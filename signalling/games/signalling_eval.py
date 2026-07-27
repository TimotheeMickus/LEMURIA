import csv
from collections import namedtuple

import numpy as np
import torch

from ..eval import compute_correlation


# Returns a copy of `signals` in which, for each row i, the first `lengths[i]` tokens are
# randomly permuted (the padding beyond that length is left untouched). Used to measure how
# much a language relies on symbol *order* rather than on which symbols are present.
def scramble_signals(signals, lengths):
    scrambled = signals.detach().clone()
    for i in range(scrambled.size(0)):
        length = int(lengths[i].item()) if torch.is_tensor(lengths[i]) else int(lengths[i])
        if(length > 1):  # nothing to permute for length 0 or 1
            scrambled[i, :length] = scrambled[i, :length][torch.randperm(length)]
    return scrambled


# Writes signal rows to `path` as CSV (a header row first). `rows` is an iterable of
# already-formatted iterables (each a full row).
def dump_signals_csv(path, header, rows):
    with open(path, 'w') as ostr:
        writer = csv.writer(ostr)
        _ = writer.writerow(header)
        for row in rows:
            _ = writer.writerow(row)


# Evaluation helpers shared by the signalling games (AliceBob, AlexBeth, and their subclasses).
# Mixed in *before* the base game, e.g. `class AliceBob(SignallingEvalMixin, CNNPretrainable, Game)`.
class SignallingEvalMixin:
    # Logs a scalar to the autologger and (unless display is 'minimal') prints it.
    def _log(self, name, value, epoch_index):
        self.autologger._write(name, value, epoch_index, direct=True)
        if(self.autologger.display != 'minimal'):
            print(f'{name}\t{value}')

    # Correctness probability of each signal for the current batch, as scored by the consumer
    # (receiver / retriever). Subclasses implement this; its shape must match the
    # `orig_correctness` passed to `_scrambling_resistance`:
    #   * AliceBob: P(target)  per item        -> shape (batch,)
    #   * AlexBeth: P(correct) per candidate   -> shape (batch, num_candidates)
    def _signal_correctness(self, batch, signals, lengths):
        raise NotImplementedError

    # Scrambling-resistance contribution of one batch. Scrambles the signals, rescoring them
    # with the consumer, and returns (kept, base) where
    #   kept = sum(min(orig_correctness, scrambled_correctness))   and   base = sum(orig_correctness).
    # Accumulate these across batches; the resistance is (total_kept / total_base), a value in
    # [0, 1]. The min prevents signals that accidentally improve after scrambling from inflating it.
    def _scrambling_resistance(self, batch, signals, lengths, orig_correctness):
        scrambled = scramble_signals(signals, lengths)
        scrambled_correctness = self._signal_correctness(batch, scrambled, lengths)
        kept = torch.minimum(orig_correctness, scrambled_correctness).sum().item()
        base = orig_correctness.sum().item()
        return kept, base


# ----------------------------------------------------------------------------------------------
# Topographic similarity (shared by AliceBob and AlexBeth)
# ----------------------------------------------------------------------------------------------

TopsimResult = namedtuple("TopsimResult", ["r", "p", "z"])

_NAN_TOPSIM = TopsimResult(float("nan"), float("nan"), float("nan"))



# Makes a meaning (or signal) hashable so it can be used as a dedup key / counted for variation.
def _key(x):
    if(isinstance(x, np.ndarray)):
        return tuple(x.ravel().tolist())
    if(isinstance(x, (list, tuple))):
        return tuple(x)
    return x


# Topographic similarity between signals and meanings: the correlation between pairwise signal
# distances and pairwise meaning distances, via the Mantel test (so we also get a permutation
# p-value and a z-score, i.e. how many standard deviations above the permutation null).
#
# signals, meanings:   aligned lists; `meaning_distance`/`signal_distance` are applied to the raw
#                      objects (or to their per-symbol-chr string form when map_*_to_str=True).
# meaning_keys:        aligned list of hashable keys identifying each meaning, used only for
#                      deduplication (defaults to `meanings`).
# deduplicate:         if True (default), each meaning is kept at most once. Duplicate
#                      meaning/signal points otherwise inflate the correlation arbitrarily.
# error_on_duplicate_meanings:
#                      if True, a repeated meaning raises instead. This is checked even when
#                      `deduplicate` is False (i.e. it takes precedence over keeping duplicates).
# method:              'spearman' (default, the topsim convention) or 'pearson'.
# correl_only:         if True (default), only the veridical correlation is computed; the Mantel
#                      permutations are skipped, so it is fast but p and z are returned as NaN.
#                      Set False to run the permutation test and get a real p-value and z-score.
#
# Returns a TopsimResult(r, p, z); r/p/z are NaN if the sample is degenerate (< 3 meanings, or
# no variation in signals or in meanings). p and z are also NaN whenever correl_only is True.
def topographic_similarity(signals, meanings, signal_distance, meaning_distance, *,
                           meaning_keys=None, deduplicate=True, map_signal_to_str=True,
                           map_meaning_to_str=False, method="spearman", perms=1000,
                           correl_only=True, error_on_duplicate_meanings=False):
    if(meaning_keys is None):
        meaning_keys = meanings

    # Handle duplicate meanings:
    #   * error_on_duplicate_meanings -> raise on the first repeat (takes precedence, checked
    #     even when deduplicate is False);
    #   * deduplicate (default)       -> keep only the first occurrence of each meaning;
    #   * otherwise                   -> keep every point, duplicates included.
    seen = set()
    u_signals, u_meanings = [], []
    for signal, meaning, mkey in zip(signals, meanings, meaning_keys):
        key = _key(mkey)
        is_duplicate = (key in seen)
        if(is_duplicate and error_on_duplicate_meanings):
            raise ValueError("topographic_similarity: duplicate meaning encountered (key=%r); the "
                             "sample must contain each meaning at most once when "
                             "error_on_duplicate_meanings=True." % (key,))
        if(is_duplicate and deduplicate):
            continue
        seen.add(key)
        u_signals.append(signal)
        u_meanings.append(meaning)

    # The Mantel test needs at least 3 objects and some variation on each side.
    if(len(u_meanings) < 3): return _NAN_TOPSIM
    if(len(set(_key(s) for s in u_signals)) < 2): return _NAN_TOPSIM
    if(len(set(_key(m) for m in u_meanings)) < 2): return _NAN_TOPSIM

    # The Mantel permutation test draws from NumPy's *global* RNG; snapshot and restore it so
    # that measuring the language does not perturb the experiment's own random stream. (When
    # correl_only is True no permutations run, so this is a cheap no-op.)
    rng_state = np.random.get_state()
    try:
        r, p, z, _r_mean = compute_correlation.mantel(
            u_signals, u_meanings,
            signal_distance=signal_distance, meaning_distance=meaning_distance,
            map_signal_to_str=map_signal_to_str, map_ctg_to_str=map_meaning_to_str,
            method=method, perms=perms, correl_only=correl_only,
        )
    finally:
        np.random.set_state(rng_state)

    # With correl_only=True the Mantel test returns p = z = None.
    return TopsimResult(
        float(r),
        float(p) if (p is not None) else float("nan"),
        float(z) if (z is not None) else float("nan"),
    )
