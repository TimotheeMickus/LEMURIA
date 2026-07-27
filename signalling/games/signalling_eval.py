import csv

import torch


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
