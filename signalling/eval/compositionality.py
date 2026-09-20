"""
Compositionality probe for the predicate signalling game.

Given (predicate, signal) pairs, we train a small sequence-to-sequence model
(encoder + LSTM decoder) to reconstruct the predicate written in Polish
(prefix) notation from the signal, and we measure the
fraction of held-out predicates that are reconstructed *exactly*. This is a
standard "can a learner recover the meaning from the signal?" probe: a language
is compositional to the extent that a generic learner generalises the
signal->meaning mapping to unseen signals.

Public entry points
--------------------
compositionality(pairs, spec, hparams, device, seed) -> float in [0, 1]
    Single-shuffle 5-fold cross-validation with early stopping. Each fold is
    trained until the held-out loss has not improved for
    `hparams.patience` epochs; the fold contributes its best held-out accuracy.
    The returned score is the mean over the 5 folds.

compositionality_per_item(pairs, spec, hparams, device, seed, n_folds, n_runs) -> (mean_acc, mean_loss, per_item)
    Like `compositionality`, but additionally reports a per-item score: the
    fraction of `n_runs` independent repeats in which that item was exactly
    recovered. With `n_folds=None` (the default), each fold holds out exactly
    one item (leave-one-out), which both sharpens the aggregate estimate (no
    partial-fold averaging) and identifies exactly which signals the probe
    fails to recover.

emergent_pairs(asker, dataset, device) -> (pairs, spec)
    Runs the asker (eval/argmax) once per predicate to obtain the emergent
    signal, and pairs it with the Polish-notation encoding of the predicate.

main(global_args, remaining_args)
    `--do compositionality_search`: random hyperparameter search. To find good
    probe hyperparameters we tune on the *reverse-Polish* control language (the
    signal for a predicate is its reverse-Polish encoding), a language that is
    compositional by construction, so the search rewards hyperparameters that let
    the probe recover composition when it is present.

main_loo(global_args, remaining_args)
    `--do compositionality_loo`: leave-one-out (or n-fold, user-specified)
    cross-validated compositionality probe on a dumped emergent language
    (--comp_signals_csv). Writes a per-signal CSV alongside the aggregate score.

Efficiency notes
----------------
* The whole pair set is padded into tensors once and reused across folds; folds
  are row-index views on the GPU.
* Training uses teacher forcing (one decoder LSTM pass per batch); evaluation
  uses batched greedy decoding.
* The encoder uses packed sequences so padding costs nothing.
"""

import json
import argparse
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn

from ..utils import predicate_data
from ..utils import cli


# ----------------------------------------------------------------------------- #
# Vocabulary and (predicate, signal) pair construction
# ----------------------------------------------------------------------------- #

class PredicateVocab:
    """
    Maps predicate atoms (values) and operators (NEG, CONJ) to contiguous ids,
    followed by the special tokens PAD, BOS, EOS. Deterministic given a dataset.
    """
    def __init__(self, dataset):
        value_tokens = sorted(v.name for v in dataset.values)
        op_tokens = [predicate_data.NEGATION_TOKEN, predicate_data.CONJUNCTION_TOKEN]
        self.tokens = value_tokens + op_tokens                 # list[str]
        self.s2i = {t: i for i, t in enumerate(self.tokens)}   # dict[str, int]

        base = len(self.tokens)
        self.pad_id = base
        self.bos_id = base + 1
        self.eos_id = base + 2
        self.size = base + 3

    # tokens: list[str] -> list[int]
    def encode(self, tokens):
        return [self.s2i[t] for t in tokens]


@dataclass
class Spec:
    """Vocabulary sizes and special-token ids for the encoder (src) and decoder (tgt)."""
    src_vocab_size: int
    src_pad_id: int
    tgt_vocab_size: int
    tgt_pad_id: int
    tgt_bos_id: int
    tgt_eos_id: int


@dataclass
class HParams:
    embed_dim: int = 64
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.033
    lr: float = 3.3e-4
    batch_size: int = 512
    max_epochs: int = 512
    patience: int = 2
    encoder: str = "bilstm"   # "bilstm" or "transformer" -- see Seq2Seq


# Command-line arguments for the compositionality probe and its hyperparameter search. Kept next to
# the probe code that reads them (see `hparams_from_args`). Returns the argparse group.
def add_compositionality_args(parser):
    import pathlib
    group = parser.add_argument_group(title='Compositionality', description='compositionality probe (encoder -> LSTM decoder) and its hyperparameter search')
    group.add_argument('--eval_compositionality', help='during evaluation, measure the compositionality of the emergent language (5-fold CV exact-match) and log it as "compositionality"', action='store_true')
    group.add_argument('--comp_encoder', help='probe: encoder architecture. "bilstm" (default) is inherently order-sensitive '
                        '(a signal and its reverse generally encode to different states). "transformer" is a self-attention '
                        'encoder with learned positional embeddings: attention is permutation-EQUIVARIANT and position only '
                        'enters additively, so it is not structurally biased either way and can learn an order-sensitive or '
                        'an order-insensitive signal->meaning mapping, whichever the language actually uses.',
                        choices=['bilstm', 'transformer'], default='bilstm')
    group.add_argument('--comp_embed_dim', help='probe: token embedding dimension', type=int, default=64)
    group.add_argument('--comp_hidden_dim', help='probe: LSTM hidden dimension', type=int, default=64)
    group.add_argument('--comp_num_layers', help='probe: number of LSTM layers (encoder and decoder)', type=int, default=1)
    group.add_argument('--comp_dropout', help='probe: dropout', type=float, default=0.033)
    group.add_argument('--comp_lr', help='probe: learning rate (Adam)', type=float, default=3.3e-4)
    group.add_argument('--comp_batch_size', help='probe: batch size', type=int, default=512)
    group.add_argument('--comp_max_epochs', help='probe: maximum training epochs per fold (early stopping usually stops earlier)', type=int, default=512)
    group.add_argument('--comp_patience', help='probe: early-stopping patience in epochs (stop when held-out exact-match has not improved for this many epochs)', type=int, default=2)
    group.add_argument('--comp_search_trials', help='compositionality_search: number of random hyperparameter trials', type=int, default=30)
    group.add_argument('--comp_search_out', help='compositionality_search: optional path to write the best hyperparameters (and history) as JSON', type=pathlib.Path, default=None)
    group.add_argument('--comp_signals_csv', help="compositionality_search / compositionality_loo: read the emergent language from a dumped-signals CSV (columns 'signal' and 'pred_str', the kind produced by --dump_signals) instead of the synthetic control language. When set, no dataset parameters are needed and the dataset is not built.", type=pathlib.Path, default=None)
    group.add_argument('--comp_target_notation', help="compositionality_search / compositionality_loo: predicate serialisation used as the probe target when --comp_signals_csv is given", choices=['polish', 'reverse_polish'], default='polish')
    group.add_argument('--comp_n_folds', help="compositionality_loo: number of CV folds. Omit for leave-one-out (n_folds = number of distinct signals), the default and most precise setting; pass a smaller number to trade precision for speed.", type=int, default=None)
    group.add_argument('--comp_runs_per_item', help="compositionality_loo: number of independent repeats of the whole CV procedure (fresh fold shuffle and probe init each time). Each signal's reported score is the fraction of runs in which it was exactly recovered, which smooths out the noise of any single stochastic training run.", type=int, default=1)
    group.add_argument('--comp_out_csv', help="compositionality_loo: path to write the per-signal results CSV. Defaults to '<comp_signals_csv stem>.comp_loo.csv' next to the input file.", type=pathlib.Path, default=None)
    return group


def hparams_from_args(args):
    return HParams(
        embed_dim=args.comp_embed_dim,
        hidden_dim=args.comp_hidden_dim,
        num_layers=args.comp_num_layers,
        dropout=args.comp_dropout,
        lr=args.comp_lr,
        batch_size=args.comp_batch_size,
        max_epochs=args.comp_max_epochs,
        patience=args.comp_patience,
        encoder=args.comp_encoder,
    )


# ----------------------------------------------------------------------------- #
# Model
# ----------------------------------------------------------------------------- #

class Seq2Seq(nn.Module):
    """Encoder (biLSTM or self-attention, see `h.encoder`) followed by an LSTM decoder.

    biLSTM: inherently order-sensitive -- a signal and its reverse generally produce different
    final states, so the model must actively learn any order-invariance it needs from examples.

    transformer: self-attention is permutation-EQUIVARIANT; position enters only as an additive
    embedding, so nothing about the architecture favours order-sensitivity over order-invariance
    (or vice versa) -- whichever the data calls for is equally easy to learn, in principle.
    """
    _TF_HEADS = 4  # requires h.embed_dim % _TF_HEADS == 0; the default embed_dim=64 satisfies this.

    def __init__(self, spec, h, max_signal_len=256):
        super().__init__()
        self.spec = spec
        self.num_layers = h.num_layers
        self.hidden_dim = h.hidden_dim
        self.encoder_kind = h.encoder

        rnn_dropout = h.dropout if (h.num_layers > 1) else 0.0

        self.src_emb = nn.Embedding(spec.src_vocab_size, h.embed_dim, padding_idx=spec.src_pad_id)

        if(self.encoder_kind == "bilstm"):
            self.encoder = nn.LSTM(h.embed_dim, h.hidden_dim, num_layers=h.num_layers,
                                   batch_first=True, bidirectional=True, dropout=rnn_dropout)
            # Bridges the (bidirectional) encoder final state to the decoder initial state.
            self.bridge_h = nn.Linear(2 * h.hidden_dim, h.hidden_dim)
            self.bridge_c = nn.Linear(2 * h.hidden_dim, h.hidden_dim)
        elif(self.encoder_kind == "transformer"):
            assert (h.embed_dim % self._TF_HEADS == 0), (
                f"--comp_encoder transformer needs --comp_embed_dim divisible by {self._TF_HEADS} "
                f"(got {h.embed_dim}).")
            self.pos_emb = nn.Embedding(max_signal_len, h.embed_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=h.embed_dim, nhead=self._TF_HEADS, dim_feedforward=4 * h.embed_dim,
                dropout=h.dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=h.num_layers)
            # Bridges the (mean-pooled) encoder output to the decoder initial state.
            self.bridge_h = nn.Linear(h.embed_dim, h.hidden_dim)
            self.bridge_c = nn.Linear(h.embed_dim, h.hidden_dim)
        else:
            raise ValueError(f"unknown --comp_encoder {self.encoder_kind!r} (expected 'bilstm' or 'transformer').")

        self.tgt_emb = nn.Embedding(spec.tgt_vocab_size, h.embed_dim, padding_idx=spec.tgt_pad_id)
        self.decoder = nn.LSTM(h.embed_dim, h.hidden_dim, num_layers=h.num_layers,
                               batch_first=True, dropout=rnn_dropout)

        self.out = nn.Linear(h.hidden_dim, spec.tgt_vocab_size)
        self.dropout = nn.Dropout(h.dropout)

    # src: (B, S) long ; src_len: (B,) long -> decoder initial state (h, c), each (num_layers, B, hidden)
    def encode(self, src, src_len):
        if(self.encoder_kind == "bilstm"):
            emb = self.dropout(self.src_emb(src))
            packed = nn.utils.rnn.pack_padded_sequence(emb, src_len.cpu(), batch_first=True, enforce_sorted=False)
            _, (h_n, c_n) = self.encoder(packed)  # (num_layers * 2, B, hidden)

            def combine(state):
                state = state.view(self.num_layers, 2, state.size(1), self.hidden_dim)  # (L, 2, B, H)
                return torch.cat([state[:, 0], state[:, 1]], dim=-1)                     # (L, B, 2H)

            h0 = torch.tanh(self.bridge_h(combine(h_n))).contiguous()  # (L, B, H)
            c0 = torch.tanh(self.bridge_c(combine(c_n))).contiguous()  # (L, B, H)
            return (h0, c0)

        # "transformer": mean-pool the (masked) encoder output into a single vector, then bridge
        # it to the decoder's initial state -- the attention/pooling combination has no encoder
        # "layers" to seed individually the way the biLSTM's stacked states do, so the same
        # pooled+bridged vector is tiled across all `num_layers` of the LSTM decoder.
        B, S = src.shape
        positions = torch.arange(S, device=src.device).unsqueeze(0).expand(B, S)
        emb = self.dropout(self.src_emb(src) + self.pos_emb(positions))
        pad_mask = torch.arange(S, device=src.device).unsqueeze(0) >= src_len.unsqueeze(1)  # (B, S), True = pad
        enc = self.encoder(emb, src_key_padding_mask=pad_mask)  # (B, S, embed_dim)
        keep = (~pad_mask).unsqueeze(-1).float()
        pooled = (enc * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)  # (B, embed_dim), mean over real tokens

        h0 = torch.tanh(self.bridge_h(pooled)).unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c0 = torch.tanh(self.bridge_c(pooled)).unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        return (h0, c0)

    # Teacher-forced training pass. tgt_in: (B, T) -> logits (B, T, V)
    def forward(self, src, src_len, tgt_in):
        state = self.encode(src, src_len)
        emb = self.dropout(self.tgt_emb(tgt_in))
        out, _ = self.decoder(emb, state)
        return self.out(self.dropout(out))

    # Batched greedy decoding. Returns produced token ids (B, <=max_len).
    @torch.no_grad()
    def greedy_decode(self, src, src_len, max_len):
        state = self.encode(src, src_len)
        batch = src.size(0)
        device = src.device

        tok = torch.full((batch, 1), self.spec.tgt_bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        outputs = []
        for _ in range(max_len):
            out, state = self.decoder(self.tgt_emb(tok), state)   # (B, 1, H)
            nxt = self.out(out.squeeze(1)).argmax(dim=-1)         # (B,)
            nxt = nxt.masked_fill(finished, self.spec.tgt_pad_id)
            outputs.append(nxt)
            finished = finished | (nxt == self.spec.tgt_eos_id)
            tok = nxt.unsqueeze(1)
            if bool(finished.all()):
                break
        return torch.stack(outputs, dim=1)


# ----------------------------------------------------------------------------- #
# Tensorization and cross-validated training
# ----------------------------------------------------------------------------- #

class _Data:
    """Pre-padded tensors for the whole pair set, shared across folds."""
    def __init__(self, pairs, spec, device):
        n = len(pairs)
        src_seqs = [p[0] for p in pairs]
        tgt_seqs = [p[1] for p in pairs]  # raw target token ids, no BOS/EOS

        smax = max((len(s) for s in src_seqs), default=1)
        src = torch.full((n, max(1, smax)), spec.src_pad_id, dtype=torch.long)
        src_len = torch.ones(n, dtype=torch.long)
        for i, s in enumerate(src_seqs):
            if len(s) > 0:
                src[i, :len(s)] = torch.tensor(s, dtype=torch.long)
                src_len[i] = len(s)
            # empty signal (should not happen): keep length 1 with a single pad token

        tmax = max((len(t) for t in tgt_seqs), default=0)
        # decoder input:  [BOS, t0, ..., t_{k-1}]  ; decoder target: [t0, ..., t_{k-1}, EOS]
        self.tgt_in = torch.full((n, tmax + 1), spec.tgt_pad_id, dtype=torch.long)
        self.tgt_out = torch.full((n, tmax + 1), spec.tgt_pad_id, dtype=torch.long)
        for i, t in enumerate(tgt_seqs):
            self.tgt_in[i, 0] = spec.tgt_bos_id
            if len(t) > 0:
                tt = torch.tensor(t, dtype=torch.long)
                self.tgt_in[i, 1:1 + len(t)] = tt
                self.tgt_out[i, :len(t)] = tt
            self.tgt_out[i, len(t)] = spec.tgt_eos_id

        self.src = src.to(device)
        self.src_len = src_len.to(device)
        self.tgt_in = self.tgt_in.to(device)
        self.tgt_out = self.tgt_out.to(device)
        self.gold = [list(t) for t in tgt_seqs]  # for exact-match comparison
        self.max_decode_len = tmax + 1           # room for EOS
        self.n = n


@torch.no_grad()
def _exact_match_per_item(model, data, idx, spec, decode_batch=256):
    """Per-item exact-match correctness and greedy-decoded predictions for `idx` (lists aligned with `idx`)."""
    model.eval()
    device = data.src.device
    eos, pad = spec.tgt_eos_id, spec.tgt_pad_id
    correct = [False] * len(idx)
    predicted = [None] * len(idx)
    for start in range(0, len(idx), decode_batch):
        rows = idx[start:start + decode_batch]
        rows_t = torch.as_tensor(rows, dtype=torch.long, device=device)
        produced = model.greedy_decode(data.src[rows_t], data.src_len[rows_t], data.max_decode_len).tolist()
        for offset, (row, seq) in enumerate(zip(rows, produced)):
            out = []
            for tok in seq:
                if tok == eos or tok == pad:
                    break
                out.append(tok)
            j = start + offset
            predicted[j] = out
            correct[j] = (out == data.gold[int(row)])
    return correct, predicted


@torch.no_grad()
def _exact_match(model, data, idx, spec, decode_batch=256):
    """Fraction of `idx` whose greedy decoding equals the gold token sequence exactly."""
    correct, _ = _exact_match_per_item(model, data, idx, spec, decode_batch=decode_batch)
    return sum(correct) / len(idx) if len(idx) > 0 else 0.0


# Minimum decrease in held-out loss to count as an improvement (avoids stopping being driven
# by negligible fluctuations).
_MIN_DELTA = 1e-4


@torch.no_grad()
def _val_loss(model, data, idx, spec, batch=512):
    """Mean per-token cross-entropy on `idx` (teacher forcing), aggregated across batches."""
    model.eval()
    device = data.src.device
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
    total_loss, total_tokens = 0.0, 0
    for start in range(0, idx_t.numel(), batch):
        rows = idx_t[start:start + batch]
        logits = model(data.src[rows], data.src_len[rows], data.tgt_in[rows])
        target = data.tgt_out[rows]
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1),
            ignore_index=spec.tgt_pad_id, reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += int((target != spec.tgt_pad_id).sum().item())
    return total_loss / max(1, total_tokens)


def _run_one_fold_detailed(data, train_idx, test_idx, spec, h, device, seed):
    """
    Trains one seq2seq on `train_idx` and returns
    (best held-out exact-match, best held-out loss, stopping epoch, ever_correct, last_predicted).

    Early stopping watches the held-out *loss* (which decreases smoothly from the first
    epoch) rather than exact-match: exact-match sits at exactly 0 during warm-up until it
    jumps, so a patience clock on exact-match would kill any run that has not yet crossed
    that transition. Loss-based stopping tracks real convergence; we still *report* the
    best exact-match, the quantity of interest.

    `ever_correct` (aligned with `test_idx`) is True for an item iff its greedy decoding matched
    the gold sequence at *some* epoch (mirroring how `best_acc` is the running max, over epochs, of
    the fold's exact-match fraction). `last_predicted` holds the most recent *wrong* decoding seen
    for each item (None if it was correct at the last epoch it was wrong, or never wrong) -- handy
    to inspect what the probe guessed instead of the gold predicate.
    """
    if(seed is not None): torch.manual_seed(seed)
    model = Seq2Seq(spec, h, max_signal_len=data.src.size(1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=h.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=spec.tgt_pad_id)

    train_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    n_train = train_t.numel()
    eval_batch = max(256, h.batch_size)

    best_acc = 0.0
    best_val = float("inf")
    stale = 0
    stopped_epoch = 0
    ever_correct = [False] * len(test_idx)
    last_predicted = [None] * len(test_idx)
    with torch.enable_grad():  # evaluate() runs under torch.no_grad(); training needs gradients
        for epoch in range(h.max_epochs):
            stopped_epoch = epoch + 1
            model.train()
            order = train_t[torch.randperm(n_train, device=device)]
            for start in range(0, n_train, h.batch_size):
                rows = order[start:start + h.batch_size]
                logits = model(data.src[rows], data.src_len[rows], data.tgt_in[rows])
                loss = criterion(logits.reshape(-1, logits.size(-1)), data.tgt_out[rows].reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            correct, predicted = _exact_match_per_item(model, data, test_idx, spec, decode_batch=eval_batch)
            best_acc = max(best_acc, sum(correct) / len(test_idx) if len(test_idx) > 0 else 0.0)
            for i, ok in enumerate(correct):
                if ok:
                    ever_correct[i] = True
                else:
                    last_predicted[i] = predicted[i]

            val = _val_loss(model, data, test_idx, spec, batch=eval_batch)
            if val < (best_val - _MIN_DELTA):
                best_val = val
                stale = 0
            else:
                stale += 1
                if stale >= h.patience:
                    break
    return best_acc, best_val, stopped_epoch, ever_correct, last_predicted


def _run_one_fold(data, train_idx, test_idx, spec, h, device, seed):
    """Trains one seq2seq on `train_idx` and returns (best held-out exact-match, best held-out loss, stopping epoch)."""
    best_acc, best_val, stopped_epoch, _, _ = _run_one_fold_detailed(data, train_idx, test_idx, spec, h, device, seed)
    return best_acc, best_val, stopped_epoch


def compositionality(pairs, spec, hparams, device, seed=None, n_folds=5, verbose=False, return_epochs=False):
    """
    Single-shuffle `n_folds`-fold cross-validation of the exact-match probe.

    pairs:   list[(src_ids: list[int], tgt_ids: list[int])]; tgt_ids are the predicate tokens (Polish notation; without any BOS/EOS).
    spec:    Spec with vocab sizes and special-token ids.
    hparams: HParams.
    Returns (mean_acc, mean_loss): the mean over folds of each fold's best held-out exact-match
    accuracy (a float in [0, 1]) and of each fold's best held-out loss ((NaN, NaN) if there are
    fewer pairs than folds). If `return_epochs`, returns (mean_acc, mean_loss, per_fold_stopping_epochs) instead.
    """
    print(f"[comp] hparams={hparams}; n_folds={n_folds}")
    
    n = len(pairs)
    if(n < n_folds):
        print(f"[comp] Aborted (n < n_folds; {n} < {n_folds}).")
        return (float("nan"), float("nan"), []) if return_epochs else (float("nan"), float("nan"))

    data = _Data(pairs, spec, device)

    # Shuffles the pairs, then partitions them into `n_folds` (near-)equal parts; each part is test once.
    perm = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(perm, n_folds)

    scores, losses, epochs = [], [], []
    for f in range(n_folds):
        test_idx = folds[f]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])
        score, val_loss, stopped_epoch = _run_one_fold(data, train_idx, test_idx, spec, hparams, device, seed=(None if (seed is None) else (seed + f)))
        scores.append(score)
        losses.append(val_loss)
        epochs.append(stopped_epoch)
        if(verbose):
            print(f"    [comp] fold {f}: best test exact-match = {score:.4f}, best test loss = {val_loss:.4f} (stopped at epoch {stopped_epoch})", flush=True)

    mean_score = float(np.mean(scores))
    mean_loss = float(np.mean(losses))
    return (mean_score, mean_loss, epochs) if return_epochs else (mean_score, mean_loss)


def compositionality_per_item(pairs, spec, hparams, device, seed=None, n_folds=None, n_runs=1, pred_strs=None):
    """
    Repeated cross-validation that additionally reports a per-item score (see module docstring).

    n_folds: number of CV folds; None (the default) means leave-one-out (n_folds = len(pairs)),
        which both sharpens the aggregate estimate (no partial-fold averaging) and lets every item
        be individually diagnosed. A smaller number trades precision for speed, as in `compositionality`.
    n_runs: number of independent repeats of the whole (shuffle, fold, train) procedure, each with
        its own seed. An item's score is the fraction of runs in which it was exactly recovered.
    pred_strs: optional, aligned with `pairs`; when given, each fold's progress line also shows the
        held-out predicate(s) and their signal.

    Returns (mean_acc, mean_loss, per_item). mean_acc/mean_loss are the mean over runs of the
    per-run fold-averaged score/loss, exactly as `compositionality` computes them (so the two
    functions' aggregate numbers are directly comparable). per_item is a list aligned with `pairs`,
    each entry a dict: {'index': i, 'score': float in [0, 1], 'n_correct': int, 'n_runs': int,
    'predicted': list[int] or None}. 'predicted' is the most recent wrong decoding seen for that
    item (None if it was recovered in every run).
    """
    n = len(pairs)
    n_folds = n if (n_folds is None) else n_folds
    print(f"[comp-loo] hparams={hparams}; n_folds={n_folds}{' (leave-one-out)' if n_folds == n else ''}; n_runs={n_runs}")
    if(n < n_folds):
        print(f"[comp-loo] Aborted (n < n_folds; {n} < {n_folds}).")
        return float("nan"), float("nan"), []

    data = _Data(pairs, spec, device)
    correct_counts = [0] * n
    last_predicted = [None] * n
    run_scores, run_losses = [], []

    for run in range(n_runs):
        run_seed = None if (seed is None) else (seed + run * 1_000_003)
        if(n_folds == n):
            # True leave-one-out: each fold is a single item, so shuffling only changes the order
            # items are processed/printed in, not which items are grouped -- use the natural (id)
            # order instead, so LOO progress is easy to follow/compare across runs.
            perm = np.arange(n)
        else:
            perm = np.random.default_rng(run_seed).permutation(n)
        folds = np.array_split(perm, n_folds)

        fold_scores, fold_losses = [], []
        for f in range(n_folds):
            test_idx = folds[f]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])
            fold_seed = None if (run_seed is None) else (run_seed + f)
            score, val_loss, stopped_epoch, ever_correct, predicted = _run_one_fold_detailed(
                data, train_idx, test_idx, spec, hparams, device, seed=fold_seed)
            fold_scores.append(score)
            fold_losses.append(val_loss)
            for local_i, global_i in enumerate(test_idx):
                global_i = int(global_i)
                if ever_correct[local_i]:
                    correct_counts[global_i] += 1
                else:
                    last_predicted[global_i] = predicted[local_i]
            if(pred_strs is None):
                detail = ""
            else:
                detail = " | " + "; ".join(
                    f"{pred_strs[int(gi)]!r} <- [{' '.join(map(str, pairs[int(gi)][0]))}]" for gi in test_idx)
            print(f"    [comp-loo] run {run + 1}/{n_runs} fold {f + 1}/{n_folds}: exact-match = {score:.4f}, "
                  f"loss = {val_loss:.4f} (stopped at epoch {stopped_epoch}, {len(test_idx)} held out){detail}", flush=True)

        run_scores.append(float(np.mean(fold_scores)))
        run_losses.append(float(np.mean(fold_losses)))
        print(f"[comp-loo] run {run + 1}/{n_runs}: mean exact-match = {run_scores[-1]:.4f}, mean loss = {run_losses[-1]:.4f}", flush=True)

    per_item = [
        {
            'index': i,
            'score': correct_counts[i] / n_runs,
            'n_correct': correct_counts[i],
            'n_runs': n_runs,
            'predicted': (None if (correct_counts[i] == n_runs) else last_predicted[i]),
        }
        for i in range(n)
    ]
    mean_score = float(np.mean(run_scores))
    mean_loss = float(np.mean(run_losses))
    return mean_score, mean_loss, per_item


# ----------------------------------------------------------------------------- #
# Pair builders
# ----------------------------------------------------------------------------- #

@torch.no_grad()
def emergent_pairs(asker, dataset, device, batch_size=256):
    """
    (predicate, emergent-signal) pairs: run the asker (eval/argmax) on every predicate
    index and pair each produced signal with the Polish encoding of the predicate.
    Returns (pairs, spec).
    """
    vocab = PredicateVocab(dataset)
    n = len(dataset.predicates)

    was_training = asker.training
    asker.eval()
    signals = []
    for start in range(0, n, batch_size):
        idx = torch.arange(start, min(start + batch_size, n), device=device)
        signal, length = asker(idx).action           # (B, L) long, (B, 1)
        signal = signal.cpu()
        length = length.view(-1).cpu()
        for i in range(signal.size(0)):
            L = max(1, int(length[i].item()))         # includes the trailing EOS; always >= 1
            signals.append(signal[i, :L].tolist())
    if was_training:
        asker.train()

    targets = [vocab.encode(pred.polish()) for pred in dataset.predicates]
    pairs = list(zip(signals, targets))

    spec = Spec(
        src_vocab_size=asker.alphabet_size,   # base_alphabet_size + 3, covers every symbol id
        src_pad_id=asker.padding_idx,
        tgt_vocab_size=vocab.size,
        tgt_pad_id=vocab.pad_id,
        tgt_bos_id=vocab.bos_id,
        tgt_eos_id=vocab.eos_id,
    )
    return pairs, spec


def reverse_polish_pairs(dataset):
    """
    (predicate, reverse-Polish-signal) pairs used as the compositional control language
    for the hyperparameter search: src = reverse-Polish tokens, tgt = Polish tokens.
    Returns (pairs, spec).
    """
    vocab = PredicateVocab(dataset)
    pairs = [(vocab.encode(p.reverse_polish()), vocab.encode(p.polish())) for p in dataset.predicates]
    spec = Spec(
        src_vocab_size=vocab.size,
        src_pad_id=vocab.pad_id,
        tgt_vocab_size=vocab.size,
        tgt_pad_id=vocab.pad_id,
        tgt_bos_id=vocab.bos_id,
        tgt_eos_id=vocab.eos_id,
    )
    return pairs, spec


# ----------------------------------------------------------------------------- #
# Reading an emergent language from a dumped-signals CSV ( --comp_signals_csv )
# ----------------------------------------------------------------------------- #

# Predicate.__str__ is fully-parenthesised infix: a value is a bare name, negation is "(¬X)",
# conjunction is "(X∧Y)". Value names are "P{i}-v{j}" and never contain the structural characters
# below, so tokenisation is unambiguous and a tiny recursive-descent parser recovers the structure.
_STRUCT_CHARS = frozenset("()¬∧")


def _peek(s, i):
    if(i >= len(s)):
        raise ValueError(f"unexpected end of {s!r}")
    return s[i]


# Parses s[i:] into a small AST: ('val', name) | ('neg', child) | ('conj', left, right).
# Returns (ast, next_index).
def _parse_pred_ast(s, i):
    if(_peek(s, i) == '('):
        i += 1
        if(_peek(s, i) == '¬'):
            child, i = _parse_pred_ast(s, i + 1)
            if(_peek(s, i) != ')'): raise ValueError(f"expected ')' at {i} in {s!r}")
            return ('neg', child), (i + 1)
        left, i = _parse_pred_ast(s, i)
        if(_peek(s, i) != '∧'): raise ValueError(f"expected '∧' at {i} in {s!r}")
        right, i = _parse_pred_ast(s, i + 1)
        if(_peek(s, i) != ')'): raise ValueError(f"expected ')' at {i} in {s!r}")
        return ('conj', left, right), (i + 1)

    # Atom: read up to the next structural character.
    j = i
    while((j < len(s)) and (s[j] not in _STRUCT_CHARS)):
        j += 1
    if(j == i): raise ValueError(f"empty atom at {i} in {s!r}")
    return ('val', s[i:j]), j


# Parses a predicate's __str__ into an AST (raises ValueError on malformed input).
def parse_pred_str(s):
    s = s.strip()
    ast, i = _parse_pred_ast(s, 0)
    if(i != len(s)): raise ValueError(f"trailing characters after position {i} in {s!r}")
    return ast


# Serialises an AST exactly like Predicate.serialize: Polish (prefix) by default, reverse-Polish
# (postfix) when reverse=True. Kept identical to predicate_data so the result matches
# .polish()/.reverse_polish() token-for-token (see selfcheck_parser).
def _serialize_ast(ast, reverse=False):
    kind = ast[0]
    if(kind == 'val'):
        return [ast[1]]
    if(kind == 'neg'):
        inner = _serialize_ast(ast[1], reverse=reverse)
        return (inner + [predicate_data.NEGATION_TOKEN]) if reverse else ([predicate_data.NEGATION_TOKEN] + inner)
    # 'conj'
    left = _serialize_ast(ast[1], reverse=reverse)
    right = _serialize_ast(ast[2], reverse=reverse)
    return (left + right + [predicate_data.CONJUNCTION_TOKEN]) if reverse else ([predicate_data.CONJUNCTION_TOKEN] + left + right)


# Faithfulness self-check: for every predicate in `dataset`, parsing its string and re-serialising
# must reproduce .polish() and .reverse_polish(). Not used on the CSV path (which has no dataset);
# handy as a test that the parser tracks Predicate.__str__. Returns the number of predicates checked.
def selfcheck_parser(dataset):
    for pred in dataset.predicates:
        ast = parse_pred_str(str(pred))
        got_p, want_p = _serialize_ast(ast, reverse=False), pred.polish()
        if(got_p != want_p): raise AssertionError(f"polish mismatch for {str(pred)!r}: {got_p} != {want_p}")
        got_r, want_r = _serialize_ast(ast, reverse=True), pred.reverse_polish()
        if(got_r != want_r): raise AssertionError(f"reverse-polish mismatch for {str(pred)!r}: {got_r} != {want_r}")
    return len(dataset.predicates)


# Builds (pairs, spec) from a dumped-signals CSV (columns: signal, pred_idx, pred_str), with NO
# dataset parameters. src = the signal token ids (verbatim, including the trailing EOS, as the
# emergent probe uses them); tgt = the predicate in `notation` ('polish' | 'reverse_polish'),
# obtained by parsing the pred_str column. Both vocabularies are inferred from the file, so nothing
# about the original dataset (properties, depth, negation/conjunction toggles) has to be specified.
def csv_pairs(path, notation="polish", return_strs=False):
    import csv as _csv

    reverse = (notation == "reverse_polish")
    NEG, CONJ = predicate_data.NEGATION_TOKEN, predicate_data.CONJUNCTION_TOKEN

    with open(str(path), newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        fields = reader.fieldnames
        if((fields is None) or ("signal" not in fields) or ("pred_str" not in fields)):
            raise ValueError(f"{path}: expected a dumped-signals CSV with at least 'signal' and 'pred_str' columns, got header {fields}.")
        rows = list(reader)
    if(len(rows) == 0):
        raise ValueError(f"{path}: no rows.")

    # Deduplicate by predicate. A dumped-signals CSV typically contains the SAME predicate many
    # times, because evaluation samples predicates with replacement and every eval item is written
    # out. The emergent language is a *function* meaning -> signal, so each predicate should
    # contribute exactly one (signal, target) pair -- matching emergent_pairs. Keeping the duplicate
    # rows would let the probe's k-fold cross-validation place identical (signal, target) pairs in
    # both the training and held-out folds, leaking the answer and greatly inflating the reported
    # compositionality. We therefore keep the first occurrence of each predicate. A later occurrence
    # with a DIFFERENT signal (the source language isn't actually a pure function of the predicate
    # in this file -- e.g. several epochs/languages concatenated, or non-deterministic decoding)
    # is not fatal: we warn and keep the first-seen signal.
    seen_signal = {}      # pred_str -> signal ids (first occurrence)
    by_idx = {}           # pred_idx -> pred_str (file-consistency check)
    src_seqs, pred_strs, pred_idxs = [], [], []
    n_dup = 0
    n_conflict = 0         # duplicate rows whose signal disagrees with the first occurrence
    conflict_preds = set() # distinct predicates affected by at least one such conflict
    for k, r in enumerate(rows):
        ps = r["pred_str"]
        try:
            sig = [int(t) for t in r["signal"].split()]
        except ValueError as e:
            raise ValueError(f"{path}: row {k}: could not parse 'signal' field {r['signal']!r} as space-separated integers ({e}).")

        pi = r["pred_idx"] if ("pred_idx" in fields) else None
        if(pi is not None):
            if((pi in by_idx) and (by_idx[pi] != ps)):
                raise ValueError(f"{path}: pred_idx {pi} maps to two different predicates ({by_idx[pi]!r} vs {ps!r}); the file looks inconsistent.")
            by_idx[pi] = ps

        if(ps in seen_signal):
            n_dup += 1
            if(seen_signal[ps] != sig):
                n_conflict += 1
                conflict_preds.add(ps)
            continue

        seen_signal[ps] = sig
        src_seqs.append(sig)
        pred_strs.append(ps)
        pred_idxs.append(pi)

    if(n_dup > 0):
        msg = (f"[comp-csv] {path}: {len(rows)} rows -> {len(src_seqs)} distinct predicates "
               f"({n_dup} duplicate rows dropped to avoid cross-validation leakage")
        if(n_conflict > 0):
            msg += (f"; {n_conflict} of those rows disagreed with their predicate's first-seen signal, "
                    f"affecting {len(conflict_preds)}/{len(src_seqs)} predicates (first-seen signal kept for each)")
        msg += ").";
        print(msg, flush=True)

    # Target: parse each (distinct) predicate and serialise in the requested notation.
    tgt_token_seqs = []
    for ps in pred_strs:
        try:
            ast = parse_pred_str(ps)
        except ValueError as e:
            raise ValueError(f"{path}: could not parse 'pred_str' field {ps!r} ({e}).")
        tgt_token_seqs.append(_serialize_ast(ast, reverse=reverse))

    # Target vocabulary inferred from the file, mirroring PredicateVocab's layout (sorted value
    # tokens, then the two operators, then PAD/BOS/EOS). Sorting keeps the ids deterministic.
    atoms = sorted({tok for seq in tgt_token_seqs for tok in seq if tok not in (NEG, CONJ)})
    tokens = atoms + [NEG, CONJ]
    s2i = {t: i for i, t in enumerate(tokens)}
    tgt_seqs = [[s2i[t] for t in seq] for seq in tgt_token_seqs]

    base = len(tokens)
    tgt_pad_id, tgt_bos_id, tgt_eos_id, tgt_vocab_size = base, base + 1, base + 2, base + 3

    # Source vocabulary inferred from the file; a fresh id above every observed symbol is the pad id.
    max_src = max((t for seq in src_seqs for t in seq), default=0)
    src_pad_id, src_vocab_size = max_src + 1, max_src + 2

    pairs = list(zip(src_seqs, tgt_seqs))
    spec = Spec(
        src_vocab_size=src_vocab_size,
        src_pad_id=src_pad_id,
        tgt_vocab_size=tgt_vocab_size,
        tgt_pad_id=tgt_pad_id,
        tgt_bos_id=tgt_bos_id,
        tgt_eos_id=tgt_eos_id,
    )
    if(return_strs):
        # id2tok: maps a target token id back to its string, for rendering a predicted sequence (see
        # main_loo). Ids above `base` (PAD/BOS/EOS) have no token string and are never emitted by the
        # probe as content, but greedy decoding could in principle predict them before hitting EOS/PAD.
        id2tok = list(tokens) + [None, None, None]
        return pairs, spec, pred_strs, pred_idxs, id2tok
    return pairs, spec


# ----------------------------------------------------------------------------- #
# Random hyperparameter search ( --do compositionality_search )
# ----------------------------------------------------------------------------- #

# Reasonable bounds for the probe.
SEARCH_SPACE = {
    "embed_dim":  [64, 128, 256],
    "hidden_dim": [64, 128, 256],
    "num_layers": [1],
    "dropout":    ("uniform", 0.0, 0.4),
    "lr":         ("loguniform", 3e-4, 3e-3),
    "batch_size": [512],
}


def _sample_hparams(rng, max_epochs, patience, encoder):
    def pick(space):
        if isinstance(space, list):
            return space[int(rng.integers(len(space)))]
        kind, lo, hi = space
        if kind == "uniform":
            return float(rng.uniform(lo, hi))
        if kind == "loguniform":
            return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        raise ValueError(f"Unknown sampling kind: {kind}")

    return HParams(
        embed_dim=int(pick(SEARCH_SPACE["embed_dim"])),
        hidden_dim=int(pick(SEARCH_SPACE["hidden_dim"])),
        num_layers=int(pick(SEARCH_SPACE["num_layers"])),
        dropout=pick(SEARCH_SPACE["dropout"]),
        lr=pick(SEARCH_SPACE["lr"]),
        batch_size=int(pick(SEARCH_SPACE["batch_size"])),
        max_epochs=max_epochs,
        patience=patience,
        encoder=encoder,   # fixed for the whole search, like max_epochs/patience -- not sampled
    )


# Lean argument parser for `--do compositionality_search`. Unlike a training run, the search only
# needs to build the control-language dataset (Data group; skipped entirely with --comp_signals_csv),
# pick a device, and read the Compositionality knobs -- so it declares exactly those, rather than
# reusing the whole predicate-game parser (which would document dozens of irrelevant training/model
# arguments in --help).
def get_args(remaining_args):
    parser = argparse.ArgumentParser(
        prog="signalling --do compositionality_search",
        description="Random hyperparameter search for the compositionality probe. Tunes the probe on a "
                    "compositional-by-construction language: the reverse-Polish control language built from the "
                    "dataset, or a dumped emergent language via --comp_signals_csv (in which case no dataset is built).")
    predicate_data.add_data_args(parser)   # the control-language dataset (unused when --comp_signals_csv is given)
    cli.add_perf_args(parser)              # --device
    add_compositionality_args(parser)      # the Compositionality group (--comp_* etc.)
    parser.add_argument('--seed', help='random seed for the search (probe init, CV splits, and hyperparameter sampling); unset => everything random', type=int, default=None)
    return parser.parse_args(remaining_args)


def main(global_args=None, remaining_args=None):
    from ..utils.predicate_data import get_data_loader

    args = get_args(remaining_args)
    device = args.device

    if(args.comp_signals_csv is not None):
        # Param-free: the CSV fully determines both the signals and the target predicates.
        dataset = None
        pairs, spec = csv_pairs(args.comp_signals_csv, notation=args.comp_target_notation)
        print(f"[comp-search] {len(pairs)} signals from {args.comp_signals_csv}; "
              f"src vocab {spec.src_vocab_size}, tgt vocab {spec.tgt_vocab_size}; target = {args.comp_target_notation}.", flush=True)
    else:
        dataset = get_data_loader(args)
        pairs, spec = reverse_polish_pairs(dataset)
        print(f"[comp-search] {len(pairs)} predicates; tuning on the reverse-Polish control language.", flush=True)

    rng = np.random.default_rng(args.seed)
    best_score, best_h = None, None
    history = []
    for trial in range(args.comp_search_trials):
        h = _sample_hparams(rng, args.comp_max_epochs, args.comp_patience, args.comp_encoder)
        score, loss, stopped_epochs = compositionality(pairs, spec, h, device, seed=args.seed, return_epochs=True)
        history.append({"score": score, "loss": loss, "stopped_epochs": stopped_epochs, **asdict(h)})
        if (best_score is None) or (score > best_score):
            best_score, best_h = score, h
        print(f"[comp-search] trial {trial + 1}/{args.comp_search_trials}: "
              f"score={score:.4f}  loss={loss:.4f}  best={best_score:.4f}  stopped@{stopped_epochs}  {h}", flush=True)

    print(f"\n[comp-search] best score = {best_score:.4f}")
    print("[comp-search] best hyperparameters (copy as CLI flags):")
    for k, v in asdict(best_h).items():
        print(f"    --comp_{k} {v}")

    if args.comp_search_out is not None:
        payload = {"best_score": best_score, "best_hparams": asdict(best_h), "history": history}
        with open(args.comp_search_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[comp-search] wrote results to {args.comp_search_out}")

    if(dataset is not None):
        try:
            dataset.close()
        except Exception:
            pass

    return best_score, best_h


# ----------------------------------------------------------------------------- #
# Leave-one-out / n-fold diagnostic probe ( --do compositionality_loo )
# ----------------------------------------------------------------------------- #

def _render_tokens(ids, id2tok):
    return " ".join((id2tok[i] if ((i is not None) and (0 <= i < len(id2tok)) and (id2tok[i] is not None)) else str(i)) for i in ids)


def _write_per_item_csv(path, pred_strs, pred_idxs, src_seqs, per_item, id2tok, notation, meta=None):
    """meta: optional dict of run/probe info (e.g. the HParams used, n_folds, seed), written as
    leading '# key: value' comment lines before the CSV header -- readable as plain text, and
    skippable by e.g. pandas.read_csv(path, comment='#')."""
    import csv as _csv
    with open(str(path), "w", newline="", encoding="utf-8") as f:
        for k, v in (meta or {}).items():
            f.write(f"# {k}: {v}\n")
        writer = _csv.writer(f)
        writer.writerow(["pred_idx", "pred_str", "signal", "target_notation", "score", "n_correct", "n_runs", "predicted"])
        for it in per_item:
            i = it['index']
            predicted = "" if (it['predicted'] is None) else _render_tokens(it['predicted'], id2tok)
            writer.writerow([
                "" if (pred_idxs[i] is None) else pred_idxs[i],
                pred_strs[i],
                " ".join(map(str, src_seqs[i])),
                notation,
                f"{it['score']:.4f}",
                it['n_correct'],
                it['n_runs'],
                predicted,
            ])


# Lean argument parser for `--do compositionality_loo`: only the Compositionality knobs plus
# --device are needed -- the signals and target predicates come entirely from --comp_signals_csv,
# so (unlike compositionality_search) no dataset arguments are declared at all.
def get_args_loo(remaining_args):
    parser = argparse.ArgumentParser(
        prog="signalling --do compositionality_loo",
        description="Leave-one-out (or n-fold, user-chosen) cross-validated compositionality probe on a "
                    "saved emergent language (a dumped-signals CSV, see --dump_signals). Reports both an "
                    "aggregate compositionality estimate and, per signal, whether the probe could recover "
                    "it from the others -- i.e. which individual signals are compositional and which are not.")
    cli.add_perf_args(parser)              # --device
    add_compositionality_args(parser)      # the Compositionality group (--comp_* etc.)
    parser.add_argument('--seed', help='random seed (probe init and CV splits); unset => everything random', type=int, default=None)
    return parser.parse_args(remaining_args)


# Length (in symbols, EOS included) and vocabulary size (EOS excluded) of a (deduplicated) set of
# (signal, target) pairs -- comparable to eval/signal_length and eval/vocab_used respectively. EOS
# is always the last symbol of a dumped signal (SignalDecoder stops right after producing it), so
# it is dropped structurally here rather than by assuming a specific eos_index.
def _language_stats(pairs):
    lengths = [len(sig) for sig, _ in pairs]
    avg_len = (sum(lengths) / len(lengths)) if lengths else float("nan")
    vocab = {tok for sig, _ in pairs for tok in sig[:-1]}
    return avg_len, len(vocab)


def main_loo(global_args=None, remaining_args=None):
    args = get_args_loo(remaining_args)
    device = args.device

    if(args.comp_signals_csv is None):
        raise SystemExit("--do compositionality_loo requires --comp_signals_csv <path to a dumped-signals CSV, see --dump_signals>.")

    pairs, spec, pred_strs, pred_idxs, id2tok = csv_pairs(args.comp_signals_csv, notation=args.comp_target_notation, return_strs=True)

    # Sort by predicate id (natural order). Together with compositionality_per_item's own
    # n_folds == n check, this makes a true leave-one-out run process items in id order instead of
    # CSV-appearance order. Harmless otherwise: a non-LOO run's fold composition is randomly
    # permuted regardless of the input order, so sorting it first changes nothing there.
    if(all(pi is not None for pi in pred_idxs)):
        order = sorted(range(len(pairs)), key=(lambda i: int(pred_idxs[i])))
        pairs = [pairs[i] for i in order]
        pred_strs = [pred_strs[i] for i in order]
        pred_idxs = [pred_idxs[i] for i in order]

    n = len(pairs)
    n_folds = n if (args.comp_n_folds is None) else args.comp_n_folds
    print(f"[comp-loo] {n} signals from {args.comp_signals_csv}; "
          f"src vocab {spec.src_vocab_size}, tgt vocab {spec.tgt_vocab_size}; target = {args.comp_target_notation}.", flush=True)
    avg_len, vocab_used = _language_stats(pairs)
    print(f"[comp-loo] language stats: avg signal length = {avg_len:.3f} symbols (EOS included; "
          f"compare eval/signal_length), vocab used = {vocab_used} symbol types (EOS excluded; "
          f"compare eval/vocab_used).", flush=True)

    hparams = hparams_from_args(args)
    mean_acc, mean_loss, per_item = compositionality_per_item(
        pairs, spec, hparams, device, seed=args.seed, n_folds=args.comp_n_folds, n_runs=args.comp_runs_per_item,
        pred_strs=pred_strs)

    n_always = sum(1 for it in per_item if it['score'] == 1.0)
    n_never = sum(1 for it in per_item if it['score'] == 0.0)
    print(f"\n[comp-loo] aggregate exact-match = {mean_acc:.4f} (mean held-out loss = {mean_loss:.4f}) over {n} signals "
          f"(n_folds={n_folds}{' = leave-one-out' if (n_folds == n) else ''}, {args.comp_runs_per_item} run(s)/item).")
    print(f"[comp-loo] {n_always}/{n} signals always recovered, {n_never}/{n} never recovered.")

    out_csv = args.comp_out_csv
    if out_csv is None:
        out_csv = args.comp_signals_csv.with_name(args.comp_signals_csv.stem + ".comp_loo.csv")
    meta = {f"comp_{k}": v for k, v in asdict(hparams).items()}
    meta.update({
        "n_folds": n_folds,
        "comp_runs_per_item": args.comp_runs_per_item,
        "seed": args.seed,
        "source_csv": args.comp_signals_csv,
        "aggregate_exact_match": f"{mean_acc:.4f}",
        "aggregate_loss": f"{mean_loss:.4f}",
    })
    _write_per_item_csv(out_csv, pred_strs, pred_idxs, [p[0] for p in pairs], per_item, id2tok, args.comp_target_notation,
                        meta=meta)
    print(f"[comp-loo] wrote per-signal results to {out_csv}")

    return mean_acc, mean_loss, per_item
