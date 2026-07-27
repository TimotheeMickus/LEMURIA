"""
Compositionality probe for the predicate signalling game.

Given (predicate, signal) pairs, we train a small sequence-to-sequence model
(bidirectional-LSTM encoder + LSTM decoder) to reconstruct the predicate
written in Polish (prefix) notation from the signal, and we measure the
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

emergent_pairs(asker, dataset, device) -> (pairs, spec)
    Runs the asker (eval/argmax) once per predicate to obtain the emergent
    signal, and pairs it with the Polish-notation encoding of the predicate.

main(global_args, remaining_args)
    `--do compositionality_search`: random hyperparameter search. To find good
    probe hyperparameters we tune on the *reverse-Polish* control language (the
    signal for a predicate is its reverse-Polish encoding), a language that is
    compositional by construction, so the search rewards hyperparameters that let
    the probe recover composition when it is present.

Efficiency notes
----------------
* The whole pair set is padded into tensors once and reused across folds; folds
  are row-index views on the GPU.
* Training uses teacher forcing (one decoder LSTM pass per batch); evaluation
  uses batched greedy decoding.
* The encoder uses packed sequences so padding costs nothing.
"""

import json
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn

from ..utils import predicate_data


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
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 512
    patience: int = 2


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
    )


# ----------------------------------------------------------------------------- #
# Model
# ----------------------------------------------------------------------------- #

class Seq2Seq(nn.Module):
    """Bidirectional-LSTM encoder followed by an LSTM decoder."""
    def __init__(self, spec, h):
        super().__init__()
        self.spec = spec
        self.num_layers = h.num_layers
        self.hidden_dim = h.hidden_dim

        rnn_dropout = h.dropout if (h.num_layers > 1) else 0.0

        self.src_emb = nn.Embedding(spec.src_vocab_size, h.embed_dim, padding_idx=spec.src_pad_id)
        self.encoder = nn.LSTM(h.embed_dim, h.hidden_dim, num_layers=h.num_layers,
                               batch_first=True, bidirectional=True, dropout=rnn_dropout)

        self.tgt_emb = nn.Embedding(spec.tgt_vocab_size, h.embed_dim, padding_idx=spec.tgt_pad_id)
        self.decoder = nn.LSTM(h.embed_dim, h.hidden_dim, num_layers=h.num_layers,
                               batch_first=True, dropout=rnn_dropout)

        # Bridges the (bidirectional) encoder final state to the decoder initial state.
        self.bridge_h = nn.Linear(2 * h.hidden_dim, h.hidden_dim)
        self.bridge_c = nn.Linear(2 * h.hidden_dim, h.hidden_dim)

        self.out = nn.Linear(h.hidden_dim, spec.tgt_vocab_size)
        self.dropout = nn.Dropout(h.dropout)

    # src: (B, S) long ; src_len: (B,) long -> decoder initial state (h, c), each (num_layers, B, hidden)
    def encode(self, src, src_len):
        emb = self.dropout(self.src_emb(src))
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_len.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, c_n) = self.encoder(packed)  # (num_layers * 2, B, hidden)

        def combine(state):
            state = state.view(self.num_layers, 2, state.size(1), self.hidden_dim)  # (L, 2, B, H)
            return torch.cat([state[:, 0], state[:, 1]], dim=-1)                     # (L, B, 2H)

        h0 = torch.tanh(self.bridge_h(combine(h_n))).contiguous()  # (L, B, H)
        c0 = torch.tanh(self.bridge_c(combine(c_n))).contiguous()  # (L, B, H)
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
def _exact_match(model, data, idx, spec, decode_batch=256):
    """Fraction of `idx` whose greedy decoding equals the gold token sequence exactly."""
    model.eval()
    device = data.src.device
    eos, pad = spec.tgt_eos_id, spec.tgt_pad_id
    correct = 0
    for start in range(0, len(idx), decode_batch):
        rows = idx[start:start + decode_batch]
        rows_t = torch.as_tensor(rows, dtype=torch.long, device=device)
        produced = model.greedy_decode(data.src[rows_t], data.src_len[rows_t], data.max_decode_len).tolist()
        for row, seq in zip(rows, produced):
            out = []
            for tok in seq:
                if tok == eos or tok == pad:
                    break
                out.append(tok)
            if out == data.gold[int(row)]:
                correct += 1
    return correct / len(idx) if len(idx) > 0 else 0.0


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


def _run_one_fold(data, train_idx, test_idx, spec, h, device, seed):
    """
    Trains one seq2seq on `train_idx` and returns (best held-out exact-match, stopping epoch).

    Early stopping watches the held-out *loss* (which decreases smoothly from the first
    epoch) rather than exact-match: exact-match sits at exactly 0 during warm-up until it
    jumps, so a patience clock on exact-match would kill any run that has not yet crossed
    that transition. Loss-based stopping tracks real convergence; we still *report* the
    best exact-match, the quantity of interest.
    """
    torch.manual_seed(seed)
    model = Seq2Seq(spec, h).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=h.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=spec.tgt_pad_id)

    train_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    n_train = train_t.numel()
    eval_batch = max(256, h.batch_size)

    best_acc = 0.0
    best_val = float("inf")
    stale = 0
    stopped_epoch = 0
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

            best_acc = max(best_acc, _exact_match(model, data, test_idx, spec, decode_batch=eval_batch))

            val = _val_loss(model, data, test_idx, spec, batch=eval_batch)
            if val < (best_val - _MIN_DELTA):
                best_val = val
                stale = 0
            else:
                stale += 1
                if stale >= h.patience:
                    break
    return best_acc, stopped_epoch


def compositionality(pairs, spec, hparams, device, seed=0, n_folds=5, verbose=False, return_epochs=False):
    """
    Single-shuffle `n_folds`-fold cross-validation of the exact-match probe.

    pairs:   list[(src_ids: list[int], tgt_ids: list[int])]; tgt_ids are the Polish
             predicate tokens (without BOS/EOS).
    spec:    Spec with vocab sizes and special-token ids.
    hparams: HParams.
    Returns the mean over folds of each fold's best held-out exact-match accuracy,
    a float in [0, 1] (NaN if there are fewer pairs than folds). If `return_epochs`,
    returns (mean_score, per_fold_stopping_epochs) instead.
    """
    n = len(pairs)
    if n < n_folds:
        return (float("nan"), []) if return_epochs else float("nan")

    data = _Data(pairs, spec, device)

    # One shuffle, then a partition into n_folds (near-)equal parts; each part is test once.
    perm = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(perm, n_folds)

    scores, epochs = [], []
    for f in range(n_folds):
        test_idx = folds[f]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])
        score, stopped_epoch = _run_one_fold(data, train_idx, test_idx, spec, hparams, device, seed=(seed + f))
        scores.append(score)
        epochs.append(stopped_epoch)
        if verbose:
            print(f"    [comp] fold {f}: best test exact-match = {score:.4f} (stopped at epoch {stopped_epoch})", flush=True)

    mean_score = float(np.mean(scores))
    return (mean_score, epochs) if return_epochs else mean_score


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
# Random hyperparameter search ( --do compositionality_search )
# ----------------------------------------------------------------------------- #

# Reasonable bounds for the probe.
SEARCH_SPACE = {
    "embed_dim":  [64],
    "hidden_dim": [64],
    "num_layers": [1],
    "dropout":    ("uniform", 0.0, 0.4),
    "lr":         ("loguniform", 3e-4, 3e-3),
    "batch_size": [512],
}


def _sample_hparams(rng, max_epochs, patience):
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
    )


def main(global_args=None, remaining_args=None):
    # Local imports to avoid an import cycle (predicate_signalling_game -> games -> alexBeth -> here).
    from ..predicate_signalling_game import get_args
    from ..utils.predicate_data import get_data_loader

    args = get_args(remaining_args)
    device = args.device

    dataset = get_data_loader(args)
    pairs, spec = reverse_polish_pairs(dataset)
    print(f"[comp-search] {len(pairs)} predicates; tuning on the reverse-Polish control language.", flush=True)

    rng = np.random.default_rng(args.comp_search_seed)
    best_score, best_h = None, None
    history = []
    for trial in range(args.comp_search_trials):
        h = _sample_hparams(rng, args.comp_max_epochs, args.comp_patience)
        score, stopped_epochs = compositionality(pairs, spec, h, device, seed=args.comp_search_seed, return_epochs=True)
        history.append({"score": score, "stopped_epochs": stopped_epochs, **asdict(h)})
        if (best_score is None) or (score > best_score):
            best_score, best_h = score, h
        print(f"[comp-search] trial {trial + 1}/{args.comp_search_trials}: "
              f"score={score:.4f}  best={best_score:.4f}  stopped@{stopped_epochs}  {h}", flush=True)

    print(f"\n[comp-search] best score = {best_score:.4f}")
    print("[comp-search] best hyperparameters (copy as CLI flags):")
    for k, v in asdict(best_h).items():
        print(f"    --comp_{k} {v}")

    if getattr(args, "comp_search_out", None) is not None:
        payload = {"best_score": best_score, "best_hparams": asdict(best_h), "history": history}
        with open(args.comp_search_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[comp-search] wrote results to {args.comp_search_out}")

    try:
        dataset.close()
    except Exception:
        pass

    return best_score, best_h
