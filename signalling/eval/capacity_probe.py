"""
Predicate/candidate satisfaction capacity probe.

In the predicate signalling game, the retriever decides whether a candidate satisfies
a predicate by encoding both into the same H-dimensional space and taking their dot
product (see `Retriever.aux_forward`, and `PredicateCandidateClassifier` in
`games/pretraining.py`, which scores (predicate, candidate) pairs the same way with a
direct predicate embedding standing in for the emergent signal). That scheme is only
representationally viable if H (`--hidden_size`) is large enough for the dataset at
hand -- e.g. separating N roughly-independent predicates from each other already needs
H = Omega(log N), and fitting arbitrary (predicate, candidate) truth tables needs more
still.

`--do capacity_probe` isolates exactly that sub-problem: it builds the predicate/
candidate dataset (the same --properties/--max_depth/... knobs as
`predicate_signalling_game`) and trains *only* a (predicate, candidate) satisfaction
classifier -- no sender/receiver game, no communication bottleneck, no population --
directly on the satisfaction task. If train/test accuracy plateaus below ~1.0,
`--hidden_size` is too small for this dataset regardless of what the communication
game could otherwise achieve; if it reaches ~1.0, hidden_size is not the bottleneck
and any remaining learning failure in the full game lies elsewhere (the discrete
channel, the training dynamics, ...).

`--scorer` selects how the encoded predicate `p` and candidate `o` (each in R^H) are
combined into a logit:
  * 'dot' (default) -- PredicateCandidateClassifier's plain dot product `p . o`, i.e.
    the retriever's own scoring rule (see `Retriever.aux_forward` / `games/pretraining.py`).
    This is a bilinear form of `p` and `o`; note that since `o` is itself freely learned
    by the candidate encoder, *any* bilinear form `p^T M o` collapses back to a plain dot
    product (the encoder just absorbs `M` into its last layer) -- so 'dot' already speaks
    for that whole family, and there is no separate 'bilinear' option.
  * 'mlp' -- `u . ReLU(... ReLU(W_1 [p, o] + b_1) ...) + b` (an MLP of `--scorer_layers`
    hidden layers of width `--scorer_hidden_size` over the concatenation `[p, o]`).
    Strictly more expressive than 'dot' (a universal approximator given enough width),
    so it separates two possible causes of low accuracy: not enough embedding dimension
    (both scorers would then struggle) vs. the dot product's bilinear form itself being
    the wrong inductive bias for this dataset (only 'dot' would struggle).
In both cases the returned score is a logit (sigmoid applied by the loss/accuracy code,
not inside the model).
"""

import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import cli
from ..utils.predicate_data import add_data_args, get_data_loader
from ..utils.modules import (
    add_graph_encoder_args,
    build_candidate_encoder_from_args,
    build_predicate_encoder_from_args,
)
from ..utils.misc import build_optimizer
from ..utils.log import Progress
from ..games.pretraining import PredicateCandidateClassifier


class MLPScorer(nn.Module):
    """Scores (predicate, candidate) pairs with an MLP over their concatenation, instead of
    `PredicateCandidateClassifier`'s dot product:
        u . ReLU(... ReLU(W_1 [p, o] + b_1) ...) + b
    (`num_layers` hidden ReLU layers of width `mlp_hidden_size`, then a linear readout to a scalar
    logit). See the module docstring for why this is a strictly richer scorer family than the dot
    product, given that `p` and `o` are themselves freely learned.

    predicate_encoder: predicate_idx (batch,)               -> (batch, H)
    candidate_encoder: candidate tensors                    -> (batch, num_candidates, H)
    forward:                                                -> (batch, num_candidates) logits
    """
    def __init__(self, predicate_encoder, candidate_encoder, hidden_size, mlp_hidden_size, num_layers):
        super().__init__()
        self.predicate_encoder = predicate_encoder
        self.candidate_encoder = candidate_encoder

        assert num_layers >= 1, "MLPScorer needs at least one hidden layer (otherwise it is just a bilinear/dot scorer)."
        layers = []
        in_size = 2 * hidden_size
        for _ in range(num_layers):
            layers += [nn.Linear(in_size, mlp_hidden_size), nn.ReLU()]
            in_size = mlp_hidden_size
        layers.append(nn.Linear(in_size, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, predicate_idx, candidate_tensors):
        encoded_predicate = self.predicate_encoder(predicate_idx)          # (batch, H)
        encoded_candidates = self.candidate_encoder(**candidate_tensors)   # (batch, num_candidates, H)
        num_candidates = encoded_candidates.size(1)
        expanded_predicate = encoded_predicate.unsqueeze(1).expand(-1, num_candidates, -1) # (batch, num_candidates, H)
        pair = torch.cat([expanded_predicate, encoded_candidates], dim=-1)                 # (batch, num_candidates, 2H)
        return self.mlp(pair).squeeze(-1)                                                  # (batch, num_candidates)


# Command-line arguments specific to the capacity probe. Kept next to the probe code
# that reads them. Returns the argparse group.
def add_capacity_probe_args(parser):
    group = parser.add_argument_group(
        title='Capacity probe',
        description="knobs for `--do capacity_probe`, which trains only a predicate/candidate "
                    "satisfaction classifier (predicate embedding <-> candidate encoder, --scorer), "
                    "without any communication game.",
    )
    group.add_argument('--hidden_size', help='dimension of the predicate/candidate embedding space', type=int, default=50)
    group.add_argument('--scorer', help="how the encoded predicate and candidate are combined into a logit: 'dot' (plain dot product, the retriever's own rule) or 'mlp' (MLP over their concatenation; see --scorer_hidden_size / --scorer_layers)", choices=['dot', 'mlp'], default='dot')
    group.add_argument('--scorer_hidden_size', help="hidden layer width for --scorer mlp (defaults to --hidden_size)", type=int, default=None)
    group.add_argument('--scorer_layers', help="number of hidden layers for --scorer mlp", type=int, default=1)
    group.add_argument('--epochs', help='number of epochs', type=int, default=100)
    group.add_argument('--steps_per_epoch', help='number of steps per epoch', type=int, default=1000)
    group.add_argument('--learning_rate', help='learning rate', type=float, default=0.01)
    group.add_argument('--seed', help='random seed', type=int, default=None)
    return group


# Lean argument parser for `--do capacity_probe`. Like `compositionality_search`, it only needs the
# dataset knobs, a device, and its own hyperparameters -- not the whole predicate-game parser.
def get_args(remaining_args):
    parser = argparse.ArgumentParser(
        prog="signalling --do capacity_probe",
        description="Trains only the predicate/candidate satisfaction classifier (predicate embedding "
                    "<-> candidate encoder, dot product, sigmoid) with no communication game, to check "
                    "whether --hidden_size is large enough to represent the dataset's (predicate, "
                    "candidate) truth table at all.",
    )
    add_data_args(parser)          # the predicate/candidate dataset
    add_graph_encoder_args(parser) # --candidate_encoder and its graph-transformer knobs
    cli.add_perf_args(parser)      # --device
    cli.add_display_args(parser)   # --display, --quiet, ...
    add_capacity_probe_args(parser)
    return parser.parse_args(remaining_args)


def main(global_args=None, remaining_args=None):
    args = get_args(remaining_args)
    do(args)


def do(args):
    if(args.seed is not None):
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    dataset = get_data_loader(args)
    # `get_batch` needs the worker pool started (see `Dataset._generate_batch`'s
    # `failure_based_distribution` argument, only materialized by `_additional_resources` inside a
    # worker); `predicate_signalling_game.do` does the same. TODO There should be command line
    # arguments for these.
    nb_workers, nb_prefetch = (2, 2)
    if((nb_workers > 0) and (nb_prefetch > 0)): dataset.turnAsynchronous(nb_workers=nb_workers, nb_prefetch=nb_prefetch)

    # Same derived args the predicate game injects before building its encoders (see
    # `predicate_signalling_game.do`).
    args.num_predicates = len(dataset.predicates)
    args.node_vocab_size = len(dataset.graph_converter.node_i2s)
    args.node_padding_idx = dataset.graph_converter.node_s2i[dataset.graph_converter.padding_token]
    args.edge_vocab_size = len(dataset.graph_converter.edge_i2s)
    if(args.candidate_encoder == 'graph_transformer'):
        if(args.graph_d_model is None):
            args.graph_d_model = args.hidden_size
        if(args.graph_d_hidden is None):
            args.graph_d_hidden = args.graph_d_model * 2

    predicate_encoder = build_predicate_encoder_from_args(args).to(args.device)
    candidate_encoder = build_candidate_encoder_from_args(args).to(args.device)
    if(args.scorer == 'dot'):
        model = PredicateCandidateClassifier(predicate_encoder, candidate_encoder).to(args.device)
        scorer_desc = "dot"
    else:
        scorer_hidden_size = args.scorer_hidden_size or args.hidden_size
        model = MLPScorer(predicate_encoder, candidate_encoder, args.hidden_size, scorer_hidden_size, args.scorer_layers).to(args.device)
        scorer_desc = f"mlp(hidden_size={scorer_hidden_size}, layers={args.scorer_layers})"
    optimizer = build_optimizer(model.parameters(), args.learning_rate)

    print(
        f"[{datetime.now()}] capacity probe: hidden_size={args.hidden_size}, scorer={scorer_desc}, "
        f"{args.num_predicates} predicates, candidate_encoder={args.candidate_encoder!r}.",
        flush=True,
    )

    def _beth_input(batch):
        return {'node_idx': batch.node_idx, 'edge_idx': batch.edge_idx, 'graph_size': batch.graph_sizes}

    total_items = 0
    for epoch_index in range(args.epochs):
        model.train()
        pbar = Progress.get_progress_cls(args.display)(args.steps_per_epoch, epoch_index, logged_items={'L', 'acc'})
        epoch_hits, epoch_items = 0., 0.
        with pbar:
            for _ in range(args.steps_per_epoch):
                optimizer.zero_grad()

                batch = dataset.get_batch(size=args.batch_size, data_type='train')
                scores = model(batch.predicate_idx, _beth_input(batch)) # (batch, num_candidates)
                truth = batch.candidate_truth

                loss = F.binary_cross_entropy_with_logits(scores, truth, reduction='mean')
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    hits = ((scores >= 0.0).float() == truth).float().sum().item() # sigmoid(score) >= 0.5 <=> score >= 0
                epoch_hits += hits
                epoch_items += truth.numel()
                total_items += batch.size
                pbar.update(L=loss.item(), acc=(epoch_hits / epoch_items))

        # Evaluation. As in `AlexBethPretrainer`, there is no held-out split (candidates are freshly
        # sampled every time and `data_type` only picks the predicate split), so this is same-
        # distribution but still an independent estimate.
        with torch.no_grad():
            model.eval()
            test_loss, test_hits, test_items = 0., 0., 0.
            for _ in range(1 + (args.steps_per_epoch // 10)):
                batch = dataset.get_batch(size=args.batch_size, data_type='test')
                scores = model(batch.predicate_idx, _beth_input(batch))
                truth = batch.candidate_truth
                test_loss += F.binary_cross_entropy_with_logits(scores, truth, reduction='sum').item()
                test_hits += ((scores >= 0.0).float() == truth).float().sum().item()
                test_items += truth.numel()
            test_loss = test_loss / test_items
            test_acc = test_hits / test_items
            print(f"[eval-capacity_probe epoch {epoch_index}] L={test_loss:.4f}, acc={test_acc:.4f}", flush=True)

    dataset.close()


if(__name__ == "__main__"):
    do(get_args(None))
