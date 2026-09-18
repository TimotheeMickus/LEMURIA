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
`predicate_signalling_game`) and trains *only* the PredicateCandidateClassifier --
no sender/receiver game, no communication bottleneck, no population -- directly on the
satisfaction task. If train/test accuracy plateaus below ~1.0, `--hidden_size` is too
small for this dataset regardless of what the communication game could otherwise
achieve; if it reaches ~1.0, hidden_size is not the bottleneck and any remaining
learning failure in the full game lies elsewhere (the discrete channel, the training
dynamics, ...).
"""

import argparse
from datetime import datetime

import torch
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


# Command-line arguments specific to the capacity probe. Kept next to the probe code
# that reads them. Returns the argparse group.
def add_capacity_probe_args(parser):
    group = parser.add_argument_group(
        title='Capacity probe',
        description="knobs for `--do capacity_probe`, which trains only the predicate/candidate "
                    "satisfaction classifier (predicate embedding <-> candidate encoder, dot product, "
                    "sigmoid), without any communication game.",
    )
    group.add_argument('--hidden_size', help='dimension of the shared predicate/candidate embedding space', type=int, default=50)
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
    model = PredicateCandidateClassifier(predicate_encoder, candidate_encoder).to(args.device)
    optimizer = build_optimizer(model.parameters(), args.learning_rate)

    print(
        f"[{datetime.now()}] capacity probe: hidden_size={args.hidden_size}, "
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
