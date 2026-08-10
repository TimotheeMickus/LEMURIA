#!/usr/bin/env python

from datetime import datetime
import uuid
import random

import torch
import json
import numpy as np

from .games import AlexBeth, AlexBethPopulation
from .utils.predicate_data import get_data_loader
from .utils.misc import path_replace
from .utils.log import AutoLogger, build_run_name, setup_wandb_logging, finish_wandb_logging

def main(global_args=None, remaining_args=None):
    args = get_args(remaining_args)
    do(args)

def do(args):
    summary_dir = path_replace(args.summary, '[now]', datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) # PosixPath
    models_dir = path_replace(args.models, '[summary]', summary_dir) # PosixPath
    # Tag used in run folder names to avoid collisions across launches.
    args.run_tag = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4().hex[:6]}"

    # Snapshot of the arguments for run naming, taken before the loop injects
    # derived values (num_predicates, graph_d_model, ...) into `args`, so every
    # run gets a consistent name.
    name_defaults = args._arg_defaults
    name_args = dict(vars(args))

    for run in range(args.runs):
        print(f'Run {run}', flush=True)
        
        run_seed = (int(args.seed) + int(run)) if(args.seed is not None) else None
        if(args.seed is not None):
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(run_seed)
            print(f"[seed] run={run} seed={run_seed}", flush=True)
        args.run_seed = run_seed

        run_name = build_run_name(name_args, run, name_with=args.name_with, defaults=name_defaults)
        run_summary_dir = summary_dir / run_name
        run_models_dir = models_dir / run_name
        signal_dump_dir = run_summary_dir if(args.dump_signals is not None) else None

        # Loads the data.
        data_loader = get_data_loader(args)
        nb_workers, nb_prefetch = (2, 2) # TODO There should be command line arguments for these.
        if((nb_workers > 0) and (nb_prefetch > 0)): data_loader.turnAsynchronous(nb_workers=nb_workers, nb_prefetch=nb_prefetch)

        # Size of the signal space (number of predicates).
        # TIMOTHÉE Why do you have to inject so much stuff in `args`? (TODO I think that you should not.)
        args.num_predicates = len(data_loader.predicates)

        # For candidate encoder
        args.node_vocab_size = len(data_loader.graph_converter.node_i2s)
        args.node_padding_idx = data_loader.graph_converter.node_s2i[data_loader.graph_converter.padding_token]
        args.edge_vocab_size = len(data_loader.graph_converter.edge_i2s)
        if(args.candidate_encoder == 'graph_transformer'):
            if(args.graph_d_model is None):
                args.graph_d_model = args.hidden_size

            # Adjusts graph defaults based on model size.
            if(args.graph_d_hidden is None):
                args.graph_d_hidden = args.graph_d_model * 2
            elif((args.graph_d_hidden < args.graph_d_model) and (not args.quiet)):
                raise ValueError(f"graph_d_hidden ({args.graph_d_hidden}) < graph_d_model ({args.graph_d_model}); consider >= {args.graph_d_model}.")

            if(args.graph_d_model % args.graph_num_heads != 0):
                raise ValueError(f"graph_d_model ({args.graph_d_model}) % graph_num_heads ({args.graph_num_heads}) must == 0.")
            
            # # Picks the largest divisor <= min(8, d_model) to avoid head mismatch.
            # max_heads = min(8, args.graph_d_model)
            # safe_heads = next((h for h in range(max_heads, 0, -1) if args.graph_d_model % h == 0), 1)
            # if not args.quiet:
            #     print(f"[warn] graph_num_heads ({args.graph_num_heads}) does not divide graph_d_model ({args.graph_d_model}); using {safe_heads}.", flush=True)
            # args.graph_num_heads = safe_heads
        
        autologger = AutoLogger(base_alphabet_size=args.base_alphabet_size, data_loader=data_loader, display=args.display, steps_per_epoch=args.steps_per_epoch, log_debug=args.log_debug, log_lang_progress=args.log_lang_progress, log_entropy=args.log_entropy, device=args.device, no_summary=args.no_summary, summary_dir=run_summary_dir, default_period=args.logging_period,) # The `data_loader` is needed because the number of categories is sometimes used.

        wandb_run = setup_wandb_logging(
            autologger=autologger,
            enabled=args.wandb,
            project=args.wandb_project,
            run_name=run_name,
            args=args,
        )

        if(not args.no_summary):
            run_summary_dir.mkdir(parents=True, exist_ok=True)
            with open(run_summary_dir / "hparams.json", "w") as f:
                json.dump(vars(args), f, indent=2, default=str)
        
        if(args.save_every > 0): run_models_dir.mkdir(parents=True, exist_ok=True)

        # Creates the model.
        # Passing --pop_size or --pop_reset_period selects the population game; the basic
        # AlexBeth game is exactly AlexBethPopulation(pop_size=1-1, pop_reset_period=0-0).
        if((args.pop_size is not None) or (args.pop_reset_period is not None)):
            model = AlexBethPopulation(args, autologger, data_loader, signal_dump_dir)
        else:
            model = AlexBeth(args, autologger, data_loader, signal_dump_dir)
        model = model.to(args.device)

        if(args.detect_anomaly):
            torch.autograd.set_detect_anomaly(True)

        # Pretrains the agents on the predicate/candidate satisfaction task, if a pretrainer is
        # configured (--pretrain). No-op otherwise. Reinitialized agents are re-pretrained by the
        # population reset hook (PopulationMixin._on_reinitialized).
        model.run_pretraining()

        # Runs the run.
        if(args.save_every > 0): model.save(run_models_dir / "model_e-1.pt")

        print(f"[{datetime.now()}] training start…", flush=True)

        total_trained_epochs = args.epochs
        model.train_agents(args.epochs, args.steps_per_epoch, data_loader, run_models_dir=run_models_dir, save_every=args.save_every, start_epoch_index=0)
        if(args.keep_training):
            while(model.max_perf < 1.0):
                user_input = input(f"Current accuracy: {model.max_perf:.6f}.\nAdd epochs?: ").strip()
                try:
                    extra_epochs = int(user_input)
                except ValueError:
                    print("Please enter an integer.")
                    continue

                if(extra_epochs <= 0): break

                total_trained_epochs += extra_epochs
                model.epochs = total_trained_epochs
                model.train_agents(extra_epochs, args.steps_per_epoch, data_loader, run_models_dir=run_models_dir, save_every=args.save_every, start_epoch_index=(total_trained_epochs - extra_epochs))

        data_loader.close()

        # If the model has not reached a certain performance threshold during training, an empty "FAILURE" file is created.
        performance_threshold = 0.9
        if(model.max_perf < performance_threshold):
            print("This run has failed (max perf = {model.max_perf} < {performance_threshold}).")
            if(not args.no_summary):
                filename = run_summary_dir / "FAILURE"
                open(filename, 'a').close()

        if(args.dump_predicate_perf):
            # Exports predicate-level performance tables and tie them to W&B as one artifact.
            model.dump_predicate_performance(run_summary_dir, wandb_run=wandb_run)
        
        if(args.dump_eval_metrics):
            model.dump_eval_metrics(run_summary_dir, wandb_run=wandb_run)
        
        finish_wandb_logging(wandb_run)


import argparse
import pathlib
import pprint

import socket # for `gethostname`

def get_args(remaining_args=None):
    from .utils import cli
    from .utils.predicate_data import add_data_args
    from .utils.modules import add_graph_encoder_args
    from .games.population import add_population_args
    from .games.pretraining import add_pretraining_args
    from .eval.compositionality import add_compositionality_args

    arg_parser = argparse.ArgumentParser()

    default_data_set = pathlib.Path('data') / 'concon'
    default_models = pathlib.Path('[summary]') / 'models'
    default_summary = pathlib.Path('runs') / 'psg' / ('[now]_' + socket.gethostname())

    # Dataset, incl. the depth curriculum (defined in utils/predicate_data.py).
    add_data_args(arg_parser)

    # Saving/logging destinations (shared) + predicate-specific dump/naming knobs.
    save = cli.add_save_args(arg_parser, default_summary, default_models)
    save.add_argument('--dump_signals', help='dump signals: "last" (default), "all", "when_hike", or "when_hike_strict"', choices=['last', 'all', 'when_hike', 'when_hike_strict'], nargs='?', const='last', default=None)
    save.add_argument('--name_with', help="build the run folder/W&B name from these argument names, e.g. --name_with [min_depth, max_depth]  ->  'mind=1__maxd=3__t=<timestamp>__run=0'. Accepts brackets/commas/spaces (quote it if it contains spaces, e.g. '[min_depth, max_depth]'). A launch timestamp and the run index are always appended. If unset, the name is generated automatically from the non-default arguments.", nargs='+', default=None)

    cli.add_display_args(arg_parser)

    # Reward (shared; --len_penalty default is game-specific) + the AlexBeth entropy betas.
    reward = cli.add_reward_args(arg_parser, len_penalty_default=0.0)
    reward.add_argument('--voc_penalty', help='coefficient for the vocabulary usage penalty (meaning depends on --voc_penalty_mode; the two modes are on different scales and need separate tuning)', default=0.0, type=float)
    reward.add_argument('--voc_penalty_mode', help="how the vocabulary penalty is applied: 'reward' is the original amortised inverse-frequency penalty subtracted from the REINFORCE reward; 'aux' is a differentiable group-sparsity penalty on the batch symbol marginal, added directly to the asker loss", choices=['reward', 'aux'], default='reward')
    reward.add_argument('--voc_penalty_p', help="exponent p in (0, 1] for the 'aux' group-sparsity penalty sum_a m_a**p over content symbols (smaller p -> closer to an L0 support count and more aggressive pruning; p=1 has no effect on the simplex). Unused when --voc_penalty_mode is 'reward'", default=0.5, type=float)
    reward.add_argument('--beta_asker', help='asker entropy penalty coefficient', type=float, default=0.01)
    reward.add_argument('--beta_retriever', help='retriever entropy penalty coefficient', type=float, default=0.0)

    cli.add_language_args(arg_parser)
    cli.add_perf_args(arg_parser)

    group = arg_parser.add_argument_group(title='Architecture', description='arguments relative to model & game architecture')
    group.add_argument('--shared', '-s', help='share the image encoder and the symbol embeddings among each couple of Alice\u00b7s and Bob\u00b7s', action='store_true')
    group.add_argument('--hidden_size', help='dimension of hidden representations', type=int, default=50)
    group.add_argument('--blind_candidates', help='debug: retriever ignores candidate features (scores become constant across candidates)', action='store_true')
    group.add_argument('--blind_signal', help='debug: retriever ignores signal embedding', action='store_true')

    # Candidate (graph) encoder (defined in utils/modules.py, next to the encoder).
    add_graph_encoder_args(arg_parser)

    # Population variant (defined in games/population.py).
    add_population_args(arg_parser, 'askers', 'retrievers')

    # Training loop (shared) + a predicate-specific training knob.
    training = cli.add_training_args(arg_parser)
    training.add_argument('--keep_training', help='after training, if max accuracy is below 1.0, interactively ask for extra epochs (0 to stop)', action='store_true')

    # Pretraining: the shared knobs (defined in games/pretraining.py) + the predicate toggle.
    pre = add_pretraining_args(arg_parser, frozen_desc='the asker predicate encoder / retriever candidate encoder')
    pre.add_argument('--pretrain', help='pretrain each agent on the predicate/candidate satisfaction task before training: askers keep their predicate encoder and are temporarily paired with a candidate encoder (same --candidate_encoder choice as retrievers), retrievers keep their candidate encoder and are temporarily paired with a predicate embedding. Temporary modules are discarded afterwards. Reinitialized agents (population reset) are pretrained again.', action='store_true')

    group = arg_parser.add_argument_group(title='Eval', description='arguments relative to evaluation routines')
    group.add_argument('--correct_only', help='analyse the language constisting of the signals produced in successful rounds only', action='store_true')
    group.add_argument('--jaccard', help='enable Jaccard-based topsim metrics (more expensive)', action='store_true')
    group.add_argument('--eval_oracle_language', help="debug feature: during fancy evaluation, replace the emergent language with a known-compositional 'oracle'/control language (an upper-bound sanity check, not part of normal runs). In AlexBeth this is the reverse-Polish encoding of the predicate. Applied to the compositionality probe and topographic similarity; signal dumping and scrambling resistance keep using the emergent language.", action='store_true')
    group.add_argument('--dump_predicate_perf', help='dump per-predicate performance tables and log them as a W&B artifact', action='store_true')
    group.add_argument('--dump_eval_metrics', help='dump per-eval-call aggregate metrics CSV and log it as a W&B artifact', action='store_true')

    cli.add_debug_arg(arg_parser)

    group = arg_parser.add_argument_group(title='WandB', description='arguments relative to Weights & Biases logging')
    group.add_argument('--wandb', help='enable Weights & Biases logging', action='store_true')
    group.add_argument('--wandb_project', help='W&B project name', default='lemuria', type=str)

    # Compositionality probe + search (defined in eval/compositionality.py).
    add_compositionality_args(arg_parser)

    args = arg_parser.parse_args(remaining_args)

    # Snapshot of the parser defaults, used by `build_run_name` so that a run name only advertises the arguments that were actually changed.
    args._arg_defaults = vars(arg_parser.parse_args([]))

    if(args.debug and (not args.log_debug)): args.log_debug = True

    if(not args.quiet):
        print("command-line arguments:")
        pprint.pprint(vars(args), indent=4)

    return args


if(__name__ == "__main__"):
    do(global_args=None, remaining_args=None)
