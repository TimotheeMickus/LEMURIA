#!/usr/bin/env python

from datetime import datetime
import uuid

import torch
import json

from .games import AlexBeth
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

    for run in range(args.runs):
        print(f'Run {run}', flush=True)

        run_name = build_run_name(args, run)
        run_summary_dir = summary_dir / run_name
        run_models_dir = models_dir / run_name
        message_dump_dir = run_summary_dir if(args.dump_message is not None) else None

        # Loads the data.
        data_loader = get_data_loader(args)
        nb_workers, nb_prefetch = (2, 2) # TODO There should be command line arguments for these.
        if((nb_workers > 0) and (nb_prefetch > 0)): data_loader.turnAsynchronous(nb_workers=nb_workers, nb_prefetch=nb_prefetch)

        # Size of the message space (number of predicates).
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
        model = AlexBeth(args, autologger, data_loader, message_dump_dir)
        model = model.to(args.device)

        if(args.detect_anomaly):
            torch.autograd.set_detect_anomaly(True)

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
            print("This run has failed.")
            if(not args.no_summary):
                filename = run_summary_dir / "FAILURE"
                open(filename, 'a').close()

        if(args.dump_predicate_perf):
            # Exports predicate-level performance tables and tie them to W&B as one artifact.
            model.dump_predicate_performance(
                run_summary_dir,
                wandb_run=wandb_run,
            )
        
        if(args.dump_eval_metrics):
            model.dump_eval_metrics(
                run_summary_dir,
                wandb_run=wandb_run,
            )
        
        finish_wandb_logging(wandb_run)


import argparse
import pathlib
import pprint

import socket # for `gethostname`

def get_args(remaining_args=None):
    arg_parser = argparse.ArgumentParser()

    default_data_set = pathlib.Path('data') / 'concon'
    default_models = pathlib.Path('[summary]') / 'models'
    default_summary = pathlib.Path('runs') / 'cbc' / ('[now]_' + socket.gethostname())

    group = arg_parser.add_argument_group(title='Data', description='arguments relative to data handling')
    group.add_argument('--properties', help='for each properties, the number of values', default='4-4', type=str)
    group.add_argument('--max_depth', help='the depth limit of the predicates considered', default=3, type=int)
    group.add_argument('--min_depth', help='minimum predicate depth to include', type=int, default=1)
    group.add_argument('--nontrivial_only', help='whether to use only predicates that are both satisfiable and falsifiable', action='store_true')
    group.add_argument('--no_negation', help='whether to allow negation in the predicates', action='store_true')
    group.add_argument('--no_conjunction', help='whether to allow conjunction in the predicates', action='store_true')
    group.add_argument('--allow_indeterminate', help='whether to allow indeterminate (neither true nor false) values in candidates', action='store_true')
    group.add_argument('--overfit', help='use a fixed small predicate/candidate pool to test memorization', action='store_true')
    group.add_argument('--batch_size', help='batch size', default=128, type=int)
    group.add_argument('--num_candidates', help='number of candidates per predicate in a batch', default=10, type=int)
    group.add_argument('--predicate_sampling', help='how candidates are sampled', choices=['random', 'difficulty'], default='random')
    group.add_argument('--candidate_sampling', help='how candidates are sampled (in particular based on their truth value distribution)', choices=['random', 'balanced'], default='balanced')

    group = arg_parser.add_argument_group(title='Save', description='arguments relative to saving models/logs')
    group.add_argument('--summary', help='the path to the TensorBoard summary for this run (\'[now]\' will be intepreted as now in the Y-m-d_H-M-S format)', default=default_summary, type=pathlib.Path)
    group.add_argument('--save_every', '-save_every', help='indicate to save the model after each __ epochs', type=int, default=0)
    group.add_argument('--models', help='the path to the saved models (\'[summary]\' will be interpreted as the value of --summary)', default=default_models, type=pathlib.Path)
    group.add_argument('--dump_message', help='dump messages: "last" (default), "all", "when_hike", or "when_hike_strict"', choices=['last', 'all', 'when_hike', 'when_hike_strict'], nargs='?', const='last', default=None)

    group = arg_parser.add_argument_group(title='Display', description='arguments relative to displayed information')
    # TODO: refactor logging: --display tqdm should be inferred from the env
    group.add_argument('--display', help='how to display the information', choices=['minimal', 'simple', 'tqdm'], default='tqdm')
    group.add_argument('--log_debug', '-ld', help='log more stuf', action='store_true')
    group.add_argument('--detect_anomaly', help='autodetect grad anomalies', action='store_true')
    group.add_argument('--no_summary', '-ns', help='do not write summaries', action='store_true')
    group.add_argument('--log_lang_progress', '-llp', help='log metrics to evaluate progress and stability of language learned', action='store_true')
    group.add_argument('--log_entropy', help='log evolution of entropy across epochs', action='store_true')
    # TODO: refactor logging: --logging_period should control the frequency of step reports when --display minimal
    group.add_argument('--logging_period', help='how often counts of logged variables are accumulated', type=int, default=10)
    # TODO: refactor logging: --quiet vs. --display quiet?
    group.add_argument('--quiet', help='display less information', action='store_true')

    group = arg_parser.add_argument_group(title='Reward', description='arguments relative to reward shaping/gradient computation')
    group.add_argument('--len_penalty', help='coefficient for the length penalty of the messages', default=0.0, type=float)
    group.add_argument('--voc_penalty', help='coefficient for the vocabulary usage penalty', default=0.0, type=float)
    group.add_argument('--use_expectation', help='use expectation of success instead of playing dice', action='store_true')
    group.add_argument('--beta_asker', help='asker entropy penalty coefficient', type=float, default=0.01)
    group.add_argument('--beta_retriever', help='retriever entropy penalty coefficient', type=float, default=0.0)
    group.add_argument("--learning_rate", help="learning rate", default=0.0001, type=float)
    group.add_argument('--grad_clipping', help='threshold for gradient clipping', default=1, type=float)
    group.add_argument('--grad_scaling', help='threshold for gradient scaling', default=None, type=float)

    group = arg_parser.add_argument_group(title='Language', description='arguments relative to language capacity')
    group.add_argument('--base_alphabet_size', help='size of the alphabet (not including special symbols)', default=10, type=int) # Previously 64. There are 32 intuitive classes of images in the data set
    group.add_argument('--max_len', help='maximum length of messages produced', default=10, type=int) # Previously 16.

    group = arg_parser.add_argument_group(title='Perfs', description='arguments relative to performances')
    # device_choices = ['cpu', 'cuda', 'mkldnn', 'opengl', 'opencl', 'ideep', 'hip', 'msnpu']
    # group.add_argument('--device', help='what to run PyTorch on (potentially available: ' + ', '.join(device_choices) + ')', choices=device_choices, default='cpu')
    group.add_argument('--device', help='what to run PyTorch on', type=torch.device, default=torch.device('cpu'))

    group = arg_parser.add_argument_group(title='Architecture', description='arguments relative to model & game architecture')
    group.add_argument('--shared', '-s', help='share the image encoder and the symbol embeddings among each couple of Alice·s and Bob·s', action='store_true')
    group.add_argument('--population', help='population size', default=None, type=int)
    group.add_argument('--reaper_step', help='population size regulator', default=None, type=int)
    group.add_argument('--hidden_size', help='dimension of hidden representations', type=int, default=50)
    group.add_argument('--candidate_encoder', help='candidate encoder type', choices=['node_averager', 'graph_transformer'], default='node_averager')
    # Graph encoder parameters (used when --candidate_encoder=graph)
    group.add_argument('--graph_num_layers', help='number of graph transformer layers', type=int, default=2)
    group.add_argument('--graph_d_model', help='graph transformer model size (defaults to hidden_size)', type=int, default=None)
    group.add_argument('--graph_num_heads', help='number of attention heads', type=int, default=4)
    group.add_argument('--graph_d_hidden', help='graph transformer feed-forward size (defaults to 2*graph_d_model)', type=int, default=None)
    group.add_argument('--graph_dropout', help='graph transformer dropout', type=float, default=0.1)
    group.add_argument('--graph_no_norm', help='disable layer norm in graph encoder', action='store_true')
    group.add_argument('--blind_candidates', help='debug: retriever ignores candidate features (scores become constant across candidates)', action='store_true')
    group.add_argument('--blind_message', help='debug: retriever ignores message embedding', action='store_true')

    group = arg_parser.add_argument_group(title='Training', description='arguments relative to training curriculum')
    group.add_argument('--use_baseline', help='use a baseline term in REINFORCE', action='store_true')
    group.add_argument('--epochs', help='number of epochs', default=100, type=int)
    group.add_argument('--steps_per_epoch', help='number of steps per epoch', default=1000, type=int)
    group.add_argument('--runs', help='number of runs', default=1, type=int)
    group.add_argument('--keep_training', help='after training, if max accuracy is below 1.0, interactively ask for extra epochs (0 to stop)', action='store_true')
    group.add_argument('--no_spigot', help='whether to replace all GradSpigot·s with usual tensor', action='store_true')
    group.add_argument('--loss_weight_temp', help='temperature parameter in the loss weighting system', default=1.0, type=float)

    group = arg_parser.add_argument_group(title='Eval', description='arguments relative to evaluation routines')
    group.add_argument('--correct_only', help='analyse the language constisting of the messages produced in successful rounds only', action='store_true')
    group.add_argument('--dump_predicate_perf', help='dump per-predicate performance tables and log them as a W&B artifact', action='store_true')
    group.add_argument('--dump_eval_metrics', help='dump per-eval-call aggregate metrics CSV and log it as a W&B artifact', action='store_true')
    
    group.add_argument('--debug', '-d', help='use this flag to change the behavior of the code to debug stuff', action='store_true')

    group = arg_parser.add_argument_group(title='WandB', description='arguments relative to Weights & Biases logging')
    group.add_argument('--wandb', help='enable Weights & Biases logging', action='store_true')
    group.add_argument('--wandb_project', help='W&B project name', default='lemuria', type=str)

    args = arg_parser.parse_args(remaining_args)
    if args.debug and not args.log_debug:
        args.log_debug = True
    if not args.quiet:
        print("command-line arguments:")
        pprint.pprint(vars(args), indent=4)
    
    return args


if(__name__ == "__main__"):
    do(global_args=None, remaining_args=None)
