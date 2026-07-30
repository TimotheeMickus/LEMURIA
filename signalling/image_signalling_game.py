#!/usr/bin/env python

from datetime import datetime
import sys
import random

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tqdm
import numpy as np

from .games import AliceBob, AliceBobPopulation, AliceBobCharlie
from .utils.image_data import get_data_loader, Batch
from .utils.misc import path_replace
from .utils.log import AutoLogger

def main(global_args=None, remaining_args=None):
    args = get_args(remaining_args)
    do(args)

def do(args):
    if(not args.data_set.is_dir()):
        print((f"Directory '{args.data_set}' not found."), flush=True)
        sys.exit()

    summary_dir = path_replace(args.summary, '[now]', datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) # PosixPath
    models_dir = path_replace(args.models, '[summary]', summary_dir) # PosixPath

    for run in range(args.runs):
        print(f'Run {run}', flush=True)
        if args.seed is not None:
            run_seed = int(args.seed) + int(run)
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(run_seed)
            print(f"[seed] run={run} seed={run_seed}", flush=True)

        run_summary_dir = summary_dir / str(run)
        run_models_dir = models_dir / str(run)
        signal_dump_dir = run_summary_dir if(args.dump_signal) else None

        # Loads the data.
        data_loader = get_data_loader(args)
        
        autologger = AutoLogger(base_alphabet_size=args.base_alphabet_size, data_loader=data_loader, display=args.display, steps_per_epoch=args.steps_per_epoch, log_debug=args.log_debug, log_lang_progress=args.log_lang_progress, log_entropy=args.log_entropy, device=args.device, no_summary=args.no_summary, summary_dir=run_summary_dir, default_period=args.logging_period,) # The `data_loader` is needed because the number of categories is sometimes used.

        if(not args.no_summary): run_summary_dir.mkdir(parents=True, exist_ok=True)
        if(args.save_every > 0): run_models_dir.mkdir(parents=True, exist_ok=True)
        
        # Creates the model.
        # Passing --pop_size or --pop_reset_period selects the population game; plain AliceBob is
        # exactly AliceBobPopulation(pop_size=1-1, pop_reset_period=0-0).
        is_population = (args.pop_size is not None) or (args.pop_reset_period is not None)
        if(args.charlie):
            assert (not is_population) # NotImplementedFeature: Charlie has no population variant.
            model = AliceBobCharlie(args, autologger, data_loader, signal_dump_dir)
        elif(is_population): model = AliceBobPopulation(args, autologger, data_loader, signal_dump_dir)
        else:
            model = AliceBob(args, autologger, data_loader, signal_dump_dir)
        model = model.to(args.device)

        if(args.detect_anomaly):
            torch.autograd.set_detect_anomaly(True)

        # Pretrains the agents' CNNs, if a pretrainer is configured (--pretrain_CNNs). No-op
        # otherwise. Returns {name: pretrained_model} (or None), used by --detect_outliers.
        pretrained_models = model.run_pretraining()

        if(args.detect_outliers): # Might not work for all pretraining methods (in fact, we are expecting a MultiHeadsClassifier). To have a more general method, record the loss for all instances, then select the ones that are far from the mean
            if(pretrained_models is None):
                raise ValueError("--detect_outliers requires pretraining; pass --pretrain_CNNs.")
            (pretrained_name, pretrained_model), *_ = list(pretrained_models.items())
            print(pretrained_name)

            outliers = []
            with torch.no_grad():
                batch_size = args.batch_size
                max_datapoints = 2 ** 15
                n = data_loader.size(data_type='any', no_evaluation=False) # We would like to see all datapoints
                if((n is None) or (n > max_datapoints)):
                    print('The dataset is too big, so we are only going to be looking at %i datapoints.' % max_datapoints)
                    n = max_datapoints
                nb_batch = int(np.ceil(n / batch_size))
                for batch_i in range(nb_batch):
                    datapoints = [data_loader.get_datapoint(i) for i in range((batch_size * batch_i), min((batch_size * (batch_i + 1)), n))]
                    batch = Batch(size=batch_size, original=[], target=[x.toInput(keep_category=True, device=args.device) for x in datapoints], base_distractors=[])
                    hits, losses = pretrained_model.forward(batch)

                    misses = 0 # Will be a vector with one value (number of misses over all heads) per element in the batch
                    for x in hits: misses += (1 - x.cpu().numpy())

                    for i, miss in enumerate(misses):
                        if(miss == 0.0): continue
                        outliers.append((miss, datapoints[i]))
                        #print('Ahah! Datapoint idx=%i (category %s) has a high miss of %s!' % (datapoints[i].idx, datapoints[i].category, miss))

            outliers.sort(key=(lambda x: x[0]), reverse=True)
            print('%i outliers (%s%%)' % (len(outliers), (100 * len(outliers) / n)))
            for i in range(len(outliers)): #range(min(len(outliers), 1000)):
                miss, datapoint = outliers[i]
                print('%i - %i' % (datapoint.idx, miss))

            sys.exit(0)

        # Runs the run.
        if(args.save_every > 0): model.save(run_models_dir / ("model_e%i.pt" % -1))

        print(("[%s] training start…" % datetime.now()), flush=True)

        model.train_agents(args.epochs, args.steps_per_epoch, data_loader, run_models_dir=run_models_dir, save_every=args.save_every)
        
        # If the model has not reached a certain performance threshold during training, an empty "FAILURE" file is created.
        performance_threshold = 0.6
        if(model.max_perf < performance_threshold):
            print("This run has failed (max perf = {model.max_perf} < {performance_threshold}).")
            filename = run_summary_dir / "FAILURE"
            open(filename, 'a').close()


import argparse
import os
import pathlib
import pprint
import sys


import socket # for `gethostname`
import torch # for device
from datetime import datetime

def get_args(remaining_args=None):
    from .utils import cli
    from .utils.image_data import add_data_args
    from .utils.modules import add_cnn_args
    from .games.population import add_population_args
    from .games.pretraining import add_pretraining_args

    arg_parser = argparse.ArgumentParser()

    default_data_set = pathlib.Path('data') / 'concon'
    default_models = pathlib.Path('[summary]') / 'models'
    default_summary = pathlib.Path('runs') / 'cbc' / ('[now]_' + cocket.gethostname())

    # Dataset (defined in utils/image_data.py, next to the loader that reads these).
    add_data_args(arg_parser, default_data_set)

    # Saving/logging destinations (shared) + an image-specific dump switch.
    save = cli.add_save_args(arg_parser, default_summary, default_models)
    save.add_argument('--dump_signal', help='whether to regularly save the signals in a file', action="store_true")

    # Display (shared), with the image-only logging knobs.
    cli.add_display_args(arg_parser, image_logging=True)

    # Reward (shared; --len_penalty default is game-specific) + the AliceBob entropy betas.
    reward = cli.add_reward_args(arg_parser, len_penalty_default=0.01)
    reward.add_argument('--beta_sender', help='sender entropy penalty coefficient', type=float, default=0.01)
    reward.add_argument('--beta_receiver', help='sender entropy penalty coefficient', type=float, default=0.001)

    cli.add_language_args(arg_parser)
    cli.add_perf_args(arg_parser)

    group = arg_parser.add_argument_group(title='Architecture', description='arguments relative to model & game architecture')
    group.add_argument('--shared', '-s', help='share the image encoder and the symbol embeddings among each couple of Alice\u00b7s and Bob\u00b7s', action='store_true')
    group.add_argument('--charlie', '-c', help='add adversary drawing agent', action='store_true')
    group.add_argument('--hidden_size', help='dimension of hidden representations', type=int, default=50)

    # Population variant (defined in games/population.py).
    add_population_args(arg_parser, 'senders', 'receivers')

    # Training loop (shared) + the image-only GradSpigot switch (read by AliceBobCharlie).
    training = cli.add_training_args(arg_parser)
    training.add_argument('--no_spigot', help='whether to replace all GradSpigot\u00b7s with usual tensor', action='store_true')

    # Convolutional structure (defined in utils/modules.py, next to the cnn factories).
    add_cnn_args(arg_parser)

    # Pretraining: the shared knobs (defined in games/pretraining.py) + the image-specific ones.
    pre = add_pretraining_args(arg_parser, frozen_desc='the pretrained CNNs')
    pre.add_argument('--pretrain_CNNs', help='pretrain CNNs on specified task', type=str, choices=['category-wise', 'feature-wise', 'auto-encoder'])
    pre.add_argument('--pretrain_CNNs_on_eval', help='pretrain CNNs on classification', action='store_true')
    pre.add_argument('--detect_outliers', help='if pretraining, then after, the trained model analyses the dataset in order to detect problems', action='store_true')
    pre.add_argument('--autoencode_receiver_inputs', help='run all receiver image inputs through a pretrained autoencoder', action='store_true')

    group = arg_parser.add_argument_group(title='Eval', description='arguments relative to evaluation routines')
    group.add_argument('--correct_only', help='analyse the language constisting of the signals produced in successful rounds only', action='store_true')
    group.add_argument('--eval_oracle_language', help="debug feature: during fancy evaluation, replace the emergent language with a known-compositional 'oracle'/control language (an upper-bound sanity check, not part of normal runs). In AliceBob this is one unique symbol per (concept, value) of the category. Applied to topographic similarity, the entropy stats, and the decision tree; signal dumping and scrambling resistance keep using the emergent language.", action='store_true')

    cli.add_debug_arg(arg_parser)

    args = arg_parser.parse_args(remaining_args)
    if not args.quiet:
        print("command-line arguments:")
        pprint.pprint(vars(args), indent=4)

    return args


if(__name__ == "__main__"):
    do(global_args=None, remaining_args=None)
