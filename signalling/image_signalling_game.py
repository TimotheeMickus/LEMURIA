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
from .games.pretraining import CNNPretrainable
from .utils.image_data import get_data_loader, Batch
from .utils.misc import build_optimizer, get_default_fn, path_replace
from .utils.modules import build_cnn_decoder_from_args, build_cnn_encoder_from_args
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
        if(args.charlie):
            assert (args.population is None) # NotImplementedFeature
            model = AliceBobCharlie(args, autologger, data_loader, signal_dump_dir)
        elif(args.population is not None): model = AliceBobPopulation(args, autologger, data_loader, signal_dump_dir)
        else:
            # --population is the population-game selector (also used by the eval scripts), so the
            # reset/size options only make sense alongside it.
            assert (args.pop_size is None) and (args.pop_reset_period is None), \
                "--pop_size / --pop_reset_period require the population game; pass --population too (e.g. --population 1)."
            model = AliceBob(args, autologger, data_loader, signal_dump_dir)
        model = model.to(args.device)

        if(args.detect_anomaly):
            torch.autograd.set_detect_anomaly(True)

        if(args.pretrain_CNNs): # Pretrains the agents.
            if(not isinstance(model, CNNPretrainable)):
                raise TypeError("--pretrain_CNNs was set, but %s does not support CNN pretraining." % type(model).__name__)

            print(("[%s] pretraining start…" % datetime.now()), flush=True)

            dcnn_factory_fn = get_default_fn(build_cnn_decoder_from_args, args)
            cnn_factory_fn = get_default_fn(build_cnn_encoder_from_args, args)
            pretrained_models = model.pretrain_CNNs(
                data_loader,
                pretrain_CNN_mode=args.pretrain_CNNs,
                freeze_pretrained_CNN=args.freeze_pretrained_CNNs,
                learning_rate=args.pretrain_learning_rate or args.learning_rate,
                epochs=args.pretrain_epochs,
                steps_per_epoch=args.steps_per_epoch,
                display_mode=args.display,
                pretrain_CNNs_on_eval=args.pretrain_CNNs_on_eval,
                deconvolution_factory=dcnn_factory_fn,
                convolution_factory=cnn_factory_fn
            )

            if(args.detect_outliers): # Might not work for all pretraining methods (in fact, we are expecting a MultiHeadsClassifier). To have a more general method, record the loss for all instances, then select the ones that are far from the mean
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

#this_path = os.path.abspath(os.path.dirname(sys.argv[0])) # The path of (the directory in which is) this file

import socket # for `gethostname`
import torch # for device
from datetime import datetime

def get_args(remaining_args=None):
    arg_parser = argparse.ArgumentParser()

    default_data_set = pathlib.Path('data') / 'concon'
    default_models = pathlib.Path('[summary]') / 'models'
    default_summary = pathlib.Path('runs') / 'cbc' / ('[now]_' + socket.gethostname())

    group = arg_parser.add_argument_group(title='Data', description='arguments relative to data handling')
    group.add_argument('--data_set', help='the path to the data set', default=default_data_set, type=pathlib.Path)
    group.add_argument('--binary_dataset', help='whether the data set contains binary or ternary images', action='store_true')
    group.add_argument('--constrain_dim', help='restrict specific dimensions in dataset', nargs=5, choices=[1,2,3], default=None, type=int)
    group.add_argument('--pair_images', '-pi', help='generates a new dataset by combining pairs of images', action='store_true')
    group.add_argument('--batch_size', help='batch size', default=128, type=int)
    group.add_argument('--noise', help='standard deviation of the normal random noise to apply to images', default=0.0, type=float)
    group.add_argument('--sampling_strategies', help='sampling strategies for the distractors, separated with \'/\' (available: hamming1, different, difficulty, random)', default='difficulty', choices=['hamming1', 'different', 'difficulty', 'random'])
    group.add_argument('--same_img', '-same_img', help='whether Bob sees Alice\'s image (or one of the same category)', action='store_true')
    group.add_argument('--evaluation_categories', help='determines whether and which categories are kept for evaluation only', default=5, type=int)

    group = arg_parser.add_argument_group(title='Save', description='arguments relative to saving models/logs')
    group.add_argument('--summary', help='the path to the TensorBoard summary for this run (\'[now]\' will be intepreted as now in the Y-m-d_H-M-S format)', default=default_summary, type=pathlib.Path)
    group.add_argument('--save_every', '-save_every', help='indicate to save the model after each __ epochs', type=int, default=0)
    group.add_argument('--models', help='the path to the saved models (\'[summary]\' will be interpreted as the value of --summary)', default=default_models, type=pathlib.Path)
    group.add_argument('--dump_signal', help='whether to regularly save the signals in a file', action="store_true")

    group = arg_parser.add_argument_group(title='Display', description='arguments relative to displayed information')
    # TODO: refactor logging: --display tqdm should be inferred from the env
    group.add_argument('--display', help='how to display the information', choices=['minimal', 'simple', 'tqdm'], default='tqdm')
    group.add_argument('--log_debug', '-ld', help='log more stuf', action='store_true')
    group.add_argument('--detect_anomaly', help='autodetect grad anomalies', action='store_true')
    group.add_argument('--no_summary', '-ns', help='do not write summaries', action='store_true')
    group.add_argument('--log_lang_progress', '-llp', help='log metrics to evaluate progress and stability of language learned', action='store_true')
    group.add_argument('--log_entropy', help='log evolution of entropy across epochs', action='store_true')
    group.add_argument('--no_log_imgs', help='do not log image samples', action='store_true')
    group.add_argument('--log_img_every', default=10, type=int, help='how often (in epochs) to log image samples')
    # TODO: refactor logging: --logging_period should control the frequency of step reports when --display minimal
    group.add_argument('--logging_period', help='how often counts of logged variables are accumulated', type=int, default=10)
    # TODO: refactor logging: --quiet vs. --display quiet?
    group.add_argument('--quiet', help='display less information', action='store_true')

    group = arg_parser.add_argument_group(title='Reward', description='arguments relative to reward shaping/gradient computation')
    group.add_argument('--len_penalty', help='coefficient for the length penalty of the signals', default=0.01, type=float)
    group.add_argument('--use_expectation', help='use expectation of success instead of playing dice', action='store_true')
    group.add_argument('--beta_sender', help='sender entropy penalty coefficient', type=float, default=0.01)
    group.add_argument('--beta_receiver', help='sender entropy penalty coefficient', type=float, default=0.001)
    group.add_argument("--learning_rate", help="learning rate", default=0.0001, type=float)
    group.add_argument('--grad_clipping', help='threshold for gradient clipping', default=1, type=float)
    group.add_argument('--grad_scaling', help='threshold for gradient scaling', default=None, type=float)

    group = arg_parser.add_argument_group(title='Language', description='arguments relative to language capacity')
    group.add_argument('--base_alphabet_size', help='size of the alphabet (not including special symbols)', default=10, type=int) # Previously 64. There are 32 intuitive classes of images in the data set
    group.add_argument('--max_len', help='maximum length of signals produced', default=10, type=int) # Previously 16.

    group = arg_parser.add_argument_group(title='Perfs', description='arguments relative to performances')
    # device_choices = ['cpu', 'cuda', 'mkldnn', 'opengl', 'opencl', 'ideep', 'hip', 'msnpu']
    # group.add_argument('--device', help='what to run PyTorch on (potentially available: ' + ', '.join(device_choices) + ')', choices=device_choices, default='cpu')
    group.add_argument('--device', help='what to run PyTorch on', type=torch.device, default=torch.device('cpu'))

    group = arg_parser.add_argument_group(title='Architecture', description='arguments relative to model & game architecture')
    group.add_argument('--shared', '-s', help='share the image encoder and the symbol embeddings among each couple of Alice·s and Bob·s', action='store_true')
    group.add_argument('--population', help='select the population game; N gives a symmetric default of N-N senders/receivers (override with --pop_size)', default=None, type=int)
    group.add_argument('--charlie', '-c', help='add adversary drawing agent', action='store_true')
    group.add_argument('--pop_size', help="population sizes as 'n-m' (n senders, m receivers); overrides the symmetric default from --population", default=None, type=str)
    group.add_argument('--pop_reset_period', help="reset periods as 'a-b'; reinitialize senders every a epochs and receivers every b epochs (0 = never); default 0-0", default=None, type=str)
    group.add_argument('--hidden_size', help='dimension of hidden representations', type=int, default=50)

    group = arg_parser.add_argument_group(title='Training', description='arguments relative to training curriculum')
    group.add_argument('--use_baseline', help='use a baseline term in REINFORCE', action='store_true')
    group.add_argument('--epochs', help='number of epochs', default=100, type=int)
    group.add_argument('--steps_per_epoch', help='number of steps per epoch', default=1000, type=int)
    group.add_argument('--runs', help='number of runs', default=1, type=int)
    group.add_argument('--seed', help='base random seed; each run uses seed+run_idx', default=None, type=int)
    group.add_argument('--no_spigot', help='whether to replace all GradSpigot·s with usual tensor', action='store_true')
    group.add_argument('--loss_weight_temp', help='temperature parameter in the loss weighting system', default=1.0, type=float)

    group = arg_parser.add_argument_group(title='Conv', description='arguments relative to convolutional structure')
    # group.add_argument('--img_channel', help='number of input channels in images', type=int, default=3)
    group.add_argument('--img_size', help='Width/height of images', type=int, default=128)
    group.add_argument('--decnn_channel_size', help="factor to determine number of channel features in deconvolutions (defaults to hidden size)", type=int, default=None)
    group.add_argument('--cnn_channel_size', help="factor to determine number of channel features in convolutions (defaults to hidden size)", type=int, default=None)
    group.add_argument('--local_batchnorm', help="indicates whether BatchNorme2D layers use global statistics (False) or not (True)", action="store_true")
    group.add_argument('--use_legacy_convolutions', help="use old architectures for both CNN and DeCNN", action="store_true")
    group.add_argument('--use_legacy_decnn', help="use old architecture for DeCNN", action="store_true")
    group.add_argument('--use_legacy_cnn', help="use old architecture for CNN", action="store_true")
    # group.add_argument('--conv_layers', help='number of convolution layers', type=int, default=8)
    # group.add_argument('--filters', help='number of filters per convolution layers', type=int, default=32)
    # group.add_argument('--kernel_size', help='size of convolution kernel', type=int, default=3)
    # group.add_argument('--strides', help='stride at each convolution layer', type=int, nargs='+', default=[2, 2, 1, 2, 1, 2, 1, 2]) # the original paper suggests 2,1,1,2,1,2,1,2, but that doesn't match the expected output of 50, 1, 1
    group.add_argument('--pretrain_CNNs', help='pretrain CNNs on specified task', type=str, choices=['category-wise', 'feature-wise', 'auto-encoder'])
    group.add_argument('--pretrain_learning_rate', help='learning rate for pretraining', type=float)
    group.add_argument('--pretrain_epochs', help='number of epochs per agent for CNN pretraining', type=int, default=5)
    group.add_argument('--pretrain_CNNs_on_eval', help='pretrain CNNs on classification', action='store_true')
    group.add_argument('--freeze_pretrained_CNNs', help='do not backpropagate gradient on pretrained CNNs', action='store_true')
    group.add_argument('--detect_outliers', help='if pretraining, then after, the trained model analyses the dataset in order to detect problems', action='store_true')
    group.add_argument('--autoencode_receiver_inputs', help='run all receiver image inputs through a pretrained autoencoder', action='store_true')

    group = arg_parser.add_argument_group(title='Eval', description='arguments relative to evaluation routines')
    group.add_argument('--correct_only', help='analyse the language constisting of the signals produced in successful rounds only', action='store_true')
    group.add_argument('--eval_oracle_language', help="debug feature: during fancy evaluation, replace the emergent language with a known-compositional 'oracle'/control language (an upper-bound sanity check, not part of normal runs). In AliceBob this is one unique symbol per (concept, value) of the category. Applied to topographic similarity, the entropy stats, and the decision tree; signal dumping and scrambling resistance keep using the emergent language.", action='store_true')
    
    group.add_argument('--debug', '-d', help='use this flag to change the behavior of the code to debug stuff', action='store_true')


    args = arg_parser.parse_args(remaining_args)
    if not args.quiet:
        print("command-line arguments:")
        pprint.pprint(vars(args), indent=4)
    
    return args


if(__name__ == "__main__"):
    do(global_args=None, remaining_args=None)
