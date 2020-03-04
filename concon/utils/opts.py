import os
import sys
import argparse

#this_path = os.path.abspath(os.path.dirname(sys.argv[0])) # The path of (the directory in which is) this file

import socket # for `gethostname`
from datetime import datetime

def get_args():

    arg_parser = argparse.ArgumentParser()

    default_data_set = os.path.join('data', 'concon')
    default_models = os.path.join('[summary]', 'models')
    default_summary = os.path.join('runs', 'cbc', ('[now]_' + socket.gethostname()))

    group = arg_parser.add_argument_group(title='data', description='arguments relative to data handling')
    group.add_argument('--data_set', help='the path to the data set', default=default_data_set)
    group.add_argument('--binary_dataset', help='whether the data set contains binary or ternary images', action='store_true')
    group.add_argument('--constrain_dim', help='restrict specific dimensions in dataset', nargs=5, choices=[1,2,3], default=None, type=int)
    group.add_argument('--batch_size', help='batch size', default=128, type=int)
    group.add_argument('--noise', help='standard deviation of the normal random noise to apply to images', default=0.0, type=float)
    group.add_argument('--sampling_strategies', help='sampling strategies for the distractors, separated with \'/\' (available: hamming1, different, difficulty)', default='difficulty')
    group.add_argument('--same_img', '-same_img', help='whether Bob sees Alice\'s image (or one of the same category)', action='store_true')
    group.add_argument('--evaluation_categories', help='determines whether and which categories are kept for evaluation only', default=5, type=int)

    group = arg_parser.add_argument_group(title='Save', description='arguments relative to saving models/logs')
    group.add_argument('--summary', help='the path to the TensorBoard summary for this run (\'[now]\' will be intepreted as now in the Y-m-d_H-M-S format)', default=default_summary)
    group.add_argument('--save_every', '-save_every', help='indicate to save the model after each __ epochs', type=int, default=0)
    group.add_argument('--models', help='the path to the saved models (\'[summary]\' will be interpreted as the value of --summary)', default=default_models)

    group = arg_parser.add_argument_group(title='Display', description='arguments relative to displayed information')
    group.add_argument('--display', help='how to display the information', choices=['minimal', 'simple', 'tqdm'], default='tqdm')
    group.add_argument('--debug', '-d', help='log more stuff', action='store_true')
    group.add_argument('--no_summary', '-ns', help='do not write summaries', action='store_true')
    group.add_argument('--log_lang_progress', '-llp', help='log metrics to evaluate progress and stability of language learned', action='store_true')
    group.add_argument('--log_entropy', help='log evolution of entropy across epochs', action='store_true')
    group.add_argument('--logging_period', help='how often counts of logged variables are accumulated', type=int, default=10)
    group.add_argument('--quiet', help='display less information', action='store_true')

    group = arg_parser.add_argument_group(title='Reward', description='arguments relative to reward shaping/gradient computation')
    group.add_argument('--penalty', help='coefficient for the length penalty of the messages', default=0.01, type=float)
    group.add_argument('--adaptative_penalty', '-ap', help='use an adaptative penalty, that depends on the performance of the agents', action='store_true') # TODO DELETE
    group.add_argument('--use_expectation', help='use expectation of success instead of playing dice', action='store_true')
    group.add_argument('--beta_sender', help='sender entropy penalty coefficient', type=float, default=0.01)
    group.add_argument('--beta_receiver', help='sender entropy penalty coefficient', type=float, default=0.001)
    group.add_argument("--learning_rate", help="learning rate", default=0.0001, type=float)
    group.add_argument('--grad_clipping', help='threshold for gradient clipping', default=1, type=float)
    group.add_argument('--grad_scaling', help='threshold for gradient scaling', default=None, type=float)

    group = arg_parser.add_argument_group(title='Language', description='arguments relative to language capacity')
    group.add_argument('--base_alphabet_size', help='size of the alphabet (not including special symbols)', default=10, type=int) # Previously 64. There are 32 intuitive classes of images in the data set
    group.add_argument('--max_len', help='maximum length of messages produced', default=10, type=int) # Previously 16.

    group = arg_parser.add_argument_group(title='Perfs', description='arguments relative to performances')
    group.add_argument('--device', help='what to run PyTorch on (potentially available: cpu, cuda, mkldnn, opengl, opencl, ideep, hip, msnpu)', default='cpu')

    group = arg_parser.add_argument_group(title='Architecture', description='arguments relative to model & game architecture')
    group.add_argument('--shared', '-s', help='share the image encoder and the symbol embeddings among each couple of Alice·s and Bob·s', action='store_true')
    group.add_argument('--population', help='population size', default=None, type=int)
    group.add_argument('--charlie', '-c', help='add adversary drawing agent', action='store_true')
    group.add_argument('--adv_pointer_training', help='non-standard pointer-based adversary_training', action='store_true')
    group.add_argument('--ignore_charlie', help='do not train bob on charlie output', action='store_true')
    group.add_argument('--hidden_size', help='dimension of hidden representations', type=int, default=50)

    group = arg_parser.add_argument_group(title='Training', description='arguments relative to training curriculum')
    group.add_argument('--epochs', help='number of epochs', default=100, type=int)
    group.add_argument('--steps_per_epoch', help='number of steps per epoch', default=1000, type=int)
    group.add_argument('--runs', help='number of runs', default=1, type=int)

    group = arg_parser.add_argument_group(title='Conv', description='arguments relative to convolutional structure')
    group.add_argument('--img_channel', help='number of input channels in images', type=int, default=3)
    group.add_argument('--conv_layers', help='number of convolution layers', type=int, default=8)
    group.add_argument('--filters', help='number of filters per convolution layers', type=int, default=32)
    group.add_argument('--kernel_size', help='size of convolution kernel', type=int, default=3)
    group.add_argument('--strides', help='stride at each convolution layer', type=int, nargs='+', default=[2, 2, 1, 2, 1, 2, 1, 2]) # the original paper suggests 2,1,1,2,1,2,1,2, but that doesn't match the expected output of 50, 1, 1
    group.add_argument('--pretrain_CNNs', help='pretrain CNNs on specified task', type=str, choices=['category-wise', 'feature-wise', 'auto-encoder'])
    group.add_argument('--pretrain_epochs', help='number of epochs per agent for CNN pretraining', type=int, default=5)
    group.add_argument('--pretrain_CNNs_on_eval', help='pretrain CNNs on classification', action='store_true')

    group = arg_parser.add_argument_group(title='Eval', description='arguments relative to evaluation routines')
    group.add_argument('--evaluate_language', help='evaluate language instead of training', action='store_true')
    group.add_argument('--visualize', help='visualize language instead of training', action='store_true')
    group.add_argument('--compute_correlation', help='compute correlation between meaning distance and message distance instead of training', action='store_true')
    # For visualize.py / evaluate_language.py
    group.add_argument('--load_model', help='the path to the model to load')
    # For evaluate_language.py
    group.add_argument('--load_other_model', help='path to a second model to load')
    group.add_argument('--message_dump_file', help='file containing language sample, or file where to dump language sample.')

    args = arg_parser.parse_args()
    if not args.quiet:
        print(args)
    return args
