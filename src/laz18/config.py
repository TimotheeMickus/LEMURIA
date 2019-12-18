import os
import sys
import argparse

this_path = os.path.abspath(os.path.dirname(sys.argv[0])) # The path of (the directory in which is) this file

import socket # for `gethostname`
from datetime import datetime

arg_parser = argparse.ArgumentParser()

default_data_set = os.path.join(this_path, os.pardir, os.pardir, 'data', 'coil', 'coil-100' ,'train')
arg_parser.add_argument('--data_set', help='the path to the data set', default=default_data_set)

default_summary = os.path.join(this_path, os.pardir, os.pardir, 'runs', 'laz18', ('[now]_' + socket.gethostname()))
arg_parser.add_argument('--summary', help='the path to the TensorBoard summary for this run (\'[now]\' will be intepreted as now in the Y-m-d_H-M-S format)', default=default_summary)

arg_parser.add_argument('-save_model', help='saves the model after each epoch', action='store_true')

arg_parser.add_argument('-simple_display', help='displays the information in a simple way (not using tqdm)', action='store_true')

#default_models = os.path.join(default_summary, 'models')
default_models = os.path.join('[summary]', 'models')
arg_parser.add_argument('--models', help='the path to the saved models (\'[summary]\' will be interpreted as the value of --summary)', default=default_models)

arg_parser.add_argument('--device', help='what to run PyTorch on (potentially available: cpu, cuda, mkldnn, opengl, opencl, ideep, hip, msnpu)', default='cpu')
#arg_parser.add_argument('-cpu', help='run PyTorch on CPU instead of GPU', action='store_true')

arg_parser.add_argument('--noise', help='standard deviation of the normal random noise to apply to images', default=0.0, type=float)

arg_parser.add_argument('--alphabet', help='size of the alphabet (not including the EOS symbol)', default=5, type=int)

arg_parser.add_argument('--distractors', help='number of distractors', default=2, type=int)

arg_parser.add_argument('--epochs', help='number of epochs', default=100, type=int)

arg_parser.add_argument('--runs', help='number of runs', default=1, type=int)

args = arg_parser.parse_args()


ALPHABET_SIZE = args.alphabet + 1 # + 1 for EOS,
EOS, PAD, BOS = 0, ALPHABET_SIZE, ALPHABET_SIZE + 1
MSG_LEN = 4
NUMBER_OF_DISTRACTORS = args.distractors
K = NUMBER_OF_DISTRACTORS + 1 # size of pools of image for listener

HIDDEN = 50

CONV_LAYERS = 8
FILTERS = 32
STRIDES = (2, 2, 1, 2, 1, 2, 1, 2) # the original paper suggests 2,1,1,2,1,2,1,2, but that doesn't match the expected output of 50, 1, 1
KERNEL_SIZE = 3

BATCH_SIZE = 32
LR = .0001
#lr tested:.0001, .00001, .001, .000001

# BETA values for reweighting entropy penalty
BETA_SENDER = .01
BETA_RECEIVER = .001

#IMG_SHAPE = (3, 124, 124) # Original dataset size
IMG_SHAPE = (3, 128, 128) # COIL size

NOISE_STD_DEV = args.noise

#DATASET_PATH = "/home/tmickus/data/img/coil/coil-100/train/"
DATASET_PATH = args.data_set

DEVICE = args.device

SUMMARY_DIR = args.summary.replace('[now]', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

SAVE_MODEL = args.save_model
MODELS_DIR = args.models.replace('[summary]', SUMMARY_DIR)

EPOCHS = args.epochs

RUNS = args.runs

SIMPLE_DISPLAY = args.simple_display
