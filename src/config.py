import os
import sys
import argparse

this_path = os.path.abspath(os.path.dirname(sys.argv[0])) # The path of (the directory in which is) this file

import socket # for `gethostname`
from datetime import datetime

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-data_set', help='the path to the data set', default=(os.path.join(this_path, os.pardir, 'data', 'coil', 'coil-100' ,'train')))
arg_parser.add_argument('-summaries', help='the path to the TensorBoard summaries for this run', default=(os.path.join(this_path, os.pardir, 'runs', (datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + socket.gethostname()))))
arg_parser.add_argument('-device', help='what to run PyTorch on', default='cpu')
#arg_parser.add_argument('-cpu', help='run PyTorch on CPU instead of GPU', action='store_true')
args = arg_parser.parse_args()

ALPHABET_SIZE = 5 + 1 # + 1 for EOS,
EOS, PAD, BOS = 0, ALPHABET_SIZE, ALPHABET_SIZE + 1
MSG_LEN = 4
NUMBER_OF_DISTRACTORS = 2
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

#DATASET_PATH = "/home/tmickus/data/img/coil/coil-100/train/"
DATASET_PATH = args.data_set

#DEVICE = "cpu"
#DEVICE = 'cpu' if(args.cpu) else 'gpu'
DEVICE = args.device

#MODEL_CKPT_DIR = "models/"
MODEL_CKPT_DIR = os.path.join(this_path, os.pardir, 'models')

SUMMARIES_DIR = args.summaries

EPOCHS = 1000
