import os
import sys
import argparse

this_path = os.path.abspath(os.path.dirname(sys.argv[0])) # The path of (the directory in which is) this file

import socket # for `gethostname`
from datetime import datetime

arg_parser = argparse.ArgumentParser()

default_data_set = os.path.join(this_path, os.pardir, os.pardir, 'data', 'cbc')
arg_parser.add_argument('--data_set', help='the path to the data set', default=default_data_set)

default_summary = os.path.join(this_path, os.pardir, os.pardir, 'runs', 'cbc', ('[now]_' + socket.gethostname()))
arg_parser.add_argument('--summary', help='the path to the TensorBoard summary for this run (\'[now]\' will be intepreted as now in the Y-m-d_H-M-S format)', default=default_summary)

arg_parser.add_argument('--save_model', '-save_model', help='saves the model after each epoch', action='store_true')

arg_parser.add_argument('-simple_display', help='displays the information in a simple way (not using tqdm)', action='store_true')

#default_models = os.path.join(default_summary, 'models')
default_models = os.path.join('[summary]', 'models')
arg_parser.add_argument('--models', help='the path to the saved models (\'[summary]\' will be interpreted as the value of --summary)', default=default_models)

arg_parser.add_argument('--device', help='what to run PyTorch on (potentially available: cpu, cuda, mkldnn, opengl, opencl, ideep, hip, msnpu)', default='cpu')
#arg_parser.add_argument('-cpu', help='run PyTorch on CPU instead of GPU', action='store_true')

arg_parser.add_argument('--batch', help='batch size', default=32, type=int)

arg_parser.add_argument('--noise', help='standard deviation of the normal random noise to apply to images', default=0.0, type=float)

arg_parser.add_argument('--penalty', help='coefficient for the length penalty of the messages', default=0.0, type=float)

arg_parser.add_argument('--alphabet', help='size of the alphabet (not including the EOS symbol)', default=10, type=int) # Previously 64. There are 32 intuitive classes of images in the data set

arg_parser.add_argument('--max_len', help='maximum length of messages produced', default=10, type=int) # Previously 16.

arg_parser.add_argument('--epochs', help='number of epochs', default=100, type=int)

arg_parser.add_argument('--runs', help='number of runs', default=1, type=int)

arg_parser.add_argument("--learning_rate", help="learning rate", default=0.0001, type=float)
#clip
arg_parser.add_argument('--clip', help='gradient clip value', default=None, type=float)

arg_parser.add_argument('--norm_clip', help='clip gradient by norm', action="store_true")


args = arg_parser.parse_args()


ALPHABET_SIZE = args.alphabet + 1 # + 1 for EOS,
EOS, PAD, BOS = 0, ALPHABET_SIZE, ALPHABET_SIZE + 1
MSG_LEN = args.max_len # Max length of a message

CLIP_VALUE = args.clip
CLIP_NORM = args.norm_clip


K = 3 # size of pools of image for listener

HIDDEN = 50

CONV_LAYERS = 8
FILTERS = 32
STRIDES = (2, 2, 1, 2, 1, 2, 1, 2) # the original paper suggests 2,1,1,2,1,2,1,2, but that doesn't match the expected output of 50, 1, 1
KERNEL_SIZE = 3

BATCH_SIZE = args.batch # Try with small values, such as 0.1
LR = args.learning_rate

# BETA values for reweighting entropy penalty
BETA_SENDER = .01
BETA_RECEIVER = .001

IMG_SHAPE = (3, 128, 128)

NOISE_STD_DEV = args.noise

DATASET_PATH = args.data_set

DEVICE = args.device

SUMMARY_DIR = args.summary.replace('[now]', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

SAVE_MODEL = args.save_model
MODELS_DIR = args.models.replace('[summary]', SUMMARY_DIR)

EPOCHS = args.epochs

RUNS = args.runs

SIMPLE_DISPLAY = args.simple_display

DEBUG_MODE = False
