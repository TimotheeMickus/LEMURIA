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

arg_parser.add_argument('--simple_display', '-simple_display', help='displays the information in a simple way (not using tqdm)', action='store_true')

#default_models = os.path.join(default_summary, 'models')
default_models = os.path.join('[summary]', 'models')
arg_parser.add_argument('--models', help='the path to the saved models (\'[summary]\' will be interpreted as the value of --summary)', default=default_models)

arg_parser.add_argument('--device', help='what to run PyTorch on (potentially available: cpu, cuda, mkldnn, opengl, opencl, ideep, hip, msnpu)', default='cpu')
#arg_parser.add_argument('-cpu', help='run PyTorch on CPU instead of GPU', action='store_true')

arg_parser.add_argument('--batch_size', help='batch size', default=128, type=int)

arg_parser.add_argument('--noise', help='standard deviation of the normal random noise to apply to images', default=0.0, type=float)

arg_parser.add_argument('--same_img', '-same_img', help='whether Bob sees Alice\'s image (or one of the same category)', action='store_true')

arg_parser.add_argument('--penalty', help='coefficient for the length penalty of the messages', default=0.0, type=float)
arg_parser.add_argument('--adaptative_penalty', '-ap', help='use an adaptative penalty, that depends on the performance of the agents', action='store_true')

arg_parser.add_argument('--use_expectation', help='use expectation of success instead of playing dice', action='store_true')

arg_parser.add_argument('--shared', '-s', help='share the image encoder and the symbol embeddings among each couple of Alice·s and Bob·s', action='store_true')

arg_parser.add_argument('--population', help='population size', default=None, type=int)

arg_parser.add_argument('--charlie', '-c', help='add adversary drawing agent', action='store_true')

arg_parser.add_argument('--base_alphabet_size', help='size of the alphabet (not including special symbols)', default=10, type=int) # Previously 64. There are 32 intuitive classes of images in the data set
arg_parser.add_argument('--max_len', help='maximum length of messages produced', default=10, type=int) # Previously 16.

arg_parser.add_argument('--epochs', help='number of epochs', default=100, type=int)
arg_parser.add_argument('--steps_per_epoch', help='number of epochs', default=1000, type=int)

arg_parser.add_argument('--runs', help='number of runs', default=1, type=int)

arg_parser.add_argument("--learning_rate", help="learning rate", default=0.0001, type=float)

arg_parser.add_argument('--grad_clipping', help='threshold for gradient clipping', default=None, type=float)
arg_parser.add_argument('--grad_scaling', help='threshold for gradient scaling', default=None, type=float)

arg_parser.add_argument('--debug', '-d', help='log more stuff', action='store_true')
arg_parser.add_argument('--no_summary', '-ns', help='do not write summaries', action='store_true')
arg_parser.add_argument('--log_lang_progress', '-llp', help='log metrics to evaluate progress and stability of language learned', action='store_true')

# For visualize.py / evaluate_language.py
arg_parser.add_argument('--load_model', help='the path to the model to load')
# For evaluate_language.py
arg_parser.add_argument('--load_other_model', help='path to a second model to load')
arg_parser.add_argument('--message_dump_file', help='output file for messages produced by model')

arg_parser.add_argument('--log_entropy', help='log evolution of entropy across epochs', action='store_true')


# misc hyper parameters
arg_parser.add_argument('--hidden_size', help='dimension of hidden representations', type=int, default=50)
arg_parser.add_argument('--beta_sender', help='sender entropy penalty coefficient', type=float, default=0.01)
arg_parser.add_argument('--beta_receiver', help='sender entropy penalty coefficient', type=float, default=0.001)

# convolutions
arg_parser.add_argument('--img_channel', help='number of input channels in images', type=int, default=3)
arg_parser.add_argument('--conv_layers', help='number of convolution layers', type=int, default=8)
arg_parser.add_argument('--filters', help='number of filters per convolution layers', type=int, default=32)
arg_parser.add_argument('--kernel_size', help='size of convolution kernel', type=int, default=3)
arg_parser.add_argument('--strides', help='stride at each convolution layer', type=int, nargs='+', default=[2, 2, 1, 2, 1, 2, 1, 2]) # the original paper suggests 2,1,1,2,1,2,1,2, but that doesn't match the expected output of 50, 1, 1

args = arg_parser.parse_args()
print(args)

# ALPHABET_SIZE = args.base_alphabet_size + 1 # + 1 for EOS; these are the symbols actually generated by the speaker
#EOS, PAD, BOS = 0, ALPHABET_SIZE, ALPHABET_SIZE + 1
MSG_LEN = args.max_len # Max length of a message

CLIP_VALUE = args.grad_clipping
SCALE_VALUE = args.grad_scaling


#K = 3 # size of pools of image for listener

HIDDEN = args.hidden_size

CONV_LAYERS = args.conv_layers
FILTERS = args.filters
STRIDES = args.strides
KERNEL_SIZE = args.kernel_size

BATCH_SIZE = args.batch_size # Try with small values, such as 0.1
LR = args.learning_rate

# BETA values for reweighting entropy penalty
BETA_SENDER = args.beta_sender
BETA_RECEIVER = args.beta_receiver

# cf. `args.img_channel`
#IMG_SHAPE = (3, 128, 128)

NOISE_STD_DEV = args.noise

DATASET_PATH = args.data_set

DEVICE = args.device

SAVE_MODEL = args.save_model

EPOCHS = args.epochs

RUNS = args.runs

SIMPLE_DISPLAY = args.simple_display

DEBUG_MODE = args.debug
