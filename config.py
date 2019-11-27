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

DATASET_PATH = "/home/tmickus/data/img/coil/coil-100/train/"

DEVICE = "cpu"
MODEL_CKPT_DIR = "models/"
EPOCHS = 1000
