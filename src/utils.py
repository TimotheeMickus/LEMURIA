from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from config import *

# Displays a tensor as an image (channels as the first dimension)
def show_img(img):
    plt.imshow(np.transpose(img, (1,2,0)), interpolation='nearest')
    plt.show()

# Displays a tensor as an image (channels as the first dimension)
def simple_show_img(img):
    torchvision.transforms.functional.to_pil_image(img).show()

def show_imgs(imgs):
    show_img(torchvision.utils.make_grid(imgs))

# Structure for policy outcomes
PolicyOutcome = namedtuple("Policy", ["entropy", "log_prob", "action"])

def build_optimizer(θ):
    """
    Factory for optimizer
    Input:
        `θ`, the model parameters
    """
    return optim.RMSprop(θ, lr=LR)


def build_cnn_encoder():
    """
    Factory for convolutionnal encoders
    """
    layers = []
    strides = STRIDES
    inputs = [IMG_SHAPE[0]] + [FILTERS] * (len(STRIDES) - 1)
    outputs = [FILTERS] * (len(STRIDES) - 1) + [HIDDEN]
    norms = [nn.BatchNorm2d] * (len(STRIDES) - 1) + [lambda _ : nn.Flatten()]
    for s,i,o,n in zip(strides, inputs, outputs, norms):
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=i,
                    out_channels=o,
                    kernel_size=KERNEL_SIZE,
                    stride=s,
                    dilation=1),
                nn.ReLU(),
                n(o),
        ))
    cnn = nn.Sequential(*layers)
    return cnn
