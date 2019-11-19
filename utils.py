from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim


from config import *

# struct for policy outcomes
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
