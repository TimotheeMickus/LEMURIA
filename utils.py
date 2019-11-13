import torch
import torch.nn as nn

from config import *

def build_cnn_encoder():
    layers = []
    strides = STRIDES
    ipts = [FILTERS] * len(STRIDES); ipts[0] = IMG_SHAPE[0]
    opts = [FILTERS] * len(STRIDES); opts[-1] = HIDDEN
    for s,i,o in zip(strides, ipts, opts):
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=i,
                    out_channels=o,
                    kernel_size=KERNEL_SIZE,
                    stride=s,
                    dilation=1),
                nn.ReLU(),
                nn.BatchNorm2d(o),
        ))
    cnn = nn.Sequential(*layers, nn.Flatten())
    return cnn
