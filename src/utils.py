import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from config import *

def max_tensor(t, dim, abs_val=True, unsqueeze=True):
    x = t.abs() if(abs_val) else t

    for i in range(dim, t.dim()):
        x = x.max(-1).values

    if(not unsqueeze): return x

    for i in range(dim, t.dim()):
        x = x.unsqueeze(-1)

    return x

# Transforms a tensor of black and white images to colour images
def to_color(t, dim):
    y = t.unsqueeze(dim)
    
    tmp = torch.ones(y.dim(), dtype=torch.int32)
    tmp[dim] = 3

    return y.repeat(tmp.tolist())

#
def max_normalize(t, dim, abs_val=True):
    x = max_tensor(t, dim, abs_val, unsqueeze=True)

    return (t / x)

def max_normalize_(t, dim, abs_val=True):
    x = max_tensor(t, dim, abs_val, unsqueeze=True)
    
    return t.div_(x)

# Displays a tensor as an image (channels as the first dimension)
def show_img(img):
    plt.imshow(np.transpose(img, (1,2,0)), interpolation='nearest')
    plt.show()

# Displays a tensor as an image (channels as the first dimension)
def simple_show_img(img):
    torchvision.transforms.functional.to_pil_image(img).show()

# `nrow` is the number of images to display in each row of the grid
def show_imgs(imgs, nrow=8):
    show_img(torchvision.utils.make_grid(imgs, nrow))

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
