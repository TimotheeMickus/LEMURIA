import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from config import *

class AverageSummaryWriter:
    def __init__(self, writer=None, log_dir=None, default_period=1, specific_periods={}, prefix=None):
        if(writer is None): writer = SummaryWriter(log_dir)
        else: assert log_dir is None

        self.writer = writer
        self.default_period = default_period
        self.specific_periods = specific_periods
        self.prefix = prefix # If not None, will be add (with ':') before all tags

        self._values = {};

    def reset_values(self):
        self._values = {}

    def add_scalar(self, tag, scalar_value, global_step=None):
        values = self._values.setdefault(tag, [])
        values.append(scalar_value)

        period = self.specific_periods.get(tag, self.default_period)
        if(len(values) == period): # If the buffer is full, prints the average and clears the buffer
            _tag = tag if(self.prefix is None) else (self.prefix + ':' +  tag)
            self.writer.add_scalar(tag=_tag, scalar_value=np.mean(values), global_step=global_step)

            values.clear()

    # `l` is a list of pairs (key, value)
    def add_scalar_list(self, l, global_step=None):
        add = False
        for key, value in l:
            self.add_scalar(key, value, global_step)

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



def build_cnn(layer_classes=(), input_channels=(), output_channels=(),
    strides=(), kernel_size=None, paddings=None, flatten_last=True,
    sigmoid_after=False,):
    """
    Factory for convolutionnal encoders.
    Input:
        `layer_classes`: a list of classes to stack, taken from `{"conv", "convTranspose", "maxpool", "avgpool"}`
        `input_channels`: a list of expected input channels per layer
        `output_channels`: a list of expected output channels per layer
        `strides`: a list of strides per layer each layer
        `kernel_size`: a valid kernel size used throughout the convolutionnal network encoder, or a list of kernel sizes per layer
        `padding`: an optional list of (output) padding per layer
        `flatten_last`: flatten output instead of performing batch normalization after the last layer.
    Output:
        `cnn`: a convolutionnal network
    Raises:
        `AssertionError` if the provided lists `layer_classes`, `input_channels`, `output_channels`, and `strides` have different lengths
        `ValueError` if a given layer class is not "conv", "maxpool", or "avgpool"
    """

    lens = map(len, (layer_classes, input_channels, output_channels, strides))
    assert len(set(lens)) == 1, "provided parameters have different lengths!"

    if paddings is None:
        paddings = ([0] * len(layer_classes))
    else:
        assert len(layer_classes) == len(paddings), "provided parameters have different lengths!"

    if (type(kernel_size) is int) or (len(kernel_size) == 2):
        kernel_size = ([kernel_size] * len(layer_classes))
    else:
        assert len(layer_classes) == len(kernel_size), "provided parameters have different lengths!"

    if flatten_last:
        norms = ([nn.BatchNorm2d] * (len(layer_classes) - 1)) + [lambda _ : nn.Flatten()]
    else:
        norms = ([nn.BatchNorm2d] * len(layer_classes))

    layers = []

    for s,i,o,n,l,p,k in zip(
        strides,
        input_channels,
        output_channels,
        norms,
        layer_classes,
        paddings,
        kernel_size,):
        if l == "conv":
            core_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=i,
                    out_channels=o,
                    kernel_size=k,
                    stride=s,
                    padding=p,),
                nn.ReLU())
        elif l == "convTranspose":
            core_layer = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=i,
                    out_channels=o,
                    kernel_size=k,
                    stride=s,
                    output_padding=p,),
                nn.ReLU())
        elif l == "maxpool":
            core_layer = nn.MaxPool2d(
                kernel_size=k,
                stride=s,
                padding=p,)
        elif l == "avgpool":
            core_layer = nn.AvgPool2d(
                kernel_size=k,
                stride=s,
                padding=p,)
        else:
            raise ValueError("layer of type %s is not supported.")
        layers.append(
            nn.Sequential(
                core_layer,
                n(o),
        ))
    if sigmoid_after:
        layers.append(nn.Sigmoid())
    cnn = nn.Sequential(*layers)
    return cnn

def build_cnn_encoder():
    """
    Factory for convolutionnal networks
    """
    layer_classes = (["conv"] * len(STRIDES))
    input_channels = ([IMG_SHAPE[0]] + [FILTERS] * (len(STRIDES) - 1))
    output_channels = ([FILTERS] * (len(STRIDES) - 1) + [HIDDEN])
    return build_cnn(
        layer_classes=layer_classes,
        input_channels=input_channels,
        output_channels=output_channels,
        strides=STRIDES,
        kernel_size=KERNEL_SIZE,
        paddings=None,)

def build_cnn_decoder():
    """
    Factory for deconvolutionnal networks
    """
    layer_classes = (["convTranspose"] * len(STRIDES))
    strides = STRIDES[::-1]
    inputs = [HIDDEN] + ([FILTERS] * (len(STRIDES) - 1))
    outputs = ([FILTERS] * (len(STRIDES) - 1)) + [IMG_SHAPE[0]]
    padding = [0, 0, 1, 0, 0, 0, 0, 1] # guessworking it out
    return build_cnn(
        layer_classes=layer_classes,
        input_channels=inputs,
        output_channels=outputs,
        strides=strides,
        padding=padding,
        kernel_size=KERNEL_SIZE,
        flatten_last=False,
        sigmoid_after=True,)
