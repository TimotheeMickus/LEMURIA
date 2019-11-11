import torch
import torch.nn as nn

from config import *

class ConvNetImageEncoder(nn.Module):
    def __init__(self):
        super(ConvNetImageEncoder, self).__init__()
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
        self.cnn = nn.Sequential(*layers)

    def forward(self, objects):
        conv_output = self.cnn(objects)
        # flatten. Last two dimensions should be 1
        return conv_output.view(*conv_output.size()[:2])
