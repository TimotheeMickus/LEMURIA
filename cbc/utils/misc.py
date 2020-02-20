import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torchvision


# A simple class the purpose of which is to assign ids to object
class Vocabulary:
    def __init__(self, entries=[]):
        self.entry_ids = {}
        self.entries = []

        for x in entries:
            self.add(x)

    def __len__(self):
        return len(self.entries)

    def __contains__(self, x):
        return (x in self.entry_ids)

    # Returns a pair composed of a boolean indicating whether a new entry was just created and the index of the entry
    def add(self, x, check_uniqueness=False):
        idx = self.entry_ids.get(x)

        if(idx is None):
            idx = len(self.entries)

            self.entry_ids[x] = idx
            self.entries.append(x)

            return (True, idx)

        assert (not check_uniqueness), "not unique"

        return (False, idx)

    def getId(self, x, default=None):
        return self.entry_ids.get(x, default)

    def __getitem__(self, idx):
        return self.entries[idx]

def pointing(scores):
    probs = F.softmax(scores, dim=-1)
    dist = Categorical(probs)
    action = dist.sample()

    return {'dist': dist, 'action': action}

def add_normal_noise(t, std_dev, clamp_values=None):
    tmp = (t + (std_dev * torch.randn(size=t.shape)))

    if(clamp_values is not None):
        clamp_min, clamp_max = clamp_values
        tmp = torch.clamp(tmp, clamp_min, clamp_max)

    return tmp

def compute_entropy(counts):
    return Categorical(counts / counts.sum()).entropy().item()

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

def build_optimizer(θ, learning_rate):
    """
    Factory for optimizer
    Input:
        `θ`, the model parameters
    """
    return optim.RMSprop(θ, lr=learning_rate)
