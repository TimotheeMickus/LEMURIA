import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torchvision

import collections
import itertools

class Averager:
    def __init__(self, size, mem_factor=2, dtype=None):
        self.size = size

        buffer_size = int(mem_factor * size)
        assert (buffer_size > size)
        self._buffer = np.empty(buffer_size, dtype=dtype)

        self._i = 0

    def update_batch(self, xs):
        if(xs.size < self.size): # Usual case: `xs` of size lower than `size`
           if((self._i + xs.size) < self._buffer.size): # Most of the time case: adding `xs` would not fill the buffer entirely
                self._buffer[self._i:(self._i + xs.size)] = xs
                self._i += xs.size
           else: # Sometime case: adding `xs` would fill the buffer entirely
                j = (self.size - xs.size)
                self._buffer[:j] = self._buffer[(self._i - j):self._i]
                self._buffer[j:self.size] = xs
                self._i = self.size
        else: # Unusual case: `xs` of size higher than `size`
            self._buffer[:self.size] = xs[-self.size:]
            self._i = self.size

    def update(self, x):
        self._buffer[self._i] = x
        self._i += 1

        if(self._i == self._buffer.size): # The buffer is full
            self._buffer[:self.size] = self._buffer[-self.size:]
            self._i = self.size

    def get(self, default=None):
        l = min(self._i, self.size) # The number of values to consider might be smaller than `size` at the beginning
        if(l == 0): return default
        
        return self._buffer[(self._i-l):self._i].mean()

def h_compress(img):
    shape = img.shape
    width = shape[-1]
        
    return img.view(shape[:-1] + ((width // 2), 2)).mean(dim=-1)
    
def combine_images(img1, img2):
    return torch.cat((h_compress(img1), h_compress(img2)), dim=-1)

# Groups elements by keys
# Returns a dictionary from keys to lists of values
# Accepts either a list `l1` of pairs (element, key) or a list `l1` of elements and a list `l2` of keys
def group_by(l1, l2=None):
    if(l2 is not None): l1 = zip(l1, l2)
    d = collections.defaultdict(list)
    for e, k in l1:
        d[k].append(e)

    return d

def count(l):
    d = collections.defaultdict(int)
    for e in l: d[e] += 1
    return d

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

def get_default_fn(base_fn, args):
    def _wrap():
        return base_fn(args)
    return _wrap

def pointing(scores, argmax=False):
    probs = F.softmax(scores, dim=-1)
    dist = Categorical(probs)

    if(argmax): action = scores.max(-1).indices
    else: action = dist.sample()

    return {'dist': dist, 'action': action}

def add_normal_noise(t, std_dev, clamp_values=None):
    tmp = (t + (std_dev * torch.randn(size=t.shape)))

    if(clamp_values is not None):
        clamp_min, clamp_max = clamp_values
        tmp = torch.clamp(tmp, clamp_min, clamp_max)

    return tmp

# `counts` must be a Torch tensor
# If `base` is None, the base is e
# The output is a Python float
def compute_entropy(counts, base=None):
    entropy = Categorical(counts / counts.sum()).entropy().item()
    if(base is not None): entropy /= np.log(base)

    return entropy

def compute_entropy_stats(sample_messages, sample_categories, base=None):
    # We starts by counting the occurences of each messages for each category
    entropy_dict = collections.defaultdict(collections.Counter)
    for msg, cat in zip(sample_messages, sample_categories): entropy_dict[cat][msg] += 1.0

    # We then computes the entropy of each category's distribution 
    entropy_cats = [compute_entropy(torch.tensor(list(messages_counter.values())), base=base) for messages_counter in entropy_dict.values()]
    entropy_cats = np.array(entropy_cats)
    
    return entropy_cats.min(), entropy_cats.mean(), np.median(entropy_cats), entropy_cats.max(), entropy_cats.var()

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

class Unflatten(nn.Module):
    def forward(self, x):
        return x.unsqueeze(-1).unsqueeze(-1)

def build_optimizer(θ, learning_rate):
    """
    Factory for optimizer
    Input:
        `θ`, the model parameters
    """
    return optim.RMSprop(θ, lr=learning_rate)
