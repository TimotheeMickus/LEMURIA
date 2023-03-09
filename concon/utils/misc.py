import collections
import itertools
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torchvision

# loss: scalar tensor
# agent: 
# spigot: GradSpigot
def get_backward_f(loss, agent=None, spigot=None):
    def backward_f(retain_graph):
        if(agent is None):
            assert spigot is None
            parameters = None
        else:
            parameters = list(agent.parameters())

        if(spigot is None):
            loss.backward(retain_graph=retain_graph, inputs=parameters)
        else:
            loss.backward(retain_graph=True, inputs=(parameters + [spigot.tensor]))
            spigot.backward(retain_graph=retain_graph, inputs=parameters)

    return backward_f

# Produces a detached version of the input tensor and provides methods used to backpropagate the gradient.
class GradSpigot:
    # t: tensor
    def __init__(self, input):
        assert input.requires_grad

        self._input = input

        self.tensor = self._input.detach()
        self.tensor.requires_grad = self._input.requires_grad

    # RMK: Practical when there are multiple GradSpigot·s in the graph.
    # mask: None or a tensor that can be multiplied by self.tensor.grad
    def backward_scalar(self, mask=None):
        grad = self.tensor.grad
        if(mask is not None): grad *= mask
        
        return (self._input * grad).sum() # Shape: ()

    # RMK: Practical when there is only one GradSpigot in the graph.
    # mask: None or a tensor that can be multiplied by self.tensor.grad
    def backward(self, mask=None, retain_graph=False, inputs=None):
        grad = self.tensor.grad
        if(mask is not None): grad *= mask
        
        self._input.backward(gradient=grad, retain_graph=retain_graph, inputs=inputs)

class Averager:
    def __init__(self, size, mem_factor=2, dtype=None, buffer_f=None):
        self.size = size # Number of values used to compute the mean.

        buffer_size = int(mem_factor * size)
        assert (buffer_size > size)
        if(buffer_f is None): self._buffer = np.empty(buffer_size, dtype=dtype)
        else: self._buffer = buffer_f(buffer_size, dtype)

        self._i = 0 # Position in the buffer where the next value should be added.

    # Adds a batch of values.
    # xs: a sequence of values
    @torch.no_grad()
    def update_batch(self, xs):
        if(len(xs) < self.size): # Usual case: `xs` of size strictly lower than `size`.
           if((self._i + len(xs)) < len(self._buffer)): # Most of the time case: adding `xs` would not fill the buffer entirely.
                self._buffer[self._i:(self._i + len(xs))] = xs
                self._i += len(xs)
           else: # Sometime case: adding `xs` would fill the buffer entirely.
                j = (self.size - len(xs)) # The number of values already in the buffer that should be kept.
                self._buffer[:j] = self._buffer[(self._i - j):self._i]
                self._buffer[j:self.size] = xs
                self._i = self.size
        else: # Unusual case: `xs` of size higher than `size`.
            self._buffer[:self.size] = xs[-self.size:]
            self._i = self.size

    # Adds a single value.
    # x: numeral
    @torch.no_grad()
    def update(self, x):
        self._buffer[self._i] = x
        self._i += 1

        if(self._i == len(self._buffer)): # If the buffer is full.
            self._buffer[:self.size] = self._buffer[-self.size:]
            self._i = self.size

    @torch.no_grad()
    def get(self, default=None):
        l = min(self._i, self.size) # The number of values to consider. This might be smaller than `size` at first.
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

# scores: tensor of shape (batch size, #options)
def pointing(scores, argmax=False):
    probs = F.softmax(scores, dim=-1) # Shape: (batch size, #options)
    dist = Categorical(probs)

    if(argmax): action = scores.max(-1).indices # Shape: (batch size)
    else: action = dist.sample() # Shape: (batch size)

    return {'dist': dist, 'action': action}

def add_normal_noise(t, std_dev, clamp_values=None):
    tmp = (t + (std_dev * torch.randn_like(t)))

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

# Doesn't affect the input tensor.
def max_normalize(t, dim, abs_val=True):
    x = max_tensor(t, dim, abs_val, unsqueeze=True)

    return (t / x)

# In-place.
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

def path_replace(path, substring, replacement):
    return pathlib.Path(str(path).replace(str(substring), str(replacement)))
