import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torchvision
import tqdm

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
        if(len(values) == period): # If the buffer is full, logs the average and clears the buffer
            _tag = tag if(self.prefix is None) else (self.prefix + ':' +  tag)
            self.writer.add_scalar(tag=_tag, scalar_value=np.mean(values), global_step=global_step)

            values.clear()

    # `l` is a list of pairs (key, value)
    def add_scalar_list(self, l, global_step=None):
        add = False
        for key, value in l:
            self.add_scalar(key, value, global_step)

class Progress:
    def __init__(self, simple_display, steps_per_epoch, epoch, logged_items={"R"}):
        self.simple_display = simple_display
        self.steps_per_epoch = steps_per_epoch
        self.epoch = epoch
        self._logged_items = logged_items

    def __enter__(self):
        if(self.simple_display): self.i = 0
        else: self.pbar = tqdm.tqdm(total=self.steps_per_epoch, postfix={i: 0.0 for i in self._logged_items}, unit="B", desc=("Epoch %i" % self.epoch)) # Do not forget to close it at the end

        return self

    def update(self, num_batches_per_episode, **logged_items):
        if(self.simple_display):
            postfix = " ".join(("%s: %f" % (k, logged_items[k])) for k in sorted(logged_items))
            print(('%i/%i - %s' % (self.i, self.steps_per_epoch, postfix)), flush=True)
            self.i += num_batches_per_episode
        else:
            self.pbar.set_postfix(logged_items, refresh=False)
            self.pbar.update(num_batches_per_episode)

    def __exit__(self, type, value, traceback):
        if(not self.simple_display): self.pbar.close()

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

def log_grads_tensorboard(parameter_list, event_writer):
    """
    Log gradient evolution
    Input:
        `parameter_list`, the model parameters
        `event_writer`, the tensorboard summary
    """
    raw_grad = torch.cat([p.grad.view(-1).detach() for p in parameters])
    median_grad = raw_grad.abs().median().item()
    mean_grad = raw_grad.abs().mean().item()
    max_grad = raw_grad.abs().max().item()

    norm_grad = torch.stack([p.grad.view(-1).detach().data.norm(2.) for p in parameters])
    mean_norm_grad = norm_grad.mean().item()
    max_norm_grad = norm_grad.max().item()
    event_writer.add_scalar('grad/median_grad', median_grad, number_ex_seen)
    event_writer.add_scalar('grad/mean_grad', mean_grad, number_ex_seen)
    event_writer.add_scalar('grad/max_grad', max_grad, number_ex_seen)
    event_writer.add_scalar('grad/mean_norm_grad', mean_norm_grad, number_ex_seen)
    event_writer.add_scalar('grad/max_norm_grad', max_norm_grad, number_ex_seen)