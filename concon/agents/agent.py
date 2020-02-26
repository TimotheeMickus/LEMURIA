import torch.nn as nn
from abc import ABCMeta, abstractmethod

class Agent(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *vargs, **kwargs):
        super(nn.Module, self).__init__(*vargs, **kwargs)

    @classmethod
    @abstractmethod
    def from_args(cls, args, *vargs, **kwargs):
        pass
