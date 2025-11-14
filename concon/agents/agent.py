import torch.nn as nn
from abc import ABCMeta, abstractmethod

class Agent(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *vargs, **kwargs):
        # MG: FIX?
        # super(nn.Module, self).__init__(*vargs, **kwargs)
        # I believe this should in fact be:
        super().__init__()
        # The substituted class should be `Agent`, not `nn.Module`
        # https://stackoverflow.com/questions/61288224/why-not-super-init-model-self-in-pytorch

    @classmethod
    @abstractmethod
    def from_args(cls, args, *vargs, **kwargs):
        pass
