import torch.nn as nn
from abc import ABCMeta, abstractmethod

class Game(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *vargs, **kwargs):
        super(nn.Module, self).__init__(*vargs, **kwargs)

    @abstractmethod
    def test_visualize(self, data_iterator, learning_rate):
        pass

    @abstractmethod
    def get_agents(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def compute_interaction(self, batch_input, *agents):
        pass
