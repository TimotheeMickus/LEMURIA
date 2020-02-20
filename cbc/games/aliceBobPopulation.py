import torch
import torch.nn as nn
import random

from .game import Game
from .aliceBob import AliceBob
from ..agents import Sender, Receiver, SenderReceiver
from ..utils.misc import build_optimizer

class AliceBobPopulation(AliceBob):
    def __init__(self, args):
        self.base_alphabet_size = args.base_alphabet_size
        self.max_len_msg = args.max_len

        size = args.population

        if(args.shared):
            self._agents = [SenderReceiver.from_args(args) for _ in range(size)]

            self.senders, self.receivers = zip(*[(agent.sender, agent.receiver) for agent in self._agents])
        else:
            self.senders = [Sender.from_args(args) for _ in range(size)]
            self.receivers = [Receiver.from_args(args) for _ in range(size)]

            self._agents = (self.senders + self.receivers)

        # PyTorch cannot find the parameters of objects that are in a list (like `self.senders` or `self.receivers`)

        self.use_expectation = args.use_expectation
        self.grad_scaling = args.grad_scaling or 0
        self.grad_clipping = args.grad_clipping or 0
        self.beta_sender = args.beta_sender
        self.beta_receiver = args.beta_receiver
        self.penalty = args.penalty
        self.adaptative_penalty = args.adaptative_penalty

        self._sender, self._receiver = None, None
        self.start_episode()

        parameters = [p for a in self._agents for p in a.parameters()]
        self._optim = build_optimizer(nn.ParameterList(parameters), args.learning_rate)

    def to(self, *vargs, **kwargs):
        #self = super().to(*vargs, **kwargs)

        #for agent in self._agents: agent.to(*args, **kwargs) # Would that be enough? I'm not sure how `.to` works

        self.senders = [sender.to(*vargs, **kwargs) for sender in self.senders]
        self.receivers = [receiver.to(*vargs, **kwargs) for receiver in self.receivers]

        return self

    def __call__(self, batch):
        """
        Input:
            `batch` is a Batch (a kind of named tuple); 'original_img' and 'target_img' are tensors of shape [args.batch_size, *IMG_SHAPE] and 'base_distractors' is a tensor of shape [args.batch_size, 2, *IMG_SHAPE]
        Output:
            `sender_outcome`, sender.Outcome
            `receiver_outcome`, receiver.Outcome
        """

        return self.compute_interaction(batch, sender, receiver)

    def start_episode(self):
        self._sender = random.choice(self.senders)
        self._receiver = random.choice(self.receivers)
        self._sender.train(), self._receiver.train()

    @property
    def agents(self):
        return self._sender, self._receiver

    @property
    def optims(self):
        return (self.optim,)
