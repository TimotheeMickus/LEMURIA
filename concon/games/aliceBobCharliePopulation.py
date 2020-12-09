import itertools as it
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import tqdm

from .game import Game
from ..agents import Sender, Receiver, Drawer
from ..utils.misc import show_imgs, max_normalize_, to_color, build_optimizer, pointing, compute_entropy_stats
from ..eval import compute_correlation

from .aliceBobPopulation import AliceBobPopulation

class AliceBobCharliePopulation(AliceBobPopulation):
    def __init__(self, args):
        size = args.population
        self.drawers = nn.ModuleList([Drawer.from_args(args) for _ in range(size)])
        self._drawer = None
        super().__init__(args)
        self._optim_alice_bob = self._optim
        self._optim_charlie = build_optimizer(self.drawers.parameters(), args.learning_rate)
        self.switch_charlie(False)
        self._agents += self.drawers

    def switch_charlie(self, train_charlie):
        self.train_charlie = train_charlie
        # for a in super().agents:
        #     for p in a.parameters():
        #         p.requires_grad = not train_charlie
        # for p in self.drawers.parameters():
        #     p.requires_grad = train_charlie
        self._optim = self._optim_charlie if train_charlie else self._optim_alice_bob

    def get_drawer(self):
        return self._drawer

    def start_episode(self, train_episode=True):
        self._drawer = random.choice(self.drawers)
        super().start_episode(train_episode=train_episode)

    @property
    def agents(self):
        return self._sender, self._receiver, self._drawer

    @property
    def optims(self):
        return (self._optim_alice_bob, self._optim_charlie)

    def compute_interaction(self, batch):
        if not self.train_charlie:
            return super().compute_interaction(batch)
        return self.compute_interaction_charlie(batch)

    def _charlied_bob_input(self, batch, charlie_img):
        return torch.cat([
            charlie_img.unsqueeze(1),
            batch.target_img(stack=True).unsqueeze(1),
            batch.base_distractors_img(stack=True)
        ], dim=1)

    def to(self, *vargs, **kwargs):
        _ = super().to(*vargs, **kwargs)
        self.drawers = self.drawers.to(*vargs, **kwargs)
        return self

    def compute_interaction_charlie(self, batch):
        sender, receiver, drawer = self.agents
        # send
        sender_outcome = sender(self._alice_input(batch))

        # adversarial step
        drawer_outcome = drawer(*sender_outcome.action)

        # receive
        receiver_outcome = receiver(self._charlied_bob_input(batch, drawer_outcome.image), *sender_outcome.action)
        receiver_loss, receiver_entropy = self.compute_receiver_loss(receiver_outcome.scores, return_entropy=True)

        # training only Charlie for now
        loss = receiver_loss
        (sender_loss, sender_successes, sender_rewards) = self.compute_sender_loss(sender_outcome, receiver_outcome.scores)
        rewards, successes = sender_rewards, sender_successes
        avg_msg_length = sender_outcome.action[1].float().mean().item()

        return loss, rewards, successes, avg_msg_length, sender_outcome.entropy.mean(), receiver_entropy
