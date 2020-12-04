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

from .aliceBob import AliceBob

class AliceBobCharlie(AliceBob):
    def __init__(self, args):
        super().__init__(args)
        self.drawer = Drawer.from_args(args)
        self.switch_charlie(False)

    def switch_charlie(self, train_charlie):
        self.train_charlie = train_charlie
        for a in super().agents:
            for p in a.parameters():
                p.requires_grad = not self.train_charlie
        for p in self.drawer.parameters():
            p.requires_grad = self.train_charlie

    @property
    def agents(self):
        return self.sender, self.receiver, self.drawer

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
        self.drawer = self.drawer.to(*vargs, **kwargs)
        return self

    def compute_interaction_charlie(self, batch):
        sender, receiver = self.get_sender(), self.get_receiver()
        # send
        sender_outcome = sender(self._alice_input(batch))

        # adversarial step
        drawer_outcome = self.drawer(*sender_outcome.action)

        # receive
        receiver_outcome = receiver(self._charlied_bob_input(batch, drawer_outcome.image), *sender_outcome.action)
        receiver_loss, receiver_entropy = self.compute_receiver_loss(receiver_outcome.scores, return_entropy=True)

        # training only Charlie for now
        loss = receiver_loss
        (sender_loss, sender_successes, sender_rewards) = self.compute_sender_loss(sender_outcome, receiver_outcome.scores)
        rewards, successes = sender_rewards, sender_successes
        avg_msg_length = sender_outcome.action[1].float().mean().item()

        return loss, rewards, successes, avg_msg_length, sender_outcome.entropy.mean(), receiver_entropy
