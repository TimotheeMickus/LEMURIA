import itertools as it
import pathlib
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torchvision
import tqdm

from .game import Game
from ..agents import Sender, Receiver, Drawer
from ..utils.misc import show_imgs, max_normalize_, to_color, build_optimizer, pointing, compute_entropy_stats
from ..eval import compute_correlation

from .aliceBobPopulation import AliceBobPopulation

class AliceBobCharliePopulation(AliceBobPopulation):
    def __init__(self, args, logger):
        size = args.population
        self.drawers = nn.ModuleList([Drawer.from_args(args) for _ in range(size)])
        self._drawer = None
        super().__init__(args, logger)
        self._optim_alice_bob = self._optim
        self._optim_charlie = build_optimizer(self.drawers.parameters(), args.learning_rate)
        self.train_charlie = None
        self._agents += self.drawers
        self._n_batches_cycle = args.n_discriminator_batches + 1
        self._n_batches_seen = 0

    def switch_charlie(self, train_charlie):
        """
        Convenience method to turn charlie training on/off
        """
        self.train_charlie = train_charlie
        for a in super().agents:
            for p in a.parameters():
                p.requires_grad = not train_charlie
        for p in self.drawers.parameters():
            p.requires_grad = train_charlie
        self._optim = self._optim_charlie if train_charlie else self._optim_alice_bob
        if(self.autologger.summary_writer is not None):
            self.autologger.summary_writer.prefix = "Charlie" if train_charlie else None

    def get_drawer(self):
        return self._drawer

    def start_episode(self, train_episode=True):
        self._drawer = random.choice(self.drawers)
        super().start_episode(train_episode=train_episode)

        if train_episode:
            if((self._n_batches_seen % self._n_batches_cycle) == 0):
                # if we are at the end of a batch cycle, turn charlie off
                self.switch_charlie(False)

            elif(((self._n_batches_seen + 1) % self._n_batches_cycle) == 0):
                # if we are at the last batch of a batch cycle, turn charlie on
                self.switch_charlie(True)

            self._n_batches_seen += 1


    @property
    def agents(self):
        return self._sender, self._receiver, self._drawer

    @property
    def optims(self):
        return (self._optim_alice_bob, self._optim_charlie)

    # def compute_interaction(self, batch):
    #     if not self.train_charlie:
    #         return super().compute_interaction(batch)
    #     return self.compute_interaction_charlie(batch)

    def _charlied_bob_input(self, batch, charlie_img):
        images = [
            batch.target_img(stack=True).unsqueeze(1),
            batch.base_distractors_img(stack=True),
            charlie_img.unsqueeze(1),
        ]
        # put charlie's img first if we're training it
        if self.train_charlie: images = images[-1] + images[:-1]
        return torch.cat(images, dim=1)

    def to(self, *vargs, **kwargs):
        _ = super().to(*vargs, **kwargs)
        self.drawers = self.drawers.to(*vargs, **kwargs)
        return self

    def compute_interaction(self, batch):
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

    def evaluate(self, data_loader, epoch):

        super().evaluate(data_loader, epoch=epoch)

        img_directory = pathlib.Path('img') / str(epoch)
        img_directory.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            train_images = self.get_images(data_loader.get_batch(data_type='train'))
            for batch_idx in range(train_images.size(0)):
                imgs = train_images[batch_idx]
                torchvision.utils.save_image(imgs, img_directory / ("img_%i.png" % batch_idx))


    def get_images(self, batch):
        """
        Produce images for batch
        """
        with torch.no_grad():
            sender, drawer = self.get_sender(), self.get_drawer()
            # send
            action = sender(self._alice_input(batch)).action
            images = drawer(*action).image
        return torch.cat([batch.target_img(stack=True).unsqueeze(1), images.unsqueeze(1)], dim=1).cpu()
