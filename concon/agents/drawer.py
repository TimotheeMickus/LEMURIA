from collections import namedtuple

import torch
import torch.nn as nn

from .agent import Agent
from ..utils.modules import MessageEncoder, Randomizer, build_cnn_decoder_from_args

Outcome = namedtuple("Outcome", ["image"])

class Drawer(Agent):
    def __init__(self, message_encoder, randomizer, image_decoder, is_gumbel=False):
        super(Agent, self).__init__()
        self.message_encoder = message_encoder
        self.randomizer = randomizer
        self.image_decoder = image_decoder
        self.is_gumbel = is_gumbel

    def forward(self, message, length):
        encoded_message = self.message_encoder(message, length)

        message_with_noise = self.randomizer(encoded_message)
        # the deconvolution expects a 1 by 1 image with D channels, hence the unsqueezing
        deconv_input = message_with_noise[:,:,None,None].view(-1, message_with_noise.size(-1), 1, 1)
        image = self.image_decoder(deconv_input)
        if self.is_gumbel and self.training:
            image = image.view(*message_with_noise.shape[:2], *image.shape[1:])

        outcome = Outcome(image=image)
        return outcome

    @classmethod
    def from_args(cls, args, message_encoder=None):
        if message_encoder is None: message_encoder = MessageEncoder.from_args(args)
        randomizer = Randomizer.from_args(args)
        image_decoder = build_cnn_decoder_from_args(args)
        return cls(message_encoder, randomizer, image_decoder, is_gumbel=args.is_gumbel)
