from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from .agent import Agent
from ..utils.modules import MessageEncoder, build_cnn_encoder_from_args

# Structure for outcomes
Outcome = namedtuple("Outcome", ["scores"])

# Scores images according to a message
class Receiver(Agent):
    """
    Defines a receiver policy.
    Based on K presented images and a given message, chooses which image the message refers to.
    """
    def __init__(self, image_encoder, message_encoder):
        super(Agent, self).__init__()

        self.image_encoder = image_encoder
        self.message_encoder = message_encoder

    def encode_message(self, message, length):
        if self.message_encoder.is_gumbel:
            return self.message_encoder(message, length).transpose(-2, -1)
        else:
            return self.message_encoder(message, length).unsqueeze(-1)

    def forward(self, images, message, length):
        return self.aux_forward(images, self.encode_message(message, length))

    def aux_forward(self, images, encoded_message):
        """
            Forward propagation.
            Input:
                `images`, of shape [args.batch_size x K x *IMG_SHAPE], where the first of each K image is the target
                `message`, of shape [args.batch_size x (<=MSG_LEN)], message produced by sender
                `length`, of shape [args.batch_size x 1], length of message produced by sender
            Output:
                `Outcome` containing action taken, entropy, log prob, dist and scores.
        """

        # Encodes the images
        original_size = images.size()[:2] #dim 1 & 2 give batch size & K (TODO Je ne comprends pas ce commentaire.)
        encoded_images = self.image_encoder(images.view(-1, *images.size()[2:]))
        encoded_images = encoded_images.view(*original_size, -1)

        # Scores the targets
        scores = torch.bmm(encoded_images, encoded_message).squeeze(-1)

        outcome = Outcome(scores=scores)
        return outcome

    @classmethod
    def from_args(cls, args, image_encoder=None, symbol_embeddings=None):
        if(image_encoder is None): image_encoder = build_cnn_encoder_from_args(args)
        message_encoder = MessageEncoder.from_args(args, symbol_embeddings=symbol_embeddings)
        return cls(image_encoder, message_encoder)
