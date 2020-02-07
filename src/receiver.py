from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from modules import MessageEncoder, build_cnn_encoder
from config import *

# Structure for outcomes
Outcome = namedtuple("Outcome", ["scores"])

# Scores images according to a message
class Receiver(nn.Module):
    """
    Defines a receiver policy.
    Based on K presented images and a given message, chooses which image the message refers to.
    """
    def __init__(self, image_encoder=None, symbol_embeddings=None):
        super(Receiver, self).__init__()

        if(image_encoder is None): image_encoder = build_cnn_encoder()
        self.image_encoder = image_encoder

        self.message_encoder = MessageEncoder(symbol_embeddings)

    def forward(self, images, message, length):
        """
            Forward propagation.
            Input:
                `images`, of shape [BATCH_SIZE x K x *IMG_SHAPE], where the first of each K image is the target
                `message`, of shape [BATCH_SIZE x (<=MSG_LEN)], message produced by sender
                `length`, of shape [BATCH_SIZE x 1], length of message produced by sender
            Output:
                `Outcome` containing action taken, entropy, log prob, dist and scores.
        """

        # Encodes the images
        original_size = images.size()[:2] #dim 1 & 2 give batch size & K (TODO Je ne comprends pas ce commentaire.)
        encoded_images = self.image_encoder(images.view(-1, *images.size()[2:]))
        encoded_images = encoded_images.view(*original_size, -1)

        # Encodes the message
        encoded_message = self.message_encoder(message, length).unsqueeze(-1)

        # Scores the targets
        scores = torch.bmm(encoded_images, encoded_message).squeeze(-1)

        outcome = Outcome(scores=scores)
        return outcome
