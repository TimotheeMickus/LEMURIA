from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from .agent import Agent
from ..utils.modules import MessageEncoder, build_cnn_encoder_from_args
from ..utils import misc

# Structure for outcomes
Outcome = namedtuple("Outcome", ["scores", "msg_spigot"])

# Scores images according to a message.
class Receiver(Agent):
    """
    Defines a receiver policy.
    Based on K presented images and a given message, chooses which image the message refers to.
    """
    def __init__(self, image_encoder, message_encoder, args, has_shared_param):
        super(Agent, self).__init__()

        self.image_encoder = image_encoder
        self.message_encoder = message_encoder
        
        self.args = args
        self.has_shared_param = has_shared_param

    def encode_message(self, message, length):
        return self.message_encoder(message, length).unsqueeze(-1)

    # images: tensor of shape (batch size, nb img, *IMG_SHAPE)
    # message: 
    # use_spigot: boolean that indicates whether to use a GradSpigot (after the encoding of the message)
    def forward(self, images, message, length, use_spigot=False):
        encoded_message = self.encode_message(message, length) # Shape (batch size, hidden size)

        return self.aux_forward(images, encoded_message, use_spigot)

    # images: tensor of shape (batch size, nb img, *IMG_SHAPE)
    # encoded_messages: tensor of shape (batch size, hidden size)
    # use_spigot: boolean that indicates whether to use a GradSpigot (after the encoding of the message)
    def aux_forward(self, images, encoded_message, use_spigot):
        """
            Forward propagation.
            Input:
                `images`, of shape [args.batch_size x K x *IMG_SHAPE], where the first of each K image is the target
            Output:
                `Outcome` containing action taken, entropy, log prob, dist and scores.
        """

        # Encodes the images.
        encoded_images = self.image_encoder(images.view(-1, *images.shape[2:])) # Shape: ((batch size * nb img), hidden size)
        encoded_images = encoded_images.view(images.shape[0], images.shape[1], -1) # Shape: (batch size, nb img, hidden size)

        if(use_spigot):
           msg_spigot = misc.GradSpigot(encoded_message)
           encoded_message = msg_spigot.tensor
        else:
            msg_spigot = None

        # Scores the targets.
        scores = torch.bmm(encoded_images, encoded_message).squeeze(-1) # Shape: (batch size, nb img)

        outcome = Outcome(scores=scores, msg_spigot=msg_spigot)

        return outcome
    
    # Randomly reinitializes the parameters of the agent. (and also the requires_grad properties)
    def reinitialize(self):
        if(self.has_shared_param):
            raise ValueError("Modules with shared parameters cannot be reinitialized.")
        
        other_receiver = Receiver.from_args(self.args)
        other_parameters = dict(other_receiver.named_parameters())
        
        for name, parameters in dict(self.named_parameters()).items():
            parameters.data = other_parameters[name].data
            parameters.requires_grad = other_parameters[name].requires_grad

    # The two optional arguments are specified when creating a SenderReceiver.
    # image_encoder: torch.nn.Module
    # symbol_embeddings: torch.nn.Embedding
    @classmethod
    def from_args(cls, args, image_encoder=None, symbol_embeddings=None):
        has_shared_param = (image_encoder is not None) or (symbol_embeddings is not None)
        
        if(image_encoder is None): image_encoder = build_cnn_encoder_from_args(args)
        message_encoder = MessageEncoder.from_args(args, symbol_embeddings=symbol_embeddings)
        
        return cls(image_encoder, message_encoder, args, has_shared_param)
