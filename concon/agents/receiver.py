from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from .agent import Agent
from ..utils.modules import SignalEncoder, build_cnn_encoder_from_args
from ..utils import misc

# Structure for outcomes
Outcome = namedtuple("Outcome", ["scores", "signal_spigot"])

# Scores images according to a signal.
class Receiver(Agent):
    """
    Defines a receiver policy.
    Based on K presented images and a given signal, chooses which image the signal refers to.
    """
    def __init__(self, image_encoder, signal_encoder, args, has_shared_param):
        super().__init__()

        self.image_encoder = image_encoder
        self.signal_encoder = signal_encoder
        
        self.args = args # Used to reinitialize the agent.
        self.has_shared_param = has_shared_param

    def encode_signal(self, signal, length):
        return self.signal_encoder(signal, length).unsqueeze(-1)

    # images: tensor of shape (batch size, nb img, *IMG_SHAPE)
    # signal: 
    # use_spigot: boolean that indicates whether to use a GradSpigot (after the encoding of the signal)
    def forward(self, images, signal, length, use_spigot=False):
        encoded_signal = self.encode_signal(signal, length) # Shape (batch size, hidden size)

        return self.aux_forward(images, encoded_signal, use_spigot)

    # images: tensor of shape (batch size, nb img, *IMG_SHAPE)
    # encoded_signals: tensor of shape (batch size, hidden size)
    # use_spigot: boolean that indicates whether to use a GradSpigot (after the encoding of the signal)
    def aux_forward(self, images, encoded_signal, use_spigot):
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
           signal_spigot = misc.GradSpigot(encoded_signal)
           encoded_signal = signal_spigot.tensor # Shape: (batch size, hidden size)
        else:
            signal_spigot = None

        # Scores the targets.
        scores = torch.bmm(encoded_images, encoded_signal).squeeze(-1) # Shape: (batch size, nb img)

        outcome = Outcome(scores=scores, signal_spigot=signal_spigot)

        return outcome
    
    # Randomly reinitializes the parameters of the agent. (and also the requires_grad properties)
    def reinitialize(self):
        if(self.has_shared_param):
            raise ValueError("Modules with shared parameters cannot be reinitialized.")
        
        other_receiver = Receiver.from_args(self.args)
        other_parameters = dict(other_receiver.named_parameters())
        
        for name, parameters in dict(self.named_parameters()).items():
            parameters.data.copy_(other_parameters[name].data.to(device=parameters.device, dtype=parameters.dtype))
            parameters.requires_grad = other_parameters[name].requires_grad

    # The two optional arguments are specified when creating a SenderReceiver.
    # image_encoder: torch.nn.Module
    # symbol_embeddings: torch.nn.Embedding
    @classmethod
    def from_args(cls, args, image_encoder=None, symbol_embeddings=None):
        has_shared_param = (image_encoder is not None) or (symbol_embeddings is not None)
        
        if(image_encoder is None): image_encoder = build_cnn_encoder_from_args(args)
        signal_encoder = SignalEncoder.from_args(args, symbol_embeddings=symbol_embeddings)
        
        return cls(image_encoder, signal_encoder, args, has_shared_param)
