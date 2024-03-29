from collections import namedtuple

import torch
import torch.nn as nn

from .agent import Agent
from ..utils.modules import MessageEncoder, NoiseAdder, build_cnn_decoder_from_args
from ..utils import misc

Outcome = namedtuple("Outcome", ["image", "img_spigot"])

# Produces an image based on a message.
class Drawer(Agent):
    def __init__(self, message_encoder, middle_nn, image_decoder, args, has_shared_param):
        super(Agent, self).__init__()

        self.message_encoder = message_encoder
        self.middle_nn = middle_nn
        self.image_decoder = image_decoder
        
        self.args = args # Used to reinitialize the agent.
        self.has_shared_param = has_shared_param

    # message: tensor of shape (batch size, max message length)
    # length: tensor of shape (batch size)
    # use_spigot: boolean that indicates whether to use a GradSpigot (after the image)
    def forward(self, message, length, use_spigot=False):
        encoded_message = self.message_encoder(message, length) # Shape: (batch size, hidden size)
        encoding = self.middle_nn(encoded_message) # Shape: (batch size, hidden size)
        encoding = encoding[:,:,None,None] # Because the deconvolution expects a 1 by 1 image with D channels. Shape: (batch size, hidden size, 1, 1)
        image = self.image_decoder(encoding) # Shape: (batch size, *IMG_SHAPE)

        if(use_spigot):
            img_spigot = misc.GradSpigot(image)
            image = img_spigot.tensor # Shape: (batch size, *IMG_SHAPE)
        else:
            img_spigot = None

        outcome = Outcome(image=image, img_spigot=img_spigot)

        return outcome
    
    # Randomly reinitializes the parameters of the agent. (and also the requires_grad properties)
    def reinitialize(self):
        if(self.has_shared_param):
            raise ValueError("Modules with shared parameters cannot be reinitialized.")
        
        other_drawer = Drawer.from_args(self.args)
        other_parameters = dict(other_drawer.named_parameters())
        
        for name, parameters in dict(self.named_parameters()).items():
            parameters.data = other_parameters[name].data
            parameters.requires_grad = other_parameters[name].requires_grad

    # Currently (2023-02-28), the optional argument is never specified.
    @classmethod
    def from_args(cls, args, message_encoder=None):
        has_shared_param = (message_encoder is not None)
        
        if(message_encoder is None): message_encoder = MessageEncoder.from_args(args)
        middle_nn = nn.Sequential(NoiseAdder.from_args(args), nn.Linear(args.hidden_size, args.hidden_size))
        image_decoder = build_cnn_decoder_from_args(args)
        
        return cls(message_encoder, middle_nn, image_decoder, args, has_shared_param)
