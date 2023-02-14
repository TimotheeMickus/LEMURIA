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
    def __init__(self, image_encoder, message_encoder, args, has_shared_param):
        super(Agent, self).__init__()

        self.image_encoder = image_encoder
        self.message_encoder = message_encoder
        
        self.args = args
        self.has_shared_param = has_shared_param

    def encode_message(self, message, length):
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
    
    # Randomly reinitializes the parameters of the agent.
    def reinitialize(self):
        if(self.has_shared_param):
            raise ValueError("Modules with shared parameters cannot be reinitialized.")
        
        other_receiver = Receiver.from_args(self.args)
        other_parameters = dict(other_receiver.named_parameters())
        
        for name, parameters in dict(self.named_parameters()).items():
            parameters.data = other_parameters[name].data

    # The two optional arguments are specified when creating a SenderReceiver.
    # image_encoder: torch.nn.Module
    # symbol_embeddings: torch.nn.Embedding
    @classmethod
    def from_args(cls, args, image_encoder=None, symbol_embeddings=None):
        has_shared_param = (image_encoder is not None) or (symbol_embeddings is not None)
        
        if(image_encoder is None): image_encoder = build_cnn_encoder_from_args(args)
        message_encoder = MessageEncoder.from_args(args, symbol_embeddings=symbol_embeddings)
        
        return cls(image_encoder, message_encoder, args, has_shared_param)
