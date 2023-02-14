import torch
import torch.nn as nn
from collections import namedtuple

from .agent import Agent
from ..utils.modules import MessageDecoder, build_cnn_encoder_from_args

# Structure for outcomes
Outcome = namedtuple("Outcome", ["entropy", "log_prob", "action"])

# Image -(vector)-> message
class Sender(Agent):
    def __init__(self, image_encoder, message_decoder, args, has_shared_param):
        super(Agent, self).__init__()
        
        self.image_encoder = image_encoder
        self.message_decoder = message_decoder
        
        self.args = args
        self.has_shared_param = has_shared_param

    def forward(self, image):
        """
            Forward propagation.
            Input:
                `image`, of shape [args.batch_size x *IMG_SHAPE]
            Output:
                `Outcome`, where `action` is the produced message
        """
        encoded_image = self.image_encoder(image)
        outputs = self.message_decoder(encoded_image)
        outcome = Outcome(
            entropy=outputs["entropy"],
            log_prob=outputs["log_probs"],
            action=(outputs["message"], outputs["message_len"]))
        return outcome
    
    # Randomly reinitializes the parameters of the agent.
    def reinitialize(self):
        if(self.has_shared_param):
            raise ValueError("Modules with shared parameters cannot be reinitialized.")
        
        other_sender = Sender.from_args(self.args)
        other_parameters = dict(other_sender.named_parameters())
        
        for name, parameters in dict(self.named_parameters()).items():
            parameters.data = other_parameters[name].data

    # The two optional arguments are specified when creating a SenderReceiver.
    # image_encoder: torch.nn.Module
    # symbol_embeddings: torch.nn.Embedding
    @classmethod
    def from_args(cls, args, image_encoder=None, symbol_embeddings=None):
        has_shared_param = (image_encoder is not None) or (symbol_embeddings is not None)
        
        if(image_encoder is None): image_encoder = build_cnn_encoder_from_args(args)
        message_decoder = MessageDecoder.from_args(args, symbol_embeddings=symbol_embeddings)
        
        return cls(image_encoder, message_decoder, args, has_shared_param)
