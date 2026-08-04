from collections import namedtuple

import torch
import torch.nn as nn

from .agent import Agent
from ..utils.modules import SignalDecoder, build_cnn_encoder_from_args

# Structure for outcomes
# `symbol_marginal` is the (differentiable) batch symbol marginal m_a = E_s[π(a|s)], of shape
# (base_alphabet_size + 1,); it is only consumed by the group-sparsity vocabulary auxiliary loss
# and defaults to None so nothing else depends on it.
Outcome = namedtuple("Outcome", ["entropy", "log_prob", "action", "symbol_marginal"], defaults=[None])

# Produces a signal based on an image.
class Sender(Agent):
    def __init__(self, image_encoder, signal_decoder, args, has_shared_param):
        super().__init__()
        
        self.image_encoder = image_encoder
        self.signal_decoder = signal_decoder
        
        self.args = args # Used to reinitialize the agent.
        self.has_shared_param = has_shared_param

        self.alphabet_size = self.signal_decoder.alphabet_size
        self.eos_index = self.signal_decoder.eos_index
        self.padding_idx = self.signal_decoder.padding_idx
        self.bos_index = self.signal_decoder.bos_index # not actually used in the signals produced

    def forward(self, image):
        """
            Forward propagation.
            Input:
                `image`, of shape [args.batch_size x *IMG_SHAPE]
            Output:
                `Outcome`, where `action` is the produced signal
        """
        encoded_image = self.image_encoder(image) # Shape: (batch size, encoding size)
        outputs = self.signal_decoder(encoded_image)

        outcome = Outcome(
            entropy=outputs["entropy"], # Shape: (batch size, 1)
            log_prob=outputs["log_probs"], # Shape: (batch, max signal length)
            action=(outputs["signal"], outputs["signal_len"]), # A list[list[Int]] and a tensor of shape (batch size, 1)
            symbol_marginal=outputs["symbol_marginal"] # Shape: (base_alphabet_size + 1,)
        )

        return outcome
    
    # Randomly reinitializes the parameters of the agent. (and also the requires_grad properties)
    def reinitialize(self):
        if(self.has_shared_param):
            raise ValueError("Modules with shared parameters cannot be reinitialized.")
        
        other_sender = Sender.from_args(self.args)
        other_parameters = dict(other_sender.named_parameters())
        
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
        signal_decoder = SignalDecoder.from_args(args, symbol_embeddings=symbol_embeddings)
        
        return cls(image_encoder, signal_decoder, args, has_shared_param)
