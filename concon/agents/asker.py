from collections import namedtuple

import torch
import torch.nn as nn

from .agent import Agent
from ..utils.modules import MessageDecoder, build_cnn_encoder_from_args

# Structure for outcomes
Outcome = namedtuple("Outcome", ["entropy", "log_prob", "action"])

# Produces a message based on an predicate.
class Asker(Agent):
    def __init__(self, predicate_encoder, message_decoder, args, has_shared_param):
        super().__init__()
        
        self.predicate_encoder = predicate_encoder # nn.Embedding(num_predicates, H)
        self.message_decoder = message_decoder # TODO LSTM that produces the message (already in place I think?)
        
        self.args = args # Used to reinitialize the agent.
        self.has_shared_param = has_shared_param

    def forward(self, predicate_idx):
        """
            Forward propagation.
            Input:
                `predicate_idx`, 1D tensor of predicate indices.
            Output:
                `Outcome`, where `action` is the produced message.
        """
        encoded_predicate = self.predicate_encoder(predicate_idx) # Shape: (number of predicate indices)
        outputs = self.message_decoder(encoded_predicate)

        outcome = Outcome(
            entropy=outputs["entropy"], # Shape: (batch size, 1)
            log_prob=outputs["log_probs"], # Shape: (batch, max msg length)
            action=(outputs["message"], outputs["message_len"]) # A list[list[Int]] and a tensor of shape (batch size, 1)
        )

        return outcome
    
    # Randomly reinitializes the parameters of the agent. (and also the requires_grad properties)
    def reinitialize(self):
        if(self.has_shared_param):
            raise ValueError("Modules with shared parameters cannot be reinitialized.")
        
        other_sender = Asker.from_args(self.args)
        other_parameters = dict(other_sender.named_parameters())
        
        for name, parameters in dict(self.named_parameters()).items():
            parameters.data = other_parameters[name].data
            parameters.requires_grad = other_parameters[name].requires_grad

    # The two optional arguments are specified when creating an AskerRetriever.
    # predicate_encoder: torch.nn.Module
    # symbol_embeddings: torch.nn.Embedding
    @classmethod
    def from_args(cls, args, predicate_encoder=None, symbol_embeddings=None):
        has_shared_param = (predicate_encoder is not None) or (symbol_embeddings is not None)
        
        num_predicates = getattr(args, "num_predicates")
        if(predicate_encoder is None): predicate_encoder = nn.Embedding(num_predicates, args.hidden_size)
        # this is essentially the signal generator
        message_decoder = MessageDecoder.from_args(args, symbol_embeddings=symbol_embeddings)
        
        return cls(predicate_encoder, message_decoder, args, has_shared_param)
