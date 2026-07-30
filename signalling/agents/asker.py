from collections import namedtuple

import torch
import torch.nn as nn

from .agent import Agent
from ..utils.modules import SignalDecoder

# Structure for outcomes
Outcome = namedtuple("Outcome", ["entropy", "log_prob", "action"])

# Produces a signal based on an predicate.
class Asker(Agent):
    def __init__(self, predicate_encoder, signal_decoder, args, has_shared_param):
        super().__init__()
        
        self.predicate_encoder = predicate_encoder # nn.Embedding(num_predicates, H)
        self.signal_decoder = signal_decoder # LSTM that produces the signal

        self.args = args # Used to reinitialize the agent.
        self.has_shared_param = has_shared_param

        self.alphabet_size = self.signal_decoder.alphabet_size
        self.eos_index = self.signal_decoder.eos_index
        self.padding_idx = self.signal_decoder.padding_idx
        self.bos_index = self.signal_decoder.bos_index # not actually used in the signals produced

    def forward(self, predicate_idx):
        """
            Forward propagation.
            Input:
                `predicate_idx`, 1D tensor of predicate indices.
            Output:
                `outcome`, Outcome where `.action` is the produced signal.
        """
        encoded_predicate = self.predicate_encoder(predicate_idx) # Shape: (batch size, embedding size)
        outputs = self.signal_decoder(encoded_predicate)

        outcome = Outcome(
            entropy=outputs["entropy"], # Shape: (batch size, 1)
            log_prob=outputs["log_probs"], # Shape: (batch, max signal length)
            action=(outputs["signal"], outputs["signal_len"]) # A list[list[Int]] and a tensor of shape (batch size, 1)
        )

        return outcome
    
    # Randomly reinitializes the parameters of the agent. (and also the requires_grad properties)
    def reinitialize(self):
        if(self.has_shared_param):
            raise ValueError("Modules with shared parameters cannot be reinitialized.")
        
        other_asker = Asker.from_args(self.args)
        other_parameters = dict(other_asker.named_parameters())
        
        for name, parameters in dict(self.named_parameters()).items():
            parameters.data.copy_(other_parameters[name].data.to(device=parameters.device, dtype=parameters.dtype))
            parameters.requires_grad = other_parameters[name].requires_grad

    # The two optional arguments are specified when creating an AskerRetriever.
    # predicate_encoder: torch.nn.Module
    # symbol_embeddings: torch.nn.Embedding
    @classmethod
    def from_args(cls, args, predicate_encoder=None, symbol_embeddings=None):
        has_shared_param = (predicate_encoder is not None) or (symbol_embeddings is not None)
        
        num_predicates = args.num_predicates
        if(predicate_encoder is None): predicate_encoder = nn.Embedding(num_predicates, args.hidden_size)

        signal_decoder = SignalDecoder.from_args(args, symbol_embeddings=symbol_embeddings)
        #signal_decoder = torch.compile(signal_decoder)
        
        return cls(predicate_encoder, signal_decoder, args, has_shared_param)
