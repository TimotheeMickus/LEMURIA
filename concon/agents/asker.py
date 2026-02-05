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
        self.message_decoder = message_decoder # LSTM that produces the signal

        self.args = args # Used to reinitialize the agent.
        self.has_shared_param = has_shared_param

    def forward(self, predicate_idx):
        """
            Forward propagation.
            Input:
                `predicate_idx`, 1D tensor of predicate indices.
            Output:
                `Outcome`, where `action` is the produced signal. (assigned as message, naming mismatch)
        """
        if not torch.isfinite(self.predicate_encoder.weight).all():
            bad_rows = (~torch.isfinite(self.predicate_encoder.weight)).any(dim=1)
            bad_idxs = bad_rows.nonzero(as_tuple=False).view(-1)[:10].tolist()
            raise ValueError(f"predicate_encoder has non-finite rows (showing up to 10): {bad_idxs}")
        if predicate_idx.numel() > 0:
            min_idx = int(predicate_idx.min().item())
            max_idx = int(predicate_idx.max().item())
            if min_idx < 0 or max_idx >= self.predicate_encoder.num_embeddings:
                raise ValueError(
                    f"predicate_idx out of range: min={min_idx} max={max_idx} "
                    f"(num_embeddings={self.predicate_encoder.num_embeddings})"
                )
        encoded_predicate = self.predicate_encoder(predicate_idx) # Shape: (number of predicate indices)
        if not torch.isfinite(encoded_predicate).all():
            bad_rows = (~torch.isfinite(encoded_predicate)).any(dim=1)
            bad_pred_idxs = predicate_idx[bad_rows].tolist()[:10]
            raise ValueError(f"encoded_predicate non-finite for predicate_idx (showing up to 10): {bad_pred_idxs}")
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
