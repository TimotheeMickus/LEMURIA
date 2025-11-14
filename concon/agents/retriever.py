from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from .agent import Agent
from ..utils.modules import MessageEncoder, CandidateAverager
from ..utils import misc

# Structure for outcomes
Outcome = namedtuple("Outcome", ["scores", "msg_spigot"])

# Determines candidate truth-value according to a message.
class Retriever(Agent):
    """
    Defines a receiver policy.
    Based on a set of candidates and a given message, determines the candidate truth values.
    """
    def __init__(self, candidate_encoder, message_encoder, args, has_shared_param):
        # FIX MG: See fix in `agent.py`
        super().__init__()

        self.candidate_encoder = candidate_encoder # TODO embedding layer 
        self.message_encoder = message_encoder # (? RNN/Transformer encoder)
        
        self.args = args # Used to reinitialize the agent.
        self.has_shared_param = has_shared_param

    def encode_message(self, message, length):
        return self.message_encoder(message, length).unsqueeze(-1)

    # candidates: tensor of shape (TODO)
    # message: TODO
    # use_spigot: boolean that indicates whether to use a GradSpigot (after the encoding of the message)
    def forward(self, candidate_tensors, message, length, use_spigot=False):
        encoded_message = self.encode_message(message, length) # Shape (batch size, hidden size)
        return self.aux_forward(candidate_tensors, encoded_message, use_spigot)

    # images: tensor of shape (batch size, nb img, *IMG_SHAPE)
    # encoded_messages: tensor of shape (batch size, hidden size)
    # use_spigot: boolean that indicates whether to use a GradSpigot (after the encoding of the message)
    def aux_forward(self, candidate_tensors, encoded_message, use_spigot):
        """
            Forward propagation.
            Input:
                `candidate_tensors`, TODO shape, where target is?
            Output:
                TODO
        """

        # Encodes the images.
        encoded_candidates = self.candidate_encoder(**candidate_tensors) # Shape: (batch_size, num_candidates, hidden_size)

        if(use_spigot):
            msg_spigot = misc.GradSpigot(encoded_message)
            encoded_message = msg_spigot.tensor # Shape: (batch size, hidden size)
        else:
            msg_spigot = None

        # Scores the targets.
        scores = torch.bmm(encoded_candidates, encoded_message).squeeze(-1) # Shape: (batch size, num_candidates)
        outcome = Outcome(scores=scores, msg_spigot=msg_spigot)

        return outcome
    
    # Randomly reinitializes the parameters of the agent. (and also the requires_grad properties)
    def reinitialize(self):
        if(self.has_shared_param):
            raise ValueError("Modules with shared parameters cannot be reinitialized.")
        
        other_receiver = Retriever.from_args(self.args)
        other_parameters = dict(other_receiver.named_parameters())
        
        for name, parameters in dict(self.named_parameters()).items():
            parameters.data = other_parameters[name].data
            parameters.requires_grad = other_parameters[name].requires_grad

    # The two optional arguments are specified when creating a SenderReceiver.
    # image_encoder: torch.nn.Module
    # symbol_embeddings: torch.nn.Embedding
    @classmethod
    def from_args(cls, args, candidate_encoder=None, symbol_embeddings=None):
        has_shared_param = (candidate_encoder is not None) or (symbol_embeddings is not None)
        
        if candidate_encoder is None:
            # TODO PredicateGraphEncoder 
            candidate_encoder = CandidateAverager(node_vocab_size=args.node_vocab_size, hidden_size=args.hidden_size, padding_id=args.node_padding_id)
        message_encoder = MessageEncoder.from_args(args, symbol_embeddings=symbol_embeddings)
        
        return cls(candidate_encoder, message_encoder, args, has_shared_param)
