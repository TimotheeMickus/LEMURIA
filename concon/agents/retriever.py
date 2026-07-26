from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from .agent import Agent
from ..utils.modules import SignalEncoder, CandidateNodeAverager, CandidateGraphEncoder
from ..utils import misc

# Structure for outcomes
Outcome = namedtuple("Outcome", ["scores", "signal_spigot"])

# Determines candidate truth-value according to a signal.
class Retriever(Agent):
    """
    Defines a retriever policy.
    Based on a set of candidates and a given signal, determines the candidate truth values.
    """
    def __init__(self, candidate_encoder, signal_encoder, args, has_shared_param):
        super().__init__()

        self.candidate_encoder = candidate_encoder # currently embedding-based averaging encoder 
        self.signal_encoder = signal_encoder # RNN (LSTM) encoder
        self.blind_candidates = getattr(args, "blind_candidates", False)
        self.blind_signal = getattr(args, "blind_signal", False)
        
        self.args = args # Used to reinitialize the agent.
        self.has_shared_param = has_shared_param
        
        self.alphabet_size = self.signal_encoder.alphabet_size

    def encode_signal(self, signal, length):
        return self.signal_encoder(signal, length).unsqueeze(-1)

    # candidates: dict with 
    #   - node_idx: (batch, n_candidates, max_nodes)
    #   - edge_idx: (batch, n_candidates, max_nodes, max_nodes)
    #   - graph_size: (batch, n_candidates)
    # signal: (batch, max_signal_len) token IDs
    # length: (batch, 1) signal lengths
    # use_spigot: boolean that indicates whether to use a GradSpigot (after the encoding of the signal)
    def forward(self, candidate_tensors, signal, length, use_spigot=False):
        encoded_signal = self.encode_signal(signal, length) # Shape (batch size, hidden size)
        return self.aux_forward(candidate_tensors, encoded_signal, use_spigot)

    # candidate_tensors: dict with 
    #   - node_idx: (batch, n_candidates, max_nodes)
    #   - edge_idx: (batch, n_candidates, max_nodes, max_nodes)
    #   - graph_size: (batch, n_candidates)
    # encoded_signals: tensor of shape (batch, hidden size)
    # use_spigot: boolean that indicates whether to use a GradSpigot (after the encoding of the signal)
    def aux_forward(self, candidate_tensors, encoded_signal, use_spigot):
        """
        Forward propagation.
        Outputs Outcome with:
            - scores: (batch, n_candidates) logits for each candidate
            - signal_spigot: GradSpigot or None
        """
        # Encodes the candidates.
        # TODO explicitate
        encoded_candidates = self.candidate_encoder(**candidate_tensors) # Shape: (batch_size, num_candidates, hidden_size)
        if self.blind_candidates:
            encoded_candidates = torch.zeros_like(encoded_candidates)

        if(use_spigot):
            signal_spigot = misc.GradSpigot(encoded_signal)
            encoded_signal = signal_spigot.tensor # Shape: (batch size, hidden size)
        else:
            signal_spigot = None
        if self.blind_signal:
            encoded_signal = torch.zeros_like(encoded_signal)

        # Scores (logits) the targets.
        scores = torch.bmm(encoded_candidates, encoded_signal).squeeze(-1) # Shape: (batch size, num_candidates)
        outcome = Outcome(scores=scores, signal_spigot=signal_spigot)

        return outcome
    
    # Randomly reinitializes the parameters of the agent. (and also the requires_grad properties)
    def reinitialize(self):
        if(self.has_shared_param):
            raise ValueError("Modules with shared parameters cannot be reinitialized.")
        
        other_receiver = Retriever.from_args(self.args)
        other_parameters = dict(other_receiver.named_parameters())
        
        for name, parameters in dict(self.named_parameters()).items():
            parameters.data.copy_(other_parameters[name].data.to(device=parameters.device, dtype=parameters.dtype))
            parameters.requires_grad = other_parameters[name].requires_grad

    # The optional argument is specified when creating an AskerRetriever.
    # candidate_encoder: torch.nn.Module
    # symbol_embeddings: torch.nn.Embedding
    @classmethod
    def from_args(cls, args, symbol_embeddings=None):
        has_shared_param = (symbol_embeddings is not None)
        
        if(args.candidate_encoder == "node_averager"):
            candidate_encoder = CandidateNodeAverager(
                node_vocab_size=args.node_vocab_size,
                hidden_size=args.hidden_size,
                padding_idx=args.node_padding_idx
            )
        elif(args.candidate_encoder == "graph_transformer"):
            candidate_encoder = CandidateGraphEncoder(
                node_vocab_size=args.node_vocab_size,
                edge_vocab_size=args.edge_vocab_size,
                num_layers=args.graph_num_layers,
                d_model=args.graph_d_model,
                num_heads=args.graph_num_heads,
                d_hidden=args.graph_d_hidden,
                dropout=args.graph_dropout,
                use_norm=not args.graph_no_norm
            )
        #candidate_encoder = torch.compile(candidate_encoder)
                
        signal_encoder = SignalEncoder.from_args(args, symbol_embeddings=symbol_embeddings)
        #signal_encoder = torch.compile(signal_encoder)
        
        return cls(candidate_encoder, signal_encoder, args, has_shared_param)
