import torch
import torch.nn as nn

from .asker import Asker
from .retriever import Retriever
from .agent import Agent
from ..utils.modules import build_cnn_encoder_from_args, build_embeddings

class AskerRetriever(Agent):
    def __init__(self, asker, retriever, check_shared_params=True):
        super(Agent, self).__init__()

        if(check_shared_params):
            assert asker.predicate_encoder is retriever.predicate_encoder, 'parameters are not shared'
            assert retriever.message_encoder.symbol_embeddings is asker.message_decoder.symbol_embeddings, 'parameters are not shared'

        self.asker = asker
        self.retriever = retriever

    @classmethod
    def from_args(cls, args):
        num_predicates = getattr(args, "num_predicates") # TODO Why this weird instruction?
        predicate_encoder = nn.Embedding(num_predicates, args.hidden_size)
        symbol_embeddings = build_embeddings(args.base_alphabet_size, args.hidden_size, use_bos=True) # +2: padding symbol, BOS symbol
        
        asker = Asker.from_args(args, predicate_encoder=predicate_encoder, symbol_embeddings=symbol_embeddings)
        retriever = Retriever.from_args(args, predicate_encoder=predicate_encoder, symbol_embeddings=symbol_embeddings)
        
        return cls(asker, retriever)
