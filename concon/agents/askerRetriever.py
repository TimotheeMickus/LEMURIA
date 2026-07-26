from .asker import Asker
from .retriever import Retriever
from .agent import Agent
from ..utils.modules import build_embeddings

class AskerRetriever(Agent):
    def __init__(self, asker, retriever, check_shared_params=True):
        super().__init__()

        if(check_shared_params):
            # The asker (via its signal decoder) and the retriever (via its signal encoder) share the symbol embeddings.
            assert retriever.signal_encoder.symbol_embeddings is asker.signal_decoder.symbol_embeddings, 'parameters are not shared'

        self.asker = asker
        self.retriever = retriever

    @classmethod
    def from_args(cls, args):
        # Only the symbol embeddings are shared.
        symbol_embeddings = build_embeddings(args.base_alphabet_size, args.hidden_size, use_bos=True) # The vocabulary size is base_alphabet_size + 3 (EOS, padding, BOS). BOS is required by the asker's signal decoder; the retriever's signal encoder does not use it.

        asker = Asker.from_args(args, symbol_embeddings=symbol_embeddings)
        retriever = Retriever.from_args(args, symbol_embeddings=symbol_embeddings)

        return cls(asker, retriever)
