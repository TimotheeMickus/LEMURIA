import torch
import torch.nn as nn

from sender import Sender
from receiver import Receiver
from modules import build_cnn_encoder_from_args

class SenderReceiver(nn.Module):
    def __init__(self, image_encoder, symbol_embeddings):
        nn.Module.__init__(self)

        self.sender = Sender(image_encoder, symbol_embeddings)
        self.receiver = Receiver(image_encoder, symbol_embeddings)

    @classmethod
    def from_args(cls, args):
        image_encoder = build_cnn_encoder_from_args(args)
        symbol_embeddings = build_embeddings(args.hidden_size, use_bos=True) # +2: padding symbol, BOS symbol
        return cls(image_encoder, symbol_embeddings)
