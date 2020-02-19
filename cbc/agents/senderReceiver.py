import torch
import torch.nn as nn

from .sender import Sender
from .receiver import Receiver
from ..utils.modules import build_cnn_encoder_from_args, build_embeddings

class SenderReceiver(nn.Module):
    def __init__(self, sender, receiver, check_shared_params=True):
        nn.Module.__init__(self)

        if check_shared_params:
            assert sender.image_encoder is receiver.image_encoder, 'parameters are not shared'
            assert receiver.message_encoder.symbol_embeddings is sender.message_decoder.symbol_embeddings, 'parameters are not shared'

        self.sender = sender
        self.receiver = receiver

    @classmethod
    def from_args(cls, args):
        image_encoder = build_cnn_encoder_from_args(args)
        symbol_embeddings = build_embeddings(args.base_alphabet_size, args.hidden_size, use_bos=True) # +2: padding symbol, BOS symbol
        sender = Sender.from_args(args, image_encoder=image_encoder, symbol_embeddings=symbol_embeddings)
        receiver = Receiver.from_args(args, image_encoder=image_encoder, symbol_embeddings=symbol_embeddings)
        return cls(sender, receiver)
