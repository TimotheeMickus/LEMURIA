import torch
import torch.nn as nn

from sender import Sender
from receiver import Receiver
from modules import build_cnn_encoder_from_args

from config import *

class SenderReceiver(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        image_encoder = build_cnn_encoder_from_args()
        symbol_embeddings = nn.Embedding((ALPHABET_SIZE + 2), args.hidden_size, padding_idx=PAD) # +2: padding symbol, BOS symbol

        self.sender = Sender(image_encoder, symbol_embeddings)
        self.receiver = Receiver(image_encoder, symbol_embeddings)
