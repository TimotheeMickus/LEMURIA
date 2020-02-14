import torch
import torch.nn as nn
from collections import namedtuple

from modules import MessageDecoder, build_cnn_encoder_from_args

from config import *

# Structure for outcomes
Outcome = namedtuple("Outcome", ["entropy", "log_prob", "action"])

# Image -(vector)-> message
class Sender(nn.Module):
    def __init__(self, image_encoder=None, symbol_embeddings=None):
        super(Sender, self).__init__()

        if(image_encoder is None): image_encoder = build_cnn_encoder_from_args()
        self.image_encoder = image_encoder
        self.message_decoder = MessageDecoder(symbol_embeddings)

    def forward(self, image):
        """
            Forward propagation.
            Input:
                `image`, of shape [args.batch_size x *IMG_SHAPE]
            Output:
                `Outcome`, where `action` is the produced message
        """
        encoded_image = self.image_encoder(image)
        outputs = self.message_decoder(encoded_image)
        outcome = Outcome(
            entropy=outputs["entropy"],
            log_prob=outputs["log_probs"],
            action=(outputs["message"], outputs["message_len"]))
        return outcome
