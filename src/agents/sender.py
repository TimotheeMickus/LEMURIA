import torch
import torch.nn as nn
from collections import namedtuple

from ..utils.modules import MessageDecoder, build_cnn_encoder_from_args

# Structure for outcomes
Outcome = namedtuple("Outcome", ["entropy", "log_prob", "action"])

# Image -(vector)-> message
class Sender(nn.Module):
    def __init__(self, image_encoder, message_decoder):
        super(Sender, self).__init__()
        self.image_encoder = image_encoder
        self.message_decoder = message_decoder

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


    @classmethod
    def from_args(cls, args, image_encoder=None, symbol_embeddings=None):
        if(image_encoder is None): image_encoder = build_cnn_encoder_from_args(args)
        message_decoder = MessageDecoder.from_args(args, symbol_embeddings=symbol_embeddings)
        return cls(image_encoder, message_decoder)
