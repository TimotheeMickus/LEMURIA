from collections import namedtuple

import torch
import torch.nn as nn

from modules import MessageEncoder, Randomizer, build_cnn_decoder_from_args

Outcome = namedtuple("Outcome", ["image"])

class Drawer(nn.Module):
    def __init__(self, message_encoder=None):
        super(Drawer, self).__init__()
        if message_encoder is None: message_encoder = MessageEncoder()
        self.message_encoder = message_encoder
        self.randomizer = Randomizer()
        self.image_decoder = build_cnn_decoder_from_args()

    def forward(self, message, length):
        encoded_message = self.message_encoder(message, length)

        message_with_noise = self.randomizer(encoded_message)
        # the deconvolution expects a 1 by 1 image with D channels, hence the unsqueezing
        deconv_input = message_with_noise[:,:,None,None]
        image = self.image_decoder(deconv_input)

        outcome = Outcome(image=image)
        return outcome
