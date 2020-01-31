import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from collections import namedtuple

from config import *
from utils import build_cnn_encoder

# Structure for outcomes
Outcome = namedtuple("Policy", ["entropy", "log_prob", "action", "dist", "scores"])

# Message -> vector
class ReceiverMessageEncoder(nn.Module):
    """
    Encodes a message of discrete symbols in a single vector.
    """
    def __init__(self, symbol_embeddings=None):
        super(ReceiverMessageEncoder, self).__init__()

        if(symbol_embeddings is None): symbol_embeddings = nn.Embedding((ALPHABET_SIZE + 1), HIDDEN, padding_idx=PAD) # +1: padding symbol
        self.symbol_embeddings = symbol_embeddings
        
        self.lstm = nn.LSTM(HIDDEN, HIDDEN, 1, batch_first=True)

    def forward(self, message, length):
        """
        Forward propagation.
        Input:
            `message`, of shape [BATCH_SIZE x <=MSG_LEN], message produced by sender
            `length`, of shape [BATCH_SIZE x 1], length of message produced by sender
        Output:
            encoded message, of shape [BATCH_SIZE x HIDDEN]
        """
        # encode
        embeddings = self.symbol_embeddings(message)
        embeddings = self.lstm(embeddings)[0]
        # select last step corresponding to message
        index = torch.arange(message.size(-1)).expand_as(message).to(DEVICE)
        output = embeddings.masked_select((index == (length-1)).unsqueeze(-1))

        return output.view(embeddings.size(0), embeddings.size(-1))

# Scores images according to a message
class Receiver(nn.Module):
    """
    Defines a receiver policy.
    Based on K presented images and a given message, chooses which image the message refers to.
    """
    def __init__(self, image_encoder=None, symbol_embeddings=None):
        super(Receiver, self).__init__()

        if(image_encoder is None): image_encoder = build_cnn_encoder()
        self.image_encoder = image_encoder

        self.message_encoder = ReceiverMessageEncoder(symbol_embeddings)

    def forward(self, images, message, length):
        """
            Forward propagation.
            Input:
                `images`, of shape [BATCH_SIZE x K x *IMG_SHAPE], where the first of each K image is the target
                `message`, of shape [BATCH_SIZE x (<=MSG_LEN)], message produced by sender
                `length`, of shape [BATCH_SIZE x 1], length of message produced by sender
            Output:
                `Outcome` containing action taken, entropy, log prob, dist and scores.
        """

        # Encodes the images
        original_size = images.size()[:2] #dim 1 & 2 give batch size & K
        encoded_images = self.image_encoder(images.view(-1, *images.size()[2:]))
        encoded_images = encoded_images.view(*original_size, -1)

        # Encodes the message
        encoded_message = self.message_encoder(message, length).unsqueeze(-1)

        # Scores the targets
        scores = torch.bmm(encoded_images, encoded_message).squeeze(-1)

        # Computes the probability distribution
        probs = F.softmax(scores, dim=-1)
        dist = Categorical(probs)

        # Generetates the outcome object
        action = dist.sample() if self.training else probs.argmax(dim=-1)
        entropy = dist.entropy()
        log_prob = dist.log_prob(action)

        outcome = Outcome(
            entropy=entropy,
            log_prob=log_prob,
            action=action,
            dist=dist,
            scores=scores)

        return outcome
