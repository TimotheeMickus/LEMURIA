import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from config import *
from utils import build_cnn_encoder, PolicyOutcome

class ReceiverMessageEncoder(nn.Module):
    """
    Encodes a message of discrete symbols in a single embedding.
    """
    def __init__(self):
        super(ReceiverMessageEncoder, self).__init__()
        self.embedding = nn.Embedding(ALPHABET_SIZE + 1, HIDDEN, padding_idx=PAD)
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
        embeddings = self.embedding(message)
        embeddings = self.lstm(embeddings)[0]
        # select last step corresponding to message
        index = torch.arange(message.size(-1)).expand_as(message).to(DEVICE)
        output = embeddings.masked_select((index == (length-1)).unsqueeze(-1))

        return output.view(embeddings.size(0), embeddings.size(-1))

class ReceiverPolicy(nn.Module):
    """
    Defines a receiver policy.
    Based on K presented images and a given message, chooses which image the message refers to.
    """
    def __init__(self):
        super(ReceiverPolicy, self).__init__()
        self.image_encoder = build_cnn_encoder()
        self.message_encoder = ReceiverMessageEncoder()

    def forward(self, images, message, length):
        """
            Forward propagation.
            Input:
                `images`, of shape [BATCH_SIZE x K x *IMG_SHAPE], where the first of each K image is the target
                `message`, of shape [BATCH_SIZE x <=MSG_LEN], message produced by sender
                `length`, of shape [BATCH_SIZE x 1], length of message produced by sender
            Output:
                `PolicyOutcome` containing action taken, entropy and log prob.
        """
        # encode images
        original_size = images.size()[:2] #dim 1 & 2 give batch size & K
        encoded_images = self.image_encoder(images.view(-1, *images.size()[2:]))
        encoded_images = encoded_images.view(*original_size, -1)

        # encode message
        encoded_message = self.message_encoder(message, length).unsqueeze(-1)

        # score targets
        scores = torch.bmm(encoded_images, encoded_message).squeeze(-1)
        # compute distribution
        probs = F.softmax(scores, dim=-1)
        dist = Categorical(probs)

        # get outcome
        action = dist.sample() if self.training else probs.argmax(dim=-1)
        entropy = dist.entropy()
        log_prob = dist.log_prob(action)

        outcome = PolicyOutcome(
            entropy=entropy,
            log_prob=log_prob,
            action=action)

        return outcome
