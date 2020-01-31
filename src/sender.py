import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from config import *
from utils import build_cnn_encoder

# Structure for outcomes
Outcome = namedtuple("Policy", ["entropy", "log_prob", "action"])

class SenderMessageDecoder(nn.Module):
    def __init__(self):
        super(SenderMessageDecoder, self).__init__()
        self.embedding = nn.Embedding(ALPHABET_SIZE + 2, HIDDEN, padding_idx=PAD)
        self.lstm = nn.LSTM(HIDDEN, HIDDEN, 1)
        # project encoded img onto cell
        self.cell_proj = nn.Linear(HIDDEN, HIDDEN)
        # project encoded img onto hidden
        self.hidden_proj = nn.Linear(HIDDEN, HIDDEN)
        # project lstm output onto action space
        self.action_space_proj = nn.Linear(HIDDEN, ALPHABET_SIZE)

    def forward(self, encoded):
        # initializes first input and state
        input = torch.ones(encoded.size(0)).long().to(DEVICE) * BOS
        input = self.embedding(input)
        cell = self.cell_proj(encoded).unsqueeze(0)
        hidden = self.hidden_proj(encoded).unsqueeze(0)
        state = (cell, hidden)

        # outputs
        message = []
        log_probs = []
        entropy = []

        # stopping mechanism when EOS has been produced
        has_stopped = torch.zeros(encoded.size(0)).bool().to(DEVICE)
        has_stopped.requires_grad = False

        # produces message
        for i in range(MSG_LEN):
            output, state = self.lstm(input.unsqueeze(0), state)
            output = self.action_space_proj(output).squeeze(0)

            # selects action for step
            probs =  F.softmax(output, dim=-1)
            dist = Categorical(probs)
            action = dist.sample() if self.training else probs.argmax(dim=-1)

            # ignores prediction for completed messages
            ent = dist.entropy() * (~has_stopped).float()
            log_p = dist.log_prob(action) * (~has_stopped).float()
            log_probs.append(log_p)
            entropy.append(ent)

            action = action.masked_fill(has_stopped, PAD)
            message.append(action)

            # If all messages are finished
            has_stopped = has_stopped | (action == EOS)
            if has_stopped.all():
                break
            # next input
            input = self.embedding(action)

        # converts output to tensor
        message = torch.stack(message, dim=1)
        message_len = (message != PAD).cumsum(dim=1)[:,-1,None]
        log_probs = torch.stack(log_probs, dim=1)

        # average entropy over timesteps, hence ignore padding
        entropy = torch.stack(entropy, dim=1)
        entropy = entropy.sum(dim=1, keepdim=True)
        entropy = entropy / message_len.float()

        outcome = Outcome(
            entropy=entropy,
            log_prob=log_probs,
            action=(message, message_len))
        return outcome


class SenderPolicy(nn.Module):
    def __init__(self):
        super(SenderPolicy, self).__init__()
        self.image_encoder = build_cnn_encoder()
        self.message_decoder = SenderMessageDecoder()

    def forward(self, image):
        """
            Forward propagation.
            Input:
                `image`, of shape [BATCH_SIZE x *IMG_SHAPE]
            Output:
                `Outcome`, where `action` is the produced message
        """
        encoded_image = self.image_encoder(image)
        outcome = self.message_decoder(encoded_image)
        return outcome
