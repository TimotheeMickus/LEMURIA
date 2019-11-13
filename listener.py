import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from config import *
from utils import build_cnn_encoder


class ListenerMessageEncoder(nn.Module):
    def __init__(self):
        super(ListenerMessageEncoder, self).__init__()
        self.embedding = nn.Embedding(ALPHABET_SIZE + 1, HIDDEN, padding_idx=PAD)
        self.lstm = nn.LSTM(HIDDEN, HIDDEN, 1, batch_first=True)

    def forward(self, messages, lengths):
        embeddings = self.embedding(messages)
        # grab the last state of the encoder
        index = torch.arange(messages.size(-1)).expand_as(messages).to(DEVICE)
        embeddings = self.lstm(embeddings)[0]
        outputs = embeddings.masked_select((index == (lengths-1)).unsqueeze(-1))
        return outputs.view(embeddings.size(0), embeddings.size(-1))

class Listener(nn.Module):
    def __init__(self):
        super(Listener, self).__init__()
        self.obj_encoder = build_cnn_encoder()
        self.msg_encoder = ListenerMessageEncoder()

    def forward(self, objects, messages, lengths):
        # encode images
        original_size = objects.size()[:2]
        objs = self.obj_encoder(objects.view(-1, *objects.size()[2:]))
        objs = objs.view(*original_size, -1)

        # encode message
        msg_embedding = self.msg_encoder(messages, lengths)

        #score
        scores = torch.bmm(objs, msg_embedding.unsqueeze(-1)).squeeze(-1)
        probs = F.softmax(scores, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        h = dist.entropy()
        log_p = dist.log_prob(action)

        return action, h, log_p
