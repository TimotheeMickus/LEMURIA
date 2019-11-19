import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


from config import *
from utils import build_cnn_encoder

class SpeakerDecoder(nn.Module):
    def __init__(self):
        super(SpeakerDecoder, self).__init__()
        self.embedding = nn.Embedding(ALPHABET_SIZE + 2, HIDDEN, padding_idx=PAD)
        self.lstm = nn.LSTM(HIDDEN, HIDDEN, 1)
        self.proj = nn.Linear(HIDDEN, ALPHABET_SIZE)

    def forward(self, encoded, return_stats=True):
        ipt = torch.ones(BATCH_SIZE).long().to(DEVICE) * BOS
        ipt = self.embedding(ipt)
        state = (encoded.unsqueeze(0), encoded.unsqueeze(0))

        # output(s)
        msg = []
        stats = {},
        if return_stats:
            log_probs = []
            h = []

        # stopping mechanism when EOS has been produced
        has_stopped = torch.zeros(BATCH_SIZE).bool().to(DEVICE)
        has_stopped.requires_grad = False

        for i in range(MSG_LEN):
            l_opt, state = self.lstm(ipt.unsqueeze(0), state)
            opt = self.proj(l_opt).squeeze(0)

            probs =  F.softmax(opt, dim=-1)
            #select action
            dist = Categorical(probs)
            action = dist.sample()

            if return_stats:
                # ignore prediction for completed messages
                ent = dist.entropy() * (~has_stopped).float()
                log_p = dist.log_prob(action) * (~has_stopped).float()
                log_probs.append(log_p)
                h.append(ent)

            action = action.masked_fill(has_stopped, PAD)
            msg.append(action)
            # early stopping
            has_stopped = has_stopped | (action == EOS)
            if has_stopped.all():
                break
            # next input
            ipt = self.embedding(action)

        # to tensor
        msg = torch.stack(msg, dim=1)
        msg_len = (msg != PAD).cumsum(dim=1)[:,-1,None]
        if return_stats:
            log_probs = torch.stack(log_probs, dim=1)
            #log_probs = log_probs.masked_fill(msg == PAD, 0.)
            h = torch.stack(h, dim=1)
            h = h.sum(dim=1, keepdim=True)
            h = h / msg_len.float()
            return (msg, msg_len), {"log_p": log_probs, "ent": h}
        return (msg, msg_len)


class Speaker(nn.Module):
    def __init__(self):
        super(Speaker, self).__init__()
        self.encoder = build_cnn_encoder()
        self.decoder = SpeakerDecoder()

    def forward(self, objects):
        encoded = self.encoder(objects)
        (decoded, decoded_len), stats = self.decoder(encoded)
        return decoded, decoded_len, stats["log_p"], stats["ent"]
