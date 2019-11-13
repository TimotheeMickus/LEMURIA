from datetime import datetime as t

import torch.nn as nn
import torch.optim as optim

from config import *
from data import get_dataloader
from speaker import Speaker
from listener import Listener

def compute_reward(action_l):
    reward = (action_l == 0).float()
    return reward

def compute_log_p(log_p_s, log_p_l):
    """
    Input:
        `log_p_q`, log probs for speaker policy
        `log_p_l`, log probs for listener policy
    Output:
        \bigg(\sum \limits_{l=1}^L \log p_{\pi^s}(m^l_t|m^{<l}_t, u) + \log p_{\pi^L}(u_{t'}|z, U) \bigg)
    """
    log_p = log_p_s.sum(dim=1) + log_p_l
    return log_p

def build_optimizer(θ):
    return optim.RMSprop(θ, lr=LR)

class CommunicationGame(nn.Module):
    def __init__(self):
        super(CommunicationGame, self).__init__()
        self.speaker = Speaker()
        self.listener = Listener()

    def forward(self, inputs, return_message=True):
        """
        Input:
            `inputs` of shape [Batch, K, 3, 124, 124]
        """
        # input[:,0] is target, the remainder are distractors
        inputs = inputs.float()
        target_img = inputs[:,0]
        message, lens, log_p_s, h_s = self.speaker(target_img)
        action_l, h_l, log_p_l = self.listener(inputs, message, lens)
        if return_message:
            return log_p_s, h_s, action_l, h_l, log_p_l, message
        return log_p_s, h_s, action_l, h_l, log_p_l

def train_epoch(model, data_iterator, optim):
    model.train()
    avg_loss = 0.
    avg_acc = 0.
    print(t.now(), "training start...")
    for i,batch in enumerate(data_iterator, start=1):
        batch = batch.to(DEVICE)
        optim.zero_grad()
        log_p_s, h_s, action_l, h_l, log_p_l, action_s = model(batch)

        R = compute_reward(action_l)
        log_p = compute_log_p(log_p_s, log_p_l)
        loss = - (R * log_p)

        loss = loss.mean()
        loss = loss - BETA_L * h_l.mean()
        loss = loss - BETA_S * h_s.mean()

        loss.backward()
        optim.step()

        # logger variables
        avg_loss += loss.item()
        acc = R.sum().item()
        avg_acc += acc / BATCH_SIZE
        # a broken entropy regularization
        if i % 100 == 0:
            print(t.now(), "loss at step %i: %f, avg. acc: %f" % (i, avg_loss / i, avg_acc /i))
    model.eval()

if __name__ == "__main__":
    model = CommunicationGame().to(DEVICE)
    optimizer = build_optimizer(model.parameters())
    import torch
    data_loader = get_dataloader()
    train_epoch(model, data_loader, optimizer)
