import torch.nn as nn
import torch.optim as optim

from config import *
from data import get_dataloader
from speaker import Speaker
from listener import Listener
from datetime import datetime as t

def compute_reward(log_p_s, action_l, log_p_l):
    """
    Input:
        `π_s` of shape [Batch x *], policy for speaker (second dim up to MSG_LEN)
        `π_l` of shape [Batch x 1], policy for listener
        `log_p_l`, log probs for speaker policy
        `h_s`  of shape [Batch x 1], entropy for speaker (averaged)
        `h_l`  of shape [Batch x K], entropy for listener
    Output: loss
    cf. §2.3 :
        The objective functon that the two agents maximize for one training instance is:
        R(t') \bigg(\sum \limits_{l=1}^L \log p_{\pi^s}(m^l_t|m^{<l}_t, u) + \log p_{\pi^L}(u_{t'}|z, U) \bigg)
        where R is the reward function returning 1 if t = t' (if the listener pointed to the correct target) and 0 otherwise.
        To maintain exploration in the speaker's policy \pi^S of generating a message, and the listener's policy \pi^L of pointing to a target, we add to the loss an entropy regularization term.
    """
    reward = (action_l == 0).float() * (log_p_s.sum(dim=1) + log_p_l)
     # + (BETA_S * h_s.view(-1)) + (BETA_L * h_l.view(-1))
    return reward

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
    print(t.now(), "training start...")
    for i,batch in enumerate(data_iterator, start=1):
        batch = batch.to(DEVICE)
        optim.zero_grad()
        log_p_s, h_s, action_l, h_l, log_p_l, action_s = model(batch)
        R = compute_reward(log_p_s, action_l, log_p_l)

        # a broken entropy regularization
        #loss = loss + BETA_L * h_l.mean()
        #loss = loss + BETA_S * h_s.mean()
        # reinforce : log_p * R(a). We probably want to backprop with respect to policy only
        # reinforce wrt. speaker policy
        (-log_p_s * R.unsqueeze(1)).sum().backward(retain_graph=True)
        # reinforce wrt. listener policy
        (-log_p_l * R).sum().backward()

        avg_loss += -R.mean().item()
        optim.step()
        if i % 100 == 0:
            print(t.now(), "loss at step %i: %f" % (i, avg_loss / i))
    model.eval()

if __name__ == "__main__":
    model = CommunicationGame().to(DEVICE)
    optimizer = build_optimizer(model.parameters())
    import torch
    data_loader = get_dataloader()
    train_epoch(model, data_loader, optimizer)
