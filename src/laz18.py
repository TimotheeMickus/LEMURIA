#!/usr/bin/env python

from datetime import datetime
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tqdm

from config import *
from data import get_dataloader
from receiver import ReceiverPolicy
from sender import SenderPolicy
from utils import build_optimizer

class CommunicationGame(nn.Module):
    def __init__(self):
        super(CommunicationGame, self).__init__()

        self.sender = SenderPolicy()
        self.receiver = ReceiverPolicy()

    def forward(self, inputs):
        """
        Input:
            `inputs` of shape [Batch, K, *IMG_SHAPE]. The target image is the first of the K images
        Output:
            `sender_outcome`, `PolicyOutcome` for sender
            `receiver_outcome`, `PolicyOutcome` for receiver
        """
        inputs = inputs.float() # Converts the images from ??? to float TODO
        target_img = inputs[:,0] # These are the targets (the others are distractors)

        sender_inputs = target_img
        if(NOISE_STD_DEV > 0.0): sender_inputs = torch.clamp((sender_inputs + (NOISE_STD_DEV * torch.randn(size=sender_inputs.shape))), 0.0, 1.0) # Adds normal random noise, then clamps
        sender_outcome = self.sender(sender_inputs)

        receiver_inputs = inputs
        if(NOISE_STD_DEV > 0.0): receiver_inputs = torch.clamp((receiver_inputs + (NOISE_STD_DEV * torch.randn(size=receiver_inputs.shape))), 0.0, 1.0) # Adds normal random noise, then clamps
        receiver_outcome = self.receiver(receiver_inputs, *sender_outcome.action)

        return sender_outcome, receiver_outcome

def compute_rewards(receiver_action):
    """
        returns the reward for each element of a batch
    """
    # by design, the first image is the target
    rewards = (receiver_action == 0).float()

    return rewards

def compute_log_prob(sender_log_prob, receiver_log_prob):
    """
    Input:
        `sender_log_prob`, log probs for sender policy
        `receiver_log_prob`, log prob for receiver policy
    Output:
        \bigg(\sum \limits_{l=1}^L \log p_{\pi^s}(m^l_t|m^{<l}_t, u) + \log p_{\pi^L}(u_{t'}|z, U) \bigg)
    """
    log_prob = sender_log_prob.sum(dim=1) + receiver_log_prob
    return log_prob


def train_epoch(model, data_iterator, optim, epoch=1, steps_per_epoch=1000, event_writer=None):
    """
        Model training function
        Input:
            `model`, a `CommunicationGame` model
            `data_iterator`, an infinite iterator over (batched) data
            `optim`, the optimizer
        Optional arguments:
            `epoch`: epoch number to display in progressbar
            `steps_per_epoch`: number of steps for epoch
            `event_writer`: tensorboard writer to log evolution of values
    """
    model.train() # sets the model in training mode
    
    total_reward = 0.0 # sum of the rewards since the beginning of the epoch 
    total_items = 0 # number of training instances since the beginning of the epoch
    start_i = ((epoch - 1) * steps_per_epoch) + 1 # (the first epoch is numbered 1, and the first iteration too)
    end_i = start_i + steps_per_epoch
    with tqdm.tqdm(total=steps_per_epoch, postfix={"R": total_reward}, unit="B", desc=("Epoch %i" % epoch)) as pbar:
        for i, batch in zip(range(start_i, end_i), data_iterator):
            batch = batch.to(DEVICE)
            optim.zero_grad()
            sender_outcome, receiver_outcome = model(batch)

            rewards = compute_rewards(receiver_outcome.action)
            log_prob = compute_log_prob(
                sender_outcome.log_prob,
                receiver_outcome.log_prob)
            loss = - (rewards * log_prob)

            loss = loss.mean()
            # entropy penalties
            loss = loss - (BETA_SENDER * sender_outcome.entropy.mean())
            loss = loss - (BETA_RECEIVER * receiver_outcome.entropy.mean())

            # backprop
            loss.backward()
            optim.step()

            # updates running average reward
            avg_reward = rewards.sum().item() / batch.size(0) # average reward of the batch
            total_reward += rewards.sum().item()
            total_items += batch.size(0)

            pbar.set_postfix({"R" : total_reward / total_items}, refresh=False)
            pbar.update()

            # logs some values
            if event_writer is not None:
                event_writer.add_scalar('train/reward', avg_reward, i)
                event_writer.add_scalar('train/loss', loss.item(), i)

    model.eval()

    return model # TODO Is there any reason to return the model?

if __name__ == "__main__":
    if(not os.path.isdir(DATASET_PATH)):
        print("Directory '%s' not found." % DATASET_PATH)
        sys.exit()

    model = CommunicationGame().to(DEVICE)
    optimizer = build_optimizer(model.parameters())
    data_loader = get_data_loader()
    event_writer = SummaryWriter(SUMMARY_DIR)

    if(SAVE_MODEL and (not os.path.isdir(MODELS_DIR))):
        os.makedirs(MODELS_DIR)

    print(datetime.now(), "training start...")
    for epoch in range(1, (EPOCHS + 1)):
        train_epoch(model, data_loader, optimizer, epoch=epoch, event_writer=event_writer)
        if(SAVE_MODEL): torch.save(model.state_dict(), os.path.join(MODELS_DIR, ("model_e%i.pt" % epoch)))
