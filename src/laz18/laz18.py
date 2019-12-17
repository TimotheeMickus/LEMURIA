#!/usr/bin/env python

from datetime import datetime
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tqdm

from config import *
from data import get_data_loader

# [START] Imports shared code from the parent directory
parent_dir_path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(parent_dir_path)

from receiver import ReceiverPolicy
from sender import SenderPolicy
from utils import build_optimizer

sys.path.remove(parent_dir_path)
# [END] Imports shared code from the parent directory

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
        inputs = inputs.float() # Makes sure the images are float tensors
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
    
    def loop(callback=None):
        total_reward = 0.0 # sum of the rewards since the beginning of the epoch 
        total_items = 0 # number of training instances since the beginning of the epoch
        start_i = ((epoch - 1) * steps_per_epoch) + 1 # (the first epoch is numbered 1, and the first iteration too)
        end_i = start_i + steps_per_epoch

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

            if(callback is not None): callback(total_reward / total_items)

            # logs some values
            if event_writer is not None:
                event_writer.add_scalar('train/reward', avg_reward, i)
                event_writer.add_scalar('train/loss', loss.item(), i)

    if(SIMPLE_DISPLAY):
        def callback(r):
            print('R: %f' % r)

        loop(callback)
    else:
        with tqdm.tqdm(total=steps_per_epoch, postfix={"R": 0.0}, unit="B", desc=("Epoch %i" % epoch)) as pbar:
            def callback(r):
                pbar.set_postfix({"R" : r}, refresh=False)
                pbar.update()

            loop(callback)

    model.eval()

    return model # TODO Is there any reason to return the model?

if __name__ == "__main__":
    if(not os.path.isdir(DATASET_PATH)):
        print("Directory '%s' not found." % DATASET_PATH)
        sys.exit()

    for run in range(RUNS):
        print('Run %i' % run)

        run_models_dir = os.path.join(MODELS_DIR, str(run))
        run_summary_dir = os.path.join(SUMMARY_DIR, str(run))

        if(not os.path.isdir(run_summary_dir)): os.makedirs(run_summary_dir)
        if(SAVE_MODEL and (not os.path.isdir(run_models_dir))): os.makedirs(run_models_dir)

        model = CommunicationGame().to(DEVICE)
        optimizer = build_optimizer(model.parameters())
        data_loader = get_data_loader()
        event_writer = SummaryWriter(run_summary_dir)

        print(datetime.now(), "training start...")
        for epoch in range(1, (EPOCHS + 1)):
            train_epoch(model, data_loader, optimizer, epoch=epoch, event_writer=event_writer)
            if(SAVE_MODEL): torch.save(model.state_dict(), os.path.join(run_models_dir, ("model_e%i.pt" % epoch)))
