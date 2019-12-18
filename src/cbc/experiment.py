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
from utils import build_optimizer, show_img, show_imgs

sys.path.remove(parent_dir_path)
# [END] Imports shared code from the parent directory

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.sender = SenderPolicy()
        self.receiver = ReceiverPolicy()

    def forward(self, batch):
        """
        Input:
            `batch` is a Batch (a kind of named tuple); 'alice_input' is a tensor of shape [BATCH_SIZE, *IMG_SHAPE] and 'bob_input' is a tensor of shape [BATCH_SIZE, K, *IMG_SHAPE]
        Output:
            `sender_outcome`, `PolicyOutcome` for sender
            `receiver_outcome`, `PolicyOutcome` for receiver
        """
        
        #show_img(batch.alice_input[0])
        #show_imgs(batch.bob_input[0])

        sender_outcome = self.sender(batch.alice_input)

        receiver_outcome = self.receiver(batch.bob_input, *sender_outcome.action)

        return sender_outcome, receiver_outcome

def compute_rewards(sender_action, receiver_action):
    """
        returns the reward as well as the success for each element of a batch
    """
    successes = (receiver_action == 0).float() # by design, the first image is the target
    
    guess_rewards = successes

    msg_lengths = torch.squeeze(sender_action[1], dim=1).float() # Il est très important de floater, sinon ça fait n'importe quoi
    length_penalties = 1.0 - (1.0 / (1.0 + args.penalty * msg_lengths)) # Equal to 0 when `args.penalty` is set to 0, increases to 1 with the length of the message otherwise

    rewards = (guess_rewards - length_penalties)

    return (rewards, successes)

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
            `model`, a `Model` model
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
            optim.zero_grad()
            sender_outcome, receiver_outcome = model(batch)

            (rewards, successes) = compute_rewards(sender_outcome.action, receiver_outcome.action)
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

            avg_reward = rewards.mean().item() # average reward of the batch
            avg_success = successes.mean().item() # average success of the batch
            msg_lengths = torch.squeeze(sender_outcome.action[1], dim=1).float() # Il est très important de floater, sinon ça fait n'importe quoi
            avg_msg_length = msg_lengths.mean().item()

            # updates running average reward
            total_reward += rewards.sum().item()
            total_items += batch.size

            if(callback is not None): callback(total_reward / total_items)

            # logs some values
            if event_writer is not None:
                event_writer.add_scalar('train/reward', avg_reward, i)
                event_writer.add_scalar('train/success', avg_success, i)
                event_writer.add_scalar('train/loss', loss.item(), i)
                event_writer.add_scalar('train/msg_length', avg_msg_length, i)

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

        model = Model().to(DEVICE)
        optimizer = build_optimizer(model.parameters())
        data_loader = get_data_loader()
        event_writer = SummaryWriter(run_summary_dir)

        print(datetime.now(), "training start...")
        for epoch in range(1, (EPOCHS + 1)):
            train_epoch(model, data_loader, optimizer, epoch=epoch, event_writer=event_writer)
            if(SAVE_MODEL): torch.save(model.state_dict(), os.path.join(run_models_dir, ("model_e%i.pt" % epoch)))
