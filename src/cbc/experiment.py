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

        sender_outcome = self.sender(batch.alice_input)

        receiver_outcome = self.receiver(batch.bob_input, *sender_outcome.action)

        return sender_outcome, receiver_outcome

def compute_rewards(sender_action, receiver_action, running_avg_success, chance_perf):
    """
        returns the reward as well as the success for each element of a batch
    """
    successes = (receiver_action == 0).float() # by design, the first image is the target

    rewards = successes

    if(args.penalty > 0.0):
        msg_lengths = sender_action[1].view(-1).float() # Float casting could be avoided if we upgrade torch to 1.3.1; cf. https://github.com/pytorch/pytorch/issues/9515 (I believe)
        length_penalties = 1.0 - (1.0 / (1.0 + args.penalty * msg_lengths)) # Equal to 0 when `args.penalty` is set to 0, increases to 1 with the length of the message otherwise
        
        # TODO J'ai peur que ce système soit un peu trop basique et qu'il encourage le système à être sous-performant - qu'on puisse obtenir plus de reward en faisant exprès de se tromper.
        if(args.adaptative_penalty):
            improvement_factor = (running_avg_success - chance_perf) / (1 - chance_perf) # Equals 0 when running average equals chance performance, reachs 1 when running average reaches 1
            length_penalties = (length_penalties * min(0.0, improvement_factor))
        
        rewards = (rewards - length_penalties)

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

def test_visualize(model, data_iterator):
    model.eval() # Sets the model in evaluation mode; good idea or not?
    
    batch_size = 2 # Maybe it would just be simpler to work with multiple batches of size 1
    batch = data_iterator.get_batch(batch_size)

    batch.alice_input.requires_grad = True
    batch.bob_input.requires_grad = True

    pseudo_optimizer = torch.optim.Optimizer(list(model.parameters()) + [batch.alice_input, batch.bob_input], {}) # I'm defining this only for its `zero_grad` method (but maybe we won't need it)

    sender_outcome, receiver_outcome = model(batch)

    pseudo_optimizer.zero_grad()
    
    # Alice's part
    sender_outcome.log_prob.sum().backward()

    batch.alice_input.grad *= 10
    torch.clamp_(batch.alice_input.grad, 0.5)
    batch.alice_input.grad += 0.5

    # Bob's part
    receiver_outcome.scores.sum().backward()

    batch.bob_input.grad *= 10
    torch.clamp_(batch.bob_input.grad, 0.5)
    batch.bob_input.grad += 0.5

    imgs = []
    img_per_batch = batch.bob_input.shape[1]
    for i in range(batch_size):
        imgs.append(batch.alice_input[i].detach())
        imgs.append(batch.alice_input.grad[i])
        for j in range(img_per_batch):
            imgs.append(batch.bob_input[i][j].detach())
            imgs.append(batch.bob_input.grad[i][j])
    show_imgs(imgs, nrow=(2 * (1 + img_per_batch)))

def train_epoch(model, data_iterator, optim, epoch=1, steps_per_epoch=10, event_writer=None):
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
    model.train() # Sets the model in training mode

    if(SIMPLE_DISPLAY):
        class Progress:
            def __enter__(self):
                self.i = 0

                return self

            def update(self, r):
                print('%i/%i - R: %f' % (self.i, steps_per_epoch, r))
                self.i += 1

            def __exit__(self, type, value, traceback):
                pass
    else:
        class Progress:
            def __enter__(self):
                self.pbar = tqdm.tqdm(total=steps_per_epoch, postfix={"R": 0.0}, unit="B", desc=("Epoch %i" % epoch)) # Do not forget to close it at the end

                return self

            def update(self, r):
                self.pbar.set_postfix({"R" : r}, refresh=False)
                self.pbar.update()

            def __exit__(self, type, value, traceback):
                self.pbar.close()
    
    with Progress() as pbar:
        total_reward = 0.0 # sum of the rewards since the beginning of the epoch
        total_success = 0.0 # sum of the successes since the beginning of the epoch
        total_items = 0 # number of training instances since the beginning of the epoch
        running_avg_reward = 0.0
        running_avg_success = 0.0
        start_i = ((epoch - 1) * steps_per_epoch) + 1 # (the first epoch is numbered 1, and the first iteration too)
        end_i = start_i + steps_per_epoch

        for i, batch in zip(range(start_i, end_i), data_iterator):
            optim.zero_grad()
            sender_outcome, receiver_outcome = model(batch)

            chance_perf = (1 / batch.bob_input.shape[1]) # The chance performance is 1 over the number of images shown to Bob
            (rewards, successes) = compute_rewards(sender_outcome.action, receiver_outcome.action, running_avg_success, chance_perf)
            log_prob = compute_log_prob(sender_outcome.log_prob, receiver_outcome.log_prob)
            loss = -(rewards * log_prob)

            loss = loss.mean()
            # entropy penalties
            loss = loss - (BETA_SENDER * sender_outcome.entropy.mean())
            loss = loss - (BETA_RECEIVER * receiver_outcome.entropy.mean())

            # backprop
            loss.backward()

            # Gradient clipping and scaling
            if((CLIP_VALUE is not None) and (CLIP_VALUE > 0)): torch.nn.utils.clip_grad_value_(model.parameters(), CLIP_VALUE)
            if((SCALE_VALUE is not None) and (SCALE_VALUE > 0)): torch.nn.utils.clip_grad_norm_(model.parameters(), SCALE_VALUE)
            
            optim.step()

            avg_reward = rewards.mean().item() # average reward of the batch
            avg_success = successes.mean().item() # average success of the batch
            avg_msg_length = sender_outcome.action[1].float().mean().item() # average message length of the batch

            # updates running average reward
            total_reward += rewards.sum().item()
            total_success += successes.sum().item()
            total_items += batch.size
            running_avg_reward = total_reward / total_items
            running_avg_success = total_success / total_items

            pbar.update(running_avg_success)

            # logs some values
            if(event_writer is not None):
                number_ex_seen = i * BATCH_SIZE
                event_writer.add_scalar('train/reward', avg_reward, number_ex_seen)
                event_writer.add_scalar('train/success', avg_success, number_ex_seen)
                event_writer.add_scalar('train/loss', loss.item(), number_ex_seen)
                event_writer.add_scalar('train/msg_length', avg_msg_length, number_ex_seen)
                if DEBUG_MODE:
                    median_grad = torch.cat([p.grad.view(-1).detach() for p in model.parameters()]).abs().median().item()
                    mean_grad = torch.cat([p.grad.view(-1).detach() for p in model.parameters()]).abs().mean().item()
                    #min_grad = torch.cat([p.grad.view(-1).detach() for p in model.parameters()]).abs().min().item()
                    max_grad = torch.cat([p.grad.view(-1).detach() for p in model.parameters()]).abs().max().item()
                    mean_norm_grad = torch.stack([p.grad.view(-1).detach().data.norm(2.) for p in model.parameters()]).mean().item()
                    max_norm_grad = torch.stack([p.grad.view(-1).detach().data.norm(2.) for p in model.parameters()]).max().item()
                    event_writer.add_scalar('train/median_grad', median_grad, number_ex_seen)
                    event_writer.add_scalar('train/mean_grad', mean_grad, number_ex_seen)
                    #event_writer.add_scalar('train/min_grad', min_grad, number_ex_seen)
                    event_writer.add_scalar('train/max_grad', max_grad, number_ex_seen)
                    event_writer.add_scalar('train/mean_norm_grad', mean_norm_grad, number_ex_seen)
                    event_writer.add_scalar('train/max_norm_grad', max_norm_grad, number_ex_seen)
    
    model.eval()

if(__name__ == "__main__"):
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
            #test_visualize(model, data_loader)
