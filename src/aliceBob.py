import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical

import tqdm

from sender import Sender
from receiver import Receiver
from senderReceiver import SenderReceiver
from utils import Progress, show_imgs, max_normalize_, to_color, pointing, add_normal_noise

from config import *

class AliceBob(nn.Module):
    def __init__(self, shared=False):
        nn.Module.__init__(self)

        if(shared):
            senderReceiver = SenderReceiver()
            self.sender = senderReceiver.sender
            self.receiver = senderReceiver.receiver
        else:
            self.sender = Sender()
            self.receiver = Receiver()

    def _bob_input(self, batch):
        return torch.cat([batch.target_img.unsqueeze(1), batch.base_distractors], dim=1)
        
    def _forward(self, batch, sender, receiver):
        sender_outcome = sender(batch.original_img)
        receiver_outcome = receiver(self._bob_input(batch), *sender_outcome.action)

        return sender_outcome, receiver_outcome

    def forward(self, batch):
        """
        Input:
            `batch` is a Batch (a kind of named tuple); 'original_img' and 'target_img' are tensors of shape [BATCH_SIZE, *IMG_SHAPE] and 'base_distractors' is a tensor of shape [BATCH_SIZE, 2, *IMG_SHAPE]
        Output:
            `sender_outcome`, sender.Outcome
            `receiver_outcome`, receiver.Outcome
        """
        return self._forward(batch, self.sender, self.receiver)

    def test_visualize(self, data_iterator):
        self.eval() # Sets the model in evaluation mode; good idea or not?

        batch_size = 4
        batch = data_iterator.get_batch(batch_size)

        batch.original_img.requires_grad = True
        batch.target_img.requires_grad = True
        batch.base_distractors.requires_grad = True

        sender_outcome, receiver_outcome = self(batch)

        # Image-specific saliency visualisation (inspired by Simonyan et al. 2013)
        pseudo_optimizer = torch.optim.Optimizer([batch.original_img, batch.target_img, batch.base_distractors], {}) # I'm defining this only for its `zero_grad` method (but maybe we won't need it)
        pseudo_optimizer.zero_grad()

        _COLOR, _INTENSITY = range(2)
        def process(t, dim, mode):
            if(mode == _COLOR):
                t = max_normalize(t, dim=dim, abs_val=True) # Normalises each image
                t *= 0.5
                t += 0.5

                return t
            elif(mode == _INTENSITY):
                t = t.abs()
                t = t.max(dim).values # Max over the colour channel
                max_normalize_(t, dim=dim, abs_val=False) # Normalises each image
                return to_color(t, dim)

        mode = _INTENSITY

        # Alice's part
        sender_outcome.log_prob.sum().backward()

        sender_part = batch.original_img.grad.detach()
        sender_part = process(sender_part, 1, mode)

        # Bob's part
        receiver_outcome.scores.sum().backward()

        receiver_part_target_img = batch.target_img.grad.detach()
        receiver_part_target_img = process(receiver_part_target_img.unsqueeze(axis=1), 2, mode).squeeze(axis=1)

        receiver_part_base_distractors = batch.base_distractors.grad.detach()
        receiver_part_base_distractors = process(receiver_part_base_distractors, 2, mode)

        # Message Bob-model visualisation (inspired by Simonyan et al. 2013)
        receiver_dream = add_normal_noise((0.5 + torch.zeros_like(batch.original_img)), std_dev=0.1, clamp_values=(0,1)) # Starts with normally-random images
        receiver_dream = receiver_dream.unsqueeze(axis=1)
        receiver_dream.requires_grad = True

        encoded_message = self.receiver.encode_message(*sender_outcome.action).detach()

        #optimizer = torch.optim.RMSprop([receiver_dream], lr=10.0*LR)
        optimizer = torch.optim.SGD([receiver_dream], lr=0.5*LR, momentum=0.9)
        #optimizer = torch.optim.Adam([receiver_dream], lr=10.0*LR)
        nb_iter = 100000
        j = 0
        for i in range(nb_iter):
            if(i >= (j + (nb_iter / 10))):
                print(i)
                j = i

            optimizer.zero_grad()

            tmp_outcome = self.receiver.aux_forward(receiver_dream, encoded_message)
            loss = -tmp_outcome.scores[:, 0].sum()
            
            regularisation_loss = 0.01 * (receiver_dream - 0.5).norm(2) # Similar to L2 regularisation but centered around 0.5
            regularisation_loss += 0.1 * (receiver_dream - 0.5).norm(1) # Similar to L1 regularisation but centered around 0.5
            loss += regularisation_loss
            
            loss.backward()

            optimizer.step()
        receiver_dream = receiver_dream.squeeze(axis=1)
        receiver_dream = torch.clamp(receiver_dream, 0, 1)

        # Displays the visualisations
        imgs = []
        for i in range(batch_size):
            imgs.append(batch.original_img[i].detach())
            imgs.append(sender_part[i])

            imgs.append(batch.target_img[i].detach())
            imgs.append(receiver_part_target_img[i])
            
            for j in range(batch.base_distractors.size(1)):
                imgs.append(batch.base_distractors[i][j].detach())
                imgs.append(receiver_part_base_distractors[i][j])

            imgs.append(receiver_dream[i].detach())
        #for img in imgs: print(img.shape)
        show_imgs(imgs, nrow=(len(imgs) // batch_size)) #show_imgs(imgs, nrow=(2 * (2 + batch.base_distractors.size(1))))

    def sender_rewards(self, sender_action, receiver_scores, running_avg_success):
        """
            returns the reward as well as the success for each element of a batch
        """
        # Generates a probability distribution from the scores and sample an action
        receiver_pointing = pointing(receiver_scores)

        # By design, the target is the first image
        if(args.use_expectation): successes = receiver_pointing['dist'].probs[:, 0].detach()
        else: successes = (receiver_pointing['action'] == 0).float() # Plays dice

        rewards = successes

        if(args.penalty > 0.0):
            msg_lengths = sender_action[1].view(-1).float() # Float casting could be avoided if we upgrade torch to 1.3.1; see https://github.com/pytorch/pytorch/issues/9515 (I believe)
            length_penalties = 1.0 - (1.0 / (1.0 + args.penalty * msg_lengths)) # Equal to 0 when `args.penalty` is set to 0, increases to 1 with the length of the message otherwise

            # TODO J'ai peur que ce système soit un peu trop basique et qu'il encourage le système à être sous-performant - qu'on puisse obtenir plus de reward en faisant exprès de se tromper.
            if(args.adaptative_penalty):
                chance_perf = (1 / receiver_scores.size(1))
                improvement_factor = (running_avg_success - chance_perf) / (1 - chance_perf) # Equals 0 when running average equals chance performance, reaches 1 when running average reaches 1
                length_penalties = (length_penalties * min(0.0, improvement_factor))

            rewards = (rewards - length_penalties)

        return (rewards, successes)

    def sender_loss(self, sender_outcome, receiver_scores, running_avg_success):
        (rewards, successes) = self.sender_rewards(sender_outcome.action, receiver_scores, running_avg_success)
        log_prob = sender_outcome.log_prob.sum(dim=1)

        loss = -(rewards * log_prob).mean()

        loss = loss - (BETA_SENDER * sender_outcome.entropy.mean()) # Entropy penalty
        
        return (loss, successes, rewards)

    def receiver_loss(self, receiver_scores):
        receiver_pointing = pointing(receiver_scores) # The sampled action is not the same as the one in `sender_rewards` but it probably does not matter

        # By design, the target is the first image
        if(args.use_expectation):
            successes = receiver_pointing['dist'].probs[:, 0].detach()
            log_prob = receiver_pointing['dist'].log_prob(torch.tensor(0).to(DEVICE))
        else: # Plays dice
            successes = (receiver_pointing['action'] == 0).float()
            log_prob = receiver_pointing['dist'].log_prob(receiver_pointing['action'])

        rewards = successes

        loss = -(rewards * log_prob).mean()

        loss = loss - (BETA_RECEIVER * receiver_pointing['dist'].entropy().mean()) # Entropy penalty
        
        return loss

    def train_epoch(self, data_iterator, optim, epoch=1, steps_per_epoch=1000, event_writer=None):
        """
            Model training function
            Input:
                `data_iterator`, an infinite iterator over (batched) data
                `optim`, the optimizer
            Optional arguments:
                `epoch`: epoch number to display in progressbar
                `steps_per_epoch`: number of steps for epoch
                `event_writer`: tensorboard writer to log evolution of values
        """
        self.train() # Sets the model in training mode

        with Progress(SIMPLE_DISPLAY, steps_per_epoch, epoch) as pbar:
            total_reward = 0.0 # sum of the rewards since the beginning of the epoch
            total_success = 0.0 # sum of the successes since the beginning of the epoch
            total_items = 0 # number of training instances since the beginning of the epoch
            running_avg_reward = 0.0
            running_avg_success = 0.0
            start_i = ((epoch - 1) * steps_per_epoch) + 1 # (the first epoch is numbered 1, and the first iteration too)
            end_i = start_i + steps_per_epoch

            for i, batch in zip(range(start_i, end_i), data_iterator):
                optim.zero_grad()
                sender_outcome, receiver_outcome = self(batch)

                # Alice's part
                (sender_loss, sender_successes, sender_rewards) = self.sender_loss(sender_outcome, receiver_outcome.scores, running_avg_success)

                # Bob's part
                receiver_loss = self.receiver_loss(receiver_outcome.scores)

                loss = sender_loss + receiver_loss

                loss.backward() # Backpropagation

                # Gradient clipping and scaling
                if((CLIP_VALUE is not None) and (CLIP_VALUE > 0)): torch.nn.utils.clip_grad_value_(self.parameters(), CLIP_VALUE)
                if((SCALE_VALUE is not None) and (SCALE_VALUE > 0)): torch.nn.utils.clip_grad_norm_(self.parameters(), SCALE_VALUE)

                optim.step()

                rewards = sender_rewards
                successes = sender_successes

                avg_reward = rewards.mean().item() # average reward of the batch
                avg_success = successes.mean().item() # average success of the batch
                avg_msg_length = sender_outcome.action[1].float().mean().item() # average message length of the batch

                # updates running average reward
                total_reward += rewards.sum().item()
                total_success += successes.sum().item()
                total_items += batch.size
                running_avg_reward = total_reward / total_items
                running_avg_success = total_success / total_items

                pbar.update(R=running_avg_success)

                # logs some values
                if(event_writer is not None):
                    number_ex_seen = i * BATCH_SIZE
                    event_writer.add_scalar('train/reward', avg_reward, number_ex_seen)
                    event_writer.add_scalar('train/success', avg_success, number_ex_seen)
                    event_writer.add_scalar('train/loss', loss.item(), number_ex_seen)
                    event_writer.add_scalar('train/msg_length', avg_msg_length, number_ex_seen)
                    if DEBUG_MODE:
                        median_grad = torch.cat([p.grad.view(-1).detach() for p in self.parameters()]).abs().median().item()
                        mean_grad = torch.cat([p.grad.view(-1).detach() for p in self.parameters()]).abs().mean().item()
                        max_grad = torch.cat([p.grad.view(-1).detach() for p in self.parameters()]).abs().max().item()
                        mean_norm_grad = torch.stack([p.grad.view(-1).detach().data.norm(2.) for p in self.parameters()]).mean().item()
                        max_norm_grad = torch.stack([p.grad.view(-1).detach().data.norm(2.) for p in self.parameters()]).max().item()
                        event_writer.add_scalar('train/median_grad', median_grad, number_ex_seen)
                        event_writer.add_scalar('train/mean_grad', mean_grad, number_ex_seen)
                        event_writer.add_scalar('train/max_grad', max_grad, number_ex_seen)
                        event_writer.add_scalar('train/mean_norm_grad', mean_norm_grad, number_ex_seen)
                        event_writer.add_scalar('train/max_norm_grad', max_norm_grad, number_ex_seen)

        self.eval()
