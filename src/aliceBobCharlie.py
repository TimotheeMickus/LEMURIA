import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from sender import Sender
from receiver import Receiver
from drawer import Drawer
from utils import show_imgs, max_normalize_, to_color, Progress

from config import *

class AliceBobCharlie(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sender = Sender()
        self.receiver = Receiver()
        self.drawer = Drawer()

    def _bob_input(self, batch, drawer_outcome):
        return torch.cat([batch.target_img.unsqueeze(1), batch.base_distractors, drawer_outcome.image.unsqueeze(1)], dim=1)

    def _forward(self, batch, sender, drawer, receiver, sender_no_grad=False, drawer_no_grad=False):
        with torch.autograd.set_grad_enabled(torch.is_grad_enabled() and (not sender_no_grad)):
            sender_outcome = sender(batch.original_img)

        with torch.autograd.set_grad_enabled(torch.is_grad_enabled() and (not drawer_no_grad)):
            drawer_outcome = drawer(*sender_outcome.action)

        receiver_outcome = receiver(self._bob_input(batch, drawer_outcome), *sender_outcome.action)

        return sender_outcome, drawer_outcome, receiver_outcome

    def forward(self, batch):
        """
        Input:
            `batch` is a Batch (a kind of named tuple); 'original_img' and 'target_img' are tensors of shape [args.batch, *IMG_SHAPE] and 'base_distractors' is a tensor of shape [args.batch, 2, *IMG_SHAPE]
        Output:
            `sender_outcome`, sender.Outcome
            `receiver_outcome`, receiver.Outcome
        """
        return self._forward(batch, self.sender, self.drawer, self.receiver)

    def compute_rewards(self, sender_action, receiver_action, running_avg_success, chance_perf):
        """
            returns the reward as well as the success for each element of a batch
        """
        successes = (receiver_action == 0).float() # by design, the first image is the target

        rewards = successes

        if(args.penalty > 0.0):
            msg_lengths = sender_action[1].view(-1).float() # Float casting could be avoided if we upgrade torch to 1.3.1; see https://github.com/pytorch/pytorch/issues/9515 (I believe)
            length_penalties = 1.0 - (1.0 / (1.0 + args.penalty * msg_lengths)) # Equal to 0 when `args.penalty` is set to 0, increases to 1 with the length of the message otherwise

            # TODO J'ai peur que ce système soit un peu trop basique et qu'il encourage le système à être sous-performant - qu'on puisse obtenir plus de reward en faisant exprès de se tromper.
            if(args.adaptative_penalty):
                improvement_factor = (running_avg_success - chance_perf) / (1 - chance_perf) # Equals 0 when running average equals chance performance, reaches 1 when running average reaches 1
                length_penalties = (length_penalties * min(0.0, improvement_factor))

            rewards = (rewards - length_penalties)

        return (rewards, successes)

    def compute_log_prob(self, sender_log_prob, receiver_log_prob):
        """
        Input:
            `sender_log_prob`, log probs for sender policy
            `receiver_log_prob`, log prob for receiver policy
        Output:
            \bigg(\sum \limits_{l=1}^L \log p_{\pi^s}(m^l_t|m^{<l}_t, u) + \log p_{\pi^L}(u_{t'}|z, U) \bigg)
        """
        log_prob = sender_log_prob.sum(dim=1) + receiver_log_prob
        return log_prob

    def train_step_alice_bob(self, batch, optim, running_avg_success):
        optim.zero_grad()
        sender_outcome, drawer_outcome, receiver_outcome = self._forward(batch, self.sender, self.drawer, self.receiver, drawer_no_grad=True)

        bob_probs = F.softmax(receiver_outcome.scores, dim=-1)
        bob_dist = Categorical(bob_probs)
        bob_action = bob_dist.sample()
        bob_entropy = bob_dist.entropy()
        bob_log_prob = bob_dist.log_prob(bob_action)

        chance_perf = (1.0 / self._bob_input(batch, drawer_outcome).size(1))
        #chance_perf = (1 / (1 + batch.base_distractors.size(1) + 1)) # The chance performance is 1 over the number of images shown to Bob
        (rewards, successes) = self.compute_rewards(sender_outcome.action, bob_action, running_avg_success, chance_perf)
        # TODO On doit séparer Alice et Bob. Comme on en a discuté longement fut un temps, on a de bonnes raisons de ne pas vouloir faire entrer en compte l'image de Charlie pour la reward d'Alice. (je sais comment faire)
        log_prob = self.compute_log_prob(sender_outcome.log_prob, bob_log_prob)
        loss = -(rewards * log_prob)

        loss = loss.mean()
        # entropy penalties
        loss = loss - (BETA_SENDER * sender_outcome.entropy.mean())
        loss = loss - (BETA_RECEIVER * bob_entropy.mean())

        # backprop
        loss.backward()

        # Gradient clipping and scaling
        if((args.grad_clipping is not None) and (args.grad_clipping > 0)): torch.nn.utils.clip_grad_value_(self.parameters(), args.grad_clipping)
        if((args.grad_scaling is not None) and (args.grad_scaling > 0)): torch.nn.utils.clip_grad_norm_(self.parameters(), args.grad_scaling)

        optim.step()

        message_length = sender_outcome.action[1]

        charlie_acc = (bob_action == (1 + batch.base_distractors.size(1))).float().sum() / bob_action.numel()

        return rewards, successes, message_length, loss, charlie_acc

    def train_step_charlie(self, batch, optim, running_avg_success, default_adv_train=True):
        optim.zero_grad()
        sender_outcome, drawer_outcome, receiver_outcome = self._forward(batch, self.sender, self.drawer, self.receiver, sender_no_grad=True)

        bob_probs = F.softmax(receiver_outcome.scores, dim=-1)
        bob_dist = Categorical(bob_probs)
        bob_action = bob_dist.sample()
        bob_entropy = bob_dist.entropy()
        bob_log_prob = bob_dist.log_prob(bob_action)

        if default_adv_train:
            fake_image_score = receiver_outcome.scores[:,1 + batch.base_distractors.size(1)]
            target = torch.ones_like(fake_image_score)
            loss = F.binary_cross_entropy(torch.sigmoid(fake_image_score), target)
            # Or, more simply in our case: loss = torch.log(fake_image_score)
        else:
            target = torch.ones_like(bob_action) * (1 + batch.base_distractors.size(1))
            loss = F.nll_loss(F.log_softmax(bob_scores, dim=1), target)

        # backprop
        loss.backward()

        # Gradient clipping and scaling
        if((args.grad_clipping is not None) and (args.grad_clipping > 0)): torch.nn.utils.clip_grad_value_(self.parameters(), args.grad_clipping)
        if((args.grad_scaling is not None) and (args.grad_scaling > 0)): torch.nn.utils.clip_grad_norm_(self.parameters(), args.grad_scaling)

        optim.step()

        message_length = sender_outcome.action[1]

        charlie_acc = (bob_action == (1 + batch.base_distractors.size(1))).float().sum() / bob_action.numel()
        chance_perf = (1 / (batch.base_distractors.size(1) + 1)) # The chance performance is 1 over the number of images shown to Bob
        (rewards, successes) = self.compute_rewards(sender_outcome.action, bob_action, running_avg_success, chance_perf)

        return rewards, successes, message_length, loss, charlie_acc


    def train_epoch(self, data_iterator, optimizers, epoch=1, steps_per_epoch=1000, event_writer=None):
        """
            Model training function
            Input:
                `data_iterator`, an infinite iterator over (batched) data
                `optim_alice_bob`, the optimizer
            Optional arguments:
                `epoch`: epoch number to display in progressbar
                `steps_per_epoch`: number of steps for epoch
                `event_writer`: tensorboard writer to log evolution of values
        """
        optim_alice_bob, optim_charlie = optimizers
        self.train() # Sets the model in training mode

        with Progress(args.simple_display, steps_per_epoch, epoch) as pbar:
            total_reward = 0.0 # sum of the rewards since the beginning of the epoch
            total_success = 0.0 # sum of the successes since the beginning of the epoch
            total_items = 0 # number of training instances since the beginning of the epoch
            running_avg_reward = 0.0
            running_avg_success = 0.0
            start_i = ((epoch - 1) * steps_per_epoch) + 1 # (the first epoch is numbered 1, and the first iteration too)
            end_i = start_i + steps_per_epoch

            for i, batch in zip(range(start_i, end_i), data_iterator):
                generator_step = (i%2 == 0) # TODO Il faudrait un paramètre à la place du 2
                if generator_step:
                    rewards, successes, message_length, loss, charlie_acc = self.train_step_alice_bob(batch, optim_alice_bob, running_avg_success)
                else:
                    rewards, successes, message_length, loss, charlie_acc = self.train_step_charlie(batch, optim_charlie, running_avg_success)

                avg_reward = rewards.mean().item() # average reward of the batch
                avg_success = successes.mean().item() # average success of the batch
                avg_msg_length = message_length.float().mean().item() # average message length of the batch

                # updates running average reward
                total_reward += rewards.sum().item()
                total_success += successes.sum().item()
                total_items += batch.size
                running_avg_reward = total_reward / total_items
                running_avg_success = total_success / total_items

                pbar.update(R=running_avg_success)

                # logs some values
                if(event_writer is not None):
                    number_ex_seen = i * args.batch
                    event_writer.add_scalar('train/reward', avg_reward, number_ex_seen)
                    event_writer.add_scalar('train/success', avg_success, number_ex_seen)
                    if generator_step:
                        event_writer.add_scalar('train/loss_charlie', loss.item(), number_ex_seen)
                    else:
                        event_writer.add_scalar('train/loss_alice_bob', loss.item(), number_ex_seen)
                    event_writer.add_scalar('train/msg_length', avg_msg_length, number_ex_seen)
                    event_writer.add_scalar('train/charlie_acc', charlie_acc.item(), number_ex_seen)
                    if args.debug:
                        median_grad = torch.cat([p.grad.view(-1).detach() for p in self.parameters()]).abs().median().item()
                        mean_grad = torch.cat([p.grad.view(-1).detach() for p in self.parameters()]).abs().mean().item()
                        #min_grad = torch.cat([p.grad.view(-1).detach() for p in self.parameters()]).abs().min().item()
                        max_grad = torch.cat([p.grad.view(-1).detach() for p in self.parameters()]).abs().max().item()
                        mean_norm_grad = torch.stack([p.grad.view(-1).detach().data.norm(2.) for p in self.parameters()]).mean().item()
                        max_norm_grad = torch.stack([p.grad.view(-1).detach().data.norm(2.) for p in self.parameters()]).max().item()
                        event_writer.add_scalar('train/median_grad', median_grad, number_ex_seen)
                        event_writer.add_scalar('train/mean_grad', mean_grad, number_ex_seen)
                        #event_writer.add_scalar('train/min_grad', min_grad, number_ex_seen)
                        event_writer.add_scalar('train/max_grad', max_grad, number_ex_seen)
                        event_writer.add_scalar('train/mean_norm_grad', mean_norm_grad, number_ex_seen)
                        event_writer.add_scalar('train/max_norm_grad', max_norm_grad, number_ex_seen)

        self.eval()
