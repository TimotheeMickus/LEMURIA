from abc import ABCMeta, abstractmethod
import itertools as it
import more_itertools as m_it
import torch
import torch.nn as nn

from ..utils.misc import Progress

class Game(metaclass=ABCMeta):

    @abstractmethod
    def test_visualize(self, data_iterator, learning_rate):
        """
        Make Bob dream again!
        """
        pass

    @property
    @abstractmethod
    def optims(self):
        """
        List agents involved in the current round of the game
        """
        pass

    @property
    @abstractmethod
    def agents(self):
        """
        List optimizers involved in the current round of the game
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        Perform evaluation at the end of each epoch
        """
        pass

    @abstractmethod
    def compute_interaction(self, batches, *agents, **state_info):
        """
        Computes one round of the game.
        Input:
            batches as required, agents
        Output:
            rewards, successes, avg_msg_length, losses
        """
        pass

    @property
    @abstractmethod
    def num_batches_per_episode(self):
        """
        Number of batches required to complete one round of the game.
        """
        pass

    def start_episode(self):
        """
        Called before starting a new round of the game. Override for setup behavior.
        """
        for agent in self.agents:  # Sets the current agents in training mode
            agent.train()


    def end_episode(self):
        """
        Called after finishing a round of the game. Override for cleanup behavior.
        """
        for agent in self.agents:
            agent.eval()

    # Trains the model for one epoch of `steps_per_epoch` steps (each step processes a batch)
    def train_epoch(self, data_iterator, epoch=1, steps_per_epoch=1000, event_writer=None, simple_display=False, debug=False, log_lang_progress=True, log_entropy=False, base_alphabet_size=None):
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

        with Progress(simple_display, steps_per_epoch, epoch) as pbar:

            total_reward = 0.0 # sum of the rewards since the beginning of the epoch
            total_success = 0.0 # sum of the successes since the beginning of the epoch
            total_items = 0 # number of training instances since the beginning of the epoch
            running_avg_reward = 0.0
            running_avg_success = 0.0
            start_i = ((epoch - 1) * steps_per_epoch) + 1 # (the first epoch is numbered 1, and the first iteration too)
            end_i = start_i + steps_per_epoch

            device = next(self.agents[0].parameters()).device

            if event_writer is not None:
                if log_lang_progress:
                    past_dist, current_dist = None, torch.zeros((base_alphabet_size, 5), dtype=torch.float).to(device) # size of embeddings
                if log_entropy:
                    symbol_counts = torch.zeros(base_alphabet_size, dtype=torch.float).to(device)

            raw_batches = range(start_i, end_i)

            for indices in m_it.chunked(raw_batches, self.num_batches_per_episode):
                indices = list(indices)
                batches = [data_iterator.get_batch(keep_category=log_lang_progress) for _ in indices]
                self.start_episode()

                for optim in self.optims:
                    optim.zero_grad()

                rewards, successes, avg_msg_length, losses = self.compute_interaction(batches, *self.agents, running_avg_success=running_avg_success)

                for loss in losses:
                    loss.backward() # Backpropagation

                # Gradient clipping and scaling
                if self.grad_clipping > 0:
                    for agent in self.agents:
                        torch.nn.utils.clip_grad_value_(agent.parameters(), self.grad_clipping)
                if self.grad_scaling > 0:
                    for agent in self.agents:
                        torch.nn.utils.clip_grad_norm_(agent.parameters(), self.grad_scaling)

                for optim in self.optims:
                    optim.step()

                avg_reward = rewards.mean().item() # average reward of the batch
                avg_success = successes.mean().item() # average success of the batch

                # updates running average reward
                total_reward += rewards.sum().item()
                total_success += successes.sum().item()
                total_items += sum(batch.size for batch in batches)
                running_avg_reward = total_reward / total_items
                running_avg_success = total_success / total_items

                if log_lang_progress:
                    for batch in copy_batches:
                        batch_msg_manyhot = torch.zeros((batch.size, base_alphabet_size + 2), dtype=torch.float).to(device) # size of embeddings + EOS + PAD
                        # message -> many-hot
                        many_hots = batch_msg_manyhot.scatter_(1, sender_outcome.action[0].detach(), 1).narrow(1,1,base_alphabet_size).float()
                        # summation along batch dimension,  and add to counts
                        current_dist += torch.einsum('bi,bj->ij', many_hots, batch.original_category.float().to(device)).detach().float()

                pbar.update(self.num_batches_per_episode, S=running_avg_success)

                # logs some values
                if(event_writer is not None):
                    number_ex_seen = indices[-1] * batches[0].size
                    event_writer.add_scalar('train/reward', avg_reward, number_ex_seen)
                    event_writer.add_scalar('train/success', avg_success, number_ex_seen)
                    event_writer.add_scalar('train/loss', loss.item(), number_ex_seen)
                    event_writer.add_scalar('llp/msg_length', avg_msg_length, number_ex_seen)
                    if debug:
                        log_grads_tensorboard([p for a in self.agents for p in a.parameters()], event_writer)
                    if log_entropy:
                        new_messages = sender_outcome.action[0].view(-1)
                        valid_indices = torch.arange(base_alphabet_size).expand(new_messages.size(0), base_alphabet_size).to(device)
                        selected_symbols = valid_indices == new_messages.unsqueeze(1).float()
                        symbol_counts += selected_symbols.sum(dim=0)

                    if log_lang_progress and any(lambda i:i % 100 == 0, indices):
                        if past_dist is None:
                            past_dist, current_dist = current_dist, torch.zeros((base_alphabet_size, 5), dtype=torch.float).to(device)
                            continue
                        else:
                            logit_c = (current_dist.view(1, -1) / current_dist.sum()).log()
                            prev_p = (past_dist.view(1, -1) / past_dist.sum())
                            kl = F.kl_div(logit_c, prev_p, reduction='batchmean').item()
                            event_writer.writer.add_scalar('llp/kl_div', kl, number_ex_seen)
                            past_dist, current_dist = current_dist, torch.zeros((base_alphabet_size, 5), dtype=torch.float).to(device)

                self.end_episode()

        if log_entropy and (event_writer is not None):
            event_writer.writer.add_scalar('llp/entropy', compute_entropy(symbol_counts), number_ex_seen)
