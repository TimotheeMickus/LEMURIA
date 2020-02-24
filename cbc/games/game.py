from abc import ABCMeta, abstractmethod
import itertools as it
import more_itertools as m_it
import torch
import torch.nn as nn
import collections

from ..utils.logging import NotALogger

class Game(metaclass=ABCMeta):

    @abstractmethod
    def test_visualize(self, data_iterator, learning_rate):
        """
        Make Bob dream again!
        """
        pass

    @property
    @abstractmethod
    def agents(self):
        """
        List agents involved in the current round of the game
        """
        pass

    @property
    @abstractmethod
    def optim(self):
        """
        Optimizer involved in the current round of the game
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

    def train(self):
        for agent in self.agents:  # Sets the agents in training mode
            agent.train()

    def eval(self):
        for agent in self.agents:  # Sets the agents in evaluation mode
            agent.eval()

    def start_episode(self):
        """
        Called before starting a new round of the game. Override for setup behavior.
        """
        self.train() # Sets the current agents in training mode


    def end_episode(self):
        """
        Called after finishing a round of the game. Override for cleanup behavior.
        """
        self.eval()

    # Trains the model for one epoch of `steps_per_epoch` steps (each step processes a batch)
    def train_epoch(self, data_iterator, epoch=1, steps_per_epoch=1000, autologger=NotALogger()):
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
        with autologger:

            start_i = (epoch * steps_per_epoch)
            end_i = (start_i + steps_per_epoch)
            running_avg_success = 0.
            for indices in range(start_i, end_i):
                batch = data_iterator.get_batch(keep_category=autologger.log_lang_progress)
                self.start_episode()

                self.optim.zero_grad()

                rewards, successes, avg_msg_length, loss = self.compute_interaction(batch, *self.agents, running_avg_success=running_avg_success)

                loss.backward() # Backpropagation

                # Gradient clipping and scaling
                if self.grad_clipping > 0:
                    for agent in self.agents:
                        torch.nn.utils.clip_grad_value_(agent.parameters(), self.grad_clipping)
                if self.grad_scaling > 0:
                    for agent in self.agents:
                        torch.nn.utils.clip_grad_norm_(agent.parameters(), self.grad_scaling)

                self.optim.step()

                running_avg_success = autologger.update(
                    rewards, successes, avg_msg_length, loss,
                    parameters=(p for a in self.agents for p in a.parameters()),
                    num_batches_per_episode=self.num_batches_per_episode,
                    batches=[batch],
                    indices=[indices],
                )

                self.end_episode()

    def save(self, path):
        state = {
            'agents_state_dicts':[ agent.state_dict() for agent in self.agents],
            'optims':[optim for optim in self.optims],
        }
        torch.save(state, path)

    @classmethod
    @abstractmethod
    def load(cls, path, args, _old_model=False):
        pass
