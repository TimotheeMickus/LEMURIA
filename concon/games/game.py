from abc import ABCMeta, abstractmethod
import itertools as it
import more_itertools as m_it
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

from ..utils.logging import DummyLogger, Progress
from ..utils.misc import build_optimizer

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
    def compute_interaction(self, batches, **state_info):
        """
        Computes one round of the game.
        Input:
            batches as required, agents
        Output:
            rewards, successes, avg_msg_length, losses
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


    def end_episode(self, **kwargs):
        """
        Called after finishing a round of the game. Override for cleanup behavior.
        """
        self.eval()

    # Trains the model for one epoch of `steps_per_epoch` steps (each step processes a batch)
    def train_epoch(self, data_iterator, epoch=1, steps_per_epoch=1000, autologger=DummyLogger()):
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
            for index in range(start_i, end_i):
                batch = data_iterator.get_batch(keep_category=autologger.log_lang_progress)
                self.start_episode()

                self.optim.zero_grad()

                loss, *external_output  = self.compute_interaction(batch)

                loss.backward() # Backpropagation

                # Gradient clipping and scaling
                if self.grad_clipping > 0:
                    for agent in self.agents:
                        torch.nn.utils.clip_grad_value_(agent.parameters(), self.grad_clipping)
                if self.grad_scaling > 0:
                    for agent in self.agents:
                        torch.nn.utils.clip_grad_norm_(agent.parameters(), self.grad_scaling)

                self.optim.step()

                udpated_state = autologger.update(
                    loss, *external_output,
                    parameters=(p for a in self.agents for p in a.parameters()),
                    batch=batch,
                    index=index,
                )
                self.end_episode(**udpated_state)

    def save(self, path):
        """
        Save model to file `path`
        """
        state = {
            'agents_state_dicts': [agent.state_dict() for agent in self.agents],
            'optims': [self.optim],
        }
        torch.save(state, path)

    @classmethod
    @abstractmethod
    def load(cls, path, args, _old_model=False):
        pass

    def pretrain_CNNs(self, data_iterator, args):
        for i,agents in enumerate(self.agents):
            self.pretrain_agent_CNN(agent, data_iterator, args, agent_name="agent %i" %i)

    def pretrain_agent_CNN(self, agent, data_iterator, args, agent_name="agent"):
        num_cats = (2 if args.binary_dataset else 3) ** 5
        model = nn.Sequential(
            agent.image_encoder,
            nn.Linear(args.hidden_size, num_cats),
            nn.LogSoftmax(dim=1)).to(args.device)
        optimizer = build_optimizer(model.parameters(), args.learning_rate)

        print("Training agent: %s" % agent_name)
        for epoch in range(args.pretrain_epochs):
            pbar = Progress(args.display, args.steps_per_epoch, epoch, logged_items={'L', 'acc'})
            avg_acc, total_items = 0., 0.
            with pbar:
                for _ in range(args.steps_per_epoch):
                    self.optim.zero_grad()
                    batch = data_iterator.get_batch(keep_category=True, no_evaluation=not args.pretrain_CNNs_on_eval)
                    pred = model(batch.target_img(stack=True))
                    tgt = batch.category(stack=True, f=data_iterator.category_idx).to(args.device)
                    loss = F.nll_loss(pred, tgt)
                    avg_acc += (pred.argmax(dim=1) == tgt).float().sum().item()
                    total_items += tgt.size(0)
                    pbar.update(L=loss.item(), acc=avg_acc / total_items)
                    loss.backward()
                    optimizer.step()
