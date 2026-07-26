from abc import ABCMeta, abstractmethod
import time

import torch

class Game(metaclass=ABCMeta):
    @abstractmethod
    def test_visualize(self, data_iterator, learning_rate):
        """
        Make Bob dream again!
        """
        pass

    @property
    @abstractmethod
    def current_agents(self):
        """
        Lists the agents involved in the current round of the game.
        """
        pass

    @property
    @abstractmethod
    def all_agents(self):
        """
        Lists all agents.
        """
        pass

    def to(self, *vargs, **kwargs):
        for agent in self.all_agents: agent.to(*vargs, **kwargs)

        return self

    @property
    @abstractmethod
    def optims(self):
        """
        Lists all optimizers.
        """
        pass

    @abstractmethod
    def compute_interaction(self, batches, **kwargs):
        """
        Computes one round of the game.
        Input:
            batches as required, agents
        Output:
            rewards, successes, avg_signal_length, losses
        """
        # TODO: change return signature to loss, {dict of things to log}
        pass

    def train(self):
        """
        Sets the agents involved in the current round of the game to train mode.
        """
        for agent in self.current_agents:
            agent.train()

    def eval(self):
        """
        Sets the agents involved in the current round of the game to eval mode.
        """
        for agent in self.current_agents:
            agent.eval()

    def start_episode(self, train_episode=True):
        """
        Called before starting a new round of the game. Override for setup behavior.
        """
        if train_episode: self.train()
        else: self.eval()

    def start_epoch(self, data_iterator, summary_writer):
        """
        Called before starting a new epoch of the game. Override for setup/pretrain behavior.
        """
        self.train()

    @property
    @abstractmethod
    def autologger(self):
        pass

    # Trains the model for one epoch of `steps_per_epoch` steps (each step processes a batch)
    def train_epoch(self, data_iterator, epoch_index, steps_per_epoch=1000, event_writer=None):
        """
        Model training function
        Input:
            `data_iterator`, an infinite iterator over (batched) data
        Optional arguments:
            `epoch_index`: epoch number to display in progressbar
            `steps_per_epoch`: number of steps for epoch
            `event_writer`: tensorboard writer to log evolution of values
        """

        self.start_epoch(data_iterator, event_writer)
        with self.autologger:
            start_index = (epoch_index * steps_per_epoch)
            end_index = (start_index + steps_per_epoch)
            for iter_index in range(start_index, end_index):
                self.start_episode()

                batch = data_iterator.get_batch(data_type='train', keep_category=self.autologger.log_lang_progress) # If `self.autologger.log_lang_progress` is True, the autologger will need to access the categories of the images in the batch.

                optimization, *external_output = self.compute_interaction(batch, epoch_index=epoch_index, iter_index=iter_index)

                # RMK: It could be easier to have all of the optimization within `compute_interaction` (possibly renamed).
                for i, (_, _, backward_f) in enumerate(optimization):
                    retain_graph = (i != (len(optimization) - 1)) # True except for the last element.
                    backward_f(retain_graph) # Backpropagation

                for (optim, loss, _) in optimization:
                    # Gradient clipping and scaling
                    if(self.grad_clipping > 0):
                        for group in optim.param_groups: torch.nn.utils.clip_grad_value_(group["params"], self.grad_clipping)
                        #for agent in self.current_agents: torch.nn.utils.clip_grad_value_(agent.parameters(), self.grad_clipping)
                    if(self.grad_scaling > 0):
                        for group in optim.param_groups: torch.nn.utils.clip_grad_norm_(group["params"], self.grad_scaling)
                        #for agent in self.current_agents: torch.nn.utils.clip_grad_norm_(agent.parameters(), self.grad_scaling)

                    optim.step() # Parameters update. Should not be performed until all gradients have been computed.
                    optim.zero_grad() # Reinitialization of the gradient buffers.

                # TODO This needs an update.
                udpated_state = self.autologger.update(
                    torch.tensor(0.0), # This is where "the loss" should be, but I'm logging the losses directly in compute_interaction.
                    *external_output,
                    parameters=(p for a in self.all_agents for p in a.parameters()), # TODO `self.all_agents` or `self.current_agents`?
                    batch=batch,
                    index=iter_index,
                )

    def save(self, path):
        """
        Saves the model to the file `path`.
        """
        state = {
            'agents_state_dicts': [agent.state_dict() for agent in self.all_agents],
            'optims': self.optims,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path, args):
        instance = cls(args)

        checkpoint = torch.load(path, map_location=args.device)
        for agent, state_dict in zip(instance.all_agents, checkpoint['agents_state_dicts']):
            agent.load_state_dict(state_dict)

        instance._optim = checkpoint['optims'][0]

        return instance

    @abstractmethod
    def evaluate(self, data_iterator, epoch_index):
        """
        Evaluates agents. (called at the end of each training epoch)
        """
        pass

    def train_agents(self, epochs, steps_per_epoch, data_loader, run_models_dir=None, save_every=0, start_epoch_index=0):
        """
        Trains all agents over a given number of epochs.
        """
        # Reference evaluation before parameter updates.
        self.evaluate(data_loader, epoch_index=-1)

        for epoch_index in range(start_epoch_index, start_epoch_index + epochs):
            timepoint_0 = time.time()

            # Training phase.
            self.train_epoch(data_loader, epoch_index=epoch_index, steps_per_epoch=steps_per_epoch)

            timepoint_1 = time.time()
            print('Training took %f s.' % (timepoint_1 - timepoint_0))
            timepoint_0 = timepoint_1
            
            # Evaluation phase.
            self.evaluate(data_loader, epoch_index=epoch_index)

            timepoint_1 = time.time()
            print('Evaluating took %f s.' % (timepoint_1 - timepoint_0))
            timepoint_0 = timepoint_1
            
            # Saving.
            if((save_every > 0) and (((epoch_index + 1) % save_every) == 0)):
                model_name = f"model_e{epoch_index}.pt"
                self.save(run_models_dir / model_name)
