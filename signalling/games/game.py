from abc import ABCMeta, abstractmethod
import argparse
import time

import torch

class Game(metaclass=ABCMeta):
    # Either None (no pretraining) or a Pretrainer instance (set by concrete games that support it).
    # Kept as a class-level default so that `run_pretraining` and PopulationMixin can reference it
    # uniformly whether or not a given game ever sets one.
    pretrainer = None

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

                # The losses carried by `optimization` are already detached (see
                # `compute_interaction`), so summing them here just aggregates
                # values for logging and does not touch the graph. This is what
                # gets logged as `train/loss`.
                total_loss = torch.tensor(0.0)
                for (_, loss, _) in optimization:
                    total_loss = total_loss + loss.detach().to(total_loss.device)

                udpated_state = self.autologger.update(
                    total_loss,
                    *external_output,
                    parameters=(p for a in self.all_agents for p in a.parameters()), # TODO `self.all_agents` or `self.current_agents`?
                    batch=batch,
                    index=iter_index,
                )

    # Builds the dataset a `load` needs to reconstruct an identically-shaped model. A checkpoint
    # stores only parameter/optimizer *state*, so `load` must first rebuild the model (whose layer
    # shapes depend on the dataset) before copying that state in. Concrete game families override
    # this to call their own data loader (image vs predicate). Only needed when `load` is not given
    # an explicit `dataset`.
    @classmethod
    def _data_loader_from_args(cls, args):
        raise NotImplementedError(
            f"{cls.__name__}.load needs a dataset to rebuild the model: either pass dataset=... "
            f"or implement {cls.__name__}._data_loader_from_args(args)."
        )

    def save(self, path):
        """
        Saves the model (agents and optimizers) to the file `path`, together with the training
        hyperparameters (`hparams`) needed to rebuild an identically-shaped model in `load`.
        """
        state = {
            # The exact args the model was constructed from (post any injection of derived fields,
            # e.g. num_predicates), stored as a plain dict so `load` can reconstruct without a
            # separate hparams file. Every game records its args in its constructor (self._args).
            'hparams': dict(vars(self._args)),
            'agents_state_dicts': [agent.state_dict() for agent in self.all_agents],
            'optims_state_dicts': [optim.state_dict() for optim in self.optims],
        }
        torch.save(state, path)

    # Reads just the embedded training hyperparameters from a checkpoint. Handy for deciding
    # *which* game class to load (e.g. whether it is a population model) before reconstructing
    # anything. Returns None only for checkpoints that predate hparams embedding.
    @staticmethod
    def peek_hparams(path, map_location="cpu"):
        # weights_only=False: the checkpoint embeds `hparams` (a dict with non-tensor objects such
        # as pathlib.Path / torch.device), which the PyTorch>=2.6 default (weights_only=True) would
        # refuse to unpickle. These are the user's own training checkpoints, so this is safe here.
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        return checkpoint.get('hparams', None)

    @classmethod
    def load(cls, path, args=None, logger=None, dataset=None, signal_dump_dir=None):
        """
        Rebuilds the game and restores agents and optimizers from `path`.

        The hyperparameters are taken from the checkpoint itself (embedded by `save`); a caller
        that only wants to override a field or two (typically `device`) may pass an `args` whose
        set fields take precedence.

        `dataset` is rebuilt from those hyperparameters via `_data_loader_from_args` unless one is
        supplied. `logger`/`signal_dump_dir` default to None (evaluation does not need them).

        Symmetric with `save`: both iterate over `all_agents`/`optims`, so this works for any game
        regardless of how many agents/optimizers it exposes. The agent/optimizer counts are checked
        against the checkpoint so a shape mismatch (e.g. loading a population checkpoint into the
        single-agent class, or with a different pop_size) fails loudly instead of silently loading
        a truncated subset.
        """
        embedded = Game.peek_hparams(path, map_location="cpu")
        if(embedded is None):
            raise RuntimeError(
                f"Cannot load {cls.__name__} from '{path}': the checkpoint has no embedded "
                "hyperparameters. Re-save it with the current code (save() now records them)."
            )

        load_args = argparse.Namespace(**embedded)
        # Let the caller override individual fields (device being the common one).
        if(args is not None):
            for key, value in vars(args).items():
                if(value is not None): setattr(load_args, key, value)

        device = load_args.device

        if(dataset is None):
            dataset = cls._data_loader_from_args(load_args)

        instance = cls(load_args, logger, dataset, signal_dump_dir)

        # weights_only=False: see peek_hparams (the checkpoint holds non-tensor `hparams`).
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        n_agents, n_saved = len(instance.all_agents), len(checkpoint['agents_state_dicts'])
        if(n_agents != n_saved):
            raise RuntimeError(
                f"Cannot restore {cls.__name__}: the rebuilt model has {n_agents} agents but the "
                f"checkpoint holds {n_saved}. This usually means the wrong game class was chosen "
                f"(e.g. a population checkpoint loaded as the single-agent game) or a mismatched "
                f"pop_size. Check --pop_size / --pop_reset_period or the loaded class."
            )
        n_optims, n_saved_optims = len(instance.optims), len(checkpoint['optims_state_dicts'])
        if(n_optims != n_saved_optims):
            raise RuntimeError(
                f"Cannot restore {cls.__name__}: the rebuilt model has {n_optims} optimizer(s) but "
                f"the checkpoint holds {n_saved_optims}."
            )

        for agent, state_dict in zip(instance.all_agents, checkpoint['agents_state_dicts']):
            agent.load_state_dict(state_dict)
        for optim, state_dict in zip(instance.optims, checkpoint['optims_state_dicts']):
            optim.load_state_dict(state_dict)

        return instance

    @abstractmethod
    def evaluate(self, data_iterator, epoch_index):
        """
        Evaluates agents. (called at the end of each training epoch)
        """
        pass

    def run_pretraining(self):
        """
        Runs pretraining if this game has a pretrainer configured (a no-op otherwise). Returns
        whatever the pretrainer's `pretrain()` returns (e.g. the trained CNN classifiers, used by
        --detect_outliers), or None when there is no pretrainer. Called by the training drivers.
        """
        if(self.pretrainer is None):
            return None
        return self.pretrainer.pretrain()

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
