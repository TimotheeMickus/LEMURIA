import torch
import torch.nn as nn
import random
import itertools
from datetime import datetime

from .game import Game
from .aliceBob import AliceBob
from ..agents import Sender, Receiver, SenderReceiver
from ..utils.misc import build_optimizer, get_default_fn
from ..utils import misc
from ..utils.modules import build_cnn_decoder_from_args

# In this game, there is a population of senders (Alice·s) and of receivers (Bob·s).
# For each training batch, a pair of (Alice, Bob) is randomly selected and trained to maximise the probability assigned by Bob to a "target image" in the following context: Alice is shown an "original image" and produces a message, Bob sees the message and then the target image and a "distractor image".
# If required, the agents are regularly reinitialized.
class AliceBobPopulation(AliceBob):
    def __init__(self, args, logger): # TODO We could consider calling super().__init__(args)
        self._logger = logger
        self.base_alphabet_size = args.base_alphabet_size
        self.max_len_msg = args.max_len

        self.use_expectation = args.use_expectation
        self.grad_scaling = args.grad_scaling or 0
        self.grad_clipping = args.grad_clipping or 0
        self.beta_sender = args.beta_sender
        self.beta_receiver = args.beta_receiver
        self.penalty = args.penalty

        size = args.population # There are `size` senders and `size` receivers.

        self.shared = args.shared
        if(self.shared):
            NotImplementedError
        else:
            senders = [Sender.from_args(args) for _ in range(size)]
            receivers = [Receiver.from_args(args) for _ in range(size)]
            agents = (senders + receivers)
        self.senders, self.receivers, self._agents = nn.ModuleList(senders), nn.ModuleList(receivers), nn.ModuleList(agents)

        self._sender, self._receiver = None, None # These properties will be set before each episode by `start_episode`.

        # Mathusalemian dynamics:
        # If `_reaper_step` is an integer, at the beginning of each `_reaper_step` training epoch, one agent is reinitialized.
        # The agents are reinitialized in turn.
        self._reaper_step = args.reaper_step
        if(self._reaper_step is not None):
            self._current_epoch = 0
            self._death_row = itertools.cycle(self._agents)
            self._pretrain_args = {
                "pretrain_CNN_mode": args.pretrain_CNNs,
                "freeze_pretrained_CNN": args.freeze_pretrained_CNNs,
                "learning_rate": args.pretrain_learning_rate or args.learning_rate,
                "nb_epochs": args.pretrain_epochs,
                "steps_per_epoch": args.steps_per_epoch,
                "display_mode": args.display,
                "pretrain_CNNs_on_eval": args.pretrain_CNNs_on_eval,
                "deconvolution_factory": get_default_fn(build_cnn_decoder_from_args, args),
            }

        self._optim = build_optimizer(self._agents.parameters(), args.learning_rate)

        self.use_baseline = args.use_baseline
        if(self.use_baseline): # In that case, the loss will take into account the "baseline term", into the average recent reward.
            # Currently, the sender and receiver's rewards are the same, but we could imagine a setting in which they are different.
            self._sender_avg_reward = misc.Averager(size=12800)
            self._receiver_avg_reward = misc.Averager(size=12800)

        self.correct_only = args.correct_only # Whether to perform the fancy language evaluation using only correct messages (leading to successful communication)

    @property
    def sender(self):
        return self._sender

    @property
    def receiver(self):
        return self._receiver

    @property
    def all_agents(self):
        return self._agents

    @property
    def current_agents(self):
        return (self._sender, self._receiver)

    # Overrides AliceBob.agents_for_CNN_pretraining.
    def agents_for_CNN_pretraining(self):
        if(self.shared): raise NotImplementedError
        return self.all_agents

    # Overrides Game.start_episode.
    def start_episode(self, train_episode=True):
        self._sender = random.choice(self.senders)
        self._receiver = random.choice(self.receivers)

        super().start_episode(train_episode=train_episode)

    # Overrides Game.start_epoch.
    def start_epoch(self, data_iterator, summary_writer):
        if(self._reaper_step is not None):
            if((self._current_epoch != 0) and (self._current_epoch % self._reaper_step == 0)):
                reborn_agent = next(self._death_row)
                reborn_agent.reinitialize()

                if(self._pretrain_args['pretrain_CNN_mode'] is not None):
                    agent_name = 'reborn agent %i' % (self._current_epoch // self._reaper_step)
                    self.pretrain_agent_CNN(reborn_agent, data_iterator, **self._pretrain_args, agent_name=agent_name)
                    print(f"[{datetime.now()}] {agent_name} reinitialized.")
            
            self._current_epoch += 1
