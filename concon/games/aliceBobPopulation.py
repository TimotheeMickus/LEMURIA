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

class AliceBobPopulation(AliceBob):
    def __init__(self, args): # TODO We could consider calling super().__init__(args)
        self.base_alphabet_size = args.base_alphabet_size
        self.max_len_msg = args.max_len

        size = args.population

        # In both cases, there are `size` senders and `size` receivers, but if `shared` is True, senders are paired with receivers so as to share their CNN and symbol embeddings
        if(args.shared):
            self._agents = [SenderReceiver.from_args(args) for _ in range(size)]

            self.senders, self.receivers = zip(*[(agent.sender, agent.receiver) for agent in self._agents])
        else:
            self.senders = [Sender.from_args(args) for _ in range(size)]
            self.receivers = [Receiver.from_args(args) for _ in range(size)]

            self._agents = (self.senders + self.receivers)

        self.use_expectation = args.use_expectation
        self.grad_scaling = args.grad_scaling or 0
        self.grad_clipping = args.grad_clipping or 0
        self.beta_sender = args.beta_sender
        self.beta_receiver = args.beta_receiver
        self.penalty = args.penalty
        self.adaptative_penalty = args.adaptative_penalty

        self._sender, self._receiver = None, None # Set before each episode by `start_episode`

        # Mathusalemian dynamics
        self._reaper_step = args.reaper_step
        if self._reaper_step is not None:
            self._current_epoch =  0
            self._death_row = itertools.cycle(self._agents)
            self._pretrain_args = {
                "pretrain_CNN_mode":args.pretrain_CNNs,
                "freeze_pretrained_CNN":args.freeze_pretrained_CNNs,
                "learning_rate":args.pretrain_learning_rate or args.learning_rate,
                "nb_epochs":args.pretrain_epochs,
                "steps_per_epoch":args.steps_per_epoch,
                "display_mode":args.display,
                "pretrain_CNNs_on_eval":args.pretrain_CNNs_on_eval,
                "deconvolution_factory":get_default_fn(build_cnn_decoder_from_args, args),
            }
            self._pretrain_shared = args.shared
        else:
            self._pretrain_args = {"pretrain_CNN_mode":args.pretrain_CNNs,}

        self.start_episode() # TODO Really useful?

        parameters = [p for a in self._agents for p in a.parameters()]
        self._optim = build_optimizer(nn.ParameterList(parameters), args.learning_rate)

        # Currently, the sender and receiver's rewards are the same, but we could imagine a setting in which they are different
        self.use_baseline = args.use_baseline
        if(self.use_baseline):
            self._sender_avg_reward = misc.Averager(size=12800)
            self._receiver_avg_reward = misc.Averager(size=12800)
        
        self.correct_only = args.correct_only # Whether to perform the fancy language evaluation using only correct messages (leading to successful communication)

    def to(self, *vargs, **kwargs):
        #self = super().to(*vargs, **kwargs)

        #for agent in self._agents: agent.to(*args, **kwargs) # Would that be enough? I'm not sure how `.to` works

        self.senders = [sender.to(*vargs, **kwargs) for sender in self.senders]
        self.receivers = [receiver.to(*vargs, **kwargs) for receiver in self.receivers]

        return self

    def get_sender(self):
        return self._sender

    def get_receiver(self):
        return self._receiver

    def start_episode(self, train_episode=True):
        self._sender = random.choice(self.senders)
        self._receiver = random.choice(self.receivers)

        super().start_episode(train_episode=train_episode)

    def start_epoch(self, data_iterator, summary_writer):
        if self._reaper_step is not None:
            if (self._current_epoch != 0) and (self._current_epoch % self._reaper_step == 0):
                reborn_agent = next(self._death_row)
                self.kill(reborn_agent)

                if self._pretrain_args['pretrain_CNN_mode'] is not None:
                    if self._pretrain_shared:
                        reborn_agent = reborn_agent.sender
                    if self._pretrain_args['freeze_pretrained_CNN']:
                        for p in reborn_agent.image_encoder.parameters():
                            p.requires_grad = True
                    agent_name = 'reborn agent %i' % (self._current_epoch // self._reaper_step)
                    self.pretrain_agent_CNN(reborn_agent, data_iterator, summary_writer, **self._pretrain_args, agent_name=agent_name)
                    print("[%s] %s reinitialized." %(datetime.now(), agent_name))
            self._current_epoch += 1

    # TODO À quoi sert cette méthode ?
    @property
    def agents(self):
        return self._sender, self._receiver

    @property
    def optims(self):
        return (self.optim,)

    def save(self, path):
        state = {
            'agents_state_dicts':[agent.state_dict() for agent in self._agents],
            'optims':[optim for optim in self.optims],
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path, args, _old_model=False):
        checkpoint = torch.load(path, map_location=args.device)
        instance = cls(args)
        for agent, state_dict in zip(instance._agents, checkpoint['agents_state_dicts']):
            agent.load_state_dict(state_dict)
        instance._optim = checkpoint['optims'][0]
        return instance

    def pretrain_CNNs(self, data_iterator, summary_writer, pretrain_CNN_mode='category-wise', freeze_pretrained_CNN=False, learning_rate=0.0001, nb_epochs=5, steps_per_epoch=1000, display_mode='', pretrain_CNNs_on_eval=False, deconvolution_factory=None, shared=False):
        agents = self._agents if not shared else [a.sender for a in self._agents]
        trained_models = {}
        for i, agent in enumerate(agents):
            agent_name = ("agent %i" % i)
            trained_models[agent_name] = self.pretrain_agent_CNN(agent, data_iterator, summary_writer, pretrain_CNN_mode, freeze_pretrained_CNN, learning_rate, nb_epochs, steps_per_epoch, display_mode, pretrain_CNNs_on_eval, deconvolution_factory, agent_name=agent_name)
        return trained_models
