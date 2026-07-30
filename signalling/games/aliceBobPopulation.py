from .aliceBob import AliceBob
from .population import PopulationMixin
from ..agents import Sender, Receiver


# Population variant of AliceBob: `n` senders (Alice·s) and `m` receivers (Bob·s), set by
# `pop_size` ("n-m"), reinitialized on a staggered schedule set by `pop_reset_period` ("a-b").
# All the population machinery lives in PopulationMixin (including re-pretraining a reinitialized
# agent's CNN through the game's pretrainer); here we only supply the AliceBob-specific hooks.
class AliceBobPopulation(PopulationMixin, AliceBob):
    def __init__(self, args, logger, dataset, signal_dump_dir):
        # Builds all the standard AliceBob state (receiver preprocessor, pretrainer, a single
        # throwaway sender/receiver/optimizer, ...); _init_population then replaces the
        # agents/optimizer with the populations.
        super().__init__(args, logger, dataset, signal_dump_dir)

        # `--pop_size n-m` sets the population sizes; absent, it defaults to 2-2.
        self._init_population(args, roles=("sender", "receiver"), default_size=(2, 2), default_period=(0, 0))

    def _population_factories(self, args):
        return ((lambda: Sender.from_args(args)), (lambda: Receiver.from_args(args)))

    def _assign_current(self, producer, consumer):
        self._sender, self._receiver = producer, consumer

    # Overrides AliceBob.agents_for_pretraining: pretrain every agent in the population.
    def agents_for_pretraining(self):
        if(self.shared):
            raise NotImplementedError
        return ([(agent, "sender") for agent in self._producers]
                + [(agent, "receiver") for agent in self._consumers])
