from .alexBeth import AlexBeth
from .population import PopulationMixin
from ..agents import Asker, Retriever


# Population variant of AlexBeth: `n` askers (Alex·es) and `m` retrievers (Beth·s), set by
# `pop_size` ("n-m"), reinitialized on a staggered schedule set by `pop_reset_period` ("a-b").
# All the population machinery lives in PopulationMixin; here we only supply the AlexBeth-specific
# hooks. See PopulationMixin for the schedule and the equivalences (e.g. AlexBeth itself is
# AlexBethPopulation(pop_size=1-1, pop_reset_period=0-0)).
class AlexBethPopulation(PopulationMixin, AlexBeth):
    def __init__(self, args, logger, dataset, signal_dump_dir):
        # Builds all the standard AlexBeth state (a single throwaway asker/retriever/optimizer
        # included), which _init_population then replaces with the populations.
        super().__init__(args, logger, dataset, signal_dump_dir)
        self._init_population(args, roles=("asker", "retriever"), default_size=(2, 2), default_period=(0, 0))

    def _population_factories(self, args):
        return ((lambda: Asker.from_args(args)), (lambda: Retriever.from_args(args)))

    def _assign_current(self, producer, consumer):
        self._asker, self._retriever = producer, consumer
