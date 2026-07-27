import random

import torch.nn as nn

from ..utils.misc import build_optimizer


# Parses a "x-y" string into an (x, y) pair of ints; None yields `default`.
def parse_pop_pair(spec, default):
    if(spec is None):
        return default
    parts = str(spec).split("-")
    assert (len(parts) == 2), f"expected a pair in the form 'x-y', got {spec!r}"
    return (int(parts[0]), int(parts[1]))


# Shared logic for population games. There is a group of `n` "producers" (senders / askers) and
# a group of `m` "consumers" (receivers / retrievers). For each round a random producer and a
# random consumer are selected and trained together; a single optimizer covers the whole
# population, so a step only updates the selected pair (Adam skips parameters whose `.grad` is
# None). Agents are reinitialized on a staggered schedule so that no producer lives more than
# `a` epochs and no consumer more than `b` epochs (0 = never); within a group the resets are
# spread over the period (one every ~period/count epochs) rather than happening together.
# During evaluation the language is produced by the *oldest* producer.
#
# A concrete game mixes this in *before* its base game, e.g.
#     class AliceBobPopulation(PopulationMixin, AliceBob): ...
# calls `self._init_population(args, ...)` at the end of its __init__ (after super().__init__),
# and provides these hooks:
#   * _population_factories(args) -> (make_producer, make_consumer): zero-arg constructors.
#   * _assign_current(producer, consumer): store the pair under the names the base game reads
#     (e.g. self._sender/self._receiver, or self._asker/self._retriever).
#   * _on_reinitialized(agent, role, epoch, data_iterator): optional post-reset hook (default
#     no-op; the image game overrides it to re-pretrain the reinitialized agent's CNN).
#
# Equivalences (for either game):
#   * the plain 1-agent game  == population(pop_size=1-1, pop_reset_period=0-0)
#   * an "every-B-epochs reaper" on the consumer == population(pop_size=1-1, pop_reset_period=0-B)
class PopulationMixin:
    # Call from the subclass __init__ *after* super().__init__(...). Replaces the base game's
    # single producer/consumer/optimizer with the populations.
    def _init_population(self, args, roles=("producer", "consumer"), default_size=(2, 2), default_period=(0, 0)):
        if(getattr(self, "shared", False)):
            raise NotImplementedError("Population games do not support shared parameters.")

        self._producer_role, self._consumer_role = roles

        n, m = parse_pop_pair(getattr(args, "pop_size", None), default_size)
        a, b = parse_pop_pair(getattr(args, "pop_reset_period", None), default_period)
        assert (n >= 1) and (m >= 1), "pop_size must be at least 1-1"
        assert (a >= 0) and (b >= 0), "pop_reset_period must be at least 0-0"
        self._producer_period, self._consumer_period = a, b

        make_producer, make_consumer = self._population_factories(args)
        producers = [make_producer() for _ in range(n)]
        consumers = [make_consumer() for _ in range(m)]
        self._producers = nn.ModuleList(producers)
        self._consumers = nn.ModuleList(consumers)
        self._agents = nn.ModuleList(producers + consumers)

        # A single optimizer over the whole population.
        self._optim = build_optimizer(self._agents.parameters(), args.learning_rate)

        # Staggered schedule: birth = -offset with offset = floor(i * period / count), so agent i
        # starts at age `offset` and is first reset at epoch period - offset. Hence the group's
        # first reset is at ~period/count and every agent's lifespan is at most `period`.
        self._producer_birth = self._initial_births(n, a)
        self._consumer_birth = self._initial_births(m, b)
        self._current_epoch = 0

        # A concrete pair must always be set; start_episode() overwrites this before every round.
        self._cur_producer, self._cur_consumer = producers[0], consumers[0]
        self._assign_current(self._cur_producer, self._cur_consumer)

    # birth = -offset, so that age(epoch) = epoch - birth = epoch + offset.
    @staticmethod
    def _initial_births(count, period):
        if(period <= 0):
            return [0] * count
        return [-((i * period) // count) for i in range(count)]

    # --- hooks a concrete game must / may provide -------------------------------------------
    def _population_factories(self, args):
        raise NotImplementedError

    def _assign_current(self, producer, consumer):
        raise NotImplementedError

    def _on_reinitialized(self, agent, role, epoch, data_iterator):
        pass  # default: nothing to do after a reinitialization
    # ----------------------------------------------------------------------------------------

    @property
    def all_agents(self):
        return self._agents

    @property
    def current_agents(self):
        return (self._cur_producer, self._cur_consumer)

    # The oldest producer = the one with the largest age = the smallest birth epoch (ties: lowest
    # index). Used to fix the language during evaluation.
    def _oldest_producer(self):
        i = min(range(len(self._producers)), key=(lambda j: self._producer_birth[j]))
        return self._producers[i]

    # Overrides Game.start_episode: select the agents used for this round. Training: a random
    # producer and a random consumer. Evaluation: the oldest producer (stable, well-trained
    # language) and a random consumer.
    def start_episode(self, train_episode=True):
        self._cur_producer = random.choice(self._producers) if train_episode else self._oldest_producer()
        self._cur_consumer = random.choice(self._consumers)
        self._assign_current(self._cur_producer, self._cur_consumer)

        super().start_episode(train_episode=train_episode)  # sets train/eval mode on current_agents

    # Overrides Game.start_epoch: apply the reinitialization schedule.
    def start_epoch(self, data_iterator, summary_writer):
        super().start_epoch(data_iterator, summary_writer)  # sets train mode

        self._reinit_due(self._producers, self._producer_birth, self._producer_period, self._producer_role, data_iterator)
        self._reinit_due(self._consumers, self._consumer_birth, self._consumer_period, self._consumer_role, data_iterator)

        self._current_epoch += 1

    # Reinitializes every agent of a group whose age has reached `period` (0 = never), clears the
    # stale optimizer state for it, and runs the post-reset hook.
    def _reinit_due(self, agents, births, period, role, data_iterator):
        if(period <= 0):
            return
        epoch = self._current_epoch
        for i, agent in enumerate(agents):
            if((epoch - births[i]) >= period):
                agent.reinitialize()
                for p in agent.parameters():
                    self._optim.state.pop(p, None)
                births[i] = epoch
                print(f"[pop-reset] epoch {epoch}: {role} {i} reinitialized.")
                self._on_reinitialized(agent, role, epoch, data_iterator)
