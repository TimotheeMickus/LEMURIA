import random

import torch.nn as nn

from ..utils.misc import build_optimizer, resolve_lr, build_optimizer_two_groups


# Parses a "x-y" string into an (x, y) pair of ints; None yields `default`.
def parse_pop_pair(spec, default):
    if(spec is None):
        return default
    parts = str(spec).split("-")
    assert (len(parts) == 2), f"expected a pair in the form 'x-y', got {spec!r}"
    return (int(parts[0]), int(parts[1]))


# Command-line arguments that select/configure the population variant of a game.
# `producers`/`consumers` are the plural role names used in the help (e.g.
# "senders"/"receivers" or "askers"/"retrievers"). Returns the argparse group.
def add_population_args(parser, producers, consumers):
    group = parser.add_argument_group(title='Population', description='arguments relative to the population variant of the game')
    group.add_argument('--pop_size', help=("population sizes as 'n-m' (n %s, m %s). Passing this (or --pop_reset_period) selects the population game; the omitted one defaults to 2-2 / 0-0." % (producers, consumers)), default=None, type=str)
    group.add_argument('--pop_reset_period', help=("reset periods as 'a-b'; reinitialize %s every a epochs and %s every b epochs (0 = never); default 0-0" % (producers, consumers)), default=None, type=str)
    return group


# Shared logic for population games.
# There is a group of `n` "producers" (senders / askers) and a group of `m` "consumers" (receivers / retrievers). For each training round a random producer and a random consumer are selected and trained together. During evaluation the oldest producer and the oldest consumer are used.
# A single optimizer covers the whole population (but possibly with per agent parameters).
# Agents are reinitialized on a staggered schedule so that no producer lives more than `a` epochs and no consumer more than `b` epochs (0 = never); within a group the resets are spread over the period (one every ~period/count epochs).
#
# A concrete game mixes this in *before* its base game, e.g.
#     class AliceBobPopulation(PopulationMixin, AliceBob): ...
# calls `self._init_population(args, ...)` at the end of its __init__ (after super().__init__), and provides these hooks:
#   * _population_factories(args) -> (make_producer, make_consumer): zero-arg constructors.
#   * _assign_current(producer, consumer): store the pair under the names the base game reads (e.g. self._sender/self._receiver, or self._asker/self._retriever).
#   * _on_reinitialized(agent, role, epoch, data_iterator): post-reset hook. The default re-pretrains the reinitialized agent through the game's pretrainer (a no-op if there is none), so concrete games no longer need to override it.
#
# Equivalence (for either game):
#   * the plain 1-agent game  == population(pop_size=1-1, pop_reset_period=0-0)
class PopulationMixin:
    # Call from the subclass __init__ *after* super().__init__(...). Replaces the base game's single producer/consumer/optimizer with the populations.
    def _init_population(self, args, roles=("producer", "consumer"), default_size=(2, 2), default_period=(0, 0)):
        if(self.shared):
            raise NotImplementedError("Population games do not support shared parameters.")

        self._producer_role, self._consumer_role = roles

        n, m = parse_pop_pair(args.pop_size, default_size)
        a, b = parse_pop_pair(args.pop_reset_period, default_period)
        assert (n >= 1) and (m >= 1), "pop_size must be at least 1-1"
        assert (a >= 0) and (b >= 0), "pop_reset_period must be at least 0-0"
        self._producer_period, self._consumer_period = a, b

        make_producer, make_consumer = self._population_factories(args)
        producers = [make_producer() for _ in range(n)]
        consumers = [make_consumer() for _ in range(m)]
        self._producers = nn.ModuleList(producers)
        self._consumers = nn.ModuleList(consumers)
        self._agents = nn.ModuleList(producers + consumers)

        self._optim = build_optimizer_two_groups(
            self._producers.parameters(), resolve_lr(args.learning_rate, args.learning_rate_a),
            self._consumers.parameters(), resolve_lr(args.learning_rate, args.learning_rate_b)
        )

        # Staggered schedule: birth = -offset with offset = floor(i * period / count), so agent i starts at age `offset` and is first reset at epoch period - offset. Hence the group's first reset is at ~period/count and every agent's lifespan is at most `period`.
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

    def _population_factories(self, args):
        raise NotImplementedError

    def _assign_current(self, producer, consumer):
        raise NotImplementedError

    def _on_reinitialized(self, agent, role, epoch, data_iterator):
        # Default: re-pretrain the reinitialized agent if this game has a pretrainer (as AliceBob
        # does for the CNN, and AlexBeth for its encoders). `role` is the group role string
        # ("sender"/"receiver" or "asker"/"retriever").
        if(self.pretrainer is not None):
            self.pretrainer.pretrain_agent(agent, role, agent_name=f"reborn {role} @epoch {epoch}")

    @property
    def all_agents(self):
        return self._agents

    @property
    def current_agents(self):
        return (self._cur_producer, self._cur_consumer)

    # The oldest agent of a group = the one with the largest age = the smallest birth epoch (ties: lowest index).
    # Used to fix the evaluated pair.
    @staticmethod
    def _oldest(agents, births):
        i = min(range(len(agents)), key=(lambda j: births[j]))
        return agents[i]

    # Overrides Game.start_episode.
    # Selects the agents used for this round. Training: a random producer and a random consumer. Evaluation: the oldest producer and the oldest consumer.
    def start_episode(self, train_episode=True):
        if(train_episode):
            self._cur_producer = random.choice(self._producers)
            self._cur_consumer = random.choice(self._consumers)
        else:
            self._cur_producer = self._oldest(self._producers, self._producer_birth)
            self._cur_consumer = self._oldest(self._consumers, self._consumer_birth)
        self._assign_current(self._cur_producer, self._cur_consumer)

    # Overrides Game.start_epoch: apply the reinitialization schedule.
    def start_epoch(self, data_iterator, summary_writer):
        super().start_epoch(data_iterator, summary_writer)  # sets train mode

        self._reinit_due(self._producers, self._producer_birth, self._producer_period, self._producer_role, data_iterator)
        self._reinit_due(self._consumers, self._consumer_birth, self._consumer_period, self._consumer_role, data_iterator)

        self._current_epoch += 1

    # Reinitializes every agent of a group whose age has reached `period` (0 = never), clears the stale optimizer state for it, and runs the post-reset hook.
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
