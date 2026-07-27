from .aliceBob import AliceBob
from .population import PopulationMixin
from ..agents import Sender, Receiver
from ..utils.misc import get_default_fn
from ..utils.modules import build_cnn_decoder_from_args


# Population variant of AliceBob: `n` senders (Alice·s) and `m` receivers (Bob·s), set by
# `pop_size` ("n-m"), reinitialized on a staggered schedule set by `pop_reset_period` ("a-b").
# For backward compatibility, `--population N` (when `--pop_size` is not given) means N-N.
# All the population machinery lives in PopulationMixin; here we only supply the AliceBob-specific
# hooks, including re-pretraining a reinitialized agent's CNN.
class AliceBobPopulation(PopulationMixin, AliceBob):
    def __init__(self, args, logger, dataset, signal_dump_dir):
        # Builds all the standard AliceBob state (receiver preprocessor, a single throwaway
        # sender/receiver/optimizer, ...); _init_population then replaces the agents/optimizer.
        super().__init__(args, logger, dataset, signal_dump_dir)

        # Arguments used to re-pretrain a reinitialized agent's CNN (see _on_reinitialized).
        self._pretrain_args = {
            "pretrain_CNN_mode": args.pretrain_CNNs,
            "freeze_pretrained_CNN": args.freeze_pretrained_CNNs,
            "learning_rate": (args.pretrain_learning_rate or args.learning_rate),
            "epochs": args.pretrain_epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "display_mode": args.display,
            "pretrain_CNNs_on_eval": args.pretrain_CNNs_on_eval,
            "deconvolution_factory": get_default_fn(build_cnn_decoder_from_args, args),
        }

        # `--population N` provides a symmetric default of N-N; `--pop_size n-m` overrides it.
        pop = getattr(args, "population", None)
        default_size = (pop, pop) if pop else (2, 2)
        self._init_population(args, roles=("sender", "receiver"), default_size=default_size, default_period=(0, 0))

    def _population_factories(self, args):
        return ((lambda: Sender.from_args(args)), (lambda: Receiver.from_args(args)))

    def _assign_current(self, producer, consumer):
        self._sender, self._receiver = producer, consumer

    # Overrides AliceBob.agents_for_CNN_pretraining: pretrain every agent in the population.
    def agents_for_CNN_pretraining(self):
        if(self.shared):
            raise NotImplementedError
        return self.all_agents

    # Overrides PopulationMixin hook: re-pretrain a reinitialized agent's CNN, if enabled.
    def _on_reinitialized(self, agent, role, epoch, data_iterator):
        if(self._pretrain_args.get("pretrain_CNN_mode") is not None):
            self.pretrain_agent_CNN(agent, data_iterator, **self._pretrain_args, agent_name=f"reborn {role} @epoch {epoch}")
