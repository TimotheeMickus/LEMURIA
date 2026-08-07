from datetime import datetime
import itertools as it

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.log import Progress
from ..utils.misc import build_optimizer, Unflatten, get_default_fn
from ..utils.modules import (
    MultiHeadsClassifier,
    build_candidate_encoder_from_args,
    build_predicate_encoder_from_args,
    build_cnn_decoder_from_args,
    build_cnn_encoder_from_args,
)

# ===========================================================================
# Pretraining
# ---------------------------------------------------------------------------
# A game owns a `pretrainer` attribute that is either `None` (no pretraining) or a `Pretrainer`
# instance encapsulating the whole procedure (see `Game.pretrainer` / `Game.run_pretraining`). This
# replaces the old `CNNPretrainable` mixin: pretraining is now a self-contained object rather than a
# set of methods bolted onto the game, which lets the common machinery live on the base classes
# (`Game.run_pretraining`, `PopulationMixin._on_reinitialized`).
#
# Concrete games expose, when they support pretraining:
#   * self.pretrainer            : a Pretrainer subclass instance (or None)
#   * agents_for_pretraining()   : list of (agent, role) pairs to pretrain (role is a short string
#                                  such as "sender"/"receiver" or "asker"/"retriever", used only for
#                                  naming/logging by the CNN pretrainer, and to pick the procedure by
#                                  the AlexBeth pretrainer).
# ===========================================================================


# ---------------------------------------------------------------------------
# CNN training routines (image games)
# ---------------------------------------------------------------------------
# These are plain functions rather than methods so that they can be reused both by `CNNPretrainer`
# and by AliceBob's receiver-preprocessor autoencoder (which is pretrained at construction time,
# independently of whether the main CNN pretraining is enabled). The only external dependency is the
# summary writer, which is passed in explicitly.

# Pretrains the CNN of an agent in category- or feature-wise mode. Returns the MultiHeadsClassifier
# (its heads are the temporary/discarded part; only the agent's image_encoder is kept).
def train_cnn_classifier(agent, data_iterator, pretrain_CNN_mode, learning_rate, epochs, steps_per_epoch, display_mode, pretrain_CNNs_on_eval, summary_writer, agent_name="agent"):
    loss_tag = 'pretrain/loss_%s_%s' % (agent_name, pretrain_CNN_mode)
    acc_tag = 'pretrain/acc_%s_%s' % (agent_name, pretrain_CNN_mode)

    concept_sizes = [len(concept) for concept in data_iterator.concepts]
    device = next(agent.parameters()).device

    if pretrain_CNN_mode == 'feature-wise':
        # Defines one classification head per non-unary concept
        heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(50, 3), # I guess the 3 could be replaced with csize
                nn.LogSoftmax(dim=1)
            ) for csize in concept_sizes if csize > 1
        ]).to(device)
        get_head_targets = (lambda cat: [v for v, csize in zip(cat, concept_sizes) if csize > 1])
    else:
        heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(50, data_iterator.nb_categories),
                nn.LogSoftmax(dim=1))
            ]
        ).to(device)
        get_head_targets = (lambda cat: [data_iterator.category_idx(cat)])

    optimizer = build_optimizer(it.chain(agent.image_encoder.parameters(), heads.parameters()), learning_rate)
    n_heads = len(heads)

    model = MultiHeadsClassifier(agent.image_encoder, optimizer, heads, n_heads, get_head_targets, device)

    total_items = 0
    for epoch_index in range(epochs):
        # Optimization
        pbar = Progress.get_progress_cls(display_mode)(steps_per_epoch, epoch_index, logged_items={'L', 'acc'})
        epoch_hits, epoch_items = 0., 0. # TODO Do they need to be floats instead of integers?
        with pbar:
            for step_i in range(steps_per_epoch):
                batch = data_iterator.get_batch(data_type='train', keep_category=True, no_evaluation=(not pretrain_CNNs_on_eval), sampling_strategies=[]) # For each instance of the batch, one original and one target image, but no distractor; only the target will be used

                hits, loss = model.run_batch(batch)

                for x in hits: epoch_hits += x.sum().item()
                epoch_items += batch.size
                total_items += batch.size

                if(summary_writer is not None):
                    summary_writer.add_scalar(loss_tag, (loss.item() / batch.size), total_items)
                    summary_writer.add_scalar(acc_tag, (epoch_hits / (epoch_items * n_heads)), total_items)
                pbar.update(L=loss.item(), acc=(epoch_hits / (epoch_items * n_heads)))

        # Evaluation
        with torch.no_grad():
            agent.image_encoder.eval()
            heads.eval()
            test_loss, test_epoch_hits, test_epoch_items =  0., 0, 0
            for step_i in range(1 + (steps_per_epoch // 10)):
                batch = data_iterator.get_batch(data_type='test', keep_category=True, no_evaluation=(not pretrain_CNNs_on_eval), sampling_strategies=[]) # For each instance of the batch, one original and one target image, but no distractor; only the target will be use
                hits, losses = model.forward(batch)
                for x in losses: test_loss += x.sum().item()
                for x in hits: test_epoch_hits += x.sum().item()
                test_epoch_items += batch.size
            test_loss = test_loss / (test_epoch_items * n_heads)
            test_acc = test_epoch_hits / (test_epoch_items * n_heads)
            if(summary_writer is not None):
                summary_writer.writer.add_scalar('eval-' + loss_tag, test_loss, epoch_index)
                summary_writer.writer.add_scalar('eval-' + acc_tag, test_acc, epoch_index)
            print(f"[eval-pretrain {agent_name} epoch {epoch_index}] L={test_loss}, acc={test_acc}")
            agent.image_encoder.train()
            heads.train()

    return model


# Pretrains the CNN of an agent in auto-encoder mode. `agent` may be None (external autoencoder, e.g.
# AliceBob's receiver preprocessor), in which case the (de)convolution factories build the modules.
# Returns the Sequential autoencoder model.
def train_cnn_autoencoder(agent, data_iterator, deconvolution_factory=None, convolution_factory=None, learning_rate=0.0001, epochs=5, steps_per_epoch=1000, display_mode='', pretrain_CNNs_on_eval=False, summary_writer=None, agent_name="agent", _is_external_ae=False, device=None):
    loss_tag = 'pretrain/loss_%s_auto-encoder' % agent_name
    if device is None:
        device = next(agent.parameters()).device

    found = False
    if agent is not None and hasattr(agent, 'image_encoder'):
        encoder = agent.image_encoder
        found = True
    else:
        encoder = convolution_factory()
    if agent is not None and hasattr(agent, 'image_decoder'):
        decoder = agent.image_decoder
        found = True
    else:
        decoder = deconvolution_factory()

    assert found or _is_external_ae, "Agent must have an image encoder or decoder!"

    model = nn.Sequential(
        encoder,
        Unflatten(),
        decoder,
    ).to(device)

    optimizer = build_optimizer(model.parameters(), learning_rate)

    total_items = 0
    data_type = 'any' if pretrain_CNNs_on_eval else 'train'

    for epoch_index in range(epochs):
        # Optimization
        epoch_loss, epoch_items = 0., 0. # TODO Do they need to be floats instead of integers?
        pbar = Progress.get_progress_cls(display_mode)(steps_per_epoch, epoch_index, logged_items={'L'})
        with pbar:
            for _ in range(steps_per_epoch):
                optimizer.zero_grad()

                batch = data_iterator.get_batch(data_type=data_type, keep_category=True, no_evaluation=(not pretrain_CNNs_on_eval), sampling_strategies=[])
                batch_img = batch.target_img(stack=True)
                output = model(batch_img)

                loss = F.mse_loss(output, batch_img, reduction="sum")

                epoch_loss += loss.item()
                epoch_items += batch_img.size(0)
                total_items += batch_img.size(0)
                if(summary_writer is not None): summary_writer.add_scalar(loss_tag, (loss.item() / batch_img.size(0)), total_items)
                pbar.update(L=(epoch_loss / epoch_items))

                loss.backward()
                optimizer.step()

        # Evaluation
        with torch.no_grad():
            model.eval()
            test_epoch_loss, test_epoch_items = 0., 0.
            for _ in range(1 + (steps_per_epoch // 10)):
                batch = data_iterator.get_batch(data_type='test', keep_category=True, no_evaluation=(not pretrain_CNNs_on_eval), sampling_strategies=[])
                batch_img = batch.target_img(stack=True)
                output = model(batch_img)
                loss = F.mse_loss(output, batch_img, reduction="sum")
                test_epoch_loss += loss.item()
                test_epoch_items += batch_img.size(0)
            test_epoch_loss = test_epoch_loss / test_epoch_items
            if(summary_writer is not None):
                summary_writer.writer.add_scalar('eval-' + loss_tag, test_epoch_loss, epoch_index)
            print(f"[eval-pretrain {agent_name} epoch {epoch_index}] L={test_epoch_loss}")
            model.train()

    return model


# ---------------------------------------------------------------------------
# Pretrainer base
# ---------------------------------------------------------------------------
class Pretrainer:
    """Base class for pretraining procedures.

    A game holds either `None` (no pretraining) or an instance of a subclass in `self.pretrainer`.
    The base class holds the reference back to the game (from which it reads the autologger and the
    agents to pretrain), the data loader, and the shared hyperparameters. It also emits a one-line
    description of the procedure at creation, including which parameters it would freeze.

    Subclasses implement:
      * `_describe()`            : short human string naming the pretraining task;
      * `_frozen_description()`  : human string naming the parameters frozen when `freeze` is set;
      * `pretrain_agent(agent, role, agent_name=None)` : pretrain a single agent (also used to
        re-pretrain a reinitialized agent after a population reset);
      * optionally `_pretrain_all()` : the full pass (defaults to pretraining every agent listed by
        the game; overridden e.g. for the shared case).
    """
    def __init__(self, game, args, dataset):
        self.game = game
        self.args = args
        self.dataset = dataset

        self.device = args.device
        self.display_mode = args.display
        self.freeze = args.freeze_pretrained_parameters
        self.epochs = args.pretrain_epochs
        self.steps_per_epoch = args.pretrain_steps_per_epoch or args.steps_per_epoch
        self.learning_rate = args.pretrain_learning_rate or args.learning_rate

        self._log_creation()

    @property
    def summary_writer(self):
        return self.game.autologger.summary_writer

    # --- description / logging at creation ---
    def _describe(self):
        raise NotImplementedError

    def _frozen_description(self):
        raise NotImplementedError

    def _log_creation(self):
        freeze_msg = (f"freezes {self._frozen_description()}" if self.freeze
                      else "keeps all pretrained parameters trainable")
        print(
            f"[{datetime.now()}] {type(self).__name__} created — task: {self._describe()}; "
            f"epochs={self.epochs}, steps/epoch={self.steps_per_epoch}, lr={self.learning_rate}; "
            f"on completion, {freeze_msg}.",
            flush=True,
        )

    # --- driving ---
    def pretrain(self):
        """Full pretraining, run once before training. Returns a dict {name: pretrained_model}
        (used by --detect_outliers for the CNN classifier); values may be None when irrelevant."""
        print(("[%s] pretraining start…" % datetime.now()), flush=True)
        return self._pretrain_all()

    def _pretrain_all(self):
        models = {}
        for i, (agent, role) in enumerate(self.game.agents_for_pretraining()):
            models[f"{role} {i}"] = self.pretrain_agent(agent, role)
        return models

    def pretrain_agent(self, agent, role, agent_name=None):
        raise NotImplementedError


# Command-line arguments common to every pretraining procedure. `frozen_desc` names, for the help
# of --freeze_pretrained_parameters, what that flag would freeze in this game. Returns the argparse
# group so a caller can append game-specific pretraining knobs (e.g. --pretrain / --pretrain_CNNs).
def add_pretraining_args(parser, frozen_desc):
    group = parser.add_argument_group(title='Pretraining', description='arguments relative to pretraining agents before the communication game')
    group.add_argument('--pretrain_epochs', help='number of epochs per agent for pretraining', type=int, default=2)
    #group.add_argument('--pretrain_epochs', help='number of epochs per agent for pretraining', type=int, default=5)
    group.add_argument('--pretrain_steps_per_epoch', help='number of steps per pretraining epoch', type=int, default=1000)
    #group.add_argument('--pretrain_steps_per_epoch', help='number of steps per pretraining epoch (defaults to --steps_per_epoch)', type=int, default=None)
    group.add_argument('--pretrain_learning_rate', help='learning rate for pretraining', type=float, default=0.01)
    #group.add_argument('--pretrain_learning_rate', help='learning rate for pretraining (defaults to --learning_rate)', type=float, default=None)
    group.add_argument('--freeze_pretrained_parameters', help=('after pretraining, freeze all pretrained parameters (here: %s) so that they are not updated during the game' % frozen_desc), action='store_true')
    return group


# ---------------------------------------------------------------------------
# CNN pretrainer (AliceBob family)
# ---------------------------------------------------------------------------
class CNNPretrainer(Pretrainer):
    """Pretrains each agent's convolutional stack on a classification (category-/feature-wise) or
    auto-encoding proxy task. The classification heads / the auto-encoder's counterpart module are
    temporary and discarded; only the agent's image_encoder (and, in auto-encoder mode, its
    image_decoder if it has one) are kept."""
    def __init__(self, game, args, dataset):
        self.mode = args.pretrain_CNNs # 'category-wise' | 'feature-wise' | 'auto-encoder'
        self.on_eval = args.pretrain_CNNs_on_eval
        self.deconvolution_factory = get_default_fn(build_cnn_decoder_from_args, args)
        self.convolution_factory = get_default_fn(build_cnn_encoder_from_args, args)
        super().__init__(game, args, dataset)

    def _describe(self):
        return f"CNN pretraining (mode={self.mode!r})"

    def _frozen_description(self):
        if(self.mode == 'auto-encoder'):
            return "each agent's pretrained image_encoder and image_decoder (whichever it has)"
        return "each agent's image_encoder (CNN)"

    # Modules that live inside the agent and must therefore keep their pretrained weights (so they
    # are what --freeze acts on). In auto-encoder mode both the encoder and the decoder may be the
    # agent's own (e.g. the drawer only has an image_decoder); in classification mode only the
    # encoder is trained/kept.
    def _kept_modules(self, agent):
        if(self.mode == 'auto-encoder'):
            modules = []
            if(hasattr(agent, 'image_encoder')): modules.append(agent.image_encoder)
            if(hasattr(agent, 'image_decoder')): modules.append(agent.image_decoder)
            return modules
        return [agent.image_encoder]

    def pretrain_agent(self, agent, role, agent_name=None):
        if(agent_name is None): agent_name = role
        print(("[%s] pretraining %s…" % (datetime.now(), agent_name)), flush=True)

        if(self.mode != 'auto-encoder'):
            model = train_cnn_classifier(
                agent, self.dataset, self.mode, self.learning_rate, self.epochs, self.steps_per_epoch,
                self.display_mode, self.on_eval, self.summary_writer, agent_name,
            )
        else:
            model = train_cnn_autoencoder(
                agent, self.dataset, deconvolution_factory=self.deconvolution_factory,
                convolution_factory=self.convolution_factory, learning_rate=self.learning_rate,
                epochs=self.epochs, steps_per_epoch=self.steps_per_epoch, display_mode=self.display_mode,
                pretrain_CNNs_on_eval=self.on_eval, summary_writer=self.summary_writer, agent_name=agent_name,
            )

        if(self.freeze):
            for module in self._kept_modules(agent):
                for p in module.parameters(): p.requires_grad = False

        return model


# ---------------------------------------------------------------------------
# AlexBeth pretrainer (predicate/candidate satisfaction task)
# ---------------------------------------------------------------------------
# The pretraining objective is, for each (predicate, candidate) pair, to predict whether the
# candidate satisfies the predicate — i.e. exactly the retriever's task, but with the emergent
# signal replaced by a direct encoding of the predicate:
#   * an asker keeps its `predicate_encoder` and is temporarily given a candidate encoder
#     (the same --candidate_encoder choice retrievers use);
#   * a retriever keeps its `candidate_encoder` and is temporarily given a predicate embedding
#     ("as if it were an asker").
# In both cases the temporary module is discarded once pretraining is over; only the encoder living
# inside the agent retains its pretrained weights. In the shared case both encoders already exist and
# are pretrained jointly in a single pass (no temporary module, and not once per role).

class PredicateCandidateClassifier(nn.Module):
    """Scores (predicate, candidate) pairs exactly the way a retriever scores (signal, candidate)
    pairs (see `Retriever.aux_forward`): the predicate is encoded to a single vector, each candidate
    to a vector, and the score of a candidate is their dot product.

    predicate_encoder: predicate_idx (batch,)               -> (batch, H)
    candidate_encoder: candidate tensors                    -> (batch, num_candidates, H)
    forward:                                                -> (batch, num_candidates) logits
    """
    def __init__(self, predicate_encoder, candidate_encoder):
        super().__init__()
        self.predicate_encoder = predicate_encoder
        self.candidate_encoder = candidate_encoder

    def forward(self, predicate_idx, candidate_tensors):
        encoded_predicate = self.predicate_encoder(predicate_idx)          # (batch, H)
        encoded_candidates = self.candidate_encoder(**candidate_tensors)   # (batch, num_candidates, H)
        # bmm((batch, num_candidates, H), (batch, H, 1)) -> (batch, num_candidates, 1) -> (batch, num_candidates)
        scores = torch.bmm(encoded_candidates, encoded_predicate.unsqueeze(-1)).squeeze(-1)
        return scores


class AlexBethPretrainer(Pretrainer):
    def __init__(self, game, args, dataset):
        self.batch_size = args.batch_size
        super().__init__(game, args, dataset)

    def _describe(self):
        return "predicate/candidate satisfaction pretraining"

    def _frozen_description(self):
        return "each agent's pretrained encoder (asker: predicate_encoder, retriever: candidate_encoder)"

    # Overrides the default full pass to special-case the shared game.
    def _pretrain_all(self):
        if(self.game.shared):
            # Shared AskerRetriever: the asker already owns the predicate encoder and the retriever
            # already owns the candidate encoder, so no temporary module is needed. Pretrain the two
            # jointly in a *single* pass — crucially not once as an asker and once as a retriever,
            # which would be redundant. (Analogous to AliceBob pretraining its shared CNN only once.)
            asker, retriever = self.game.asker, self.game.retriever
            self._run(
                predicate_encoder=asker.predicate_encoder,
                candidate_encoder=retriever.candidate_encoder,
                kept_modules=[asker.predicate_encoder, retriever.candidate_encoder],
                agent_name="asker+retriever (shared)",
            )
            return {}
        return super()._pretrain_all()

    def pretrain_agent(self, agent, role, agent_name=None):
        if(agent_name is None): agent_name = role

        if(role == "asker"):
            # Keep the asker's predicate encoder; pair it with a throwaway candidate encoder built
            # with the same command-line choice the retriever uses.
            self._run(
                predicate_encoder=agent.predicate_encoder,
                candidate_encoder=build_candidate_encoder_from_args(self.args).to(self.device),
                kept_modules=[agent.predicate_encoder],
                agent_name=agent_name,
            )
        elif(role == "retriever"):
            # Keep the retriever's candidate encoder; pair it with a throwaway predicate embedding
            # ("as if it were an asker").
            self._run(
                predicate_encoder=build_predicate_encoder_from_args(self.args).to(self.device),
                candidate_encoder=agent.candidate_encoder,
                kept_modules=[agent.candidate_encoder],
                agent_name=agent_name,
            )
        else:
            raise ValueError(f"Cannot pretrain agent with unknown role {role!r} (expected 'asker' or 'retriever').")

        return None

    # Runs the actual optimization. `kept_modules` are the modules that live inside an agent and must
    # therefore retain their pretrained weights; every other parameter of the classifier belongs to a
    # temporary module that is discarded when this method returns (its local `model` goes out of scope).
    def _run(self, predicate_encoder, candidate_encoder, kept_modules, agent_name):
        loss_tag = f'pretrain/loss_{agent_name}'
        acc_tag = f'pretrain/acc_{agent_name}'
        summary_writer = self.summary_writer

        model = PredicateCandidateClassifier(predicate_encoder, candidate_encoder).to(self.device)
        optimizer = build_optimizer(model.parameters(), self.learning_rate)

        print(("[%s] pretraining %s…" % (datetime.now(), agent_name)), flush=True)

        total_items = 0
        for epoch_index in range(self.epochs):
            # Optimization
            model.train()
            pbar = Progress.get_progress_cls(self.display_mode)(self.steps_per_epoch, epoch_index, logged_items={'L', 'acc'})
            epoch_hits, epoch_items = 0., 0.
            with pbar:
                for step_i in range(self.steps_per_epoch):
                    optimizer.zero_grad()

                    batch = self.dataset.get_batch(size=self.batch_size, data_type='train')
                    scores = model(self.game._alex_input(batch), self.game._beth_input(batch)) # (batch, num_candidates)
                    truth = batch.candidate_truth                             # (batch, num_candidates)

                    loss = F.binary_cross_entropy_with_logits(scores, truth, reduction='mean')

                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        hits = (((scores >= 0.0).float() == truth).float().sum().item()) # sigmoid(score) >= 0.5 <=> score >= 0
                    epoch_hits += hits
                    epoch_items += truth.numel()
                    total_items += batch.size

                    if(summary_writer is not None):
                        summary_writer.add_scalar(loss_tag, loss.item(), total_items)
                        summary_writer.add_scalar(acc_tag, (epoch_hits / epoch_items), total_items)
                    pbar.update(L=loss.item(), acc=(epoch_hits / epoch_items))

            # Evaluation. The predicate loader has no held-out split (candidates are freshly sampled
            # every time and `data_type` is ignored), so these batches are same-distribution; they
            # still give an independent estimate of the pretraining task accuracy.
            with torch.no_grad():
                model.eval()
                test_loss, test_hits, test_items = 0., 0., 0.
                for _ in range(1 + (self.steps_per_epoch // 10)):
                    batch = self.dataset.get_batch(size=self.batch_size, data_type='test')
                    scores = model(self.game._alex_input(batch), self.game._beth_input(batch))
                    truth = batch.candidate_truth
                    test_loss += F.binary_cross_entropy_with_logits(scores, truth, reduction='sum').item()
                    test_hits += ((scores >= 0.0).float() == truth).float().sum().item()
                    test_items += truth.numel()
                test_loss = (test_loss / test_items)
                test_acc = (test_hits / test_items)
                if(summary_writer is not None):
                    summary_writer.writer.add_scalar('eval-' + loss_tag, test_loss, epoch_index)
                    summary_writer.writer.add_scalar('eval-' + acc_tag, test_acc, epoch_index)
                print(f"[eval-pretrain {agent_name} epoch {epoch_index}] L={test_loss}, acc={test_acc}")

        # If asked, freeze the pretrained encoder(s) that live inside the agent(s).
        if(self.freeze):
            for module in kept_modules:
                for p in module.parameters(): p.requires_grad = False

        return model
