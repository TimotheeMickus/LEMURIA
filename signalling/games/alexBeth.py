import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
import itertools as it
import csv, os # for dumping signals

import tqdm
from collections import defaultdict
import random
import time

from ..agents import Asker, Retriever, AskerRetriever
from ..utils.misc import build_optimizer, compute_entropy_stats
# from ..utils.predicate_data import 
from ..utils import misc, predicate_data

from ..eval import compute_correlation
from ..eval import decision_tree
from ..eval import compositionality

from .game import Game
from .signalling_eval import SignallingEvalMixin, dump_signals_csv, topographic_similarity
from .depth_curriculum import DepthCurriculum

# In this game, there is one asker (Alex) and one retriever (Beth).
# They are both trained to maximise either the probability assigned by Beth to an object (if the object satisfies the predicate), or its opposite (otherwise), in the following context: 
# Alex is shown a predicate and produces a signal, Beth sees both the signal and an object, and produces a probability.
# Alex is trained with REINFORCE; Beth is trained by log-likelihood maximization.
class AlexBeth(SignallingEvalMixin, Game):
    def __init__(self, args, logger, dataset, signal_dump_dir):
        self.max_perf = 0.0

        self._logger = logger
        self._dataset = dataset

        self.use_expectation = args.use_expectation
        self.grad_scaling = (args.grad_scaling or 0)
        self.grad_clipping = (args.grad_clipping or 0)
        self.beta_asker = args.beta_asker
        self.beta_retriever = args.beta_retriever
        self.len_penalty = args.len_penalty
        self.voc_penalty = args.voc_penalty

        self.base_alphabet_size = args.base_alphabet_size # Number of symbols excluding special ones (padding, EOS, etc.)

        self.shared = args.shared # Whether some parameters are shared between Alex and Beth.
        if(self.shared):
            raise NotImplementedError
            askerRetriever = AskerRetriever.from_args(args)

            self._asker = askerRetriever.asker
            self._retriever = askerRetriever.retriever

            parameters = askerRetriever.parameters()
        
            assert (self._asker.alphabet_size == self._retriever.alphabet_size) # The asker and the retriever have the exact same vocabulary.
        else:
            self._asker = Asker.from_args(args)
            self._retriever = Retriever.from_args(args)

            parameters = it.chain(self.asker.parameters(), self.retriever.parameters())
            
            assert (self._asker.alphabet_size == (self._retriever.alphabet_size + 1)) # Only the asker has the BOS symbol in its vocabulary.
        
        self.full_alphabet_size = self._asker.alphabet_size - 1 # Number of symbols that can be found in the signals; this includes padding and EOS but excludes BOS.
        assert (self.full_alphabet_size == (self.base_alphabet_size + 2))
        
        self.max_len_signal = args.max_len

        self._optim = build_optimizer(parameters, args.learning_rate)
        
        self.use_baseline = args.use_baseline
        if(self.use_baseline): # In that case, the loss will take into account the "baseline term" into the average recent reward.
            # Currently, the asker and retriever's rewards are the same, but we could imagine a setting in which they are different.
            self._asker_avg_reward = misc.Averager(size=12800)
            self._retriever_avg_reward = misc.Averager(size=12800)

        self.dump_signal_mode = getattr(args, "dump_signals", None)
        self.dump_predicate_perf = getattr(args, "dump_predicate_perf", False)
        self.dump_eval_metrics_enabled = getattr(args, "dump_eval_metrics", False)
        # Fancy language eval is only needed for eval metrics.
        self.run_fancy_lang_eval = bool(self.dump_eval_metrics_enabled)
        self.correct_only = args.correct_only # Whether to perform the fancy language evaluation using only correct signals (i.e., the one that leads to successful communication).
        self.use_jaccard_eval = getattr(args, "jaccard", False)
        self._topsim_correl_only = True # If True, skips the Mantel permutations (fast; correlation only, no p/z).
        # Compositionality probe (biLSTM->LSTM seq2seq): measured during evaluation when enabled.
        self.eval_compositionality = getattr(args, "eval_compositionality", False)
        self._comp_hparams = compositionality.hparams_from_args(args) if self.eval_compositionality else None
        # DEBUG FEATURE (--eval_oracle_language): replace the emergent language with a known-
        # compositional "control"/oracle language (the reverse-Polish encoding of the predicate)
        # during fancy evaluation. This is a debugging/diagnostic aid rather than part of normal
        # runs: it shows what the language metrics report for a language that is compositional by
        # construction (an upper-bound sanity check). Applied to the compositionality probe and
        # topographic similarity.
        self.eval_oracle_language = getattr(args, "eval_oracle_language", False)
        self._oracle_signals_cache = None  # list[list[int]] indexed by predicate index; built lazily
        self._comp_seed = getattr(args, "comp_search_seed", 0)
        self.epochs = getattr(args, "epochs", None)
        # Negation metrics only run when negation exists.
        self.no_negation = getattr(args, "no_negation", False)
        # Predicate-depth training curriculum (disabled when --depth_curriculum_threshold is
        # None). All of its state is encapsulated in this single object.
        self._depth_curriculum = DepthCurriculum(getattr(args, "depth_curriculum_threshold", None), dataset)
        if(self._depth_curriculum.enabled):
            print(
                f"[depth-curriculum] enabled: starting at depth {self._depth_curriculum.current_max_depth} "
                f"(cap {self._depth_curriculum.max_depth}), unlock threshold eval/accuracy >= {self._depth_curriculum.threshold}"
            )
        # Used to decide whether to dump signals during a hike in performance.
        self._prev_eval_perf = None
        self._best_eval_perf = None
        self._predicate_negation_idx = self._build_negation_correspondence()
        # For topographic similarity: candidate id = position in list
        self._topsim_candidates = self._build_candidate_vector()
        # Row-level predicate diagnostics accumulated across eval calls.
        self._predicate_perf_rows = []  # (epoch, pred_idx, perf, acc)
        self._predicate_text_by_idx = {}
        # One row per evaluate() call (epoch-level aggregate metrics).
        self._eval_metrics_rows = []

        if(self.dump_eval_metrics_enabled and (not self.run_fancy_lang_eval)):
            raise ValueError("--dump_eval_metrics requires fancy eval metrics; enable --dump_signals.")
        
        self.debug = args.debug
        self.signal_dump_dir = signal_dump_dir # str|None

    @property
    def asker(self):
        return self._asker

    @property
    def retriever(self):
        return self._retriever

    @property
    def all_agents(self):
        return (self.asker, self.retriever)

    @property
    def current_agents(self):
        return self.all_agents

    @property
    def optims(self):
        return [self._optim]

    @property
    def autologger(self):
        return self._logger

    def _build_candidate_vector(self):
        candidate_vector = [
            predicate_data.Candidate({prop: value for prop, value in zip(self._dataset.properties, values)})
            for values in it.product(*(prop.values for prop in self._dataset.properties))
        ]
        return candidate_vector
    
    def _polarity_repr(self, predicate):
        """Returns a frozenset of signed literals where literal identity is property-level."""
        signed_literals = []
        # stack entries are (predicate, sign ∈ {+1, -1})
        stack = [(predicate, 1)]

        while stack:
            p, sign = stack.pop()
            if(isinstance(p, predicate_data.Conjunction)):
                stack.append((p.pred2, sign))
                stack.append((p.pred1, sign))
            elif(isinstance(p, predicate_data.Negation)):
                stack.append((p.predicate, -sign))
            else:
                if(isinstance(p, predicate_data.Value)):
                    atom_name = p.prop.name
                else:
                    atom_name = str(p)
                signed_literals.append(("+" if(sign > 0) else "-", atom_name))

        return frozenset(signed_literals)


    def _build_negation_correspondence(self):
        '''
        Returns a dictionary of {predicate index: index of its negation}.
        Builds pairs of the indices of predicates in `self._dataset.predicates`.
        This works both ways: if p[i]=¬p and p[j]=p, store both i: j and j: i.
            partner: dict[int, int]
        '''
        partner = {} # dict[int,int]

        # Map predicates to their indices
        # This should perhaps be a feature of the Dataset
        pred2idx = {pred: i for i, pred in enumerate(self._dataset.predicates)}
        # For each negative predicate and its index, Negation stores its positive "base"
        for i, pred in enumerate(self._dataset.predicates):
            if(isinstance(pred, predicate_data.Negation)):
                # Map the negation and the base to each other
                j = pred2idx[pred.predicate]
                partner[i] = j
                partner[j] = i

        return partner

    def dump_predicate_performance(self, output_dir, wandb_run=None, artifact_name=None):
        # Save one raw row-level table at the end of the run.
        if((not self.dump_predicate_perf) or (len(self._predicate_perf_rows) == 0)):
            return

        os.makedirs(output_dir, exist_ok=True)
        rows_path = os.path.join(output_dir, "predicate_perf_rows.csv")

        with open(rows_path, "w") as ostr:
            writer = csv.writer(ostr)
            writer.writerow(["epoch", "pred_idx", "pred_str", "row_perf", "row_acc"])
            for epoch, pred_idx, row_perf, row_acc in self._predicate_perf_rows:
                writer.writerow([epoch, pred_idx, self._predicate_text_by_idx.get(pred_idx, ""), row_perf, row_acc])

        if(wandb_run is not None):
            import wandb
            artifact = wandb.Artifact(name=f"predicate-performance-{wandb_run.id}", type="analysis")
            artifact.add_file(rows_path)
            wandb_run.log_artifact(artifact)

    def dump_eval_metrics(self, output_dir, wandb_run=None):
        # Save one epoch-level table at the end of the run.
        if((not self.dump_eval_metrics_enabled) or (len(self._eval_metrics_rows) == 0)):
            return

        os.makedirs(output_dir, exist_ok=True)
        rows_path = os.path.join(output_dir, "eval_metrics_rows.csv")
        fieldnames = [
            "epoch",
            "eval/retriever_loss",
            "eval/perf",
            "eval/accuracy",
            "eval/retriever_entropy",
            "eval/asker_entropy",
            "eval/signal_length",
            "eval/vocab_used",
            "eval/compositionality_acc",
            "eval/compositionality_loss",
            "eval/scrambling-resistance",
            "eval/topsim_extensional_norm_levenshtein",
            "eval/topsim_extensional_norm_levenshtein_p",
            "eval/topsim_extensional_norm_levenshtein_z",
            "eval/topsim_extensional_multi_jaccard",
            "eval/topsim_extensional_multi_jaccard_p",
            "eval/topsim_extensional_multi_jaccard_z",
            "eval/topsim_intensional_norm_levenshtein",
            "eval/topsim_intensional_norm_levenshtein_p",
            "eval/topsim_intensional_norm_levenshtein_z",
            "eval/topsim_intensional_multi_jaccard",
            "eval/topsim_intensional_multi_jaccard_p",
            "eval/topsim_intensional_multi_jaccard_z"
        ]
        # Present only when the depth curriculum was active (matches the row dict).
        if(self._depth_curriculum.enabled):
            fieldnames.append("eval/curriculum_max_depth")

        with open(rows_path, "w") as ostr:
            writer = csv.DictWriter(ostr, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._eval_metrics_rows)

        if(wandb_run is not None):
            import wandb
            artifact = wandb.Artifact(name=f"eval-metrics-{wandb_run.id}", type="analysis")
            artifact.add_file(rows_path)
            wandb_run.log_artifact(artifact)
    
    # The name is misleading (reflects an older version): this only converts truth to a tensor on the device.
    def _compute_truth_targets(self, batch):
        """
        Returns a float tensor of shape (batch size, nb candidates) where 1.0 denotes that the predicate holds for the candidate, and 0.0 otherwise.
        """
        return batch.candidate_truth
    
    # batch: Batch
    def _alex_input(self, batch):
        return batch.predicate_idx

    # batch: Batch
    def _beth_input(self, batch):
        return {
            'node_idx': batch.node_idx, 
            'edge_idx': batch.edge_idx, 
            'graph_size': batch.graph_sizes, 
        }

    def __call__(self, batch):
        """
        Input:
            batch: Batch
        Output:
            asker_outcome: asker.Outcome
            retriever_outcome: retriever.Outcome
        """
        return self.alex_to_beth(batch)

    # batch: Batch
    def alex_to_beth(self, batch):
        asker = self.asker
        retriever = self.retriever

        asker_outcome = asker(self._alex_input(batch))
        retriever_outcome = retriever(self._beth_input(batch), *asker_outcome.action)

        return asker_outcome, retriever_outcome

    # batch: Batch
    def compute_interaction(self, batch, **kwargs):
        asker_outcome, retriever_outcome = self(batch)
        truth_targets = self._compute_truth_targets(batch)

        # Alex's part
        (asker_loss, asker_perf, asker_rewards) = self.compute_asker_loss(asker_outcome, retriever_outcome.scores, truth_targets)
        asker_entropy = asker_outcome.entropy.mean()

        # Beth's part
        (retriever_loss, _, retriever_entropy) = self.compute_retriever_loss(retriever_outcome.scores, truth_targets, return_entropy=True)

        loss = asker_loss + retriever_loss
        if(torch.isnan(loss)): print(f"[warn] loss is {loss}")
        optimization = [(self._optim, loss.detach(), misc.get_backward_f(loss))]

        signal_length = asker_outcome.action[1].float().mean()

        metrics = {
            "rewards": asker_rewards,
            "successes": asker_perf,
            "signal_length": signal_length,
            "sender_entropy": asker_entropy,
            "receiver_entropy": retriever_entropy,
        }

        return optimization, metrics

    # Returns two tensors of shape (batch size).
    # asker_action: pair (signal, length) where signal is a tensor of shape (batch size, max signal length) and length a tensor of shape (batch size)
    # retriever_scores: logits pair (batch, num_candidates)
    # truth_targets: (batch, num_candidates)
    def compute_asker_rewards(self, asker_action, retriever_scores, truth_targets):
        """
        Returns reward and performance tensors (both shaped [batch size]) 
        based on the probability Beth assigns to the correct truth value.
        """
        # Selects whether the retriever is right.
        scores_right = torch.where((truth_targets > 0.5), retriever_scores, -retriever_scores) # Shape: (batch, candidates)
        selecting_right = misc.selecting(scores_right)

        perf = selecting_right["dist"].probs.mean(dim=1).detach() # Shape: (batch,)
        
        if(self.use_expectation):
            rewards = perf.clone() # Expected average accuracy of the retriever over the candidates. # Shape: (batch,)
        else:
            rewards = selecting_right["actions"].mean(dim=1) # Shape: (batch,)

        # Generates probabilities from the scores and selects candidates.
        #retriever_selecting = misc.selecting(retriever_scores)
        #
        #dist = retriever_selecting["dist"]
        #perf = torch.where((truth_targets > 0.5), dist.probs, (1.0 - dist.probs)).mean(dim=1).detach() # Shape: (batch,)
        #
        #if(self.use_expectation):
        #    rewards = perf.clone() # Expected average accuracy of the retriever over the candidates. # Shape: (batch,)
        #else:
        #    rewards = torch.isclose(retriever_selecting["actions"], truth_targets).float().mean(dim=1) # Shape: (batch,)

        signal_lengths = asker_action[1].view(-1).float() # Shape: (batch,), includes the EOS symbol (usualy 0).

        rewards += -1 * (signal_lengths >= self.max_len_signal) # Penalty related to signals exceeding the length limit.

        if(self.len_penalty > 0.0):
            # The penalty equals 0 when `len_penalty` is set to 0, and increases (the faster the higher `len_penalty` is) to 1 with the length of the signal otherwise.
            # RMK: We could imagine a non-uniform penalty (that depends on the position of the token for symbols ≠ EOS).
            length_penalties = 1.0 - (1.0 / (1.0 + self.len_penalty * signal_lengths)) # Shape: (batch,)

            rewards = (rewards - length_penalties) # Shape: (batch,)

        if(self.voc_penalty > 0.0):
            # Each symbol of the base alphabet is associated with a total penalty equal to `batch_size` * `voc_penalty`, distributed over all signals in proportion of their use of the symbol (as if the total penalty were distributed equally over all occurrences of the symbol).
            # Ex: If each signal uses a single symbol and at least once, and all signals use different symbols, each signal gets a penalty equal to `voc_penalty`.
            # Ex: If all signal uses a single symbol overall, each signal gets a penalty equal to `voc_penalty` * its length * `batch_size` / sum of lengths, which is `voc_penalty` when the signals are of the same length ≥ 1.
            # RMK: We could imagine a non-uniform penalty (that depends on the position of the token).
            vocabulary_counts = torch.bincount(asker_action[0].view(-1), minlength=self.full_alphabet_size) # Shape: (full_alphabet_size,), includes padding and EOS
            #vocabulary_counts[self.asker.eos_index] = 0 # no penalty for using EOS
            #vocabulary_counts[self.asker.padding_idx] = 0 # no penalty for "using" padding

            vocabulary_freqs = 1.0 / vocabulary_counts # Shape: (full_alphabet_size,)
            vocabulary_freqs[self.asker.eos_index] = 0.0 # no penalty for using EOS
            vocabulary_freqs[self.asker.padding_idx] = 0.0 # no penalty for "using" padding

            batch_size = asker_action[0].shape[0]
            symbol_penalty = (self.voc_penalty * batch_size) * vocabulary_freqs # Shape: (full_alphabet_size,)

            vocabulary_penalties = symbol_penalty[asker_action[0]].sum(dim=1) # Shape (batch,)
            
            #print(asker_action[0])
            #print(vocabulary_counts)
            #print(symbol_penalty)
            #input(vocabulary_penalties)

            rewards = (rewards - vocabulary_penalties) # Shape: (batch,)

        return (rewards, perf)

    # Returns (loss, perf, rewards) where loss is a scalar and perf and rewards are of shape (batch,)
    # asker_outcome: (log_prob tensor of shape (batch, max signal len), entropy tensor of shape (batch, 1))
    # retriever_scores: tensor of shape (batch size, number of candidates)
    # truth_targets: tensor of shape (batch size, number of candidates)
    def compute_asker_loss(self, asker_outcome, retriever_scores, truth_targets):
        (rewards, perf) = self.compute_asker_rewards(asker_outcome.action, retriever_scores, truth_targets)

        loss = 0.0

        # REINFORCE loss
        log_prob = asker_outcome.log_prob.sum(dim=1) # The per-episode sum of the log-probabilies of the selection actions (they all get the same reward). Shape: (batch size)

        if(self.use_baseline):
            r_baseline = self._asker_avg_reward.get(default=0.0)
            self._asker_avg_reward.update_batch(rewards.cpu().numpy())
        else: r_baseline = 0.0

        reinforce_loss = -((rewards - r_baseline) * log_prob).mean()
        loss += reinforce_loss

        # Entropy penalty
        entropy_loss = -(self.beta_asker * asker_outcome.entropy.mean()) # Could be normalised (divided) by (base_alphabet_size + 1).
        loss += entropy_loss

        return (loss, perf, rewards)

    # Returns the loss (a scalar tensor) and, if asked, also the average entropy of the pointing distributions (a scalar tensor).
    # retriever_scores: logits pair tensor of shape (batch size, number of candidates)
    # truth_targets: tensor of shape (batch size, number of candidates)
    # return_entropy: bool
    def compute_retriever_loss(self, retriever_scores, truth_targets, return_entropy=False):
        # Generates probabilities from the scores and selects candidates.
        retriever_selecting = misc.selecting(retriever_scores)
        dist = retriever_selecting["dist"] # RMK: dist.logits == retriever_scores

        perf = torch.where((truth_targets > 0.5), dist.probs, (1.0 - dist.probs)).detach() # Shape: (batch_size, num candidates)
        
        entropy = dist.entropy().mean()
        
        loss = 0.0
        
        # Main loss
        #cross_entropy_loss = F.binary_cross_entropy(dist.probs, truth_targets, reduction='mean')
        cross_entropy_loss = F.binary_cross_entropy_with_logits(retriever_scores, truth_targets, reduction='mean')
        loss += cross_entropy_loss

        # Entropy penalty
        if(self.beta_retriever != 0.0):
            entropy_loss = -(self.beta_retriever * entropy)
            loss += entropy_loss

        if(return_entropy): return (loss, perf, entropy)
        return (loss, perf)

    # Measures the compositionality of the emergent language: the asker produces one
    # signal per predicate, and a seq2seq probe is trained (5-fold CV, early stopping) to
    # reconstruct the predicate in Polish notation from the signal. Returns the mean over
    # folds of each fold's best held-out exact-match rate (a float in [0, 1]) and best
    # held-out loss, as an (accuracy, loss) pair.
    def _compute_compositionality(self):
        device = next(self.asker.parameters()).device
        if(self.eval_oracle_language):
            # Oracle: feed the probe the reverse-Polish encoding of each predicate instead of
            # the emergent signal (targets stay in Polish notation); this is exactly the
            # compositional control language, so the probe should reconstruct it near-perfectly.
            pairs, spec = compositionality.reverse_polish_pairs(self._dataset)
        else:
            pairs, spec = compositionality.emergent_pairs(self.asker, self._dataset, device)
        return compositionality.compositionality(pairs, spec, self._comp_hparams, device, seed=self._comp_seed)

    # Reverse-Polish "oracle" signal for each predicate, as a list of int symbols, indexed by
    # predicate index (parallel to self._dataset.predicates). Built once and cached. Used only
    # when self.eval_oracle_language is set, to replace the emergent signals during fancy eval.
    def _oracle_signals_by_predicate(self):
        if(self._oracle_signals_cache is None):
            vocab = compositionality.PredicateVocab(self._dataset)
            self._oracle_signals_cache = [vocab.encode(pred.reverse_polish()) for pred in self._dataset.predicates]
        return self._oracle_signals_cache

    # Overrides SignallingEvalMixin._signal_correctness: P(correct) under the retriever for each
    # candidate (P(true) where the predicate holds, P(false) otherwise), given (possibly
    # scrambled) signals.
    def _signal_correctness(self, batch, signals, lengths):
        truth_targets = self._compute_truth_targets(batch)
        outcome = self.retriever(self._beth_input(batch), signal=signals, length=lengths)
        probs = torch.sigmoid(outcome.scores)
        return torch.where(truth_targets > 0.5, probs, 1.0 - probs)

    # Called at the end of each training epoch.
    # data_loader: Dataset
    # epoch_index: int
    @torch.no_grad()
    def evaluate(self, data_loader, epoch_index):
        def log(name, value): self._log(name, value, epoch_index)

        # We try to visit each predicate on average 8 times.
        batch_size = data_loader.batch_size
        max_datapoints = 32768 # (2^15)
        n = (8 * data_loader.nb_categories)
        n = min(max_datapoints, n)
        nb_batch = int(np.ceil(n / batch_size))

        # Epoch-level totals (we aggregate batch by batch, then normalize once at the end).
        total_items = 0
        total_retriever_loss = 0.0
        total_perf = 0.0 # average probability of the retriever selecting correctly
        total_accuracy = 0.0 # average accuracy of the retriever argmax selection
        total_entropy = 0.0
        total_asker_entropy = 0.0
        total_signal_length = 0.0
        # Scrambling resistance = preserved correctness after shuffling / original correctness.
        perf_scrambled = 0.0
        perf_baseline = 0.0
        # Count symbol usage across eval batches.
        total_vocab_counts = None

        # Cache for signal dumps.
        dump_cache = None
        if(self.dump_signal_mode):
            dump_cache = {
                "signals": [],
                "predicate_ids": [],
                "predicate_texts": [],
            }

        # Cache for language-level eval metrics.
        eval_cache = None
        if(self.run_fancy_lang_eval):
            eval_cache = dump_cache if(dump_cache is not None) else {
                "signals": [],
                "predicate_ids": [],
                "predicate_texts": [],
            }

        batch_numbers = range(nb_batch)
        if(self.autologger.display == 'tqdm'): batch_numbers = tqdm.tqdm(batch_numbers, desc='Eval.')

        for batch_index in batch_numbers:
            self.start_episode(train_episode=False)
            
            batch = data_loader.get_batch(size=batch_size, data_type='test', predicate_sampling='random') # RMK: `data_type` currently has no effect.
            truth_targets = self._compute_truth_targets(batch) # Shape: (batch_size, num_candidates)

            asker_outcome, retriever_outcome = self.alex_to_beth(batch)

            retriever_selecting = misc.selecting(retriever_outcome.scores, argmax=True)
            dist = retriever_selecting["dist"] # RMK: dist.logits == retriever_outcome.scores
            
            entropy = dist.entropy().mean().item()
            asker_entropy = asker_outcome.entropy.mean().item()
            
            retriever_loss = F.binary_cross_entropy_with_logits(retriever_outcome.scores, truth_targets, reduction='mean').item()

            correct_prob = torch.where((truth_targets > 0.5), dist.probs, (1.0 - dist.probs)) # Shape: (batch_size, num_candidates)
            perf = correct_prob.mean().item() # Shape: (batch_size, num_candidates)
            
            correct_pred = torch.isclose(retriever_selecting["actions"], truth_targets).float() # Shape: (batch_size, num_candidates)
            accuracy = correct_pred.mean().item() # average over all candidates

            # Updates the failure-based distribution.
            num_candidates = retriever_outcome.scores.shape[-1]
            predicate_idx = batch.predicate_idx.repeat_interleave(num_candidates).cpu().numpy() # Shape: (batch_size * num_candidates)
            failure = (1.0 - correct_prob).view(-1).cpu().numpy() # Shape: (batch_size * num_candidates)
            data_loader.failure_based_distribution.update(predicate_idx, failure, new_epoch=(batch_index == 0))

            # `signal_length`: average length (including EOS) of the signals in this batch.
            signal_length = asker_outcome.action[1].float().mean().item()
            # `total_vocab_counts`: count symbols used in signals (excluding EOS and padding).
            signal_tokens = asker_outcome.action[0]
            max_len = signal_tokens.size(1)
            positions = torch.arange(max_len, device=signal_tokens.device).unsqueeze(0) # int tensor of shape (1, max_len)
            signal_lens = asker_outcome.action[1].int().view(-1)
            in_signal = positions < (signal_lens.unsqueeze(1) - 1) # boolean tensor of shape (batch_size, max_len), True for positions strictly before EOS in each signal
            if in_signal.any():
                used_tokens = signal_tokens[in_signal]
                vocab_counts = torch.bincount(used_tokens, minlength=self.full_alphabet_size).to("cpu")
                vocab_counts[self.asker.eos_index] = 0
                vocab_counts[self.asker.padding_idx] = 0
                if(total_vocab_counts is None): # TODO Instead of doing this, initialise `total_vocab_counts` correctly.
                    total_vocab_counts = vocab_counts
                else:
                    total_vocab_counts += vocab_counts

            # Stores row-level performance for predicate diagnostics only when requested.
            if(self.dump_predicate_perf):
                row_perf = correct_prob.mean(dim=1).cpu().tolist()
                row_acc = correct_pred.mean(dim=1).cpu().tolist()
                for i, pred_idx in enumerate(batch.predicate_idx):
                    pred_idx = int(pred_idx)
                    if(pred_idx not in self._predicate_text_by_idx):
                        self._predicate_text_by_idx[pred_idx] = str(batch.predicate[i])
                    self._predicate_perf_rows.append((epoch_index, pred_idx, float(row_perf[i]), float(row_acc[i])))

            batch_items = truth_targets.numel()
            total_items += batch_items
            total_retriever_loss += retriever_loss * batch_items
            total_perf += perf * batch_items
            total_accuracy += accuracy * batch_items
            total_entropy += entropy * batch_items
            total_asker_entropy += asker_entropy * batch_items
            total_signal_length += signal_length * batch_items

            # Scrambling resistance: how much correctness is kept when the symbol order is
            # shuffled. High retention suggests the semantics rely less on order/composition.
            if(self.run_fancy_lang_eval):
                kept, base = self._scrambling_resistance(batch, asker_outcome.action[0], asker_outcome.action[1], correct_prob)
                perf_scrambled += kept
                perf_baseline += base

            # Cache signals once so dump and fancy eval can reuse them.
            # If `correct_only` is True, only correct items are cached.
            cache = dump_cache if(dump_cache is not None) else eval_cache
            if(cache is not None):
                batch_signals = asker_outcome.action[0].detach().clone()
                batch_lens = asker_outcome.action[1].detach().clone()
                accuracy_per_item = correct_pred.mean(dim=1) # (batch,) mean across candidates
                for i in range(batch_signals.size(0)):
                    if(self.correct_only and (not torch.isclose(accuracy_per_item[i], torch.tensor(1.0, device=accuracy_per_item.device)))):
                        # not: (accuracy_per_item[i].item() < 0.5):
                        continue # skip low accuracy items
                    # truncate padding away from signals
                    signal = batch_signals[i].tolist()[:batch_lens[i].item()]
                    cache["signals"].append(signal)
                    cache["predicate_ids"].append(int(batch.predicate_idx[i]))
                    # TODO This was crashing
                    # eval_cache["predicate_texts"].append(str(batch.predicate[i]))
                    # ---- and with this it works now:
                    pred_list = getattr(batch, "predicate", None)
                    if(pred_list is not None): 
                        cache["predicate_texts"].append(str(pred_list[i]))
                    else: 
                        cache["predicate_texts"].append(str(self._dataset.predicates[int(batch.predicate_idx[i])]))
                    

        # --- logging and stdout --- #
        #                            #
        # Normalise the accumulated sums and push them to TensorBoard / stdout.
        eval_retriever_loss = total_retriever_loss / total_items
        eval_perf = total_perf / total_items
        eval_accuracy = total_accuracy / total_items
        eval_retriever_entropy = total_entropy / total_items
        eval_asker_entropy = total_asker_entropy / total_items
        eval_signal_length = total_signal_length / total_items
        if(total_vocab_counts is None): eval_vocab_used = 0
        else: eval_vocab_used = int((total_vocab_counts > 0).sum().item())
        
        log('eval/retriever_loss', eval_retriever_loss)
        log('eval/perf', eval_perf)
        log('eval/accuracy', eval_accuracy)
        log('eval/retriever_entropy', eval_retriever_entropy)
        log('eval/asker_entropy', eval_asker_entropy)
        log('eval/signal_length', eval_signal_length)  # Average number of symbols Alex produced.
        log('eval/vocab_used', eval_vocab_used)

        # Compositionality: can a generic seq2seq learner recover each predicate (in Polish
        # notation) from its emergent signal? Reported as the 5-fold-CV exact-match rate
        # ('eval/compositionality_acc') and the corresponding held-out loss ('eval/compositionality_loss').
        compositionality_acc = float("nan")
        compositionality_loss = float("nan")
        if(self.eval_compositionality):
            compositionality_acc, compositionality_loss = self._compute_compositionality()
            log('eval/compositionality_acc', compositionality_acc)
            log('eval/compositionality_loss', compositionality_loss)
        
        avg_accuracy = eval_accuracy
        if(avg_accuracy > self.max_perf): self.max_perf = avg_accuracy

        is_perf_hike = False
        # Depth curriculum: unlock the next predicate depth once eval accuracy reaches the threshold.
        if(self._depth_curriculum.maybe_unlock(eval_accuracy, epoch_index)):
            is_perf_hike = True
            print(
                f"[depth-curriculum] epoch {epoch_index}: eval/accuracy={eval_accuracy:.6f} "
                f">= {self._depth_curriculum.threshold} -> unlocked depth {self._depth_curriculum.current_max_depth} "
                f"(now using depths {self._depth_curriculum.min_depth}..{self._depth_curriculum.current_max_depth})"
            )
        # Log the current max depth (only when the curriculum is active).
        if(self._depth_curriculum.enabled):
            log('eval/curriculum_max_depth', float(self._depth_curriculum.current_max_depth))
            if(self.autologger.display != 'minimal'):
                print(self._depth_curriculum.status_line(epoch_index))

        # Fancy metrics
        scrambling_ratio = None
        topsim_ext_levenshtein = float("nan")
        topsim_ext_levenshtein_p = float("nan")
        topsim_ext_levenshtein_z = float("nan")
        topsim_ext_jaccard = float("nan")
        topsim_ext_jaccard_p = float("nan")
        topsim_ext_jaccard_z = float("nan")
        topsim_int_levenshtein = float("nan")
        topsim_int_levenshtein_p = float("nan")
        topsim_int_levenshtein_z = float("nan")
        topsim_int_jaccard = float("nan")
        topsim_int_jaccard_p = float("nan")
        topsim_int_jaccard_z = float("nan")
        if(self.run_fancy_lang_eval):
            scrambling_ratio = 0
            if(perf_baseline > 0.0): scrambling_ratio = perf_scrambled / perf_baseline
            log('eval/scrambling-resistance', scrambling_ratio)

            # Topographic similarity
            # Extensional meaning is a binary vector over candidate IDs; used with Hamming distance.
            # Intensional meaning is the predicate embedding ("what the robots think"); used with cosine distance.
            if((eval_cache is not None) and (len(eval_cache["signals"]) > 1)):
                num_predicates = len(self._dataset.predicates)
                sample_size = min(256, num_predicates)
                sample = list(zip(eval_cache["signals"], eval_cache["predicate_ids"])) # sample = [([s0, s1], id0), ([s2], id1), ...]
                random.shuffle(sample)
                sample = sample[:sample_size]
                sample_signals = [tuple(s) for (s, _) in sample]
                sample_pred_ids = [int(pid) for (_, pid) in sample]

                # Oracle: replace the emergent signals with the reverse-Polish encoding of each
                # predicate (meaning side is left untouched). The dump cache is not affected.
                if(self.eval_oracle_language):
                    oracle_by_pred = self._oracle_signals_by_predicate()
                    sample_signals = [tuple(oracle_by_pred[pid]) for pid in sample_pred_ids]

                # Extensional meaning
                sample_cand_vecs = []
                for pid in sample_pred_ids:
                    pred = self._dataset.predicates[int(pid)]
                    candidate_vec = tuple(1 if(pred.check(cand) == 1) else 0 for cand in self._topsim_candidates)
                    sample_cand_vecs.append(candidate_vec)

                # Levenshtein (order dependent) vs. Jaccard (order invariant)
                # Levenshtein is normalised to account for varying signal length.
                if((len(set(sample_signals)) > 1) and (len(set(sample_cand_vecs)) > 1)):
                    # Intensional meaning
                    device = next(self.asker.parameters()).device
                    pred_idx_tensor = torch.tensor(sample_pred_ids, dtype=torch.long, device=device)
                    sender_vecs = [row for row in self.asker.predicate_encoder(pred_idx_tensor).cpu().numpy()] # per-predicate embedding

                    # Topographic similarity via the Mantel test (Spearman), deduplicating by meaning
                    # (predicate id) so repeated categories don't inflate the correlation.
                    def _topsim(meanings, meaning_distance, signal_distance, map_signal_to_str):
                        return topographic_similarity(
                            sample_signals, meanings,
                            signal_distance=signal_distance, meaning_distance=meaning_distance,
                            meaning_keys=sample_pred_ids, map_signal_to_str=map_signal_to_str, map_meaning_to_str=False,
                            method='spearman', deduplicate=True, correl_only=self._topsim_correl_only, error_on_duplicate_meanings=False,
                        )

                    ext_lev = _topsim(sample_cand_vecs, compute_correlation.hamming, compute_correlation.levenshtein_normalised, True)
                    topsim_ext_levenshtein, topsim_ext_levenshtein_p, topsim_ext_levenshtein_z = ext_lev
                    log('eval/topsim_extensional_norm_levenshtein', ext_lev.r)
                    if(not self._topsim_correl_only):
                        log('eval/topsim_extensional_norm_levenshtein_p', ext_lev.p)
                        log('eval/topsim_extensional_norm_levenshtein_z', ext_lev.z)
                    
                    int_lev = _topsim(sender_vecs, scipy.spatial.distance.cosine, compute_correlation.levenshtein_normalised, True)
                    topsim_int_levenshtein, topsim_int_levenshtein_p, topsim_int_levenshtein_z = int_lev
                    log('eval/topsim_intensional_norm_levenshtein', int_lev.r)
                    if(not self._topsim_correl_only):
                        log('eval/topsim_intensional_norm_levenshtein_p', int_lev.p)
                        log('eval/topsim_intensional_norm_levenshtein_z', int_lev.z)

                    if(self.use_jaccard_eval):
                        ext_jac = _topsim(sample_cand_vecs, compute_correlation.hamming, compute_correlation.jaccard, False)
                        topsim_ext_jaccard, topsim_ext_jaccard_p, topsim_ext_jaccard_z = ext_jac
                        log('eval/topsim_extensional_multi_jaccard', ext_jac.r)
                        if(not self._topsim_correl_only):
                            log('eval/topsim_extensional_multi_jaccard_p', ext_jac.p)
                            log('eval/topsim_extensional_multi_jaccard_z', ext_jac.z)
                        
                        int_jac = _topsim(sender_vecs, scipy.spatial.distance.cosine, compute_correlation.jaccard, False)
                        topsim_int_jaccard, topsim_int_jaccard_p, topsim_int_jaccard_z = int_jac
                        log('eval/topsim_intensional_multi_jaccard', int_jac.r)
                        if(not self._topsim_correl_only):
                            log('eval/topsim_intensional_multi_jaccard_p', int_jac.p)
                            log('eval/topsim_intensional_multi_jaccard_z', int_jac.z)
                else:
                    log('eval/topsim_extensional_norm_levenshtein', topsim_ext_levenshtein)
                    log('eval/topsim_intensional_norm_levenshtein', topsim_int_levenshtein)
                    if(self.use_jaccard_eval):
                        log('eval/topsim_extensional_multi_jaccard', topsim_ext_jaccard)
                        log('eval/topsim_intensional_multi_jaccard', topsim_int_jaccard)
                    if(self.autologger.display != 'minimal'):
                        print('eval/topsim\tnot enough variation in sampled signals/meanings')

                # Decision tree TODO: how easily predicate identity can be recovered from signals.

        if(self.dump_eval_metrics_enabled):
            row = {
                "epoch": epoch_index,
                "eval/retriever_loss": float(eval_retriever_loss),
                "eval/perf": float(eval_perf),
                "eval/accuracy": float(eval_accuracy),
                "eval/retriever_entropy": float(eval_retriever_entropy),
                "eval/asker_entropy": float(eval_asker_entropy),
                "eval/signal_length": float(eval_signal_length),
                "eval/vocab_used": eval_vocab_used,
                "eval/compositionality_acc": float(compositionality_acc),
                "eval/compositionality_loss": float(compositionality_loss),
                "eval/scrambling-resistance": scrambling_ratio,
                "eval/topsim_extensional_norm_levenshtein": topsim_ext_levenshtein,
                "eval/topsim_extensional_norm_levenshtein_p": topsim_ext_levenshtein_p,
                "eval/topsim_extensional_norm_levenshtein_z": topsim_ext_levenshtein_z,
                "eval/topsim_extensional_multi_jaccard": topsim_ext_jaccard,
                "eval/topsim_extensional_multi_jaccard_p": topsim_ext_jaccard_p,
                "eval/topsim_extensional_multi_jaccard_z": topsim_ext_jaccard_z,
                "eval/topsim_intensional_norm_levenshtein": topsim_int_levenshtein,
                "eval/topsim_intensional_norm_levenshtein_p": topsim_int_levenshtein_p,
                "eval/topsim_intensional_norm_levenshtein_z": topsim_int_levenshtein_z,
                "eval/topsim_intensional_multi_jaccard": topsim_int_jaccard,
                "eval/topsim_intensional_multi_jaccard_p": topsim_int_jaccard_p,
                "eval/topsim_intensional_multi_jaccard_z": topsim_int_jaccard_z,
            }
            # Only recorded when the depth curriculum is active (constant within a run).
            if(self._depth_curriculum.enabled):
                row["eval/curriculum_max_depth"] = int(self._depth_curriculum.current_max_depth)
            missing = [k for k, v in row.items() if((k != "epoch") and (v is None))]
            if(missing):
                raise RuntimeError(
                    "Missing eval metrics for epoch "
                    f"{epoch_index}: {', '.join(missing)}. "
                    "Enable fancy eval with --dump_eval_metrics."
                )
            self._eval_metrics_rows.append(row)

        # Decides if there is a performance hike.
        def _min_jump(best):
            if(best < 0.50): return 0.10
            if(best < 0.70): return 0.05
            if(best < 0.90): return 0.01
            if(best < 0.95): return 0.005
            return 0.001
        
        if(self.dump_signal_mode in ('when_hike', 'when_hike_strict')):
            if(self._depth_curriculum.is_epoch_after_unlock(epoch_index)):
                # Reset performance baseline (the predicate set just enlarged).
                self._best_eval_perf = eval_perf
                self._prev_eval_perf = eval_perf
                is_perf_hike = False
            else:
                if(self._best_eval_perf is None):
                    self._best_eval_perf = eval_perf
                if(not is_perf_hike):
                    if((eval_perf >= 1.0) and (eval_perf > self._best_eval_perf)):
                        is_perf_hike = True
                    else:
                        delta_min = _min_jump(self._best_eval_perf)
                        if(self.dump_signal_mode == 'when_hike_strict'):
                            is_perf_hike = (eval_perf > self._best_eval_perf + delta_min) and (eval_perf > 0.95)
                        else:
                            is_perf_hike = (eval_perf > self._best_eval_perf + delta_min)

                if(is_perf_hike and (eval_perf > self._best_eval_perf)):
                    self._best_eval_perf = eval_perf
                self._prev_eval_perf = eval_perf
        # Dumps signals into file every epoch or on the last epoch, depending on the flag.
        if(self.signal_dump_dir and (dump_cache is not None) and (
            (self.dump_signal_mode == 'all') or 
            ((self.dump_signal_mode == 'last') and (epoch_index == (self.epochs - 1))) or
            ((self.dump_signal_mode in ('when_hike', 'when_hike_strict')) and (
                 is_perf_hike 
                 or self._depth_curriculum.is_epoch_after_unlock(epoch_index)
                 or (epoch_index == (self.epochs - 1))))
            )):
            filename = os.path.join(self.signal_dump_dir, f"signals.e{epoch_index}.csv")
            rows = [
                [' '.join(map(str, signal)), pred_idx, pred_text]
                for signal, pred_idx, pred_text in zip(
                    dump_cache["signals"],
                    dump_cache["predicate_ids"],
                    dump_cache["predicate_texts"],
                )
            ]
            dump_signals_csv(filename, ['signal', 'pred_idx', 'pred_str'], rows)
        
        return
