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

# In this game, there is one asker (Alex) and one retriever (Beth).
# They are both trained to maximise either the probability assigned by Beth to an object (if the object satisfies the predicate), or its opposite (otherwise), in the following context: 
# Alex is shown a predicate and produces a signal, Beth sees both the signal and an object, and produces a probability.
# Alex is trained with REINFORCE; Beth is trained by log-likelihood maximization.
class AlexBeth(Game):
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
        # Compositionality probe (biLSTM->LSTM seq2seq): measured during evaluation when enabled.
        self.eval_compositionality = getattr(args, "eval_compositionality", False)
        self._comp_hparams = compositionality.hparams_from_args(args) if self.eval_compositionality else None
        self._comp_seed = getattr(args, "comp_search_seed", 0)
        self.epochs = getattr(args, "epochs", None)
        # Negation metrics only run when negation exists.
        self.no_negation = getattr(args, "no_negation", False)
        # "Curriculum learning"; if not None, then some accuracy threshold in [0,1] unlocks it
        self.curriculum_negation_acc = getattr(args, "curriculum_negation", None)
        self._curriculum_unlocked = (self.curriculum_negation_acc is None) or self.no_negation
        self._curriculum_unlock_epoch = None
        self._beth_reaper_step = getattr(args, "beth_reaper_step", None)
        self._current_epoch = 0
        if((self.curriculum_negation_acc is not None) and (not self.no_negation)):
            self._dataset.use_positive_predicates_only()
            print(
                f"[curriculum] positive-only phase enabled "
                f"(unlock_acc={self.curriculum_negation_acc})"
            )
        # Used to decide whether to dump signals during a hike in performance
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

    # Beth reaper.
    def start_epoch(self, data_iterator, summary_writer):
        super().start_epoch(data_iterator, summary_writer)

        if((self._beth_reaper_step is not None) and (self._current_epoch != 0) and ((self._current_epoch % self._beth_reaper_step) == 0)):
            self.retriever.reinitialize()
            for p in self.retriever.parameters(): self._optim.state.pop(p, None)

            print(f"[beth-reaper] epoch {self._current_epoch}: retriever reinitialized.")

        self._current_epoch += 1
    
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
            "eval/signal_length",
            "eval/vocab_used",
            "eval/compositionality",
            "eval/c.e._verify",
            "eval/c.e._falsify",
            "eval/scrambling-resistance",
            "eval/neg_consistency",
            "eval/curriculum_unlocked",
            "eval/topsim_extensional_norm_levenshtein",
            "eval/topsim_extensional_multi_jaccard",
            "eval/topsim_intensional_norm_levenshtein",
            "eval/topsim_intensional_multi_jaccard",
            "eval/topsim_polarity_norm_levenshtein"
        ]

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
    # folds of each fold's best held-out exact-match rate (a float in [0, 1]).
    def _compute_compositionality(self):
        device = next(self.asker.parameters()).device
        pairs, spec = compositionality.emergent_pairs(self.asker, self._dataset, device)
        return compositionality.compositionality(pairs, spec, self._comp_hparams, device, seed=self._comp_seed)

    # Called at the end of each training epoch.
    # data_loader: Dataset
    # epoch_index: int
    @torch.no_grad()
    def evaluate(self, data_loader, epoch_index):
        def log(name, value):
            self.autologger._write(name, value, epoch_index, direct=True)
            if(self.autologger.display != 'minimal'): print(f'{name}\t{value}')

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
        total_signal_length = 0.0
        # Communication-efficiency split by truth label (model can be ex. good at positives and bad at negatives).
        total_verify_ce = 0.0 # verify: predicate true for candidate
        total_falsify_ce = 0.0 # falsify: predicate false for candidate.
        total_verify_items = 0 # denominators for ratios
        total_falsify_items = 0
        # Scrambling resistance = preserved correctness after shuffling / original correctness.
        perf_scrambled = 0.0
        perf_baseline = 0.0
        # Mean negation consistency over matched (p, ¬p) rows.
        neg_consistency_total = 0.0
        neg_consistency_count = 0
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
            total_signal_length += signal_length * batch_items

            # Splits candidate decisions into verify/falsify subsets.
            verify_mask = (truth_targets > 0.5)
            falsify_mask = ~verify_mask
            if(verify_mask.any()):
                # c.e._verify: average P(true) on true instances.
                total_verify_ce += dist.probs[verify_mask].sum().item()
                total_verify_items += int(verify_mask.sum().item())
            if(falsify_mask.any()):
                # c.e._falsify: average P(false)=1-P(true) on false instances.
                total_falsify_ce += (1.0 - dist.probs[falsify_mask]).sum().item()
                total_falsify_items += int(falsify_mask.sum().item())

            # Scrambles the symbol order inside signals and recompute correctness probs on these.
            # Then checks how much original performance is kept.
            # If retention is high, semantics likely rely less on order/composition.
            if(self.run_fancy_lang_eval):
                batch_signals = asker_outcome.action[0].detach().clone()
                batch_lens = asker_outcome.action[1].detach().clone()
                scrambled_signals = batch_signals.clone()
                for i in range(scrambled_signals.size(0)):
                    signal_len = int(batch_lens[i].item())
                    if(signal_len > 1):
                        scrambled_signals[i, :signal_len] = scrambled_signals[i, :signal_len][torch.randperm(signal_len)]

                scrambled_outcome = self.retriever(self._beth_input(batch), signal=scrambled_signals, length=batch_lens)
                scrambled_probs = torch.sigmoid(scrambled_outcome.scores)

                scrambled_correct_prob = torch.where(truth_targets > 0.5, scrambled_probs, 1.0 - scrambled_probs)

                # Limits at min(original, scrambled) so scrambling does not increase score.
                perf_scrambled += torch.minimum(correct_prob, scrambled_correct_prob).sum().item()
                perf_baseline += correct_prob.sum().item()

            # Negation consistency: Is it the case that for any predicate p and candidate c, P(p true | c) + P(¬p true | c) = 1?
            # Consistency for (p, c): cons(p, c) = 1 - |P(p true | c ) + P(¬p true | c) - 1|
            # Per batch: avg of cons(p, c) over all (p, c)
            if(self.run_fancy_lang_eval and (not self.no_negation)):
                paired_rows = [] # list[int], row indices (of rows in the batch about a predicate that has or is a negation)
                neg_pred_indices = [] # list[int], predicate indices
                for row_i, pred_i in enumerate(batch.predicate_idx):
                    neg_i = self._predicate_negation_idx.get(int(pred_i))
                    if(neg_i is not None):
                        paired_rows.append(row_i)
                        neg_pred_indices.append(neg_i)

                if(len(paired_rows) > 0):
                    paired_rows = torch.tensor(paired_rows, device=dist.probs.device, dtype=torch.long)
                    
                    p_prob = dist.probs.index_select(dim=0, index=paired_rows) # Shape: (batch size, num candidates)
                    
                    # Generates signals for negated predicates.
                    neg_pred_indices = torch.tensor(neg_pred_indices, device=dist.probs.device, dtype=torch.long)
                    neg_asker_outcome = self.asker(neg_pred_indices)

                    # Collects the candidates used for predicates that have or are a negation.
                    beth_input = self._beth_input(batch)
                    beth_input_subset = {k: v.index_select(dim=0, index=paired_rows) for k, v in beth_input.items()}
                    
                    # Scores candidates based on ¬p signals.
                    neg_retriever_outcome = self.retriever(beth_input_subset, *neg_asker_outcome.action)
                    not_p_prob = torch.sigmoid(neg_retriever_outcome.scores) # Shape: (batch size, num candidates)

                    neg_consistency = 1.0 - torch.abs((p_prob + not_p_prob) - 1.0) # Shape: (batch size, num candidates)

                    # Accumulates sum and count.
                    neg_consistency_total += neg_consistency.sum().item()
                    neg_consistency_count += neg_consistency.numel()

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
        eval_signal_length = total_signal_length / total_items
        if(total_vocab_counts is None): eval_vocab_used = 0
        else: eval_vocab_used = int((total_vocab_counts > 0).sum().item())
        
        log('eval/retriever_loss', eval_retriever_loss)
        log('eval/perf', eval_perf)
        log('eval/accuracy', eval_accuracy)
        log('eval/retriever_entropy', eval_retriever_entropy)
        log('eval/signal_length', eval_signal_length)  # Average number of symbols Alex produced.
        log('eval/vocab_used', eval_vocab_used)

        # Compositionality: can a generic seq2seq learner recover each predicate (in Polish
        # notation) from its emergent signal? Reported as the 5-fold-CV exact-match rate.
        compositionality_score = float("nan")
        if(self.eval_compositionality):
            compositionality_score = self._compute_compositionality()
            log('eval/compositionality', compositionality_score)
        
        avg_accuracy = eval_accuracy
        if(avg_accuracy > self.max_perf): self.max_perf = avg_accuracy

        is_perf_hike = False
        # Curriculum: start with positive-only predicates, unlock all once eval accuracy reaches threshold.
        if((self.curriculum_negation_acc is not None) and (not self._curriculum_unlocked)):
            if(eval_accuracy >= self.curriculum_negation_acc):
                data_loader.use_all_predicates()
                self._curriculum_unlocked = True
                self._curriculum_unlock_epoch = epoch_index
                is_perf_hike = True
                print(
                    f"[curriculum] unlocked all predicates at epoch {epoch_index} "
                    f"(eval/accuracy={eval_accuracy:.6f})"
                )
        curriculum_unlocked = float((self._curriculum_unlock_epoch is not None) and (epoch_index > self._curriculum_unlock_epoch))
        log('eval/curriculum_unlocked', curriculum_unlocked)

        # Fancy metrics
        verify_ratio = None
        falsify_ratio = None
        scrambling_ratio = None
        neg_consistency_ratio = (float("nan") if(self.no_negation) else None)
        topsim_ext_levenshtein = float("nan")
        topsim_ext_jaccard = float("nan")
        topsim_int_levenshtein = float("nan")
        topsim_int_jaccard = float("nan")
        topsim_polarity_norm_levenshtein = float("nan")
        if(self.run_fancy_lang_eval):
            verify_ratio = 0
            falsify_ratio = 0
            if(total_verify_items > 0): verify_ratio = total_verify_ce / total_verify_items
            if(total_falsify_items > 0): falsify_ratio = total_falsify_ce / total_falsify_items
            log('eval/c.e._verify', verify_ratio)
            log('eval/c.e._falsify', falsify_ratio)

            scrambling_ratio = 0
            if(perf_baseline > 0.0): scrambling_ratio = perf_scrambled / perf_baseline
            log('eval/scrambling-resistance', scrambling_ratio)
            # Only meaningful when negation predicates exist.
            if(not self.no_negation): 
                neg_consistency_ratio = 0
                if(neg_consistency_count > 0): 
                    neg_consistency_ratio = neg_consistency_total / neg_consistency_count
                log('eval/neg_consistency', neg_consistency_ratio)

            # Topographic similarity
            if((eval_cache is not None) and (len(eval_cache["signals"]) > 1)):
                num_predicates = len(self._dataset.predicates)
                sample_size = int(min(1024, max(128, 12 * np.sqrt(max(1, num_predicates)))))
                # sample = [([s0, s1], id0), ([s2], id1), ...]
                sample = list(zip(eval_cache["signals"], eval_cache["predicate_ids"]))
                random.shuffle(sample)
                sample = sample[:sample_size]
                # Consider only unique (predicate, signal) pairs.
                sample_type = []
                seen_pairs = set()
                for signal, pid in sample:
                    key = (int(pid), tuple(signal))
                    if(key in seen_pairs):
                        continue
                    seen_pairs.add(key)
                    sample_type.append((signal, int(pid)))
                sample = sample_type
                # sample_signals = [(s0,s1), (s2,), (s0,s3), ...]
                sample_signals = [tuple(s) for (s, _) in sample]
                sample_pred_ids = [int(pid) for (_, pid) in sample]
                # signed-literal representation (polarity topsim)
                if(not self.no_negation):
                    sample_signed_literals = [self._polarity_repr(self._dataset.predicates[pid]) for pid in sample_pred_ids]

                # Extensional meaning is expressed as a binary vector over candidate IDs
                # sample_cand_vec = [(1,1,0,0), (0,0,1,1), ...]
                sample_cand_vecs = []
                # For each predicate build meaning vector
                for _, pid in sample:
                    pred = self._dataset.predicates[int(pid)]
                    # candidate_vec[k] = 1 if predicate verifies candidate_k, 0 otherwise
                    candidate_vec = tuple(1 if(pred.check(cand) == 1) else 0 for cand in self._topsim_candidates)
                    sample_cand_vecs.append(candidate_vec)

                # Topographic similarity:
                # extensional: predicate meaning expressed as the binary vector of whether a candidate satisfies it
                # intensional: the meaning is the embedding ("what the robots think")
                # Levenshtein vs. Jaccard: order-dependent vs. invariant
                # report 2 x 2
                # Polarity (Levenshtein): topsim over signed literals
                # Levenshtein is normalised to account for varying signal length
                if((len(set(sample_signals)) > 1) and (len(set(sample_cand_vecs)) > 1)):
                    # Intensional topsim: meaning distance is cosine distance between predicate embeddings.
                    asker_device = next(self.asker.parameters()).device
                    pred_idx_tensor = torch.tensor(sample_pred_ids, dtype=torch.long, device=asker_device)
                    # sender_vecs: (n, d) predicate embeddings for intensional distance.
                    sender_vecs = self.asker.predicate_encoder(pred_idx_tensor).cpu().numpy()
                    # Compute one condensed pairwise vector per distance notion.
                    # signal_strings: length-n list of signal strings for Levenshtein.
                    signal_strings = [''.join(map(chr, signal)) for signal in sample_signals]
                    n = len(sample_signals)
                    # Pairwise distance vectors built only on non-identical (predicate, signal) pairs.
                    signal_lev_d = []
                    ext_d = []
                    int_d = []
                    pol_d = [] if(not self.no_negation) else None
                    signal_jac_d = [] if(self.use_jaccard_eval) else None

                    for i in range(n - 1):
                        sig_i_str = signal_strings[i]
                        sig_i = sample_signals[i]
                        pid_i = sample_pred_ids[i]
                        ext_i = sample_cand_vecs[i]
                        vec_i = sender_vecs[i]
                        for j in range(i + 1, n):
                            # Keep one of each pair type.
                            if((pid_i == sample_pred_ids[j]) and (sig_i == sample_signals[j])):
                                continue
                            # signal_lev_d: normalized Levenshtein (order-sensitive, normalize by length).
                            signal_lev_d.append(compute_correlation.levenshtein_normalised(sig_i_str, signal_strings[j]))
                            # ext_d: Hamming distance over candidate truth vectors
                            # Hamming is equally sensitive to verify and falsify.
                            ext_d.append(sum(int(a != b) for a, b in zip(ext_i, sample_cand_vecs[j])))
                            # int_d: cosine distance between predicate embeddings (intensional meaning).
                            int_d.append(float(scipy.spatial.distance.cosine(vec_i, sender_vecs[j])))
                            # pol_d: symmetric difference over signed literals (equiv. Hamming over binary).
                            if(not self.no_negation):
                                pol_d.append(float(len(sample_signed_literals[i] ^ sample_signed_literals[j])))
                            if(self.use_jaccard_eval):
                                # multiset Jaccard over token sequences (order-invariant).
                                signal_jac_d.append(compute_correlation.jaccard(sig_i, sample_signals[j]))

                    signal_lev_d = np.asarray(signal_lev_d, dtype=float)
                    ext_d = np.asarray(ext_d, dtype=float)
                    int_d = np.asarray(int_d, dtype=float)
                    if(not self.no_negation):
                        pol_d = np.asarray(pol_d, dtype=float)
                    if(self.use_jaccard_eval):
                        signal_jac_d = np.asarray(signal_jac_d, dtype=float)

                    def _safe_spearman(x, y):
                        # Spearman correlation on pairwise distance vectors; NaN if degenerate.
                        if((x.size == 0) or (y.size == 0)):
                            return float("nan")
                        if(np.all(x == x[0]) or np.all(y == y[0])):
                            return float("nan")
                        return float(scipy.stats.spearmanr(x, y).correlation)

                    topsim_ext_levenshtein = _safe_spearman(signal_lev_d, ext_d)
                    topsim_int_levenshtein = _safe_spearman(signal_lev_d, int_d)
                    if(not self.no_negation):
                        topsim_polarity_norm_levenshtein = _safe_spearman(signal_lev_d, pol_d)
                    if(self.use_jaccard_eval):
                        topsim_ext_jaccard = _safe_spearman(signal_jac_d, ext_d)
                        topsim_int_jaccard = _safe_spearman(signal_jac_d, int_d)

                    log('eval/topsim_extensional_norm_levenshtein', topsim_ext_levenshtein)
                    log('eval/topsim_intensional_norm_levenshtein', topsim_int_levenshtein)
                    if(not self.no_negation):
                        log('eval/topsim_polarity_norm_levenshtein', topsim_polarity_norm_levenshtein)
                    if(self.use_jaccard_eval):
                        log('eval/topsim_extensional_multi_jaccard', topsim_ext_jaccard)
                        log('eval/topsim_intensional_multi_jaccard', topsim_int_jaccard)
                else:
                    log('eval/topsim_extensional_norm_levenshtein', topsim_ext_levenshtein)
                    log('eval/topsim_intensional_norm_levenshtein', topsim_int_levenshtein)
                    if(not self.no_negation):
                        log('eval/topsim_polarity_norm_levenshtein', topsim_polarity_norm_levenshtein)
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
                "eval/signal_length": float(eval_signal_length),
                "eval/vocab_used": eval_vocab_used,
                "eval/compositionality": float(compositionality_score),
                "eval/c.e._verify": verify_ratio,
                "eval/c.e._falsify": falsify_ratio,
                "eval/scrambling-resistance": scrambling_ratio,
                "eval/neg_consistency": neg_consistency_ratio,
                "eval/curriculum_unlocked": curriculum_unlocked,
                "eval/topsim_extensional_norm_levenshtein": topsim_ext_levenshtein,
                "eval/topsim_extensional_multi_jaccard": topsim_ext_jaccard,
                "eval/topsim_intensional_norm_levenshtein": topsim_int_levenshtein,
                "eval/topsim_intensional_multi_jaccard": topsim_int_jaccard,
                "eval/topsim_polarity_norm_levenshtein": topsim_polarity_norm_levenshtein,
            }
            missing = [k for k, v in row.items() if(k != "epoch" and v is None)]
            if(missing):
                raise RuntimeError(
                    "Missing eval metrics for epoch "
                    f"{epoch_index}: {', '.join(missing)}. "
                    "Enable fancy eval with --dump_eval_metrics."
                )
            self._eval_metrics_rows.append(row)

        # Decide if there is a performance hike.
        def _min_jump(best):
            if(best < 0.50): return 0.10
            if(best < 0.70): return 0.05
            if(best < 0.90): return 0.01
            if(best < 0.95): return 0.005
            return 0.001
        
        if(self.dump_signal_mode in ('when_hike', 'when_hike_strict')):
            if((self._curriculum_unlock_epoch is not None) and (epoch_index == (self._curriculum_unlock_epoch + 1))):
                # Reset performance baseline.
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
                 or ((self._curriculum_unlock_epoch is not None) and (epoch_index == (self._curriculum_unlock_epoch + 1)))
                 or (epoch_index == (self.epochs - 1))))
            )):
            filename = os.path.join(self.signal_dump_dir, f"signals.e{epoch_index}.csv")
            with open(filename, 'w') as ostr:
                writer = csv.writer(ostr)
                _ = writer.writerow(['signal', 'pred_idx', 'pred_str'])
                for signal, pred_idx, pred_text in zip(
                    dump_cache["signals"],
                    dump_cache["predicate_ids"],
                    dump_cache["predicate_texts"],
                ):
                    signal = ' '.join(map(str, signal))
                    row = [signal, pred_idx, pred_text]
                    _ = writer.writerow(row)
        
        return
