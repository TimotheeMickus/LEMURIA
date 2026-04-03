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

from .game import Game

# In this game, there is one asker (Alex) and one retriever (Beth).
# They are both trained to maximise either the probability assigned by Beth to an object (if the object satisfies the predicate), or its opposite (otherwise), in the following context: 
# Alex is shown a predicate and produces a signal, Beth sees both the signal and an object, and produces a probability.
# Alex is trained with REINFORCE; Beth is trained by log-likelihood maximization.
class AlexBeth(Game):
    def __init__(self, args, logger, dataset, message_dump_dir):
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
        
        self.max_len_msg = args.max_len

        self._optim = build_optimizer(parameters, args.learning_rate)
        
        self.use_baseline = args.use_baseline
        if(self.use_baseline): # In that case, the loss will take into account the "baseline term" into the average recent reward.
            # Currently, the asker and retriever's rewards are the same, but we could imagine a setting in which they are different.
            self._asker_avg_reward = misc.Averager(size=12800)
            self._retriever_avg_reward = misc.Averager(size=12800)

        self.dump_message_mode = getattr(args, "dump_message", None)
        self.dump_predicate_perf = getattr(args, "dump_predicate_perf", False)
        self.dump_eval_metrics_enabled = getattr(args, "dump_eval_metrics", False)
        # Fancy language eval is only needed for eval metrics.
        self.run_fancy_lang_eval = bool(self.dump_eval_metrics_enabled)
        self.correct_only = args.correct_only # Whether to perform the fancy language evaluation using only correct messages (i.e., the one that leads to successful communication).
        self.epochs = getattr(args, "epochs", None)
        # Negation metrics only run when negation exists.
        self.no_negation = getattr(args, "no_negation", False)
        # Used to decide whether to dump messages during a hike in performance
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

        if self.dump_eval_metrics_enabled and (not self.run_fancy_lang_eval):
            raise ValueError("--dump_eval_metrics requires fancy eval metrics; enable --dump_message.")
        
        self.debug = args.debug
        self.message_dump_dir = message_dump_dir # str|None

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

    def _build_negation_correspondence(self):
        '''
        Returns a dictionary of {predicate index: index of its negation}.
        Builds pairs of the indices of predicates in `self._dataset.predicates`.
        This works both ways: if p[i]=¬p and p[j]=p, store both i: j and j: i.
            partner: dict[int, int]
        '''
        partner = {}
        # Map predicates to their indices
        # This should perhaps be a feature of the Dataset
        pred2idx = {pred: i for i, pred in enumerate(self._dataset.predicates)}
        # For each negative predicate and its index, Negation stores its positive "base"
        for i, pred in enumerate(self._dataset.predicates):
            if isinstance(pred, predicate_data.Negation):
                # Map the negation and the base to each other
                j = pred2idx[pred.predicate]
                partner[i] = j
                partner[j] = i
        return partner

    def dump_predicate_performance(self, output_dir, wandb_run=None, artifact_name=None):
        # Save one raw row-level table at the end of the run.
        if (not self.dump_predicate_perf) or (len(self._predicate_perf_rows) == 0):
            return

        os.makedirs(output_dir, exist_ok=True)
        rows_path = os.path.join(output_dir, "predicate_perf_rows.csv")

        with open(rows_path, "w") as ostr:
            writer = csv.writer(ostr)
            writer.writerow(["epoch", "pred_idx", "pred_str", "row_perf", "row_acc"])
            for epoch, pred_idx, row_perf, row_acc in self._predicate_perf_rows:
                writer.writerow([epoch, pred_idx, self._predicate_text_by_idx.get(pred_idx, ""), row_perf, row_acc])

        if wandb_run is not None:
            import wandb
            artifact = wandb.Artifact(name=f"predicate-performance-{wandb_run.id}", type="analysis")
            artifact.add_file(rows_path)
            wandb_run.log_artifact(artifact)

    def dump_eval_metrics(self, output_dir, wandb_run=None):
        # Save one epoch-level table at the end of the run.
        if (not self.dump_eval_metrics_enabled) or (len(self._eval_metrics_rows) == 0):
            return

        os.makedirs(output_dir, exist_ok=True)
        rows_path = os.path.join(output_dir, "eval_metrics_rows.csv")
        fieldnames = [
            "epoch",
            "eval/loss",
            "eval/accuracy",
            "eval/perf",
            "eval/retriever_entropy",
            "eval/msg_length",
            "eval/vocab_used",
            "eval/c.e._verify",
            "eval/c.e._falsify",
            "eval/scrambling-resistance",
            "eval/neg_consistency",
            "eval/topsim_extensional_levenshtein",
            "eval/topsim_extensional_jaccard",
            "eval/topsim_intensional_levenshtein",
            "eval/topsim_intensional_jaccard",
        ]

        with open(rows_path, "w") as ostr:
            writer = csv.DictWriter(ostr, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._eval_metrics_rows)

        if wandb_run is not None:
            import wandb
            artifact = wandb.Artifact(name=f"eval-metrics-{wandb_run.id}", type="analysis")
            artifact.add_file(rows_path)
            wandb_run.log_artifact(artifact)
    
    # The name is misleading (reflects an older version): this only converts truth to a tensor on the device
    def _compute_truth_targets(self, batch):
        """
        Returns a float tensor of shape (batch size,) where 1.0 denotes that the
        predicate holds for the candidate, and 0.0 otherwise.
        """
        return batch.candidate_truth
    
    def agents_for_CNN_pretraining(self):
        raise NotImplementedError # In fact, the method should not even exist (the superclass should be modified).

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
        if torch.isnan(loss): # DEBUG
            print(f"[warn] loss is {loss}")
        optimization = [(self._optim, loss.detach(), misc.get_backward_f(loss))]

        msg_length = asker_outcome.action[1].float().mean()

        metrics = {
            "rewards": asker_rewards,
            "successes": asker_perf,
            "msg_length": msg_length,
            "sender_entropy": asker_entropy,
            "receiver_entropy": retriever_entropy,
        }

        return optimization, metrics

    # Returns two tensors of shape (batch size).
    # asker_action: pair (message, length) where message is a tensor of shape (batch size, max message length) and length a tensor of shape (batch size)
    # retriever_scores: logits pair (batch, num_candidates)
    # truth_targets: (batch, num_candidates)
    def compute_asker_rewards(self, asker_action, retriever_scores, truth_targets):
        """
        Returns reward and performance tensors (both shaped [batch size]) 
        based on the probability Beth assigns to the correct truth value.
        """
        logits = retriever_scores # Shape: (batch, num_candidates)
        probs = torch.sigmoid(logits) # Shape: (batch, num_candidates)
        correct_prob = torch.where(truth_targets > 0.5, probs, 1.0 - probs) # Shape: (batch, num_candidates)
        perf = correct_prob.mean(dim=1).detach() # Shape: (batch,)

        if(self.use_expectation):
            rewards = perf.clone() # Expected average accuracy of the retriever over the sequence. # Shape: (batch,)
        else:
            rewards = torch.bernoulli(correct_prob).mean(dim=1).detach() # We sample whether the retriever is right according to the probability of the retriever being right; the reward is 1 when the retriever is right, 0 otherwise. # Shape: (batch,)

        msg_lengths = asker_action[1].view(-1).float() # Shape: (batch,), includes the EOS symbol (usualy 0).

        rewards += -1 * (msg_lengths >= self.max_len_msg) # Penalty related to messages exceeding the length limit.

        if(self.len_penalty > 0.0):
            # The penalty equals 0 when `len_penalty` is set to 0, and increases (the faster the higher `len_penalty` is) to 1 with the length of the message otherwise.
            # RMK: We could imagine a non-uniform penalty (that depends on the position of the token for symbols ≠ EOS).
            length_penalties = 1.0 - (1.0 / (1.0 + self.len_penalty * msg_lengths)) # Shape: (batch,)

            rewards = (rewards - length_penalties) # Shape: (batch,)

        if(self.voc_penalty > 0.0):
            # Each symbol of the base alphabet is associated with a total penalty equal to `batch_size` * `voc_penalty`, distributed over all messages in proportion of their use of the symbol (as if the total penalty were distributed equally over all occurrences of the symbol).
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

    # Returns (loss, perf, rewards) where loss is scalar and perf/rewards are (batch,)
    # asker_outcome: (log_prob of (batch, max_msg_len), entropy (batch, 1))
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
        logits = retriever_scores # Shape: (batch, num_candidates)
        probs = torch.sigmoid(logits) # Shape: (batch, num_candidates)

        loss = F.binary_cross_entropy_with_logits(logits, truth_targets.float())

        eps = 1e-8
        entropy = (-(probs * torch.log(probs + eps) + (1.0 - probs) * torch.log(1.0 - probs + eps))).mean()
        # entropy penalty (addition)
        loss += (self.beta_retriever * entropy)

        perf = torch.where(truth_targets > 0.5, probs, 1.0 - probs).detach() # Shape: (batch, num_candidates)

        if return_entropy: return (loss, perf, entropy)
        return (loss, perf)

    # Called at the end of each training epoch.
    @torch.no_grad()
    def evaluate(self, data_loader, epoch_index):
        def log(name, value):
            self.autologger._write(name, value, epoch_index, direct=True)
            if(self.autologger.display != 'minimal'): print(f'{name}\t{value}')

        # Use the dataset batch size but cap the number of batches to keep eval fast.
        batch_size = data_loader.batch_size
        max_batches = 128
        # TODO: consider stratified eval with a fixed number of occurrences per predicate
        # (e.g., 8 each) instead of random nb_batch sampling for more uniform coverage.
        nb_batch = max(1, min(max_batches, (2 ** 15) // max(1, batch_size)))

        # Epoch-level totals (we aggregate batch by batch, then normalize once at the end).
        total_items = 0
        total_loss = 0.0
        total_accuracy = 0.0 # average accuracy (computed from success~1, failure~0).
        total_entropy = 0.0
        total_msg_length = 0.0
        total_perf = 0.0
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

        # Cache for message dumps.
        dump_cache = None
        if self.dump_message_mode:
            dump_cache = {
                "messages": [],
                "predicate_ids": [],
                "predicate_texts": [],
                "candidate_texts": []
            }

        # Cache for language-level eval metrics.
        eval_cache = None
        if self.run_fancy_lang_eval:
            eval_cache = dump_cache if dump_cache is not None else {
                "messages": [],
                "predicate_ids": [],
                "predicate_texts": [],
                "candidate_texts": []
            }

        iterator = range(nb_batch)
        if(self.autologger.display == 'tqdm'):
            iterator = tqdm.tqdm(iterator, desc='Eval.')

        for _ in iterator:
            self.start_episode(train_episode=False)
            
            batch = data_loader.get_batch(size=batch_size, data_type='test')

            asker_outcome, retriever_outcome = self.alex_to_beth(batch)
            truth_targets = self._compute_truth_targets(batch) # Shape: (batch, n_candidates)

            # Beth outputs a logit per candidate; we interpret it as the log-odds that the predicate holds for the candidate.
            logits = retriever_outcome.scores # Shape: (batch, num_candidates)
            probs = torch.sigmoid(logits) # Shape: (batch, num_candidates)
           
            num_candidates = retriever_outcome.scores.shape[-1]
            predicate_idx = batch.predicate_idx.repeat_interleave(num_candidates).cpu().numpy() # Shape: (batch * num_candidates)
            failure = ((1.0 - truth_targets) * probs + truth_targets * (1.0 - probs)).view(-1).cpu().numpy() # Shape: (batch * num_candidates)
            data_loader.failure_based_distribution.update(predicate_idx, failure)

            # Scalar BCE averaged over the batch (used for logging only).
            loss = F.binary_cross_entropy_with_logits(logits, truth_targets, reduction='mean').item()
            # Accuracy is the thresholded probability vs. the binary target.
            preds = (probs >= 0.5).float() # Shape: (batch, num_candidates)
            accuracy = (preds == truth_targets).float().mean().item() # per candidate (not predicate)
            # `perf` measures how much probability mass Beth assigns to the correct truth value.
            correct_prob = torch.where(truth_targets > 0.5, probs, 1.0 - probs)
            perf = correct_prob.mean().item()

            # Entropy of Beth's Bernoulli output; useful to detect collapsed predictions. TODO Is this really useful?
            entropy = (-(probs * torch.log(probs + 1e-8) + (1.0 - probs) * torch.log(1.0 - probs + 1e-8))).mean().item() # TODO Instead of adding 1e-8 factors, use something like torch.where((a != 0), (a * b), 0.).
            # Average symbol count for Alex's message in this batch.
            msg_length = asker_outcome.action[1].float().mean().item()
            # Vocabulary usage: count symbols used in signals (excluding EOS and padding).
            msg_tokens = asker_outcome.action[0]
            msg_lens = asker_outcome.action[1].int().view(-1)
            max_len = msg_tokens.size(1)
            positions = torch.arange(max_len, device=msg_tokens.device).unsqueeze(0)
            # True for positions strictly before EOS in each signal.
            in_message = positions < (msg_lens.unsqueeze(1) - 1)
            if in_message.any():
                used_tokens = msg_tokens[in_message]
                vocab_counts = torch.bincount(used_tokens, minlength=self.full_alphabet_size).to("cpu")
                vocab_counts[self.asker.eos_index] = 0
                vocab_counts[self.asker.padding_idx] = 0
                if total_vocab_counts is None:
                    total_vocab_counts = vocab_counts
                else:
                    total_vocab_counts += vocab_counts

            # Store row-level performance for predicate diagnostics only when requested.
            if self.dump_predicate_perf:
                row_perf = correct_prob.mean(dim=1).detach().cpu().tolist()
                row_acc = (preds == truth_targets).float().mean(dim=1).detach().cpu().tolist()
                for i, pred_idx in enumerate(batch.predicate_idx):
                    pred_idx = int(pred_idx)
                    if pred_idx not in self._predicate_text_by_idx:
                        self._predicate_text_by_idx[pred_idx] = str(batch.predicate[i])
                    self._predicate_perf_rows.append((int(epoch_index), pred_idx, float(row_perf[i]), float(row_acc[i])))

            batch_items = truth_targets.numel()
            total_items += batch_items
            total_loss += loss * batch_items
            total_accuracy += accuracy * batch_items
            total_entropy += entropy * batch_items
            total_msg_length += msg_length * batch_items
            total_perf += perf * batch_items

            # Split candidate decisions into verify/falsify subsets.
            verify_mask = (truth_targets > 0.5)
            falsify_mask = ~verify_mask
            if verify_mask.any():
                # c.e._verify: average P(true) on true instances.
                total_verify_ce += probs[verify_mask].sum().item()
                total_verify_items += int(verify_mask.sum().item())
            if falsify_mask.any():
                # c.e._falsify: average P(false)=1-P(true) on false instances.
                total_falsify_ce += (1.0 - probs[falsify_mask]).sum().item()
                total_falsify_items += int(falsify_mask.sum().item())

            # Scramble the symbol order inside signals and recompute correctness probs on these.
            # Then we check how much original performance is kept.
            # If retention is high, semantics likely rely less on order/composition.
            if self.run_fancy_lang_eval:
                batch_messages = asker_outcome.action[0].detach().clone()
                batch_lens = asker_outcome.action[1].detach().clone()
                scrambled_messages = batch_messages
                for i in range(scrambled_messages.size(0)):
                    msg_len = int(batch_lens[i].item())
                    if msg_len > 1:
                        scrambled_messages[i, :msg_len] = scrambled_messages[i, :msg_len][torch.randperm(msg_len)]

                scrambled_outcome = self.retriever(self._beth_input(batch), message=scrambled_messages, length=batch_lens)
                scrambled_probs = torch.sigmoid(scrambled_outcome.scores)

                scrambled_correct_prob = torch.where(truth_targets > 0.5, scrambled_probs, 1.0 - scrambled_probs)

                # Limit at min(original, scrambled) so scrambling does not increase score.
                perf_scrambled += torch.minimum(correct_prob, scrambled_correct_prob).sum().item()
                perf_baseline += correct_prob.sum().item()

            # Negation consistency: for predicate p and its negation ¬p on the same candidate, P(p=true) + P(¬p=true) should be close to 1.
            # In particular: in the current evaluation batch select rows with predicate p
            # Then run the model on the same candidates showing the model ¬p
            # Alex generates a signal for ¬p, Beth outputs P(¬p=true | signal_¬p, candidate)
            # Consistency per item: c = 1 - |P(p=true) + P(¬p=true) - 1|
            # Per batch: C = 1/N * (sum for i=1 to N) of c(x_i)
            if self.run_fancy_lang_eval and (not self.no_negation):
                paired_rows = []
                neg_pred_indices = []
                # Store idx for Batch rows and their negations if exist
                # inb4: Rows already contain P(p=true) probabilities
                for row_i, pred_i in enumerate(batch.predicate_idx):
                    neg_i = self._predicate_negation_idx.get(int(pred_i))
                    if neg_i is not None:
                        paired_rows.append(row_i)
                        neg_pred_indices.append(neg_i)

                if len(paired_rows) > 0:
                    # Build tensor of row positions in current batch and aligned negated predicate ids
                    row_idx = torch.tensor(paired_rows, device=probs.device, dtype=torch.long)
                    neg_pred_idx = torch.tensor(neg_pred_indices, device=probs.device, dtype=torch.long)

                    # Generate signals for negated predicates
                    neg_asker_outcome = self.asker(neg_pred_idx)
                    # From candidate tensors for this batch, keep only rows that were paired
                    beth_input = self._beth_input(batch)
                    beth_input_subset = {k: v.index_select(0, row_idx) for k, v in beth_input.items()}
                    # Score candidates based on ¬p signals and convert to probabilities
                    neg_retriever_outcome = self.retriever(beth_input_subset, *neg_asker_outcome.action)
                    p_true_given_not_p = torch.sigmoid(neg_retriever_outcome.scores)
                    # Original probabilities of paired rows
                    p_true_given_p = probs.index_select(0, row_idx)

                    neg_consistency = 1.0 - torch.abs((p_true_given_p + p_true_given_not_p) - 1.0)
                    # Accumulate sum and count: C = 1/N * (sum for i=1 to N) of c(x_i)
                    neg_consistency_total += neg_consistency.sum().item() # sum_i c(x_i)
                    neg_consistency_count += neg_consistency.numel() # N

            # Cache signals once so dump and fancy eval can reuse them.
            # If `correct_only` is True, only correct items are cached.
            cache = dump_cache if dump_cache is not None else eval_cache
            if cache is not None:
                batch_messages = asker_outcome.action[0].detach().clone()
                batch_lens = asker_outcome.action[1].detach().clone()
                accuracy_per_item = (preds == truth_targets).float().mean(dim=1) # (batch,) mean across candidates
                for i in range(batch_messages.size(0)):
                    if self.correct_only and not torch.isclose(accuracy_per_item[i], torch.tensor(1.0, device=accuracy_per_item.device)):
                        # not: (accuracy_per_item[i].item() < 0.5):
                        continue # skip low accuracy items
                    # truncate padding away from signals
                    message = batch_messages[i].tolist()[:batch_lens[i].item()]
                    cache["messages"].append(message)
                    cache["predicate_ids"].append(int(batch.predicate_idx[i]))
                    # TODO This was crashing
                    # eval_cache["predicate_texts"].append(str(batch.predicate[i]))
                    # eval_cache["candidate_texts"].append(",".join(str(c) for c in batch.candidate[i]))
                    # ---- and with this it works now:
                    pred_list = getattr(batch, "predicate", None)
                    if pred_list is not None: 
                        cache["predicate_texts"].append(str(pred_list[i]))
                    else: 
                        cache["predicate_texts"].append(str(self._dataset.predicates[int(batch.predicate_idx[i])]))
                    cand_texts = getattr(batch, "candidate_texts", None)
                    if cand_texts is not None: 
                        cache["candidate_texts"].append(cand_texts[i])
                    else:  
                        cand_list = getattr(batch, "candidate", None)
                        if cand_list is not None:
                            cache["candidate_texts"].append(",".join(str(c) for c in cand_list[i]))
                        else:
                            cache["candidate_texts"].append("")
                    

        # --- logging and stdout --- #
        #                            #
        # Normalise the accumulated sums and push them to TensorBoard / stdout.
        eval_loss = total_loss / total_items
        eval_accuracy = total_accuracy / total_items
        eval_perf = total_perf / total_items
        eval_retriever_entropy = total_entropy / total_items
        eval_msg_length = total_msg_length / total_items
        if total_vocab_counts is None:
            eval_vocab_used = 0.0
        else:
            eval_vocab_used = float((total_vocab_counts > 0).sum().item())
        log('eval/loss', eval_loss)
        log('eval/accuracy', eval_accuracy)
        log('eval/perf', eval_perf)
        log('eval/retriever_entropy', eval_retriever_entropy)
        log('eval/msg_length', eval_msg_length)  # Average number of symbols Alex produced.
        log('eval/vocab_used', eval_vocab_used)
        avg_accuracy = eval_accuracy
        if(avg_accuracy > self.max_perf): self.max_perf = avg_accuracy

        # Fancy metrics
        verify_ratio = None
        falsify_ratio = None
        scrambling_ratio = None
        neg_consistency_ratio = (float("nan") if self.no_negation else None)
        topsim_ext_levenshtein = float("nan")
        topsim_ext_jaccard = float("nan")
        topsim_int_levenshtein = float("nan")
        topsim_int_jaccard = float("nan")
        if self.run_fancy_lang_eval:
            verify_ratio = 0
            falsify_ratio = 0
            if total_verify_items > 0: verify_ratio = total_verify_ce / total_verify_items
            if total_falsify_items > 0: falsify_ratio = total_falsify_ce / total_falsify_items
            log('eval/c.e._verify', verify_ratio)
            log('eval/c.e._falsify', falsify_ratio)

            scrambling_ratio = 0
            if perf_baseline > 0.0: scrambling_ratio = perf_scrambled / perf_baseline
            log('eval/scrambling-resistance', scrambling_ratio)
            # Only meaningful when negation predicates exist.
            if not self.no_negation: 
                neg_consistency_ratio = 0
                if neg_consistency_count > 0: 
                    neg_consistency_ratio = neg_consistency_total / neg_consistency_count
                log('eval/neg_consistency', neg_consistency_ratio)

            # Topographic similarity
            if eval_cache is not None and len(eval_cache["messages"]) > 1:
                num_predicates = len(self._dataset.predicates)
                sample_size = int(min(1024, max(128, 12 * np.sqrt(max(1, num_predicates)))))
                # sample = [([s0, s1], id0), ([s2], id1), ...]
                sample = list(zip(eval_cache["messages"], eval_cache["predicate_ids"]))
                random.shuffle(sample)
                sample = sample[:sample_size]
                # sample_signals = [(s0,s1), (s2,), (s0,s3), ...]
                sample_signals = [tuple(s) for (s, _) in sample]
                sample_pred_ids = [int(pid) for (_, pid) in sample]

                # Meaning is expressed as a binary vector over candidate IDs
                # sample_cand_vec = [(1,1,0,0), (0,0,1,1), ...]
                sample_cand_vecs = []
                # For each predicate build meaning vector
                for _, pid in sample:
                    pred = self._dataset.predicates[int(pid)]
                    # candidate_vec[k] = 1 if predicate verifies candidate_k, 0 otherwise
                    candidate_vec = tuple(1 if pred.check(cand) == 1 else 0 for cand in self._topsim_candidates)
                    sample_cand_vecs.append(candidate_vec)

                # Topographic similarity:
                # extensional: predicate meaning expressed as the binary vector of whether a candidate satisfies it
                # intensional: the meaning is the embedding ("what the robots think")
                # Levenshtein vs. Jaccard: order-dependent vs. invariant
                # report 2 x 2
                if len(set(sample_signals)) > 1 and len(set(sample_cand_vecs)) > 1:
                    # Intensional topsim: meaning distance is cosine distance between predicate embeddings.
                    asker_device = next(self.asker.parameters()).device
                    pred_idx_tensor = torch.tensor(sample_pred_ids, dtype=torch.long, device=asker_device)
                    # sender_vecs: (n, d) predicate embeddings for intensional distance.
                    sender_vecs = self.asker.predicate_encoder(pred_idx_tensor).detach().cpu().numpy()
                    # Compute each pairwise distance vector once, then reuse it for all 4 topsims.
                    # This avoids recomputing expensive Jaccard distances multiple times.
                    # signal_strings: length-n list of message strings for Levenshtein.
                    signal_strings = [''.join(map(chr, msg)) for msg in sample_signals]
                    n = len(sample_signals)
                    pair_count = (n * (n - 1)) // 2
                    # Condensed upper-triangle distance vectors.
                    msg_lev_d = np.empty(pair_count, dtype=float)
                    ext_d = np.empty(pair_count, dtype=float)
                    int_d = np.empty(pair_count, dtype=float)
                    # msg_jac_d uses multiset Jaccard over token sequences (order-invariant).
                    msg_jac_d = compute_correlation.pairwise_multiset_jaccard_distances(sample_signals)

                    k = 0
                    for i in range(n - 1):
                        sig_i_str = signal_strings[i]
                        ext_i = sample_cand_vecs[i]
                        vec_i = sender_vecs[i]
                        for j in range(i + 1, n):
                            # msg_lev_d: normalized Levenshtein (order-sensitive, normalize by length).
                            msg_lev_d[k] = compute_correlation.levenshtein_normalised(sig_i_str, signal_strings[j])
                            # ext_d: Hamming distance over candidate truth vectors
                            # Hamming is equally sensitive to verify and falsify.
                            ext_d[k] = sum(int(a != b) for a, b in zip(ext_i, sample_cand_vecs[j]))
                            # int_d: cosine distance between predicate embeddings (intensional meaning).
                            int_d[k] = float(scipy.spatial.distance.cosine(vec_i, sender_vecs[j]))
                            k += 1

                    def _safe_spearman(x, y):
                        # Spearman correlation on pairwise distance vectors; NaN if degenerate.
                        if x.size == 0 or y.size == 0:
                            return float("nan")
                        if np.all(x == x[0]) or np.all(y == y[0]):
                            return float("nan")
                        return float(scipy.stats.spearmanr(x, y).correlation)

                    topsim_ext_levenshtein = _safe_spearman(msg_lev_d, ext_d)
                    topsim_ext_jaccard = _safe_spearman(msg_jac_d, ext_d)
                    topsim_int_levenshtein = _safe_spearman(msg_lev_d, int_d)
                    topsim_int_jaccard = _safe_spearman(msg_jac_d, int_d)

                    log('eval/topsim_extensional_levenshtein', topsim_ext_levenshtein)
                    log('eval/topsim_extensional_jaccard', topsim_ext_jaccard)
                    log('eval/topsim_intensional_levenshtein', topsim_int_levenshtein)
                    log('eval/topsim_intensional_jaccard', topsim_int_jaccard)
                else:
                    log('eval/topsim_extensional_levenshtein', topsim_ext_levenshtein)
                    log('eval/topsim_extensional_jaccard', topsim_ext_jaccard)
                    log('eval/topsim_intensional_levenshtein', topsim_int_levenshtein)
                    log('eval/topsim_intensional_jaccard', topsim_int_jaccard)
                    if self.autologger.display != 'minimal':
                        print('eval/topsim\tnot enough variation in sampled messages/meanings')

                # Decision tree TODO: how easily predicate identity can be recovered from messages.

        if self.dump_eval_metrics_enabled:
            row = {
                "epoch": int(epoch_index),
                "eval/loss": float(eval_loss),
                "eval/accuracy": float(eval_accuracy),
                "eval/perf": float(eval_perf),
                "eval/retriever_entropy": float(eval_retriever_entropy),
                "eval/msg_length": float(eval_msg_length),
                "eval/vocab_used": float(eval_vocab_used),
                "eval/c.e._verify": verify_ratio,
                "eval/c.e._falsify": falsify_ratio,
                "eval/scrambling-resistance": scrambling_ratio,
                "eval/neg_consistency": neg_consistency_ratio,
                "eval/topsim_extensional_levenshtein": topsim_ext_levenshtein,
                "eval/topsim_extensional_jaccard": topsim_ext_jaccard,
                "eval/topsim_intensional_levenshtein": topsim_int_levenshtein,
                "eval/topsim_intensional_jaccard": topsim_int_jaccard,
            }
            missing = [k for k, v in row.items() if (k != "epoch" and v is None)]
            if missing:
                raise RuntimeError(
                    "Missing eval metrics for epoch "
                    f"{epoch_index}: {', '.join(missing)}. "
                    "Enable fancy eval with --dump_eval_metrics."
                )
            self._eval_metrics_rows.append(row)

        #                            #
        # -------------------------- #
        # Decide if there is a performance hike
        is_perf_hike = False

        def _min_jump(best):
                if best < 0.50: return 0.10
                if best < 0.70: return 0.05
                if best < 0.90: return 0.01
                if best < 0.95: return 0.005
                return 0.001
        
        if self.dump_message_mode in ('when_hike', 'when_hike_strict'):
            if self._best_eval_perf is None:
                self._best_eval_perf = eval_perf
            if eval_perf >= 1.0 and eval_perf > self._best_eval_perf:
                is_perf_hike = True
            else:
                delta_min = _min_jump(self._best_eval_perf)
                if self.dump_message_mode == 'when_hike_strict':
                    is_perf_hike = (eval_perf > self._best_eval_perf + delta_min) and (eval_perf > 0.95)
                else:
                    is_perf_hike = (eval_perf > self._best_eval_perf + delta_min)

            if is_perf_hike and eval_perf > self._best_eval_perf:
                self._best_eval_perf = eval_perf
            self._prev_eval_perf = eval_perf
        # Dumps signals into file every epoch or on the last epoch, depending on the flag
        if self.message_dump_dir and dump_cache is not None and (
            self.dump_message_mode == 'all' or 
            (self.dump_message_mode == 'last' and epoch_index == self.epochs - 1) or
            (self.dump_message_mode in ('when_hike', 'when_hike_strict') and is_perf_hike)
            ):
            filename = os.path.join(self.message_dump_dir, f"msgs.e{epoch_index}.csv")
            with open(filename, 'w') as ostr:
                writer = csv.writer(ostr)
                _ = writer.writerow(['msg', 'pred_idx', 'pred_str', 'candidates'])
                for msg, pred_idx, pred_text, cand_text in zip(
                    dump_cache["messages"],
                    dump_cache["predicate_ids"],
                    dump_cache["predicate_texts"],
                    dump_cache["candidate_texts"],
                ):
                    msg = ' '.join(map(str, msg))
                    row = [msg, pred_idx, pred_text, cand_text]
                    _ = writer.writerow(row)
        
        return

    def test_visualize(self, data_loader, learning_rate):
        print("No visualisation defined.")
