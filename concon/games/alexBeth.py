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

        self.base_alphabet_size = args.base_alphabet_size # Number of symbols without special ones (padding, EOS, etc.)
        self.max_len_msg = args.max_len

        self.use_expectation = args.use_expectation
        self.grad_scaling = (args.grad_scaling or 0)
        self.grad_clipping = (args.grad_clipping or 0)
        self.beta_asker = args.beta_asker
        self.beta_retriever = args.beta_retriever
        self.penalty = args.penalty

        self.shared = args.shared # Whether some parameters are shared between Alex and Beth.
        if(self.shared):
            raise NotImplementedError
            askerRetriever = AskerRetriever.from_args(args)

            self._asker = askerRetriever.asker
            self._retriever = askerRetriever.retriever

            parameters = askerRetriever.parameters()
        else:
            self._asker = Asker.from_args(args)
            self._retriever = Retriever.from_args(args)

            parameters = it.chain(self.asker.parameters(), self.retriever.parameters())

        self._optim = build_optimizer(parameters, args.learning_rate)
        
        self.use_baseline = args.use_baseline
        if(self.use_baseline): # In that case, the loss will take into account the "baseline term" into the average recent reward.
            # Currently, the asker and retriever's rewards are the same, but we could imagine a setting in which they are different.
            self._asker_avg_reward = misc.Averager(size=12800)
            self._retriever_avg_reward = misc.Averager(size=12800)

        self.correct_only = args.correct_only # Whether to perform the fancy language evaluation using only correct messages (i.e., the one that leads to successful communication).
        self.dump_message_mode = getattr(args, "dump_message", None)
        self.epochs = getattr(args, "epochs", None)
        
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
    
    # The name is misleading (reflects an older version): this only converts truth to a tensor on the device
    def _compute_truth_targets(self, batch, device):
        """
        Returns a float tensor of shape (batch size,) where 1.0 denotes that the
        predicate holds for the candidate, and 0.0 otherwise.
        """
        return torch.tensor(batch.candidate_truth, dtype=torch.float32, device=device)
    
    def agents_for_CNN_pretraining(self):
        raise NotImplementedError # In fact, the method should not even exist (the superclass should be modified).

    # batch: Batch
    def _alex_input(self, batch):
        predicate_idx = batch.encode_predicates('sparse') # return list/np array of ints
        device = next(self.asker.parameters()).device
        return torch.tensor(predicate_idx, device=device, dtype=torch.long)

    # batch: Batch
    def _beth_input(self, batch):
        if batch.node_idx is None or batch.edge_idx is None or batch.graph_sizes is None:
            batch.tensorize(self._dataset)
        # self.retriever is nn.Module; PyTorch modules don’t expose a .device attribute. (that's why no self.retriever.device)
        device = next(self.retriever.parameters()).device
        return {
            'node_idx': batch.node_idx.to(device, non_blocking=True), 
            'edge_idx': batch.edge_idx.to(device, non_blocking=True), 
            'graph_size': batch.graph_sizes.to(device, non_blocking=True)
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

    def alex_to_beth(self, batch):
        asker = self.asker
        retriever = self.retriever

        asker_outcome = asker(self._alex_input(batch))
        retriever_outcome = retriever(self._beth_input(batch), *asker_outcome.action)

        return asker_outcome, retriever_outcome

    # batch: Batch
    def compute_interaction(self, batch, **kwargs):
        asker_outcome, retriever_outcome = self(batch)
        truth_targets = self._compute_truth_targets(batch, device=retriever_outcome.scores.device)

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

        msg_lengths = asker_action[1].view(-1).float() # Shape: (batch,)
        rewards += -1 * (msg_lengths >= self.max_len_msg) # Penalty related to messages exceeding the length limit.

        if(self.penalty > 0.0):
            length_penalties = 1.0 - (1.0 / (1.0 + self.penalty * msg_lengths)) # Shape: (batch,)
            rewards = rewards - length_penalties # Shape: (batch,)

        return (rewards, perf)

    # Returns (loss, pref, rewards) where loss is scalar and perf/rewards are (batch,)
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
    def evaluate(self, data_iterator, epoch_index):
        def log(name, value):
            self.autologger._write(name, value, epoch_index, direct=True)
            if(self.autologger.display != 'minimal'): print(f'{name}\t{value}')

        # Predicate datasets lack the image-category metadata relied upon by the legacy Alice-Bob evaluation. 
        # When that's the case we switch to a simpler evaluation that reuses the supervised truth labels we've already defined for training.
        if(not hasattr(data_iterator, 'category_idx')):
            # Use the dataset batch size but cap the number of batches to keep eval snappy.
            batch_size = data_iterator.batch_size
            max_batches = 128
            nb_batch = max(1, min(max_batches, (2 ** 15) // max(1, batch_size)))

            # Running sums so we can compute dataset-averaged metrics without materialising every batch.
            total_items = 0
            total_loss = 0.0
            total_accuracy = 0.0 # average accuracy (computed from success~1, failure~0).
            total_entropy = 0.0
            total_msg_length = 0.0
            total_perf = 0.0 # TODO Remove perf for now; in the end, we want communication effectiveness (the average probability assigned to the right answer).

            messages = []
            predicate_ids = []

            iterator = range(nb_batch)
            if(self.autologger.display == 'tqdm'):
                iterator = tqdm.tqdm(iterator, desc='Eval.')

            for _ in iterator:
                self.start_episode(train_episode=False)
                
                batch = data_iterator.get_batch(size=batch_size, data_type='test')

                asker_outcome, retriever_outcome = self.alex_to_beth(batch)
                truth_targets = self._compute_truth_targets(batch, device=retriever_outcome.scores.device) # Shape: (batch, n_candidates)

                # Beth outputs a logit per candidate; we interpret it as the log-odds that the predicate holds for the candidate.
                logits = retriever_outcome.scores # Shape: (batch, num_candidates)
                probs = torch.sigmoid(logits) # Shape: (batch, num_candidates)

                # Scalar BCE averaged over the batch (used for logging only).
                loss = F.binary_cross_entropy_with_logits(logits, truth_targets, reduction='mean').item()
                # Accuracy is the thresholded probability vs. the binary target.
                preds = (probs >= 0.5).float() # Shape: (batch, num_candidates)
                accuracy = (preds == truth_targets).float().mean().item() # per candidate (not predicate)
                # `perf` measures how much probability mass Beth assigns to the correct truth value.
                perf = torch.where(truth_targets > 0.5, probs, 1.0 - probs).mean().item()
                # Entropy of Beth's Bernoulli output; useful to detect collapsed predictions.
                entropy = (-(probs * torch.log(probs + 1e-8) + (1.0 - probs) * torch.log(1.0 - probs + 1e-8))).mean().item()
                # Average symbol count for Alex's message in this batch.
                msg_length = asker_outcome.action[1].float().mean().item()

                batch_items = truth_targets.numel()
                total_items += batch_items
                total_loss += loss * batch_items
                total_accuracy += accuracy * batch_items
                total_entropy += entropy * batch_items
                total_msg_length += msg_length * batch_items
                total_perf += perf * batch_items

                # This block stores signals produced by the agents that are dumped at the end of eval
                # If `correct_only` is True, only signals yielding non-random accuracy are stored
                if(self.message_dump_dir is not None):
                    batch_messages = asker_outcome.action[0].detach()
                    batch_lens     = asker_outcome.action[1].detach()
                    accuracy_per_item = (preds == truth_targets).float().mean(dim=1) # (batch,) mean across candidates
                    for i in range(batch_messages.size(0)):
                        if self.correct_only and (accuracy_per_item[i].item() < 0.5):
                            continue # skip low accuracy items
                        # truncate padding away from signals
                        message = batch_messages[i].tolist()[:batch_lens[i].item()]
                        messages.append(message)
                        predicate_ids.append(int(batch.predicate_idx[i]))

            # Normalise the accumulated sums and push them to TensorBoard / stdout.
            avg_accuracy = (total_accuracy / total_items)
            log('eval/loss', total_loss / total_items)
            log('eval/accuracy', avg_accuracy)
            log('eval/perf', total_perf / total_items)
            log('eval/retriever_entropy', total_entropy / total_items)
            log('eval/msg_length', total_msg_length / total_items)  # Average number of symbols Alex produced.
            if(avg_accuracy > self.max_perf):
                self.max_perf = avg_accuracy

            # Dumps signals into file every epoch or on the last epoch, depending on the flag
            if self.message_dump_dir and (self.dump_message_mode == 'all' or epoch_index == self.epochs - 1):
                filename = os.path.join(self.message_dump_dir, f"msgs.e{epoch_index}.csv")
                with open(filename, 'w') as ostr:
                    writer = csv.writer(ostr)
                    _ = writer.writerow(['msg', 'pred_idx'])
                    for msg, pred_idx in zip(messages, predicate_ids):
                        msg = ' '.join(map(str, msg))
                        row = [msg, pred_idx]
                        _ = writer.writerow(row)
            
            return
        
        # ----------------------------
        # hic incipit quod neglegitur
        # ----------------------------

        # TODO above to run, the below is ignored for now
        counts_matrix = np.zeros((data_iterator.nb_categories, data_iterator.nb_categories))
        failure_matrix = np.zeros((data_iterator.nb_categories, data_iterator.nb_categories))

        # We try to visit each pair of categories on average 8 times.
        batch_size = 256
        max_datapoints = 32768 # (2^15)
        n = (8 * (data_iterator.nb_categories**2))
        #n = data_iterator.size(data_type='test', no_evaluation=False)
        n = min(max_datapoints, n)
        nb_batch = int(np.ceil(n / batch_size))

        messages = []
        categories = []
        input_ids = []
        batch_numbers = range(nb_batch)
        if(self.autologger.display == 'tqdm'): batch_numbers = tqdm.tqdm(batch_numbers, desc='Eval.')
        success = [] # Binary
        success_prob = [] # Probabilities
        scrambled_success_prob = [] # Probabilities

        for _ in batch_numbers:
            self.start_episode(train_episode=False)

            batch = data_iterator.get_batch(batch_size, data_type='test', no_evaluation=False, sampling_strategies=['different'], keep_category=True, keep_idx=True) # We use all categories and use only one distractor from a different category. The target image is selected in the same way as it is selected during training (equal to the original image vs a different one).

            asker_outcome, retriever_outcome = self.alex_to_beth(batch)

            retriever_pointing = misc.pointing(retriever_outcome.scores, argmax=True)
            success.append((retriever_pointing['action'] == 0).float())
            success_prob.append(retriever_pointing['dist'].probs[:, 0]) # Probability of the target

            target_category = [data_iterator.category_idx(x.category) for x in batch.original]
            distractor_category = [data_iterator.category_idx(x.category) for base_distractors in batch.base_distractors for x in base_distractors]

            failure = retriever_pointing['dist'].probs[:, 1].cpu().numpy() # Probability of the distractor
            data_iterator.failure_based_distribution.update(target_category, distractor_category, failure)

            np.add.at(counts_matrix, (target_category, distractor_category), 1.0)
            np.add.at(failure_matrix, (target_category, distractor_category), failure)

            scrambled_messages = asker_outcome.action[0].clone().detach() # We have to be careful as we probably don't want to modify the original messages
            for i, datapoint in enumerate(batch.original): # Saves the (message, category) pairs and prepares for scrambling
                msg = asker_outcome.action[0][i]
                msg_len = asker_outcome.action[1][i]
                cat = datapoint.category

                if((not self.correct_only) or (retriever_pointing['action'][i] == 0)):
                    messages.append(msg.tolist()[:msg_len])
                    categories.append(cat)
                    input_ids.append(datapoint.idx)
                # Scrambles the whole message, including the EOS (but not the padding symbols, of course)
                l = msg_len.item()
                scrambled_messages[i, :l] = scrambled_messages[i][torch.randperm(l)]

            scrambled_retriever_outcome = self.retriever(self._beth_input(batch), message=scrambled_messages, length=asker_outcome.action[1])
            scrambled_retriever_pointing = misc.pointing(scrambled_retriever_outcome.scores)
            scrambled_success_prob.append(scrambled_retriever_pointing['dist'].probs[:, 0])

        if(self.message_dump_dir is not None):
            filename = os.path.join(self.message_dump_dir, f"msgs.e{epoch_index}.csv")
            with open(filename, 'w') as ostr:
                writer = csv.writer(ostr)
                _ = writer.writerow(['msg', 'cat', 'idx'])
                for msg, cat, idx in zip(messages, categories, input_ids):
                    msg = ' '.join(map(str, msg))
                    cat = ' '.join(map(str, cat))
                    row = [msg, cat, idx]
                    _ = writer.writerow(row)

        success_prob = torch.stack(success_prob)
        scrambled_success_prob = torch.stack(scrambled_success_prob)
        scrambling_resistance = (torch.stack([success_prob, scrambled_success_prob]).min(0).values.mean().item() / success_prob.mean().item()) # Between 0 and 1. We take the min in order to not count messages that become accidentaly better after scrambling
        log('eval/scrambling-resistance', scrambling_resistance)

        # Here, we try to see how much the messages describe the categories and not the particular images
        # To do so, we use the original image as target, and an image of the same category as distractor
        abstractness = []
        n = (32 * data_iterator.nb_categories)
        n = min(max_datapoints, n)
        nb_batch = int(np.ceil(n / batch_size))
        for _ in range(nb_batch):
            self.start_episode(train_episode=False)

            batch = data_iterator.get_batch(batch_size, data_type='test', no_evaluation=False, sampling_strategies=['same'], target_is_original=True, keep_category=True) # We use only one "distractor" from a different category.

            asker_outcome, retriever_outcome = self.alex_to_beth(batch)

            retriever_pointing = misc.pointing(retriever_outcome.scores)
            abstractness.append(retriever_pointing['dist'].probs[:, 1] * 2.0)

        abstractness = torch.stack(abstractness)
        abstractness_rate = abstractness.mean().item()
        log('eval/abstractness', abstractness_rate)

        use_legacy_names = False # New (non-legacy) names are the ones used in the ACL submission.
        if(not use_legacy_names):
            name_acc = "accuracy"
            name_c_e = "c.e."
            name_base_c_e = "c.e._base" # Two base categories.
            name_gen_c_e = "c.e._gen." # Two generalization categories.
            name_mixed_c_e = "c.e._mixed" # One base category and one generalization category.
            name_tgen_c_e = "c.e._t:gen" # At least one generalization category (target/original images).
            name_dgen_c_e = "c.e._d:gen" # At least one generalization category (distractor).
        else:
            name_acc = "success_rate"
            name_c_e = "accuracy"
            name_base_c_e = "accuracy-train-td" # Two base categories.
            name_gen_c_e = "accuracy-eval-td" # Two generalization categories.
            name_mixed_c_e = "accuracy-one" # One base category and one generalization category.
            name_tgen_c_e = "accuracy-eval-t" # At least one generalization category (target/original images).
            name_dgen_c_e = "accuracy-eval-d" # At least one generalization category (distractor).

        # Here, we compute the actual accuracy rate with argmax pointing, and not the mean expected accuracy based on probabilities like is done after (for communication efficiency).
        success = torch.stack(success)
        accuracy = success.mean().item()
        log(f'eval/{name_acc}', accuracy)

        # Computes the communication efficiency when the images are selected from all categories.
        c_e = 1 - (failure_matrix.sum() / counts_matrix.sum())
        log(f'eval/{name_c_e}', c_e)
        if(not data_iterator.same_img): main_perf = c_e

        train_categories = data_iterator.training_categories_idx
        eval_categories = data_iterator.evaluation_categories_idx
        if(len(eval_categories) > 0):
            # Computes the communication efficiency when both the target and the distractor are selected from training categories.
            failure_matrix_train_td = failure_matrix[np.ix_(train_categories, train_categories)]
            counts_matrix_train_td = counts_matrix[np.ix_(train_categories, train_categories)]

            counts = counts_matrix_train_td.sum()
            base_c_e = (1 - (failure_matrix_train_td.sum() / counts)) if(counts > 0.0) else -1
            log(f'eval/{name_base_c_e}', base_c_e)

            # Computes the communication efficiency when both the target and the distractor are selected from evaluation categories (never seen during training).
            failure_matrix_eval_td = failure_matrix[np.ix_(eval_categories, eval_categories)]
            counts_matrix_eval_td = counts_matrix[np.ix_(eval_categories, eval_categories)]

            counts = counts_matrix_eval_td.sum()
            gen_c_e = (1 - (failure_matrix_eval_td.sum() / counts)) if(counts > 0.0) else -1
            log(f'eval/{name_gen_c_e}', gen_c_e)
            
            # Computes the communication efficiency when exactly one evaluation category (never seen during training) is used.
            failure_matrix_tbase_dgen = failure_matrix[np.ix_(train_categories, eval_categories)]
            counts_matrix_tbase_dgen = counts_matrix[np.ix_(train_categories, eval_categories)]
            failure_matrix_tgen_dbase = failure_matrix[np.ix_(eval_categories, train_categories)]
            counts_matrix_tgen_dbase = counts_matrix[np.ix_(eval_categories, train_categories)]

            counts = counts_matrix_tbase_dgen.sum() + counts_matrix_tgen_dbase.sum()
            mixed_c_e = (1 - ((failure_matrix_tbase_dgen.sum() + failure_matrix_tgen_dbase.sum()) / counts)) if(counts > 0.0) else -1
            log(f'eval/{name_mixed_c_e}', mixed_c_e)

            # Computes the communication efficiency when the target is selected from an evaluation category (never seen during training).
            failure_matrix_eval_t = failure_matrix[eval_categories, :]
            counts_matrix_eval_t = counts_matrix[eval_categories, :]

            counts = counts_matrix_eval_t.sum()
            tgen_c_e = (1 - (failure_matrix_eval_t.sum() / counts)) if(counts > 0.0) else -1
            log(f'eval/{name_tgen_c_e}', tgen_c_e)

            # Computes the communication efficiency when the distractor is selected from an evaluation category (never seen during training).
            failure_matrix_eval_d = failure_matrix[:, eval_categories]
            counts_matrix_eval_d = counts_matrix[:, eval_categories]

            counts = counts_matrix_eval_d.sum()
            dgen_c_e = (1 - (failure_matrix_eval_d.sum() / counts)) if(counts > 0.0) else -1
            log(f'eval/{name_dgen_c_e}', dgen_c_e)

        # If the "same_img" option is used, the communication efficiency is also computed without this feature.
        if(data_iterator.same_img):
            success_prob = []
            n = (32 * data_iterator.nb_categories)
            n = min(max_datapoints, n)
            nb_batch = int(np.ceil(n / batch_size))
            for batch_index in range(nb_batch):
                self.start_episode(train_episode=False)

                batch = data_iterator.get_batch(batch_size, data_type='test', no_evaluation=False, sampling_strategies=['different'], target_is_original=False, keep_category=True) # We use all categories and use only one distractor from a different category. The target image is selected uniformly from the original image's category.

                asker_outcome, retriever_outcome = self.alex_to_beth(batch)

                retriever_pointing = misc.pointing(retriever_outcome.scores)
                success_prob.append(retriever_pointing['dist'].probs[:, 0]) # Probability of the target.

            success_prob = torch.stack(success_prob)
            diff_tgt_c_e = success_prob.mean().item()
            log(f'eval/{name_c_e}_diff_tgt', diff_tgt_c_e)
            main_perf = diff_tgt_c_e

        if(main_perf > self.max_perf): self.max_perf = main_perf

        # Computes metrics related to symbol-order.
        # First tries to rank each symbol according to its average relative position in messages.
        rel_positions = {} # From symbol to list of relative positions
        for message in messages:
            for pos, sym in enumerate(message[:-1]): # For each symbol except the EOS
                if(sym not in rel_positions): rel_positions[sym] = []
                rel_positions[sym].append(pos / len(message)) # Relative position of the symbol in the message

        avg_positions = [(np.mean(l), sym) for (sym, l) in rel_positions.items()]
        avg_positions.sort()
        mapping = {sym: i for (i, (_, sym)) in enumerate(avg_positions)} # From symbol to rank

        # Then builds the two lists that we want to test the correlation of.
        value_list = []
        position_list = []
        for message in messages:
            for pos, sym in enumerate(message[:-1]): # For each symbol except the EOS
                value_list.append(mapping[sym])
                position_list.append(pos / len(message)) # Relative position of the symbol in the message

        res_spearman = scipy.stats.spearmanr(value_list, position_list)
        log('eval/sym-order-corr', res_spearman.correlation)
        log('eval/sym-order-pval', res_spearman.pvalue) # Should be compared to the factorial of the size of the vocabulary (the number of possible ordering of the symbols)


        # Computes compositionality measures
        # First selects a sample of (message, category) pairs
        size_sample = 1024

        sample = list(zip(messages, categories))
        random.shuffle(sample)
        sample = sample[:size_sample]
        # (To sample from each category instead, start with: d = misc.group_by(messages, categories))

        # Checks that the sample contains at least two different categories and two differents messages
        ok = False
        mes = set()
        cat = set()
        for m, c in sample:
            mes.add(tuple(m))
            cat.add(tuple(c))
            if((len(mes) > 1) and (len(cat) > 1)):
                ok = True
                break

        if(ok == False):
            print(f'Compositionality measures cannot be computed ({len(mes)} messages and {len(cat)} categories in the sample).') # Unique messages and unique categories.
        else:
            sample_messages, sample_categories = zip(*sample)
            sample_messages, sample_categories = list(map(tuple, sample_messages)), list(map(tuple, sample_categories))

            l_cor, *_ = compute_correlation.mantel(sample_messages, sample_categories, correl_only=True)
            log('FM_corr/Lev-based comp', l_cor)
            #log('FM_corr/Lev-based comp (z-score)', l_cor_n)
            #log('FM_corr/Lev-based comp (random)', l_cor_rd)

            l_n_cor, *_ = compute_correlation.mantel(sample_messages, sample_categories, message_distance=compute_correlation.levenshtein_normalised, correl_only=True)
            log('FM_corr/Normalised Lev-based comp', l_n_cor)
            #log('FM_corr/Normalised Lev-based comp (z-score)', l_n_cor_n)
            #log('FM_corr/Normalised Lev-based comp (random)', l_n_cor_rd)

            j_cor, *_ = compute_correlation.mantel(sample_messages, sample_categories, message_distance=compute_correlation.jaccard, map_msg_to_str=False, correl_only=True)
            log('FM_corr/Jaccard-based comp', j_cor)
            #log('FM_corr/Jaccard-based comp (z-score)', j_cor_n)
            #log('FM_corr/Jaccard-based comp (random)', j_cor_rd)

            if(l_n_cor > 0.0): log('FM_corr/Jaccard-n.Lev ratio', (j_cor / l_n_cor))

            minH, meanH, medH, maxH, varH = compute_entropy_stats(sample_messages, sample_categories, base=2)
            log('FM_corr/min Entropy category per msgs', minH)
            log('FM_corr/mean Entropy category per msgs', meanH)
            log('FM_corr/med Entropy category per msgs', medH)
            log('FM_corr/max Entropy category per msgs', maxH)
            log('FM_corr/var Entropy category per msgs', varH)

        # Decision tree stuff
        alphabet_size = (self.base_alphabet_size + 1)
        gram_size = 1 # Max size of n-grams to consider
        tmp = decision_tree.analyse(messages, categories, alphabet_size, data_iterator.concepts, gram_size)
        (full_tree, full_tree_accuracy) = tmp['full_tree']
        conceptual_trees = tmp['conceptual_trees']

        n_leaves, depth = full_tree.get_n_leaves(), full_tree.get_depth()
        log('decision_tree/full_accuracy', full_tree_accuracy)
        log('decision_tree/full_n_leaves', n_leaves)
        log('decision_tree/full_depth', depth)

        for i, (tree, accuracy) in conceptual_trees:
            name = data_iterator.concept_names[i]

            n_leaves, depth = tree.get_n_leaves(), tree.get_depth()
            log(('decision_tree/%s_accuracy' % name), accuracy)
            log(('decision_tree/%s_n_leaves' % name), n_leaves)
            log(('decision_tree/%s_depth' % name), depth)

        prod_conceptual_accuracy = np.array([accuracy for (_, (_, accuracy)) in conceptual_trees]).prod()
        if(prod_conceptual_accuracy > 0.0):
            tree_accuracy_ratio = (full_tree_accuracy / prod_conceptual_accuracy)
            log('decision_tree/accuracy_ratio', tree_accuracy_ratio)

        prod_conceptual_n_leaves = np.array([tree.get_n_leaves() for (_, (tree, _)) in conceptual_trees]).prod()
        if(prod_conceptual_n_leaves > 0):
            tree_n_leaves_ratio = (full_tree.get_n_leaves() / prod_conceptual_n_leaves)
            log('decision_tree/n_leaves_ratio', tree_n_leaves_ratio)

        sum_conceptual_depth = sum([tree.get_depth() for (_, (tree, _)) in conceptual_trees])
        if(sum_conceptual_depth > 0):
            tree_depth_ratio = (full_tree.get_depth() / sum_conceptual_depth)
            log('decision_tree/depth_ratio', tree_depth_ratio)
    
    def test_visualize(self, data_iterator, learning_rate):
        print("No visualisation defined.")
