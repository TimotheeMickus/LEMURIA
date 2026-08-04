import torch
import torch.nn as nn
import numpy as np
import scipy
import itertools as it

import tqdm
from collections import defaultdict
import random
import time

from ..agents import Sender, Receiver, SenderReceiver
from ..utils.misc import show_imgs, max_normalize_, to_color, add_normal_noise, build_optimizer, compute_entropy_stats
from ..utils import misc
from ..utils.modules import build_cnn_decoder_from_args, build_cnn_encoder_from_args

from ..eval import compute_correlation
from ..eval import decision_tree

from .game import Game
from .pretraining import CNNPretrainer, train_cnn_autoencoder
from .signalling_eval import SignallingEvalMixin, dump_signals_csv, topographic_similarity
from .vocab_penalty import VocabularyPenaltyMixin

# In this game, there is one sender (Alice) and one receiver (Bob).
# They are both trained to maximise the probability assigned by Bob to a "target image" in the following context: Alice is shown an "original image" and produces a signal, Bob sees the signal and then the target image and a "distractor image".
# Alice is trained with REINFORCE; Bob is trained by log-likelihood maximization.
class AliceBob(VocabularyPenaltyMixin, SignallingEvalMixin, Game):
    # Builds the image dataset used by Game.load to rebuild the model.
    @classmethod
    def _data_loader_from_args(cls, args):
        from ..utils.image_data import get_data_loader
        return get_data_loader(args)

    def __init__(self, args, logger, dataset, signal_dump_dir):
        self.max_perf = 0.0

        # Kept so save() can embed the exact (post-injection) args in the checkpoint; load() rebuilds from them.
        self._args = args

        self._logger = logger
        self._dataset = dataset # kept for meaning (image category) lookups in the per-meaning baseline
        self.base_alphabet_size = args.base_alphabet_size
        self.max_len_signal = args.max_len

        self.use_expectation = args.use_expectation
        self.grad_scaling = (args.grad_scaling or 0)
        self.grad_clipping = (args.grad_clipping or 0)
        self.beta_sender = args.beta_sender
        self.beta_receiver = args.beta_receiver
        self.len_penalty = args.len_penalty
        self._setup_vocab_penalty(args)
        self.full_alphabet_size = self.base_alphabet_size + 2 # EOS + content + padding (excludes BOS); matches SignalDecoder's layout.

        self.shared = args.shared
        if(self.shared):
            senderReceiver = SenderReceiver.from_args(args)

            self._sender = senderReceiver.sender
            self._receiver = senderReceiver.receiver

            parameters = senderReceiver.parameters()
            assert (args.learning_rate_a is None) and (args.learning_rate_b is None)
            self._optim = build_optimizer(parameters, args.learning_rate)
        else:
            self._sender = Sender.from_args(args)
            self._receiver = Receiver.from_args(args)

            #parameters = it.chain(self.sender.parameters(), self.receiver.parameters()) 
            self._optim = misc.build_optimizer_two_groups(
                self.sender.parameters(), misc.resolve_lr(args.learning_rate, args.learning_rate_a), 
                self.receiver.parameters(), misc.resolve_lr(args.learning_rate, args.learning_rate_b)
            )

        self.baseline_mode = args.baseline
        # REINFORCE baselines; 'per_meaning' keys on the target image category. The receiver baseline
        # is only exercised when the receiver is trained by REINFORCE (compute_receiver_loss).
        self._sender_baseline = misc.RewardBaseline(self.baseline_mode, n_meanings=dataset.nb_categories, momentum=args.baseline_momentum)
        self._receiver_baseline = misc.RewardBaseline(self.baseline_mode, n_meanings=dataset.nb_categories, momentum=args.baseline_momentum)

        self.correct_only = args.correct_only # Whether to perform the fancy language evaluation using only correct signals (i.e., the one that leads to successful communication).
        self._topsim_correl_only = True # If True, skips the Mantel permutations (fast; correlation only, no p/z).
        self.eval_oracle_language = args.eval_oracle_language
        self._oracle_concept_offsets = None  # list[int]; per-concept symbol base, built lazily
        self._oracle_alphabet_size = None     # int; total number of (concept, value) symbols
        
        self.debug = args.debug
        self.signal_dump_dir = signal_dump_dir # str|None
        self._init_receiver_preprocessor(args, dataset)
        self._init_cnn_pretrainer(args, dataset)

    # Sets self.pretrainer to a CNNPretrainer when --pretrain_CNNs is given, else None. Factored out
    # so that AliceBobCharlie (which does not call AliceBob.__init__) can reuse it.
    def _init_cnn_pretrainer(self, args, dataset):
        self.pretrainer = CNNPretrainer(self, args, dataset) if(args.pretrain_CNNs) else None
    
    #TODO: the preprocessor currently requires the dataloader to be passed as argument upon construction.
    # a cleaner fix would be to implement a flag to signal the preprocessor needs to be pretrained before actual training can start
    # or include the preprocessor in the pre-training round.
    def _init_receiver_preprocessor(self, args, dataset):
        dcnn_factory_fn = misc.get_default_fn(build_cnn_decoder_from_args, args)
        cnn_factory_fn = misc.get_default_fn(build_cnn_encoder_from_args, args)
        if args.autoencode_receiver_inputs:
            self.receiver_preprocessor = train_cnn_autoencoder(
                None, # no agent
                dataset, #
                convolution_factory=cnn_factory_fn, 
                deconvolution_factory=dcnn_factory_fn, 
                pretrain_CNNs_on_eval=True, 
                _is_external_ae=True,
                device=args.device,
                display_mode=args.display,
                summary_writer=self.autologger.summary_writer,
                agent_name='receiver preprocessor AE',
                epochs=args.pretrain_epochs,
                learning_rate=args.pretrain_learning_rate,
            )
            self.receiver_preprocessor.requires_grad_(False)
        else:
            self.receiver_preprocessor = nn.Identity()


    @property
    def sender(self):
        return self._sender

    @property
    def receiver(self):
        return self._receiver

    @property
    def all_agents(self):
        return (self.sender, self.receiver)

    @property
    def current_agents(self):
        return self.all_agents

    @property
    def optims(self):
        return [self._optim]

    @property
    def autologger(self):
        return self._logger

    # Lists the (agent, role) pairs to pretrain. In the shared case only the sender is listed,
    # because the CNN is shared between Alice and Bob (so pretraining the sender's CNN pretrains
    # both). `role` is used only for naming/logging by the CNN pretrainer.
    def agents_for_pretraining(self):
        if(self.shared): return [(self.sender, "sender")]
        return [(self.sender, "sender"), (self.receiver, "receiver")]

    # batch: Batch
    def _alice_input(self, batch):
        return batch.original_img(stack=True)

    # batch: Batch
    def _bob_input(self, batch):
        with torch.no_grad():
            ipts = torch.cat([batch.target_img(stack=True).unsqueeze(1), batch.base_distractors_img(stack=True)], dim=1)
            ipts = self.receiver_preprocessor(ipts.flatten(0, 1)).view(*ipts.shape).detach()
        return ipts

    def __call__(self, batch):
        """
        Input:
            batch: Batch
        Output:
            sender_outcome: sender.Outcome
            receiver_outcome: receiver.Outcome
        """
        return self.alice_to_bob(batch)

    def alice_to_bob(self, batch):
        sender = self.sender
        receiver = self.receiver

        sender_outcome = sender(self._alice_input(batch))
        receiver_outcome = receiver(self._bob_input(batch), *sender_outcome.action)

        return sender_outcome, receiver_outcome

    # batch: Batch
    def compute_interaction(self, batch, **kwargs):
        # TODO: change return signature to loss, {dict of things to log}

        sender_outcome, receiver_outcome = self(batch)

        # Per-item meaning ids (target image category) for the per-meaning REINFORCE baseline.
        meanings = batch.target_category(stack=True, f=self._dataset.category_idx)

        # Alice's part
        (sender_loss, sender_perf, sender_rewards) = self.compute_sender_loss(sender_outcome, receiver_outcome.scores, meanings=meanings)
        sender_entropy = sender_outcome.entropy.mean()

        # Bob's part
        (receiver_loss, _, receiver_entropy) = self.compute_receiver_loss(receiver_outcome.scores, return_entropy=True, meanings=meanings)

        loss = sender_loss + receiver_loss
        optimization = [(self._optim, loss.detach(), misc.get_backward_f(loss))]

        signal_length = sender_outcome.action[1].float().mean()

        return optimization, sender_rewards, sender_perf, signal_length, sender_entropy, receiver_entropy

    # Returns two tensors of shape (batch size).
    # sender_action: pair (signal, length) where signal is a tensor of shape (batch size, max signal length) and length a tensor of shape (batch size)
    # img_scores: tensor of shape (batch size, nb img)
    def compute_sender_rewards(self, sender_action, img_scores, target_idx):
        """
            returns the reward as well as the performance for each element of a batch
        """
        # Generates a probability distribution from the scores and points at an image.
        receiver_pointing = misc.pointing(img_scores)

        perf = receiver_pointing['dist'].probs[:, target_idx].detach() # Shape: (batch size)

        if(self.use_expectation): rewards = perf.clone() # Shape: (batch size)
        else: rewards = (receiver_pointing['action'] == target_idx).float() # Shape: (batch size)

        signal_lengths = sender_action[1].view(-1).float() # Shape: (batch size)

        rewards += -1 * (signal_lengths >= self.max_len_signal) # -1 reward anytime we reach the signal length limit

        if(self.len_penalty > 0.0):
            # The penalty equals to 0 when `args.len_penalty` is set to 0, and increases to 1 with the length of the signal otherwise.
            length_penalties = 1.0 - (1.0 / (1.0 + self.len_penalty * signal_lengths)) # Shape: (batch,)

            rewards = (rewards - length_penalties) # Shape: (batch size)

        # Vocabulary penalty, 'reward' mode (a no-op in 'aux' mode; see compute_sender_loss). Shape: (batch,) or 0.0.
        rewards = rewards - self.vocabulary_reward_penalty(sender_action[0], self.sender.eos_index, self.sender.padding_idx, self.full_alphabet_size)

        return (rewards, perf)

    # Returns a scalar tensor and two tensors of shape (batch size).
    # receiver_scores: tensor of shape (batch size, nb img)
    # contending_imgs: None or a list[int] containing the indices of the contending images
    def compute_sender_loss(self, sender_outcome, receiver_scores, target_idx=0, contending_imgs=None, meanings=None):
        if(contending_imgs is None): img_scores = receiver_scores # Shape: (batch size, nb img)
        else: img_scores = torch.stack([receiver_scores[:,i] for i in contending_imgs], dim=1) # Shape: (batch size, len(contending_imgs))

        (rewards, perf) = self.compute_sender_rewards(sender_outcome.action, img_scores, target_idx) # Two tensors of shape (batch size).

        loss = 0.0

        # REINFORCE loss
        log_prob = sender_outcome.log_prob.sum(dim=1) # The per-episode sum of the log-probabilies of the selection actions (they all get the same reward). Shape: (batch size)

        r_baseline = self._sender_baseline(rewards, meanings) # 0.0, a float ('global'), or a (batch,) tensor ('per_meaning')

        reinforce_loss = -((rewards - r_baseline) * log_prob).mean()
        loss += reinforce_loss

        # Entropy penalty
        entropy_loss = -(self.beta_sender * sender_outcome.entropy.mean()) # Could be normalised (divided) by (base_alphabet_size + 1).
        loss += entropy_loss

        # Vocabulary penalty, 'aux' mode: a direct, differentiable group-sparsity loss on the batch
        # symbol marginal (the 'reward' mode counterpart lives in compute_sender_rewards instead).
        loss += self.vocabulary_aux_loss(sender_outcome.symbol_marginal, self.sender.eos_index)

        return (loss, perf, rewards)

    # Returns the loss (a scalar tensor) and, if asked, also the average entropy of the pointing distributions (a scalar tensor).
    # receiver_scores: tensor of shape (batch size, nb img)
    # use_REINFORCE: if true, the REINFORCE loss is used, otherwise, the cross-entropy loss is used
    # contending_imgs: None or a list[int] containing the indices of the contending images
    def compute_receiver_loss(self, receiver_scores, use_REINFORCE=False, target_idx=0, contending_imgs=None, return_entropy=False, meanings=None):
        if(contending_imgs is None): img_scores = receiver_scores # Shape: (batch size, nb img)
        else: img_scores = torch.stack([receiver_scores[:,i] for i in contending_imgs], dim=1) # Shape: (batch size, len(contending_imgs))

        # Generates a probability distribution from the scores and points at an image.
        receiver_pointing = misc.pointing(img_scores)

        perf = receiver_pointing['dist'].probs[:, target_idx].detach() # Shape: (batch size)
        
        entropy = receiver_pointing['dist'].entropy().mean() # Shape: ()

        loss = 0.0

        # Main loss
        if(use_REINFORCE): # REINFORCE
            log_prob = receiver_pointing['dist'].log_prob(receiver_pointing['action']) # The log-probabilities of the selected images. Shape: (batch size)

            if(self.use_expectation): rewards = perf.clone() # Shape: (batch size)
            else: rewards = (receiver_pointing['action'] == target_idx).float() # Shape: (batch size)

            r_baseline = self._receiver_baseline(rewards, meanings) # 0.0, a float ('global'), or a (batch,) tensor ('per_meaning')

            reinforce_loss = -((rewards - r_baseline) * log_prob).mean()
            loss += reinforce_loss
        else: # Cross-entropy maximization
            log_prob = receiver_pointing['dist'].log_prob(torch.tensor(target_idx, device=img_scores.device)) # The log-probabilities of the target images. Shape: (batch size)

            cross_entropy_loss = -log_prob.mean() # Shape: ()
            loss += cross_entropy_loss

        # Entropy penalty
        if(self.beta_receiver != 0.0):
            entropy_loss = -(self.beta_receiver * entropy)
            loss += entropy_loss

        if return_entropy: return (loss, perf, entropy)
        return (loss, perf)

    # Lazily computes the per-concept symbol bases for the "oracle" language: concept d, value v
    # maps to the unique symbol (offset[d] + v), so both the concept (position/identity) and its
    # value are recoverable even under an order-invariant (set-based) metric like Jaccard.
    def _ensure_oracle_setup(self, data_iterator):
        if(self._oracle_concept_offsets is None):
            sizes = [len(concept) for concept in data_iterator.concepts]
            offsets, running = [], 0
            for size in sizes:
                offsets.append(running)
                running += size
            self._oracle_concept_offsets = offsets
            self._oracle_alphabet_size = running

    # Oracle signal for a category (a tuple of per-concept value indices), as a list of int
    # symbols. Used only when self.eval_oracle_language is set, to replace the emergent signals
    # during fancy eval.
    def _oracle_signal_for_category(self, category):
        return [self._oracle_concept_offsets[d] + int(v) for d, v in enumerate(category)]

    # Overrides SignallingEvalMixin._signal_correctness: P(target) under the receiver for each
    # item, given (possibly scrambled) signals.
    def _signal_correctness(self, batch, signals, lengths):
        outcome = self.receiver(self._bob_input(batch), signal=signals, length=lengths)
        return misc.pointing(outcome.scores)['dist'].probs[:, 0]

    # Called at the end of each training epoch.
    @torch.no_grad()
    def evaluate(self, data_iterator, epoch_index):
        def log(name, value): self._log(name, value, epoch_index)

        counts_matrix = np.zeros((data_iterator.nb_categories, data_iterator.nb_categories))
        failure_matrix = np.zeros((data_iterator.nb_categories, data_iterator.nb_categories))

        # We try to visit each pair of categories on average 8 times.
        batch_size = 256
        max_datapoints = 32768 # (2^15)
        n = (8 * (data_iterator.nb_categories**2))
        #n = data_iterator.size(data_type='test', no_evaluation=False)
        n = min(max_datapoints, n)
        nb_batch = int(np.ceil(n / batch_size))

        signals = []
        categories = []
        input_ids = []
        batch_numbers = range(nb_batch)
        if(self.autologger.display == 'tqdm'): batch_numbers = tqdm.tqdm(batch_numbers, desc='Eval.')
        success = [] # Binary
        # Scrambling resistance: preserved correctness after shuffling / original correctness.
        perf_scrambled = 0.0
        perf_baseline = 0.0

        for batch_index in batch_numbers:
            self.start_episode(train_episode=False)

            batch = data_iterator.get_batch(batch_size, data_type='test', no_evaluation=False, sampling_strategies=['different'], keep_category=True, keep_idx=True) # We use all categories and use only one distractor from a different category. The target image is selected in the same way as it is selected during training (equal to the original image vs a different one).

            sender_outcome, receiver_outcome = self.alice_to_bob(batch)

            receiver_pointing = misc.pointing(receiver_outcome.scores, argmax=True)
            success.append((receiver_pointing['action'] == 0).float())
            success_prob = receiver_pointing['dist'].probs[:, 0] # Probability of the target

            target_category = [data_iterator.category_idx(x.category) for x in batch.original]
            distractor_category = [data_iterator.category_idx(x.category) for base_distractors in batch.base_distractors for x in base_distractors]

            failure = receiver_pointing['dist'].probs[:, 1].cpu().numpy() # Probability of the distractor
            data_iterator.failure_based_distribution.update(target_category, distractor_category, failure)

            np.add.at(counts_matrix, (target_category, distractor_category), 1.0)
            np.add.at(failure_matrix, (target_category, distractor_category), failure)

            for i, datapoint in enumerate(batch.original): # Saves the (signal, category) pairs
                signal = sender_outcome.action[0][i]
                signal_len = sender_outcome.action[1][i]
                cat = datapoint.category

                if((not self.correct_only) or (receiver_pointing['action'][i] == 0)):
                    signals.append(signal.tolist()[:signal_len])
                    categories.append(cat)
                    input_ids.append(datapoint.idx)

            # Scrambling resistance (scrambles signal order, incl. EOS, and rescores).
            kept, base = self._scrambling_resistance(batch, sender_outcome.action[0], sender_outcome.action[1], success_prob)
            perf_scrambled += kept
            perf_baseline += base

        if(self.signal_dump_dir is not None):
            import os

            filename = os.path.join(self.signal_dump_dir, f"signals.e{epoch_index}.csv")
            rows = [
                [' '.join(map(str, signal)), ' '.join(map(str, cat)), idx]
                for signal, cat, idx in zip(signals, categories, input_ids)
            ]
            dump_signals_csv(filename, ['signal', 'cat', 'idx'], rows)

        scrambling_resistance = (perf_scrambled / perf_baseline) if(perf_baseline > 0.0) else 0.0 # Between 0 and 1. The min inside _scrambling_resistance avoids counting signals that accidentally improve after scrambling.
        log('eval/scrambling-resistance', scrambling_resistance)

        # Here, we try to see how much the signals describe the categories and not the particular images
        # To do so, we use the original image as target, and an image of the same category as distractor
        abstractness = []
        n = (32 * data_iterator.nb_categories)
        n = min(max_datapoints, n)
        nb_batch = int(np.ceil(n / batch_size))
        for batch_index in range(nb_batch):
            self.start_episode(train_episode=False)

            batch = data_iterator.get_batch(batch_size, data_type='test', no_evaluation=False, sampling_strategies=['same'], target_is_original=True, keep_category=True) # We use only one "distractor" from a different category.

            sender_outcome, receiver_outcome = self.alice_to_bob(batch)

            receiver_pointing = misc.pointing(receiver_outcome.scores)
            abstractness.append(receiver_pointing['dist'].probs[:, 1] * 2.0)

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

                sender_outcome, receiver_outcome = self.alice_to_bob(batch)

                receiver_pointing = misc.pointing(receiver_outcome.scores)
                success_prob.append(receiver_pointing['dist'].probs[:, 0]) # Probability of the target.

            success_prob = torch.stack(success_prob)
            diff_tgt_c_e = success_prob.mean().item()
            log(f'eval/{name_c_e}_diff_tgt', diff_tgt_c_e)
            main_perf = diff_tgt_c_e

        if(main_perf > self.max_perf): self.max_perf = main_perf

        # Computes metrics related to symbol-order.
        # First tries to rank each symbol according to its average relative position in signals.
        rel_positions = {} # From symbol to list of relative positions
        for signal in signals:
            for pos, sym in enumerate(signal[:-1]): # For each symbol except the EOS
                if(sym not in rel_positions): rel_positions[sym] = []
                rel_positions[sym].append(pos / len(signal)) # Relative position of the symbol in the signal

        avg_positions = [(np.mean(l), sym) for (sym, l) in rel_positions.items()]
        avg_positions.sort()
        mapping = {sym: i for (i, (_, sym)) in enumerate(avg_positions)} # From symbol to rank

        # Then builds the two lists that we want to test the correlation of.
        value_list = []
        position_list = []
        for signal in signals:
            for pos, sym in enumerate(signal[:-1]): # For each symbol except the EOS
                value_list.append(mapping[sym])
                position_list.append(pos / len(signal)) # Relative position of the symbol in the signal

        res_spearman = scipy.stats.spearmanr(value_list, position_list)
        log('eval/sym-order-corr', res_spearman.correlation)
        log('eval/sym-order-pval', res_spearman.pvalue) # Should be compared to the factorial of the size of the vocabulary (the number of possible ordering of the symbols)


        # Computes compositionality measures
        # First selects a sample of (signal, category) pairs
        size_sample = 1024

        sample = list(zip(signals, categories))
        random.shuffle(sample)
        sample = sample[:size_sample]
        # (To sample from each category instead, start with: d = misc.group_by(signals, categories))

        # Checks that the sample contains at least two different categories and two differents signals.
        ok = False
        s_signals = set()
        s_cat = set()
        for s, c in sample:
            s_signals.add(tuple(s))
            s_cat.add(tuple(c))
            if((len(s_signals) > 1) and (len(s_cat) > 1)):
                ok = True
                break

        if(ok == False):
            print(f'Compositionality measures cannot be computed ({len(s_signals)} signals and {len(s_cat)} categories in the sample).') # Unique signals and unique categories.
        else:
            sample_signals, sample_categories = zip(*sample)
            sample_signals = list(map(tuple, sample_signals))
            sample_categories = list(map(tuple, sample_categories))

            # DEBUG FEATURE (--eval_oracle_language): replace the emergent language with a known-compositional "control"/oracle language (the reverse-Polish encoding of the predicate during fancy evaluation. This is a debugging/diagnostic aid; it shows what the language metrics report for a language that is compositional by construction (an upper-bound sanity check). Applied to the category-per-signal entropy and topographic similarity.
            if(self.eval_oracle_language):
                self._ensure_oracle_setup(data_iterator)
                analysis_signals = [tuple(self._oracle_signal_for_category(c)) for c in sample_categories]
            else:
                analysis_signals = sample_signals

            # Topographic similarity via the Mantel test (Spearman), deduplicating by meaning (category) so repeated categories don't inflate the correlation.
            def _topsim(signal_distance, map_signal_to_str):
                return topographic_similarity(
                    analysis_signals, sample_categories,
                    signal_distance=signal_distance, meaning_distance=compute_correlation.hamming_str,
                    meaning_keys=sample_categories, map_signal_to_str=map_signal_to_str, map_meaning_to_str=True,
                    method='spearman', deduplicate=True, correl_only=self._topsim_correl_only, error_on_duplicate_meanings=False,
                )

            lev = _topsim(compute_correlation.levenshtein, True)
            log('FM_corr/Lev-based comp', lev.r)
            if(not self._topsim_correl_only):
                log('FM_corr/Lev-based comp (p)', lev.p)
                log('FM_corr/Lev-based comp (z)', lev.z)

            n_lev = _topsim(compute_correlation.levenshtein_normalised, True)
            log('FM_corr/Normalised Lev-based comp', n_lev.r)
            if(not self._topsim_correl_only):
                log('FM_corr/Normalised Lev-based comp (p)', n_lev.p)
                log('FM_corr/Normalised Lev-based comp (z)', n_lev.z)

            jac = _topsim(compute_correlation.jaccard, False)
            log('FM_corr/Jaccard-based comp', jac.r)
            if(not self._topsim_correl_only):
                log('FM_corr/Jaccard-based comp (p)', jac.p)
                log('FM_corr/Jaccard-based comp (z)', jac.z)

            if(n_lev.r > 0.0): log('FM_corr/Jaccard-n.Lev ratio', (jac.r / n_lev.r))

            # Uses `analysis_signals`, so under --eval_oracle_language this entropy is computed on
            # the control language too (see the debug-feature note above).
            minH, meanH, medH, maxH, varH = compute_entropy_stats(analysis_signals, sample_categories, base=2)
            log('FM_corr/min Entropy category per signals', minH)
            log('FM_corr/mean Entropy category per signals', meanH)
            log('FM_corr/med Entropy category per signals', medH)
            log('FM_corr/max Entropy category per signals', maxH)
            log('FM_corr/var Entropy category per signals', varH)

        # Decision tree stuff
        alphabet_size = (self.base_alphabet_size + 1)
        gram_size = 1 # Max size of n-grams to consider
        # Oracle: recover the category from the compositional (concept, value) symbols instead of
        # the emergent signal. `alphabet_size` here is only the out-of-signal sentinel, so it must
        # stay strictly above every symbol actually present.
        if(self.eval_oracle_language):
            self._ensure_oracle_setup(data_iterator)
            dt_signals = [self._oracle_signal_for_category(c) for c in categories]
            dt_alphabet_size = (self._oracle_alphabet_size + 1)
        else:
            dt_signals = signals
            dt_alphabet_size = alphabet_size
        tmp = decision_tree.analyse(dt_signals, categories, dt_alphabet_size, data_iterator.concepts, gram_size)
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
        self.start_episode(train_episode=False)

        batch_size = 4
        batch = data_iterator.get_batch(batch_size, data_type='any', sampling_strategies=["different"])

        batch.require_grad()

        sender_outcome, receiver_outcome = self.alice_to_bob(batch)

        # Image-specific saliency visualisation (inspired by Simonyan et al. 2013)
        _COLOR, _INTENSITY = range(2)
        def process(t, dim, mode):
            if(mode == _COLOR):
                t = max_normalize(t, dim=dim, abs_val=True) # Normalises each image
                t *= 0.5
                t += 0.5

                return t
            elif(mode == _INTENSITY):
                t = t.abs()
                t = t.max(dim).values # Max over the colour channel

                max_normalize_(t, dim=dim, abs_val=False) # Normalises each image

                return to_color(t, dim)

        mode = _INTENSITY

        # Alice's part
        sender_outcome.log_prob.sum().backward()

        sender_part = batch.original_img(stack=True, f=(lambda img: img.grad.detach()))
        sender_part = process(sender_part, 1, mode)

        # Bob's part
        receiver_outcome.scores.sum().backward()

        receiver_part_target_img = batch.target_img(stack=True, f=(lambda img: img.grad.detach()))
        receiver_part_target_img = process(receiver_part_target_img.unsqueeze(axis=1), 2, mode).squeeze(axis=1)

        receiver_part_base_distractors = batch.base_distractors_img(stack=True, f=(lambda img: img.grad.detach()))
        receiver_part_base_distractors = process(receiver_part_base_distractors, 2, mode)

        # Signal Bob-model visualisation (inspired by Simonyan et al. 2013)
        #receiver_dream = add_normal_noise((0.5 + torch.zeros_like(batch.original_img)), std_dev=0.1, clamp_values=(0,1)) # Starts with normally-random images
        receiver_dream = torch.stack([data_iterator.average_image() for _ in range(batch_size)]) # Starts with the average of the dataset
        #show_imgs([data_iterator.average_image()], 1)
        receiver_dream = receiver_dream.unsqueeze(axis=1) # Because the receiver expect a 1D array of images per batch instance; shape: [batch_size, 1, 3, height, width]
        receiver_dream = receiver_dream.clone().detach() # Creates a leaf that is a copy of `receiver_dream`
        receiver_dream.requires_grad = True

        encoded_signal = self.receiver.encode_signal(*sender_outcome.action).detach()

        # Defines a filter for checking smoothness
        channels = 3
        filter_weight = torch.tensor([[1.2, 2, 1.2], [2, -12.8, 2], [1.2, 2, 1.2]]) # -12.8 (at the center) is equal to the opposite of the sum of the other coefficients
        filter_weight = filter_weight.view(1, 1, 3, 3)
        filter_weight = filter_weight.repeat(channels, 1, 1, 1) # Shape: [channel, 1, 3, 3]
        filter_layer = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, groups=channels, bias=False)
        filter_layer.weight.data = filter_weight
        filter_layer.weight.requires_grad = False

        #optimizer = torch.optim.RMSprop([receiver_dream], lr=10.0*args.learning_rate)
        optimizer = torch.optim.SGD([receiver_dream], lr=2*learning_rate, momentum=0.9)
        #optimizer = torch.optim.Adam([receiver_dream], lr=10.0*args.learning_rate)
        nb_iter = 1000
        j = 0
        for i in range(nb_iter):
            if(i >= (j + (nb_iter / 10))):
                print(i)
                j = i

            tmp_outcome = self.receiver.aux_forward(receiver_dream, encoded_signal)
            loss = -tmp_outcome.scores[:, 0].sum()

            regularisation_loss = 0.0
            #regularisation_loss += 0.05 * (receiver_dream - 0.5).norm(2) # Similar to L2 regularisation but centered around 0.5
            regularisation_loss += 0.01 * (receiver_dream - 0.5).norm(1) # Similar to L1 regularisation but centered around 0.5
            #regularisation_loss += -0.1 * torch.log(1.0 - (2 * torch.abs(receiver_dream - 0.5))).sum() # "Wall" at 0 and 1
            loss += regularisation_loss

            #smoothness_loss = 20 * torch.abs(filter_layer(receiver_dream.squeeze(axis=1))).sum()
            smoothness_loss = 20 * torch.abs(filter_layer(receiver_dream.squeeze(axis=1))).norm(1)
            loss += smoothness_loss

            loss.backward()

            # TODO In Deep Dream, they blur the gradient before applying it. (https://hackernoon.com/deep-dream-with-tensorflow-a-practical-guide-to-build-your-first-deep-dream-experience-f91df601f479)
            # This can probably be done by modifying receiver_dream.grad.

            optimizer.step()
            optimizer.zero_grad()

        receiver_dream = receiver_dream.squeeze(axis=1)
        receiver_dream = torch.clamp(receiver_dream, 0, 1)

        # Displays the visualisations
        imgs = []
        for i in range(batch_size):
            imgs.append(batch.original[i].img)
            imgs.append(sender_part[i])

            imgs.append(batch.target[i].img)
            imgs.append(receiver_part_target_img[i])

            for j in range(len(batch.base_distractors[i])):
                imgs.append(batch.base_distractors[i][j].img)
                imgs.append(receiver_part_base_distractors[i][j])

            imgs.append(receiver_dream[i])
        #for img in imgs: print(img.shape)
        show_imgs([img.detach() for img in imgs], nrow=(len(imgs) // batch_size)) #show_imgs(imgs, nrow=(2 * (2 + batch.base_distractors.size(1))))
