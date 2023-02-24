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

from ..eval import compute_correlation
from ..eval import decision_tree

from .game import Game

# In this game, there is one sender (Alice) and one receiver (Bob).
# They are both trained to maximise the probability assigned by Bob to a "target image" in the following context: Alice is shown an "original image" and produces a message, Bob sees the message and then the target image and a "distractor image".
# They are currently both trained with REINFORCE.
class AliceBob(Game):
    def __init__(self, args, logger):
        self._logger = logger
        self.base_alphabet_size = args.base_alphabet_size
        self.max_len_msg = args.max_len

        self.use_expectation = args.use_expectation
        self.grad_scaling = args.grad_scaling or 0
        self.grad_clipping = args.grad_clipping or 0
        self.beta_sender = args.beta_sender
        self.beta_receiver = args.beta_receiver
        self.penalty = args.penalty

        self.shared = args.shared
        if(self.shared):
            senderReceiver = SenderReceiver.from_args(args)
            
            self.sender = senderReceiver.sender
            self.receiver = senderReceiver.receiver
            
            parameters = senderReceiver.parameters()
        else:
            self.sender = Sender.from_args(args)
            self.receiver = Receiver.from_args(args)
            
            parameters = it.chain(self.sender.parameters(), self.receiver.parameters())

        self._optim = build_optimizer(parameters, args.learning_rate)
        
        self.use_baseline = args.use_baseline
        if(self.use_baseline): # In that case, the loss will take into account the "baseline term" into the average recent reward.
            # Currently, the sender and receiver's rewards are the same, but we could imagine a setting in which they are different.
            self._sender_avg_reward = misc.Averager(size=12800)
            self._receiver_avg_reward = misc.Averager(size=12800)

        self.correct_only = args.correct_only # Whether to perform the fancy language evaluation using only correct messages (i.e., the one that leads to successful communication).

    def get_sender(self):
        return self.sender

    def get_receiver(self):
        return self.receiver

    @property
    def all_agents(self):
        return (self.sender, self.receiver)
    
    @property
    def current_agents(self):
        return self.all_agents

    @property
    def optim(self):
        return self._optim

    @property
    def autologger(self):
        return self._logger

    def agents_for_CNN_pretraining(self):
        if(self.shared): return [self.sender] # Because the CNN is shared between Alice and Bob, no need to pretrain the CNN of both agents.
        return [self.sender, self.receiver]

    # batch: Batch
    def _alice_input(self, batch):
        return batch.original_img(stack=True)

    # batch: Batch
    def _bob_input(self, batch):
        return torch.cat([batch.target_img(stack=True).unsqueeze(1), batch.base_distractors_img(stack=True)], dim=1)

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
        sender = self.get_sender()
        receiver = self.get_receiver()

        sender_outcome = sender(self._alice_input(batch))
        receiver_outcome = receiver(self._bob_input(batch), *sender_outcome.action)

        return sender_outcome, receiver_outcome

    # batch: Batch
    def compute_interaction(self, batch):
        sender_outcome, receiver_outcome = self(batch)

        # Alice's part
        (sender_loss, sender_perf, sender_rewards) = self.compute_sender_loss(sender_outcome, receiver_outcome.scores)
        sender_entropy = sender_outcome.entropy.mean()

        # Bob's part
        (receiver_loss, _, receiver_entropy) = self.compute_receiver_loss(receiver_outcome.scores, return_entropy=True)

        loss = sender_loss + receiver_loss
        losses = [(self.optim, loss)]

        msg_length = sender_outcome.action[1].float().mean()

        return losses, sender_rewards, sender_perf, msg_length, sender_entropy, receiver_entropy

    # Returns two tensors of shape (batch size).
    # sender_action: pair (message, length) where message is a tensor of shape (batch size, max message length) and length a tensor of shape (batch size)
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

        msg_lengths = sender_action[1].view(-1).float() # Shape: (batch size)

        rewards += -1 * (msg_lengths >= self.max_len_msg) # -1 reward anytime we reach the message length limit

        if(self.penalty > 0.0):
            length_penalties = 1.0 - (1.0 / (1.0 + self.penalty * msg_lengths.float())) # Equal to 0 when `args.penalty` is set to 0, increases to 1 with the length of the message otherwise

            rewards = (rewards - length_penalties) # Shape: (batch size)

        return (rewards, perf)

    # Returns a scalar tensor and two tensors of shape (batch size).
    # receiver_scores: tensor of shape (batch size, nb img)
    # contending_imgs: None or a list[int] containing the indices of the contending images
    def compute_sender_loss(self, sender_outcome, receiver_scores, target_idx=0, contending_imgs=None):
        if(contending_imgs is None): img_scores = receiver_scores # Shape: (batch size, nb img)
        else: img_scores = torch.stack([receiver_scores[:,i] for i in contending_imgs], dim=1) # Shape: (batch size, len(contending_imgs))

        (rewards, perf) = self.compute_sender_rewards(sender_outcome.action, img_scores, target_idx) # Two tensor of shape (batch size).

        log_prob = sender_outcome.log_prob.sum(dim=1)

        loss = torch.tensor(0.0).to(log_prob.device)

        if(self.use_baseline):
            r_baseline = self._sender_avg_reward.get(default=0.0)
            self._sender_avg_reward.update_batch(rewards.cpu().numpy())
        else: r_baseline = 0.0

        reinforce_loss = -((rewards - r_baseline) * log_prob).mean() # REINFORCE loss
        loss += reinforce_loss

        entropy_loss = -(self.beta_sender * sender_outcome.entropy.mean()) # Entropy penalty; could be normalised (divided) by (base_alphabet_size + 1)
        loss += entropy_loss

        return (loss, perf, rewards)

    # Returns the loss (a scalar tensor) and, if asked, also the average entropy of the pointing distributions (a scalar tensor).
    # receiver_scores: tensor of shape (batch size, nb img)
    # contending_imgs: None or a list[int] containing the indices of the contending images
    # TODO Currently, this computes the REINFORCE Loss. Add a flag to use a log-likelihood loss instead.
    def compute_receiver_loss(self, receiver_scores, target_idx=0, contending_imgs=None, return_entropy=False):
        if(contending_imgs is None): img_scores = receiver_scores # Shape: (batch size, nb img)
        else: img_scores = torch.stack([receiver_scores[:,i] for i in contending_imgs], dim=1) # Shape: (batch size, len(contending_imgs))
        
        # Generates a probability distribution from the scores and points at an image.
        receiver_pointing = misc.pointing(img_scores)

        perf = receiver_pointing['dist'].probs[:, target_idx].detach() # Shape: (batch size)

        if(self.use_expectation): rewards = perf.clone() # Shape: (batch size)
        else: rewards = (receiver_pointing['action'] == target_idx).float() # Shape: (batch size)

        log_prob = receiver_pointing['dist'].log_prob(receiver_pointing['action']) # The log-probabilities of the selected actions. Shape: (batch size)

        loss = torch.tensor(0.0).to(log_prob.device)

        if(self.use_baseline):
            r_baseline = self._receiver_avg_reward.get(default=0.0)
            self._receiver_avg_reward.update_batch(rewards.cpu().numpy())
        else: r_baseline = 0.0

        reinforce_loss = -((rewards - r_baseline) * log_prob).mean() # REINFORCE loss
        loss += reinforce_loss

        entropy = receiver_pointing['dist'].entropy().mean() # Shape: ()

        entropy_loss = -(self.beta_receiver * entropy) # Entropy penalty
        loss += entropy_loss

        if return_entropy: return (loss, perf, entropy)
        return (loss, perf)

    # Called at the end of each training epoch.
    def evaluate(self, data_iterator, epoch):
        def log(name, value):
            self.autologger._write(name, value, epoch, direct=True)
            if(self.autologger.display != 'minimal'): print(f'{name}\t{value}')

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
        batch_numbers = range(nb_batch)
        if(self.autologger.display == 'tqdm'): batch_numbers = tqdm.tqdm(batch_numbers, desc='Eval.')
        with torch.no_grad():
            success = [] # Binary
            success_prob = [] # Probabilities
            scrambled_success_prob = [] # Probabilities

            for _ in batch_numbers:
                self.start_episode(train_episode=False)

                batch = data_iterator.get_batch(batch_size, data_type='test', no_evaluation=False, sampling_strategies=['different'], keep_category=True) # We use all categories and use only one distractor from a different category
                sender_outcome, receiver_outcome = self.alice_to_bob(batch)

                receiver_pointing = misc.pointing(receiver_outcome.scores, argmax=True)
                success.append((receiver_pointing['action'] == 0).float())
                success_prob.append(receiver_pointing['dist'].probs[:, 0]) # Probability of the target

                target_category = [data_iterator.category_idx(x.category) for x in batch.original]
                distractor_category = [data_iterator.category_idx(x.category) for base_distractors in batch.base_distractors for x in base_distractors]

                failure = receiver_pointing['dist'].probs[:, 1].cpu().numpy() # Probability of the distractor
                data_iterator.failure_based_distribution.update(target_category, distractor_category, failure)

                np.add.at(counts_matrix, (target_category, distractor_category), 1.0)
                np.add.at(failure_matrix, (target_category, distractor_category), failure)

                scrambled_messages = sender_outcome.action[0].clone().detach() # We have to be careful as we probably don't want to modify the original messages
                for i, datapoint in enumerate(batch.original): # Saves the (message, category) pairs and prepares for scrambling
                    msg = sender_outcome.action[0][i]
                    msg_len = sender_outcome.action[1][i]
                    cat = datapoint.category

                    if((not self.correct_only) or (receiver_pointing['action'][i] == 0)):
                        messages.append(msg.tolist()[:msg_len])
                        categories.append(cat)

                    # Scrambles the whole message, including the EOS (but not the padding symbols, of course)
                    l = msg_len.item()
                    scrambled_messages[i, :l] = scrambled_messages[i][torch.randperm(l)]

                scrambled_receiver_outcome = self.get_receiver()(self._bob_input(batch), message=scrambled_messages, length=sender_outcome.action[1])
                scrambled_receiver_pointing = misc.pointing(scrambled_receiver_outcome.scores)
                scrambled_success_prob.append(scrambled_receiver_pointing['dist'].probs[:, 0])

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

                batch = data_iterator.get_batch(batch_size, data_type='test', no_evaluation=False, sampling_strategies=['same'], target_is_original=True, keep_category=True)

                sender_outcome, receiver_outcome = self.alice_to_bob(batch)

                receiver_pointing = misc.pointing(receiver_outcome.scores)
                abstractness.append(receiver_pointing['dist'].probs[:, 1] * 2.0)

            abstractness = torch.stack(abstractness)
            abstractness_rate = abstractness.mean().item()
            log('eval/abstractness', abstractness_rate)

        # Here, we compute the actual success rate with argmax pointing, and not the mean expected success based on probabilities like is done after (for accuracies).
        success = torch.stack(success)
        success_rate = success.mean().item()
        log('eval/success_rate', success_rate)

        # Computes the accuracy when the images are selected from all categories
        accuracy_all = 1 - (failure_matrix.sum() / counts_matrix.sum())
        log('eval/accuracy', accuracy_all)

        train_categories = data_iterator.training_categories_idx
        eval_categories = data_iterator.evaluation_categories_idx
        if(eval_categories != []):
            # Computes the accuracy when both the target and the distractor are selected from training categories
            failure_matrix_train_td = failure_matrix[np.ix_(train_categories, train_categories)]
            counts_matrix_train_td = counts_matrix[np.ix_(train_categories, train_categories)]

            counts = counts_matrix_train_td.sum()
            accuracy_train_td = (1 - (failure_matrix_train_td.sum() / counts)) if(counts > 0.0) else -1
            log('eval/accuracy-train-td', accuracy_train_td)

            # Computes the accuracy when the target is selected from an evaluation category (never seen during training)
            failure_matrix_eval_t = failure_matrix[eval_categories, :]
            counts_matrix_eval_t = counts_matrix[eval_categories, :]

            counts = counts_matrix_eval_t.sum()
            accuracy_eval_t = (1 - (failure_matrix_eval_t.sum() / counts)) if(counts > 0.0) else -1
            log('eval/accuracy-eval-t', accuracy_eval_t)

            # Computes the accuracy when the distractor is selected from an evaluation category (never seen during training)
            failure_matrix_eval_d = failure_matrix[:, eval_categories]
            counts_matrix_eval_d = counts_matrix[:, eval_categories]

            counts = counts_matrix_eval_d.sum()
            accuracy_eval_d = (1 - (failure_matrix_eval_d.sum() / counts)) if(counts > 0.0) else -1
            log('eval/accuracy-eval-d', accuracy_eval_d)

            # Computes the accuracy when both the target and the distractor are selected from evaluation categories (never seen during training)
            failure_matrix_eval_td = failure_matrix[np.ix_(eval_categories, eval_categories)]
            counts_matrix_eval_td = counts_matrix[np.ix_(eval_categories, eval_categories)]

            counts = counts_matrix_eval_td.sum()
            accuracy_eval_td = (1 - (failure_matrix_eval_td.sum() / counts)) if(counts > 0.0) else -1
            log('eval/accuracy-eval-td', accuracy_eval_td)

        # Computes values related to symbol-order
        # First tries to rank each symbol according to its average relative position in messages
        rel_positions = {} # From symbol to list of relative positions
        for message in messages:
            for pos, sym in enumerate(message[:-1]): # For each symbol except the EOS
                if(sym not in rel_positions): rel_positions[sym] = []
                rel_positions[sym].append(pos / len(message)) # Relative position of the symbol in the message

        avg_positions = [(np.mean(l), sym) for (sym, l) in rel_positions.items()]
        avg_positions.sort()
        mapping = {sym: i for (i, (_, sym)) in enumerate(avg_positions)} # From symbol to rank

        # Then builds the two lists that we want to test the correlation of
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
            print('Compositionality measures cannot be computed (%i messages and %i categories in the sample).' % (len(mes), len(cat))) # Unique messages and unique categories
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
        self.start_episode(train_episode=False)

        batch_size = 4
        batch = data_iterator.get_batch(batch_size, data_type='any') # Standard training batch

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

        # Message Bob-model visualisation (inspired by Simonyan et al. 2013)
        #receiver_dream = add_normal_noise((0.5 + torch.zeros_like(batch.original_img)), std_dev=0.1, clamp_values=(0,1)) # Starts with normally-random images
        receiver_dream = torch.stack([data_iterator.average_image() for _ in range(batch_size)]) # Starts with the average of the dataset
        #show_imgs([data_iterator.average_image()], 1)
        receiver_dream = receiver_dream.unsqueeze(axis=1) # Because the receiver expect a 1D array of images per batch instance; shape: [batch_size, 1, 3, height, width]
        receiver_dream = receiver_dream.clone().detach() # Creates a leaf that is a copy of `receiver_dream`
        receiver_dream.requires_grad = True

        encoded_message = self.get_receiver().encode_message(*sender_outcome.action).detach()

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

            tmp_outcome = self.get_receiver().aux_forward(receiver_dream, encoded_message)
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
