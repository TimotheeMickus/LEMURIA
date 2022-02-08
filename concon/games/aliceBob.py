import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools as it
import tqdm
from datetime import datetime

from collections import defaultdict
import random
import time

from ..agents import Sender, Receiver, SenderReceiver
from ..utils.misc import show_imgs, max_normalize_, to_color, pointing, add_normal_noise, build_optimizer, compute_entropy_stats
from ..utils import misc

from ..eval import compute_correlation
from ..eval import decision_tree

from .game import Game

class AliceBob(Game):
    def __init__(self, args, logger):
        self._logger = logger
        self.base_alphabet_size = args.base_alphabet_size
        self.max_len_msg = args.max_len
        self.is_gumbel = args.is_gumbel

        if(args.shared):
            print('You currently cannot use AliceBob with shared CNNs. Please use AliceBobPopulation with a population of size 1 instead.')
            raise ValueError

            senderReceiver = SenderReceiver.from_args(args)
            self.sender = senderReceiver.sender
            self.receiver = senderReceiver.receiver
            parameters = senderReceiver.parameters()
        else:
            self.sender = Sender.from_args(args)
            self.receiver = Receiver.from_args(args)
            parameters = it.chain(self.sender.parameters(), self.receiver.parameters())

        self.use_expectation = args.use_expectation
        self.grad_scaling = args.grad_scaling or 0
        self.grad_clipping = args.grad_clipping or 0
        self.beta_sender = args.beta_sender
        self.beta_receiver = args.beta_receiver
        self.penalty = args.penalty

        self._optim = build_optimizer(parameters, args.learning_rate)

        # Currently, the sender and receiver's rewards are the same, but we could imagine a setting in which they are different
        self.use_baseline = args.use_baseline
        if(self.use_baseline):
            self._sender_avg_reward = misc.Averager(size=12800)
            self._receiver_avg_reward = misc.Averager(size=12800)

        self.correct_only = args.correct_only # Whether to perform the fancy language evaluation using only correct messages (leading to successful communication)

    def _alice_input(self, batch):
        return batch.original_img(stack=True)

    def _bob_input(self, batch):
        return torch.cat([batch.target_img(stack=True).unsqueeze(1), batch.base_distractors_img(stack=True)], dim=1)

    def compute_interaction_gumbel(self, batch):
        sender_outcome, receiver_outcome = self(batch)
        probs = F.log_softmax(receiver_outcome.scores.transpose(1, 2), dim=-1)
        flat_probs = probs.view(-1, probs.size(-1))
        targets = torch.zeros(flat_probs.size(0), dtype=torch.long).to(probs.device)
        unweighted_loss = F.nll_loss(flat_probs, targets, reduction='none').view(probs.shape[:-1])
        # what's the probability that the message will continue after timestep t?
        prob_continues = (1 - sender_outcome.eos_probs).cumprod(1)
        # what's the probability that the message has continued until timestep t - 1?
        prob_has_lasted = torch.cat([torch.ones_like(prob_continues[:,0]).unsqueeze(1), prob_continues], dim=1)[:, :prob_continues.size(1)]
        # what's the probability that timestep t is the last?
        prob_last_step = prob_has_lasted * sender_outcome.eos_probs
        # weight loss according to the probability that this is the last step
        weighted_loss = prob_last_step * unweighted_loss
        loss = weighted_loss.sum(1).mean()

        # compute success, weighted by the likelihood of stopping at step t
        successes = (prob_last_step * (probs.argmax(-1) == 0)).sum(1)
        successes = successes.detach()
        # compute suclengthcess, weighted by the likelihood of stopping at step t
        lengths = (torch.arange(1, prob_continues.size(1)+1).to(prob_last_step.device) * prob_last_step)
        avg_msg_length = lengths.sum(1).mean().item()
        return loss, torch.tensor(0), successes, avg_msg_length, torch.tensor(0), torch.tensor(0) #TODO

    def compute_interaction(self, batch):
        if self.is_gumbel and self.sender.training:
            return self.compute_interaction_gumbel(batch)

        sender_outcome, receiver_outcome = self(batch)

        # Alice's part
        (sender_loss, sender_successes, sender_rewards) = self.compute_sender_loss(sender_outcome, receiver_outcome.scores)

        # Bob's part
        receiver_loss, receiver_entropy = self.compute_receiver_loss(receiver_outcome.scores, return_entropy=True)

        loss = sender_loss + receiver_loss

        rewards, successes = sender_rewards, sender_successes
        avg_msg_length = sender_outcome.action[1].float().mean().item()

        return loss, rewards, successes, avg_msg_length, sender_outcome.entropy.mean(), receiver_entropy

    def to(self, *vargs, **kwargs):
        self.sender, self.receiver = self.sender.to(*vargs, **kwargs), self.receiver.to(*vargs, **kwargs)
        return self

    def get_sender(self):
        return self.sender

    def get_receiver(self):
        return self.receiver

    def __call__(self, batch):
        """
        Input:
            `batch` is a Batch (a kind of named tuple); 'original_img' and 'target_img' are tensors of shape [args.batch_size, *IMG_SHAPE] and 'base_distractors' is a tensor of shape [args.batch_size, 2, *IMG_SHAPE]
        Output:
            `sender_outcome`, sender.Outcome
            `receiver_outcome`, receiver.Outcome
        """
        sender = self.get_sender()
        receiver = self.get_receiver()

        sender_outcome = sender(self._alice_input(batch))
        receiver_outcome = receiver(self._bob_input(batch), *sender_outcome.action)

        return sender_outcome, receiver_outcome

    #TODO: move this  somewhere else; e.g. `eval`?
    def test_visualize(self, data_iterator, learning_rate):
        self.start_episode(train_episode=False)

        batch_size = 4
        batch = data_iterator.get_batch(batch_size, data_type='any') # Standard training batch

        batch.require_grad()

        sender_outcome, receiver_outcome = self(batch)

        # Image-specific saliency visualisation (inspired by Simonyan et al. 2013)
        pseudo_optimizer = torch.optim.Optimizer(batch.get_images(), {}) # I'm defining this only for its `zero_grad` method (but maybe we won't need it)
        pseudo_optimizer.zero_grad()

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

            optimizer.zero_grad()

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

    def compute_sender_rewards(self, sender_action, receiver_scores):
        """
            returns the reward as well as the success for each element of a batch
        """
        # Generates a probability distribution from the scores and sample an action
        receiver_pointing = pointing(receiver_scores)

        # By design, the target is the first image
        if(self.use_expectation): successes = receiver_pointing['dist'].probs[:, 0].detach()
        else: successes = (receiver_pointing['action'] == 0).float() # Plays dice

        rewards = successes.clone()

        msg_lengths = sender_action[1].view(-1).float()

        rewards += -1 * (msg_lengths >= self.max_len_msg) # -1 reward anytime we reach the message length limit

        if(self.penalty > 0.0):
            length_penalties = 1.0 - (1.0 / (1.0 + self.penalty * msg_lengths.float())) # Equal to 0 when `args.penalty` is set to 0, increases to 1 with the length of the message otherwise

            rewards = (rewards - length_penalties)

        return (rewards, successes)

    def compute_sender_loss(self, sender_outcome, receiver_scores):
        (rewards, successes) = self.compute_sender_rewards(sender_outcome.action, receiver_scores)
        log_prob = sender_outcome.log_prob.sum(dim=1)

        loss = torch.tensor(0.0).to(log_prob.device)

        if(self.use_baseline):
            r_baseline = self._sender_avg_reward.get(default=0.0)
            self._sender_avg_reward.update_batch(rewards.cpu().numpy())
        else: r_baseline = 0.0

        reinforce_loss = -((rewards - r_baseline) * log_prob).mean() # REINFORCE loss
        loss += reinforce_loss

        entropy_loss = -(self.beta_sender * sender_outcome.entropy.mean()) # Entropy loss; could be normalised (divided) by (base_alphabet_size + 1)
        loss += entropy_loss

        return (loss, successes, rewards)

    def compute_receiver_loss(self, receiver_scores, return_entropy=False):
        receiver_pointing = pointing(receiver_scores) # The sampled action is not the same as the one in `sender_rewards` but it probably does not matter

        # By design, the target is the first image
        if(self.use_expectation):
            successes = receiver_pointing['dist'].probs[:, 0].detach()
            log_prob = receiver_pointing['dist'].log_prob(torch.tensor(0).to(receiver_scores.device))
        else: # Plays dice
            successes = (receiver_pointing['action'] == 0).float()
            log_prob = receiver_pointing['dist'].log_prob(receiver_pointing['action'])

        rewards = successes.clone()

        loss = torch.tensor(0.0).to(log_prob.device)

        if(self.use_baseline):
            r_baseline = self._receiver_avg_reward.get(default=0.0)
            self._receiver_avg_reward.update_batch(rewards.cpu().numpy())
        else: r_baseline = 0.0

        reinforce_loss = -((rewards - r_baseline) * log_prob).mean()
        loss += reinforce_loss

        entropy_loss = -(self.beta_receiver * receiver_pointing['dist'].entropy().mean()) # Entropy penalty
        loss += entropy_loss

        if return_entropy: return (loss, receiver_pointing['dist'].entropy().mean())
        return loss

    def evaluate(self, data_iterator, epoch):
        def log(name, value):
            self.autologger._write(name, value, epoch, direct=True)
            if(self.autologger.display != 'minimal'): print('%s\t%s' % (name, value))

        counts_matrix = np.zeros((data_iterator.nb_categories, data_iterator.nb_categories))
        failure_matrix = np.zeros((data_iterator.nb_categories, data_iterator.nb_categories))

        # We try to visit each pair of categories on average 8 times
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
                self.start_episode(train_episode=False) # Select agents at random if necessary

                batch = data_iterator.get_batch(batch_size, data_type='test', no_evaluation=False, sampling_strategies=['different'], keep_category=True) # We use all categories and use only one distractor from a different category
                sender_outcome, receiver_outcome = self(batch)

                receiver_pointing = pointing(receiver_outcome.scores, argmax=True)
                success.append((receiver_pointing['action'] == 0).float())
                success_prob.append(receiver_pointing['dist'].probs[:, 0]) # Probability of the target

                target_category = [data_iterator.category_idx(x.category) for x in batch.original]
                distractor_category = [data_iterator.category_idx(x.category) for base_distractors in batch.base_distractors for x in base_distractors]

                failure = receiver_pointing['dist'].probs[:, 1].cpu().numpy() # Probability of the distractor
                data_iterator.failure_based_distribution.update(target_category, distractor_category, failure)

                np.add.at(counts_matrix, (target_category, distractor_category), 1.0)
                np.add.at(failure_matrix, (target_category, distractor_category), failure)

                scrambled_messages = sender_outcome.action[0].clone().detach() # We have to be careful as we probably don't want to modify the original messages
                for i, datapoint in enumerate(batch.original): # Saves the (message, category) pairs and prepare for scrambling
                    msg = sender_outcome.action[0][i]
                    msg_len = sender_outcome.action[1][i]
                    cat = datapoint.category

                    if((not self.correct_only) or (receiver_pointing['action'][i] == 0)):
                        messages.append(msg.tolist()[:msg_len])
                        categories.append(cat)

                    # Here, I am scrambling the whole message, including the EOS (but not the padding symbols, of course)
                    l = msg_len.item()
                    scrambled_messages[i, :l] = scrambled_messages[i][torch.randperm(l)]

                scrambled_receiver_outcome = self.get_receiver()(self._bob_input(batch), message=scrambled_messages, length=sender_outcome.action[1])
                scrambled_receiver_pointing = misc.pointing(scrambled_receiver_outcome.scores)
                scrambled_success_prob.append(scrambled_receiver_pointing['dist'].probs[:, 0])

            success_prob = torch.stack(success_prob)
            scrambled_success_prob = torch.stack(scrambled_success_prob)
            scrambling_resistance = (torch.stack([success_prob, scrambled_success_prob]).min(0).values.mean().item() / success_prob.mean().item()) # Between 0 and 1. We take the min in order to not count messages that become accidentaly better after scrambling
            log('eval/scrambling-resistance', scrambling_resistance)

            # Here, we try to see how much the messages describe the categories and not the praticular images
            # To do so, we use the original image as target, and an image of the same category as distractor
            abstractness = []
            n = (32 * data_iterator.nb_categories)
            n = min(max_datapoints, n)
            nb_batch = int(np.ceil(n / batch_size))
            for _ in range(nb_batch):
                self.start_episode(train_episode=False) # Selects agents at random if necessary

                batch = data_iterator.get_batch(batch_size, data_type='test', no_evaluation=False, sampling_strategies=['same'], target_is_original=True, keep_category=True)

                sender_outcome, receiver_outcome = self(batch)

                receiver_pointing = misc.pointing(receiver_outcome.scores)
                abstractness.append(receiver_pointing['dist'].probs[:, 1] * 2.0)

            abstractness = torch.stack(abstractness)
            abstractness_rate = abstractness.mean().item()
            log('eval/abstractness', abstractness_rate)

        # Here we computing the actual success rate with argmax pointing, and not the mean expected success based on probabilities like is done after
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

    # TODO À quoi sert cette méthode ? (Ici, ça me semble assez naturel, mais l'équivalent dans AliceBobPopulation pourrait porter à confusion.)
    @property
    def agents(self):
        return self.sender, self.receiver

    @property
    def optim(self):
        return self._optim

    @property
    def autologger(self):
        return self._logger

    @classmethod
    def load(cls, path, args, _old_model=False):
        checkpoint = torch.load(path, map_location=args.device)
        instance = cls(args)
        if _old_model:
            sender_state_dict = {
                k[len('sender.'):]:checkpoint[k] for k in checkpoint.keys() if k.startswith('sender.')
            }
            instance.sender.load_state_dict(sender_state_dict)
            receiver_state_dict = {
                k[len('receiver.'):]:checkpoint[k] for k in checkpoint.keys() if k.startswith('receiver.')
            }
            instance.receiver.load_state_dict(receiver_state_dict)
        else:
            for agent, state_dict in zip(instance.agents, checkpoint['agents_state_dicts']):
                agent.load_state_dict(state_dict)
            instance._optim = checkpoint['optims'][0]
        return instance

    """def pretrain_CNNs(self, data_iterator, summary_writer, args):
        if args.shared:
            named_agents = [['sender-receiver', self.sender],]
        else:
            named_agents = [['sender', self.sender], ['receiver', self.receiver]]

        for name, agent in named_agents:
            print(("[%s] pretraining %s…" % (datetime.now(), name)), flush=True)
            self.pretrain_agent_CNN(agent, data_iterator, summary_writer, args, agent_name=name)"""
