import torch
import torch.nn as nn
import numpy as np
import itertools as it
import tqdm
from datetime import datetime

from collections import defaultdict
import random
import time

from ..agents import Sender, Receiver, SenderReceiver
from ..utils.misc import show_imgs, max_normalize_, to_color, pointing, add_normal_noise, compute_entropy, build_optimizer, compute_entropy_stats
from ..utils import misc

from ..eval import compute_correlation
from ..eval import decision_tree

from .game import Game

class AliceBob(Game):
    def __init__(self, args):
        self.base_alphabet_size = args.base_alphabet_size
        self.max_len_msg = args.max_len

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

    def _alice_input(self, batch):
        return batch.original_img(stack=True)

    def _bob_input(self, batch):
        return torch.cat([batch.target_img(stack=True).unsqueeze(1), batch.base_distractors_img(stack=True)], dim=1)

    def compute_interaction(self, batch):
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

    def __call__(self, batch, sender=None, receiver=None):
        """
        Input:
            `batch` is a Batch (a kind of named tuple); 'original_img' and 'target_img' are tensors of shape [args.batch_size, *IMG_SHAPE] and 'base_distractors' is a tensor of shape [args.batch_size, 2, *IMG_SHAPE]
        Output:
            `sender_outcome`, sender.Outcome
            `receiver_outcome`, receiver.Outcome
        """
        sender = sender or self.sender
        receiver = receiver or self.receiver
        sender_outcome = sender(self._alice_input(batch))
        receiver_outcome = receiver(self._bob_input(batch), *sender_outcome.action)

        return sender_outcome, receiver_outcome

    def test_visualize(self, data_iterator, learning_rate):
        self.eval() # Sets the model in evaluation mode; good idea or not?

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

        encoded_message = self.receiver.encode_message(*sender_outcome.action).detach()

        # Defines a filter for checking smoothness
        channels = 3
        filter_weight = torch.tensor([[1.2, 2, 1.2], [2, -12.8, 2], [1.2, 2, 1.2]]) # -12.8 (at the center) is equal to the opposite of sum of the other coefficient
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

            tmp_outcome = self.receiver.aux_forward(receiver_dream, encoded_message)
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

        loss = -(rewards * log_prob).mean()

        loss = loss - (self.beta_sender * sender_outcome.entropy.mean()) # Entropy penalty

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

        rewards = successes

        loss = -(rewards * log_prob).mean()

        loss = loss - (self.beta_receiver * receiver_pointing['dist'].entropy().mean()) # Entropy penalty
        if return_entropy:
            return loss, receiver_pointing['dist'].entropy().mean()
        return loss

    def evaluate(self, data_iterator, epoch, event_writer=None, display='tqdm', debug=False, log_lang_progress=True):
        self.eval()

        counts_matrix = np.zeros((data_iterator.nb_categories, data_iterator.nb_categories))
        failure_matrix = np.zeros((data_iterator.nb_categories, data_iterator.nb_categories))

        batch_size = 256
        n = len(data_iterator)
        if(n is None): n = 10000
        nb_batch = int(np.ceil(n / batch_size)) # Doing so makes us see on average at least each data point once; this also means that each cell of the failure matrix is updated (len(data_iterator) / ((nb_categories) * (nb_categories - 1))) time, which can be quite low (~10)

        messages = []
        categories = []
        batch_numbers = range(nb_batch)
        if(display == 'tqdm'): batch_numbers = tqdm.tqdm(range(nb_batch), desc='Eval.')
        for _ in batch_numbers:
            self.start_episode(train_episode=False) # Select agents at random if necessary

            with torch.no_grad():
                batch = data_iterator.get_batch(batch_size, data_type='test', no_evaluation=False, sampling_strategies=['different'], keep_category=True) # We use all categories and use only one distractor from a different category
                sender_outcome, receiver_outcome = self(batch)

                messages.extend([msg.tolist()[:l] for msg, l in zip(*sender_outcome.action)])
                categories.extend([x.category for x in batch.original])

                receiver_pointing = pointing(receiver_outcome.scores)
                failure = receiver_pointing['dist'].probs[:, 1].cpu().numpy() # Probability of the distractor

                target_category = [data_iterator.category_idx(x.category) for x in batch.original]
                distractor_category = [data_iterator.category_idx(x.category) for base_distractors in batch.base_distractors for x in base_distractors]

                data_iterator.failure_based_distribution.update(target_category, distractor_category, failure)

                np.add.at(counts_matrix, (target_category, distractor_category), 1.0)
                np.add.at(failure_matrix, (target_category, distractor_category), failure)

        # Computes the accuracy when the images are selected from all categorsie
        accuracy_all = 1 - (failure_matrix.sum() / counts_matrix.sum())
        if(event_writer is not None): event_writer.add_scalar('eval/accuracy', accuracy_all, epoch, period=1)
        if(display != 'minimal'): print('Accuracy: %s' % accuracy_all)

        train_categories = data_iterator.training_categories_idx
        eval_categories = data_iterator.evaluation_categories_idx
        if(eval_categories != []):
            # Computes the accuracy when both the target and the distractor are selected from training categories
            failure_matrix_train_td = failure_matrix[np.ix_(train_categories, train_categories)]
            counts_matrix_train_td = counts_matrix[np.ix_(train_categories, train_categories)]

            counts = counts_matrix_train_td.sum()
            accuracy_train_td = (1 - (failure_matrix_train_td.sum() / counts)) if(counts > 0.0) else -1
            if(event_writer is not None): event_writer.add_scalar('eval/accuracy-train-td', accuracy_train_td, epoch, period=1)
            if(display != 'minimal'): print('Accuracy train-td %s' % accuracy_train_td)

            # Computes the accuracy when the target is selected from an evaluation category (never seen during training)
            failure_matrix_eval_t = failure_matrix[eval_categories, :]
            counts_matrix_eval_t = counts_matrix[eval_categories, :]

            counts = counts_matrix_eval_t.sum()
            accuracy_eval_t = (1 - (failure_matrix_eval_t.sum() / counts)) if(counts > 0.0) else -1
            if(event_writer is not None): event_writer.add_scalar('eval/accuracy-eval-t', accuracy_eval_t, epoch, period=1)
            if(display != 'minimal'): print('Accuracy eval-t: %s' % accuracy_eval_t)

            # Computes the accuracy when the distractor is selected from an evaluation category (never seen during training)
            failure_matrix_eval_d = failure_matrix[:, eval_categories]
            counts_matrix_eval_d = counts_matrix[:, eval_categories]

            counts = counts_matrix_eval_d.sum()
            accuracy_eval_d = (1 - (failure_matrix_eval_d.sum() / counts)) if(counts > 0.0) else -1
            if(event_writer is not None): event_writer.add_scalar('eval/accuracy-eval-d', accuracy_eval_d, epoch, period=1)
            if(display != 'minimal'): print('Accuracy eval-d %s' % accuracy_eval_d)

            # Computes the accuracy when both the target and the distractor are selected from evaluation categories (never seen during training)
            failure_matrix_eval_td = failure_matrix[np.ix_(eval_categories, eval_categories)]
            counts_matrix_eval_td = counts_matrix[np.ix_(eval_categories, eval_categories)]

            counts = counts_matrix_eval_td.sum()
            accuracy_eval_td = (1 - (failure_matrix_eval_td.sum() / counts)) if(counts > 0.0) else -1
            if(event_writer is not None): event_writer.add_scalar('eval/accuracy-eval-td', accuracy_eval_td, epoch, period=1)
            if(display != 'minimal'): print('Accuracy eval-td %s' % accuracy_eval_td)

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
            print('Compositionality measures cannot be computed (%i messages and %i different categories in the sample).' % (len(mes), len(cat)))
        else:
            sample_messages, sample_categories = zip(*sample)
            sample_messages, sample_categories = list(map(tuple, sample_messages)), list(map(tuple, sample_categories))

            #timepoint = time.time()
            l_cor, *_, l_cor_n = compute_correlation.mantel(sample_messages, sample_categories)
            if(display != 'minimal'): print('Levenshtein: %f - %f' % (l_cor, l_cor_n))

            #timepoint2 = time.time()
            #print(timepoint2 - timepoint)
            #timepoint2 = timepoint

            l_n_cor, *_, l_n_cor_n = compute_correlation.mantel(sample_messages, sample_categories, message_distance=compute_correlation.levenshtein_normalised)
            if(display != 'minimal'): print('Levenshtein (normalised): %f - %f' % (l_n_cor, l_n_cor_n))

            #timepoint2 = time.time()
            #print(timepoint2 - timepoint)
            #timepoint2 = timepoint

            j_cor, *_, j_cor_n = compute_correlation.mantel(sample_messages, sample_categories, message_distance=compute_correlation.jaccard, map_msg_to_str=False)
            if(display != 'minimal'): print('Jaccard: %f - %f' % (j_cor, j_cor_n))

            #timepoint2 = time.time()
            #print(timepoint2 - timepoint)
            #timepoint2 = timepoint

            entropy_stats = compute_entropy_stats(sample_messages, sample_categories)
            if(display != 'minimal'):
                print('Entropy category/msgs: min: %f, mean: %f, median: %f, max: %f, var: %f' % (entropy_stats))

            if(event_writer is not None):
                event_writer.add_scalar('eval/Lev-based comp', l_cor, epoch, period=1)
                event_writer.add_scalar('eval/Lev-based comp (normalised)', l_cor_n, epoch, period=1)
                event_writer.add_scalar('eval/Normalised Lev-based comp', l_n_cor, epoch, period=1)
                event_writer.add_scalar('eval/Normalised Lev-based comp (normalised)', l_n_cor_n, epoch, period=1)
                event_writer.add_scalar('eval/Jaccard-based comp', j_cor, epoch, period=1)
                event_writer.add_scalar('eval/Jaccard-based comp (normalised)', j_cor_n, epoch, period=1)
                minH, meanH, medH, maxH, varH = entropy_stats
                event_writer.add_scalar('eval/min Entropy category per msgs', minH, epoch, period=1)
                event_writer.add_scalar('eval/mean Entropy category per msgs', meanH, epoch, period=1)
                event_writer.add_scalar('eval/med Entropy category per msgs', medH, epoch, period=1)
                event_writer.add_scalar('eval/max Entropy category per msgs', maxH, epoch, period=1)
                event_writer.add_scalar('eval/var Entropy category per msgs', varH, epoch, period=1)

        # Decision tree stuff
        alphabet_size = (self.base_alphabet_size + 1)
        n = 1 # Max size of n-grams to consider
        tmp = decision_tree.analyse(messages, categories, alphabet_size, data_iterator.concepts, n)
        (full_tree, full_tree_accuracy) = tmp['full_tree']
        conceptual_trees = tmp['conceptual_trees']

        n_leaves, depth = full_tree.get_n_leaves(), full_tree.get_depth()
        if(event_writer is not None): event_writer.add_scalar('decision_tree/full_accuracy', full_tree_accuracy, epoch, period=1)
        if(display != 'minimal'): print('Full tree accuracy: %s' % full_tree_accuracy)
        if(event_writer is not None): event_writer.add_scalar('decision_tree/full_n_leaves', n_leaves, epoch, period=1)
        if(display != 'minimal'): print('Full tree n leaves: %s' % n_leaves)
        if(event_writer is not None): event_writer.add_scalar('decision_tree/full_depth', depth, epoch, period=1)
        if(display != 'minimal'): print('Full tree depth: %s' % depth)

        for i, (tree, accuracy) in conceptual_trees:
            name = data_iterator.concept_names[i]

            n_leaves, depth = tree.get_n_leaves(), tree.get_depth()
            if(event_writer is not None): event_writer.add_scalar(('decision_tree/%s_accuracy' % name), accuracy, epoch, period=1)
            if(display != 'minimal'): print('%s tree accuracy: %s' % (name, accuracy))
            if(event_writer is not None): event_writer.add_scalar(('decision_tree/%s_n_leaves' % name), n_leaves, epoch, period=1)
            if(display != 'minimal'): print('%s tree n leaves: %s' % (name, n_leaves))
            if(event_writer is not None): event_writer.add_scalar(('decision_tree/%s_depth' % name), depth, epoch, period=1)
            if(display != 'minimal'): print('%s tree depth: %s' % (name, depth))

    @property
    def agents(self):
        return self.sender, self.receiver

    @property
    def optim(self):
        return self._optim

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
