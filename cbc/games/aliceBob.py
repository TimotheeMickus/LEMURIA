import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
import numpy as np
import itertools as it
import tqdm

from collections import defaultdict
from deprecated import deprecated
import random
import time

from ..agents import Sender, Receiver, SenderReceiver
from ..utils.misc import show_imgs, max_normalize_, to_color, pointing, add_normal_noise, compute_entropy, build_optimizer
from ..utils import misc

from ..eval import compute_correlation

from .game import Game

class AliceBob(Game):
    def __init__(self, args):
        self.base_alphabet_size = args.base_alphabet_size
        self.max_len_msg = args.max_len

        if(args.shared):
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
        self.adaptative_penalty = args.adaptative_penalty

        self.optim = build_optimizer(parameters, args.learning_rate)

    def _alice_input(self, batch):
        return batch.original_img(stack=True)

    def _bob_input(self, batch):
        return torch.cat([batch.target_img(stack=True).unsqueeze(1), batch.base_distractors_img(stack=True)], dim=1)

    def compute_interaction(self, batches, *agents, **state_info):
        batch = batches[0]
        sender, receiver = agents
        sender_outcome = sender(self._alice_input(batch))
        receiver_outcome = receiver(self._bob_input(batch), *sender_outcome.action)

        # Alice's part
        (sender_loss, sender_successes, sender_rewards) = self.compute_sender_loss(sender_outcome, receiver_outcome.scores, state_info['running_avg_success'])

        # Bob's part
        receiver_loss = self.compute_receiver_loss(receiver_outcome.scores)

        loss = sender_loss + receiver_loss

        rewards, successes = sender_rewards, sender_successes
        avg_msg_length = sender_outcome.action[1].float().mean().item()
        losses = (loss,)

        return rewards, successes, avg_msg_length, losses

    def to(self, *vargs, **kwargs):
        self.sender, self.receiver = self.sender.to(*vargs, **kwargs), self.receiver.to(*vargs, **kwargs)
        return self

    def __call__(self, batch):
        """
        Input:
            `batch` is a Batch (a kind of named tuple); 'original_img' and 'target_img' are tensors of shape [args.batch_size, *IMG_SHAPE] and 'base_distractors' is a tensor of shape [args.batch_size, 2, *IMG_SHAPE]
        Output:
            `sender_outcome`, sender.Outcome
            `receiver_outcome`, receiver.Outcome
        """
        return self.compute_interaction(batch, self.sender, self.receiver)

    def test_visualize(self, data_iterator, learning_rate):
        self.eval() # Sets the model in evaluation mode; good idea or not?

        batch_size = 4
        batch = data_iterator.get_batch(batch_size) # Standard training batch

        batch.require_grad()

        sender_outcome, receiver_outcome = self([batch])

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

            for j in range(batch.base_distractors.size(1)):
                imgs.append(batch.base_distractors[i][j].img)
                imgs.append(receiver_part_base_distractors[i][j])

            imgs.append(receiver_dream[i].detach())
        #for img in imgs: print(img.shape)
        show_imgs(imgs, nrow=(len(imgs) // batch_size)) #show_imgs(imgs, nrow=(2 * (2 + batch.base_distractors.size(1))))

    def compute_sender_rewards(self, sender_action, receiver_scores, running_avg_success):
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

            # TODO J'ai peur que ce système soit un peu trop basique et qu'il encourage le système à être sous-performant - qu'on puisse obtenir plus de reward en faisant exprès de se tromper.
            if(self.adaptative_penalty):
                chance_perf = (1 / receiver_scores.size(1))
                improvement_factor = (running_avg_success - chance_perf) / (1 - chance_perf) # Equals 0 when running average equals chance performance, reaches 1 when running average reaches 1
                length_penalties = (length_penalties * min(0.0, improvement_factor))

            rewards = (rewards - length_penalties)

        return (rewards, successes)

    def compute_sender_loss(self, sender_outcome, receiver_scores, running_avg_success):
        (rewards, successes) = self.compute_sender_rewards(sender_outcome.action, receiver_scores, running_avg_success)
        log_prob = sender_outcome.log_prob.sum(dim=1)

        loss = -(rewards * log_prob).mean()

        loss = loss - (self.beta_sender * sender_outcome.entropy.mean()) # Entropy penalty

        return (loss, successes, rewards)

    def compute_receiver_loss(self, receiver_scores):
        receiver_pointing = pointing(receiver_scores) # The sampled action is not the same as the one in `sender_rewards` but it probably does not matter

        # By design, the target is the first image
        if(self.use_expectation):
            successes = receiver_pointing['dist'].probs[:, 0].detach()
            log_prob = receiver_pointing['dist'].log_prob(torch.tensor(0).to(probs.device))
        else: # Plays dice
            successes = (receiver_pointing['action'] == 0).float()
            log_prob = receiver_pointing['dist'].log_prob(receiver_pointing['action'])

        rewards = successes

        loss = -(rewards * log_prob).mean()

        loss = loss - (self.beta_receiver * receiver_pointing['dist'].entropy().mean()) # Entropy penalty

        return loss

    def evaluate(self, data_iterator, epoch, event_writer=None, simple_display=False, debug=False, log_lang_progress=True):
        self.eval()

        counts_matrix = np.zeros((data_iterator.nb_categories, data_iterator.nb_categories))
        failure_matrix = np.zeros((data_iterator.nb_categories, data_iterator.nb_categories))

        batch_size = 256
        nb_batch = int(np.ceil(len(data_iterator) / batch_size)) # Doing so makes us see on average at least each data point once; this also means that each cell of the failure matrix is updated (len(data_iterator) / ((nb_categories) * (nb_categories - 1))) time, which can be quite low (~10)

        batch_numbers = range(nb_batch)
        messages = []
        categories = []
        if(not simple_display): batch_numbers = tqdm.tqdm(range(nb_batch), desc='Eval.')
        for _ in batch_numbers:
            with torch.no_grad():
                batch = data_iterator.get_batch(batch_size, no_evaluation=False, sampling_strategies=['different'], keep_category=True) # We use all categories and use only one distractor from a different category
                sender, receiver = self.agents
                sender_outcome = sender(self._alice_input(batch))
                receiver_outcome = receiver(self._bob_input(batch), *sender_outcome.action)

                messages.extend([msg.tolist()[:l] for msg, l in zip(*sender_outcome.action)])
                categories.extend([x.category for x in batch.original])

                receiver_pointing = pointing(receiver_outcome.scores)
                failure = receiver_pointing['dist'].probs[:, 1].cpu().numpy() # Probability of the distractor

                target_category = [data_iterator.category_idx(x.category) for x in batch.original]
                distractor_category = [data_iterator.category_idx(x.category) for base_distractors in batch.base_distractors for x in base_distractors]

                data_iterator.failure_based_distribution.update(target_category, distractor_category, failure)

                np.add.at(counts_matrix, (target_category, distractor_category), 1.0)
                np.add.at(failure_matrix, (target_category, distractor_category), failure)

        # Computes the accuracy when the target is selected from any category
        accuracy_all = 1 - (failure_matrix.sum() / counts_matrix.sum())
        if(event_writer is not None): event_writer.add_scalar('eval/accuracy', accuracy_all, epoch, period=1)
        print('Accuracy: %s' % accuracy_all)

        eval_categories = data_iterator.evaluation_categories_idx
        if(eval_categories != []):
            # Computes the accuracy when the target is selected from an evaluation category (never seen during training)
            failure_matrix_eval_t = failure_matrix[eval_categories, :]
            counts_matrix_eval_t = counts_matrix[eval_categories, :]

            counts = counts_matrix_eval_t.sum()
            accuracy_eval_t = (1 - (failure_matrix_eval_t.sum() / counts)) if(counts > 0.0) else -1
            if(event_writer is not None): event_writer.add_scalar('eval/accuracy-eval-t', accuracy_eval_t, epoch, period=1)
            print('Accuracy eval-t: %s' % accuracy_eval_t)

            # Computes the accuracy when the distractor is selected from an evaluation category (never seen during training)
            failure_matrix_eval_d = failure_matrix[:, eval_categories]
            counts_matrix_eval_d = counts_matrix[:, eval_categories]

            counts = counts_matrix_eval_d.sum()
            accuracy_eval_d = (1 - (failure_matrix_eval_d.sum() / counts)) if(counts > 0.0) else -1
            if(event_writer is not None): event_writer.add_scalar('eval/accuracy-eval-d', accuracy_eval_d, epoch, period=1)
            print('Accuracy eval-d %s' % accuracy_eval_d)

            # Computes the accuracy when both the target and the distractor are selected from evaluation categories (never seen during training)
            failure_matrix_eval_td = failure_matrix[np.ix_(eval_categories, eval_categories)]
            counts_matrix_eval_td = counts_matrix[np.ix_(eval_categories, eval_categories)]

            counts = counts_matrix_eval_td.sum()
            accuracy_eval_td = (1 - (failure_matrix_eval_td.sum() / counts)) if(counts > 0.0) else -1
            if(event_writer is not None): event_writer.add_scalar('eval/accuracy-eval-td', accuracy_eval_td, epoch, period=1)
            print('Accuracy eval-td %s' % accuracy_eval_td)

        # Computes compositionality measures
        # First selects a sample of (message, category) pairs
        size_sample = (30 * data_iterator.nb_categories)

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
            l_cor, _, _, l_cor_n = compute_correlation.analyze_correlation(sample_messages, sample_categories, scrambling_pool_size=30)
            print('Levenshtein: %f - %f' % (l_cor, l_cor_n))

            #timepoint2 = time.time()
            #print(timepoint2 - timepoint)
            #timepoint2 = timepoint

            l_n_cor, _, _, l_n_cor_n = compute_correlation.analyze_correlation(sample_messages, sample_categories, scrambling_pool_size=30, message_distance=compute_correlation.levenshtein_normalised)
            print('Levenshtein (normalised): %f - %f' % (l_n_cor, l_n_cor_n))

            #timepoint2 = time.time()
            #print(timepoint2 - timepoint)
            #timepoint2 = timepoint

            j_cor, _, _, j_cor_n = compute_correlation.analyze_correlation(sample_messages, sample_categories, scrambling_pool_size=30, message_distance=compute_correlation.jaccard, map_msg_to_str=False)
            print('Jaccard: %f - %f' % (j_cor, j_cor_n))

            #timepoint2 = time.time()
            #print(timepoint2 - timepoint)
            #timepoint2 = timepoint

            if(event_writer is not None):
                event_writer.add_scalar('eval/Lev-based comp', l_cor, epoch, period=1)
                event_writer.add_scalar('eval/Lev-based comp (normalised)', l_cor_n, epoch, period=1)
                event_writer.add_scalar('eval/Normalised Lev-based comp', l_n_cor, epoch, period=1)
                event_writer.add_scalar('eval/Normalised Lev-based comp (normalised)', l_n_cor_n, epoch, period=1)
                event_writer.add_scalar('eval/Jaccard-based comp', j_cor, epoch, period=1)
                event_writer.add_scalar('eval/Jaccard-based comp (normalised)', j_cor_n, epoch, period=1)

    @property
    def num_batches_per_episode(self):
        return 1

    @property
    def agents(self):
        return self.sender, self.receiver

    @property
    def optims(self):
        return (self.optim,)

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
            instance.optim = checkpoint['optims'][0]
        return instance
