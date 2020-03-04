import itertools as it
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import tqdm

from .game import Game
from ..agents import Sender, Receiver, Drawer
from ..utils.misc import show_imgs, max_normalize_, to_color, build_optimizer, pointing
from ..eval import compute_correlation

class AliceBobCharlie(Game):
    def __init__(self, args):
        super(Game, self).__init__()
        self.sender = Sender.from_args(args)
        self.receiver = Receiver.from_args(args)
        self.drawer = Drawer.from_args(args)

        self.penalty = args.penalty
        self.adaptative_penalty = args.adaptative_penalty
        self.grad_clipping = args.grad_clipping or 0
        self.grad_scaling = args.grad_scaling or 0
        self.beta_sender = args.beta_sender
        self.beta_receiver = args.beta_receiver

        self._optim_alice_bob = build_optimizer(it.chain(self.sender.parameters(), self.receiver.parameters()), args.learning_rate)
        self._optim_charlie = build_optimizer(self.drawer.parameters(), args.learning_rate)

        self._current_step = 0
        self._running_average_success = 0
        self.default_adv_train = not args.adv_pointer_training
        self._ignore_charlie = args.ignore_charlie

    def _charlie_turn(self):
        return self._current_step % 2 != 0

    def _alice_input(self, batch):
        return batch.original_img(stack=True)

    def _bob_input(self, batch, drawer_outcome=None):
        if drawer_outcome is not None:
            return torch.cat([batch.target_img(stack=True).unsqueeze(1), batch.base_distractors_img(stack=True), drawer_outcome.image.unsqueeze(1)], dim=1)
        else:
            return torch.cat([batch.target_img(stack=True).unsqueeze(1), batch.base_distractors_img(stack=True)], dim=1)

    def __call__(self, batch, sender_no_grad=False, drawer_no_grad=False):
        """
        Input:
            `batch` is a Batch (a kind of named tuple); 'original_img' and 'target_img' are tensors of shape [args.batch_size, *IMG_SHAPE] and 'base_distractors' is a tensor of shape [args.batch_size, 2, *IMG_SHAPE]
        Output:
            `sender_outcome`, sender.Outcome
            `receiver_outcome`, receiver.Outcome
        """
        with torch.autograd.set_grad_enabled(torch.is_grad_enabled() and (not sender_no_grad)):
            sender_outcome = self.sender(self._alice_input(batch))

        with torch.autograd.set_grad_enabled(torch.is_grad_enabled() and (not drawer_no_grad)):
            drawer_outcome = self.drawer(*sender_outcome.action)

        drawer_outcome_ = None if (not self._charlie_turn() and self._ignore_charlie) else drawer_outcome # Set dummy drawer_outcome if Bob is not trained on Charlie output
        receiver_outcome = self.receiver(self._bob_input(batch, drawer_outcome_), *sender_outcome.action)

        return sender_outcome, drawer_outcome, receiver_outcome

    def to(self, *vargs, **kwargs):
        self.sender, self.receiver, self.drawer = self.sender.to(*vargs, **kwargs), self.receiver.to(*vargs, **kwargs), self.drawer.to(*vargs, **kwargs)
        return self

    def compute_rewards(self, sender_action, receiver_action, running_avg_success, chance_perf):
        """
            returns the reward as well as the success for each element of a batch
        """
        successes = (receiver_action == 0).float() # by design, the first image is the target

        rewards = successes

        if(self.penalty > 0.0):
            msg_lengths = sender_action[1].view(-1).float() # Float casting could be avoided if we upgrade torch to 1.3.1; see https://github.com/pytorch/pytorch/issues/9515 (I believe)
            length_penalties = 1.0 - (1.0 / (1.0 + self.penalty * msg_lengths)) # Equal to 0 when `args.penalty` is set to 0, increases to 1 with the length of the message otherwise

            # TODO J'ai peur que ce système soit un peu trop basique et qu'il encourage le système à être sous-performant - qu'on puisse obtenir plus de reward en faisant exprès de se tromper.
            # TODO DELETE
            if(self.adaptative_penalty):
                improvement_factor = (running_avg_success - chance_perf) / (1 - chance_perf) # Equals 0 when running average equals chance performance, reaches 1 when running average reaches 1
                length_penalties = (length_penalties * min(0.0, improvement_factor))

            rewards = (rewards - length_penalties)

        return (rewards, successes)

    def compute_log_prob(self, sender_log_prob, receiver_log_prob):
        """
        Input:
            `sender_log_prob`, log probs for sender policy
            `receiver_log_prob`, log prob for receiver policy
        Output:
            \bigg(\sum \limits_{l=1}^L \log p_{\pi^s}(m^l_t|m^{<l}_t, u) + \log p_{\pi^L}(u_{t'}|z, U) \bigg)
        """
        log_prob = sender_log_prob.sum(dim=1) + receiver_log_prob
        return log_prob

    def compute_interaction(self, batch, **supplementary_info):
        if not self._charlie_turn():
            rewards, successes, message_length, loss, sender_entropy, receiver_entropy, charlie_acc = self._train_step_alice_bob(batch, self._running_average_success)
        else:
            rewards, successes, message_length, loss, sender_entropy, receiver_entropy, charlie_acc = self._train_step_charlie(batch, self._running_average_success)
        return loss.mean(), rewards, successes, message_length.float().mean().item(), sender_entropy, receiver_entropy, charlie_acc, self._charlie_turn()


    def _train_step_alice_bob(self, batch, running_avg_success):
        #optim.zero_grad()
        sender_outcome, drawer_outcome, receiver_outcome = self(batch, drawer_no_grad=not self._charlie_turn())

        bob_probs = F.softmax(receiver_outcome.scores, dim=-1)
        bob_dist = Categorical(bob_probs)
        bob_action = bob_dist.sample()
        bob_entropy = bob_dist.entropy()
        bob_log_prob = bob_dist.log_prob(bob_action)

        chance_perf = (1.0 / self._bob_input(batch, drawer_outcome).size(1))
        #chance_perf = (1 / (1 + batch.base_distractors.size(1) + 1)) # The chance performance is 1 over the number of images shown to Bob
        (rewards, successes) = self.compute_rewards(sender_outcome.action, bob_action, running_avg_success, chance_perf)
        # TODO On doit séparer Alice et Bob. Comme on en a discuté longement fut un temps, on a de bonnes raisons de ne pas vouloir faire entrer en compte l'image de Charlie pour la reward d'Alice. (je sais comment faire)
        log_prob = self.compute_log_prob(sender_outcome.log_prob, bob_log_prob)
        loss = -(rewards * log_prob)

        #loss = loss.mean()
        # entropy penalties
        loss = loss - (self.beta_sender * sender_outcome.entropy.mean())
        loss = loss - (self.beta_receiver * bob_entropy.mean())

        # backprop
        #loss.backward()

        # Gradient clipping and scaling
        #if(self.grad_clipping > 0): torch.nn.utils.clip_grad_value_(self.parameters(), self.grad_clipping)
        #if(self.grad_scaling > 0): torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_scaling)

        #optim.step()

        message_length = sender_outcome.action[1]

        charlie_idx = self._bob_input(batch, drawer_outcome).size(1)
        charlie_acc = (bob_action == charlie_idx).float().sum() / bob_action.numel()

        return rewards, successes, message_length, loss, sender_outcome.entropy.mean(),  bob_entropy.mean(), charlie_acc

    def _train_step_charlie(self, batch, running_avg_success):
        #optim.zero_grad()
        sender_outcome, drawer_outcome, receiver_outcome = self(batch, sender_no_grad=self._charlie_turn())

        bob_dist = Categorical(F.softmax(receiver_outcome.scores, dim=-1))
        bob_action = bob_dist.sample()

        total_items = self._bob_input(batch, drawer_outcome).size(1)
        charlie_idx = total_items - 1 # By design, Charlie's image is the last one presented to Bob

        if self.default_adv_train:
            fake_image_score = receiver_outcome.scores[:,charlie_idx]
            #target = torch.ones_like(fake_image_score)
            #loss = F.binary_cross_entropy(torch.sigmoid(fake_image_score), target)
            loss = torch.sigmoid(fake_image_score).log() # Or, more simply in our case: loss = fake_image_score)
        else:
            target = torch.ones_like(bob_action) * charlie_idx
            loss = F.nll_loss(F.log_softmax(receiver_outcome.scores, dim=1), target)

        message_length = sender_outcome.action[1]

        chance_perf = 1. / total_items
        charlie_acc = (bob_action == charlie_idx).float().sum() / bob_action.numel()

        (rewards, successes) = self.compute_rewards(sender_outcome.action, bob_action, running_avg_success, chance_perf)

        return rewards, successes, message_length, loss, sender_outcome.entropy.mean(),  bob_dist.entropy().mean(), charlie_acc

    def end_episode(self, **kwargs):
        self.eval()
        self._current_step += 1
        self._running_average_success = kwargs.get('running_avg_success', None)

    def test_visualize(self, data_iterator, learning_rate):
        #TODO
        pass

    def evaluate(self, data_iterator, epoch, event_writer=None, display='tqdm', debug=False, log_lang_progress=True):
        self.eval()

        counts_matrix = np.zeros((data_iterator.nb_categories, data_iterator.nb_categories))
        failure_matrix = np.zeros((data_iterator.nb_categories, data_iterator.nb_categories))

        batch_size = 256
        nb_batch = int(np.ceil(len(data_iterator) / batch_size)) # Doing so makes us see on average at least each data point once; this also means that each cell of the failure matrix is updated (len(data_iterator) / ((nb_categories) * (nb_categories - 1))) time, which can be quite low (~10)

        batch_numbers = range(nb_batch)
        messages = []
        categories = []
        if(display == 'tqdm'): batch_numbers = tqdm.tqdm(range(nb_batch), desc='Eval.')
        for _ in batch_numbers:
            with torch.no_grad():
                batch = data_iterator.get_batch(batch_size, no_evaluation=False, sampling_strategies=['different'], keep_category=True) # We use all categories and use only one distractor from a different category
                sender_outcome = self.sender(self._alice_input(batch))
                receiver_outcome = self.receiver(self._bob_input(batch, None), *sender_outcome.action)


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
        if(display != 'minimal'): print('Accuracy: %s' % accuracy_all)

        eval_categories = data_iterator.evaluation_categories_idx
        if(eval_categories != []):
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
            l_cor, _, _, l_cor_n = compute_correlation.analyze_correlation(sample_messages, sample_categories, scrambling_pool_size=30)
            if(display != 'minimal'): print('Levenshtein: %f - %f' % (l_cor, l_cor_n))

            #timepoint2 = time.time()
            #print(timepoint2 - timepoint)
            #timepoint2 = timepoint

            l_n_cor, _, _, l_n_cor_n = compute_correlation.analyze_correlation(sample_messages, sample_categories, scrambling_pool_size=30, message_distance=compute_correlation.levenshtein_normalised)
            if(display != 'minimal'): print('Levenshtein (normalised): %f - %f' % (l_n_cor, l_n_cor_n))

            #timepoint2 = time.time()
            #print(timepoint2 - timepoint)
            #timepoint2 = timepoint

            j_cor, _, _, j_cor_n = compute_correlation.analyze_correlation(sample_messages, sample_categories, scrambling_pool_size=30, message_distance=compute_correlation.jaccard, map_msg_to_str=False)
            if(display != 'minimal'): print('Jaccard: %f - %f' % (j_cor, j_cor_n))

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
    def agents(self):
        return self.sender, self.receiver, self.drawer

    @property
    def optim(self):
        if self._charlie_turn:
            return self._optim_charlie
        else:
            return self._optim_alice_bob

    def save(self, path):
        """
        Save model to file `path`
        """
        state = {
            'agents_state_dicts': [agent.state_dict() for agent in self.agents],
            'optims': [self._optim_alice_bob, self._optim_charlie],
            'current_step': self._current_step,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path, args, _old_model=False):
        checkpoint = torch.load(path, map_location=args.device)
        instance = cls(args)
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
            drawer_state_dict = {
                k[len('drawer.'):]:checkpoint[k] for k in checkpoint.keys() if k.startswith('drawer.')
            }
            instance.drawer.load_state_dict(drawer_state_dict)
        else:
            for agent, state_dict in zip(instance.agents, checkpoint['agents_state_dicts']):
                agent.load_state_dict(state_dict)
            instance._optim_alice_bob, instance._optim_charlie = checkpoint['optims']
            instance._current_step = checkpoint['current_step']
        return instance

    def pretrain_CNNs(self, data_iterator, summary_writer, args):
        for name, agent in [['sender', self.sender], ['receiver', self.receiver]]:
            self.pretrain_agent_CNN(agent, data_iterator, args, summary_writer, agent_name=name)
