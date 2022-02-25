from collections import deque
import itertools as it
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import tqdm
import torchvision

from .game import Game
from ..agents import Sender, Receiver, Drawer
from ..utils.misc import max_normalize_, to_color, build_optimizer, pointing, compute_entropy_stats, pointing
from ..utils.logging import LoggingData
from ..eval import compute_correlation

from .aliceBob import AliceBob

class AliceBobCharlie(AliceBob):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.drawer = Drawer.from_args(args)
        self._optim = None
        self._optim_alice = build_optimizer(self.sender.parameters(), args.learning_rate)
        self._optim_bob = build_optimizer(self.receiver.parameters(), args.learning_rate)
        self._optim_charlie = build_optimizer(self.drawer.parameters(), args.learning_rate)
        # self.switch_charlie(False)
        self.trained_agent = None
        self._cycle_length = args.n_discriminator_batches
        # self._n_batches_seen = 0

        self.test_output_charlie_path = args.test_output_charlie_path
        self._agent_cycle = it.cycle(['Alice', 'Bob', 'Charlie'])
        self.dynamic_batch_cycle = args.dynamic_batch_cycle
        if self.dynamic_batch_cycle:
            self.success_rate_trackers = {
                'Alice': deque(maxlen=self._cycle_length), #deque(maxlen=args.n_discriminator_batches)),
                'Bob': deque(maxlen=self._cycle_length), #deque(maxlen=args.n_discriminator_batches)),
                'Charlie': deque(maxlen=self._cycle_length), #deque(maxlen=args.n_discriminator_batches)),
            }
            self._current_step = 1


    def _mean_success_rate(self, agent):
        n_obs = len(self.success_rate_trackers[agent])
        if n_obs == 0:
            return 0
        if agent == 'Alice':
            return sum(
                success_rate / 2
                for success_rate in self.success_rate_trackers['Alice']
            ) / n_obs
        return sum(
            success_rate # should earlier success be less and less relevant?
            for success_rate in self.success_rate_trackers[agent]
        ) / n_obs

    def switch_trained_agent(self):
        if self.dynamic_batch_cycle:
            self.trained_agent = sorted(self.success_rate_trackers, key=self._mean_success_rate)[0]
        else:
            self.trained_agent = next(self._agent_cycle)
        alice, bob, charlie = self.agents
        # turn everything off
        alice.eval()
        #     for p in agent.parameters():
        #         p.requires_grad = False
        # pick and choose the agent to train
        if self.trained_agent == 'Alice':
            # for p in self.sender.parameters():
            #     p.requires_grad = True
            alice.train()
            self._optim = self._optim_alice
        elif self.trained_agent == 'Bob':
            # for p in self.receiver.parameters():
            #     p.requires_grad = True
            # bob.train()
            self._optim = self._optim_bob
        elif self.trained_agent == 'Charlie':
            # for p in self.drawer.parameters():
            #     p.requires_grad = True
            # charlie.train()
            self._optim = self._optim_charlie
        # configure autologger
        # if self.autologger.summary_writer is not None:
        #     self.autologger.summary_writer.prefix = self.trained_agent
    # def switch_charlie(self, train_charlie):
    #     self.train_charlie = train_charlie
    #     for a in super().agents:
    #         for p in a.parameters():
    #             p.requires_grad = not train_charlie
    #     for p in self.drawer.parameters():
    #         p.requires_grad = train_charlie
    #     self._optim = self._optim_charlie if train_charlie else self._optim_alice_bob
    #     if(self.autologger.summary_writer is not None):
    #         self.autologger.summary_writer.prefix = "Charlie" if train_charlie else None

    def start_episode(self, train_episode=True):
        super().start_episode(train_episode=train_episode)
        self.switch_trained_agent()
        #
        # if train_episode:
        #     if((self._n_batches_seen % self._n_batches_cycle) == 0):
        #         # if we are at the end of a batch cycle, turn charlie off
        #         self.switch_charlie(False)
        #
        #     elif(((self._n_batches_seen + 1) % self._n_batches_cycle) == 0):
        #         # if we are at the last batch of a batch cycle, turn charlie on
        #         self.switch_charlie(True)
        #
        #     self._n_batches_seen += 1

    @property
    def optims(self):
        return (self._optim_alice, self._optim_bob, self._optim_charlie)

    @property
    def agents(self):
        return self.sender, self.receiver, self.drawer

    def get_label_and_prefix(self):
        return self.trained_agent[0], self.trained_agent + '-'

    def compute_interaction(self, batch):
        if self.trained_agent == 'Alice':
            loss, logging_data_ = super().compute_interaction(batch)
            short_label, prefix = self.get_label_and_prefix()
            logging_data = LoggingData(
                number_ex_seen=logging_data_.number_ex_seen,
                pbar_items={short_label: v for v in logging_data_.pbar_items.values()},
                summary_items={
                    prefix + k: v
                    for k,v in logging_data_.summary_items.items()
                }
            )
        else:
            loss, logging_data = self.compute_interaction_charlie(batch)
        if self.dynamic_batch_cycle and any(agent.training for agent in self.agents):
            self.success_rate_trackers[self.trained_agent].append(logging_data.summary_items[f'{self.trained_agent}-train/success'])
            # if we have perfs stats for all agents, plug that to logger
            if all(map(len, self.success_rate_trackers.values())):
                A = self.success_rate_trackers['Alice'][-1]
                C = 2 * abs(0.5 - self.success_rate_trackers['Charlie'][-1])
                logging_data.summary_items[f'Shared-train/perf_overview'] = (2 * A * C) / (A+C)
            # reset trackers if necessary
            if (self._current_step % self._cycle_length) == 0:
                #print(f'{self._current_step} % {self._cycle_length} == 0, clearing trackers')
                for agent in self.success_rate_trackers.keys():
                    prop_batch_assigned = len(self.success_rate_trackers[agent]) / self._cycle_length
                    logging_data.summary_items[f'Shared-train/{agent}_batch_prop'] = prop_batch_assigned
                    self.success_rate_trackers[agent] = deque(maxlen=self._cycle_length)
            self._current_step += 1
        return loss, logging_data

    def _charlied_bob_input(self, batch, charlie_img, deterministic):
        images = [
            batch.target_img(stack=True).unsqueeze(1),
            batch.base_distractors_img(stack=True),
            charlie_img.unsqueeze(1),
        ]
        # repl targets and distractors if message is gumbelish
        if not deterministic:
            images[0] = images[0].unsqueeze(2).expand_as(images[2])
            images[1] = images[1].unsqueeze(2).expand_as(images[2])
        # put charlie's img first if we're training it
        if self.trained_agent == 'Charlie':
            images = [images[-1]] + images[:-1]
        return torch.cat(images, dim=1)

    def to(self, *vargs, **kwargs):
        _ = super().to(*vargs, **kwargs)
        self.drawer = self.drawer.to(*vargs, **kwargs)
        return self

    def get_drawer(self):
        return self.drawer

    def _compute_interaction_charlie_gumbel(self, batch):
        sender, receiver, drawer = self.agents
        # send
        sender_outcome = sender(self._alice_input(batch))

        # adversarial step
        drawer_outcome = drawer(*sender_outcome.action)

        # receive
        deterministic = sender_outcome.action[2]
        input_with_charlie_img = self._charlied_bob_input(batch, drawer_outcome.image, deterministic)
        receiver_outcome = receiver(input_with_charlie_img, *sender_outcome.action)

        probs = F.log_softmax(receiver_outcome.scores, dim=-1)
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

        short_label, prefix = self.get_label_and_prefix()
        logging_data = LoggingData(
            number_ex_seen=batch.size,
            pbar_items={short_label: successes.mean().item()},
            summary_items={
                prefix + 'train/loss': loss.mean().item(),
                prefix + 'train/success':  successes.mean().item(),
                prefix + 'train/msg_length': avg_msg_length,
            }
        )
        return loss, logging_data

    def compute_interaction_charlie(self, batch):
        if self.is_gumbel and self.trained_agent == 'Alice':
            return self._compute_interaction_charlie_gumbel(batch)
        sender, receiver, drawer = self.agents
        # send
        sender_outcome = sender(self._alice_input(batch))

        # adversarial step
        drawer_outcome = drawer(*sender_outcome.action)
        deterministic = sender_outcome.action[2]
        # receive
        receiver_outcome = receiver(self._charlied_bob_input(batch, drawer_outcome.image, deterministic), *sender_outcome.action)

        if not self.is_gumbel:
            receiver_loss = self.compute_receiver_loss(receiver_outcome.scores, return_entropy=True)

            # training only Charlie for now
            loss = receiver_loss
            (sender_loss, sender_successes, sender_rewards) = self.compute_sender_loss(sender_outcome, receiver_outcome.scores)
            rewards, successes = sender_rewards, sender_successes
            avg_msg_length = sender_outcome.action[1].float().mean().item()

            short_label, prefix = self.get_label_and_prefix()
            logging_data = LoggingData(
                number_ex_seen=batch.size,
                pbar_items={short_label: successes.mean().item()},
                summary_items={
                    prefix + 'train/loss': loss.mean().item(),
                    prefix + 'train/sender_loss': sender_loss.mean.item(),
                    prefix + 'train/receiver_loss': receiver_loss.mean.item(),
                    prefix + 'train/rewards': rewards,
                    prefix + 'train/success':  successes.mean().item(),
                    prefix + 'train/msg_length': avg_msg_length,
                    prefix + 'train/sender_entropy': sender_outcome.entropy.mean().item(),
                    prefix + 'train/receiver_entropy': receiver_entropy.item(),
                }
            )
            return loss, rewards, successes, avg_msg_length, sender_outcome.entropy.mean(), receiver_entropy

        else:
            probs = F.log_softmax(receiver_outcome.scores, dim=-1)
            targets = torch.zeros(probs.size(0), dtype=torch.long).to(probs.device)
            loss = F.nll_loss(probs, targets)
            avg_msg_length = sender_outcome.action[1].float().mean().item()
            successes = (probs.argmax(-1) == 0).float()
            short_label, prefix = self.get_label_and_prefix()
            logging_data = LoggingData(
                number_ex_seen=batch.size,
                pbar_items={short_label: successes.mean().item()},
                summary_items={
                    prefix + 'train/loss': loss.mean().item(),
                    prefix + 'train/success':  successes.mean().item(),
                    prefix + 'train/msg_length': avg_msg_length,
                }
            )
            return loss, logging_data

    def get_images(self, batch):
        with torch.no_grad():
            sender, drawer = self.get_sender(), self.get_drawer()
            # send
            action = sender(self._alice_input(batch)).action
            images = drawer(*action).image
        return torch.cat([batch.target_img(stack=True).unsqueeze(1), images.unsqueeze(1)], dim=1)

    def evaluate(self, data_iterator, epoch):
        for agents in self.agents:
            agents.eval()
        if(self.autologger.summary_writer is not None):
            self.autologger.summary_writer.prefix = 'shared'
        if((epoch % 10 == 0) and (self.test_output_charlie_path is not  None)):
            if(self.autologger.display != 'minimal'): print('constructing sample for epoch', epoch)
            batch = data_iterator.get_batch(256, data_type='test', no_evaluation=False, sampling_strategies=['different'])
            images = self.get_images(batch).flatten(end_dim=1)
            images = torchvision.utils.make_grid(images, 8)
            torchvision.transforms.functional.to_pil_image(images).save(f"{self.test_output_charlie_path}epoch_{epoch}.png")

        self.trained_agent = None  # also controls image ordering
        #self.switch_charlie(False)
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

        # these are not going to change, since this is not inherited by the population settings
        sender, receiver, drawer = self.agents


        messages = []
        categories = []
        batch_numbers = range(nb_batch)
        if(self.autologger.display == 'tqdm'): batch_numbers = tqdm.tqdm(batch_numbers, desc='Eval.')
        with torch.no_grad():
            success = [] # Binary
            success_prob = [] # Probabilities
            scrambled_success_prob = [] # Probabilities
            charlie_success = [] # Binary

            for _ in batch_numbers:
                #self.start_episode(train_episode=False) # Select agents at random if necessary
                batch = data_iterator.get_batch(batch_size, data_type='test', no_evaluation=False, sampling_strategies=['different'], keep_category=True) # We use all categories and use only one distractor from a different category
                # send
                sender_outcome = sender(self._alice_input(batch))
                # adversarial step
                drawer_outcome = drawer(*sender_outcome.action)
                deterministic = sender_outcome.action[2]
                assert deterministic
                # receive
                bob_input = self._charlied_bob_input(batch, drawer_outcome.image, deterministic)
                receiver_outcome = receiver(bob_input, *sender_outcome.action)

                receiver_pointing = pointing(receiver_outcome.scores, argmax=True)
                success.append((receiver_pointing['action'] == 0).float())
                success_prob.append(receiver_pointing['dist'].probs[:, 0]) # Probability of the target
                charlie_success.append((receiver_pointing['action'] == 2).float())


                target_category = [data_iterator.category_idx(x.category) for x in batch.original]
                distractor_category = [data_iterator.category_idx(x.category) for base_distractors in batch.base_distractors for x in base_distractors]

                failure = receiver_pointing['dist'].probs[:, 1:].sum(1).cpu().numpy() # Probability of the distractor(s)
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

                scrambled_receiver_outcome = self.get_receiver()(bob_input, message=scrambled_messages, length=sender_outcome.action[1], deterministic=True)
                scrambled_receiver_pointing = pointing(scrambled_receiver_outcome.scores)
                scrambled_success_prob.append(scrambled_receiver_pointing['dist'].probs[:, 0])

            success_prob = torch.stack(success_prob)
            scrambled_success_prob = torch.stack(scrambled_success_prob)
            scrambling_resistance = (torch.stack([success_prob, scrambled_success_prob]).min(0).values.mean().item() / success_prob.mean().item()) # Between 0 and 1. We take the min in order to not count messages that become accidentaly better after scrambling
            log('Shared-eval/scrambling-resistance', scrambling_resistance)

            # Here, we try to see how much the messages describe the categories and not the praticular images
            # To do so, we use the original image as target, and an image of the same category as distractor
            abstractness = []
            n = (32 * data_iterator.nb_categories)
            n = min(max_datapoints, n)
            nb_batch = int(np.ceil(n / batch_size))
            for _ in range(nb_batch):
                batch = data_iterator.get_batch(batch_size, data_type='test', no_evaluation=False, sampling_strategies=['same'], target_is_original=True, keep_category=True)
                # send
                sender_outcome = sender(self._alice_input(batch))
                # adversarial step
                drawer_outcome = drawer(*sender_outcome.action)
                deterministic = sender_outcome.action[2]
                assert deterministic
                # receive
                bob_input = self._charlied_bob_input(batch, drawer_outcome.image, deterministic)
                receiver_outcome = receiver(bob_input, *sender_outcome.action)
                receiver_pointing = pointing(receiver_outcome.scores)
                abstractness.append(1 - (receiver_pointing['dist'].probs[:, 1] - receiver_pointing['dist'].probs[:, 0]).abs())

            abstractness = torch.stack(abstractness)
            abstractness_rate = abstractness.mean().item()
            log('Shared-eval/abstractness2', abstractness_rate)

            abstractness = []
            n = (32 * data_iterator.nb_categories)
            n = min(max_datapoints, n)
            nb_batch = int(np.ceil(n / batch_size))
            for _ in range(nb_batch):
                batch = data_iterator.get_batch(batch_size, data_type='test', no_evaluation=False, sampling_strategies=['same'], target_is_original=True, keep_category=True)
                # send
                sender_outcome = sender(self._alice_input(batch))
                # adversarial step
                drawer_outcome = drawer(*sender_outcome.action)
                deterministic = sender_outcome.action[2]
                assert deterministic
                # receive
                bob_input = self._charlied_bob_input(batch, drawer_outcome.image, deterministic)
                receiver_outcome = receiver(bob_input, *sender_outcome.action)

                receiver_pointing = pointing(receiver_outcome.scores)
                abstractness.append(receiver_pointing['dist'].probs[:, 1] * 2.0)

            abstractness = torch.stack(abstractness)
            abstractness_rate = abstractness.mean().item()
            log('Shared-eval/abstractness', abstractness_rate)

            # Here we computing the actual success rate with argmax pointing, and not the mean expected success based on probabilities like is done after
            success = torch.stack(success)
            success_rate = success.mean().item()
            log('Shared-eval/success_rate', success_rate)

            charlie_success = torch.stack(charlie_success)
            charlie_success_rate = charlie_success.mean().item()
            log('Shared-eval/charlie_success_rate', charlie_success_rate)

            # Computes the accuracy when the images are selected from all categories
            accuracy_all = 1 - (failure_matrix.sum() / counts_matrix.sum())
            log('Shared-eval/accuracy', accuracy_all)

            train_categories = data_iterator.training_categories_idx
            eval_categories = data_iterator.evaluation_categories_idx
            if(eval_categories != []):
                # Computes the accuracy when both the target and the distractor are selected from training categories
                failure_matrix_train_td = failure_matrix[np.ix_(train_categories, train_categories)]
                counts_matrix_train_td = counts_matrix[np.ix_(train_categories, train_categories)]

                counts = counts_matrix_train_td.sum()
                accuracy_train_td = (1 - (failure_matrix_train_td.sum() / counts)) if(counts > 0.0) else -1
                log('Shared-eval/accuracy-train-td', accuracy_train_td)

                # Computes the accuracy when the target is selected from an evaluation category (never seen during training)
                failure_matrix_eval_t = failure_matrix[eval_categories, :]
                counts_matrix_eval_t = counts_matrix[eval_categories, :]

                counts = counts_matrix_eval_t.sum()
                accuracy_eval_t = (1 - (failure_matrix_eval_t.sum() / counts)) if(counts > 0.0) else -1
                log('Shared-eval/accuracy-eval-t', accuracy_eval_t)

                # Computes the accuracy when the distractor is selected from an evaluation category (never seen during training)
                failure_matrix_eval_d = failure_matrix[:, eval_categories]
                counts_matrix_eval_d = counts_matrix[:, eval_categories]

                counts = counts_matrix_eval_d.sum()
                accuracy_eval_d = (1 - (failure_matrix_eval_d.sum() / counts)) if(counts > 0.0) else -1
                log('Shared-eval/accuracy-eval-d', accuracy_eval_d)

                # Computes the accuracy when both the target and the distractor are selected from evaluation categories (never seen during training)
                failure_matrix_eval_td = failure_matrix[np.ix_(eval_categories, eval_categories)]
                counts_matrix_eval_td = counts_matrix[np.ix_(eval_categories, eval_categories)]

                counts = counts_matrix_eval_td.sum()
                accuracy_eval_td = (1 - (failure_matrix_eval_td.sum() / counts)) if(counts > 0.0) else -1
                log('Shared-eval/accuracy-eval-td', accuracy_eval_td)

        if(self.autologger.summary_writer is not None):
            self.autologger.summary_writer.prefix = 'AliceBob'
        super().evaluate(data_iterator, epoch)
        # for agents in self.agents:
        #     agents.train()
        #self.switch_charlie(training_state)
