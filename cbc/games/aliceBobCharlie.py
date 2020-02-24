import itertools as it

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from .game import Game
from ..agents import Sender, Receiver, Drawer
from ..utils.logging import Progress
from ..utils.misc import show_imgs, max_normalize_, to_color, build_optimizer

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
            rewards, successes, message_length, loss, charlie_acc = self._train_step_alice_bob(batch, self._running_average_success)
        else:
            rewards, successes, message_length, loss, charlie_acc = self._train_step_charlie(batch, self._running_average_success)
        return loss.mean(), rewards, successes, message_length.float().mean().item(), charlie_acc


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

        return rewards, successes, message_length, loss, charlie_acc

    def _train_step_charlie(self, batch, running_avg_success):
        #optim.zero_grad()
        sender_outcome, drawer_outcome, receiver_outcome = self(batch, sender_no_grad=self._charlie_turn())

        bob_action = Categorical(F.softmax(receiver_outcome.scores, dim=-1)).sample()

        total_items = self._bob_input(batch, drawer_outcome).size(1)
        charlie_idx = total_items - 1 # By design, Charlie's image is the last one presented to Bob

        if self.default_adv_train:
            fake_image_score = receiver_outcome.scores[:,charlie_idx]
            target = torch.ones_like(fake_image_score)
            loss = F.binary_cross_entropy(torch.sigmoid(fake_image_score), target)
            # Or, more simply in our case: loss = torch.log(fake_image_score)
        else:
            target = torch.ones_like(bob_action) * charlie_idx
            loss = F.nll_loss(F.log_softmax(receiver_outcome.scores, dim=1), target)

        message_length = sender_outcome.action[1]

        chance_perf = 1. / total_items
        charlie_acc = (bob_action == charlie_idx).float().sum() / bob_action.numel()

        (rewards, successes) = self.compute_rewards(sender_outcome.action, bob_action, running_avg_success, chance_perf)

        return rewards, successes, message_length, loss, charlie_acc

    def end_episode(self, **kwargs):
        self.eval()
        self._current_step += 1
        self._running_average_success = kwargs.get('running_average_success', None)

    def test_visualize(self, data_iterator, learning_rate):
        #TODO
        pass

    def evaluate(self, *vargs, **kwargs):
        pass
        #TODO

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
