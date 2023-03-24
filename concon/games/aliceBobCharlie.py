import torch
import numpy as np
import scipy

from .game import Game
from .aliceBob import AliceBob

from ..agents import Sender, Receiver, Drawer
from ..utils import misc
from ..utils.misc import build_optimizer

# In this game, there is one sender (Alice), one receiver (Bob) and one drawer (Charlie).
# Alice is shown an "original image" and produces a message, Charlie sees the message and produces a "forged image", Bob sees the message and then a "target image", the forged image and a "distractor image".
# Alice is trained to maximize the probability that Bob assigns to the target image when comparing the target image and the distractor image.
# Bob is trained to maximize the probability that he assigns to the target image when comparing the three images.
# Charlie is trained to maximize the probability that Bob assigns to the fake image when comparing the target image and the fake image.
# Alice is trained with REINFORCE; Bob is trained by log-likelihood maximization; Charlie is trained by log-likelihood maximization.
class AliceBobCharlie(AliceBob):
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
            raise NotImplementedError
        else:
            self._sender = Sender.from_args(args)
            self._receiver = Receiver.from_args(args)
            self._drawer = Drawer.from_args(args)

        self.use_spigot = (not args.no_spigot) # A boolean that indicates whether to use GradSpigot·s (one after Charlie's image and one after Bob's encoding of Alice's message). GradSpigot·s are meant for use during training only.
        self.loss_weight_temp = args.loss_weight_temp

        # TODO Using different learning rates would probably prove beneficial.
        self._optim_sender = build_optimizer(self.sender.parameters(), args.learning_rate)
        self._optim_receiver = build_optimizer(self.receiver.parameters(), args.learning_rate)
        self._optim_drawer = build_optimizer(self.drawer.parameters(), args.learning_rate)

        self.score_trackers = {
            'sender': misc.Averager(1280, buffer_f=(lambda size, dtype: torch.zeros(size, dtype=dtype).to(args.device))),
            'receiver': misc.Averager(1280, buffer_f=(lambda size, dtype: torch.zeros(size, dtype=dtype).to(args.device))),
            'drawer': misc.Averager(1280, buffer_f=(lambda size, dtype: torch.zeros(size, dtype=dtype).to(args.device))),
        }

        self.weights_sum = torch.zeros(3, device=args.device) # Shape: (3)
        self.weights_average_log_frequency = 10
        self.weights_average_log_counter = 0

        self.use_baseline = args.use_baseline
        if(self.use_baseline): # In that case, the sender loss will take into account the "baseline term" into the average recent reward.
            self._sender_avg_reward = misc.Averager(size=12800)

        self.correct_only = args.correct_only # Whether to perform the fancy language evaluation using only correct messages (i.e., the one that leads to successful communication).

        self.debug = args.debug

    @property
    def drawer(self):
        return self._drawer

    # Overrides AliceBob.all_agents.
    @property
    def all_agents(self):
        return (self.sender, self.receiver, self.drawer)

    # Overrides AliceBob.optims.
    @property
    def optims(self):
        return [self._optim_sender, self._optim_receiver, self._optim_drawer]

    # Overrides AliceBob.agents_for_CNN_pretraining.
    def agents_for_CNN_pretraining(self):
        if(self.shared): raise NotImplementedError
        return self.all_agents

    # batch: Batch
    # forged_img: tensor of shape [args.batch_size, *IMG_SHAPE]
    # Overrides AliceBob._bob_input.
    def _bob_input(self, batch, forged_img=None):
        if(forged_img is None): return torch.cat([batch.target_img(stack=True).unsqueeze(1), batch.base_distractors_img(stack=True)], dim=1)
        return torch.cat([batch.target_img(stack=True).unsqueeze(1), batch.base_distractors_img(stack=True), forged_img.unsqueeze(1)], dim=1)

    # Overrides AliceBob.__call__.
    # use_spigot: boolean that indicates whether to use GradSpigot·s (one after Charlie's image and one after Bob's encoding of Alice's message); GradSpigot·s are meant for use during training only
    def __call__(self, batch, use_spigot=False):
        """
        Input:
            batch: Batch
        Output:
            sender_outcome: sender.Outcome
            receiver_outcome: receiver.Outcome
        """
        sender = self.sender
        receiver = self.receiver
        drawer = self.drawer

        sender_outcome = sender(self._alice_input(batch))
        drawer_outcome = drawer(*sender_outcome.action, use_spigot=use_spigot)
        receiver_outcome = receiver(self._bob_input(batch, drawer_outcome.image), *sender_outcome.action, use_spigot=use_spigot)

        return (sender_outcome, drawer_outcome, receiver_outcome)

    # Overrides AliceBob.compute_interaction.
    def compute_interaction(self, batch):
        # TODO: change return signature to loss, {dict of things to log}

        # Predictions.
        (sender_outcome, drawer_outcome, receiver_outcome) = self(batch, use_spigot=self.use_spigot)

        # Alice's loss.
        (sender_loss, sender_perf, sender_rewards) = self.compute_sender_loss(sender_outcome, receiver_outcome.scores, contending_imgs=[0, 1])
        sender_entropy = sender_outcome.entropy.mean()

        # Bob's loss.
        (receiver_loss, receiver_perf, receiver_entropy) = self.compute_receiver_loss(receiver_outcome.scores, return_entropy=True)

        # Charlie's loss.
        (drawer_loss, drawer_perf) = self.compute_drawer_loss(receiver_outcome.scores, contending_imgs=[2, 0])

        scores = torch.tensor([self.score_trackers[role].get(default=0.0) for role in ["sender", "receiver", "drawer"]], device=sender_loss.device) # Shape: (3)
        if(self.loss_weight_temp != 0.0): weights = torch.softmax((-scores / self.loss_weight_temp), dim=0) # Shape: (3)
        else: weights = torch.nn.functional.one_hot(torch.argmax(-scores), 3) # Shape: (3)
        #else: weights = torch.ones_like(-scores) # Only one of the value will be used. Shape: (3)
        if(self.debug): weights = torch.tensor([1.0, 1.0, 0.0], device=weights.device) # DEBUG ONLY 2023-03-09 Deactivate Charlie's training.
        losses = torch.stack([sender_loss, receiver_loss, drawer_loss]) # Shape: (3)
        weighted_losses = weights * losses # Shape: (3)

        self.weights_sum += weights
        self.weights_average_log_counter += 1
        if(self.weights_average_log_counter % self.weights_average_log_frequency) == 0:
            weights_average = self.weights_sum / self.weights_average_log_frequency # Shape: (3)
            self.weights_sum = torch.zeros_like(self.weights_sum) # Shape: (3)
            # self.weights_average_log_counter = 0
            for idx, agent in enumerate('ABC'):
                self.autologger._write(
                    f'train/weight_{agent}',
                    weights_average[idx].item(),
                    self.weights_average_log_counter,
                    direct=True,
                )
                self.autologger._write(
                    f'train/score_{agent}',
                    scores[idx].item(),
                    self.weights_average_log_counter,
                    direct=True,
                )
                self.autologger._write(
                    f'train/loss_{agent}',
                    losses[idx].item(),
                    self.weights_average_log_counter,
                    direct=True,
                )

        optimization = []

        # Alice backward.
        optim = self._optim_sender
        loss = weighted_losses[0]
        agent = self.sender

        optimization.append((optim, loss.detach(), misc.get_backward_f(loss, agent)))

        # Bob backward.
        optim = self._optim_receiver
        loss = weighted_losses[1]
        agent = self.receiver
        spigot = receiver_outcome.msg_spigot # None or a GradSpigot.

        optimization.append((optim, loss.detach(), misc.get_backward_f(loss, agent, spigot)))

        # Charlie backward.
        optim = self._optim_drawer
        loss = weighted_losses[2]
        agent = self.drawer
        spigot = drawer_outcome.img_spigot # None or a GradSpigot.

        optimization.append((optim, loss.detach(), misc.get_backward_f(loss, agent, spigot)))

        if(self.loss_weight_temp == 0.0): optimization = [optimization[np.argmax(-scores)]]

        # Updates each agent's success rate tracker.
        sender_score = ((2 * sender_perf) - 1) # Values usually in [0, 1] (otherwise, there might be a problem). Shape: (batch size)
        self.score_trackers["sender"].update_batch(sender_score.detach())

        receiver_score = ((3 * receiver_perf) - 1) # Values usually in [0, 2] (otherwise, there might be a problem). Shape: (batch size)
        self.score_trackers["receiver"].update_batch(receiver_score.detach())

        drawer_score = (2 * drawer_perf) # Values usually in [0, 1] (otherwise, there might be a problem). Shape: (batch size)
        self.score_trackers["drawer"].update_batch(drawer_score.detach())

        msg_length = sender_outcome.action[1].float().mean()

        return optimization, sender_rewards, sender_perf, msg_length, sender_entropy, receiver_entropy

    # receiver_scores: tensor of shape (batch size, nb img)
    # contending_imgs: None or a list[int] containing the indices of the contending images
    def compute_drawer_loss(self, receiver_scores, target_idx=0, contending_imgs=None):
        if(contending_imgs is None): img_scores = receiver_scores # Shape: (batch size, nb img)
        else: img_scores = torch.stack([receiver_scores[:,i] for i in contending_imgs]) # Shape: (batch size, len(contending_imgs))

        # Generates a probability distribution from the scores and points at an image.
        receiver_pointing = misc.pointing(img_scores)

        perf = receiver_pointing['dist'].probs[:, target_idx].detach() # Shape: (batch size)

        loss = 0.0

        log_prob = receiver_pointing['dist'].log_prob(torch.tensor(target_idx, device=img_scores.device)) # The log-probabilities of the target images. Shape: (batch size)

        cross_entropy_loss = -log_prob.mean() # Shape: ()
        loss += cross_entropy_loss

        return (loss, perf)

    def test_visualize(self, data_iterator, learning_rate):
        raise NotImplementedError
