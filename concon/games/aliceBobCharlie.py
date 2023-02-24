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
    def __init__(self, args, logger): # TODO Currently, the default auto-logger does not log enough information.
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
            self.sender = Sender.from_args(args)
            self.receiver = Receiver.from_args(args)
            self.drawer = Drawer.from_args(args)

        # TODO Using different learning rates would probably prove beneficial.
        self._optim_sender = build_optimizer(self.sender.parameters(), args.learning_rate)
        self._optim_receiver = build_optimizer(self.receiver.parameters(), args.learning_rate)
        self._optim_drawer = build_optimizer(self.drawer.parameters(), args.learning_rate)

        self.score_trackers = {
            'sender': misc.Averager(12800, buffer_f=(lambda size, dtype: torch.zeros(size, dtype=dtype))),
            'receiver': misc.Averager(12800, buffer_f=(lambda size, dtype: torch.zeros(size, dtype=dtype))),
            'drawer': misc.Averager(12800, buffer_f=(lambda size, dtype: torch.zeros(size, dtype=dtype))),
        }

        self.use_baseline = args.use_baseline
        if(self.use_baseline): # In that case, the sender loss will take into account the "baseline term" into the average recent reward.
            self._sender_avg_reward = misc.Averager(size=12800)

        self.correct_only = args.correct_only # Whether to perform the fancy language evaluation using only correct messages (i.e., the one that leads to successful communication).

    def get_drawer(self):
        return self.drawer

    # Overrides AliceBob.all_agents.
    @property
    def all_agents(self):
        return (self.sender, self.receiver, self.drawer)

    # Overrides AliceBob.optim.
    @property
    def optim(self):
        pass # TODO

    # TODO Check what happens to Charlie in terms of pre-training.
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
    def __call__(self, batch):
        """
        Input:
            batch: Batch
        Output:
            sender_outcome: sender.Outcome
            receiver_outcome: receiver.Outcome
        """
        sender = self.get_sender()
        receiver = self.get_receiver()
        drawer = self.get_drawer()

        sender_outcome = sender(self._alice_input(batch))
        drawer_outcome = drawer(*sender_outcome.action)
        receiver_outcome = receiver(self._bob_input(batch, drawer_outcome.image), *sender_outcome.action)

        return sender_outcome, drawer_outcome, receiver_outcome

    # Overrides AliceBob.compute_interaction.
    def compute_interaction(self, batch):
        (sender_outcome, drawer_outcome, receiver_outcome) = self(batch)

        # Alice's part.
        (sender_loss, sender_perf, sender_rewards) = self.compute_sender_loss(sender_outcome, receiver_outcome.scores, contending_imgs=[0, 1])
        sender_entropy = sender_outcome.entropy.mean()

        # Bob's part.
        (receiver_loss, receiver_perf, receiver_entropy) = self.compute_receiver_loss(receiver_outcome.scores, return_entropy=True)

        # Charlie's part.
        (drawer_loss, drawer_perf) = self.compute_drawer_loss(receiver_outcome.scores, contending_imgs=[2, 0])

        # TODO It might be possible to save a lot of computation in the computation
        # of the gradients (for example, when differentiating Bob's loss, no need
        # to propagate the gradient through Charlie). See misc.GradSpigot. 
        optimizers = [self._optim_sender, self._optim_receiver, self._optim_drawer]
        losses = [sender_loss, receiver_loss, drawer_loss]
        scores = torch.tensor([-self.score_trackers[role].get(default=0.0) for role in ["sender", "receiver", "drawer"]])
        temperature = 1.0
        if(temperature != 0.0):
            weights = torch.softmax((scores / temperature), dim=0) # Shape: (3)

            losses = [(o, (w * l)) for (o, w, l) in zip(optimizers, weights, losses)]
        else:
            i = np.argmax(scores)
            losses = [(optimizers[i], losses[i])]

        # Updates each agent's success rate tracker.
        sender_score = ((2 * sender_perf) - 1) # Values usually in [0, 1] (otherwise, there might be a problem). Shape: (batch size)
        self.score_trackers["sender"].update_batch(sender_score.detach())

        receiver_score = ((3 * receiver_perf) - 1) # Values usually in [0, 2] (otherwise, there might be a problem). Shape: (batch size)
        self.score_trackers["receiver"].update_batch(receiver_score.detach())
        
        drawer_score = (2 * drawer_perf) # Values usually in [0, 1] (otherwise, there might be a problem). Shape: (batch size)
        self.score_trackers["drawer"].update_batch(drawer_score.detach())
        
        msg_length = sender_outcome.action[1].float().mean()

        return losses, sender_rewards, sender_perf, msg_length, sender_entropy, receiver_entropy

    # receiver_scores: tensor of shape (batch size, nb img)
    # contending_imgs: None or a list[int] containing the indices of the contending images
    def compute_drawer_loss(self, receiver_scores, target_idx=0, contending_imgs=None):
        if(contending_imgs is None): img_scores = receiver_scores # Shape: (batch size, nb img)
        else: img_scores = torch.stack([receiver_scores[:,i] for i in contending_imgs]) # Shape: (batch size, len(contending_imgs))
        
        # Generates a probability distribution from the scores and points at an image.
        receiver_pointing = misc.pointing(img_scores)

        perf = receiver_pointing['dist'].probs[:, target_idx].detach() # Shape: (batch size)

        loss = -receiver_pointing['dist'].log_prob(torch.tensor(target_idx).to(img_scores.device)).mean() # Shape: ()

        return (loss, perf)

    def test_visualize(self, data_iterator, learning_rate):
        raise NotImplementedError
