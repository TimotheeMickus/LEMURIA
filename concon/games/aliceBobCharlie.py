
from .game import Game

from ..utils import misc

# In this game, there is one sender (Alice), one receiver (Bob) and one drawer (Charlie).
# Alice is shown an "original image" and produces a message, Charlie sees the message and produces a "forged image", Bob sees the message and then a "target image", the forged image and a "distractor image".
# Alice is trained to maximize the probability that Bob assigns to the target image when comparing the target image and the distractor image.
# Bob is trained to maximize the probability that he assigns to the target image when comparing the three images.
# Charlie is trained to maximize the probability that Bob assigns to the fake image when comparing the target image and the fake image.
class AliceBobCharlie(Game):
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
            self.sender = Sender.from_args(args)
            self.receiver = Receiver.from_args(args)
            self.drawer = Drawer.from_args(args)
            
            parameters = it.chain(self.sender.parameters(), self.receiver.parameters(), self.drawer.parameters())

        self._optim = build_optimizer(parameters, args.learning_rate)

        self.success_rate_trackers = {
            'sender': misc.Averager(12800),
            'receiver': misc.Averager(12800),
            'drawer': misc.Averager(12800),
        }
        
        self.use_baseline = args.use_baseline
        if(self.use_baseline): # In that case, the loss will take into account the "baseline term", into the average recent reward.
            self._sender_avg_reward = misc.Averager(size=12800)
            self._receiver_avg_reward = misc.Averager(size=12800)
            self._drawer_avg_reward = misc.Averager(size=12800)

        self.correct_only = args.correct_only # Whether to perform the fancy language evaluation using only correct messages (i.e., the one that leads to successful communication).

    def get_sender(self):
        return self._sender

    def get_receiver(self):
        return self._receiver

    def get_drawer(self):
        return self._drawer

    @property
    def all_agents(self):
        return (self._sender, self._receiver, self._drawer)

    @property
    def current_agents(self):
        return all_agents

    def agents_for_CNN_pretraining(self):
        if(self.shared): raise NotImplementedError
        return self.all_agents

    # batch: Batch
    # forged_img: tensor of shape [args.batch_size, *IMG_SHAPE]
    # Overrides AliceBob._bob_input.
    def _bob_input(self, batch, forged_img):
        return torch.cat([batch.target_img(stack=True).unsqueeze(1), batch.base_distractors_img(stack=True)], forged_img, dim=1)

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

