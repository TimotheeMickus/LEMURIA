import torch
import torch.nn as nn
import random

from aliceBob import AliceBob
from sender import Sender
from receiver import Receiver
from senderReceiver import SenderReceiver

class AliceBobPopulation(AliceBob):
    def __init__(self, size, shared):
        nn.Module.__init__(self)

        if(shared):
            self._agents = [SenderReceiver() for _ in range(size)]

            self.senders, self.receivers = zip(*[(agent.sender, agent.receiver) for agent in self._agents])
        else:
            self.senders = [Sender() for _ in range(size)]
            self.receivers = [Receiver() for _ in range(size)]
            
            self._agents = (self.senders + self.receivers)
        
        # PyTorch cannot find the parameters of objects that are in a list (like `self.senders` or `self.receivers`)
        # TODO Maybe it is safer to use __setattr__ And be careful about parameters appearing multiple times… I'm not sure what would happen
        parameters = []
        for agent in self._agents: parameters += list(agent.parameters())
        self.agent_parameters = nn.ParameterList(parameters)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        
        #for agent in self._agents: agent.to(*args, **kwargs) # Would that be enough? I'm not sure how `.to` works

        self.senders = [sender.to(*args, **kwargs) for sender in self.senders]
        self.receivers = [receiver.to(*args, **kwargs) for receiver in self.receivers]
        
        return self

    def forward(self, batch):
        """
        Input:
            `batch` is a Batch (a kind of named tuple); 'original_img' and 'target_img' are tensors of shape [BATCH_SIZE, *IMG_SHAPE] and 'base_distractors' is a tensor of shape [BATCH_SIZE, 2, *IMG_SHAPE]
        Output:
            `sender_outcome`, sender.Outcome
            `receiver_outcome`, receiver.Outcome
        """

        sender = random.choice(self.senders)
        receiver = random.choice(self.receivers)

        return self._forward(batch, sender, receiver)
