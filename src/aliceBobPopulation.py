import torch
import torch.nn as nn
import random

from aliceBob import AliceBob
from sender import Sender
from receiver import Receiver
from senderReceiver import SenderReceiver

class AliceBobPopulation(AliceBob):
    def __init__(self, size, shared=True):
        nn.Module.__init__(self)

        if(shared):
            agents = [SenderReceiver() for _ in range(size)]

            self.senders, self.receivers = zip(*[(agent.sender, agent.receiver) for agent in agents])
        else:
            self.senders = [Sender() for _ in range(size)]
            self.receivers = [Receiver() for _ in range(size)]
            
            agents = (self.senders + self.receivers)
        
        # PyTorch cannot find the parameters of objects that are in a list (like `self.senders` or `self.receivers`)
        parameters = []
        for agent in agents: parameters += list(agent.parameters())
        self.agent_parameters = nn.ParameterList(parameters)

    def forward(self, batch):
        """
        Input:
            `batch` is a Batch (a kind of named tuple); 'alice_input' is a tensor of shape [BATCH_SIZE, *IMG_SHAPE] and 'bob_input' is a tensor of shape [BATCH_SIZE, K, *IMG_SHAPE]
        Output:
            `sender_outcome`, sender.Outcome
            `receiver_outcome`, receiver.Outcome
        """

        sender = random.choice(self.senders)
        receiver = random.choice(self.receivers)
        
        sender_outcome = sender(batch.alice_input)
        receiver_outcome = receiver(batch.bob_input, *sender_outcome.action)

        return sender_outcome, receiver_outcome
