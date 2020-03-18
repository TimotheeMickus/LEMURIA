#!/usr/bin/env python3

from datetime import datetime
import os

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import tqdm

from .decision_tree import decision_tree

from ..games import AliceBob, AliceBobPopulation
from ..utils.misc import build_optimizer, compute_entropy
from ..utils.data import get_data_loader

def main(args):
    if(not os.path.isdir(args.data_set)):
        print("Directory '%s' not found." % args.data_set)
        sys.exit()
    assert args.load_model is not None, "a valid path to a trained model is required."
    #assert args.message_dump_file is not None, "a valid output file is required."

    if(args.population is not None): model = AliceBobPopulation.load(args.load_model, args)
    else: model = AliceBob.load(args.load_model, args)
    #print(model)

    data_loader = get_data_loader(args)

    #model.eval()
    counts = torch.zeros(args.base_alphabet_size, dtype=torch.float).to(args.device)

    if args.load_other_model is not None:
        other_model = type(model).load(args.load_other_model, args)
        counts_other_model == torch.zeros(args.base_alphabet_size, dtype=torch.float).to(args.device)

    ostr = open(args.message_dump_file, 'w') if(args.message_dump_file is not None) else None
    #with open(args.message_dump_file, 'w') as ostr:
   
    n = len(data_loader)
    if(n is None): n = 10000
    dataset = np.array([data_loader.get_datapoint(i) for i in range(n)])
    
    print("Generating the messages…")
    messages = []
    with torch.no_grad():
        for datapoint in tqdm.tqdm(dataset):
            sender_outcome = model.sender(datapoint.img.unsqueeze(0).to(args.device))
            message = sender_outcome.action[0].view(-1)
            
            messages.append(message.tolist())
            
            message_str = ' '.join(map(str, message.tolist()))
            category_str = ' '.join(map(str, datapoint.category))
            counts += (torch.arange(args.base_alphabet_size).expand(message.size(0), args.base_alphabet_size) == message.unsqueeze(1)).float().sum(dim=0)
            
            if args.load_other_model is not None:
                other_sender_outcome = other_model.sender(datapoint.img.unsqueeze(0).to(args.device))
                other_message = other_sender_outcome.action[0].view(-1)
                counts_other_model += (torch.arange(args.base_alphabet_size).expand(other_message.size(0), args.base_alphabet_size) == other_message.unsqueeze(1)).float().sum(dim=0)

            if(ostr is not None): print(datapoint.idx, category_str, message_str, sep='\t', file=ostr)
    messages = np.array(messages)

    if(ostr is not None): ostr.close()

    # Computes entropy stuff
    uniform = torch.ones_like(counts)

    if args.load_other_model is not None:
        print('entropy:', compute_entropy(counts), 'compared model:', compute_entropy(counts_other_model), 'ref uniform:', compute_entropy(uniform))
    else:
        print('entropy:', compute_entropy(counts), 'ref uniform:', compute_entropy(uniform))

    # Decision tree stuff
    categories = np.array([datapoint.category for datapoint in dataset])
    decision_tree(messages=messages, categories=categories, alphabet_size=(model.base_alphabet_size + 1), concepts=data_loader.concepts)
