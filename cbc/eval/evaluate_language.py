#!/usr/bin/env python3

from datetime import datetime
import os

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import tqdm

from ..games import AliceBob, AliceBobPopulation
from ..utils.misc import build_optimizer, compute_entropy
from ..utils.data import get_data_loader

def main(args):
    if(not os.path.isdir(args.data_set)):
        print("Directory '%s' not found." % args.data_set)
        sys.exit()
    assert args.load_model is not None, "a valid path to a trained model is required."
    assert args.message_dump_file is not None, "a valid output file is required."

    if(args.population is not None): model = AliceBobPopulation(args)
    else: model = AliceBob.load(args.load_model, args)
    #print(model)

    data_loader = get_data_loader(args)

    #model.eval()
    counts = torch.zeros(args.base_alphabet_size, dtype=torch.float).to(args.device)

    if args.load_other_model is not None:
        other_model = type(model)()
        other_model.load_state_dict(torch.load(args.load_other_model, map_location=args.device))
        other_model.to(args.device)
        counts_other_model == torch.zeros(args.base_alphabet_size, dtype=torch.float).to(args.device)

    with open(args.message_dump_file, 'w') as ostr:
        for datapoint in tqdm.tqdm(data_loader.dataset):
            sender_outcome = model.sender(datapoint.img.unsqueeze(0).to(args.device))
            message = sender_outcome.action[0].view(-1)
            message_str = ' '.join(map(str, message.tolist()))
            category_str = ' '.join(map(str, datapoint.category))
            counts += (torch.arange(args.base_alphabet_size).expand(message.size(0), args.base_alphabet_size) == message.unsqueeze(1)).float().sum(dim=0)
            if args.load_other_model is not None:
                other_sender_outcome = other_model.sender(datapoint.img.unsqueeze(0).to(args.device))
                other_message = other_sender_outcome.action[0].view(-1)
                counts_other_model += (torch.arange(args.base_alphabet_size).expand(other_message.size(0), args.base_alphabet_size) == other_message.unsqueeze(1)).float().sum(dim=0)
            print(datapoint.idx, category_str, message_str, sep='\t', file=ostr)

    uniform = torch.ones_like(counts)

    if args.load_other_model is not None:
        print('entropy:', compute_entropy(counts), 'compared model:', compute_entropy(counts_other_model), 'ref uniform:', compute_entropy(uniform))
    else:
        print('entropy:', compute_entropy(counts), 'ref uniform:', compute_entropy(uniform))
