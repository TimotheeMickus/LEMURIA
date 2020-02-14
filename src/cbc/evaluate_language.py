#!/usr/bin/env python3

from datetime import datetime
import os

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import tqdm

from data import get_data_loader

from config import *

# [START] Imports shared code from the parent directory
parent_dir_path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(parent_dir_path)

from aliceBob import AliceBob
from aliceBobPopulation import AliceBobPopulation
from utils import build_optimizer, compute_entropy

sys.path.remove(parent_dir_path)

if(__name__ == "__main__"):
    if(not os.path.isdir(args.data_set)):
        print("Directory '%s' not found." % args.data_set)
        sys.exit()
    assert args.load_model is not None, "a valid path to a trained model is required."
    assert args.message_dump_file is not None, "a valid output file is required."

    if(args.population > 1): model = AliceBobPopulation(args.population)
    else: model = AliceBob()
    model.load_state_dict(torch.load(args.load_model, map_location=args.device))
    model = model.to(args.device)
    #print(model)

    data_loader = get_data_loader(args.same_img)

    model.eval()
    counts = torch.zeros(args.alphabet, dtype=torch.float).to(args.device)

    if args.load_other_model is not None:
        other_model = type(model)()
        other_model.load_state_dict(torch.load(args.load_other_model, map_location=args.device))
        other_model.to(args.device)
        counts_other_model == torch.zeros(args.alphabet, dtype=torch.float).to(args.device)

    with open(args.message_dump_file, 'w') as ostr:
        for datapoint in tqdm.tqdm(data_loader.dataset):
            sender_outcome = model.sender(datapoint.img.unsqueeze(0).to(args.device))
            message = sender_outcome.action[0].view(-1)
            message_str = ' '.join(map(str, message.tolist()))
            category_str = ' '.join(map(str, datapoint.category))
            counts += (torch.arange(args.alphabet_size).expand(message.size(0), args.alphabet_size) == message.unsqueeze(1)).float().sum(dim=0)
            if args.load_other_model is not None:
                other_sender_outcome = other_model.sender(datapoint.img.unsqueeze(0).to(args.device))
                other_message = other_sender_outcome.action[0].view(-1)
                counts_other_model += (torch.arange(args.alphabet).expand(other_message.size(0), args.alphabet) == other_message.unsqueeze(1)).float().sum(dim=0)
            print(datapoint.idx, category_str, message_str, sep='\t', file=ostr)

uniform = torch.ones_like(counts)

if args.load_other_model is not None:
    print('entropy:', compute_entropy(counts), 'compared model:', compute_entropy(counts_other_model), 'ref uniform:', compute_entropy(uniform))
else:
    print('entropy:', compute_entropy(counts), 'ref uniform:', compute_entropy(uniform))
