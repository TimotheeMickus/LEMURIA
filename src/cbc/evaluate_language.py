#!/usr/bin/env python3

from datetime import datetime
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tqdm

from data import get_data_loader

from config import *

# [START] Imports shared code from the parent directory
parent_dir_path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(parent_dir_path)

from aliceBob import AliceBob
from aliceBobPopulation import AliceBobPopulation
from utils import build_optimizer

sys.path.remove(parent_dir_path)

if(__name__ == "__main__"):
    if(not os.path.isdir(DATASET_PATH)):
        print("Directory '%s' not found." % DATASET_PATH)
        sys.exit()
    assert args.load_model is not None, "a valid path to a trained model is required."

    if(args.population > 1): model = AliceBobPopulation(args.population)
    else: model = AliceBob()
    model.load_state_dict(torch.load(args.load_model, map_location=DEVICE))
    model = model.to(DEVICE)
    #print(model)

    data_loader = get_data_loader(args.same_img)

    model.eval()

    with open('messages.txt', 'w') as ostr:
        for datapoint in tqdm.tqdm(data_loader.dataset):
            sender_outcome = model.sender(datapoint.img.unsqueeze(0))
            message_str = ' '.join(map(str, sender_outcome.action[0].view(-1).tolist()))
            category_str = ' '.join(map(str, datapoint.category))
            print(datapoint.idx, category_str, message_str, sep='\t', file=ostr)
