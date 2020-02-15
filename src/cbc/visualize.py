#!/usr/bin/env python

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
# [END] Imports shared code from the parent directory

if(__name__ == "__main__"):
    if(not os.path.isdir(args.data_set)):
        print("Directory '%s' not found." % args.data_set)
        sys.exit()

    # Loads the model
    if(args.load_model is None):
        print("'load_model' must be indicated")
        sys.exit()

    if(args.population is not None): model = AliceBobPopulation(args)
    else: model = AliceBob(args)
    model.load_state_dict(torch.load(args.load_model, map_location=args.device))
    model = model.to(args.device)
    #print(model)

    data_loader = get_data_loader(args.same_img)

    model.decision_tree(data_loader, args.base_alphabet_size)
    while(True):
        model.test_visualize(data_loader, args.learning_rate)
