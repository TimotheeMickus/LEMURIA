#!/usr/bin/env python

from datetime import datetime
import os
import sys

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tqdm

from ..games import AliceBob, AliceBobPopulation
from ..utils.misc import build_optimizer
from ..utils.data import get_data_loader


def main(args):
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

    data_loader = get_data_loader(args)

    model.decision_tree(data_loader)
    while(True):
        model.test_visualize(data_loader, args.learning_rate)
