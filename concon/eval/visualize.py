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

from .decision_tree import decision_tree_standalone

def main(args):
    if(not os.path.isdir(args.data_set)):
        print("Directory '%s' not found." % args.data_set)
        sys.exit()

    # Loads the model
    if(args.load_model is None):
        print("'load_model' must be indicated")
        sys.exit()

    if(args.population is not None): model = AliceBobPopulation.load_model(args.load_model, args)
    else: model = AliceBob.load(args.load_model, args)
    #print(model)

    data_loader = get_data_loader(args)

    decision_tree_standalone(model, data_loader)

    while(True): model.test_visualize(data_loader, args.learning_rate)
