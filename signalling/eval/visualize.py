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
from ..utils.image_data import get_data_loader

from .decision_tree import decision_tree_standalone

def main(args):
    if(not os.path.isdir(args.data_set)):
        print("Directory '%s' not found." % args.data_set)
        sys.exit()

    # Loads the model
    if(args.load_model is None):
        print("'load_model' must be indicated")
        sys.exit()

    # Population game iff a size or reset period was set. Read this from the checkpoint's embedded
    # hyperparameters when available (robust), otherwise fall back to the args (legacy checkpoints).
    hparams = AliceBob.peek_hparams(args.load_model)
    if(hparams is not None):
        is_population = (hparams.get("pop_size") is not None) or (hparams.get("pop_reset_period") is not None)
    else:
        is_population = (getattr(args, "pop_size", None) is not None) or (getattr(args, "pop_reset_period", None) is not None)
    model = (AliceBobPopulation if is_population else AliceBob).load(args.load_model, args)
    #print(model)

    data_loader = get_data_loader(args)

    decision_tree_standalone(model, data_loader)

    while(True): model.test_visualize(data_loader, args.learning_rate)
