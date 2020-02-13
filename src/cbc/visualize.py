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
    if(not os.path.isdir(DATASET_PATH)):
        print("Directory '%s' not found." % DATASET_PATH)
        sys.exit()

    # Loads the model
    if(args.load_model is None):
        print("'load_model' must be indicated")
        sys.exit()

    if(args.population is not None): model = AliceBobPopulation(size=args.population, shared=args.shared)
    else: model = AliceBob(shared=args.shared)
    model.load_state_dict(torch.load(args.load_model, map_location=DEVICE))
    model = model.to(DEVICE)
    #print(model)

    data_loader = get_data_loader(args.same_img)
   
    model.decision_tree(data_loader)
    while(True):
        model.test_visualize(data_loader)
