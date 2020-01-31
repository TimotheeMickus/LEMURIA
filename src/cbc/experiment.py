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

    for run in range(RUNS):
        print('Run %i' % run)

        run_models_dir = os.path.join(MODELS_DIR, str(run))
        run_summary_dir = os.path.join(SUMMARY_DIR, str(run))

        if(not os.path.isdir(run_summary_dir)): os.makedirs(run_summary_dir)
        if(SAVE_MODEL and (not os.path.isdir(run_models_dir))): os.makedirs(run_models_dir)

        if(args.population > 1): model = AliceBobPopulation(args.population).to(DEVICE)
        else: model = AliceBob().to(DEVICE)
        print(model)

        optimizer = build_optimizer(model.parameters())
        data_loader = get_data_loader()
        event_writer = SummaryWriter(run_summary_dir)

        print(datetime.now(), "training start...")
        for epoch in range(1, (EPOCHS + 1)):
            model.train_epoch(data_loader, optimizer, epoch=epoch, event_writer=event_writer)
            if(SAVE_MODEL): torch.save(model.state_dict(), os.path.join(run_models_dir, ("model_e%i.pt" % epoch)))
            #model.test_visualize(data_loader)
