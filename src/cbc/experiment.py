#!/usr/bin/env python

from datetime import datetime
from itertools import chain
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
from aliceBobCharlie import AliceBobCharlie
from aliceBobPopulation import AliceBobPopulation
from utils import build_optimizer, AverageSummaryWriter

sys.path.remove(parent_dir_path)
# [END] Imports shared code from the parent directory

if(__name__ == "__main__"):
    if(not os.path.isdir(args.data_set)):
        print(("Directory '%s' not found." % args.data_set), flush=True)
        sys.exit()

    summary_dir = args.summary.replace('[now]', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    models_dir = args.models.replace('[summary]', summary_dir)

    for run in range(args.runs):
        print(('Run %i' % run), flush=True)

        run_summary_dir = os.path.join(summary_dir, str(run))
        run_models_dir = os.path.join(models_dir, str(run))

        if((not args.no_summary) and (not os.path.isdir(run_summary_dir))): os.makedirs(run_summary_dir)
        if(args.save_model and (not os.path.isdir(run_models_dir))): os.makedirs(run_models_dir)

        if(args.population is not None): model = AliceBobPopulation(args)
        elif args.charlie: model = AliceBobCharlie(args)
        else: model = AliceBob(args)
        model = model.to(args.device)
        #print(model)

        if args.charlie:
            optimizer = (
                build_optimizer(chain(model.sender.parameters(), model.receiver.parameters()), args.learning_rate),
                build_optimizer(model.drawer.parameters(), args.learning_rate))
        else:
            optimizer = build_optimizer(model.parameters(), args.learning_rate)

        data_loader = get_data_loader(args.same_img)

        if(args.no_summary): event_writer = None
        else:
            tmp_writer = SummaryWriter(run_summary_dir)
            event_writer = AverageSummaryWriter(writer=tmp_writer, default_period=10)

        print(("[%s] training start..." % datetime.now()), flush=True)
        for epoch in range(args.epochs):
            model.train_epoch(data_loader, optimizer, epoch=epoch, event_writer=event_writer, log_lang_progress=args.log_lang_progress, simple_display=args.simple_display, debug=args.debug, log_entropy=args.log_entropy, steps_per_epoch=args.steps_per_epoch)
            model.evaluate(data_loader, event_writer=event_writer, log_lang_progress=args.log_lang_progress, simple_display=args.simple_display, debug=args.debug)

            if(args.save_model): torch.save(model.state_dict(), os.path.join(run_models_dir, ("model_e%i.pt" % epoch)))
