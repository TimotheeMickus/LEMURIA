#!/usr/bin/env python

from datetime import datetime
from itertools import chain
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tqdm

from .games import AliceBob, AliceBobCharlie, AliceBobPopulation
from .utils.data import get_data_loader
from .utils.opts import get_args
from .utils.misc import build_optimizer
from .utils.logging import AutoLogger


def train(args):
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

        data_loader = get_data_loader(args)

        autologger = AutoLogger(simple_display=args.simple_display, steps_per_epoch=args.steps_per_epoch, debug=args.debug, log_lang_progress=args.log_lang_progress, log_entropy=args.log_entropy, base_alphabet_size=args.base_alphabet_size, device=args.device, no_summary=args.no_summary, summary_dir=args.summary, default_period=args.logging_period)

        print(("[%s] training start..." % datetime.now()), flush=True)
        for epoch in range(args.epochs):
            model.train_epoch(data_loader, epoch=epoch, autologger=autologger, steps_per_epoch=args.steps_per_epoch)
            model.evaluate(data_loader, epoch=epoch, event_writer=autologger.summary_writer, log_lang_progress=args.log_lang_progress, simple_display=args.simple_display, debug=args.debug)

            if(args.save_model):
                model.save(os.path.join(run_models_dir, "model_e%i.pt" % epoch))
                AliceBob.load(os.path.join(run_models_dir, "model_e%i.pt" % epoch), args)

if(__name__ == "__main__"):
    args = get_args()
    if args.evaluate_language:
        from .eval.evaluate_language import main
        main(args)
    elif args.visualize:
        from .eval.visualize import main
        main(args)
    elif args.compute_correlation:
        from .eval.compute_correlation import main
        main(args)
    else:
        train(args)
