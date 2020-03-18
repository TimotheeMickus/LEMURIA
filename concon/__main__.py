#!/usr/bin/env python

from datetime import datetime
from itertools import chain
import os
import sys

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tqdm

from .games import AliceBob, AliceBobCharlie, AliceBobPopulation
from .utils.data import get_data_loader
from .utils.opts import get_args
from .utils.misc import build_optimizer, get_default_fn
from .utils.modules import build_cnn_decoder_from_args
from .utils.logging import AutoLogger
from .utils.data import Batch

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
        if((args.save_every > 0) and (not os.path.isdir(run_models_dir))): os.makedirs(run_models_dir)

        if(args.population is not None): model = AliceBobPopulation(args)
        elif args.charlie: model = AliceBobCharlie(args)
        else: model = AliceBob(args)
        model = model.to(args.device)
        #print(model)

        data_loader = get_data_loader(args)
        autologger = AutoLogger(game=model, data_loader=data_loader, display=args.display, steps_per_epoch=args.steps_per_epoch, debug=args.debug, log_lang_progress=args.log_lang_progress, log_entropy=args.log_entropy, device=args.device, no_summary=args.no_summary, summary_dir=run_summary_dir, default_period=args.logging_period, log_charlie_acc=args.charlie)

        if args.pretrain_CNNs:
            print(("[%s] pretraining start…" % datetime.now()), flush=True)

            dcnn_factory_fn = get_default_fn(build_cnn_decoder_from_args, args)
            pretrained_models = model.pretrain_CNNs(data_loader, autologger.summary_writer, pretrain_CNN_mode=args.pretrain_CNNs, freeze_pretrained_CNN=args.freeze_pretrained_CNNs, learning_rate=args.pretrain_learning_rate or args.learning_rate, nb_epochs=args.pretrain_epochs, steps_per_epoch=args.steps_per_epoch, display_mode=args.display, pretrain_CNNs_on_eval=args.pretrain_CNNs_on_eval, deconvolution_factory=dcnn_factory_fn, shared=args.shared)

            if(args.detect_outliers): # Might not work for all pertraining methods (in fact, we are expecting a MultiHeadsClassifier)
                (pretrained_name, pretrained_model), *_ = list(pretrained_models.items())
                print(pretrained_name)

                outliers = []
                with torch.no_grad():
                    n = len(data_loader)
                    if(n is None): n = 10000

                    batch_size = 128
                    for batch_i in range(n // batch_size):
                        datapoints = [data_loader.get_datapoint(i) for i in range((batch_size * batch_i), min((batch_size * (batch_i + 1)), n))]
                        batch = Batch(size=batch_size, original=[], target=datapoints, base_distractors=[])
                        hits, losses = pretrained_model.forward(batch)

                        misses = 0 # Will be a vector with one value (number of misses over all heads) per element in the batch
                        for x in hits: misses += (1 - x.cpu().numpy())

                        for i, miss in enumerate(misses):
                            if(miss == 0.0): continue
                            outliers.append((miss, datapoints[i]))
                            #print('Ahah! Datapoint idx=%i (category %s) has a high miss of %s!' % (datapoints[i].idx, datapoints[i].category, miss))

                outliers.sort(key=(lambda x: x[0]), reverse=True)
                print(len(outliers))
                for i in range(1000):
                    miss, datapoint = outliers[i]
                    print('%i - %i' % (datapoint.idx, miss))

                sys.exit(0)

        if(args.save_every > 0): model.save(os.path.join(run_models_dir, ("model_e%i.pt" % -1)))

        # TODO There is an asymmetry between pretraining (which handles the epochs itself) and training (which does not)
        print(("[%s] training start…" % datetime.now()), flush=True)
        for epoch in range(args.epochs):
            model.train_epoch(data_loader, epoch=epoch, autologger=autologger, steps_per_epoch=args.steps_per_epoch)
            model.evaluate(data_loader, epoch=epoch, event_writer=autologger.summary_writer, log_lang_progress=args.log_lang_progress, display=args.display, debug=args.debug)

            if((args.save_every > 0) and (((epoch + 1) % args.save_every) == 0)):
                model.save(os.path.join(run_models_dir, ("model_e%i.pt" % epoch)))

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
