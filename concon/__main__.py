#!/usr/bin/env python

from datetime import datetime
import sys

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tqdm

from .games import AliceBob, AliceBobCharlie, AliceBobPopulation, AliceBobCharliePopulation
from .utils.data import get_data_loader
from .utils.opts import get_args
from .utils.misc import build_optimizer, get_default_fn, path_replace
from .utils.modules import build_cnn_decoder_from_args, build_cnn_encoder_from_args
from .utils.logging import AutoLogger
from .utils.data import Batch

def train(args):
    if(not args.data_set.is_dir()):
        print(("Directory '%s' not found." % args.data_set), flush=True)
        sys.exit()

    summary_dir = path_replace(args.summary, '[now]', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    models_dir = path_replace(args.models, '[summary]', summary_dir)

    for run in range(args.runs):
        print(('Run %i' % run), flush=True)

        run_summary_dir = summary_dir / str(run)
        run_models_dir = models_dir / str(run)

        data_loader = get_data_loader(args)
        autologger = AutoLogger(base_alphabet_size=args.base_alphabet_size, data_loader=data_loader, display=args.display, steps_per_epoch=args.steps_per_epoch, debug=args.debug, log_lang_progress=args.log_lang_progress, log_entropy=args.log_entropy, device=args.device, no_summary=args.no_summary, summary_dir=run_summary_dir, default_period=args.logging_period,)# log_charlie_acc=args.charlie)

        if(not args.no_summary): run_summary_dir.mkdir(parents=True, exist_ok=True)
        if(args.save_every > 0): run_models_dir.mkdir(parents=True, exist_ok=True)

        if((args.population is not None) and (args.charlie)): model = AliceBobCharliePopulation(args, autologger)
        elif(args.population is not None): model = AliceBobPopulation(args, autologger)
        elif(args.charlie): model = AliceBobCharlie(args, autologger)
        else: model = AliceBob(args, autologger)
        model = model.to(args.device)


        if args.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

        if args.pretrain_CNNs:
            print(("[%s] pretraining start…" % datetime.now()), flush=True)

            dcnn_factory_fn = get_default_fn(build_cnn_decoder_from_args, args)
            cnn_factory_fn = get_default_fn(build_cnn_encoder_from_args, args)
            pretrained_models = model.pretrain_CNNs(
                data_loader,
                pretrain_CNN_mode=args.pretrain_CNNs,
                freeze_pretrained_CNN=args.freeze_pretrained_CNNs,
                learning_rate=args.pretrain_learning_rate or args.learning_rate,
                nb_epochs=args.pretrain_epochs,
                steps_per_epoch=args.steps_per_epoch,
                display_mode=args.display,
                pretrain_CNNs_on_eval=args.pretrain_CNNs_on_eval,
                deconvolution_factory=dcnn_factory_fn,
                convolution_factory=cnn_factory_fn,
                shared=args.shared,
                pretrain_charlie=args.pretrain_charlie)

            if(args.detect_outliers): # Might not work for all pretraining methods (in fact, we are expecting a MultiHeadsClassifier). To have a more general method, record the loss for all instances, then select the ones that are far from the mean
                (pretrained_name, pretrained_model), *_ = list(pretrained_models.items())
                print(pretrained_name)

                outliers = []
                with torch.no_grad():
                    batch_size = args.batch_size
                    max_datapoints = 2 ** 15
                    n = data_loader.size(data_type='any', no_evaluation=False) # We would like to see all datapoints
                    if((n is None) or (n > max_datapoints)):
                        print('The dataset is too big, so we are only going to be looking at %i datapoints.' % max_datapoints)
                        n = max_datapoints
                    nb_batch = int(np.ceil(n / batch_size))
                    for batch_i in range(nb_batch):
                        datapoints = [data_loader.get_datapoint(i) for i in range((batch_size * batch_i), min((batch_size * (batch_i + 1)), n))]
                        batch = Batch(size=batch_size, original=[], target=[x.toInput(keep_category=True, device=args.device) for x in datapoints], base_distractors=[])
                        hits, losses = pretrained_model.forward(batch)

                        misses = 0 # Will be a vector with one value (number of misses over all heads) per element in the batch
                        for x in hits: misses += (1 - x.cpu().numpy())

                        for i, miss in enumerate(misses):
                            if(miss == 0.0): continue
                            outliers.append((miss, datapoints[i]))
                            #print('Ahah! Datapoint idx=%i (category %s) has a high miss of %s!' % (datapoints[i].idx, datapoints[i].category, miss))

                outliers.sort(key=(lambda x: x[0]), reverse=True)
                print('%i outliers (%s%%)' % (len(outliers), (100 * len(outliers) / n)))
                for i in range(len(outliers)): #range(min(len(outliers), 1000)):
                    miss, datapoint = outliers[i]
                    print('%i - %i' % (datapoint.idx, miss))

                sys.exit(0)

        if(args.save_every > 0): model.save(run_models_dir / ("model_e%i.pt" % -1))

        print(("[%s] training start…" % datetime.now()), flush=True)
        # make Charlie not trainable
        if(args.charlie):
            model.switch_charlie(False)
        model.train_agents(args.epochs, args.steps_per_epoch, data_loader, run_models_dir=run_models_dir, save_every=args.save_every)

        if args.charlie:
            # make Charlie trainable
            model.switch_charlie(True)

            model.train_agents(args.epochs, args.steps_per_epoch, data_loader, run_models_dir=run_models_dir, save_every=args.save_every)

            # make Charlie not trainable
            model.switch_charlie(False)

    # train_images = model.get_images(data_loader.get_batch(data_type='train'))
    # for batch_idx in range(train_images.size(0)):
    #     show_imgs(train_images[batch_idx])

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
    elif args.threeway_correlation:
        from .eval.three_way_correlation import main
        main(args)
    else:
        train(args)
