#!/usr/bin/env python3

from datetime import datetime
import os

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import tqdm

from .decision_tree import decision_tree

from ..games import AliceBob, AliceBobPopulation
from ..utils.misc import build_optimizer, compute_entropy
from ..utils.data import get_data_loader

def main(args):
    if(not os.path.isdir(args.data_set)):
        print("Directory '%s' not found." % args.data_set)
        sys.exit()
    assert args.load_model is not None, "You need to specify 'load_model'"
    #assert args.message_dump_file is not None, "a valid output file is required."

    if(args.population is not None): model = AliceBobPopulation.load(args.load_model, args)
    else: model = AliceBob.load(args.load_model, args)
    #print(model)

    data_iterator = get_data_loader(args)

    #model.eval()
    '''
    counts = torch.zeros(args.base_alphabet_size, dtype=torch.float).to(args.device)

    if args.load_other_model is not None:
        other_model = type(model).load(args.load_other_model, args)
        counts_other_model == torch.zeros(args.base_alphabet_size, dtype=torch.float).to(args.device)

    ostr = open(args.message_dump_file, 'w') if(args.message_dump_file is not None) else None
    #with open(args.message_dump_file, 'w') as ostr:
    '''
   
    # We try to visit each category on average 32 times
    batch_size = 256
    max_datapoints = 32768 # (2^15)
    n = (32 * data_iterator.nb_categories)
    #n = data_iterator.size(data_type='test', no_evaluation=False)
    n = min(max_datapoints, n) 
    nb_batch = int(np.ceil(n / batch_size))
    print('%i datapoints (%i batches)' % (n, nb_batch))
    
    print("Generating the messages…")
    messages = []
    categories = []
    batch_numbers = range(nb_batch)
    if(args.display == 'tqdm'): batch_numbers = tqdm.tqdm(batch_numbers)
    with torch.no_grad():
        for _ in batch_numbers:
            model.start_episode(train_episode=False) # Selects agents at random if necessary

            batch = data_iterator.get_batch(batch_size, data_type='test', no_evaluation=False, sampling_strategies=['different'], keep_category=True) # Standard evaluation batch
            sender_outcome, receiver_outcome = model(batch)

            messages.extend([msg.tolist()[:l] for msg, l in zip(*sender_outcome.action)])
            categories.extend([x.category for x in batch.original])
   
    '''
    messages = []
    categories = []
    with torch.no_grad():
        batch_numbers = range(nb_batch)
        if(display == 'tqdm'): batch_numbers = tqdm.tqdm(range(nb_batch))
        for _ in batch_numbers:
        
        for datapoint in tqdm.tqdm(dataset):
            sender_outcome = model.sender(datapoint.img.unsqueeze(0).to(args.device))
            message = sender_outcome.action[0].view(-1)
            
            messages.append(message.tolist())
            
            message_str = ' '.join(map(str, message.tolist()))
            category_str = ' '.join(map(str, datapoint.category))
            counts += (torch.arange(args.base_alphabet_size).expand(message.size(0), args.base_alphabet_size) == message.unsqueeze(1)).float().sum(dim=0)
            
            if args.load_other_model is not None:
                other_sender_outcome = other_model.sender(datapoint.img.unsqueeze(0).to(args.device))
                other_message = other_sender_outcome.action[0].view(-1)
                counts_other_model += (torch.arange(args.base_alphabet_size).expand(other_message.size(0), args.base_alphabet_size) == other_message.unsqueeze(1)).float().sum(dim=0)

            if(ostr is not None): print(datapoint.idx, category_str, message_str, sep='\t', file=ostr)
    '''
    categories = np.array(categories) # Numpyfies the categories (but not the messages, as there are list of various length)

    '''
    if(ostr is not None): ostr.close()

    # Computes entropy stuff
    uniform = torch.ones_like(counts)

    if args.load_other_model is not None:
        print('entropy:', compute_entropy(counts, base=2), 'compared model:', compute_entropy(counts_other_model, base=2), 'ref uniform:', compute_entropy(uniform, base=2))
    else:
        print('entropy:', compute_entropy(counts, base=2), 'ref uniform:', compute_entropy(uniform, base=2))
    '''

    # Decision tree stuff
    decision_tree(messages=messages, categories=categories, alphabet_size=(model.base_alphabet_size + 1), concepts=data_iterator.concepts, gram_size=args.analysis_gram_size, disj_size=args.analysis_disj_size)
