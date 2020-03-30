import itertools as it
import functools as ft
import csv
import collections
import random
import os

import torch
from scipy.stats import pearsonr as spearman
import scipy
import numpy as np

import tqdm
import Levenshtein

from ..utils.Mantel import test as mantel_test
from .decision_tree import decision_tree

from ..games import AliceBob, AliceBobPopulation
from ..utils.misc import build_optimizer, compute_entropy
from ..utils.data import get_data_loader

# The output values `messages` and `categories` are both lists of tuples of integers.
# If `string_msgs` is set to True, then the messages in the file are considered string and then converted to tuples of integers.
def read_csv(csv_filename, string_msgs=False):
    """
    Open a message TSV file, and return messages paired with categories
    """
    with open(csv_filename) as istr:
        data = list(csv.reader(istr, delimiter="\t"))

    _, categories, messages = zip(*data)

    if string_msgs:
        c2i = collections.defaultdict(it.count().__next__)
        messages = map(str.strip, messages)
        messages = [tuple(map(c2i.__getitem__, msg)) for msg in messages]

        alphabet_size = len(c2i)
    else:
        messages = map(str.strip, messages)
        messages = map(str.split, messages)
        messages = [tuple(map(int, msg)) for msg in messages]

        alphabet_size = (1 + max([max(msg, default=0) for msg in messages]))

    categories = map(str.strip, categories)
    categories = map(str.split, categories)
    categories = [tuple(map(int, ctg)) for ctg in categories]

    # We assume all categories have the same length
    concepts = [] # List of dictionaries {value name -> value idx}
    for i in range(len(categories[0])):
        #concept_name = ('d%i' % i)
        max_value = max([category[i] for category in categories])
        concept_values = dict([(('d%i.v%i' % (i, v)), v) for v in range(max_value + 1)])
        concepts.append(concept_values)

    return messages, categories, alphabet_size, concepts

def l2(v1, v2): return np.linalg.norm(v1 - v2)

# `seq1` and `seq2` can be any sequences
# `embeddings` must currently have a get method for element of the sequences to Numpy arrays (or None)
# `average_distance` and `r` are two hyperparameters: substituting two items the embeddings of which are at distance `average_distance` (in L2) will have a cost `r`. For smaller (resp. higher) distance, the cost of the substitution will be smaller (resp. higher), but, of course, always between 0 and 1.
def word_embedding_levenshtein(seq1, seq2, embeddings, average_distance, r=0.9, normalise=False):
    x1 = 1 + len(seq1)
    x2 = 1 + len(seq2)

    alpha = r / ((1 - r) * average_distance)

    # Initialisation of the matrix
    d = [] # Using Numpy structures for this is probably not more efficient
    d.append(list(range(x2)))
    for i in range(1, x1):
        d.append([i] * x2)

    # Core of the algorithm
    for i in range(1, x1):
        for j in range(1, x2):
            e1 = seq1[i-1]
            e2 = seq2[j-1]

            if(e1 == e2): c = 0
            else:
                v1 = embeddings.get(e1)
                v2 = embeddings.get(e2)

                if((v1 is None) or (v2 is None)): c = 1
                else:
                    dst = np.linalg.norm(v1 - v2) # Distance 2 (or L2 norm of the difference)

                    # Now, we need a function increasing function mapping 0 to 0 and +inf to 1
                    c = 1 - (1 / (1 + (alpha * dst)))

                    #c /= r # If you uncomment this line, the cost of a substitution at distance `average_distance` will be 1 and substitutions might have higher cost, up to 1/r. This might be justified as long as `r` is above 0.5 (otherwise, some substitutions might be more expensive than an insertion followed by a deletion).

            d[i][j] = min(
                (d[(i-1)][j] + 1), # Deletion of seq1[i]
                (d[i][(j-1)] + 1), # Insertion of seq2[j]
                (d[(i-1)][(j-1)] + c) # Substitution from seq1[i] to seq2[j]
            )

    raw = d[-1][-1]

    if(normalise): return (raw / (len(seq1) + len(seq2)))
    return raw

# `weights` is a dictionary of word to weight (between 0 and 1)
# One possibility for computing these weights: for a given token, let f be its document frequency (the proportion of documents in which it appears), then w = 1 - f
def weighted_levenshtein(seq1, seq2, weights, normalise=False):
    x1 = 1 + len(seq1)
    x2 = 1 + len(seq2)

    alpha = r / ((1 - r) * average_distance)

    # Initialisation of the matrix
    d = [] # Using Numpy structures for this is probably not more efficient
    tmp = 0.0
    first_line = [tmp]
    for e in seq2:
        tmp += weights.get(e, 1)
        first_line.append(tmp)
    d.append(first_line)
    tmp = 0
    for e in seq1:
        tmp += weights.get(e, 1)
        d.append([tmp] * x2)

    # Core of the algorithm
    for i in range(1, x1):
        for j in range(1, x2):
            e1 = seq1[i-1]
            e2 = seq2[j-1]

            w1 = weights.get(e1, 1)
            w2 = weights.get(e2, 1)

            d[i][j] = min(
                (d[(i-1)][j] + w1), # Deletion of seq1[i]
                (d[i][(j-1)] + w2), # Insertion of seq2[j]
                (d[(i-1)][(j-1)] + (int(e1 != e2) * max(w1, w2))) # Substitution from seq1[i] to seq2[j]
            )

    raw = d[-1][-1]

    if(normalise): return (raw / (r[0][-1] + r[-1][0]))
    return raw

#@ft.lru_cache(maxsize=32768)
def levenshtein(str1, str2, normalise=False):
    tmp = Levenshtein.distance(str1, str2)
    if(normalise): tmp /= (len(str1) + len(str2))

    return tmp

#@ft.lru_cache(maxsize=32768)
# `str1` and `str2` must be two strings of the same length
def hamming(str1, str2):
    return Levenshtein.hamming(str1, str2)

#@ft.lru_cache(maxsize=32768)
def levenshtein_normalised(str1, str2):
    return levenshtein(str1, str2, normalise=True)

#@ft.lru_cache(maxsize=32768)
def jaccard(seq1, seq2):
    union = len(seq1) # Will contain the total number of symbols (with repetition) in both sequences
    intersection = 0 # Will contain the nubme of symbols (with repetition) in common in the two sequences
    d = collections.defaultdict(int) # This dictionary is used to track which symbols in seq1 are not exhausted by seq2
    for i in seq1:
        d[i] += 1
    for i in seq2:
        x = d[i]
        if(x > 0): # i·s from seq1 not exhausted yet
            d[i] -= 1
            intersection += 1
        else: # i·s from seq1 already exhausted
            union += 1
    return 1 - (intersection / union)

"""
@ft.lru_cache(maxsize=32768)
def jaccard2(seq1, seq2):
    proto_union = len(seq1) + len(seq2)
    intersection = 0
    d = collections.defaultdict(int)
    for i in seq1:
        d[i] += 1
    for i in seq2:
        x = d[i]
        if(x > 0):
            d[i] -= 1
            intersection += 1
    return 1 - (intersection / (proto_union - intersection))
"""

def compute_correlation(messages, categories, message_distance=levenshtein, meaning_distance=hamming, map_msg_to_str=True, map_ctg_to_str=True):
    """
    Compute correlation of message distance and meaning distance.
    """

    # Some distance functions are defined over strings only
    if map_msg_to_str:
        messages = [''.join(map(chr, msg)) for msg in messages]
    if map_ctg_to_str:
        categories = [''.join(map(chr, ctg)) for ctg in categories]

    # Compute pairwise distances
    dm = list(it.starmap(message_distance, it.combinations(messages, 2)))
    dc = list(it.starmap(meaning_distance, it.combinations(categories, 2)))

    return spearman(dc, dm)

def compute_correlation_baseline(messages, categories, scrambling_pool_size, **kwargs):
    """
    Compute baseline for correlation of message distance and meaning distance.
    """
    uniq_cats = list(set(map(tuple, categories)))
    num_cats = len(uniq_cats)
    results = []
    for _ in range(scrambling_pool_size):
        mapping = dict(zip(uniq_cats, random.sample(uniq_cats, num_cats)))
        remapped_categories = list(map(mapping.__getitem__, map(tuple, categories)))
        results.append(compute_correlation(messages, remapped_categories, **kwargs)[0])
    results = np.array(results)

    return results.mean(), results.std()

# Normalise value `x` using a mean and standard deviation
def score(x, mean, stdev):
    return ((x - mean) / stdev)

def analyze_correlation(messages, categories, scrambling_pool_size=1000, **kwargs):
    cor = compute_correlation(messages, categories, **kwargs)[0]
    μ, σ = compute_correlation_baseline(messages, categories, scrambling_pool_size, **kwargs)
    impr = score(cor, μ, σ)

    return cor, μ, σ, impr

def mantel(messages, categories, message_distance=levenshtein, meaning_distance=hamming, perms=1000, method='pearson', map_msg_to_str=True, map_ctg_to_str=True):
    assert len(messages) == len(categories)

    if map_msg_to_str:
        messages = [''.join(map(chr, msg)) for msg in messages] # Each integer is mapped to the corresponding unicode character

    if map_ctg_to_str:
        categories = [''.join(map(chr, ctg)) for ctg in categories] # Each integer is mapped to the corresponding unicode character

    #tM = np.array(list(it.starmap(message_distance, it.combinations(messages, 2))))
    tM = np.array([message_distance(msg_1, msg_2) for msg_1, msg_2 in it.combinations(messages, 2)])
    #sM = np.array(list(it.starmap(meaning_distance, it.combinations(categories, 2))))
    sM = np.array([meaning_distance(ctg_1, ctg_2) for ctg_1, ctg_2 in it.combinations(categories, 2)])

    return mantel_test(tM, sM, method=method, perms=perms)

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
    images = []
    batch_numbers = range(nb_batch)
    if(args.display == 'tqdm'): batch_numbers = tqdm.tqdm(batch_numbers)
    with torch.no_grad():
        sender = model._sender if (args.population is not None) else model.sender
        for _ in batch_numbers:
            model.start_episode(train_episode=False) # Selects agents at random if necessary

            batch = data_iterator.get_batch(batch_size, data_type='test', no_evaluation=False, sampling_strategies=['different'], keep_category=True) # Standard evaluation batch
            sender_outcome, receiver_outcome = model(batch)
            image = sender.image_encoder(batch.original_img(stack=True))

            messages.extend([msg.tolist()[:l] for msg, l in zip(*sender_outcome.action)])
            categories.extend([x.category for x in batch.original])
            images.extend([vector for vector in image.numpy()])

    mc_ln = mantel(messages, categories, message_distance=levenshtein_normalised)
    mc_j = mantel(messages, categories, message_distance=jaccard, map_msg_to_str=False)

    ic_ln = mantel(images, categories, message_distance=l2, map_msg_to_str=False)
    ic_j = mantel(images, categories, message_distance=l2, map_msg_to_str=False)

    mi_ln = mantel(messages, images, message_distance=levenshtein_normalised, meaning_distance=l2, map_ctg_to_str=False)
    mi_j = mantel(messages, images, message_distance=jaccard, map_msg_to_str=False, meaning_distance=l2, map_ctg_to_str=False)

    print('levenshtein normalized\n\tmsg/ctg:', *mc_ln, '\n\timg/ctg:',  *ic_ln, '\n\tmsg/img:',  *mi_ln)
    print('jaccard\n\tmsg/ctg:', *mc_j, '\n\timg/ctg:',  *ic_j, '\n\tmsg/img:',  *mi_j)
