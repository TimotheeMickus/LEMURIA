import itertools as it
import functools as ft
import csv
import collections
import random

import torch
from scipy.stats import pearsonr as spearman
import scipy
import scipy.spatial
import numpy as np

import Levenshtein

from ..utils.Mantel import test as mantel_test
from .decision_tree import decision_tree

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

# Computes a value between 0 and 1
class CosineDissimilarity:
    def __call__(self, u, v):
        return 0.5 * scipy.spatial.distance.cosine(u, v)
        
# Computes a value between 0 and 1
# If the distance (2) between `u` and `v` is higher (resp. lower) than the typical distance, returns a value that is higher (resp. lower) than the typical dissimilarity
class EuclideanDissimilarity:
    def __init__(self, typical_dst, typical_diss=0.9):
        self.alpha = typical_diss / ((1 - typical_diss) * typical_dst)

    def __call__(self, u, v):
         dst = np.linalg.norm(u - v) # Distance 2 (or L2 norm of the difference)
         c = 1 - (1 / (1 + (self.alpha * dst)))

         return c

# `seq1` and `seq2` can be any sequences
# `embeddings` must currently have a get method for element of the sequences to Numpy arrays (or None)
# 2020/07/02 I have made some change, now you need can supply a dissimilarity function as argument. To get the same behaviour as previously, use EclideanDissimilarity(typical_dst=average_distance, typical_diss=r)
def word_embedding_levenshtein(seq1, seq2, embeddings, diss_f=None, normalise=False):
    x1 = 1 + len(seq1)
    x2 = 1 + len(seq2)

    if(diss_f is None): diss_f = CosineDissimilarity()

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
                else: c = diss_f(v1, v2)

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

@ft.lru_cache(maxsize=32768)
def levenshtein(str1, str2, normalise=False):
    tmp = Levenshtein.distance(str1, str2)
    if(normalise): tmp /= (len(str1) + len(str2))

    return tmp

@ft.lru_cache(maxsize=32768)
# `str1` and `str2` must be two strings of the same length
def hamming(str1, str2):
    return Levenshtein.hamming(str1, str2)

@ft.lru_cache(maxsize=32768)
def levenshtein_normalised(str1, str2):
    return levenshtein(str1, str2, normalise=True)

@ft.lru_cache(maxsize=32768)
def jaccard(seq1, seq2):
    union = len(seq1) # Will contain the total number of symbols (with repetition) in both sequences
    intersection = 0 # Will contain the number of symbols (with repetition) in common in the two sequences
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

def mantel(messages, categories, message_distance=levenshtein, meaning_distance=hamming, perms=1000, method='pearson', map_msg_to_str=True, map_ctg_to_str=True, correl_only=False):
    assert len(messages) == len(categories)

    if map_msg_to_str:
        messages = [''.join(map(chr, msg)) for msg in messages] # Each integer is mapped to the corresponding unicode character

    if map_ctg_to_str:
        categories = [''.join(map(chr, ctg)) for ctg in categories] # Each integer is mapped to the corresponding unicode character

    tM = np.array(list(it.starmap(message_distance, it.combinations(messages, 2))))
    sM = np.array(list(it.starmap(meaning_distance, it.combinations(categories, 2))))

    return mantel_test(tM, sM, method=method, perms=perms, correl_only=correl_only)

def main(args):
    assert args.message_dump_file is not None, "'message_dump_file' is required."

    messages, categories, alphabet_size, concepts = read_csv(args.message_dump_file, string_msgs=args.string_msgs)

    # Logs some information
    print('Number of messages: %i' % len(messages))
    print('Alphabet size: %i' % alphabet_size)
    print('Concepts: %s' % concepts)

    # Correlation tests
    with torch.no_grad():
        """l_cor, l_bμ, l_bσ, l_bi = analyze_correlation(messages, categories)
        l_n_cor, l_n_bμ, l_n_bσ, l_n_bi = analyze_correlation(messages, categories, message_distance=levenshtein_normalised)
        j_cor, j_bμ, j_bσ, j_bi = analyze_correlation(messages, categories, message_distance=jaccard, map_msg_to_str=False)

        print(
            'file: %s' % args.message_dump_file,
            'Levenshtein: %f (μ=%f, σ=%f, impr=%f)' % (l_cor, l_bμ, l_bσ, l_bi),
            'Levenshtein (normalized): %f (μ=%f, σ=%f, impr=%f)' % (l_n_cor, l_n_bμ, l_n_bσ, l_n_bi),
            'Jaccard: %f (μ=%f, σ=%f, impr=%f)' % (j_cor, j_bμ, j_bσ, j_bi),
            sep='\t')
        """
        m_l = mantel(messages, categories)
        m_ln = mantel(messages, categories, message_distance=levenshtein_normalised)
        m_j = mantel(messages, categories, message_distance=jaccard, map_msg_to_str=False)

        print(args.message_dump_file, 'levenshtein', *m_l, 'levenshtein normalized', *m_ln, 'jaccard', *m_j)

    # Decision tree stuff
    print('\nDecision tree stuff')
    decision_tree(messages=messages, categories=categories, alphabet_size=alphabet_size, concepts=concepts)
