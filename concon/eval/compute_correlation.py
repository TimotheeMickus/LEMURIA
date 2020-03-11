import itertools as it
import functools as ft
import csv
import collections
import random

import torch
import Levenshtein
from scipy.stats import pearsonr as spearman
import scipy
import numpy as np
from ..utils.Mantel import test as mantel_test


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
    else:
        messages = map(str.strip, messages)
        messages = map(str.split, messages)
        messages = [tuple(map(int, msg)) for msg in messages]

    categories = map(str.strip, categories)
    categories = map(str.split, categories)
    categories = [tuple(map(int, ctg)) for ctg in categories]

    return messages, categories

# `str1` and `str2` can be any sequences
# `embeddings` must currently have a get method for element of the sequences to Numpy arrays (or None)
# `average_distance` and `r` are two hyperparameters: substituting two items the embeddings of which are at distance `average_distance` (in L2) will have a cost `r`. For smaller (resp. higher) distance, the cost of the substitution will be smaller (resp. higher), but, of course, always between 0 and 1.
def word_embedding_levenshtein(str1, str2, embeddings, average_distance, r=0.9, normalise=False):
    x1 = 1 + len(str1)
    x2 = 1 + len(str2)

    alpha = r / ((1 - r) * average_distance)

    # Initialisation of the matrix
    d = [] # Using Numpy structures for this is probably not more efficient
    d.append(list(range(x2)))
    for i in range(1, x1):
        d.append([i] * x2)

    # Core of the algorithm
    for i in range(1, x1):
        for j in range(1, x2):
            wa = str1[i-1]
            wb = str2[j-1]

            if(wa == wb): c = 0
            else:
                ea = embeddings.get(wa)
                eb = embeddings.get(wb)

                if((ea is None) or (eb is None)): c = 1
                else:
                    dst = np.linalg.norm(ea - eb) # Distance 2 (or L2 norm of the difference)

                    # Now, we need a function increasing function mapping 0 to 0 and +inf to 1
                    c = 1 - (1 / (1 + (alpha * dst)))

            d[i][j] = min(
                (d[(i-1)][j] + 1), # Deletion of str1[i]
                (d[i][(j-1)] + 1), # Insertion of str2[j]
                (d[(i-1)][(j-1)] + c)) # Substitution from str1[i] to str2[j]
            )

    raw = d[-1][-1]

    if(normalise): return (raw / (len(str1) + len(str2)))
    return raw

@ft.lru_cache(maxsize=32768)
def levenshtein(str1, str2, normalise=False):
    tmp = Levenshtein.distance(str1, str2)
    if(normalise): tmp /= (len(str1) + len(str2))

    return tmp

@ft.lru_cache(maxsize=32768)
def hamming(str1, str2):
    return Levenshtein.hamming(str1, str2)

@ft.lru_cache(maxsize=32768)
def levenshtein_normalised(str1, str2):
    return levenshtein(str1, str2, normalise=True)

@ft.lru_cache(maxsize=32768)
def jaccard(seq1, seq2):
    union = len(seq1)
    intersection = 0
    d = collections.defaultdict(int)
    for i in seq1:
        d[i] += 1
    for i in seq2:
        x = d[i]
        if(x > 0):
            d[i] -= 1
            intersection += 1
        else:
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
    messages = list(it.starmap(message_distance, it.combinations(messages, 2)))
    categories = list(it.starmap(meaning_distance, it.combinations(categories, 2)))

    return spearman(categories, messages)

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

def score(cor, μ, σ):
    return (cor - μ) / σ

def analyze_correlation(messages, categories, scrambling_pool_size=1000, **kwargs):
    cor = compute_correlation(messages, categories, **kwargs)[0]
    μ, σ = compute_correlation_baseline(messages, categories, scrambling_pool_size, **kwargs)
    impr = score(cor, μ, σ)
    return cor, μ, σ, impr

def mantel(messages, categories, message_distance=levenshtein, meaning_distance=hamming, perms=1000, method='pearson', map_msg_to_str=True, map_ctg_to_str=True):

    if map_msg_to_str:
        messages = [''.join(map(chr, msg)) for msg in messages]
    if map_ctg_to_str:
        categories = [''.join(map(chr, ctg)) for ctg in categories]

    assert len(messages) == len(categories)
    tM = np.array(list(it.starmap(message_distance, it.combinations(messages, 2))))
    sM = np.array(list(it.starmap(meaning_distance, it.combinations(categories, 2))))
    return mantel_test(tM, sM, method=method, perms=perms)

def main(args):
    assert args.message_dump_file is not None, "Messages are required."
    with torch.no_grad():
        messages, categories = read_csv(args.message_dump_file, string_msgs=args.string_msgs)

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
