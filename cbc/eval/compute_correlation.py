import itertools as it
import functools as ft
import csv
from collections import Counter
import random

from Levenshtein import hamming as _hamming
import Levenshtein
from scipy.stats import spearmanr as spearman
import numpy as np

def read_csv(csv_filename):
    """
    Open a message TSV file, and return messages paired with categories
    """
    with open(csv_filename) as istr:
        data = list(csv.reader(istr, delimiter="\t"))

    _, categories, messages = zip(*data)

    messages = map(str.strip, messages)
    messages = map(str.split, messages)
    messages = [tuple(map(int, msg)) for msg in messages]

    categories = map(str.strip, categories)
    categories = map(str.split, categories)
    categories = [tuple(map(int, ctg)) for ctg in categories]

    return messages, categories

@ft.lru_cache(maxsize=128)
def levenshtein(str1, str2, normalise=False):
    tmp = Levenshtein.distance(str1, str2)
    if(normalise): tmp /= (len(str1) + len(str2))

    return tmp

@ft.lru_cache(maxsize=128)
def hamming(str1, str2):
    return _hamming(str1, str2)

@ft.lru_cache(maxsize=128)
def levenshtein_normalised(str1, str2):
    return levenshtein(str1, str2, normalise=True)

@ft.lru_cache(maxsize=128)
def jaccard(seq1, seq2):
    cnt1, cnt2 = Counter(seq1), Counter(seq2)
    intersection = len(cnt1 & cnt2)
    union = (len(cnt1) + len(cnt2) - intersection)
    return 1.0 - intersection / union

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
        results.append(compute_correlation(messages, remapped_categories, **kwargs).correlation)
    results = np.array(results)
    return results.mean(), results.std()

def score(cor, μ, σ):
    return (cor - μ) / σ

def analyze_correlation(messages, categories, scrambling_pool_size, **kwargs):
    cor = compute_correlation(messages, categories, **kwargs).correlation
    μ, σ = compute_correlation_baseline(messages, categories, scrambling_pool_size, **kwargs)
    impr = score(cor, μ, σ)
    return cor, μ, σ, impr

def main(args):
    assert args.message_dump_file is not None, "Messages are required."

    messages, categories = read_csv(args.message_dump_file)

    l_cor, l_bμ, l_bσ, l_bi = analyze_correlation(messages, categories, 10)
    l_n_cor, l_n_bμ, l_n_bσ, l_n_bi = analyze_correlation(messages, categories, 10, message_distance=levenshtein_normalised)
    j_cor, j_bμ, j_bσ, j_bi = analyze_correlation(messages, categories, 10, message_distance=jaccard, map_msg_to_str=False)

    if args.simple_display:
        print(args.message_dump_file, l_cor, l_bi, l_n_cor, l_n_bi, j_cor, j_bi, sep='\t')
    else:
        print(
            'file: %s' % args.message_dump_file,
            'Levenshtein: %f (μ=%f, σ=%f, impr=%f)' % (l_cor, l_bμ, l_bσ, l_bi),
            'Levenshtein (normalized): %f (μ=%f, σ=%f, impr=%f)' % (l_n_cor, l_n_bμ, l_n_bσ, l_n_bi),
            'Jaccard: %f (μ=%f, σ=%f, impr=%f)' % (j_cor, j_bμ, j_bσ, j_bi),
            sep='\t')
