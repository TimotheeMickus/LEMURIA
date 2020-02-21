import itertools as it
import csv
from collections import Counter

from Levenshtein import hamming
import Levenshtein
from scipy.stats import spearmanr as spearman

def read_csv(csv_filename):
    """
    Open a message TSV file, and return messages paired with categories
    """
    with open(csv_filename) as istr:
        data = list(csv.reader(istr, delimiter="\t"))

    _, categories, messages = zip(*data)

    messages = map(str.strip, messages)
    messages = map(str.split, messages)
    messages = [list(map(int, msg)) for msg in messages]

    categories = map(str.strip, categories)
    categories = map(str.split, categories)
    categories = [list(map(int, ctg)) for ctg in categories]

    return messages, categories

def levenshtein(str1, str2, normalise=False):
    tmp = Levenshtein.distance(str1, str2)
    if(normalise): tmp /= (len(str1) + len(str2))

    return tmp

def levenshtein_normalised(str1, str2):
    return levenshtein(str1, str2, normalise=True)

def jaccard(seq1, seq2):
    cnt1, cnt2 = Counter(seq1), Counter(seq2)
    return 1.0 - len(cnt1 & cnt2) / len(cnt1 | cnt2)

def compute_correlation(messages, categories, message_distance=levenshtein, meaning_distance=hamming, map_msg_to_str=True, map_ctg_to_str=True):
    """
    Compute correlation of message distance and meaning distance.
    """

    # Some distance functions are defined over strings only
    if map_msg_to_str:
        messages = [''.join(map(chr, msg)) for msg in messages]
    if map_ctg_to_str:
        categories = [''.join(map(chr, map(int, ctg))) for ctg in categories]

    # Compute pairwise distances
    messages = list(it.starmap(message_distance, it.combinations(messages, 2)))
    categories = list(it.starmap(meaning_distance, it.combinations(categories, 2)))

    return spearman(categories, messages)

def main(args):
    assert args.message_dump_file is not None, "Messages are required."

    messages, categories = read_csv(args.message_dump_file)
    l_cor = compute_correlation(messages, categories).correlation
    l_n_cor = compute_correlation(messages, categories, message_distance=levenshtein_normalised).correlation
    j_cor = compute_correlation(messages, categories, message_distance=jaccard, map_msg_to_str=False).correlation
    if args.simple_display:
        print(args.message_dump_file, l_cor, l_n_cor, j_cor, sep='\t')
    else:
        print(
            'file: %s' % args.message_dump_file,
            'Levenshtein: %f' % l_cor,
            'Levenshtein (normalised): %f' % l_n_cor,
            'Jaccard: %f' % j_cor,
            sep='\t')
