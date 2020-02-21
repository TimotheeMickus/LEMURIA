import itertools as it
import csv

from Levenshtein import hamming
from Levenshtein import distance as levenshtein
from scipy.stats import spearmanr as spearman

def read_csv(csv_filename):

    with open(csv_filename) as istr:
        data = list(csv.reader(istr, delimiter="\t"))

    _, categories, messages = zip(*data)

    messages = map(str.strip, messages)
    messages = map(str.split, messages)
    messages = [list(map(int, msg)) for msg in messages]

    categories = map(str.strip, categories)
    categories = map(str.split, categories)
    categories = [list(map(eval, ctg)) for ctg in categories]

    return messages, categories

def compute_correlation(messages, categories):

    messages = [''.join(map(chr, msg)) for msg in messages]
    categories = [''.join(map(chr, map(int, ctg))) for ctg in categories]

    messages = list(it.starmap(levenshtein, it.combinations(messages, 2)))
    categories = list(it.starmap(hamming, it.combinations(categories, 2)))

    return spearman(categories, messages)

def main(args):
    assert args.message_dump_file is not None, "Messages are required."

    messages, categories = read_csv(args.message_dump_file)

    print(compute_correlation(messages, categories))
