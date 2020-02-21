import itertools as it
import more_itertools as m_it
import functools as ft
import random
import os


OUTPUT_DIR = 'data/gen-langs'
NUMBER_MSG_PER_CATEGORY = 10
MAX_SWAP = 2
MAX_SYNONYMS = 3
MAX_UPPER_MWE_LEN = 3

#0. Create result dir
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

@ft.lru_cache(maxsize=32)
def _base_msg(category):
    """
    Given category, produce message.
    """
    return  [2 * i + int(c) for i,c in enumerate(category)]

def _pretty(message, synonym_table, end_symbol):
    """
    Clean up message.
    """
    return [s for sub in message for s in random.choice(synonym_table[sub])] + [end_symbol]

def generate_synonym_table(num_synonyms, upper_mwe_len):
    """
    Input:
        - `num_synonyms` number of synonyms per category
        - `upper_mwe_len` upper bound for the length of MWE
    Output:
        - `synonym_table`  synonym lookup table
        - `base_alphabet_size` number of symbols used
    """
    if upper_mwe_len == 1:
        base_alphabet_size = 10 * num_synonyms
        items = [(i,) for i in range(base_alphabet_size)]
        random.shuffle(items)
        synonym_table = list(m_it.chunked(items, num_synonyms))
    else:
        base_alphabet_size = 10
        items = (
            it.product(range(base_alphabet_size), repeat=i)
            for i in range(1, upper_mwe_len + 1)
        )
        items = list(it.chain.from_iterable(items))
        random.shuffle(items)
        synonym_table = list(m_it.chunked(items, num_synonyms))[:10]
    return synonym_table, base_alphabet_size


#1. Generate regular language
with open(os.path.join(OUTPUT_DIR, 'regular.tsv'), 'w') as ostr:
    for id, category in enumerate(it.product([0, 1], repeat=5)):
            message = _base_msg(category) + [10]
            print(id, ' '.join(map(str, category)), ' '.join(map(str, message)), sep="\t", file=ostr)

for synonyms in range(1, MAX_SYNONYMS + 1):

    for upper_mwe_len in range(1, MAX_UPPER_MWE_LEN + 1):

        base_name = 'syn-%i_mwe-%i_' % (synonyms, upper_mwe_len)

        for swaps in range(MAX_SWAP + 1):

            synonym_table, base_alphabet_size = generate_synonym_table(synonyms, upper_mwe_len)
            if swaps == 0:
                # no scramble
                with open(os.path.join(OUTPUT_DIR, base_name + 'strict-order.tsv'), 'w') as ostr:
                    idx = 0
                    for i in range(NUMBER_MSG_PER_CATEGORY):
                        for category in it.product([0, 1], repeat=5):
                            message = _base_msg(category)
                            message = _pretty(message, synonym_table, base_alphabet_size)
                            print(idx, ' '.join(map(str, category)), ' '.join(map(str, message)), sep="\t", file=ostr)
                            idx += 1

                synonym_table, base_alphabet_size = generate_synonym_table(synonyms, upper_mwe_len)
                # total random scramble
                with open(os.path.join(OUTPUT_DIR, base_name + 'random-order.tsv'), 'w') as ostr:
                    idx = 0
                    for i in range(NUMBER_MSG_PER_CATEGORY):
                        for category in it.product([0, 1], repeat=5):
                            message = _base_msg(category)
                            random.shuffle(message)
                            message = _pretty(message, synonym_table, base_alphabet_size)
                            print(idx, ' '.join(map(str, category)), ' '.join(map(str, message)), sep="\t", file=ostr)
                            idx += 1

            # random inversion of `swap` pairs
            else:
                with open(os.path.join(OUTPUT_DIR, base_name + 'swap-%i.tsv' % swaps), 'w') as ostr:
                    idx = 0
                    for i in range(NUMBER_MSG_PER_CATEGORY):
                        for category in it.product([0, 1], repeat=5):
                            message = _base_msg(category)
                            for _ in range(swaps):
                                i1, i2 = random.sample(range(5), 2)
                                message[i1], message[i2] = message[i2], message[i1]
                            message = _pretty(message, synonym_table, base_alphabet_size)
                            print(idx, ' '.join(map(str, category)), ' '.join(map(str, message)), sep="\t", file=ostr)
                            idx += 1
