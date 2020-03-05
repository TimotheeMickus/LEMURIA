import itertools as it
import more_itertools as m_it
import functools as ft
import random
import os

OUTPUT_DIR = 'data/gen-langs'
ARITIES = [2,3,4,5]

#0. Create result dir
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

for arity in ARITIES:
    with open(os.path.join(OUTPUT_DIR, 'arity-%i-regular.tsv' % arity), 'w') as ostr:
        for id, category in enumerate(it.product([0, 1], repeat=5)):
            message = numberToBase(id, arity) + [arity]
            print(id, ' '.join(map(str, category)), ' '.join(map(str, message)), sep="\t", file=ostr)
    with open(os.path.join(OUTPUT_DIR, 'arity-%i-random.tsv' % arity), 'w') as ostr:
        ids, categories = zip(*enumerate(it.product([0, 1], repeat=5)))
        categories = list(map(list, categories))
        random.shuffle(categories)
        for id, category in zip(ids, categories):
            message = numberToBase(id, arity) + [arity]
            print(id, ' '.join(map(str, category)), ' '.join(map(str, message)), sep="\t", file=ostr)

ARITIES = [7,9,11]

for arity in ARITIES:
    with open(os.path.join(OUTPUT_DIR, 'saint-khmers-arity-%i-regular.tsv' % arity), 'w') as ostr:
        for id, category in enumerate(it.product(list(range(0,6)), repeat=4)):
            message = numberToBase(id, arity) + [arity]
            print(id, ' '.join(map(str, category)), ' '.join(map(str, message)), sep="\t", file=ostr)
    with open(os.path.join(OUTPUT_DIR, 'saint-khmers-arity-%i-random.tsv' % arity), 'w') as ostr:
        ids, categories = zip(*enumerate(it.product(list(range(0,6)), repeat=4)))
        categories = list(map(list, categories))
        random.shuffle(categories)
        for id, category in zip(ids, categories):
            message = numberToBase(id, arity) + [arity]
            print(id, ' '.join(map(str, category)), ' '.join(map(str, message)), sep="\t", file=ostr)
