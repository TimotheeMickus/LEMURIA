import itertools as it
import random
import os

OUTPUT_DIR = 'data/gen-langs'
ROLLS = 20
MAX_SWAP = 5

if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, 'regular.tsv'), 'w') as ostr:
    for id, category in enumerate(it.product([True, False], repeat=5)):
            message = [2 * i + int(c) for i,c in enumerate(category)] + [10]
            print(id, ' '.join(map(str, category)), ' '.join(map(str, message)), sep="\t", file=ostr)

with open(os.path.join(OUTPUT_DIR, 'random-order.tsv'), 'w') as ostr:
    idx = 0
    for i in range(ROLLS):
        for category in it.product([True, False], repeat=5):
            message = [2 * i + int(c) for i,c in enumerate(category)]
            random.shuffle(message)
            message += [10]
            print(idx, ' '.join(map(str, category)), ' '.join(map(str, message)), sep="\t", file=ostr)
            idx += 1

for swaps in range(1, MAX_SWAP + 1):
    with open(os.path.join(OUTPUT_DIR, 'swap-%i.tsv' % swaps), 'w') as ostr:
        idx = 0
        for i in range(ROLLS):
            for category in it.product([True, False], repeat=5):
                message = [2 * i + int(c) for i,c in enumerate(category)]
                for _ in range(swaps):
                    i1, i2 = random.sample(range(5), 2)
                    message[i1], message[i2] = message[i2], message[i1]
                message += [10]
                print(idx, ' '.join(map(str, category)), ' '.join(map(str, message)), sep="\t", file=ostr)
                idx += 1
