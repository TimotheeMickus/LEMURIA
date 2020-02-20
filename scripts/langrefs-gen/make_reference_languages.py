import random
import itertools as it

with open("regular.tsv", "w") as ostr:
    for id, category in enumerate(it.product([True, False], repeat=5)):
            message = [2 * i + int(c) for i,c in enumerate(category)] + [10]
            print(id, ' '.join(map(str, category)), ' '.join(map(str, message)), sep="\t", file=ostr)

rolls = 5
with open("random-order.tsv", "w") as ostr:
    idx = 0
    for i in range(rolls):
        for category in it.product([True, False], repeat=5):
            message = [2 * i + int(c) for i,c in enumerate(category)]
            random.shuffle(message)
            message += [10]
            print(idx, ' '.join(map(str, category)), ' '.join(map(str, message)), sep="\t", file=ostr)
            idx += 1
