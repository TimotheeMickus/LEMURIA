import itertools
import torch

import torch.nn as nn
import tqdm
import sklearn.tree
import matplotlib.pyplot as plt

from collections import defaultdict

def decision_tree(model, data_iterator):
    model.eval()

    print("Generating the messages…")
    messages = []
    with torch.no_grad():
        n = len(data_iterator)
        n = n or 100000 # If the dataset is infinite (None), use 100000 data points
        for i in tqdm.tqdm(range(n)):
            datapoint = data_iterator.get_datapoint(i)

            sender_outcome = model.sender(datapoint.img.unsqueeze(0))
            message = sender_outcome.action[0].view(-1).tolist()
            messages.append(message)
            #print((datapoint.category, message))

    # As features, we will use the presence of n-grams
    import numpy as np

    # TODO Add one OUT_OF_SENTENCE pseudo-word to the alphabet
    n = 3
    alphabet_size = model.base_alphabet_size + 1
    nb_ngrams = alphabet_size * (alphabet_size**n - 1) // (alphabet_size - 1)
    print('Number of possible %i-grams: %i' % (n, nb_ngrams))

    ngrams = [()] * nb_ngrams
    def ngram_to_idx(ngram): # `ngram` is a list of integers
        idx = 0
        for i, k in enumerate(ngram): # We read the n-grams as numbers in base ALPHABET_SIZE written in reversed and with '0' used as the unit, instead of '1' (because message (0, 0) is different from (0))
            idx += (k + 1) * (alphabet_size**i) # '+1' because symbol '0' is used as the unit

        idx -= 1 # Because the 0-gram is not taken into account

        assert (ngrams[idx] == () or ngrams[idx] == ngram) # Checks that we are not assigning the same id to two different n-grams

        ngrams[idx] = ngram

        return idx

    last_symbol = alphabet_size - 1 # Because the alphabet starts with 0
    last_tuple = tuple([last_symbol] * n)
    print('Id of %s: %i' % (last_tuple, ngram_to_idx(last_tuple)))

    feature_vectors = []
    for message in messages:
        # We could consider adding the BOM symbol
        v = np.zeros(nb_ngrams, dtype=bool)
        s = set()
        for l in range(1, (n + 1)): # Length on the n-gram
            for i in range(len(message) - l + 1):
                ngram = tuple(message[i:(i + l)])
                s.add(ngram)
                idx = ngram_to_idx(ngram)
                v[idx] = True
                #print((ngram, idx))
        #input((message, v, s))
        feature_vectors.append(v)

    feature_vectors = np.array(feature_vectors)

    rule_precision_threshold = 0.95
    rule_frequence_threshold = 0.05
    rules = defaultdict(list) # From ngram to list of RHS·s (to be conjuncted)

    results_decision_tree = []
    max_depth = 2 # None # We could successively try with increasing depth
    max_conjunctions = 3 # data_iterator.nb_concepts
    for size_conjunctions in range(1, (max_conjunctions + 1)):
        results_binary_classifier = []

        for concept_indices in itertools.combinations(range(data_iterator.nb_concepts), size_conjunctions): # Iterates over all subsets of [|0, `data_iterator.nb_concepts`|[ of size `size_conjunctions`
            #print([data_iterator.concept_names[idx] for idx in concept_indices])

            # For each selected concept, we pick a value
            conjunctions = itertools.product(*[data_iterator.concepts[idx].keys() for idx in concept_indices])

            for conjunction in conjunctions:
                #print('\t class: %s' % str(conjunction))

                def in_class(category):
                    for i, idx in enumerate(concept_indices):
                        if(category[idx] != data_iterator.concepts[idx][conjunction[i]]): return False

                    return True

                in_class_aux = np.vectorize(lambda datapoint: in_class(datapoint.category))

                # For each n-gram, check if it is a good predictor of the class (equivalent to building a decision tree of depth 1)
                gold = in_class_aux(data_iterator.dataset)
                for feature_idx in range(nb_ngrams):
                    ngram = ngrams[feature_idx]

                    if(ngram == ()): continue


                    ratio = gold.mean()
                    baseline_accuracy = max(ratio, (1.0 - ratio)) # Precision of the majority class baseline

                    feature_type = 'presence'
                    prediction = feature_vectors[:, feature_idx]

                    matches = (gold == prediction)

                    accuracy = matches.mean()
                    error_reduction = (1 - baseline_accuracy) / (1 - accuracy)

                    precision = gold[prediction].mean() # 1 means that the symbol entails the property
                    if((precision > rule_precision_threshold) and (prediction.sum() > rule_frequence_threshold * prediction.size)):
                        rules[ngram].append((set(conjunction), precision))
                        #print('%s means %s (%f)' % (ngram, conjunction, precision))
                    recall = prediction[gold].mean() # 1 means that the property entails the symbol
                    f1 = (2 * precision * recall / (precision + recall)) if(precision + recall > 0.0) else 0.0

                    item = (accuracy, baseline_accuracy, error_reduction, precision, recall, f1, conjunction, ngram, feature_type)
                    results_binary_classifier.append(item)

                    feature_type = 'absence'
                    prediction = (prediction ^ True)

                    matches = (gold == prediction)

                    accuracy = matches.mean()
                    error_reduction = (1 - baseline_accuracy) / (1 - accuracy)

                    precision = gold[prediction].mean() # 1 means that the absence of the symbol entails the property
                    if((precision > rule_precision_threshold) and (prediction.sum() < (1 - rule_frequence_threshold) * prediction.size)):
                        rules[('NOT', ngram)].append((set(conjunction), precision))
                        #print('NOT %s means %s (%f)' % (ngram, conjunction, precision))
                    recall = prediction[gold].mean() # 1 means that the property entails the absence of the symbol
                    f1 = (2 * precision * recall / (precision + recall)) if(precision + recall > 0.0) else 0.0

                    item = (accuracy, baseline_accuracy, error_reduction, precision, recall, f1, conjunction, ngram, feature_type)
                    results_binary_classifier.append(item)


                if(True): continue

                # Decision trees
                X = feature_vectors
                Y = gold # in_class_aux(data_iterator.dataset) # [in_class(datapoint.category) for datapoint in data_iterator.dataset]

                classifier = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth).fit(X, Y)

                n_leaves, depth = classifier.get_n_leaves(), classifier.get_depth()
                precision = classifier.score(X, Y) # Precision on the 'training set'
                ratio = (np.sum(Y) / Y.size)
                baseline_precision = max(ratio, (1.0 - ratio)) # Precision of the majority class baseline

                item = (
                    precision,
                    baseline_precision,
                    (precision / baseline_precision),
                    ((1 - baseline_precision) / (1 - precision)),
                    conjunction,
                    n_leaves,
                    depth,
                    classifier
                )

                results_decision_tree.append(item)

                #if(precision > 0.9):
                if(item[3] > 2.0):
                    print(item)
                    print(sklearn.tree.export_text(classifier, feature_names=ngrams, show_weights=True))

                    plt.figure(figsize=(12, 12))
                    sklearn.tree.plot_tree(classifier, filled=True)
                    plt.show()

        print("\nBest binary classifiers")
        print("\tby error reduction")
        results_binary_classifier.sort(reverse=True, key=(lambda e: e[2]))
        for e in results_binary_classifier[:10]:
            print(e)

        print("\tby F1")
        results_binary_classifier.sort(reverse=True, key=(lambda e: e[5]))
        for e in results_binary_classifier[:10]:
            print(e)

    clean_rules = []
    for ngram, l in rules.items():
        lhs = ngram # In fact it could be ('NOT', ngram)
        rhs = set.union(*[e[0] for e in l])
        clean_rules.append((lhs, rhs))

    # Removes redundant rules (i.e., if (x1, …, xn) => Y, then (x1, …, x{n+1}) => Y)
    # Does currently nothing for negative rules (whereas if NOT(x1, …, x{n+1}) => Y, then NOT(x1, …, xn) => Y)
    for lhs, rhs in clean_rules:
        ok = True
        for lhs2, rhs2 in clean_rules:
            if(lhs == lhs2): continue
            if(rhs != rhs2): continue

            # Checks whether lhs2 is a subpart of lhs
            for i in range(1 + len(lhs) - len(lhs2)):
                if(lhs[i:(i + len(lhs2))] == lhs2):
                    ok = False
                    break

            if(not ok): break

        if(ok or lhs[0] == 'NOT'): print('%s => %s' % (lhs, rhs))

    print("\nBest decision trees")
    results_decision_tree.sort(reverse=True, key=(lambda e: e[3]))
    for e in results_decision_tree[:10]:
        print(e)
