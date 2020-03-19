import torch
import torch.nn as nn
import numpy as np

import itertools
import tqdm
import sklearn.tree
import matplotlib.pyplot as plt

from collections import defaultdict

# `data` should either be a pair (dataset, messages) or None. If None, then messages will be generated
# Except for generating the messages,
#   we need model for _.base_alphabet_size
#   and data_iterator for _.concepts
def decision_tree_standalone(model, data_iterator):
    n = len(data_iterator)
    if(n is None): n = 10000
    dataset = np.array([data_iterator.get_datapoint(i) for i in range(n)])
    
    categories = np.array([datapoint.category for datapoint in dataset])

    print("Generating the messages…")
    messages = []
    with torch.no_grad():
        model.eval()
        for datapoint in tqdm.tqdm(dataset):
            sender_outcome = model.sender(datapoint.img.unsqueeze(0))
            message = sender_outcome.action[0].view(-1).tolist()
            messages.append(message)
            #print((datapoint.category, message))
    messages = np.array(messages)

    return decision_tree(messages=messages, categories=categories, alphabet_size=(1 + model.base_alphabet_size), concepts=data_iterator.concepts)

# The messages must be iterables of integers between 0 (included) and `alphabet_size` (excluded)
def decision_tree(messages, categories, alphabet_size, concepts):
    # As features, we will use the presence of n-grams
    n = 3
    print('We will consider %i-grams' % n)

    def get_ngrams(message, max_length):
        for s in message: yield (s,) # I separate unigrams so that we don't return the unigram composed of the out-of-message symbol

        message_tmp = [alphabet_size] + list(message) + [alphabet_size] # I add an out-of-message symbol at the beginning and at the end
        for l in range(2, (min(max_length, len(message_tmp)) + 1)): # Length of the n-gram
            for i in range(len(message_tmp) - l + 1):
                yield tuple(message_tmp[i:(i + l)])

    # Determines the set of n-grams
    ngrams_idx = defaultdict(itertools.count().__next__) # From ngrams (tuples) to indices
    for message in messages:
        for ngram in get_ngrams(message, n): _ = ngrams_idx[ngram]
    ngrams = sorted(list(ngrams_idx.keys()), key=len) # From indices to ngrams (tuple); we will stick to a Python list as tuples are a bit tricky to put into Numpy arrays
    ngrams_idx = dict([(ngram, i) for (i, ngram) in enumerate(ngrams)])

    # Generates the feature vectors
    feature_vectors = []
    for message in messages:
        v = np.zeros(len(ngrams), dtype=np.int)
        for ngram in get_ngrams(message, n):
            idx = ngrams_idx[ngram]
            v[idx] += 1
        feature_vectors.append(v)
    
    feature_vectors = np.array(feature_vectors) # Integer tensor
    boolean_feature_vectors = (feature_vectors != 0) # Boolean tensor

    # Super decision-tree
    print('Full tree')

    def category_to_name(category):
        return '_'.join([str(v) for v in category])

    X = feature_vectors # We might want to use only some of these (e.g., only the unigrams)
    Y = np.array([category_to_name(category) for category in categories]) 

    classifier = sklearn.tree.DecisionTreeClassifier(max_depth=64).fit(X, Y)

    n_leaves, depth = classifier.get_n_leaves(), classifier.get_depth()
    accuracy = classifier.score(X, Y) # Accuracy on the 'training set'

    print('Decision tree accuracy: %s' % accuracy)
    print('Number of leaves: %i' % n_leaves)
    print('Depth: %i' % depth)
    #print(sklearn.tree.export_text(classifier, feature_names=ngrams, show_weights=True))

    #plt.figure(figsize=(12, 12))
    #sklearn.tree.plot_tree(classifier, filled=True)
    #plt.show()

    # Conceptual decision-tree
    for dim in range(len(concepts)):
        print('Tree for concept %i' % dim)

        def category_to_name(category):
            return str(category[dim])

        X = feature_vectors # We might want to use only some of these (e.g., only the unigrams)
        Y = np.array([category_to_name(category) for category in categories]) 

        classifier = sklearn.tree.DecisionTreeClassifier(max_depth=64).fit(X, Y)

        n_leaves, depth = classifier.get_n_leaves(), classifier.get_depth()
        accuracy = classifier.score(X, Y) # Accuracy on the 'training set'

        print('Decision tree accuracy: %s' % accuracy)
        print('Number of leaves: %i' % n_leaves)
        print('Depth: %i' % depth)
        #print(sklearn.tree.export_text(classifier, feature_names=ngrams, show_weights=True))

        #plt.figure(figsize=(12, 12))
        #sklearn.tree.plot_tree(classifier, filled=True)
        #plt.show()


    # Rules stuff
    rule_precision_threshold = 0.95
    rule_frequence_threshold = 0.05
    rules = defaultdict(list) # From ngram to list of RHS·s (to be conjuncted)

    results_decision_tree = []
    max_depth = 2 # None # We could successively try with increasing depth
    max_conjunctions = 3 # len(concepts)
    for size_conjunctions in range(1, (max_conjunctions + 1)):
        print('\nConjunctions of %i concepts' % size_conjunctions)
        results_binary_classifier = []

        for concept_indices in itertools.combinations(range(len(concepts)), size_conjunctions): # Iterates over all subsets of [|0, `len(concepts)`|[ of size `size_conjunctions`
            #print([data_iterator.concept_names[idx] for idx in concept_indices])

            # For each selected concept, we pick a value
            conjunctions = itertools.product(*[concepts[idx].keys() for idx in concept_indices])

            for conjunction in conjunctions:
                #print('\t class: %s' % str(conjunction))

                def in_class(category):
                    for i, idx in enumerate(concept_indices):
                        if(category[idx] != concepts[idx][conjunction[i]]): return False

                    return True

                #in_class_aux = np.vectorize(lambda datapoint: in_class(datapoint.category))

                # For each n-gram, check if it is a good predictor of the class (equivalent to building a decision tree of depth 1)
                #gold = in_class_aux(dataset)
                gold = np.array([in_class(category) for category in categories])
                for feature_idx, ngram in enumerate(ngrams):

                    ratio = gold.mean()
                    baseline_accuracy = max(ratio, (1.0 - ratio)) # Precision of the majority class baseline

                    feature_type = 'presence'
                    prediction = boolean_feature_vectors[:, feature_idx]

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

                if(True): continue # TODO only for simple concepts, and over all values

                # Decision trees
                X = feature_vectors
                Y = gold # in_class_aux(dataset) # [in_class(datapoint.category) for datapoint in dataset]

                classifier = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth).fit(X, Y)

                n_leaves, depth = classifier.get_n_leaves(), classifier.get_depth()
                accuracy = classifier.score(X, Y) # Accuracy on the 'training set'
                ratio = (np.sum(Y) / Y.size)
                baseline_accuracy = max(ratio, (1.0 - ratio)) # Accuracy of the majority class baseline

                item = (
                    accuracy,
                    baseline_accuracy,
                    (accuracy / baseline_accuracy),
                    ((1 - baseline_accuracy) / (1 - accuracy)),
                    conjunction,
                    n_leaves,
                    depth,
                    classifier
                )

                results_decision_tree.append(item)

                #if(accuracy > 0.9):
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
    clean_rules_by_lhs = defaultdict(list)
    for ngram, l in rules.items():
        lhs = ngram # In fact it could be ('NOT', ngram)
        rhs = set.union(*[e[0] for e in l])
        rule = (lhs, rhs)

        clean_rules.append(rule)
        clean_rules_by_lhs[lhs].append(rule)

    # Removes redundant rules (i.e., if (x1, …, xn) => Y, then (x1, …, x{n+1}) => Y, so we don't need the latter)
    # Does currently nothing for negative rules (whereas if NOT(x1, …, x{n+1}) => Y, then NOT(x1, …, xn) => Y, so we don't need the latter)

    # Iterates over all sublists of `l`
    def iter_sublists(l, non_empty=True, strict=True):
        n = len(l)
        max_k = n
        if(strict): max_k -= 1

        if(not non_empty): yield []

        for k in range(max_k):
            for i in range(n - k):
                yield l[i:(i + k + 1)]

    for lhs, rhs in clean_rules:
        ok = True
        
        if(lhs[0] != 'NOT'):
            for lhs2, rhs2 in clean_rules:
                if(lhs == lhs2): continue
                if(rhs != rhs2): continue

                # Checks whether lhs2 is a subpart of lhs
                for i in range(1 + len(lhs) - len(lhs2)):
                    if(lhs[i:(i + len(lhs2))] == lhs2):
                        ok = False
                        break

                if(not ok): break
        
        # Checks whether the rule can be obtained compositionaly from other rules
        if(lhs[0] != 'NOT'):
            rhs_remainder = set(rhs)
            for lhs2 in iter_sublists(lhs):
                for _, rhs2 in clean_rules_by_lhs[lhs2]:
                    rhs_remainder.difference_update(rhs2)
                
                    if(not rhs_remainder):
                        ok = False
                        break

        if(ok): print('%s => %s' % (lhs, rhs))

    print("\nBest decision trees")
    results_decision_tree.sort(reverse=True, key=(lambda e: e[3]))
    for e in results_decision_tree[:10]:
        print(e)
