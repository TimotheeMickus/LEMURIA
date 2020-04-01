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
    # We try to visit each category on average 32 times
    batch_size = 256
    max_datapoints = 32768 # (2^15)
    n = (32 * data_iterator.nb_categories)
    #n = data_iterator.size(data_type='test', no_evaluation=False)
    n = min(max_datapoints, n) 
    nb_batch = int(np.ceil(n / batch_size))
    print('%i datapoints (%i batches)' % (n, nb_batch))
    
    messages = []
    categories = []
    print("Generating the messages…")
    batch_numbers = range(nb_batch)
    with torch.no_grad():
        for _ in batch_numbers:
            model.start_episode(train_episode=False) # Selects agents at random if necessary

            batch = data_iterator.get_batch(batch_size, data_type='test', no_evaluation=False, sampling_strategies=['different'], keep_category=True) # Standard evaluation batch
            sender_outcome, receiver_outcome = model(batch)

            messages.extend([msg.tolist()[:l] for msg, l in zip(*sender_outcome.action)])
            categories.extend([x.category for x in batch.original])
    
    categories = np.array(categories) # Numpyfies the categories (but not the messages, as there are list of various length)

    return decision_tree(messages=messages, categories=categories, alphabet_size=(1 + model.base_alphabet_size), concepts=data_iterator.concepts)

# For a sequence of symbols `l`, yields all tuples of strictly increasing tuples of size between 1 and `max_disj`, with values between 0 and `max_val`, and containing the corresponding element in `l` (i.e., the ith tuple will contain l[i])
def apply_disj(l, max_disj, max_val):
    # Increment a list `l` such that it stays strictly increasing, without modifying a pivot
    # `pos` is the position of the pivot
    # `max_val` is the maximum allowed value
    # Should return None on invalid input
    def increment(target, pos):
        k = len(target)
        
        i = 0 # The position we are working on
        
        incr_loop = ((target[-1] <= max_val) and ((pos == 0) or target[pos-1] < target[pos]))
        while(incr_loop):
            if(i == pos):
                i += 1
                continue
            
            if(i == k): # The end
                return None
            
            target[i] += 1
            
            max_allowed = (max_val+1) if(i == (k-1)) else target[i+1]
            if(target[i] >= max_allowed):
                min_val = 0 if(i == 0) else (target[i-1]+1)
                target[i] = min_val
                i += 1
            else:
                return target
        return None
    
    # Initialisation
    tmp_output = [(s, ) for s in l]
    infos = [0 for _ in l] # Position of the original symbol in the inner tuple
    
    outer_loop = True
    while(outer_loop):
        yield tuple(tmp_output)
        
        # Incrementation
        i = 0 # The index of the inner tuple we are working on
        pos = infos[i] # Position of the original symbol in the inner tuple
        target = list(tmp_output[i]) # Inner tuple
        k = len(target) # Size of the inner tuple
        
        incr_loop = True
        while(incr_loop):
            #print(i, target, pos, k)
            tmp = increment(target, pos)
            #print('tmp: %s ' % tmp)
            if(tmp is not None):
                tmp_output[i] = tuple(tmp)
                incr_loop = False
            else: # We have to update one of the parameter (pos, k), or move to next inner tuple (i)
                pos += 1 # We try to increase the position of the original symbol
                if(pos < k):
                    x = l[i]
                    target = list(range(pos)) + [x] + list(range((x+1), (x+k-pos))) # Should be of same size (k)
                    #print('try target: %s' %target)
                    if((target[pos-1] < x) and (target[-1] <= max_val)):
                        tmp_output[i] = tuple(target)
                        infos[i] = pos
                        incr_loop = False
                        # Otherwise, in the next loop, `increment` will output None and we will try to increase `pos` again
                else: # We have to update one of the parameter (k), or move to next inner tuple (i)
                    pos = 0
                    
                    k += 1 # We try to increase the size of the disjunction
                    if(k <= max_disj):
                        x = l[i]
                        target = [x] + list(range((x+1), (x+k))) # Should be of size k (the incremented value)
                        #print('new target: %s' % target)
                        if(target[-1] <= max_val):
                            tmp_output[i] = tuple(target)
                            infos[i] = pos
                            incr_loop = False
                            # Otherwise, in the next loop, `increment` will output None and we will try to increase `pos`
                    else:
                        k = 1 # We reinitialise the inner tuple to (k, )
                        tmp_output[i] = (l[i], )
                        infos[i] = 0
                        
                        i += 1 # We have to move to the next inner tuple (i)
                        if(i < len(l)):
                            pos = infos[i]
                            target = list(tmp_output[i])
                            k = len(target)
                        else: # It's over
                            incr_loop = False
                            outer_loop = False


# Used to represent disjunctions of n-grams
class DisjTerm():
    def __init__(self, seq_terms):
        assert (len(seq_terms) > 1) # Everywhere else I am assuming this
        self.seq_terms = seq_terms # A set of SeqTerm·s

    # A DisjTerm entails another DisjTerm or SeqTerm if all of the disjuncts entails the other term
    def entails(self, other_term):
        for seq_term in self.seq_terms:
            if(not other_term.is_entailed(seq_term)): return False
        return True
    
    # A DisjTerm is entailed by a SeqTerm if each SeqTerm of the former is entailed by the latter
    def is_entailed(self, other_seq_term):
        for seq_term in self.seq_terms:
            if(not seq_term.is_entailed(other_seq_term)): return False
        return True

    def __str__(self):
        return ('(%s)' % '|'.join([str(seq_term) for seq_term in self.seq_terms]))

    def can_equal(self, other):
        return isinstance(other, DisjTerm)

    def __eq__(self, other):
        return (self.can_equal(other) and other.can_equal(self) and (self.seq_terms == other.seq_terms))

    def __hash__(self): # 1 plus the sum of the hash of each SeqTerm
        tmp_hash = 1
        for seq_term in self.seq_terms: tmp_hash += hash(seq_term)
        return tmp_hash

# Used to represent n-grams
class SeqTerm():
    def __init__(self, t):
        self.t = t # A tuple

    def entails(self, other_term):
        return other_term.is_entailed(self)

    # A SeqTerm is entailed by another one if the former is a subsequence of the latter
    def is_entailed(self, other_seq_term):
        for i in range(1 + len(other_seq_term) - len(self)):
            other_part = other_seq_term.t[i:(i + len(self))]
            if(other_part == self.t): return True
        return False

    def __len__(self):
        return len(self.t)

    def __str__(self):
        return str(self.t)

    def can_equal(self, other):
        return isinstance(other, SeqTerm)

    def __eq__(self, other):
        return (self.can_equal(other) and other.can_equal(self) and (self.t == other.t))

    def __hash__(self):
        return hash(self.t)

# The messages must be iterables of integers between 0 (included) and `alphabet_size` (excluded)
# `gram_size` is the k max of k-grams to consider
def analyse(messages, categories, alphabet_size, concepts, gram_size, disj_size=1, feature_vectors=None, full_max_depth=128, conceptual_max_depth=64):
    result = {}

    # Returns all the k-grams for 0 < k <= `max_length` in the message with an out-of-message symbol at the beginning and the end
    def get_ngrams(message, max_length):
        assert (max_length > 0)
        for s in message: yield SeqTerm((s,)) # I separate unigrams so that we don't return the unigram composed of the out-of-message symbol

        message_tmp = [alphabet_size] + list(message) + [alphabet_size] # I add an out-of-message symbol at the beginning and at the end
        for k in range(2, (min(max_length, len(message_tmp)) + 1)): # Length of the n-gram
            for i in range(len(message_tmp) - k + 1):
                yield SeqTerm(tuple(message_tmp[i:(i + k)]))

    # Can be used to have disjunctions directly in the n-grams, but we will not do that
    def get_disj_ngrams(message, max_length):
        if(disj_size == 1):
            for ngram in get_ngrams(message, max_length): yield ngram
        else:
            for ngram in get_ngrams(message, max_length):
                for disj in apply_disj(ngram, disj_size, (alphabet_size + 1)): yield disj

    if(feature_vectors is None):
        # Determines the set of n-grams
        ngrams_idx = defaultdict(itertools.count().__next__) # From ngrams (tuples) to indices
        for message in messages:
            for ngram in get_ngrams(message, gram_size):
                _ = ngrams_idx[ngram]
        ngrams = np.array(sorted(ngrams_idx.keys(), key=len)) # From indices to ngrams (SeqTerm)
        ngrams_idx = {ngram: i for (i, ngram) in enumerate(ngrams)}

        # Generates the (n-gram) feature vectors
        ngram_vectors = np.zeros((len(messages), len(ngrams)), dtype=np.int)
        for i, message in enumerate(messages):
            for ngram in get_ngrams(message, gram_size):
                idx = ngrams_idx[ngram]
                ngram_vectors[i, idx] += 1

        if(disj_size == 1):
            features = ngrams
            features_idx = ngrams_idx

            feature_vectors = ngram_vectors
        else: # Now, we add the disjunctions of ngrams
            # Set of features
            disjunctions = []
            for i in range(2, (disj_size + 1)):
                for disjuncts in itertools.combinations(ngrams, i):
                    disjunction = DisjTerm(set(disjuncts))
                    disjunctions.append(disjunction)
            features = np.empty((len(ngrams) + len(disjunctions)), dtype=object)
            features[:len(ngrams)] = ngrams
            features[len(ngrams):] = disjunctions
            features_idx = {ngram: i for (i, ngram) in enumerate(features)}

            # Feature vectors
            feature_vectors = np.zeros((len(messages), len(features)), dtype=np.int)
            feature_vectors[:, :len(ngrams)] = ngram_vectors
            for i in range(len(ngrams), len(features)): # For all disjunctive terms (by index)
                disjunction = features[i]
                for ngram in disjunction.seq_terms:
                    feature_vectors[:, i] += feature_vectors[:, features_idx[ngram]]

        result['features'] = features
        result['features_idx'] = features_idx

    result['feature_vectors'] = feature_vectors

    # Super decision-tree
    def category_to_name(category):
        return '_'.join([str(v) for v in category])

    X = feature_vectors # We might want to use only some of these (e.g., only the unigrams)
    Y = np.array([category_to_name(category) for category in categories]) 

    classifier = sklearn.tree.DecisionTreeClassifier(max_depth=full_max_depth).fit(X, Y)
    accuracy = classifier.score(X, Y) # Accuracy on the 'training set'

    result['full_tree'] = (classifier, accuracy)

    # Conceptual decision-tree
    conceptual_trees = []
    for dim in range(len(concepts)):
        if(len(concepts[dim]) < 2): continue

        def category_to_name(category):
            return str(category[dim])

        X = feature_vectors # We might want to use only some of these (e.g., only the unigrams)
        Y = np.array([category_to_name(category) for category in categories]) 

        classifier = sklearn.tree.DecisionTreeClassifier(max_depth=conceptual_max_depth).fit(X, Y)
        accuracy = classifier.score(X, Y) # Accuracy on the 'training set'
        
        conceptual_trees.append((dim, (classifier, accuracy)))

    result['conceptual_trees'] = conceptual_trees

    return result

# The messages must be iterables of integers between 0 (included) and `alphabet_size` (excluded)
# `gram_size` corresponds to the size of the n-grams to consider
# `disj_size` corresponds to 
def decision_tree(messages, categories, alphabet_size, concepts, gram_size=1, disj_size=1):
    print('First, some messages and categories:')
    for i in range(min(30, len(messages))):
        print('message', messages[i], '; category', categories[i])
    print()

    # As features, we will use the presence of n-grams
    print('We will consider %i-grams and disjunctions up to size %i' % (gram_size, disj_size))

    tmp = analyse(messages, categories, alphabet_size, concepts, gram_size, disj_size=disj_size)
    features = tmp['features']
    features_idx = tmp['features_idx']
    feature_vectors = tmp['feature_vectors']
    (full_tree, full_tree_accuracy) = tmp['full_tree']
    conceptual_trees = tmp['conceptual_trees']
    
    boolean_feature_vectors = (feature_vectors != 0) # Boolean tensor

    # Super decision-tree
    print('Full tree')
    n_leaves, depth = full_tree.get_n_leaves(), full_tree.get_depth()

    print('Decision tree accuracy: %s' % full_tree_accuracy)
    print('Number of leaves: %i' % n_leaves)
    print('Depth: %i' % depth)
    if(True):
        print(sklearn.tree.export_text(full_tree, feature_names=[str(f) for f in features], show_weights=True))
        
        plt.figure(figsize=(12, 12))
        sklearn.tree.plot_tree(full_tree, filled=True)
        plt.show()

    # Conceptual decision-tree
    for dim, (tree, accuracy) in conceptual_trees:
        print('Tree for concept %i' % dim)
        n_leaves, depth = tree.get_n_leaves(), tree.get_depth()

        print('Decision tree accuracy: %s' % accuracy)
        print('Number of leaves: %i' % n_leaves)
        print('Depth: %i' % depth)
        #print(sklearn.tree.export_text(tree, feature_names=[str(f) for f in features], show_weights=True))

        #plt.figure(figsize=(12, 12))
        #sklearn.tree.plot_tree(tree, filled=True)
        #plt.show()

    # Rules stuff
    rule_precision_threshold = 0.95
    rule_frequence_threshold = 0.05
    rules = defaultdict(list) # From LSH (feature or ('NOT', feature)) to list of RHS·s, which are sets of properties (we also indicate the precision of the rule)

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
                for feature_idx, feature in enumerate(features):

                    ratio = gold.mean()
                    baseline_accuracy = max(ratio, (1.0 - ratio)) # Precision of the majority class baseline

                    feature_type = 'presence'
                    prediction = boolean_feature_vectors[:, feature_idx]

                    matches = (gold == prediction)

                    accuracy = matches.mean()
                    error_reduction = (1 - baseline_accuracy) / (1 - accuracy)

                    precision = gold[prediction].mean() # 1 means that the symbol entails the property
                    if((precision > rule_precision_threshold) and (prediction.sum() > rule_frequence_threshold * prediction.size)):
                        rules[('', feature)].append((set(conjunction), precision))
                        #print('%s means %s (%f)' % (feature, conjunction, precision))
                    recall = prediction[gold].mean() # 1 means that the property entails the symbol
                    f1 = (2 * precision * recall / (precision + recall)) if(precision + recall > 0.0) else 0.0

                    item = (accuracy, baseline_accuracy, error_reduction, precision, recall, f1, conjunction, str(feature), feature_type)
                    results_binary_classifier.append(item)

                    feature_type = 'absence'
                    prediction = (prediction ^ True)

                    matches = (gold == prediction)

                    accuracy = matches.mean()
                    error_reduction = (1 - baseline_accuracy) / (1 - accuracy)

                    precision = gold[prediction].mean() # 1 means that the absence of the symbol entails the property
                    if((precision > rule_precision_threshold) and (prediction.sum() < (1 - rule_frequence_threshold) * prediction.size)):
                        rules[('NOT', feature)].append((set(conjunction), precision))
                        #print('NOT %s means %s (%f)' % (feature, conjunction, precision))
                    recall = prediction[gold].mean() # 1 means that the property entails the absence of the symbol
                    f1 = (2 * precision * recall / (precision + recall)) if(precision + recall > 0.0) else 0.0

                    item = (accuracy, baseline_accuracy, error_reduction, precision, recall, f1, conjunction, str(feature), feature_type)
                    results_binary_classifier.append(item)

        print("\nBest binary classifiers")
        print("\tby error reduction")
        results_binary_classifier.sort(reverse=True, key=(lambda e: e[2]))
        for e in results_binary_classifier[:10]:
            print(e)

        print("\tby F1")
        results_binary_classifier.sort(reverse=True, key=(lambda e: e[5]))
        for e in results_binary_classifier[:10]:
            print(e)

    # We potentially have many rules for the same LHS
    clean_rules = []
    clean_rules_by_lhs = defaultdict(list)
    for lhs, l in rules.items(): # `lhs` is (x, feature) where x is '' or 'NOT'
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

    # Determines whether a formula entails another
    def entails(p1, p2):
        t1, f1 = p1
        t2, f2 = p2

        if(t1 != t2): # A positive rule and a negative one
            return False
        
        elif(t1 == ''): # Positive rules
            return f1.entails(f2)

        else: # Negative rules
            return f2.entails(f1)

    for lhs1, rhs1 in clean_rules:
        ok = True
        
        for lhs2, rhs2 in clean_rules:
            if((rhs1 == rhs2) and (lhs1 != lhs2)):
                if(entails(lhs1, lhs2)):
                    ok = False
                    break

            if((lhs1 == lhs2) and (rhs1 != rhs2)):
                if(entails(rhs2, lhs1)):
                    ok = False
                    break
        
        # TODO Checks whether the rule can be obtained compositionaly from other rules (many situations are possible)
        t, f = lhs1
        if(t != 'NOT'):
            if(isinstance(f, DisjTerm)): # Handling of trivial disjunction (if a -> b and a' -> b, obviously a|a' -> b)
                trivial = True
                for seq_term in f.seq_terms:
                    also_entails = False
                    for _, rhs2 in clean_rules_by_lhs[('', seq_term)]:
                        if(rhs1.issubset(rhs2)):
                            also_entails = True
                            break
                    if(not also_entails):
                        trivial = False
                        break
                if(trivial): ok = False
                
            # Si DisjTerm
            # On regarde si tous les disjuncts entail la rhs
            # Si c'est le cas, on jarte
            

        if(ok):
            str_lhs1 = str(lhs1[1])
            if(lhs1[0] == 'NOT'): str_lhs1 = 'NOT(%s)' % str_lhs1
            print('%s => %s' % (str_lhs1, sorted(list(rhs1))))
