import numpy as np

import sys
import os
from collections import namedtuple, defaultdict
import itertools
import random

class Batch():
    def __init__(self, size, predicate, candidates):
       self.size = size # int
       self.predicate = predicate # list[Predicate]
       self.candidates = candidates # list[list[Candidate]]

    def __eq__(self, other):
        if(not isinstance(other, Batch)): return NotImplemented

        if(self.size != other.size):
            print("Batch.__eq__//size")
            return False
        if(self.predicate != other.predicate):
            print("Batch.__eq__//predicate")
            return False
        if(self.candidates != other.candidates):
            print("Batch.__eq__//candidates")
            return False

        return True

    # Used for debugging.
    # Returns (list[None|int], list[None|int], list[None|int]).
    #def indices(self):
    #    return ([dp.idx for dp in self.original], [dp.idx for dp in self.target], [dp.idx for l in self.base_distractors for dp in l])

    # Used for debugging.
    # Returns (list[None|tuple[int]], list[None|tuple[int]], list[None|tuple[int]]).
    #def categories(self):
    #    return ([dp.category for dp in self.original], [dp.category for dp in self.target], [dp.category for l in self.base_distractors for dp in l])

    # Used for debugging.
    # Returns an int.
    #def signature(self):
    #    a, b, c = self.indices()
    #    d, e, f = self.categories()
    #
    #    return hash(tuple([tuple(x) for x in [a, b, c, d, e, f]]))

    #def original_img(self, stack=False, f=None):
    #    if(f is None): f = (lambda x: x)
    #
    #    tmp = [f(x.img) for x in self.original]
    #
    #    if(stack): return torch.stack(tmp)
    #    else: return tmp
    
    #def target_img(self, stack=False, f=None):
    #    if(f is None): f = (lambda x: x)
    #
    #    tmp = [f(x.img) for x in self.target]
    #
    #    if(stack): return torch.stack(tmp)
    #    else: return tmp

    # Returns a list or a tensor of the original/target categories of the batch, possibly transformed by a function first.
    # stack: whether to return a tensor (True) or a list (False)
    # f: the function (if any) to apply to each category
    #def target_category(self, stack=False, f=None):
    #    if(f is None): f = (lambda x: x)
    #
    #    tmp = [f(x.category) for x in self.target]
    #
    #    if(stack): return torch.tensor(tmp)
    #    else: return tmp
    
    #def base_distractors_img(self, flat=False, stack=False, f=None):
    #    if(f is None): f = (lambda x: x)
    #
    #    if(not flat):
    #        tmp = [[f(x.img) for x in base_distractor] for base_distractor in self.base_distractors] # list[list[tensor of shape (*IMG_SHAPE)]]
    #        if(stack): tmp = list(map(torch.stack, tmp)) # list[tensor of shape (1, *IMG_SHAPE)]
    #    else: tmp = [f(x.img) for base_distractor in self.base_distractors for x in base_distractor] # list[tensor of shape (*IMG_SHAPE)]
    #
    #    if(stack): return torch.stack(tmp)
    #    else: return tmp
    
    #def get_images(self, original=True, target=True, base_distractors=True):
    #    images = []
    #    if(original): images.extend(self.original_img())
    #    if(target): images.extend(self.target_img())
    #    if(base_distractors): images.extend(self.base_distractors_img(flat=True))
    #
    #    return images

class Property():
    # name: str
    # values: set[Value]
    def __init__(self, name, values=set()):
        self.name = name
        self.values = values

    def __str__(self):
        return self.name


# A predicate is equivalent to a first-order logic formula.
class Predicate():
    # Computes the truth value in {-1, 0, 1} of the predicate applied on a given candidate based on Kleene logic. (-1 for false, 0 for unknown, 1 for true)
    # candidate: Candidate
    # Outputs an int.
    def check(self, candidate):
        raise NotImplementedError

    # target: -1, 0 or 1
    # Outputs a list[Candidate].
    def build(self, target=1):
        raise NotImplementedError

    # Outputs a Bool.
    def isVerifiable(self):
        return (len(self.build(target=1)) > 0)

    # Outputs a Bool.
    def isFalsifiable(self):
        return (len(self.build(target=-1)) > 0)

    # Outputs a Bool.
    def isNontrivial(self):
        return self.isVerifiable() and self.isFalsifiable()

    # other: Predicate
    # Outputs a Bool.
    def isAsStrongAs(self, other):
        for c in self.build(target=1):
            if(other.check(c) < 1): return False

        for c in self.build(target=0):
            if(other.check(c) < 0): return False

        return True
    
    # other: Predicate
    # Outputs a Bool.
    def isEquivalentTo(self, other):
        return self.isAsStrongAs(other) and other.isAsStrongAs(self)

    # others: iterable[Predicate]
    # Outputs a Bool.
    def hasEquivalentIn(self, others):
        for other in others:
            if(self.isEquivalentTo(other)):
                return True
        return False

    def __repr__(self):
        return str(self)

class Value(Predicate):
    # name: str
    # prop: Property
    def __init__(self, name, prop):
        self.name = name
        self.prop = prop

    # candidate: Candidate
    # Outputs a truth value in {-1, 0, 1} based on Kleene logic. (-1 for false, 0 for unknown, 1 for true)
    def check(self, candidate):
        v = candidate.get(self.prop)
        if(v is None): return 0
        if(v == self): return 1
        return -1
    
    # target: -1, 0 or 1
    # Outputs a list[Candidate].
    def build(self, target=1):
        if(target == 1): return [Candidate(prop2value={self.prop: self})]
        if(target == -1): return [Candidate(prop2value={self.prop: value}) for value in self.prop.values if value != self]
        if(target == 0): return [Candidate(prop2value={self.prop: None})]
        assert False

    def __str__(self): return self.name

class Negation(Predicate):
    def __init__(self, predicate):
        self.predicate = predicate # Predicate

    # candidate: Candidate
    # Outputs a truth value in {-1, 0, 1} based on Kleene logic. (-1 for false, 0 for unknown, 1 for true)
    def check(self, candidate):
        return -self.predicate.check(candidate)
    
    # target: -1, 0 or 1
    # Outputs a list[Candidate].
    def build(self, target=1):
        return self.predicate.build(target=(-target))

    def __str__(self): return f"(¬{self.predicate})"

class Conjunction(Predicate):
    # pred1, pred2: Predicate
    def __init__(self, pred1, pred2):
        self.pred1 = pred1
        self.pred2 = pred2

    # candidate: Candidate
    # Outputs a truth value in {-1, 0, 1} based on Kleene logic. (-1 for false, 0 for unknown, 1 for true)
    def check(self, candidate):
        return min(self.pred1.check(candidate), self.pred2.check(candidate))
    
    # target: -1, 0 or 1
    # Outputs a list[Candidate].
    def build(self, target=1):
        if(target == 1):
            l1 = self.pred1.build(target=1)
            l2 = self.pred2.build(target=1)
            s = set()
            for c1, c2 in itertools.product(l1, l2):
                c = c1.merge(c2)
                if(c is not None): s.add(c)
            
            return list(s)
            
        if(target == 0):
            l1 = self.pred1.build(target=0)
            l2 = self.pred2.build(target=0)
            s = set()
            for c1 in l1:
                if(self.pred2.check(c1) >= 0): s.add(c1)
            for c2 in l2:
                if(self.pred1.check(c2) >= 0): s.add(c2)
            
            return list(s)
            
        if(target == -1):
            l1 = self.pred1.build(target=-1)
            l2 = self.pred2.build(target=-1)
            s = set()
            for c in itertools.chain(l1, l2): s.add(c)
            
            return list(s)
        
        assert False
            

    def __str__(self):
        return f"({self.pred1}∧{self.pred2})"


class Candidate():
    def __init__(self, prop2value):
        self.prop2value = prop2value # dict[Property, Value|NoneType]

    # prop: Property
    # Outputs a Value|NoneType.
    def get(self, prop):
        return self.prop2value.get(prop)

    # other: Candidate
    # Outputs a Candidate|NoneType.
    def merge(self, other):
        prop2value = dict(self.prop2value) # copy
        for (p, v) in other.prop2value.items():
            if(not (p in prop2value)): prop2value[p] = v
            elif(prop2value[p] != v): return None

        return Candidate(prop2value)

    def __str__(self):
        return f"{{{','.join([str(value) for value in self.prop2value.values()])}}}"


class failureBasedDistribution():
    def __init__(self, nb_categories, momentum_factor=0.99, smoothing_factor=1.0):
        self.momentum_factor = momentum_factor

        # Initialisation with smoothing
        self.counts_matrix = np.full((nb_categories, nb_categories), smoothing_factor)
        self.failure_matrix = np.full((nb_categories, nb_categories), (0.5 * smoothing_factor))
        np.fill_diagonal(self.failure_matrix, 0.0)

    def update(self, target_category_idx, distractor_category_idx, failure):
        # Note: if the same pair (target, distractor) appears multiple time, the momentum factor would still be applied only once
        self.counts_matrix[target_category_idx, distractor_category_idx] *= self.momentum_factor
        np.add.at(self.counts_matrix, (target_category_idx, distractor_category_idx), 1.0)
        self.failure_matrix[target_category_idx, distractor_category_idx] *= self.momentum_factor
        np.add.at(self.failure_matrix, (target_category_idx, distractor_category_idx), failure)

    def distribution(self, category_idx, allowed_categories_idx=None):
        if(allowed_categories_idx is None): unnormalised_dist = (self.failure_matrix[category_idx] / self.counts_matrix[category_idx])
        else: unnormalised_dist = (self.failure_matrix[category_idx, allowed_categories_idx] / self.counts_matrix[category_idx, allowed_categories_idx])

        return (unnormalised_dist / np.linalg.norm(unnormalised_dist, 1))

    # Returns a distractor category based on an original/target category.
    def sample(self, category_idx, allowed_categories_idx=None):
        dist = self.distribution(category_idx, allowed_categories_idx)

        if(allowed_categories_idx is None): allowed_categories_idx = range(dist.shape[0])

        return np.random.choice(a=allowed_categories_idx, p=dist)

class SimpleDataset():
    def __init__(self, device='cpu', batch_size=128, nb_candidates=16, sampling_strategies=["random"], properties="3-4", max_depth=2, nontrivial_only=True, allow_negation=True, allow_conjunction=True):
        self.device = device
        self.batch_size = batch_size
        self.nb_candidates = nb_candidates
        self.sampling_strategies = sampling_strategies

        # Generates the properties and the values ("3-4" means a 3-valued property and a 4-valued one).
        self.properties = set() # set[Property]
        self.values = set() # set[Value]
        for i, n in enumerate([int(s) for s in properties.split("-")]):
            property_name = f"P{i}"
            prop = Property(name=property_name)
            prop.values = {Value(prop=prop, name=f"{property_name}-v{j}") for j in range(n)}
            
            self.properties.add(prop)
            self.values.update(prop.values)

        # Generates predicates.
        self.predicates = self.generateAllPredicates(max_depth=max_depth, nontrivial_only=nontrivial_only, allow_negation=allow_negation, allow_conjunction=allow_conjunction)

    # nontrivial_only: bool, indicates whether all subpredicates should be nontrivial
    # max_depth: int
    # Outputs a list[Predicate]
    def generateAllPredicates(self, max_depth, nontrivial_only, allow_negation, allow_conjunction):
        depth2predicates = [] # list[list[Predicate]]
        depth2predicates.append([value for value in self.values if (not nontrivial_only or value.isNontrivial())]) # All predicates of depth 1
        while(len(depth2predicates) < max_depth):
            predicates = list() # list[Predicate]

            if(allow_negation):
                for predicate in depth2predicates[-1]:
                    pred = Negation(predicate=predicate)

                    if(pred.hasEquivalentIn(itertools.chain(*depth2predicates, predicates))): continue
                    #if(nontrivial_only and (not pred.isNontrivial())): continue

                    predicates.append(pred)
             
            if(allow_conjunction):
                for pred1 in depth2predicates[-1]:
                    for pred2 in itertools.chain.from_iterable(depth2predicates):
                        pred = Conjunction(pred1=pred1, pred2=pred2)

                        if(pred.hasEquivalentIn(itertools.chain(*depth2predicates, predicates))): continue
                        if(nontrivial_only and (not pred.isNontrivial())): continue

                        predicates.append(pred)

            depth2predicates.append(predicates)

        return list(itertools.chain.from_iterable(depth2predicates)) # list[Predicate]

    def print_info(self):
        print(f"{len(self.properties)} properties:")
        #for prop in self.properties: print(f"{prop} (size {len(prop.values)})")
        for prop in self.properties: print(f"{prop} ({prop.values})")

        print(f"{len(self.predicates)} predicates ({self.predicates})")

    # Returns a Batch.
    def get_batch(self, size=None, nb_candidates=None, data_type='any', sampling_strategies=None):
        """Generates a batch as a Batch object.
        size: int, the size of the batch.
        data_type: string ("train", "test" or "any"), indicates from what part the candidates are selected.
        sampling_strategies: list[string], indicates how the candidates are determined.
        """
        batch = []
        if(size is None): size = self.batch_size
        if(nb_candidates is None): nb_candidates = self.nb_candidates
        if(sampling_strategies is None): sampling_strategies = self.sampling_strategies
        for _ in range(size):
            # Selects a predicate.
            # TODO

            # Selects candidates.
            # TODO

            # Original image
            _original = self.category_to_datapoint(target_category, data_type).toInput(keep_category=keep_category, device=self.device, keep_idx=keep_idx)

            # Target image
            if(target_is_original): _target = _original.copy(deep=False)
            else: _target = self.category_to_datapoint(target_category, data_type).toInput(keep_category=keep_category, device=self.device, keep_idx=keep_idx) # Same category
            
            # Base distractors
            _base_distractors = []
            for sampling_strategy in sampling_strategies:
                distractor_category = self.sample_category(sampling_strategy, target_category, no_evaluation)
                distractor = self.category_to_datapoint(distractor_category, data_type).toInput(keep_category=keep_category, device=self.device, keep_idx=keep_idx)
                distractor.add_normal_noise_(self.noise)

                _base_distractors.append(distractor)

            batch.append((_original, _target, _base_distractors))

        original, target, base_distractors = zip(*batch) # Unzips the list of pairs (to a pair of lists).

        return Batch(size=size, original=original, target=target, base_distractors=base_distractors)

    # If `d` is -1, all categories are used during training.
    # Otherwise, a reference category and all categories with a distance from it that is a multiple of `d` are reserved for evaluation.
    # The reference category is picked randomly except if given as `ref_category`.
    def set_evaluation_categories(self, concepts, d, ref_category=None, random_ref=False):
        training_categories = set()
        evaluation_categories = set()

        if(ref_category is not None): assert (not random_ref), "One cannot both specify a reference category and ask for a random one at the same time."
        else: ref_category = np.array([np.random.randint(len(concept)) for concept in concepts]) if(random_ref) else np.full(len(concepts), 0)

        category = np.full(len(concepts), 0) # Encodes the current category.
        while(True): # Iterates over all categories to categorise them. Alternatively, we could use the number of categories
            dist = (category != ref_category).sum()
            if((d >= 0) and ((dist % d) == 0)):
                evaluation_categories.add(tuple(category))
            else:
                training_categories.add(tuple(category))

            # Let's go to the next category
            for i, concept in enumerate(concepts):
                if(category[i] < (len(concept) - 1)):
                    category[i] += 1
                    break

                category[i] = 0
            if(category.sum() == 0): break # If we're back to (0,0,…,0), then we've seen all categories

        self.training_categories = training_categories
        self.evaluation_categories = evaluation_categories

        return ref_category

    def _different_category(self, category, no_evaluation):
        """Returns a category that is different from `category`.
        If `no_evaluation` is True, evaluation categories are ignored."""
        categories = self.training_categories
        if(not no_evaluation): categories = categories.union(self.evaluation_categories)
        categories = list(categories.difference(set([category])))
        assert (categories != []), f"There is no other category than category {category} (with{'out considering' if(no_evaluation) else ''} evaluation categories)."

        return random.choice(categories)

        # The following code was very efficient, but only works when there is no split between training and evaluation categories
        #distance = np.random.randint(self.nb_concepts) + 1
        #return self._distance_to_category(category, distance)

    def sample_category(self, sampling_strategy, category, no_evaluation):
        if(sampling_strategy == 'hamming1'): # Selects a category at distance 1 in the concept space
            return self._distance_to_category(category, 1, no_evaluation)

        if(sampling_strategy == 'different'): # Selects a different category
            return self._different_category(category, no_evaluation)

        if(sampling_strategy == 'difficulty'): # Selects a category based on the difficulty scores, that are softmaxed
            category_idx = self.category_idx(category)
            allowed_categories_idx = self.training_categories_idx if(no_evaluation) else None
            sample_idx = self.failure_based_distribution.sample(category_idx, allowed_categories_idx)

            return self.category_tuple(sample_idx)

        if(sampling_strategy == 'same'): # Should not be used during training
            return category

        if(sampling_strategy == "random"):
            return random.choice(list(self.training_categories))

        assert False, ('Sampling strategy \'%s\' unknown.' % sampling_strategy)

    # Should only be used for debugging purpose. Use `get_batch` instead
    # Returns a DataPoint.
    # i: int
    def get_datapoint(self, i):
        return self._dataset[i]

    # Category tuples are read from left to right (contrary to usual numbers)
    # Return a tuple[int].
    # category_idx: int
    def category_tuple(self, category_idx):
        ks = []
        k = 1
        for i, concept in enumerate(self.concepts):
            ks.append(k)
            k *= len(concept)
        ks.reverse()

        l = []
        remainder = category_idx
        for k in ks:
            l.append(remainder // k)
            remainder = (remainder % k)
        l.reverse()

        category_tuple = tuple(l)

        #if(np.random.randint(2)): assert self.category_idx(category_tuple) == category_idx # DEBUG ONLY

        return category_tuple

    # Returns an int.
    # category_tuple: tuple[int]
    def category_idx(self, category_tuple):
        category_idx = 0
        k = 1
        for i, concept in enumerate(self.concepts):
            category_idx += category_tuple[i] * k
            k *= len(concept)

        #if(np.random.randint(2)): assert self.category_tuple(category_idx) == category_tuple # DEBUG ONLY

        return category_idx
    
    # Should be consistant with `category_to_datapoint`
    # data_type: string ("train", "test" or "any")
    # no_evaluation: bool
    def size(self, data_type, no_evaluation):
        size = 0
        
        categories = self.training_categories
        if(not no_evaluation): categories = categories.union(self.evaluation_categories)

        for category in categories:
            size += self.category_size(category, data_type)

        return size

    #def __len__(self):
    #    return len(self._dataset)

    # Returns an int.
    # category: tuple[int]
    # data_type: string ("train", "test" or "any")
    def category_size(self, category, data_type):
        split = self.category_split(category)
        
        if(data_type == 'train'): return split[1] - split[0]
        elif(data_type == 'test'): return split[2] - split[1]
        elif(data_type == 'any'): return split[-1] - split[0]
        else: assert False, ('Data type \'%s\' unknown.' % data_type)

    # Returns of list[int] of length 3.
    # category: tuple[int]
    def category_split(self, category):
        l = len(self.categories[category])
        
        if(category in self.training_categories):
            split_point = ((4 * l) // 5)
            return [0, split_point, l] # 4/5th in the train portion, 1/5th in the test portion
        
        return [0, 0, l] # Everything in the test portion

    # Returns a Datapoint.
    # category: tuple[int]
    # data_type: string
    def category_to_datapoint(self, category, data_type):
        split = self.category_split(category)

        if(data_type == 'train'): a, b = split[0], (split[1]-1)
        elif(data_type == 'test'): a, b = split[1], (split[2]-1)
        elif(data_type == 'any'): a, b = split[0], (split[-1]-1)
        else: assert False, ('Data type \'%s\' unknown.' % data_type)
        i = random.randint(a, b) # A random integer between a and b (included)

        return self.categories[category][i]

def get_data_loader(args):
    sampling_strategies = args.sampling_strategies.split('/')

    dataset = SimpleDataset(args.same_img, evaluation_categories=args.evaluation_categories, data_set=args.data_set, display=args.display, noise=args.noise, device=args.device, batch_size=args.batch_size, sampling_strategies=sampling_strategies, binary=args.binary_dataset, constrain_dim=args.constrain_dim, args=args)

    dataset.print_info()

    return dataset


if(__name__ == "__main__"):
    dataset = SimpleDataset(device='cpu', batch_size=128, nb_candidates=16, sampling_strategies=["random"], properties="3-3", max_depth=3, nontrivial_only=True, allow_negation=True, allow_conjunction=True)
    dataset.print_info()
