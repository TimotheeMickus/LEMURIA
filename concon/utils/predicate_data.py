import numpy as np

import sys
import os
from collections import namedtuple, defaultdict
import itertools
import random

class Batch():
    # TODO See TODO below. This class has to be changed to store tensors ready to be fed to the model.
    # MG: Done-ish
    def __init__(self, size, predicate, predicate_idx, candidate):
        self.size = size # int
        self.predicate = predicate # list[Predicate]
        self.predicate_idx = predicate_idx # list[int]
        self.candidate = candidate # list[Candidate]
        
        self.node_idx = None
        self.edge_idx = None
        self.graph_sizes = None

    # Predicate encoding
    def encode_predicates(self, mode='predicate', num_predicates=None):
        '''May encode predicates as simply their indices (sparse one-hot encoding).'''
        assert mode in ['predicate', 'sparse']
        if(mode=='predicate'): return self.predicate
        if(mode=='sparse'): return np.array(self.predicate_idx)

    # Tensorize this batch (using Dataset vocabulary)
    def tensorize(self, dataset):
        graphs = [[{p.name: v.name.split('-',1)[1] for p, v in c.prop2value.items() if v is not None}] for c in self.candidate]

        nl_s2i = dataset.node_s2i
        el_s2i = dataset.edge_s2i

        node_idx = []
        edge_idx = []
        graph_sizes = []

        padding_length = max(sum(len(o) + 1 for o in g) for g in graphs) if graphs else 1

        for g in graphs:
            n, e, s = dataset._tensorize_graph(g, nl_s2i, el_s2i, padding_length)
            node_idx.append(n)
            edge_idx.append(e)
            graph_sizes.append(s)

        self.node_idx = node_idx
        self.edge_idx = edge_idx
        self.graph_sizes = graph_sizes

        return self  # allow chaining

    def __eq__(self, other):
        if(not isinstance(other, Batch)): return NotImplemented

        if(self.size != other.size):
            print("Batch.__eq__//size")
            return False
        if(self.predicate != other.predicate):
            print("Batch.__eq__//predicate")
            return False
        if(self.candidate != other.candidate):
            print("Batch.__eq__//candidate")
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
    # values: list[Value]
    def __init__(self, name, values=list()):
        self.name = name
        # This is a shared list across instances, should it be? consider:
        # self.values = [] if values is None else list(values)
        self.values = values

    def __str__(self):
        return self.name


# A predicate is equivalent to a first-order logic formula.
class Predicate():
    def __init__(self):
        self._build_cache = dict() # dict[int, list[Candidate]]
    
    # Computes the truth value in {-1, 0, 1} of the predicate applied on a given candidate based on Kleene logic. (-1 for false, 0 for unknown, 1 for true)
    # candidate: Candidate
    # Outputs an int.
    def check(self, candidate):
        # MG TODO I guess I should do this
        raise NotImplementedError
    
    # target: -1, 0 or 1
    # Outputs a list[Candidate].
    def build(self, target=1):
        # Uncached version
        #return self._build(target)
        
        # Cached version
        if(target not in self._build_cache): self._build_cache[target] = self._build(target)
        return self._build_cache[target]
    
    # target: -1, 0 or 1
    # Outputs a list[Candidate].
    def _build(self, target=1):
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
        return str(self) # MG TODO fix representation (this causes infinite recursion (repr calls str etc.))


class Value(Predicate):
    # name: str
    # prop: Property
    def __init__(self, name, prop):
        super(Value, self).__init__()
        
        self.name = name
        self.prop = prop

    # Computes the truth value in {-1, 0, 1} of the predicate applied on a given candidate based on Kleene logic. (-1 for false, 0 for unknown, 1 for true)
    # candidate: Candidate
    # Outputs an int.
    def check(self, candidate):
        v = candidate.get(self.prop)
        if(v is None): return 0
        if(v == self): return 1
        return -1
    
    # target: -1, 0 or 1
    # Outputs a list[Candidate].
    def _build(self, target=1):
        if(target == 1): return [Candidate(prop2value={self.prop: self})]
        if(target == -1): return [Candidate(prop2value={self.prop: value}) for value in self.prop.values if value != self]
        if(target == 0): return [Candidate(prop2value={self.prop: None})]
        assert False

    def __str__(self): return self.name

class Negation(Predicate):
    def __init__(self, predicate):
        super(Negation, self).__init__()
        
        self.predicate = predicate # Predicate

    # Computes the truth value in {-1, 0, 1} of the predicate applied on a given candidate based on Kleene logic. (-1 for false, 0 for unknown, 1 for true)
    # candidate: Candidate
    # Outputs an int.
    def check(self, candidate):
        return -self.predicate.check(candidate)
    
    # target: -1, 0 or 1
    # Outputs a list[Candidate].
    def _build(self, target=1):
        return self.predicate.build(target=(-target))

    def __str__(self): return f"(¬{self.predicate})"

class Conjunction(Predicate):
    # pred1, pred2: Predicate
    def __init__(self, pred1, pred2):
        super(Conjunction, self).__init__()
        
        self.pred1 = pred1
        self.pred2 = pred2

    # Computes the truth value in {-1, 0, 1} of the predicate applied on a given candidate based on Kleene logic. (-1 for false, 0 for unknown, 1 for true)
    # candidate: Candidate
    # Outputs an int.
    def check(self, candidate):
        return min(self.pred1.check(candidate), self.pred2.check(candidate))
    
    # target: -1, 0 or 1
    # Outputs a list[Candidate].
    def _build(self, target=1):
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
    
    def __repr__(self):
        return str(self)


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

class Dataset():
    def __init__(self, device='cpu', batch_size=128, properties="3-4", max_depth=2, nontrivial_only=False, no_negation=False, no_conjunction=False, allow_indeterminate=False):
        self.device = device
        self.batch_size = batch_size
        self.allow_indeterminate = allow_indeterminate

        # Generates the properties and the values ("3-4" means a 3-valued property and a 4-valued one).
        self.properties = list() # list[Property]
        self.values = list() # list[Value]
        for i, n in enumerate([int(s) for s in properties.split("-")]):
            property_name = f"P{i}"
            prop = Property(name=property_name)
            prop.values = [Value(prop=prop, name=f"{property_name}-v{j}") for j in range(n)]
            
            self.properties.append(prop)
            self.values.extend(prop.values)

        # Generates predicates.
        self.predicates = self.generateAllPredicates(max_depth=max_depth, nontrivial_only=nontrivial_only, no_negation=no_negation, no_conjunction=no_conjunction) # ndarray[Predicate]

        # Stores predicates as tensors.
        self.object_token='<obj>'
        self.padding_token='<pad>'
        self.selfedge_label='<self>'
        self.noedge_label='<noedge>'
        self.obj2feat_edge='obj2feat'
        self.feat2obj_edge='feat2obj'

        # Build a global node vocabulary based on all predicates
        node_labels = {self.object_token}
        for property in self.properties:
            for value in property.values:
                # Store ("Px", "vx") pairs as node labels for features
                node_labels.add((property.name, value.name.split('-',1)[1]))

        self.node_i2s, self.node_s2i = self._vocabulary(node_labels, unknown=None)
        self.node_i2s.append(self.padding_token)
        self.node_s2i[self.padding_token] = len(self.node_s2i)

        # Global edge vocabulary
        edge_labels = {self.selfedge_label, self.noedge_label, self.obj2feat_edge, self.feat2obj_edge}
        self.edge_i2s, self.edge_s2i = self._vocabulary(edge_labels, unknown=None)

    # nontrivial_only: bool, indicates whether all subpredicates should be nontrivial
    # max_depth: int
    # Outputs a ndarray[Predicate]
    def generateAllPredicates(self, max_depth, nontrivial_only, no_negation, no_conjunction):
        depth2predicates = [] # list[list[Predicate]]
        depth2predicates.append([value for value in self.values if (not nontrivial_only or value.isNontrivial())]) # All predicates of depth 1
        while(len(depth2predicates) < max_depth):
            predicates = list() # list[Predicate]

            if(not no_negation):
                for predicate in depth2predicates[-1]:
                    pred = Negation(predicate=predicate)

                    if(pred.hasEquivalentIn(itertools.chain(*depth2predicates, predicates))): continue
                    #if(nontrivial_only and (not pred.isNontrivial())): continue

                    predicates.append(pred)
             
            if(not no_conjunction):
                for pred1 in depth2predicates[-1]:
                    for pred2 in itertools.chain.from_iterable(depth2predicates):
                        pred = Conjunction(pred1=pred1, pred2=pred2)

                        if(pred.hasEquivalentIn(itertools.chain(*depth2predicates, predicates))): continue
                        if(nontrivial_only and (not pred.isNontrivial())): continue

                        predicates.append(pred)

            depth2predicates.append(predicates)

        return np.array(list(itertools.chain.from_iterable(depth2predicates))) # ndarray[Predicate]

    def print_info(self):
        print(f"{len(self.properties)} properties:")
        #for prop in self.properties: print(f"{prop} (size {len(prop.values)})")
        for prop in self.properties: print(f"{prop} ({prop.values})")
        
        print(f"{len(self.predicates)} predicates ({self.predicates})")

    # Generates a batch.
    # Outputs a Batch.
    def get_batch(self, size=None, data_type='any', allow_indeterminate=None, **kwargs):
        """Generates a batch as a Batch object.
        size: int, the size of the batch.
        data_type: string ("train", "test" or "any"), indicates from what part the candidates are selected.

        Additional kwargs are accepted for compatibility with image-oriented data iterators but ignored here.
        """
        batch = []
        if(size is None): size = self.batch_size
        if(allow_indeterminate is None): allow_indeterminate = self.allow_indeterminate

        for _ in range(size):
            # Selects a predicate.
            pred_idx, predicate = self.selectPredicate()
            # Generates a candidate.
            candidate = self.generateCandidate(allow_indeterminate=allow_indeterminate) # Candidate
            batch.append((pred_idx, predicate, candidate))

        predicate_idx, pred_objects, candidates = zip(*batch) # Unzips the list of tuples (to a tuple of lists).
        # TODO In fact, it would be better to store in the batch tensors ready to be fed to the model.
        # So, the predicate indices instead of the predicates, and for the candidates, use graphTensorize here https://colab.research.google.com/drive/1C5iUSxX-MIJXIb4wfUzYBExRTF-OhsWn?usp=sharing
        # MG: The `store_tensor` flag triggers candidate dictionary translation into tensors
        return Batch(size=size, predicate=list(pred_objects), predicate_idx=list(predicate_idx), candidate=list(candidates))

    # Outputs a (int, Predicate).
    def selectPredicate(self):
        idx = np.random.randint(0, len(self.predicates))
        return (idx, self.predicates[idx])

    # allow_indeterminate: Bool
    # Outputs a Candidate.
    def generateCandidate(self, allow_indeterminate):
        prop2value = dict() # dict[Property, Value|NoneType]
        
        for prop in self.properties:
            if(allow_indeterminate and (np.random.rand() < (1 / (1 + len(prop.values))))): continue
            
            prop2value[prop] = np.random.choice(prop.values) # All values are equiprobable.
        
        return Candidate(prop2value)
    
    def _vocabulary(self, symbols, unknown='<unk>'):
        '''Given a set of strings, returns mappings: index2string and string2index.'''
        symbols = set(symbols)
        if unknown is not None:
            symbols.add(unknown)
        i2s = list(symbols)
        s2i = {s:i for (i,s) in enumerate(i2s)}
        return i2s, s2i

    def _tensorize_graph(self, graph, node_labels_sym2idx, edge_labels_sym2id, padding_length=None):
        '''
        Given a graph expressed as a list of dictionaries and a set of node and edge labels,
        returns node, edge indices and graph depth.
        '''
        # graph: list[dict[str, str]]
        graph_size = sum([(len(o) + 1) for o in graph])
        length = padding_length if padding_length is not None else graph_size

        node_idx = [node_labels_sym2idx[self.padding_token]] * length # list[int]
        edge_idx = [[edge_labels_sym2id[self.noedge_label]] * length for _ in range(length)] # list[list[int]]

        root_id = 0
        for o in graph:
            node_idx[root_id] = node_labels_sym2idx[self.object_token]
            edge_idx[root_id][root_id] = edge_labels_sym2id[self.selfedge_label]

            i = root_id + 1
            for feature, value in o.items():
                node_idx[i] = node_labels_sym2idx[(feature, value)]
                edge_idx[i][i] = edge_labels_sym2id[self.selfedge_label]

                edge_idx[root_id][i] = edge_labels_sym2id[self.obj2feat_edge]
                edge_idx[i][root_id] = edge_labels_sym2id[self.feat2obj_edge]
                i += 1

            root_id = i

        return node_idx, edge_idx, graph_size


def get_data_loader(args):
    dataset = Dataset(device=args.device, batch_size=args.batch_size, properties=args.properties, max_depth=args.max_depth, nontrivial_only=args.nontrivial_only, no_negation=args.no_negation, no_conjunction=args.no_conjunction)
    dataset.print_info()

    return dataset

if(__name__ == "__main__"):
    # Creates a dataset.
    dataset = Dataset(device='cpu', batch_size=128, properties="4-4", max_depth=3, nontrivial_only=False, no_negation=False, no_conjunction=False)
    print("\nDataset info: ")
    dataset.print_info()
    # Estimates the probability that a random candidate satisfy a random predicate.
    print("\nSatisfaction probability test (logical)")
    nb = 10_000
    for allow_indeterminate in [True, False]:
        counts = dict() # dict[int, int]
        for _ in range(nb):
            _, predicate = dataset.selectPredicate() # FYI BUG_FIX: `predicate` is a tuple (idx, predicate@idx): select only predicate[-1]
            candidate = dataset.generateCandidate(allow_indeterminate=allow_indeterminate)
            truth_value = predicate.check(candidate)
            counts[predicate.check(candidate)] = counts.get(predicate.check(candidate), 0) + 1
        
        print(f"Satisfaction probabilities (allow_indeterminate={allow_indeterminate}): ", end="")
        print({truth_value: (100 * c / nb) for (truth_value, c) in counts.items()})

    print("\nEncoding correctness test: ")
    batch = dataset.get_batch(size=256)
    
    # Sparse encoding should preserve indices
    sparse = batch.encode_predicates('sparse')
    assert np.all(sparse == np.array(batch.predicate_idx)), "Sparse encoding mismatch"
    print("Sparse encoding OK")
    
    # Tensorization test: component shapes must be consistent
    batch.tensorize(dataset)
    assert batch.node_idx is not None
    assert batch.edge_idx is not None
    assert batch.graph_sizes is not None
    assert len(batch.node_idx) == batch.size
    assert len(batch.edge_idx) == batch.size
    print("Graph tensorization OK")

    print("\nAll tests passed")

    print(batch)
