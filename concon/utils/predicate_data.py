import numpy as np
import torch
import itertools
import random

from multiprocessing import shared_memory

try:
    from .dataset import SeqAsyncDataset # When loading this file from outside of its directory.
except:
    from dataset import SeqAsyncDataset # When directly executing this file.

class Batch():
    def __init__(self, size, predicate, predicate_idx, candidate, node_idx, edge_idx, graph_sizes, candidate_truth=None):
        self.size = size # int, in number of instances

        #self.predicate = predicate # list[Predicate]
        self.predicate_idx = predicate_idx # list[int] or tensor of shape (batch)
        
        # RMK: The Candidate·s contain Property·s and Value·s (which refer to each other), and these don't get serialised easily (which is necessary for multiprocessing).
        #self.candidate = candidate # list[list[Candidate]]

        self.node_idx = node_idx # list[list[list[int]]] or tensor of shape (batch, candidate, node)
        self.edge_idx = edge_idx # list[list[list[list[int]]]] or tensor of shape (batch, candidate, node, node)
        self.graph_sizes = graph_sizes # list[list[int]] tensor of shape (batch, candidate), in number of nodes
        
        self.candidate_truth = candidate_truth # None or list[list[int]] or tensor of shape (batch, candidate), aligned with candidate

    # pin_memory: bool
    def tensorize(self, device=None, pin_memory=False):
        assert ((pin_memory == False) or (device is None) or (device == "cpu"))

        self.predicate_idx = torch.tensor(self.predicate_idx, device=device, dtype=torch.long, pin_memory=pin_memory)
        
        self.node_idx = torch.tensor(self.node_idx, device=device, dtype=torch.long, pin_memory=pin_memory)
        self.edge_idx = torch.tensor(self.edge_idx, device=device, dtype=torch.long, pin_memory=pin_memory)
        self.graph_sizes = torch.tensor(self.graph_sizes, device=device, dtype=torch.int, pin_memory=pin_memory)
        
        if(self.candidate_truth is not None): self.candidate_truth = torch.tensor(self.candidate_truth, device=device, dtype=torch.float32, pin_memory=pin_memory)

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
    
    def __str__(self):
        return f"Batch(size={self.size}, predicate={self.predicate}, predicate_idx={self.predicate_idx}, candidate={self.candidate}, candidate_truth={self.candidate_truth})"

    def pretty_print(self, dataset):
        """
        Decodes node indices for a batch item into human-readable labels.
        If candidate_idx is None, prints all candidates for the item.
        """
        def decode_row(d, row):
            return [d[i] for i in row]

        return [(
            [decode_row(dataset.graph_converter.node_i2s, row) for row in self.node_idx[item_idx]],
            [decode_row(dataset.edge_i2s, row) for mtx in self.edge_idx[item_idx] for row in mtx],
            self.graph_sizes[item_idx],
        ) for item_idx in range(self.size)]

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


# A predicate is equivalent to a propositional logic formula.
class Predicate():
    def __init__(self):
        self._build_cache = dict() # dict[int, list[Candidate]]
    
    # Computes the truth value in {-1, 0, 1} of the predicate applied on a given candidate based on Kleene logic. (-1 for false, 0 for unknown, 1 for true)
    # candidate: Candidate
    # Outputs an int.
    def check(self, candidate):
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

    # Outputs a bool.
    def isVerifiable(self):
        return (len(self.build(target=1)) > 0)

    # Outputs a bool.
    def isFalsifiable(self):
        return (len(self.build(target=-1)) > 0)

    # Outputs a bool.
    # TODO To be trivial is to always have the same truth value so this function should check that at least two truth values are available (this matters when considering also a third truth value).
    def isNontrivial(self):
        return self.isVerifiable() and self.isFalsifiable()

    # other: Predicate
    # consider_indeterminate: bool
    # Outputs a bool.
    def isAsStrongAs(self, other, consider_indeterminate):
        for c in self.build(target=1):
            if(other.check(c) < 1): return False

        if(not consider_indeterminate): return True

        for c in self.build(target=0):
            if(other.check(c) < 0): return False

        for c in other.build(target=0):
            if(self.check(c) > 0): return False
        
        for c in other.build(target=-1):
            if(self.check(c) > -1): return False

        return True
    
    # other: Predicate
    # consider_indeterminate: bool
    # Outputs a bool.
    def isEquivalentTo(self, other, consider_indeterminate):
        return self.isAsStrongAs(other, consider_indeterminate) and other.isAsStrongAs(self, consider_indeterminate)

    # others: iterable[Predicate]
    # consider_indeterminate: bool
    # Outputs a bool.
    def hasEquivalentIn(self, others, consider_indeterminate):
        for other in others:
            if(self.isEquivalentTo(other, consider_indeterminate)):
                #print(f"{self} is equivalent with {other}") # DEBUG
                return True

        return False

    def __repr__(self):
        return str(self)

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
        assert False, f"Unknown target value ({target})."

    def __str__(self): return self.name

    # A value predicate is always verifiable: assign this value to its property.
    def isVerifiable(self):
        return True

    # It is falsifiable iff the property has at least one different value.
    def isFalsifiable(self):
        return (len(self.prop.values) > 1)

    # O(1) exact nontriviality for atomic predicates.
    def isNontrivial(self):
        return (len(self.prop.values) > 1)

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

    # Negation swaps verifiable/falsifiable and preserves nontriviality.
    def isVerifiable(self):
        return self.predicate.isFalsifiable()

    def isFalsifiable(self):
        return self.predicate.isVerifiable()

    def isNontrivial(self):
        return self.predicate.isNontrivial()

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
        
        assert False, f"Unknown target value ({target})."

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


class FailureBasedDistribution:
    # nb_predicates: int
    # momentum_factor: float (∈ [0,1[)
    # smoothing_factor: float (> 0)
    # from_shared_memory: [int, tuple[str]] | None
    def __init__(self, nb_predicates, momentum_factor=0.99, smoothing_factor=1.0, from_shared_memory=None):
        self.momentum_factor = momentum_factor

        self.shared_memory = [0, ()] # RMK: A list and not a tuple because I want a mutable type.

        if(from_shared_memory is None):
            # Initialisation with smoothing
            self.counts_vector = np.full((nb_predicates,), smoothing_factor)
            self.failure_vector = self.counts_vector / 2
        else:
            _, names = from_shared_memory

            self.counts_vector_shm = shared_memory.SharedMemory(name=names[0])
            self.counts_vector = np.ndarray((nb_predicates,), dtype=np.float64, buffer=self.counts_vector_shm.buf)
            
            self.failure_vector_shm = shared_memory.SharedMemory(name=names[1])
            self.failure_vector = np.ndarray((nb_predicates,), dtype=np.float64, buffer=self.failure_vector_shm.buf)

            self.shared_memory[0] = -1
            self.shared_memory[1] = names

        self.info = {"nb_predicates": nb_predicates, "momentum_factor": momentum_factor, "smoothing_factor": smoothing_factor, "from_shared_memory": self.shared_memory}

    def to_shared_memory(self):
        status, names = self.shared_memory
        assert (status == 0)

        self.counts_vector_shm = shared_memory.SharedMemory(create=True, size=self.counts_vector.size * self.counts_vector[0].nbytes)
        counts_vector = np.ndarray(self.counts_vector.shape, dtype=np.float64, buffer=self.counts_vector_shm.buf)
        counts_vector[:] = self.counts_vector
        self.counts_vector = counts_vector

        self.failure_vector_shm = shared_memory.SharedMemory(create=True, size=self.failure_vector.size * self.failure_vector[0].nbytes)
        failure_vector = np.ndarray(self.failure_vector.shape, dtype=np.float64, buffer=self.failure_vector_shm.buf)
        failure_vector[:] = self.failure_vector
        self.failure_vector = failure_vector

        self.shared_memory[0] = 1
        self.shared_memory[1] = (self.counts_vector_shm.name, self.failure_vector_shm.name)

    # Might be important to avoid memory leaks.
    def close(self):
        status, names = self.shared_memory
        if(status == -1):
            self.counts_vector_shm.close()
            self.failure_vector_shm.close()
        elif(status == 1):
            self.counts_vector_shm.close()
            self.counts_vector_shm.unlink()
            self.failure_vector_shm.close()
            self.failure_vector_shm.unlink()

    # predicate_idx: np.array[int]
    # failure: np.array[float]
    # new_epoch: boolean
    def update(self, predicate_idx, failure, new_epoch):
        if(new_epoch):
            self.counts_vector *= self.momentum_factor
            self.failure_vector *= self.momentum_factor
        
        ## Note: if the same predicate appeared multiple time, the momentum factor would still be applied only once
        #self.counts_vector[predicate_idx] *= self.momentum_factor
        #self.failure_vector[predicate_idx] *= self.momentum_factor
        
        np.add.at(self.counts_vector, predicate_idx, 1.0)
        np.add.at(self.failure_vector, predicate_idx, failure)

    # Returns a probability distribution over (a subset of the) predicate indices.
    # allowed_predicates_idx: np.array[int] | None
    def distribution(self, allowed_predicates_idx=None):
        if(allowed_predicates_idx is None): unnormalised_dist = (self.failure_vector / self.counts_vector)
        else: unnormalised_dist = (self.failure_vector[allowed_predicates_idx] / self.counts_vector[allowed_predicates_idx])

        #print(self.failure_vector) # DEBUG
        #print(self.counts_vector) # DEBUG

        return (unnormalised_dist / np.linalg.norm(unnormalised_dist, 1))

    # Returns predicate indices.
    # nb: int
    # allowed_predicates_idx: np.array[int] | None
    def sample(self, nb, allowed_predicates_idx=None):
        dist = self.distribution(allowed_predicates_idx)
        
        #print(dist) # DEBUG

        if(allowed_predicates_idx is None): allowed_predicates_idx = range(dist.shape[0])

        return np.random.choice(a=allowed_predicates_idx, size=nb, p=dist)

class GraphConverter:
    object_token='<obj>'
    padding_token='<pad>'
    selfedge_label='<self>'
    noedge_label='<noedge>'
    obj2feat_edge='obj2feat'
    feat2obj_edge='feat2obj'

    def __init__(self, node_i2s, node_s2i, edge_i2s, edge_s2i):
        self.node_i2s = node_i2s
        self.node_s2i = node_s2i
        self.edge_i2s = edge_i2s
        self.edge_s2i = edge_s2i
    
    # Turns a graph into a triplet (node indices, edge indices, size).
    # graph: list[dict[str, str]]
    # padding_length: int|None
    # Ouputs an (list[int], list[list[int]], int).
    def convertGraph(self, graph, padding_length=None):
        graph_size = sum([(len(o) + 1) for o in graph])
        length = padding_length if(padding_length is not None) else graph_size

        node_idx = [self.node_s2i[self.padding_token]] * length # list[int]
        edge_idx = [[self.edge_s2i[self.noedge_label]] * length for _ in range(length)] # list[list[int]]

        root_id = 0
        for o in graph:
            node_idx[root_id] = self.node_s2i[self.object_token]
            edge_idx[root_id][root_id] = self.edge_s2i[self.selfedge_label]

            i = root_id + 1
            for feature, value in o.items():
                node_idx[i] = self.node_s2i[(feature, value)]
                edge_idx[i][i] = self.edge_s2i[self.selfedge_label]

                edge_idx[root_id][i] = self.edge_s2i[self.obj2feat_edge]
                edge_idx[i][root_id] = self.edge_s2i[self.feat2obj_edge]
                i += 1

            root_id = i

        return (node_idx, edge_idx, graph_size)
    
    # candidates: list[list[Candidate]]
    def convert(self, candidates):
        # A candidate is an object that is turned into a dictionary (e.g. {"P0": "P0-v2", "P1", "P1-v0"}). This dictionary is wrapped into a list because in general a single input could consists of multiple objects; here we have a single object so a list of size 1.
        batch = [
            [
                [
                    {p.name: v.name for p, v in candidate.prop2value.items() if v is not None}
                ]
                for candidate in candidate_list
            ] 
            for candidate_list in candidates
        ] # list[list[list[dict[str, str]]]]; dimensions are (instances, candidates, objects)

        node_idx = []    # (batch, num_candidates, max_nodes)
        edge_idx = []    # (batch, num_candidates, max_nodes, max_nodes)
        graph_sizes = [] # (batch, num_candidates), in number of nodes

        padding_length = max(sum((len(o) + 1) for o in graph) for item in batch for graph in item) if(batch != []) else 0 # largest total node count across all graphs

        for item in batch:
            item_nodes, item_edges, item_sizes = zip(*[self.convertGraph(graph, padding_length) for graph in item])
            node_idx.append(item_nodes)
            edge_idx.append(item_edges)
            graph_sizes.append(item_sizes)

        # RMK: The data is not yet converted in PyTorch tensors due to efficiency reasons (tensors seem slow to serialise and thus do not work so well with multiprocessing).
        #node_idx = torch.tensor(node_idx, dtype=torch.long, pin_memory=pin_memory)
        #edge_idx = torch.tensor(edge_idx, dtype=torch.long, pin_memory=pin_memory)
        #graph_sizes = torch.tensor(graph_sizes, dtype=torch.long, pin_memory=pin_memory)

        return (node_idx, edge_idx, graph_sizes)


# A dataset contains properties and (property) values, but also mappings (to indices) used to tensories objects.
class Dataset(SeqAsyncDataset):
    def __init__(self, device='cpu', batch_size=128, properties="3-4", max_depth=2, min_depth=1, num_candidates=2, predicate_sampling='random', candidate_sampling='random', nontrivial_only=False, no_negation=False, no_conjunction=False, allow_indeterminate=False, overfit=False):
        self.device = device
        self.batch_size = batch_size
        self.allow_indeterminate = allow_indeterminate
        self.num_candidates = num_candidates
        self.predicate_sampling = predicate_sampling
        self.candidate_sampling = candidate_sampling
    
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
        assert (nontrivial_only or candidate_sampling != "balanced"), "It is impossible to garantee the existence of both positive and negative candidates if there exist trivial predicates."
        self.predicates = self.generateAllPredicates(max_depth=max_depth, min_depth=min_depth, nontrivial_only=nontrivial_only, no_negation=no_negation, no_conjunction=no_conjunction, consider_indeterminate=allow_indeterminate) # ndarray[Predicate]
        # naming convention inconsistent internally but compatible with modules; TIMOTHÉE: be more specific
        self.nb_categories = len(self.predicates) 
        # Predicate-selection mask used by get_batch. True means "can be sampled".
        self.predicate_sampling_mask = np.ones((self.nb_categories,), dtype=bool)
        # Positive predicates are predicates that do not contain any negation node.
        self.positive_predicate_mask = np.array([not self._has_negation(pred) for pred in self.predicates], dtype=bool)

        # Builds a global node vocabulary.
        node_labels = {GraphConverter.object_token} # set[str | (str, str)]
        for prop in self.properties:
            for value in prop.values:
                node_labels.add((prop.name, value.name))

        node_i2s, node_s2i = self._vocabulary(node_labels, unknown=None)
        node_i2s.append(GraphConverter.padding_token) # TIMOTHÉE: why is padding_token not handled the same way object_token is?
        node_s2i[GraphConverter.padding_token] = len(node_s2i)

        # Builds a global edge vocabulary.
        edge_labels = {GraphConverter.selfedge_label, GraphConverter.noedge_label, GraphConverter.obj2feat_edge, GraphConverter.feat2obj_edge} # set[str]
        edge_i2s, edge_s2i = self._vocabulary(edge_labels, unknown=None)
        
        self.graph_converter = GraphConverter(node_i2s=node_i2s, node_s2i=node_s2i, edge_i2s=edge_i2s, edge_s2i=edge_s2i)

        # Builds (or not) a pool of batches used in overfitting regime.
        self.overfit_pool = self.init_overfit_pool() if(overfit) else None # list[(int, Predicate, list[Candidate], list[int])]|None
        
        # A momentum factor of (1 - 1/k) (e.g. 0.99) corresponds to each cell of the failure vector containing a statistics over k (e.g. 100) epochs.
        self.failure_based_distribution = FailureBasedDistribution(len(self.predicates), momentum_factor=0.9, smoothing_factor=8.0)
        
        super().__init__(overfit_pool=self.overfit_pool, predicates=self.predicates, properties=self.properties, graph_converter=self.graph_converter, failure_based_distribution_info=self.failure_based_distribution.info);
        #super().__init__();
   
    def turnAsynchronous(self, *args, **kwargs):
        self.failure_based_distribution.to_shared_memory()
        
        super(Dataset, self).turnAsynchronous(*args, **kwargs) 

    def _close(self):
        self.failure_based_distribution.close()

    # failure_based_distribution_shared_memory: [1, (str, str)]
    @staticmethod
    def _additional_resources(failure_based_distribution_info, **kwargs): 
        return {"failure_based_distribution": FailureBasedDistribution(**failure_based_distribution_info)}

    # Builds a fixed pool of instances. (Used for overfitting tests.)
    def init_overfit_pool(self, size=100):
        instances = [] # list[(int, Predicate, list[Candidate], list[int])]
        for pred_idx in random.choices(range(len(self.predicates)), k=size):
            predicate = self.predicates[pred_idx]

            if(self.candidate_sampling == 'balanced'):
                candidates, truths = self.generateCandidatesBalanced(predicate, self.num_candidates, self.allow_indeterminate)
            else:
                assert (self.candidate_sampling == 'random')

                candidates, truths = self.generateCandidates(predicate, self.num_candidates, self.allow_indeterminate)

            instances.append((pred_idx, predicate, candidates, truths))

        return instances

    # TODO Find ways to optimise this function.
    # nontrivial_only: bool, indicates whether all subpredicates should be nontrivial
    # max_depth: int (a single node is of depth one)
    # min_depth: int
    # consider_indeterminate: bool
    # Outputs a ndarray[Predicate]
    def generateAllPredicates(self, max_depth, min_depth, nontrivial_only, no_negation, no_conjunction, consider_indeterminate):
        # Computes an upper-bound of the number of predicates (ignoring equivalence and non-trivial pruning).
        est_by_depth = []
        est_by_depth.append(len(self.values))
        for d in range(1, max_depth):
            prev = est_by_depth[-1]
            total_prev = sum(est_by_depth)
            neg = prev if not no_negation else 0
            conj = prev * total_prev if not no_conjunction else 0
            est_by_depth.append(neg + conj)
        print(f"Upper-bound of the number of predicates by depth: {est_by_depth}; total: {sum(est_by_depth)}")

        depth2predicates = [] # list[list[Predicate]]

        print(f"Generating depth={len(depth2predicates)+1} predicates…")
        depth2predicates.append([value for value in self.values if(not nontrivial_only or value.isNontrivial())]) # All predicates of depth 1
        print(f"Depth: {len(depth2predicates)}: {len(depth2predicates[-1])} predicates")

        # In order to not include in the dataset two (syntactically) distinct predicates logically equivalent to each other, the equivalence of each new predicate with already generated predicates is checked. To speed up this process, a list of objects is used to compute a "logical signature" for each predicate. (Then, only the equivalence of predicates with the same signature is directly checked.) The signature of a predicate is the list of truth values of the predicate on a list of objects.
        # Computes the list of objects used to compute signatures.
        signature_items = [] # list[Candidate]
        for value in self.values:
            l = value.build(1) # list[Candidate]
            assert (len(l) == 1)
            signature_items.append(self.extendCandidate(candidate=l[0], allow_indeterminate=consider_indeterminate))

        # pred: Predicate
        # Outputs a tuple[int].
        def computeSignature(pred):
            return tuple(pred.check(item) for item in signature_items)

        signatures = {} # dict[tuple[int], Set[Predicate]]

        # pred: Predicate
        # Outputs a bool.
        def checkEquivalence(pred):
            signature = computeSignature(pred)
            if((signature in signatures) and (pred.hasEquivalentIn(signatures[signature], consider_indeterminate))): return False
            
            signatures[signature] = signatures.get(signature, set())
            signatures[signature].add(pred)

            return True
        
        for pred in depth2predicates[0]: assert checkEquivalence(pred)


        while(len(depth2predicates) < max_depth):
            print(f"Generating depth={len(depth2predicates)+1} predicates…")
            predicates = list() # list[Predicate]

            if(not no_negation):
                for predicate in depth2predicates[-1]:
                    pred = Negation(predicate=predicate)

                    #if(nontrivial_only and (not pred.isNontrivial())): continue

                    #if(pred.hasEquivalentIn(itertools.chain(*depth2predicates, predicates), consider_indeterminate)): continue
                    if(not checkEquivalence(pred)): continue

                    predicates.append(pred)
             
            if(not no_conjunction):
                for pred1 in depth2predicates[-1]:
                    for pred2 in itertools.chain.from_iterable(depth2predicates):
                        if(pred1 == pred2): break

                        pred = Conjunction(pred1=pred1, pred2=pred2)
                        
                        if(nontrivial_only and (not pred.isNontrivial())): continue

                        #if(pred.hasEquivalentIn(itertools.chain(*depth2predicates, predicates), consider_indeterminate)): continue
                        if(not checkEquivalence(pred)): continue

                        predicates.append(pred)

            depth2predicates.append(predicates)
            print(f"Depth: {len(depth2predicates)}: {len(predicates)} predicates")

        return np.array(list(itertools.chain.from_iterable(depth2predicates[min_depth-1:]))) # ndarray[Predicate]
        
    def print_info(self):
        print(f"{len(self.properties)} properties:")
        #for prop in self.properties: print(f"{prop} (size {len(prop.values)})")
        for prop in self.properties: print(f"{prop} ({prop.values})")
        
        print(f"{len(self.predicates)} predicates ({self.predicates})")

        print(f"device: {self.device}")
        print(f"batch_size: {self.batch_size}")
        print(f"allow_indeterminate: {self.allow_indeterminate}")
        print(f"num_candidates: {self.num_candidates}")
        print(f"predicate_sampling: {self.predicate_sampling}")
        print(f"candidate_sampling: {self.candidate_sampling}")

    @staticmethod
    def _has_negation(predicate):
        if isinstance(predicate, Negation):
            return True
        if isinstance(predicate, Conjunction):
            return Dataset._has_negation(predicate.pred1) or Dataset._has_negation(predicate.pred2)
        return False

    def set_predicate_sampling_mask(self, mask):
        mask = np.asarray(mask, dtype=bool)
        assert mask.shape == (self.nb_categories,), f"Mask shape {mask.shape} does not match number of predicates ({self.nb_categories})."
        assert np.any(mask), "Predicate sampling mask cannot be empty."
        self.predicate_sampling_mask = mask.copy()

    def use_positive_predicates_only(self):
        self.set_predicate_sampling_mask(self.positive_predicate_mask)

    def use_all_predicates(self):
        self.predicate_sampling_mask = np.ones((self.nb_categories,), dtype=bool)
   
    # symbols: collection[str]
    # unknown: str
    # Outputs a (list[str], dict[int, str]).
    def _vocabulary(self, symbols, unknown='<unk>'):
        '''Given a set of strings, returns mappings: index2string and string2index.'''
        symbols = set(symbols)
        if(unknown is not None): symbols.add(unknown)

        i2s = list(symbols) # list[str]
        s2i = {s: i for (i, s) in enumerate(i2s)} # dict[int, str]
        
        return (i2s, s2i)

    # Generates a batch.
    # Outputs a Batch with candidate list(s) and aligned truth labels.
    def get_batch(self, size=None, data_type='any', allow_indeterminate=None, num_candidates=None, predicate_sampling=None, candidate_sampling=None, device=None, pin_memory=False, **kwargs):
        """Generates a batch as a Batch object.
        size: int, the size of the batch.
        data_type: string ("train", "test" or "any"), indicates from what part the candidates are selected.
        Additional kwargs are accepted for compatibility with image data iterators but ignored here.
        """
        if(size is None): size = self.batch_size
        if(allow_indeterminate is None): allow_indeterminate = self.allow_indeterminate
        if(num_candidates is None): num_candidates = self.num_candidates
        if(predicate_sampling is None): predicate_sampling = self.predicate_sampling
        if(candidate_sampling is None): candidate_sampling = self.candidate_sampling

        allowed_predicates_idx = kwargs.pop('allowed_predicates_idx', None)
        if allowed_predicates_idx is None:
            if not np.all(self.predicate_sampling_mask):
                allowed_predicates_idx = tuple(np.flatnonzero(self.predicate_sampling_mask).tolist())
        elif isinstance(allowed_predicates_idx, np.ndarray):
            allowed_predicates_idx = tuple(allowed_predicates_idx.tolist())
        elif isinstance(allowed_predicates_idx, list):
            allowed_predicates_idx = tuple(allowed_predicates_idx)

        batch = self._get_batch(size=size, data_type=data_type, allow_indeterminate=allow_indeterminate, num_candidates=num_candidates, predicate_sampling=predicate_sampling, candidate_sampling=candidate_sampling, allowed_predicates_idx=allowed_predicates_idx, **kwargs);

        batch.predicate = [self.predicates[pred_idx] for pred_idx in batch.predicate_idx]
        
        if(device is None): device = self.device
        batch.tensorize(device=device, pin_memory=pin_memory)

        return batch
        #return self._get_batch(
        #    size, data_type, allow_indeterminate, num_candidates, candidate_sampling, # request arguments
        #    self.overfit_pool, self.predicates, self.properties # resource arguments
        #)

    @staticmethod
    def _generate_batch(
        size, data_type, allow_indeterminate, num_candidates, predicate_sampling, candidate_sampling, # request arguments
        overfit_pool, predicates, properties, graph_converter, failure_based_distribution, # resource arguments
        **kwargs
    ):
        allowed_predicates_idx = kwargs.get('allowed_predicates_idx', None)
        batch = []
        if(overfit_pool is not None): # Specific procedure for overfitting mode.
            for _ in range(size):
                pred_idx, predicate, candidates, truths = random.choice(overfit_pool)
                batch.append((pred_idx, predicate, list(candidates), list(truths))) # TIMOTHÉE Why are the two lists copied (with `list`)?
        else:
            # Samples `size` predicates.
            for (pred_idx, predicate) in zip(*Dataset._selectPredicates(size, predicates, predicate_sampling, failure_based_distribution, allowed_predicates_idx=allowed_predicates_idx)):
                # Samples `num_candidates` candidates.
                if(candidate_sampling == 'balanced'):
                    candidates, truths = Dataset._generateCandidatesBalanced(predicate, num_candidates, allow_indeterminate, properties)
                elif(candidate_sampling == 'random'):
                    candidates, truths = Dataset._generateCandidates(predicate, num_candidates, allow_indeterminate, properties)
                else:
                    raise ValueError(f"Candidate sampling strategy unknown: {candidate_sampling}.")

                batch.append((pred_idx, predicate, candidates, truths))

        predicate_idx, predicate, candidates, truths = zip(*batch)

        node_idx, edge_idx, graph_sizes = graph_converter.convert(candidates)

        return Batch(size=size, predicate=predicate, predicate_idx=predicate_idx, candidate=candidates, node_idx=node_idx, edge_idx=edge_idx, graph_sizes=graph_sizes, candidate_truth=truths)

    # nb: int
    # predicate_sampling: str
    # Outputs a (np.array[int], np.array[Predicate]).
    def selectPredicates(self, nb, predicate_sampling, allowed_predicates_idx=None):
        return self._selectPredicates(nb, self.predicates, predicate_sampling, self.failure_based_distribution, allowed_predicates_idx=allowed_predicates_idx)

    # nb: int
    # predicates: list[Predicate]
    # predicate_sampling: str
    # failure_based_distribution: FailureBasedDistribution
    # Outputs a (np.array[int], np.array[Predicate]).
    @staticmethod
    def _selectPredicates(nb, predicates, predicate_sampling, failure_based_distribution, allowed_predicates_idx=None):
        if allowed_predicates_idx is None:
            allowed_predicates_idx = np.arange(len(predicates))
        else:
            allowed_predicates_idx = np.asarray(allowed_predicates_idx, dtype=np.int64)
        if allowed_predicates_idx.size == 0:
            raise ValueError("No predicates are available for sampling.")

        if(predicate_sampling == "random"):
            indices = np.random.choice(allowed_predicates_idx, size=(nb,), replace=True)
        elif(predicate_sampling == "difficulty"):
            indices = failure_based_distribution.sample(nb, allowed_predicates_idx=allowed_predicates_idx)
        else:
            raise ValueError(f"Predicate sampling strategy unknown: {predicate_sampling}.")
        
        return (indices, predicates[indices])

    # candidate: Candidate
    # allow_indeterminate
    # Outputs a Candidate.
    def extendCandidate(self, candidate, allow_indeterminate):
        return self._extendCandidate(candidate, allow_indeterminate, self.properties)

    # properties: list[Property]
    @staticmethod
    def _extendCandidate(candidate, allow_indeterminate, properties):
        prop2value = dict(candidate.prop2value) # dict[Property, Value|NoneType]

        for prop in properties:
            if(prop in prop2value): continue
            if(allow_indeterminate and (np.random.rand() < (1 / (1 + len(prop.values))))): continue
            
            prop2value[prop] = np.random.choice(prop.values) # All values are equiprobable.
        
        return Candidate(prop2value)

    # Generates `n` candidates satisfying (`target`=1) or falsifying (`target`=-1) `predicate`.
    # predicate: Predicate
    # target: int
    # n: int
    # allow_indeterminate: bool
    # Outputs a list[Candidate].
    def generateCandidatesTarget(self, predicate, target, n, allow_indeterminate):
        return self._generateCandidatesTarget(predicate, target, n, allow_indeterminate, self.properties)

    # properties: list[Property]
    @staticmethod
    def _generateCandidatesTarget(predicate, target, n, allow_indeterminate, properties):
        candidates = [] # list[Candidate]

        base_candidates = predicate.build(target=target) # list[Candidate]
        for _ in range(n):
            base_candidate = random.choice(base_candidates)
            candidate = Dataset._extendCandidate(base_candidate, allow_indeterminate, properties)
            
            candidates.append(candidate)

        return candidates

    # allow_indeterminate: bool
    # Outputs a Candidate.
    def generateCandidate(self, allow_indeterminate):
        return self._generateCandidate(allow_indeterminate, self.properties)

    # properties: list[Property]
    @staticmethod
    def _generateCandidate(allow_indeterminate, properties):
        prop2value = dict() # dict[Property, Value|NoneType]
        
        for prop in properties:
            if(allow_indeterminate and (np.random.rand() < (1 / (1 + len(prop.values))))): continue
            
            prop2value[prop] = np.random.choice(prop.values) # All values are equiprobable.
        
        return Candidate(prop2value)
    
    # Samples candidates balacing satisfaction given a predicate. (This ensures that the performance of the random baseline is 0.5.)
    # predicate: Predicate
    # num_candidates: int
    # allow_indeterminate: bool
    # Outputs a (list[Candidate], list[int]).
    def generateCandidatesBalanced(self, predicate, num_candidates, allow_indeterminate):
        return self._generateCandidatesBalanced(predicate, num_candidates, allow_indeterminate, self.properties)

    # properties: list[Property]
    @staticmethod
    def _generateCandidatesBalanced(predicate, num_candidates, allow_indeterminate, properties):
        assert (num_candidates % 2 == 0), f"It is impossible to balance an odd number ({num_candidates}) of candidates."
        num_true = num_candidates // 2
        num_false = num_candidates // 2 #num_candidates - num_true

        candidates = [] # list[Candidate]
        truths = [] # list[int]

        candidates.extend(Dataset._generateCandidatesTarget(predicate, 1, num_true, allow_indeterminate, properties))
        truths.extend([1] * num_true)

        candidates.extend(Dataset._generateCandidatesTarget(predicate, -1, num_false, allow_indeterminate, properties))
        truths.extend([0] * num_false)

        return (candidates, truths)

    # num_candidates: int
    # predicate: Predicate
    # allow_indeterminate: bool
    # Outputs a (list[Candidate], list[int]).
    def generateCandidates(self, predicate, num_candidates, allow_indeterminate):
        return self._generateCandidates(predicate, num_candidates, allow_indeterminate, self.properties)

    # properties: list[Property]
    @staticmethod
    def _generateCandidates(predicate, num_candidates, allow_indeterminate, properties):
        candidates = [] # list[Candidate]
        truths = [] # list[int]
        for _ in range(num_candidates):
            candidate = Dataset._generateCandidate(allow_indeterminate=allow_indeterminate, properties=properties)
            candidates.append(candidate)
            truths.append(1 if predicate.check(candidate) == 1 else 0)

        return (candidates, truths)


def get_data_loader(args):
    dataset = Dataset(device=args.device, batch_size=args.batch_size, properties=args.properties, max_depth=args.max_depth, min_depth=args.min_depth, nontrivial_only=args.nontrivial_only, no_negation=args.no_negation, no_conjunction=args.no_conjunction, allow_indeterminate=args.allow_indeterminate, num_candidates=args.num_candidates, predicate_sampling=args.predicate_sampling, candidate_sampling=args.candidate_sampling, overfit=args.overfit)
    dataset.print_info()

    return dataset

if(__name__ == "__main__"):
    import time

    # Creates a dataset.
    t1 = time.time()
    dataset = Dataset(device='cpu', batch_size=128, properties="1024", max_depth=2, nontrivial_only=True, no_negation=False, no_conjunction=True)
    #dataset = Dataset(device='cpu', batch_size=128, properties="2-3", max_depth=3, nontrivial_only=False, no_negation=False, no_conjunction=False)
    #dataset = Dataset(device='cpu', batch_size=128, properties="2-3", max_depth=3, nontrivial_only=False, no_negation=False, no_conjunction=False, allow_indeterminate=True)
    #dataset = Dataset(device='cpu', batch_size=128, properties="5", max_depth=4, nontrivial_only=False, no_negation=False, no_conjunction=False)
    t2 = time.time()
    print("\nDataset info: ")
    dataset.print_info()
    print(f"(generation took {t2-t1}s)")

    # Estimates the probability that a random candidate satisfy a random predicate.
    print("\nSatisfaction probability test (logical)")
    nb = 10_000
    for allow_indeterminate in [True, False]:
        counts = dict() # dict[int, int]
        for _, predicate in zip(*dataset.selectPredicate(nb, predicate_sampling='random')):
            candidate = dataset.generateCandidate(allow_indeterminate=allow_indeterminate)
            truth_value = predicate.check(candidate)
            counts[predicate.check(candidate)] = counts.get(predicate.check(candidate), 0) + 1
        
        print(f"Satisfaction probabilities (allow_indeterminate={allow_indeterminate}): ", end="")
        print({truth_value: (100 * c / nb) for (truth_value, c) in counts.items()})

    print("\nEncoding correctness test: ")
    batch = dataset.get_batch(size=32)
    
    # Conversion test: component shapes must be consistent
    assert len(batch.node_idx) == batch.size
    assert len(batch.edge_idx) == batch.size
    print("Graph tensorization OK")

    print("\nAll tests passed")

    print("\nBatch used:")
    print(batch)
    
    # Asynchronous
    print("\nSwitching to asynchronous mode.")
    dataset.turnAsynchronous(nb_workers=2, nb_prefetch=2);
    batch = dataset.get_batch(size=32)
    print("\nBatch:")
    print(batch)

    dataset.close()
