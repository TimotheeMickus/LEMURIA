from abc import ABCMeta, abstractmethod
import torch, torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import PIL
import numpy as np
import scipy
import scipy.special as scispe

import tqdm

import os
from collections import namedtuple, defaultdict
import itertools
import random
from deprecated import deprecated

from .misc import add_normal_noise, show_imgs, combine_images

class Batch():
    def __init__(self, size, original, target, base_distractors):
       self.size = size
       self.original = original # List of InputDataPoint·s
       self.target = target # List of InputDataPoint·s
       self.base_distractors = base_distractors # List of lists of InputDataPoint·s

    def original_img(self, stack=False, f=None):
        if(f is None): f = (lambda x: x)

        tmp = [f(x.img) for x in self.original]

        if(stack): return torch.stack(tmp)
        else: return tmp

    def target_img(self, stack=False, f=None):
        if(f is None): f = (lambda x: x)

        tmp = [f(x.img) for x in self.target]

        if(stack): return torch.stack(tmp)
        else: return tmp

    # Returns a list or a tensor of the original/target categories of the batch, possibly transformed by a function first.
    # stack: whether to return a tensor (True) or a list (False)
    # f: the function (if any) to apply to each category
    def target_category(self, stack=False, f=None):
        if(f is None): f = (lambda x: x)

        tmp = [f(x.category) for x in self.target]

        if(stack): return torch.tensor(tmp)
        else: return tmp

    # TODO If there is no base distractor, then the function might fail (when asking to stack an empty list)
    def base_distractors_img(self, flat=False, stack=False, f=None):
        if(f is None): f = (lambda x: x)

        if(not flat):
            tmp = [[f(x.img) for x in base_distractor] for base_distractor in self.base_distractors]
            if(stack): tmp = list(map(torch.stack, tmp))
        else: tmp = [f(x.img) for base_distractor in self.base_distractors for x in base_distractor]

        if(stack): return torch.stack(tmp)
        else: return tmp

    def get_images(self, original=True, target=True, base_distractors=True):
        images = []
        if(original): images.extend(self.original_img())
        if(target): images.extend(self.target_img())
        if(base_distractors): images.extend(self.base_distractors_img(flat=True))

        return images

    def require_grad(self, for_original=True, for_target=True, for_base_distractors=True):
        for img in self.get_images(original=for_original, target=for_target, base_distractors=for_base_distractors):
            img.requires_grad = True

class InputDataPoint():
    def __init__(self, img, category=None):
        self.img = img
        self.category = category

    def copy(self):
        return InputDataPoint(self.img, self.category)

    # Adds noise if necessary (normal random noise + clamping)
    def add_normal_noise_(self, noise):
        if(noise <= 0.0): return

        self.img = add_normal_noise(self.img, std_dev=noise, clamp_values=(0.0, 1.0))
    
    def add_normal_noise(self, noise):
        img = self.img if(noise <= 0.0) else add_normal_noise(self.img, std_dev=noise, clamp_values=(0.0, 1.0))
        
        return InputDataPoint(img, self.category)

class DataPoint():
    def __init__(self, idx, category, img):
        self.idx = idx
        self.category = category
        self.img = img

    def toInput(self, keep_category=False, device=None):
        img = self.img if(device is None) else self.img.to(device)
        category = self.category if keep_category else None

        return InputDataPoint(img, category)

class FailureBasedDistribution():
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
    def __init__(self, same_img, device, noise, batch_size, sampling_strategies):
        self.same_img = same_img # Whether Bob sees Alice's image or another one (of the same category); but any noise will be applied independently
        self.device = device
        self.noise = noise
        self.batch_size = batch_size
        self.sampling_strategies = sampling_strategies

        # Other properties that will be needed: self.nb_concepts, self.evaluation_categories, self.training_categories_idx, failure_based_distribution

    @abstractmethod
    def category_tuple(self, category_idx):
        pass

    @abstractmethod
    def category_idx(self, category_tuple):
        pass

    @abstractmethod
    def average_image(self):
        pass
    
    # None for an infinite dataset?
    @abstractmethod
    def size(self, data_type='any', no_evaluation=True):
        pass
    
    # None for an infinite dataset?
    #def __len__(self):
    #    pass
    
    # Should only be used for debugging purpose. Use `get_batch` instead
    @abstractmethod
    def get_datapoint(self, i):
        pass
    
    # Data type is for train data or test data
    @abstractmethod
    def category_to_datapoint(self, category_tuple, data_type):
        pass

    def print_info(self):
        for i, name in enumerate(self.concept_names): print('%s: %s' % (name, self.concepts[i]))
        
        print('Total number of categories: %i' % self.nb_categories)
        print('Training categories: %s' % sorted(self.training_categories))
        print('Evaluation categories: %s' % sorted(self.evaluation_categories))
        #print('(reference category: %s)' % ref_category)

        print('Size (total): %i' % self.size(data_type='any', no_evaluation=False))

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

    def _distance_to_categories(self, category, distance, no_evaluation):
        """Returns the list of all categories at distance `distance` from category `category`.
        If `no_evaluation` is True, evaluation categories are ignored."""
        categories = []
        for changed_dim in itertools.combinations(range(self.nb_concepts), distance): # Iterates over all subsets of [|0, `self.nb_concepts`|[ of size `distance`.
            changed_dim = set(changed_dim)
            new_category = tuple(e if(i not in changed_dim) else (not e) for i,e in enumerate(category)) # A category at distance `distance` from `category`.

            if(no_evaluation and (new_category in self.evaluation_categories)): continue
            categories.append(new_category)

        return categories

    def _distance_to_category(self, category, distance, no_evaluation):
        """Returns a category that is at distance `distance` from category `category`.
        If `no_evaluation` is True, evaluation categories are ignored."""
        categories = self._distance_to_categories(category, distance, no_evaluation)
        assert (categories != []), f"There is no category at distance {distance} from category {category} (with{'out considering' if(no_evaluation) else ''} evaluation categories)."

        return random.choice(categories)

        # The following code was very efficient, but only works when there is no split between training and evaluation categories
        #changed_dim = np.random.choice(self.nb_concepts, distance, replace=False)
        #changed_dim = set(changed_dim)
        #new_category = tuple(e if(i not in changed_dim) else (not e) for i,e in enumerate(category))
        #return new_category

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

        assert False, ('Sampling strategy \'%s\' unknown.' % sampling_strategy)

    def get_batch(self, size=None, data_type='any', sampling_strategies=None, no_evaluation=True, target_evaluation=False, target_is_original=None, keep_category=False):
        """Generates a batch as a Batch object.
        'size' is the size of the batch.
        'data_type' (train, test, any) indicates, when selecting an image from a category, from what part of this category we take it.
        'sampling_strategies' indicates how the distractor·s are determined.
        'no_evaluation' indicates whether we avoid evaluation categories.
        'target_evaluation' indicates whether the original/target category must be one of the evaluation categories.
        'noise' indicates how much (random normal) noise must be added to the images.
        'device' is a PyTorch parameter.
        """
        batch = []
        if(size is None): size = self.batch_size
        if(sampling_strategies is None): sampling_strategies = self.sampling_strategies
        if(target_is_original is None): target_is_original = self.same_img
        for _ in range(size):
            # Choice of the original/target category
            categories = self.training_categories
            if(not no_evaluation): categories = categories.union(self.evaluation_categories)
            if(target_evaluation): categories = categories.intersection(self.evaluation_categories)
            target_category = random.choice(list(categories))

            # Original image
            _original = self.category_to_datapoint(target_category, data_type).toInput(keep_category, self.device)

            # Target image
            if(target_is_original): _target = _original.copy()
            else: _target = self.category_to_datapoint(target_category, data_type).toInput(keep_category, self.device) # Same category
            
            # Noise is applied independently to the two images
            _original.add_normal_noise_(self.noise)
            _target.add_normal_noise_(self.noise)

            # Base distractors
            _base_distractors = []
            for sampling_strategy in sampling_strategies:
                distractor_category = self.sample_category(sampling_strategy, target_category, no_evaluation)
                distractor = self.category_to_datapoint(distractor_category, data_type).toInput(keep_category, self.device)
                distractor.add_normal_noise_(self.noise)

                _base_distractors.append(distractor)

            batch.append((_original, _target, _base_distractors))

        original, target, base_distractors = zip(*batch) # Unzips the list of pairs (to a pair of lists)

        return Batch(size=size, original=original, target=target, base_distractors=base_distractors)

class SimpleDataset(Dataset):
    def __init__(self, same_img=False, evaluation_categories=-1, data_set=None, display='tqdm', noise=0.0, device='cpu', batch_size=128, sampling_strategies=['different'], binary=False, constrain_dim=None):
        super().__init__(same_img, device, noise, batch_size, sampling_strategies)

        # The concepts
        possible_shapes = ['cube', 'sphere'] if binary else ['cube', 'sphere', 'ring']
        possible_colours = ['blue', 'red'] if binary else ['blue', 'red', 'green']
        possible_v_positions = ['down', 'up'] if binary else ['down', 'up', 'v-mid']
        possible_h_positions = ['left', 'right'] if binary else ['left', 'right', 'h-mid']
        possible_sizes = ['small', 'big'] if binary else ['small', 'big', 'medium']
        possibilities = [possible_shapes, possible_colours, possible_v_positions, possible_h_positions, possible_sizes]

        if(constrain_dim is None): constrain_dim = [len(concept) for concept in possibilities]
        assert all([(x >= 1) for x in constrain_dim])

        self.concepts = [] # List of dictionaries {value name -> value idx}
        for i, nb_values in enumerate(constrain_dim):
            values = possibilities[i]
            #random.shuffle(values)
            values = values[-1:] if(nb_values == 1) else values[:nb_values]

            self.concepts.append({v: i for (i, v) in enumerate(values)})

        self.nb_categories = np.prod([len(concept) for concept in self.concepts])
        self.nb_concepts = len(self.concepts)
        self.concept_names = ['shape', 'colour', 'vertical-pos', 'horizontal-pos', 'size']

        # If `evaluation_categories` is -1, all categories are used during training
        # Otherwise, a random category `ref_category` and all categories with a distance from it that is a multiple of `evaluation_categories` are reserved for evaluation
        ref_category = self.set_evaluation_categories(self.concepts, d=evaluation_categories, random_ref=False)

        self.training_categories_idx = np.array([self.category_idx(category) for category in self.training_categories])
        self.evaluation_categories_idx = np.array([self.category_idx(category) for category in self.evaluation_categories])

        def analyse_filename(filename):
            name, ext = os.path.splitext(filename)
            infos = name.split('_') # idx, shape, colour, vertical position, horizontal position, size

            idx = int(infos[0])
            category = tuple([concept.get(value) for (concept, value) in zip(self.concepts, infos[1:])])

            if(None in category): category = None # The value for one of the concepts is not accepted

            return (idx, category)

        print('Loading data from \'%s\'…' % data_set)

        # Loads all images from data_set as DataPoint
        dataset = [] # Will end as a Numpy array of DataPoint·s
        #for filename in os.listdir(data_set):
        tmp_data = os.listdir(data_set)
        if(display == 'tqdm'): tmp_data = tqdm.tqdm(tmp_data)
        for filename in tmp_data:
            full_path = os.path.join(data_set, filename)
            if(not os.path.isfile(full_path)): continue # We are only interested in files (not directories)

            idx, category = analyse_filename(filename)
            if(category is None): continue

            pil_img = PIL.Image.open(full_path).convert('RGB')
            #torchvision.transforms.ToTensor
            tensor_img = torchvision.transforms.functional.to_tensor(pil_img)

            dataset.append(DataPoint(idx, category, tensor_img))
        self._dataset = np.array(dataset)

        categories = defaultdict(list) # Will end as a dictionary from category (tuple) to Numpy array of DataPoint·s
        for img in self._dataset:
            categories[img.category].append(img)
        self.categories = {k: np.array(v) for (k, v) in categories.items()}

        # A momentum factor of 0.99 means that each cell of the failure matrix contains a statistics over 100 examples.
        # In our setting, each evaluation phase updates each cell 10 times, so the matrix is renewed every 10 epochs.
        self.failure_based_distribution = FailureBasedDistribution(self.nb_categories, momentum_factor=0.99, smoothing_factor=10.0)

        if(display != 'tqdm'): print('Loading done')
        
        #show_imgs([self.average_image()], 1)
    
    # Should only be used for debugging purpose. Use `get_batch` instead
    def get_datapoint(self, i):
        return self._dataset[i]

    # Category tuples are read from left to right (contrary to usual numbers)
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

    def category_idx(self, category_tuple):
        category_idx = 0
        k = 1
        for i, concept in enumerate(self.concepts):
            category_idx += category_tuple[i] * k
            k *= len(concept)

        #if(np.random.randint(2)): assert self.category_tuple(category_idx) == category_tuple # DEBUG ONLY

        return category_idx

    _average_image = None
    def average_image(self):
        if(self._average_image is None):
            tmp = torch.stack([x.img for x in self._dataset])
            self._average_image = tmp.mean(axis=0)

        return self._average_image
    
    # Should be consistant with `category_to_datapoint`
    def size(self, data_type, no_evaluation):
        size = 0
        
        categories = self.training_categories
        if(not no_evaluation): categories = categories.union(self.evaluation_categories)

        for category in categories:
            size += self.category_size(category, data_type)

        return size

    #def __len__(self):
    #    return len(self._dataset)

    def category_size(self, category, data_type):
        split = self.category_split(category)
        
        if(data_type == 'train'): return split[1] - split[0]
        elif(data_type == 'test'): return split[2] - split[1]
        elif(data_type == 'any'): return split[-1] - split[0]
        else: assert False, ('Data type \'%s\' unknown.' % data_type)

    def category_split(self, category):
        l = len(self.categories[category])
        
        if(category in self.training_categories):
            split_point = ((4 * l) // 5)
            return [0, split_point, l] # 4/5th in the train portion, 1/5th in the test portion
        
        return [0, 0, l] # Everything in the test portion

    def category_to_datapoint(self, category, data_type):
        split = self.category_split(category)

        if(data_type == 'train'): a, b = split[0], (split[1]-1)
        elif(data_type == 'test'): a, b = split[1], (split[2]-1)
        elif(data_type == 'any'): a, b = split[0], (split[-1]-1)
        else: assert False, ('Data type \'%s\' unknown.' % data_type)
        i = random.randint(a, b) # A random integer between a and b (included)

        return self.categories[category][i]

class PairDataset(Dataset):
    def __init__(self, same_img=False, evaluation_categories=-1, data_set=None, display='tqdm', noise=0.0, device='cpu', batch_size=128, sampling_strategies=['different'], binary=False, constrain_dim=None):
        super().__init__(same_img, device, noise, batch_size, sampling_strategies)

        self.base_dataset = SimpleDataset(same_img=None, evaluation_categories=-1, data_set=data_set, display=display, noise=None, device=None, batch_size=None, sampling_strategies=None, binary=binary, constrain_dim=constrain_dim) # Maybe the display argument should be modified. Also note that some batch_size and sampling strategies should be useless, and probably evaluation_categories and device too

        self.concepts = (self.base_dataset.concepts * 2)
        self.nb_categories = (self.base_dataset.nb_categories * self.base_dataset.nb_categories)
        self.nb_concepts = (2 * self.base_dataset.nb_concepts)
        self.concept_names = ([('left_' + concept_name) for concept_name in self.base_dataset.concept_names] + [('right_' + concept_name) for concept_name in self.base_dataset.concept_names])

        # If `evaluation_categories` is -1, all categories are used during training
        # Otherwise, a random category `ref_category` and all categories with a distance from it that is a multiple of `evaluation_categories` are reserved for evaluation
        ref_category = self.set_evaluation_categories(self.concepts, d=evaluation_categories, random_ref=False)

        self.training_categories_idx = np.array([self.category_idx(category) for category in self.training_categories])
        self.evaluation_categories_idx = np.array([self.category_idx(category) for category in self.evaluation_categories])
        
        # A momentum factor of 0.99 means that each cell of the failure matrix contains a statistics over 100 examples.
        # In our setting, each evaluation phase updates each cell 10 times, so the matrix is renewed every 10 epochs.
        self.failure_based_distribution = FailureBasedDistribution(self.nb_categories, momentum_factor=0.99, smoothing_factor=10.0)

        #show_imgs([self.average_image()], 1)
    
    # Should only be used for debugging purpose. Use `get_batch` instead
    def get_datapoint(self, i):
        base_len = len(self.base_dataset)
        left_i = (i % base_len)
        right_i = (i // base_len)

        left_datapoint = self.base_dataset.get_datapoint(left_i)
        right_datapoint = self.base_dataset.get_datapoint(left_i)

        return self.combine_datapoint(left_datapoint, right_datapoint)
    
    def category_tuple(self, category_idx):
        left_idx = (category_idx % self.base_dataset.nb_categories)
        left_tuple = self.base_dataset.category_tuple(left_idx)
        
        right_idx = (category_idx // self.base_dataset.nb_categories)
        right_tuple = self.base_dataset.category_tuple(right_idx)

        return (left_tuple + right_tuple)
    
    def divide_category(self, category_tuple):
        return (category_tuple[:self.base_dataset.nb_concepts], category_tuple[self.base_dataset.nb_concepts:])

    def category_idx(self, category_tuple):
        left_tuple, right_tuple = self.divide_category(category_tuple)

        left_idx = self.base_dataset.category_idx(left_tuple)
        right_idx = self.base_dataset.category_idx(right_tuple)

        return (left_idx + (self.base_dataset.nb_categories * right_idx))
        
    _average_image = None
    def average_image(self):
        if(self._average_image is None):
            base_img = self.base_dataset.average_image()

            self._average_image = combine_images(base_img, base_img)

        return self._average_image

    # Should be consistant with `category_to_datapoint`
    def size(self, data_type, no_evaluation):
        size = 0
        
        categories = self.training_categories
        if(not no_evaluation): categories = categories.union(self.evaluation_categories)

        for category in categories:
            size += self.category_size(category, data_type)

        return size

    #def __len__(self):
    #    return len(self._dataset)

    def category_size(self, category, data_type):
        if(category in self.evaluation_categories):
            assert data_type != 'train'
            base_data_type = 'any'
        else:
            base_data_type = data_type
        
        left_tuple, right_tuple = self.divide_category(category)
        left_size, right_size = self.base_dataset.category_size(left_tuple, base_data_type), self.base_dataset.category_size(right_tuple, 'any') 

        return (left_size * right_size)

    def combine_datapoint(self, datapoint1, datapoint2):
        idx = None # The following would work if the indices of the images are exactly the number between 0 and the size of the dataset: (left_datapoint.idx + (len(self.base_dataset) * right_datapoint_idx))
        category = (datapoint1.category + datapoint2.category)
        img = combine_images(datapoint1.img, datapoint2.img)
        #show_imgs([img], 1)
        
        return DataPoint(idx, category, img)
  
    # For training categories, training (resp. testing) images are composed of a training (resp. testing) images and an 'any' image
    # For evaluation categories, there is no training image and testing images are composed of two 'any' images
    # As a consequence, there is no overlap of the combinations even though, for instance, an image from an evaluation category may be composed of two sub-images appearing independently in the training portions of two training categories
    # I'm doing this because otherwise the dataset is harder to manage (think about 'any' containing more datapoints than the sum of the 'train' and the 'test' portion).
    # This might not seem optimal at first, but (i) combined images will still behave as expected and (ii) the combined dataset will probably be gigantic so overfitting might be less of a concern.
    def category_to_datapoint(self, category_tuple, data_type):
        if(category_tuple in self.evaluation_categories):
            assert data_type != 'train'
            base_data_type = 'any'
        else:
            base_data_type = data_type

        left_tuple, right_tuple = self.divide_category(category_tuple)
        left_datapoint, right_datapoint = self.base_dataset.category_to_datapoint(left_tuple, base_data_type), self.base_dataset.category_to_datapoint(right_tuple, 'any')

        return self.combine_datapoint(left_datapoint, right_datapoint)

def get_data_loader(args):
    sampling_strategies = args.sampling_strategies.split('/')

    if(args.pair_images): dataset = PairDataset(args.same_img, evaluation_categories=args.evaluation_categories, data_set=args.data_set, display=args.display, noise=args.noise, device=args.device, batch_size=args.batch_size, sampling_strategies=sampling_strategies, binary=args.binary_dataset, constrain_dim=args.constrain_dim)
    else: dataset = SimpleDataset(args.same_img, evaluation_categories=args.evaluation_categories, data_set=args.data_set, display=args.display, noise=args.noise, device=args.device, batch_size=args.batch_size, sampling_strategies=sampling_strategies, binary=args.binary_dataset, constrain_dim=args.constrain_dim)

    dataset.print_info()

    return dataset
