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

from .misc import add_normal_noise


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

    def base_distractors_img(self, flat=False, stack=False, f=None):
        if(f is None): f = (lambda x: x)

        if(not flat):
            tmp = [[f(x.img) for x in base_distractor] for base_distractor in self.base_distractors]
            if(stack): tmp = list(map(torch.stack, tmp))
        else: tmp = [f(x.img) for base_distractor in self.distractors for x in base_distractor]

        if(stack): return torch.stack(tmp)
        else: return tmp

    def get_images(self, original=True, target=True, base_distractors=True):
        images = []
        if(for_original): images.extend(self.original_img())
        if(for_target): images.extend(self.target_img())
        if(for_base_distractors): images.extend(self.base_distractors_img(flat=True))

        return images

    def require_grad(self, for_original=True, for_target=True, for_base_distractors=True):
        for img in self.get_images(original=for_original, target=for_target, base_distractors=for_base_distractors):
            img.requires_grad = True

class InputDataPoint():
    def __init__(self, img, category=None):
        self.img = img
        self.category = category

    # Adds noise if necessary (normal random noise + clamping)
    def add_normal_noise_(self, noise):
        if(noise <= 0.0): return

        self.img = add_normal_noise(self.img, std_dev=noise, clamp_values=(0.0, 1.0))

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
    def __init__(self, nb_categories, momentum_factor=0.79, smoothing_factor=1.0):
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
        unnormalised_dist = (self.failure_matrix[category_idx, allowed_categories_idx] / self.counts_matrix[category_idx, allowed_categories_idx])
        return (unnormalised_dist / np.linalg.norm(unnormalised_dist, 1))

    def sample(self, category_idx, allowed_categories_idx=None):
        dist = self.distribution(category_idx, allowed_categories_idx)

        if(allowed_categories_idx is None): allowed_categories_idx = range(dist.shape[1])

        return np.random.choice(a=allowed_categories_idx, p=dist)

class DistinctTargetClassDataLoader():
    # The binary concepts
    shapes = {'cube': 0, 'sphere': 1}
    colours = {'blue': 0, 'red': 1}
    v_positions = {'down': 0, 'up': 1}
    h_positions = {'left': 0, 'right': 1}
    sizes = {'small': 0, 'big': 1}
    _concepts = [shapes, colours, v_positions, h_positions, sizes]
    nb_categories = np.prod([len(concept) for concept in _concepts])
    nb_concepts = len(_concepts)
    concept_names = ['shape', 'colour', 'vertical-pos', 'horizontal-pos', 'size']

    def category_tuple(self, category_idx):
        ks = []
        k = 1
        for i, concept in enumerate(self._concepts):
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
        for i, concept in enumerate(self._concepts):
            category_idx += category_tuple[i] * k
            k *= len(concept)

        #if(np.random.randint(2)): assert self.category_tuple(category_idx) == category_tuple # DEBUG ONLY

        return category_idx

    def __init__(self, same_img=False, evaluation_categories=None, data_set=None, simple_display=False, noise=0.0, device='cpu', batch_size=128):
        self.device = device
        self.noise = noise
        self.batch_size=batch_size
        self.same_img = same_img # Whether Bob sees Alice's image or another one (of the same category)

        # If `evaluation_categories` is None, all categories are used during training
        # Otherwise, category (0,0,…,0) and all categories with a distance from (0,0,…,0) that is a multiple of `evaluation_categories` are reserved for evaluation
        self.training_categories = set()
        self.evaluation_categories = set()
        category = np.full(self.nb_concepts, False)
        while(True): # Enumerates all categories to sort them
            dist = category.sum()
            if((evaluation_categories is not None) and ((dist % evaluation_categories) == 0)):
                self.evaluation_categories.add(tuple(category))
            else:
                self.training_categories.add(tuple(category))

            if(category.all()): break # If category is (1,1,…,1), we've seen all categories
            for j in range(len(category)): # Let's go to the next category
                if(category[j] == False):
                    category[j] = True
                    break

                category[j] = False

        self.training_categories_idx = np.array([self.category_idx(category) for category in self.training_categories])
        self.evaluation_categories_idx = np.array([self.category_idx(category) for category in self.evaluation_categories])

        print('Training categories: %s' % self.training_categories)
        print('Evaluation categories: %s' % self.evaluation_categories)

        def analyse_filename(filename):
            name, ext = os.path.splitext(filename)
            infos = name.split('_') # idx, shape, colour, vertical position, horizontal position, size

            idx = int(infos[0])
            category = tuple(map(dict.__getitem__, self._concepts, infos[1:])) # equivalent to (self.shapes[infos[1]], self.colours[infos[2]], self.v_positions[infos[3]], self.h_positions[infos[4]], self.sizes[infos[5]])

            return (idx, category)

        print('Loading data from \'%s\'...' % data_set)

        # Loads all images from data_set as DataPoint
        dataset = [] # Will end as a Numpy array of DataPoint·s
        #for filename in os.listdir(data_set):
        tmp_data = os.listdir(data_set)
        if(not simple_display): tmp_data = tqdm.tqdm(tmp_data)
        for filename in tmp_data:
            full_path = os.path.join(data_set, filename)
            if(not os.path.isfile(full_path)): continue # We are only interested in files (not directories)

            idx, category = analyse_filename(filename)

            pil_img = PIL.Image.open(full_path).convert('RGB')
            #torchvision.transforms.ToTensor
            tensor_img = torchvision.transforms.functional.to_tensor(pil_img)

            dataset.append(DataPoint(idx, category, tensor_img))
        self.dataset = np.array(dataset)

        categories = defaultdict(list) # Will end as a dictionary from category (tuple) to Numpy array of DataPoint·s
        for img in self.dataset:
            categories[img.category].append(img)
        self.categories = {k: np.array(v) for (k, v) in categories.items()}

        self.failure_based_distribution = FailureBasedDistribution(self.nb_categories)

        if(simple_display): print('Loading done')

    def __len__(self):
        return len(self.dataset)

    _average_image = None
    def average_image(self):
        if(self._average_image is None):
            tmp = torch.stack([x.img for x in self.dataset])
            self._average_image = tmp.mean(axis=0)

        return self._average_image

    def _distance_to_categories(self, category, distance, no_evaluation):
        """Returns the list of all categories at distance `distance` from category `category`."""
        categories = []
        for changed_dim in itertools.combinations(range(self.nb_concepts), distance): # Iterates over all subsets of [|0, `self.nb_concepts`|[ of size `distance`
            changed_dim = set(changed_dim)
            new_category = tuple(e if(i not in changed_dim) else (not e) for i,e in enumerate(category))

            if(no_evaluation and (new_category in self.evaluation_categories)): continue
            categories.append(new_category)

        return categories

    def _distance_to_category(self, category, distance, no_evaluation):
        """Returns a category that is at distance `distance` from category `category`."""
        categories = self._distance_to_categories(category, distance, no_evaluation)

        return random.choice(categories)

        # The following code was very efficient, but only works when there is no split between training and evaluation categories
        #changed_dim = np.random.choice(self.nb_concepts, distance, replace=False)
        #changed_dim = set(changed_dim)
        #new_category = tuple(e if(i not in changed_dim) else (not e) for i,e in enumerate(category))
        #return new_category

    def _different_category(self, category, no_evaluation):
        """Returns a category that is different from `category`."""
        categories = self.training_categories
        if(not no_evaluation): categories = categories.union(self.evaluation_categories)
        categories = categories.difference(set([category]))

        return random.choice(list(categories))

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

        assert False, ('Sampling strategy \'%s\' unknown.' % sampling_strategy)

    def get_batch(self, size=self.batch_size, sampling_strategies=['difficulty'], no_evaluation=True, target_evaluation=False, noise=self.noise, keep_category=False, device=self.device):
        """Generates a batch as a Batch object.
        'size' is the size of the batch.
        'sampling_strategies' indicates how the distractor·s are determined.
        'no_evaluation' indicates whether we avoid evaluation categories.
        'target_evaluation' indicates whether the original/target category must be one of the evaluation categories.
        'noise' indicates how much (random normal) noise must be added to the images.
        'device' is a PyTorch parameter.
        """
        batch = []
        for _ in range(size):
            # Choice of the original/target category
            categories = self.training_categories
            if(not no_evaluation): categories = categories.union(self.evaluation_categories)
            if(target_evaluation): categories = categories.intersection(self.evaluation_categories)
            target_category = random.choice(list(categories))

            # Original image
            _original = np.random.choice(self.categories[target_category]).toInput(keep_category, device)
            _original.add_normal_noise_(noise)

            # Target image
            if(self.same_img): _target = _original
            else: # Same category
                _target = np.random.choice(self.categories[target_category]).toInput(keep_category, device)
                _target.add_normal_noise_(noise)

            # Base distractors
            _base_distractors = []
            for sampling_strategy in sampling_strategies:
                distractor_category = self.sample_category(sampling_strategy, target_category, no_evaluation)
                distractor = np.random.choice(self.categories[distractor_category]).toInput(keep_category, device)
                distractor.add_normal_noise_(noise)

                _base_distractors.append(distractor)

            batch.append((_original, _target, _base_distractors))

        original, target, base_distractors = zip(*batch) # Unzips the list of pairs (to a pair of lists)

        return Batch(size=size, original=original, target=target, base_distractors=base_distractors)

    @deprecated(version='[on n\'a pas de numéro de version, mais aujourd\'hui est le 2019/02/19, un mercredi]', reason='pas vraiment de raison ; je fais un test')
    def __iter__(self):
        """Iterates over batches of default parameters"""
        while True:
            yield self.get_batch()


def get_data_loader(args):
    return DistinctTargetClassDataLoader(args.same_img, data_set=args.data_set, simple_display=args.simple_display, noise=args.noise, device=args.device, batch_size=args.batch_size)
