import itertools as it
import os
from collections import namedtuple, defaultdict
import itertools
import random

import tqdm

import torch, torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import PIL
import numpy as np

from config import *

# [START] Imports shared code from the parent directory
parent_dir_path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(parent_dir_path)

from utils import add_normal_noise

sys.path.remove(parent_dir_path)
# [END] Imports shared code from the parent directory

Batch = namedtuple("Batch", ["size", "original_img", "target_img", "base_distractors"])

class DataPoint():
    def __init__(self, idx, category, img):
        self.idx = idx
        self.category = category
        self.img = img

class DistinctTargetClassDataLoader():
    # The binary concepts
    shapes = {'sphere': True, 'cube': False}
    colours = {'red': True, 'blue': False}
    v_positions = {'up': True, 'down': False}
    h_positions = {'right': True, 'left': False}
    sizes = {'big': True, 'small': False}
    _concepts = [shapes, colours, v_positions, h_positions, sizes]
    concept_names = ['shape', 'colour', 'vertical-pos', 'horizontal-pos', 'size']
    nb_concepts = len(_concepts)

    def __init__(self, same_img=False, evaluation_categories=None):
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
        print('Training categories: %s' % self.training_categories)
        print('Evaluation categories: %s' % self.evaluation_categories)

        def analyse_filename(filename):
            name, ext = os.path.splitext(filename)
            infos = name.split('_') # idx, shape, colour, vertical position, horizontal position, size

            idx = int(infos[0])
            category = tuple(map(dict.__getitem__, self._concepts, infos[1:])) # equivalent to (self.shapes[infos[1]], self.colours[infos[2]], self.v_positions[infos[3]], self.h_positions[infos[4]], self.sizes[infos[5]])

            return (idx, category)

        print('Loading data from \'%s\'...' % DATASET_PATH)

        # Loads all images from DATASET_PATH as DataPoint
        dataset = [] # Will end as a Numpy array of DataPoint·s
        #for filename in os.listdir(DATASET_PATH):
        tmp_data = os.listdir(DATASET_PATH)
        if(not SIMPLE_DISPLAY): tmp_data = tqdm.tqdm(tmp_data)
        for filename in tmp_data:
            full_path = os.path.join(DATASET_PATH, filename)
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

        if(SIMPLE_DISPLAY): print('Loading done')

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

    def get_batch(self, size, no_evaluation=True, target_evaluation=False):
        """Generates a batch as a Batch object.
        'original_img', 'target_img' and 'base_distractors' are all tensors of outer dimension of size `size`.
        If 'no_evaluation' is True, none of the image is from an evaluation category.
        Each element of 'original_img' is an image. In 'target_evaluation' is True, then this image is from an evaluation category.
        Each element of 'target_img' is an image of the same category as its counterpart in 'original_img'.
        Each element of 'base_distractors' is the stack of two images related to their counterpart in 'original_img':
            - base_distractors[0] is an image of a neighbouring category (distance = 1), and
            - base_distractors[1] is an image of a different category (distance != 0)."""
        batch = []
        for _ in range(size):
            categories = self.training_categories
            if(not no_evaluation): categories = categories.union(self.evaluation_categories)
            if(target_evaluation): categories = categories.intersection(self.evaluation_categories)
            target_category = random.choice(list(categories))

            _original_img = np.random.choice(self.categories[target_category])
            #_original_img = np.random.choice(self.dataset)
            _target_img = _original_img if(self.same_img) else np.random.choice(self.categories[_original_img.category]) # same category
            distractor_1 = np.random.choice(self.categories[self._distance_to_category(_original_img.category, 1, no_evaluation)]) # neighbouring category
            distractor_2 = np.random.choice(self.categories[self._different_category(_original_img.category, no_evaluation)]) # different category

            _base_distractors = [distractor_1, distractor_2]
            batch.append((_original_img.img, _target_img.img, torch.stack([x.img for x in _base_distractors])))

        original_img, target_img, base_distractors = map(torch.stack, zip(*batch)) # Unzips the list of pairs (to a pair of lists) and then stacks

        # Adds noise if necessary (normal random noise + clamping)
        if(NOISE_STD_DEV > 0.0):
            original_img = add_normal_noise(original_img, std_dev=NOISE_STD_DEV, clamp_values=(0.0, 1.0))
            target_img = add_normal_noise(target_img, std_dev=NOISE_STD_DEV, clamp_values=(0.0, 1.0))
            base_distractors = add_normal_noise(base_distractors, std_dev=NOISE_STD_DEV, clamp_values=(0.0, 1.0))

        return Batch(size=BATCH_SIZE, original_img=original_img.to(DEVICE), target_img=target_img.to(DEVICE), base_distractors=base_distractors.to(DEVICE))

    def __iter__(self):
        """Iterates over batches of size `BATCH_SIZE`"""
        while True:
            yield self.get_batch(BATCH_SIZE)

def get_data_loader(same_img=False):
    return DistinctTargetClassDataLoader(same_img)
