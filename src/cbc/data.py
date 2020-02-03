import itertools as it
import os
from collections import namedtuple, defaultdict

import torch, torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import PIL
import numpy as np

from config import *

Batch = namedtuple("Batch", ["size", "alice_input", "bob_input"])

class DataPoint():
    def __init__(self, idx, category, img):
        self.idx = idx
        self.category = category
        self.img = img

class DistinctTargetClassDataLoader():
    # The binary concepts
    nb_concepts = 5
    shapes = {'sphere': True, 'cube': False}
    colours = {'red': True, 'blue': False}
    v_positions = {'up': True, 'down': False}
    h_positions = {'right': True, 'left': False}
    sizes = {'big': True, 'small': False}
    _concepts = [shapes, colours, v_positions, h_positions, sizes]

    def __init__(self, same_img=False):
        self.same_img = same_img # Whether Bob sees Alice's image or another one (of the same category)

        def analyse_filename(filename):
            name, ext = os.path.splitext(filename)
            infos = name.split('_') # idx, shape, colour, vertical position, horizontal position, size

            idx = int(infos[0])
            category = tuple(map(dict.__getitem__, self._concepts, infos[1:])) # equivalent to (self.shapes[infos[1]], self.colours[infos[2]], self.v_positions[infos[3]], self.h_positions[infos[4]], self.sizes[infos[5]])

            return (idx, category)

        print('Loading data from \'%s\'...' % DATASET_PATH)

        # Loads all images from DATASET_PATH as DataPoint
        dataset = [] # Will end as a Numpy array of DataPoint·s
        for filename in os.listdir(DATASET_PATH):
            full_path = os.path.join(DATASET_PATH, filename)
            if(not os.path.isfile(full_path)): continue # We are only interested in files (not directories)

            idx, category = analyse_filename(filename)

            pil_img = PIL.Image.open(full_path).convert('RGB')
            #torchvision.transforms.ToTensor
            tensor_img = torchvision.transforms.functional.to_tensor(pil_img)

            dataset.append(DataPoint(idx, category, tensor_img))
        self.dataset = np.array(dataset)

        categories = defaultdict(list)# Will end as a dictionary from category (tuple) to Numpy array of DataPoint·s
        for img in self.dataset:
            categories[img.category].append(img)
        self.categories = {k: np.array(v) for (k, v) in categories.items()}
        
        print('Loading done')


    def _distance_to_category(self, category, distance):
        """Returns a category that is at distance `distance` from category `category`"""
        changed_dim = np.random.choice(self.nb_concepts, distance, replace=False)
        changed_dim = set(changed_dim)
        new_category = tuple(e if(i not in changed_dim) else not e for i,e in enumerate(category))
        return new_category


    def _different_category(self, category):
        """Returns a category that is different from `category`"""
        distance = np.random.randint(self.nb_concepts) + 1
        return self._distance_to_category(category, distance)

    def get_batch(self, size):
        """Generates a batch as a Batch object.
        'alice_input' and 'bob_input' are both tensors of outer dimension of size `size`
        Each element of 'alice_input' is an image
        Each element of 'bob_input' is the stack of three images related to their counterpart in 'alice_input':
            - bob_input[0] is an image of the same category,
            - bob_input[1] is an image of a neighbouring category (distance = 1), and
            - bob_input[2] is an image of a different category (distance != 0)"""
        batch = []
        for _ in range(size):
            alice_data = np.random.choice(self.dataset)
            bob_a = alice_data if(self.same_img) else np.random.choice(self.categories[alice_data.category]) # same category
            bob_b = np.random.choice(self.categories[self._distance_to_category(alice_data.category, 1)]) # neighbouring category
            bob_c = np.random.choice(self.categories[self._different_category(alice_data.category)]) # different category

            l = [bob_a, bob_b, bob_c]
            batch.append((alice_data.img, torch.stack([x.img for x in l])))
        alice_input, bob_input = map(torch.stack, zip(*batch)) # Unzips the list of pairs (to a pair of lists) and then stacks
        # Adds noise if necessary (normal random noise + clamping)
        if(NOISE_STD_DEV > 0.0):
            alice_input = torch.clamp((alice_input + (NOISE_STD_DEV * torch.randn(size=alice_input.shape))), 0.0, 1.0)
            bob_input = torch.clamp((bob_input + (NOISE_STD_DEV * torch.randn(size=bob_input.shape))), 0.0, 1.0)

        return Batch(size=BATCH_SIZE, alice_input=alice_input.to(DEVICE), bob_input=bob_input.to(DEVICE))

    def __iter__(self):
        """Iterates over batches of size `BATCH_SIZE`"""
        while True:
            yield self.get_batch(BATCH_SIZE)

def get_data_loader(same_img=False):
    return DistinctTargetClassDataLoader(same_img)
