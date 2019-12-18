import itertools as it
import functools as ft
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

    def __init__(self):
        def analyse_filename(filename):
            name, ext = os.path.splitext(filename)
            infos = name.split('_') # nothing, idx, shape, colour, vertical position, horizontal position, size

            idx = int(infos[1])
            category = tuple(map(dict.__getitem__, self._concepts, infos[2:])) # equivalent to (self.shapes[infos[1]], self.colours[infos[2]], self.v_positions[infos[3]], self.h_positions[infos[4]], self.sizes[infos[5]])

            return (idx, category)

        # Loads all images from DATASET_PATH as DataPoint
        dataset = [] # Will end as a Numpy array of DataPoint·s
        for filename in os.listdir(DATASET_PATH):
            full_path = os.path.join(DATASET_PATH, filename)
            if not os.path.isfile(full_path): continue # We are only interested in files (not directories)

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


    def _distance_to_category(self, category, distance):
        """Returns a category that is at distance `distance` from category `category`"""
        changed_dim = np.random.choice(self.nb_concepts, distance, replace=False)
        changed_dim = set(changed_dim)
        new_category = tuple(e if i not in changed_dim else not e for i,e in enumerate(category))
        return new_category


    def _different_category(self, category):
        """Returns a category that is different from `category`"""
        distance = np.random.randint(self.nb_concepts) + 1
        return self._distance_to_category(category, distance)


    def _get_img(self, row):
        return torch.stack([e.img for e in row])

    def _make_bob_examples(self, alice_data_row):  # TODO: this can probably be vectorized better, probably using indexing
        alice_data_category = alice_data_row[0].category
        return [
            np.random.choice(self.categories[alice_data_category]),
            np.random.choice(self.categories[self._distance_to_category(alice_data_category, 1)]),
            np.random.choice(self.categories[self._different_category(alice_data_category)]),
        ]

    def _get_batch(self):
        """Generates a batch as a Batch object.
        'alice_input' and 'bob_input' are both tensors of outer dimension of size BATCH_SIZE
        Each element of 'alice_input' is an image
        Each element of 'bob_input' is the stack of three images related to their counterpart in 'alice_input':
            - bob_input[0] is an image of the same category,
            - bob_input[1] is an image of a neighbouring category (distance = 1), and
            - bob_input[2] is an image of a different category (distance != 0)"""
        alice_examples = np.random.choice(self.dataset, BATCH_SIZE)
        bob_examples = np.apply_along_axis(self._make_bob_examples, 1, alice_examples[:,None])

        alice_input = torch.from_numpy(np.apply_along_axis(self._get_img, 0, alice_examples))
        bob_input = torch.from_numpy(np.apply_along_axis(self._get_img, 1, bob_examples))

        # Adds noise if necessary (normal random noise + clamping)
        if NOISE_STD_DEV > 0.0:
            alice_input = torch.clamp((alice_input + (NOISE_STD_DEV * torch.randn(size=alice_input.shape))), 0.0, 1.0)
            bob_input = torch.clamp((bob_input + (NOISE_STD_DEV * torch.randn(size=bob_input.shape))), 0.0, 1.0)

        return Batch(size=BATCH_SIZE, alice_input=alice_input.to(DEVICE), bob_input=bob_input.to(DEVICE))

    def __iter__(self):
        """Iterates over batches"""
        while True:
            yield self._get_batch()

def get_data_loader():
    return DistinctTargetClassDataLoader()
