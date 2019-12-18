import itertools as it
import os
from collections import namedtuple

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

    def __init__(self):
        def analyse_filename(filename):
            name, ext = os.path.splitext(filename)
            infos = name.split('_') # nothing, idx, shape, colour, vertical position, horizontal position, size
            
            idx = int(infos[1])
            category = (self.shapes[infos[2]], self.colours[infos[3]], self.v_positions[infos[4]], self.h_positions[infos[5]], self.sizes[infos[6]])

            return (idx, category)

        # Loads all images from DATASET_PATH as DataPoint
        dataset = [] # Will end as a Numpy array of DataPoint·s
        for filename in os.listdir(DATASET_PATH):
            full_path = os.path.join(DATASET_PATH, filename)
            if(not os.path.isfile(full_path)): continue # We are only interested in files (not directories)
            
            idx, category = analyse_filename(filename)
            
            pil_img = PIL.Image.open(full_path)
            #torchvision.transforms.ToTensor
            tensor_img = torchvision.transforms.functional.to_tensor(pil_img)
            tensor_img = tensor_img[:-1].contiguous() # Removes the alpha channel TODO Peut-être faudrait-il aussi appeler 'float()'

            dataset.append(DataPoint(idx, category, tensor_img))
        self.dataset = np.array(dataset)

        categories = {} # Will end as a dictionary from category (tuple) to Numpy array of DataPoint·s
        for img in self.dataset:
            categories.setdefault(img.category, []).append(img)
        self.categories = {k: np.array(v) for (k, v) in categories.items()}

    def _random_category(self):
        return tuple(np.random.choice([True, False], self.nb_concepts))

    # Returns a category that is at distance `distance` from category `category`
    def _distance_to_category(self, category, distance):
        tmp_category = list(category)

        changed_dim = np.random.choice(self.nb_concepts, distance, replace=False)
        for i in changed_dim:
            tmp_category[i] = (not tmp_category[i])

        return tuple(tmp_category)

    # Returns a category that is different from `category`
    def _different_category(self, category):
        distance = np.random.randint(self.nb_concepts) + 1
        return self._distance_to_category(category, distance)

    # Generates a batch
    # A batch is a Batch
    # 'alice_input' and 'bob_input' are both tensors of outer dimension of size BATCH_SIZE
    # Each element of 'alice_input' is an image
    # Each element of 'bob_input' is the stack of four images related to their counterpart in 'alice_input': an image of the same category, an image of a neighbouring category (distance = 1) and an image of a different category (distance != 0)
    def _get_batch(self):
        batch = []
        for _ in range(BATCH_SIZE):
            alice_data = np.random.choice(self.dataset)
            bob_a = np.random.choice(self.categories[alice_data.category]) # same category
            bob_b = np.random.choice(self.categories[self._distance_to_category(alice_data.category, 1)]) # neighbouring category
            bob_c = np.random.choice(self.categories[self._different_category(alice_data.category)]) # different category

            l = [bob_a, bob_b, bob_c]
            batch.append((alice_data.img, torch.stack([x.img for x in l])))

        alice_input, bob_input = list(map((lambda l: torch.stack(l)), zip(*batch))) # Unzips the list of pairs (to a pair of lists) and then stacks

        # Adds noise if necessary (normal random noise + clamping)
        if(NOISE_STD_DEV > 0.0):
            alice_input = torch.clamp((alice_input + (NOISE_STD_DEV * torch.randn(size=alice_input.shape))), 0.0, 1.0)
            bob_input = torch.clamp((bob_input + (NOISE_STD_DEV * torch.randn(size=bob_input.shape))), 0.0, 1.0)

        # TODO: Faudrait-il appeler '.contiguous()' à un moment ?

        return Batch(size=BATCH_SIZE, alice_input=alice_input.to(DEVICE), bob_input=bob_input.to(DEVICE))

    # Iterates over batches
    def __iter__(self):
        while True:
            yield self._get_batch()

def get_data_loader():
    return DistinctTargetClassDataLoader()
