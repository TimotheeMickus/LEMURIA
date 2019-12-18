import itertools as it
import os

import torch, torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import PIL
import numpy as np

from config import *

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
            tensor_img = tensor_img[:-1] # Removes the alpha channel

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
    # A batch is a stack of `BATCH_SIZE` instance
    # An instance is a stack of four images: alice_img, an image of the same category, an image of a neighbouring category (distance = 1) and an image of a different category (distance != 0)
    def _get_batch(self):
        batch = []
        for _ in range(BATCH_SIZE):
            alice_img = np.random.choice(self.dataset)
            bob_a = np.random.choice(self.categories[alice_img.category]) # same category
            bob_b = np.random.choice(self.categories[self._distance_to_category(alice_img.category, 1)]) # neighbouring category
            bob_c = np.random.choice(self.categories[self._different_category(alice_img.category)]) # different category

            l = [alice_img, bob_a, bob_b, bob_c]
            batch.append(torch.stack([x.img for x in l]))

        return torch.stack(batch)

    # Iterates over batches
    def __iter__(self):
        while True:
            yield self._get_batch()

def get_data_loader():
    return DistinctTargetClassDataLoader()
