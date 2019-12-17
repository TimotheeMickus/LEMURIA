import itertools as it
import os

import torch, torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import PIL
import numpy as np

from config import *

class DataPoint():
    def __init__(self, id, class, img):
        self.id = id
        self.class = class
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
            infos = filename.split('_') # nothing, id, shape, colour, vertical position, horizontal position, size
            
            id = int(infos[1])
            class = (shapes[infos[2]], colours[infos[3]], v_positions[infos[4]], h_positions[infos[5]], sizes[infos[6]])

            return (id, class)

        # Loads all images from DATASET_PATH as DataPoint
        dataset = [] # Will end as a Numpy array of DataPoint·s
        for filename in os.listdir(DATASET_PATH):
            full_path = os.join(DATASET_PATH, filename)
            if(not os.isfile(full_path)): continue # We are only interested in files (not directories)
            
            id, class = analyse_filename(filename)
            
            pil_img = PIL.Image.open(full_path)
            #torchvision.transforms.ToTensor
            tensor_img = torchvision.transforms.functional.to_tensor(pil_img)

            dataset.append(DataPoint(id, class, tensor_img))
        self.dataset = np.array(dataset)

        classes = {} # Will end as a dictionary from class (tuple) to Numpy array of DataPoint·s
        for img in self.dataset:
            classes.setdefault(img.class, []).append(img)
        self.classes = {k: np.array(v) for (k, v) in classes.items()}

    def _random_class(self):
        return tuple(np.random.choice([True, False], nb_concepts))

    # Returns a class that is at distance `distance` from class `class`
    def _dinstance_to_class(self, class, distance):
        tmp_class = list(class)

        changed_dim = np.random.choice(nb_concepts, distance, replace=False)
        for i in changed_dim:
            tmp_class[i] = (not tmp_class[i])

        return tuple(tmp_class)

    # Returns a class that is different from `class`
    def _different_class(self, class):
        distance = np.random.randint(nb_concepts) + 1
        return self._distance_to_class(class, distance)

    # Generates a batch
    # A batch is a stack of `BATCH_SIZE` instance
    # An instance is a stack of four images: alice_img, an image of the same class, an image of a neighbouring class (distance = 1) and an image of a different class (distance != 0)
    def _get_batch(self):
        batch = []
        for _ in range(BATCH_SIZE):
            alice_img = np.random.choice(self.dataset)
            bob_a = np.random.choice(self.classes[alice_img.class]) # same class
            bob_b = np.random.choice(self.classes[self._distance_to_class(alice_img.class, 1)]) # neighbouring class
            bob_c = np.random.choice(self.classes[self._different_class(alice_img.class)]) # different class

            l = [alice_img, bob_a, bob_b, bob_c]
            batch.append(torch.stack([x.img for x in l]))

        return torch.stack(batch)

    # Iterates over batches
    def __iter__(self):
        while True:
            yield self._get_batch()

def get_data_loader():
    return DistinctTargetClassDataLoader()
