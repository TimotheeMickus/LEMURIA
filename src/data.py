import itertools as it

import torch, torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import *

class DistinctTargetClassDataLoader():
    def __init__(self) :
        # target images, batched by BATCH_SIZE
        self.target_dataset = ImageFolder(
            root=DATASET_PATH,
            transform=torchvision.transforms.ToTensor())
        self.target_loader = DataLoader(
            self.target_dataset,
            batch_size=BATCH_SIZE,
            num_workers=1,
            shuffle=True)
        # distractor images, that require to be matched with targets
        self.distractor_dataset = ImageFolder(
            root=DATASET_PATH,
            transform=torchvision.transforms.ToTensor())
        self.distractor_loader = DataLoader(
            self.distractor_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=True)

    def iter_distractors(self):
        """
        Infinite generator of distractors
        """
        while True:
            for distractor in self.distractor_loader:
                yield distractor

    def _get_examples(self):
        """
        Generator yielding examples constructed on the fly.
        Ensures no distractor has the same class as the target it is paired with
        """
        def _first_valid(distractor_class, target_classes, examples_filled):
            """
            Return the first target that can be matched with this distractor
            """
            for batch_item in target_classes:
                if distractor_class == target_classes[batch_item]:
                    # ignore this target if the distractor has the same class
                    continue
                if batch_item in examples_filled:
                    # ignore this target if it has already enough distractors
                    continue
                # return first match
                return batch_item
            # return no match
            return None
        distractors_generator = self.iter_distractors()
        for examples, classes in self.target_loader:
            # load distractors
            current_batch_size = classes.size(0)
            distractors = [[] for i in range(current_batch_size)] # will contain distractors
            target_classes = dict(enumerate(classes)) # holds information necessary to discard distractors
            examples_filled = set() # container for targets with enough distractors

            while len(examples_filled) != current_batch_size:
                # get next distractor
                distractor, distractor_class = next(distractors_generator)
                # match distractor with target
                first_valid_match = _first_valid(distractor_class, target_classes, examples_filled)
                if first_valid_match is None:
                    # no matching target for this distractor
                    continue

                # add where match
                distractors[first_valid_match].append(distractor.squeeze(0))

                if len(distractors[first_valid_match]) == K - 1:
                    # target has enough distractors
                    examples_filled.add(first_valid_match)

            # to tensors
            distractors = torch.stack([torch.stack(d) for d in distractors])
            batch = torch.cat([examples.unsqueeze(1), distractors], dim=1)
            yield batch

    def __iter__(self):
        """cycling iterator over examples"""
        while True:
            for example in self._get_examples():
                yield example

def get_dataloader():
    return DistinctTargetClassDataLoader()
