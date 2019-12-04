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

    def _get_batch(self):
        """
        Generator yielding a batch (target + list of distractors) constructed on the fly.
        Ensures no distractor has the same class as the target it is paired with
        """
        def match_with_target(distractor_class, target_classes, examples_filled):
            """
            Returns the index of the first target that can be matched with this distractor, or None
            """
            for target_idx, target_class in enumerate(target_classes):
                if distractor_class == target_class:
                    # ignores this target because the distractor is of the same class
                    continue
                if target_idx in examples_filled:
                    # ignores this target because it has already enough distractors
                    continue

                # returns first match
                return target_idx
                
            # returns no match
            return None

        distractors_generator = self.iter_distractors()
        for targets, target_classes in self.target_loader: # a tensor of images and the tensor of corresponding classes
            current_batch_size = target_classes.size(0)

            distractors = [[] for _ in range(current_batch_size)] # will contain the list of distractors for each target of the batch

            # instead of iterating over the targets and finding distractors for them, we iterate over potential distractors and assign them (when possible) to a target
            examples_filled = set() # will contain the index of the targets with enough distractors
            while len(examples_filled) != current_batch_size:
                # gets the next distractor
                distractor, distractor_class = next(distractors_generator)

                # tries to match the distractor with a target
                match_idx = match_with_target(distractor_class, target_classes, examples_filled)
                if match_idx is None:
                    # no possible target for this distractor
                    continue

                # adds the distractor to its matched target
                distractors[match_idx].append(distractor.squeeze(0))

                if len(distractors[match_idx]) == (K - 1):
                    # the target has enough distractors
                    examples_filled.add(match_idx)

            # conversion to tensors
            distractors = torch.stack([torch.stack(d) for d in distractors])
            batch = torch.cat([targets.unsqueeze(1), distractors], dim=1)

            yield batch

    def __iter__(self):
        """cycling iterator over examples"""
        while True:
            for example in self._get_batch():
                yield example

def get_dataloader():
    return DistinctTargetClassDataLoader()
