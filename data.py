import itertools as it

import torch, torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import *

class SingleClassDataLoader():
    def __init__(self) :
        self.target_dataset = ImageFolder(
            root=DATASET_PATH,
            transform=torchvision.transforms.ToTensor())
        self.target_loader = DataLoader(
            self.target_dataset,
            batch_size=BATCH_SIZE,
            num_workers=1,
            shuffle=True
        )
        self.distractor_dataset = ImageFolder(
            root=DATASET_PATH,
            transform=torchvision.transforms.ToTensor())
        self.distractor_loader = DataLoader(
            self.distractor_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=True
        )

    def _get_examples(self):
        def _fst_valid(d_class, forbidden, examples_filled):
            for i in forbidden:
                if i in examples_filled or d_class == forbidden[i]: continue
                return i
            return None
        distractors_iterator = (d
            for _ in it.count()
            for d in iter(self.distractor_loader))
        for examples, classes in self.target_loader:
            # load distractors
            batch_size = classes.size(0)
            distractors = [[] for i in range(batch_size)]
            forbidden = {i:c for i,c in enumerate(classes)}
            filled = set()
            while True:
                distractor, d_class = next(distractors_iterator)
                fst_valid = _fst_valid(d_class, forbidden, filled)
                if fst_valid is None: continue
                distractors[fst_valid].append(distractor.squeeze(0))
                if len(distractors[fst_valid]) == K - 1: filled.add(fst_valid)
                if len(filled) == batch_size: break
            yield torch.cat([
                examples.unsqueeze(1),
                torch.stack([
                    torch.stack(d)
                    for d in distractors])],
                dim=1,)

    def __iter__(self):
        """cycling iterator"""
        while True:
            for example in self._get_examples():
                yield example

def get_dataloader():
    return SingleClassDataLoader()
