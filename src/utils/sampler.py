import torch
from torch.utils.data import Sampler

from typing import Union, Dict

import numpy as np

class SequentialSampler(Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class BalanceSampler(Sampler):
    def __init__(self, class_indexes: Dict, n_classes: int = 2, mode: str = "downsampling"):
        pass


class WeightedRandomSampler(Sampler):
    def __init__(
            self, 
            distribution: Union[np.ndarray, torch.Tensor], 
            n_samples: int = 10_000,
            replacement: bool = False
        ) -> None:

        self.distribution = distribution
        self.indexes = np.arange(len(self.distribution))

        self.n_samples = n_samples
        self.replace = replacement

    def __iter__(self):
        indexes = np.random.choice(
            a=self.indexes,
            size=self.n_samples,
            p=self.distribution,
            replace=self.replace
        )
        return iter(indexes)
    
    def __len__(self):
        return self.n_samples


class SamplerFactory:
    def __init__(self, targets: np.ndarray, n_samples: int = 10_000, replacement: bool = False):
        self.n_samples = n_samples
        self.replacement = replacement

        weights = np.asarray(
            [
                1. / sum(targets == label) for label in np.unique(targets)
            ]
        )

        self.distribution = weights[targets].squeeze()
        self.distribution = self.distribution / sum(self.distribution)

    def weighted_random_sampler(self):
        return WeightedRandomSampler(
            distribution=self.distribution,
            n_samples=self.n_samples,
            replacement=self.replacement
        )
