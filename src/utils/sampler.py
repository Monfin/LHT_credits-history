import torch
from torch.utils.data import Sampler

from typing import Union, Dict, Iterator

import numpy as np

class SequentialSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class BalanceSampler(Sampler):
    def __init__(self, targets: np.array, class_indexes: Dict):
        self.targets = targets
        self.class_indexes = class_indexes

        self.samples_per_class = min([len(lbl2idx) for lbl2idx in self.class_indexes.values()])

        self.length = self.samples_per_class * len(set(self.targets))

    def __iter__(self) -> Iterator[int]:
        indexes = list()

        for label in sorted(self.class_indexes):
            replace_flag = self.samples_per_class > len(self.class_indexes[label])

            indexes = np.concatenate(
                (
                    indexes, 
                    np.random.choice(
                        self.class_indexes[label],
                        self.samples_per_class,
                        replace_flag
                    )
                )
            )

        indexes = indexes.astype(int).tolist()
        
        assert len(indexes) == self.__len__()
        np.random.shuffle(indexes)

        return iter(indexes)
    
    def __len__(self) -> int:
        return self.length


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

        self.targets = targets

        labels = set(self.targets)

        self.lbl2idx = {
            label: np.arange(len(targets))[targets == label] for label in labels
        }

    
    def balanced_random_sampler(self):
        return BalanceSampler(
            targets=self.targets,
            class_indexes=self.lbl2idx
        )


    def weighted_random_sampler(self):
        weights = np.asarray(
            [
                1. / len(one_lbl2idx) for one_lbl2idx in self.lbl2idx.values()
            ]
        )

        self.distribution = weights[self.targets].squeeze()
        self.distribution = self.distribution / sum(self.distribution)

        return WeightedRandomSampler(
            distribution=self.distribution,
            n_samples=self.n_samples,
            replacement=self.replacement
        )
