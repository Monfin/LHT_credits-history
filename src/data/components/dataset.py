from src.data.components.data_reader import DataReader

import torch

import numpy as np

from typing import Dict, Union, List


class CreditsHistoryDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            data: DataReader, 
            targets: Union[np.ndarray, torch.Tensor],
            indexes: Union[np.ndarray, torch.Tensor],
            features: Dict
        ) -> None:

        self.data = data
        self.targets = targets
        self.indexes = indexes

        self.numerical_features = features["numerical"]
        self.categorical_features = features["categorical"]

    def __contains__(self, idx: int) -> bool:
        return idx in range(len(self.indexes))

    def __getitem__(self, idx: int) -> dict:
        if self.__contains__(idx):
            idx = self.indexes[idx] # map int idx to idx of data

            item = self.data[idx]

            numerical = torch.as_tensor(
                [
                    item[feature] for feature in self.numerical_features
                ], dtype=torch.float32
            ).T
            categorical = torch.as_tensor(
                [
                    item[feature] for feature in self.categorical_features.keys()
                ], dtype=torch.long
            ).T

            return {
                    "numerical": numerical,
                    "categorical": categorical,
                    "target": self.targets[idx],
                    "length": categorical.size(0),
                    "sample_index": idx
            }
        
        else:
            raise KeyError(f"incorrect index: {idx}")
    
    def __len__(self) -> int:
        return len(self.indexes)