import torch
from typing import Dict, List, Optional

from abc import ABC
from dataclasses import dataclass

from collections import namedtuple

# abstract class for model batches (data type)
# returns class with __init__, __repr__ and other
_model_batch = namedtuple("model_batch", ["numerical", "categorical", "mask", "sample_indexes", "targets"])
class ModelBatch(_model_batch):
    numerical: Optional[torch.Tensor]
    categorical: Optional[torch.Tensor]
    mask: Optional[torch.Tensor]
    sample_indexes: Optional[List]
    targets: Optional[torch.Tensor]

    def __new__(
        cls, 
        numerical: Optional[torch.Tensor],
        categorical: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        sample_indexes: Optional[List[str]] = None,
        targets: Optional[torch.Tensor] = None
    ): 
        return super().__new__(
            cls,
            numerical=numerical,
            categorical=categorical,
            mask=mask,
            sample_indexes=sample_indexes,
            targets=targets
    )


_model_input = namedtuple("model_input", ["numerical", "categorical", "mask"])
class ModelInput(_model_input):
    numerical: Optional[torch.Tensor]
    categorical: Optional[torch.Tensor]
    mask: Optional[torch.Tensor]

    def __new__(
        cls, 
        numerical: Optional[torch.Tensor],
        categorical: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ): 
        return super().__new__(
            cls,
            numerical=numerical,
            categorical=categorical,
            mask=mask
    )


@dataclass
class ModelOutput:
    representations: Optional[torch.Tensor]
    logits: Optional[torch.Tensor]


@dataclass
class SingleForwardState:
    sequences: Optional[torch.Tensor]
    mask: Optional[torch.Tensor]


@dataclass
class TwoBranchForwardState:
    main_sequences: Optional[torch.Tensor]
    aggregates: Optional[torch.Tensor]
    mask: Optional[torch.Tensor]


class Collator(ABC):
    def __call__(self, batch: List[Dict]) -> ModelBatch:
        raise NotImplementedError


class BaseCollator(Collator):
    def __init__(self, max_seq_len: int = None):
        self.max_seq_len = max_seq_len

    def get_padded_tensors(self, data: List[Dict], field: str):
        features = [item[field] for item in data]

        padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True) # size(batch_size, max_seq_len, num_features = num_sequences) 

        if self.max_seq_len is not None:
            padded_features = torch.nn.functional.pad(
                padded_features, (0, 0, 0, self.max_seq_len - padded_features.size(1))
            ) # size(batch_size, new max_seq_len, num_features)

        return padded_features

    def __call__(self, batch: List[Dict]) -> ModelBatch:
        lengths = torch.as_tensor([item["length"] for item in batch]).unsqueeze(dim=1)
        sample_indexes = [item["sample_index"] for item in batch]
 
        padded_numerical_features = self.get_padded_tensors(batch, "numerical")
        padded_categorical_features = self.get_padded_tensors(batch, "categorical")
        
        mask = torch.tile(torch.arange(self.max_seq_len), (lengths.size(0), 1)) >= lengths

        targets = torch.as_tensor([item["target"] for item in batch], dtype=torch.float32).unsqueeze(dim=-1)

        return ModelBatch(
            numerical=padded_numerical_features,
            categorical=padded_categorical_features, 
            mask=mask,
            sample_indexes=sample_indexes,
            targets=targets
        )

@dataclass
class InferenceBaseCollator(Collator):
    max_seq_len: int = 30

    def get_padded_tensors(self, data: List[Dict], field: str):
        features = [item[field] for item in data]

        padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True) # size(batch_size, max_seq_len, num_features = num_sequences) 

        if self.max_seq_len is not None:
            padded_features = torch.nn.functional.pad(
                padded_features, (0, 0, 0, self.max_seq_len - padded_features.size(1))
            ) # size(batch_size, new max_seq_len, num_features)

        return padded_features

    def __call__(self, batch: List[Dict]) -> ModelBatch:
        lengths = torch.as_tensor([item["length"] for item in batch]).unsqueeze(dim=1)
        sample_indexes = [item["sample_index"] for item in batch]

        padded_numerical_features = self.get_padded_tensors(batch, "numerical")
        padded_categorical_features = self.get_padded_tensors(batch, "categorical")
        
        mask = torch.tile(torch.arange(self.max_seq_len), (lengths.size(0), 1)) >= lengths

        return ModelBatch(
            numerical=padded_numerical_features,
            categorical=padded_categorical_features, 
            mask=mask,
            targets=None,
            sample_indexes=sample_indexes
        )