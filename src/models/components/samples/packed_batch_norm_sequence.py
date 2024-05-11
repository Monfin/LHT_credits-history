import torch
from torch import nn

class PackedBatchNormSequence(nn.BatchNorm1d):
    def forward(self, inputs: torch.Tensor) -> nn.utils.rnn.PackedSequence:
        return nn.utils.rnn.PackedSequence(
            data=super().forward(inputs.data),
            batch_sizes=inputs.batch_sizes,
            sorted_indices=inputs.sorted_indices
        )
    
