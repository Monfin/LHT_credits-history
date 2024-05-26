import torch
from torch import nn

from src.data.components.collate import ModelOutput, SingleForwardState


class UseMainSeq(nn.Module):
    def __init__(self):
        super(UseMainSeq, self).__init__()

    def forward(self, x: SingleForwardState) -> ModelOutput:
        return ModelOutput(
            representations=x.sequences,
            logits=None
        )