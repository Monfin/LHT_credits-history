import torch
from torch import nn

from src.data.components.collate import ModelOutput, SingleForwardState, TwoBranchForwardState


class TBFS2SFS(nn.Module):
    def __init__(self):
        super(TBFS2SFS, self).__init__()

    def forward(self, x: TwoBranchForwardState) -> SingleForwardState:
        return SingleForwardState(
            sequences=x.main_sequences,
            mask=x.mask
        )
    

class SFS2ModelOutput(nn.Module):
    def __init__(self):
        super(SFS2ModelOutput, self).__init__()

    def forward(self, x: SingleForwardState) -> ModelOutput:
        return ModelOutput(
            representations=x.sequences,
            logits=None
        )