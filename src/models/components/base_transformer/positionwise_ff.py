import torch 
from torch import nn

from src.models.components.utils.activation_type_mapping import ACTIVATION_TYPE_MAPPING

from src.data.components.collate import SingleForwardState


class Generator(nn.Module):
    def __init__(self, d_model: int, out_size: int):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(d_model, out_size),
            nn.LogSoftmax(dim=-1)
        )


    def forward(self, x: SingleForwardState) -> SingleForwardState:
        return SingleForwardState(
            sequences=self.projection(x.sequences),
            mask=x.mask
        )


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, activation_type: str = "relu"):
        super().__init__()

        d_ff = d_ff if d_ff is not None else d_model * 2

        if activation_type in ACTIVATION_TYPE_MAPPING.keys():
            self.act = ACTIVATION_TYPE_MAPPING[activation_type]
        else: 
            NotImplementedError(f"activation_type must be in <{list(ACTIVATION_TYPE_MAPPING.keys())}>")

        self.position_wise_ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            self.act(),
            nn.Linear(d_ff, d_model)
        )

        self.d_model = d_model
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # dropout(ReLU(x @ W1 + b1)) @ W2 + b2
        return self.position_wise_ff(x)