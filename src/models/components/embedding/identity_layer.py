import torch
from torch import nn

from typing import List, Dict

from src.data.components.collate import ModelInput, SingleForwardState

from einops import rearrange


class IdentityEncoderLayer(nn.Module):
    def __init__(
            self,
            numerical_features: List[str],
            categorical_features: Dict[str, List[int]],
            embedding_dim: int = 16,
            dropout_inputs: float = 0.5,
            non_linear: bool = False,
            num_batch_norm: bool = True
        ) -> None:
        super(IdentityEncoderLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout_inputs)

        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

        if non_linear:
            self.out_linear_block = nn.Sequential(
                nn.Linear(
                    len(self.categorical_features) + len(self.numerical_features), 
                    embedding_dim
                ),
                nn.ReLU(),

                nn.Linear(embedding_dim, embedding_dim)
            )
        else:
            self.out_linear_block = nn.Linear(
                len(self.categorical_features) + len(self.numerical_features), 
                embedding_dim
            )

        if num_batch_norm:
            self.num_bn = nn.BatchNorm1d(len(self.numerical_features))
        else: 
            self.num_bn = None


    def forward(self, inputs: ModelInput) -> SingleForwardState:

        if self.num_bn is None:
            x_num = inputs.numerical
        else:
            x_num = rearrange(inputs.numerical, "N L H -> N H L")

            x_num = self.num_bn(x_num)

            x_num = rearrange(x_num, "N H L -> N L H")

        x = torch.concatenate((x_num, inputs.categorical), dim=-1)

        x = self.dropout(x)
        x = self.out_linear_block(x)

        return SingleForwardState(
            sequences=x, 
            mask=inputs.mask
        )