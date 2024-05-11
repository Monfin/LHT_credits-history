import torch
from torch import nn

from src.data.components.collate import SingleForwardState
    

class SimpleAttention1d(nn.Module):
    def __init__(self, features_dim: int, use_batch_norm: bool = False):
        super(SimpleAttention1d, self).__init__()

        self.features_dim = features_dim

        if use_batch_norm:
            self.layer_norm = nn.BatchNorm1d
        else:
            self.layer_norm = nn.LayerNorm

        self.att_block = nn.Sequential(
            nn.Linear(in_features=self.features_dim, out_features=self.features_dim),
            self.layer_norm(self.features_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x: SingleForwardState) -> SingleForwardState:
        att_sequences = x.sequences * self.att_block(x.sequences)

        return SingleForwardState(
            sequences=att_sequences,
            lengths=x.lengths
        )


class PositionwiseFeedForward(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = None, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()

        hidden_size = hidden_size if hidden_size is not None else input_size * 2

        self.position_wise_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: SingleForwardState) -> SingleForwardState:
        # Norm(dropout(ReLU(x @ W1 + b1) @ W2 + b2))

        residual = x.sequences
        lengths = x.lengths

        x = self.dropout(self.position_wise_layer(x.sequences))
        
        x += residual

        x = self.layer_norm(x)

        return SingleForwardState(
            sequences=x,
            lengths=lengths
        )