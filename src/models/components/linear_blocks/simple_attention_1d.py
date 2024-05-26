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