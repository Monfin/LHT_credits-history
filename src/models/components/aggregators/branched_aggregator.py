import torch
from torch import nn

from src.data.components.collate import SingleForwardState

from typing import List


class BranchedAggregator(nn.Module):
    def __init__(
            self,
            branches: List[nn.Module],
            embedding_dim: int = 16
    ) -> None:
        super().__init__()

        self.branches = nn.ModuleList(branches)

        self.out_block = nn.Linear(embedding_dim * len(self.branches), embedding_dim)


    def forward(self, state: SingleForwardState) -> SingleForwardState:
        branched_states = [
            branch(state).sequences for branch in self.branches
        ]

        sequences = torch.concatenate(branched_states, dim=-1)
        sequences = self.out_block(sequences)

        return SingleForwardState(
            sequences=sequences,
            mask=state.mask
        )