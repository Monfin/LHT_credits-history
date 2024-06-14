import torch
from torch import nn

from src.data.components.collate import SingleForwardState


class GRUAggregator(nn.Module):
    def __init__(
            self,
            hidden_size: int, 
            num_layers_gru: int = 1,
            bidirectional: bool = False,
            dropout_gru: float = 0.0
    ) -> None:
        super(GRUAggregator, self).__init__()

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers_gru,
            bidirectional=bidirectional,
            dropout=dropout_gru
        )

    def forward(self, x: SingleForwardState) -> SingleForwardState:

        lengths = (~x.mask).sum(dim=1)

        packed_sequences = torch.nn.utils.rnn.pack_padded_sequence(
            input=x.sequences, 
            lengths=lengths, 
            batch_first=True, 
            enforce_sorted=False
        )

        _, hidden_state = self.gru(packed_sequences) # h ~ (num_layers, batch_size, hidden_size)

        return SingleForwardState(
            sequences=hidden_state[-1],
            mask=x.mask
        )