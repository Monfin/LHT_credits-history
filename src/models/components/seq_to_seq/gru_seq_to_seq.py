import torch
from torch import nn

from src.data.components.collate import SingleForwardState

    
class GRUSeqToSeq(nn.Module):
    def __init__(
            self,
            hidden_size: int, 
            num_layers_gru: int = 1,
            bidirectional: bool = False,
            dropout_gru: float = 0.0
    ) -> None:
        super(GRUSeqToSeq, self).__init__()

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers_gru,
            bidirectional=bidirectional,
            dropout=dropout_gru
        )

    def forward(self, x: SingleForwardState) -> SingleForwardState:

        packed_sequences = torch.nn.utils.rnn.pack_padded_sequence(
            input=x.sequences, 
            lengths=x.lengths, 
            batch_first=True, 
            enforce_sorted=False
        )

        state, _ = self.gru(packed_sequences)

        padded_state, _ = torch.nn.utils.rnn.pad_packed_sequence(state, batch_first=True)

        # unpacked_sequences = torch.nn.utils.rnn.unpack_sequence(packed_sequences=packed_sequences)

        return SingleForwardState(
            sequences=padded_state,
            lengths=x.lengths
        )