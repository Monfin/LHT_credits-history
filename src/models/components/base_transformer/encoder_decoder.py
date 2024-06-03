import torch
from torch import nn

from src.models.components.utils.clone_modules import clone_modules
from src.data.components.collate import SingleForwardState


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model: int, max_len: int = 40, dropout: float = 0.1, encoding_capacity: float = 10_000.0):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        
        encoding_capacity = torch.tensor(encoding_capacity)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(-torch.arange(0, d_model, 2) * torch.log(encoding_capacity) / d_model)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)
        

    def forward(self, x: SingleForwardState) -> SingleForwardState:
        embeddings = x.sequences

        positioned_embs = embeddings + self.pe[:, :embeddings.size(1)]

        return SingleForwardState(
            sequences=self.dropout(positioned_embs),
            mask=x.mask
        )


class Encoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, n_layers: int = 1):
        super().__init__()

        self.layers = clone_modules(encoder_layer, n_layers)

        self.layer_norm = nn.LayerNorm(encoder_layer.d_model)

    def forward(self, x: SingleForwardState) -> SingleForwardState:

        for layer in self.layers:
            x = layer(x)
            
        return SingleForwardState(
            sequences=self.layer_norm(x.sequences),
            mask=x.mask
        )
    

class Decoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, n_layers: int = 1):
        super().__init__()

        self.layers = clone_modules(decoder_layer, n_layers)

        self.layer_norm = nn.LayerNorm(decoder_layer.d_model)

    def forward(self, x: SingleForwardState, memory: SingleForwardState) -> SingleForwardState:

        for layer in self.layers:
            x = layer(x, memory)
            
        return SingleForwardState(
            sequences=self.layer_norm(x.sequences),
            mask=x.mask
        )