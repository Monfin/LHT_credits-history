import torch
from torch import nn

from src.models.components.utils.clone_modules import clone_modules
from src.data.components.collate import SingleForwardState, TwoBranchForwardState


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model: int, max_len: int = 40, dropout: float = 0.1, encoding_capacity: float = 10_000.0):
        super(PositionalEncoding, self).__init__()

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
        super(Encoder, self).__init__()

        self.layers = clone_modules(encoder_layer, n_layers)

        self.layer_norm = nn.LayerNorm(encoder_layer.d_model)


    def forward(self, x: SingleForwardState) -> SingleForwardState:
        
        for layer in self.layers:
            x = layer(x)
            
        return SingleForwardState(
            sequences=self.layer_norm(x.sequences),
            mask=x.mask
        )
    

class LUNAEncoder(nn.Module):
    def __init__(self, luna_encoder_layer: nn.Module, n_aggregates: int = 4, n_layers: int = 1):
        super(LUNAEncoder, self).__init__()

        self.layers = clone_modules(luna_encoder_layer, n_layers)

        self.main_layer_norm = nn.LayerNorm(luna_encoder_layer.d_model)
        self.agg_layer_norm = nn.LayerNorm(luna_encoder_layer.d_model)
        
        self.aggregates = nn.init.normal_(
            nn.Parameter(torch.Tensor(1, n_aggregates, luna_encoder_layer.d_model), requires_grad=True), 
            mean=0.0, 
            std=luna_encoder_layer.d_model ** (-0.5)
        )


    def forward(self, x: SingleForwardState) -> TwoBranchForwardState:
        aggregates = self.aggregates.expand(x.sequences.size(0), -1, -1)

        x = TwoBranchForwardState(
            main_sequences=x.sequences,
            aggregates=aggregates,
            mask=x.mask
        )

        for layer in self.layers:
            x = layer(x)
            
        return TwoBranchForwardState(
            main_sequences=self.main_layer_norm(x.main_sequences),
            aggregates=self.agg_layer_norm(x.aggregates),
            mask=x.mask
        )
    

class Decoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, n_layers: int = 1):
        super(Decoder, self).__init__()

        self.layers = clone_modules(decoder_layer, n_layers)

        self.layer_norm = nn.LayerNorm(decoder_layer.d_model)


    def forward(self, x: SingleForwardState, memory: SingleForwardState) -> SingleForwardState:

        for layer in self.layers:
            x = layer(x, memory)
            
        return SingleForwardState(
            sequences=self.layer_norm(x.sequences),
            mask=x.mask
        )