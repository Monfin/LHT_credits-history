from torch import nn

from src.models.components.utils.clone_modules import clone_modules
from src.data.components.collate import SingleForwardState


class Encoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, n_layers: int):
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