from torch import nn

from src.data.components.collate import SingleForwardState


class BaseTransformer(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module = None):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder


    def forward(self, src: SingleForwardState, tgt: SingleForwardState = None) -> SingleForwardState:
        
        state = self.encoder(src)

        if self.decoder is not None:
            assert tgt is None, "tgt is None"

            state = self.decoder(tgt, state)

        return state