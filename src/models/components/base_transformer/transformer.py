from torch import nn

from src.data.components.collate import SingleForwardState


class BaseTransformer(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, merged_src: bool = True):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.merged_src = merged_src

    def forward(self, src: SingleForwardState, tgt: SingleForwardState = None) -> SingleForwardState:

        assert self.merged_src == True and tgt is None, "tgt is None"

        if self.merged_src:
            tgt = src

        memory = self.encoder(src)

        state = self.decoder(tgt, memory)

        return state