from torch import nn

from src.data.components.collate import SingleForwardState


class BaseTransformer(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module = None, merged_src: bool = True, use_decoder: bool = True):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.merged_src = merged_src
        self.use_decoder = use_decoder

    def forward(self, src: SingleForwardState, tgt: SingleForwardState = None) -> SingleForwardState:
        
        state = self.encoder(src)

        if self.use_decoder:
            assert self.merged_src == True and tgt is None, "tgt is None"

            if self.merged_src:
                tgt = src


            state = self.decoder(tgt, state)

        return state