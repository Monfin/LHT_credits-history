import torch
from torch import nn

from src.models.components.utils.clone_modules import clone_modules

from src.data.components.collate import SingleForwardState


class SubLayer(nn.Module):
    """ Residual connection followed by a layer norm """
    def __init__(self, d_model: int, dropout: float = 0.5):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sub_layer: nn.Module) -> torch.Tensor:
        return x + self.dropout(sub_layer(self.layer_norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, attention: nn.Module, feed_forward: nn.Module, residual_dropout: float = 0.3):
        super().__init__()

        self.attn_layer = attention # z = x + dropout(attention(norm(x)))
        self.ff_layer = feed_forward # y = z + dropout(ff(norm(z)))

        self.d_model = self.attn_layer.d_model

        assert self.d_model == self.ff_layer.d_model, \
            f"The size <{self.d_model}> of attention layer must be the same as size <{self.ff_layer.d_model}> of feed forward layer"

        self.sub_layers = nn.ModuleDict(
            dict(
                zip(
                    ["attention", "feed_forward"],
                    clone_modules(SubLayer(self.d_model, residual_dropout), 2)
                )
            )
        )

    def forward(self, state: SingleForwardState) -> SingleForwardState:
        mask = state.mask
        x = state.sequences

        x = self.sub_layers["attention"](x, lambda x: self.attn_layer(x, x, x, mask))

        x = self.sub_layers["feed_forward"](x, self.ff_layer)

        return SingleForwardState(
            sequences=x,
            mask=mask
        )


class DecoderLayer(nn.Module):
    def __init__(self, src_attention: nn.Module, tgt_attention: nn.Module, feed_forward: nn.Module, residual_dropout: float = 0.3):
        super().__init__()

        self.tgt_attn_layer = tgt_attention
        self.src_attn_layer = src_attention
        self.ff_layer = feed_forward

        self.d_model = self.src_attn_layer.d_model

        assert self.d_model == self.ff_layer.d_model, \
            f"The size <{self.d_model}> of attention layer must be the same as size <{self.ff_layer.d_model}> of feed forward layer"

        self.sub_layers = nn.ModuleDict(
            dict(
                zip(
                    ["attention", "src_attention", "feed_forward"],
                    clone_modules(SubLayer(self.d_model, residual_dropout), 3)
                )
            )
        )

    def forward(self, x: SingleForwardState, memory: SingleForwardState) -> SingleForwardState:

        tgt_mask = x.mask
        x = x.sequences

        src_mask = memory.mask
        memory = memory.sequences

        x = self.sub_layers["attention"](x, lambda x: self.tgt_attn_layer(x, x, x, tgt_mask))

        x = self.sub_layers["src_attention"](x, lambda x: self.src_attn_layer(x, memory, memory, src_mask))

        x = self.sub_layers["feed_forward"](x, self.ff_layer)

        return SingleForwardState(
            sequences=x,
            mask=tgt_mask
        )