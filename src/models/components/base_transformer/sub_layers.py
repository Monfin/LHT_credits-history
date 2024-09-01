import torch
from torch import nn

from src.models.components.utils.clone_modules import clone_modules

from src.data.components.collate import SingleForwardState, TwoBranchForwardState


class SubLayer(nn.Module):
    """ Residual connection followed by a layer norm """
    def __init__(self, d_model: int, dropout: float = 0.5):
        super(SubLayer, self).__init__()

        self.layer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor, sub_layer: nn.Module) -> torch.Tensor:
        return x + self.dropout(sub_layer(self.layer_norm(x)))
    

class LUNASubLayer(nn.Module):
    """ Residual connection followed by a layer norm """
    def __init__(self, d_model: int, main_dropout: float = 0.3, agg_dropout: float = 0.3):
        super(LUNASubLayer, self).__init__()

        self.main_layer_norm = nn.LayerNorm(d_model)
        self.agg_layer_norm = nn.LayerNorm(d_model)

        self.main_dropout = nn.Dropout(main_dropout)
        self.agg_dropout = nn.Dropout(agg_dropout)


    def forward(self, context: torch.Tensor, aggregates: torch.Tensor, sub_layer: nn.Module) -> torch.Tensor:

        _context, _aggregates = sub_layer(self.main_layer_norm(context), self.agg_layer_norm(aggregates))

        return context + self.main_dropout(_context), aggregates + self.agg_dropout(_aggregates)


class EncoderLayer(nn.Module):
    def __init__(self, attention: nn.Module, feed_forward: nn.Module, residual_dropout: float = 0.3):
        super(EncoderLayer, self).__init__()

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
        mask = state.mask.unsqueeze(dim=1).unsqueeze(dim=-1)
        context = state.sequences

        context = self.sub_layers["attention"](
            context, 
            lambda context: self.attn_layer(context, context, context, mask)
        )

        x = self.sub_layers["feed_forward"](context, self.ff_layer)

        return SingleForwardState(
            sequences=x,
            mask=state.mask
        )
    

class LUNAEncoderLayer(nn.Module):
    def __init__(self, luna: nn.Module, feed_forward: nn.Module = None, residual_dropout: float = 0.3):
        super(LUNAEncoderLayer, self).__init__()

        self.luna_layer = luna # z = x + dropout(attention(norm(x)))
        self.ff_layer = feed_forward # y = z + dropout(ff(norm(z)))

        self.d_model = self.luna_layer.d_model

        if self.ff_layer is not None:
            assert self.d_model == self.ff_layer.d_model, \
                f"The size <{self.d_model}> of attention layer must be the same as size <{self.ff_layer.d_model}> of feed forward layer"

            self.sub_layers = nn.ModuleDict(
                dict(
                    zip(
                        ["luna", "feed_forward"],
                        [LUNASubLayer(self.d_model, residual_dropout), SubLayer(self.d_model, residual_dropout)]
                    )
                )
            )
        else:
            self.sub_layers = nn.ModuleDict(
                {
                    "luna": LUNASubLayer(self.d_model, residual_dropout)
                    
                }
            )


    def forward(self, state: TwoBranchForwardState) -> TwoBranchForwardState:
        mask = state.mask.unsqueeze(dim=1).unsqueeze(dim=1)
        context = state.main_sequences
        aggregates = state.aggregates

        context, aggregates = self.sub_layers["luna"](
            context, 
            aggregates,
            lambda context, aggregates: self.luna_layer(context, context, context, aggregates, mask)
        )

        if self.ff_layer is not None:
            context = self.sub_layers["feed_forward"](context, self.ff_layer)

        return TwoBranchForwardState(
            main_sequences=context,
            aggregates=aggregates,
            mask=state.mask
        )


class DecoderLayer(nn.Module):
    def __init__(self, src_attention: nn.Module, tgt_attention: nn.Module, feed_forward: nn.Module, residual_dropout: float = 0.3):
        super(DecoderLayer, self).__init__()

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