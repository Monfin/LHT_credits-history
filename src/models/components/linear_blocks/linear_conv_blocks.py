import torch
from torch import nn
from typing import Optional, Union, List

from src.data.components.collate import ModelOutput


ACTIVATION_TYPE_MAPPING = {
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "relu": nn.ReLU
}

def init_linear_block_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(layer.bias)

class LinearBlock(nn.Module):
    def __init__(
            self, 
            in_features: int, 
            out_features: int = 1, 
            num_layers: int = 3, 
            dropout_rate: float = 0.0, 
            activation_type: str = "tanh",
            use_batch_norm: bool = False,
            bias: bool = True
        ) -> None:
        super(LinearBlock, self).__init__()

        self.in_features = in_features

        self.dropout = nn.Dropout(p=dropout_rate)

        if activation_type is None:
            self.act = ACTIVATION_TYPE_MAPPING["tanh"]
        elif activation_type in ACTIVATION_TYPE_MAPPING.keys():
            self.act = ACTIVATION_TYPE_MAPPING[activation_type]
        else: NotImplementedError(f"activation_type must be in <{list(ACTIVATION_TYPE_MAPPING.keys())}>")

        if use_batch_norm:
            self.layer_norm = nn.BatchNorm1d
        else:
            self.layer_norm = nn.LayerNorm

        self.linear_block = nn.Sequential(
            *[
                nn.Sequential(
                    *[
                        nn.Linear(in_features // (2 ** i), in_features // (2 ** (i + 1)), bias),
                        self.act(),
                        self.layer_norm(in_features // (2 ** (i + 1)))
                    ]
                ) for i in range(num_layers)
            ]
        )
        

        self.out_block = nn.Linear(
            in_features=in_features // (2 ** num_layers), out_features=out_features
        )

        self.cls_layers = nn.Sequential(
            self.dropout,
            self.linear_block,
            self.out_block,
            self.act()
        )

        # weights init
        self.cls_layers.apply(init_linear_block_weights)


    def forward(self, x: ModelOutput) -> ModelOutput:
        logits = self.cls_layers(x.representations)

        return ModelOutput(
            representations=x.representations,
            logits=logits
        )


class MultiOutputLinearBlock(nn.Module):
    def __init__(
            self,
            heads: List[LinearBlock]
    ) -> None: 
        super(MultiOutputLinearBlock, self).__init__()

        self.heads = nn.ModuleList(heads)

    def forward(self, x: ModelOutput) -> ModelOutput:
        multi_state = [
            head(x).logits for head in self.heads
        ]

        logits = torch.concat(multi_state, dim=1) # size(batch_size, num_outputs)
        
        return ModelOutput(
            representations=x.representations,
            logits=logits
        )

    
class Conv1dBlock(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            num_layers: int = 3, 
            dropout_rate: float = 0.0, 
            activation_type: str = "tanh",
            padding: Optional[Union[str, int]] = "same"
        ) -> None:
        super(Conv1dBlock, self).__init__()

        self.dropout = nn.Dropout(p=dropout_rate)

        if activation_type is None:
            self.act = ACTIVATION_TYPE_MAPPING["tanh"]
        elif activation_type in ACTIVATION_TYPE_MAPPING.keys():
            self.act = ACTIVATION_TYPE_MAPPING[activation_type]
        else: NotImplementedError(f"activation_type must be in <{list(ACTIVATION_TYPE_MAPPING.keys())}>")

        self.conv_block = nn.Sequential(
            *[
                nn.Sequential(
                    *[
                        nn.Conv1d(
                            in_channels=in_channels // (2 ** i), 
                            out_channels=in_channels // (2 ** (i + 1)), 
                            kernel_size=3, 
                            padding=padding,
                            bias=False
                        ),
                        self.act(),
                        nn.BatchNorm1d(num_features=in_channels // (2 ** (i + 1)))
                    ]
                ) for i in range(num_layers)
            ]
        )

        self.out_block = nn.Conv1d(
            in_channels=in_channels // (2 ** num_layers),
            out_channels=out_channels,
            kernel_size=3,
            padding=padding
        )

        self.cls_layers = nn.Sequential(
            self.dropout,
            self.conv_block,
            self.out_block,
            self.act()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.cls_layers(inputs)