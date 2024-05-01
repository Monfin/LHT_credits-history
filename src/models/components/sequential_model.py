from torch import nn

from src.data.components.collate import ModelInput, ModelOutput

from typing import List

class SequentialLitModel(nn.Module):
    def __init__(
            self, 
            layers: List[nn.Module],
            embedding_dim: int = 32
        ) -> None:
        super(SequentialLitModel, self).__init__()

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: ModelInput) -> ModelOutput:
        
        state = self.layers(inputs)

        return state