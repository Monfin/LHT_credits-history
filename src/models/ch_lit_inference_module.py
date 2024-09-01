import torch
from torch import nn

from src.data.components.collate import ModelInput, ModelBatch, ModelOutput

from typing import Dict, List

import lightning as L


class CHLitInferenceModule(L.LightningModule):
    def __init__(
            self, 
            net: nn.Module,
            compile: bool = False
        ) -> None:
        super().__init__()

        self.net = net


    def forward(self, inputs: ModelInput) -> ModelOutput:
        return self.net(inputs)
    

    def predict_step(self, inputs: ModelBatch) -> Dict[str, torch.Tensor]:
        x = ModelInput(
            numerical=inputs.numerical.type(self.dtype).to(self.device), 
            categorical=inputs.categorical.to(self.device), 
            mask=inputs.mask
        )
        preds = self.forward(x)

        return {
            "logits": preds.logits,
            "embeddings": preds.representations,
            "indexes": inputs.sample_indexes
        }