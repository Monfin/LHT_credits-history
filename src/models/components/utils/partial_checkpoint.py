import torch
from torch import nn

from typing import Union


class PartialCheckpoint(nn.Module):
    def __init__(self, target_model: nn.Module, ckpt_path: str = None, ckpt_prefix: str = "", strict: bool = True):
        super().__init__()

        self.partial_model = target_model

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            model_weights = ckpt["state_dict"]

            partial_state_dict = {
                key[len(ckpt_prefix) + 1:]: model_weights[key] for key in model_weights.keys() if key.startswith(ckpt_prefix) 
            }

            self.partial_model.load_state_dict(partial_state_dict, strict=strict)
        

    def forward(self, inputs: Union[torch.Tensor]) -> torch.Tensor:
        return self.partial_model(inputs)