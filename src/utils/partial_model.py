import torch

from typing import Union


class PartialCheckpoint(torch.nn.Module):
    def __init__(self, ckpt: Union[str, dict], freeze: bool = True, weights_only: bool = True):
        super().__init__()

        ckpt = torch.load(ckpt, weights_only=weights_only) if isinstance(ckpt, str) else ckpt

        assert ckpt.get("state_dict"), "ckpt does not have 'state_dict'"

        