from torch import nn
from copy import deepcopy


def clone_modules(module: nn.Module, N: int = 1):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])