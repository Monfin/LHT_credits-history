from torchmetrics.classification import BinaryAUROC
import torch

# todo ~ failure with .compute()
class GINI(BinaryAUROC):
    def __init__(self) -> None:
        super(GINI, self).__init__()
        
    def compute(self) -> torch.Tensor:
        return (2 * super().compute() - 1) * 100.