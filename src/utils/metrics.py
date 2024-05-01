from torchmetrics.classification import BinaryAUROC
import torch

class GINI(BinaryAUROC):
    def __init__(self) -> None:
        super(GINI, self).__init__()
        
    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return (2. * super().forward(y_true, y_pred) - 1.) * 100.