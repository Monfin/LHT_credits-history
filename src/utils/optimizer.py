import torch
from torch import nn 

class Optimizer:
    def __init__(self, learning_rate: float = 1e-3, scheduler_gamma: float = 0.99, scheduler_step_size: int = 10):
        self.criterion = nn.BCELoss()

        self.lr = learning_rate

        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma

    def optimizer(self, model_params) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            model_params, lr=self.lr
        )
    
    def scheduler(self, opt) -> torch.optim.lr_scheduler.StepLR:
        return torch.optim.lr_scheduler.StepLR(
            opt, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma
        )