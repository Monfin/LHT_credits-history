import torch
from torch import nn

import numpy as np

from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import BinaryAccuracy
from src.utils.metrics import GINI

from src.data.components.collate import ModelInput, ModelBatch, ModelOutput

from typing import Dict, Tuple, List

import lightning as L

import logging
log = logging.getLogger(__name__)


METRICS_MAPPING = {
    "gini": GINI,
    "accuracy": BinaryAccuracy
}


class TabularRNNLitModule(L.LightningModule):
    def __init__(
            self, 
            net: nn.Module,
            train_batch_size: int,
            val_batch_size: int,

            outputs_names: List[str],
            outputs_weights: List[float],

            metric_names: List[str],
            monitor_metric: str,

            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,

            compile: bool = False
        ) -> None:

        super(TabularRNNLitModule, self).__init__()
        self.save_hyperparameters(logger=False)

        self.monitor_metric = monitor_metric

        self.monitor_metric_name = self.monitor_metric.split("/")[-1].split("_")[0]
        # self.monitor_output_name = self.monitor_metric.split("/")[1]

        self.net = net

        self.criterion = nn.BCEWithLogitsLoss(
            reduction="none"
        )

        # if the nrof logis is less than outputs_names, then zip automatically reduces them
        self.outputs_names = outputs_names
        self.outputs_weights = torch.tensor(outputs_weights)

        self.metrics_names = metric_names

        # metric objects for calculating BinaryAUROC (~GINI) / Accuracy across batches
        self.train_metrics = nn.ModuleDict({
            output_name: nn.ModuleDict({
                metric_name: METRICS_MAPPING[metric_name]() for metric_name in self.metrics_names
            }) for output_name in self.outputs_names
        }) 
        self.val_metrics = nn.ModuleDict({
            output_name: nn.ModuleDict({
                metric_name: METRICS_MAPPING[metric_name]() for metric_name in self.metrics_names
            }) for output_name in self.outputs_names
        }) 
        self.test_metrics = nn.ModuleDict({
            output_name: nn.ModuleDict({
                metric_name: METRICS_MAPPING[metric_name]() for metric_name in self.metrics_names
            }) for output_name in self.outputs_names
        })
        
        self.monitor_metric = sum(
            [
                self.val_metrics[output_name][self.monitor_metric_name] for output_name in self.outputs_names
            ]
        ) / len(self.outputs_names)
        
        # Compositional metrics
        # for metric_name in self.metrics_names:
        #     self.train_metrics[f"avg_outputs_{metric_name}"] = / len(self.outputs_names)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.train_branched_loss = nn.ModuleDict({
            output_name: MeanMetric() for output_name in self.outputs_names
        })

        self.val_loss = MeanMetric()
        self.val_branched_loss = nn.ModuleDict({
            output_name: MeanMetric() for output_name in self.outputs_names
        })

        self.test_loss = MeanMetric()
        self.test_branched_loss = nn.ModuleDict({
            output_name: MeanMetric() for output_name in self.outputs_names
        })

        # for tracking best so far validation gini
        self.val_best_metric = MaxMetric()


    def forward(self, inputs: ModelInput) -> Dict[str, ModelOutput]:
        return self.net(inputs)
    

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.monitor_metric.reset()

        for output_name in self.outputs_names:
            self.val_best_metric[output_name].reset()

            for metric_name in self.metrics_names:
                self.val_metrics[output_name][metric_name].reset()


    def multioutput_loss(self, logits: ModelOutput, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # logits size is (batch_size, num_outputs)
        # targets size is (batch_size, 1)

        targets = targets.expand(size=(-1, len(self.outputs_names))) # to size like logits

        weighted_loss = self.outputs_weights * self.criterion(logits, targets)
        
        # (self.outputs_weights * self.criterion(logits, targets)).sum() / len(self.outputs_names)
        loss = weighted_loss.sum() / (len(weighted_loss) * len(self.outputs_names))
        branched_loss = (weighted_loss.sum(dim=0) / len(weighted_loss)).detach()

        return loss, branched_loss 
        
        
    def model_step(self, batch: ModelBatch) -> Tuple[torch.Tensor, ModelOutput]:
        x = ModelInput(
            categorical=batch.categorical,
            mask=batch.mask,
            lengths=batch.lengths
        )

        labels = batch.targets

        outputs = self.forward(x)
        
        loss, branched_loss = self.multioutput_loss(
            logits=outputs.logits,
            targets=labels
        )
    
        return loss, branched_loss, outputs.logits, labels.squeeze()


    def training_step(self, batch: ModelBatch, batch_idx: int) -> Dict:
        loss, branched_loss, outputs, labels = self.model_step(batch)

        self.train_loss(loss)
        self.log(
            "train/loss", 
            self.train_loss, 
            batch_size=self.hparams.train_batch_size,
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=False
        )

        for i, output_name in enumerate(self.outputs_names):
            self.train_branched_loss[output_name](branched_loss[i])

            self.log(
                f"train/loss_{output_name}",
                self.train_branched_loss[output_name],
                batch_size=self.hparams.train_batch_size,
                on_step=True, on_epoch=True, prog_bar=True, sync_dist=False
            )

            self.log(
                f"train/batch_cnt_ones_{output_name}",
                sum(labels == 1).type(torch.float32),
                batch_size=self.hparams.train_batch_size,
                on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
            )

            for metric_name in self.metrics_names:
                self.train_metrics[output_name][metric_name](outputs[:, i], labels)
                
                self.log(
                    f"train/{output_name}/{metric_name}", 
                    self.train_metrics[output_name][metric_name], 
                    batch_size=self.hparams.train_batch_size,
                    on_step=True, on_epoch=True, prog_bar=True, sync_dist=False
                )

        return {"loss": loss}


    def validation_step(self, batch: ModelBatch, batch_idx: int) -> Dict:
        loss, branched_loss, outputs, labels = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log(
            "val/loss", 
            self.val_loss, 
            batch_size=self.hparams.val_batch_size,
            on_step=False, on_epoch=True, prog_bar=False, sync_dist=True
        )

        for i, output_name in enumerate(self.outputs_names):
            self.val_branched_loss[output_name](branched_loss[i])

            self.log(
                f"val/loss_{output_name}",
                self.val_branched_loss[output_name],
                batch_size=self.hparams.val_batch_size,
                on_step=False, on_epoch=True, prog_bar=False, sync_dist=True
            )

            for metric_name in self.metrics_names:
                self.val_metrics[output_name][metric_name](outputs[:, i], labels)
                
                self.log(
                    f"val/{output_name}/{metric_name}", 
                    self.val_metrics[output_name][metric_name], 
                    batch_size=self.hparams.val_batch_size,
                    on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
                )

        return {"loss": loss}
    

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # update best so far val gini
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

        # self.val_best_metric(self.monitor_metric.compute())
        self.val_best_metric(self.monitor_metric.compute())

        # print("epoch end")

        self.log(
            f"val/{self.monitor_metric_name}_avg_best", 
            self.val_best_metric.compute(), 
            sync_dist=True, prog_bar=True
        )


    def test_step(self, batch: ModelBatch, batch_idx: int) -> Dict:
        loss, branched_loss, outputs, labels = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log(
            "test/loss", 
            self.test_loss, 
            batch_size=self.hparams.val_batch_size,
            on_step=False, on_epoch=True, prog_bar=False, sync_dist=True
        )

        for i, output_name in enumerate(self.outputs_names):
            self.test_branched_loss[output_name](branched_loss[i])

            self.log(
                f"test/loss_{output_name}",
                self.test_branched_loss[output_name],
                batch_size=self.hparams.val_batch_size,
                on_step=False, on_epoch=True, prog_bar=False, sync_dist=True
            )

            for metric_name in self.metrics_names:
                self.test_metrics[output_name][metric_name](outputs[:, i], labels)
                
                self.log(
                    f"test/{output_name}/{metric_name}", 
                    self.test_metrics[output_name][metric_name], 
                    batch_size=self.hparams.val_batch_size,
                    on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
                )

        return {"loss": loss}


    def setup(self, stage: str) -> None:
        """
        Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)


    def configure_optimizers(self) -> Dict:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        requires_grad_filter = lambda param: param.requires_grad
        optimizer = self.hparams.optimizer(params=filter(requires_grad_filter, self.parameters()))

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.monitor_metric,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        
        return {"optimizer": optimizer}