import hydra
from omegaconf import DictConfig, OmegaConf

import rootutils
import pickle

import torch

from src.data.components.collate import ModelBatch

from src.data.components.transforms import Compose

import lightning as L

from typing import Dict

from timeit import timeit


import logging
log = logging.getLogger(__name__)

rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("len", len)


class Inference(torch.nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        # set seed for random number generators in pytorch, numpy and python.random
        if cfg.get("seed"):
            L.seed_everything(cfg.seed, workers=True)

        log.info(f"Instantiating model <{cfg.model._target_}>")

        self.model: L.LightningModule = hydra.utils.instantiate(cfg.model)
        self.model.eval()
        self.model.freeze()

        if cfg.get("transforms"):
            self.transforms: Compose = hydra.utils.instantiate(cfg.transforms)
        else:
            self.transforms = None


    def forward(self, sample: ModelBatch) -> Dict[str, torch.Tensor]:
        sample = self.transforms(sample) if self.transforms else sample

        preds = self.model.predict_step(sample)

        return preds


@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:

    inference_module = Inference(cfg=cfg)

    with open("data/raw_sample.pickle", "rb") as inputs:
        sample = pickle.load(inputs)

    inference_time = timeit("inference_module(sample)", number=cfg.n_iters) / cfg.n_iters

    print(inference_time)



if __name__ == "__main__": main()