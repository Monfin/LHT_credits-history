import hydra
from omegaconf import DictConfig, OmegaConf

import rootutils

import logging
from lightning import LightningDataModule

from typing import Tuple, Dict, Any, List

import lightning as L

import torch

log = logging.getLogger(__name__)

rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)

OmegaConf.register_new_resolver("eval", eval)
# OmegaConf.register_new_resolver("bool2int", lambda x: 2 if x else 1) # to bidirectional
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #


def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """

    torch.set_float32_matmul_precision("medium")

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Setup datamodule...")
    datamodule.setup()

    val_dataloader = datamodule.val_dataloader()

    samples = list()

    for batch in val_dataloader:
        samples.append(batch)

    print(len(samples))


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    train(cfg)

    print("Done.")



if __name__ == "__main__": main()