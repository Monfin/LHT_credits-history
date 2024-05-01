import hydra
from omegaconf import DictConfig, OmegaConf

import rootutils

import logging
from lightning import LightningModule

from typing import Tuple, Dict, Any, List

import lightning as L

import torch

log = logging.getLogger(__name__)

rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)

from src.models.tabnn import TabularRNNLitModule
from src.data.components.collate import ModelBatch

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


def compile(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

    
    log.info(f"Load model from ckpt... <{cfg.get("ckpt_path")}>")
    model = TabularRNNLitModule

    trained_model = model.load_from_checkpoint(
        checkpoint_path=cfg.get("ckpt_path"),
        map_location="cpu"
    )
    
    trained_model.eval()

    trained_model.freeze()

    sample = ModelBatch(
        categorical=torch.zeros(
            size=(1, cfg.data.collator.max_seq_len, len(cfg.data.features)), 
            dtype=torch.float
        ),
        targets=torch.zeros(
            size=(1, 1), 
            dtype=torch.float
        ),
        sample_indexes=[
            " " for _ in range(1)
        ],
        mask=torch.zeros(
            size=(1, ), 
            dtype=torch.bool
        )
    )
    
    trained_model.to_onnx(
        file_path="test/gru_all_poolings_model.onnx", 
        input_sample=sample,
        export_params=True
    )


@hydra.main(version_base=None, config_path="../configs", config_name="compile_model.yaml")
def main(cfg: DictConfig) -> None:
    compile(cfg)

    print("Done.")



if __name__ == "__main__": main()