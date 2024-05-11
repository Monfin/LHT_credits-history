from lightning import LightningDataModule
import torch

import numpy as np

import hydra

# from torch.utils.data import random_split
from src.data.components.collate import BaseCollator

from typing import Optional, Any, Dict, Union

from src.utils.sampler import SamplerFactory

from src.data.components.data_reader import DataReader
from src.data.components.targets_indexes_reader import IndexesReader, TargetsReader

from src.data.components.dataset import CreditsHistoryDataset

import logging
log = logging.getLogger(__name__)


class LitDataModule(LightningDataModule):
    def __init__(
            self,
            data_reader: DataReader,
            features: Dict,

            collator: BaseCollator,

            indexes_reader: IndexesReader,

            targets_reader: TargetsReader,

            balance_sampler: bool = True,
            batch_sampler: bool = False,
            n_samples: int = 10_000,

            train_batch_size: int = 128,
            val_batch_size: int = 128,

            pin_memory: bool = False,
            num_workers: int = 0,
            persistent_workers: bool = False
    ) -> None:
        super(LitDataModule, self).__init__()
        
        self.save_hyperparameters()

        self.data_reader = hydra.utils.instantiate(data_reader)

        self.indexes_reader = hydra.utils.instantiate(indexes_reader)

        self.targets_reader = hydra.utils.instantiate(targets_reader)

        self.data_train: Optional[torch.utils.data.Dataset] = None
        self.data_val: Optional[torch.utils.data.Dataset] = None
        self.data_test: Optional[torch.utils.data.Dataset] = None

    @property
    def num_classes(self):
        return 1


    def prepare_data(self) -> None:
        """
        Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass


    def set_sampler(self, targets: np.ndarray = None, n_samples: int = 10_000, replacement: bool = False) -> torch.utils.data.Sampler:

        targets = np.asarray(list(targets.values()))

        if self.hparams.balance_sampler == "weighted":
            sampler_factory = SamplerFactory(
                targets=targets, 
                n_samples=n_samples,
                replacement=replacement
            )
            return sampler_factory.weighted_random_sampler()
        
        elif self.hparams.balance_sampler == "balanced":
            sampler_factory = SamplerFactory(
                targets=targets
            )
            return sampler_factory.balanced_random_sampler()

        else:
            return torch.utils.data.SequentialSampler(data_source=np.arange(len(targets)))
        

    def set_dataset(self, indexes: np.ndarray, targets: np.ndarray) -> CreditsHistoryDataset:
        return CreditsHistoryDataset(
            data=self.data_reader,
            targets=targets,
            indexes=indexes,
            features=self.hparams.features
        )


    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """

        # setup data
        self.data_reader.setup()

        targets = self.targets_reader.targets

        setup_data = lambda indexes, targets: CreditsHistoryDataset(
            data=self.data_reader,
            targets=targets,
            indexes=indexes,
            features=self.hparams.features
        )

        setup_sampler = lambda targets, n_samples, replacement: self.set_sampler(
            targets=targets,
            n_samples=n_samples,
            replacement=replacement
        )

        # TRAIN DATA
        if self.indexes_reader.train_path is not None:
            log.info("Setup train data...")

            train_indexes = self.indexes_reader.train_indexes
            train_targets = {idx: targets.get(idx) for idx in train_indexes}

            self.train_data = setup_data(indexes=train_indexes, targets=train_targets)

            log.info(f"Train data shape... <{len(self.train_data.indexes)}>")

            log.info(f"Setting train sampler...")
            self.train_sampler = setup_sampler(
                targets=self.train_data.targets, 
                n_samples=self.hparams.n_samples,
                replacement=True
            )
        else: 
            log.info("Skipping setup train data...")
            self.train_data = None
        
        # VALID DATA
        if self.indexes_reader.val_path is not None:
            log.info("Setup val data...")

            val_indexes = self.indexes_reader.val_indexes
            val_targets = {idx: targets.get(idx) for idx in val_indexes}

            self.val_data = setup_data(indexes=val_indexes, targets=val_targets)

            log.info(f"Val data shape... <{len(self.val_data.indexes)}>")
        else: 
            log.info("Skipping setup val data...")
            self.val_data = None


        # TEST DATA
        if self.indexes_reader.test_path is not None:
            log.info("Setup test data...")

            test_indexes = self.indexes_reader.test_indexes
            test_targets = {idx: targets.get(idx) for idx in test_indexes}

            self.test_data = setup_data(indexes=test_indexes, targets=test_targets)

            log.info(f"Test data shape... <{len(self.test_data.indexes)}>")
        else: 
            log.info("Skipping setup test data...")
            self.test_data = None


        if self.test_data is None:
            self.test_data = self.val_data


    def train_dataloader(self) -> torch.utils.data.DataLoader[Any]:
        """
        Create and return the train dataloader.

        :return: The train dataloader.
        """

        collator = hydra.utils.instantiate(self.hparams.collator)

        return torch.utils.data.DataLoader(
            dataset=self.train_data,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collator,
            sampler=self.train_sampler,
            persistent_workers=self.hparams.persistent_workers
        ) 


    def val_dataloader(self) -> torch.utils.data.DataLoader[Any]:
        """
        Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        collator = hydra.utils.instantiate(self.hparams.collator)

        return torch.utils.data.DataLoader(
            dataset=self.val_data,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collator,
            persistent_workers=self.hparams.persistent_workers
        ) 


    def test_dataloader(self) -> torch.utils.data.DataLoader[Any]:
        """
        Create and return the test dataloader.

        :return: The test dataloader.
        """
        collator = hydra.utils.instantiate(self.hparams.collator)

        return torch.utils.data.DataLoader(
            dataset=self.test_data,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collator,
            persistent_workers=self.hparams.persistent_workers
        ) 
    