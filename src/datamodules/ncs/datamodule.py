from typing import Any, Dict, Optional, Tuple
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from .utils.prepare import prepare_ncs_dataset
from .dataset.dataset import NCSDataset


class NCSDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size=64,
        num_workers: int = 0,
        pin_memory: bool = False,
        root_dir="data/ncs/",
        train_dir="train/",
        test_dir="validation/",
        train_ratio=0.7,
        val_ratio=0.15,
        create=False,
        download=False,
        google_id=None,
        interval_length=20,
        transform=None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["transform"])
        self.transform = transform

        # Dataset info
        self.spectrogram_size = (1, 84, 1723)
        self.num_classes = 24

    def prepare_data(self):
        prepare_ncs_dataset(
            root_dir=self.hparams.root_dir,
            train_dir=self.hparams.train_dir,
            test_dir=self.hparams.test_dir,
            train_ratio=self.hparams.train_ratio,
            download=self.hparams.download,
            create=self.hparams.create,
            interval_length=self.hparams.interval_length,
        )

    def setup(self, stage=None):
        # Train / Validation
        if stage == "fit" or stage is None:
            dataset = NCSDataset(
                root_dir=os.path.join(self.hparams.root_dir, self.hparams.train_dir),
                transform=self.transform,
            )
            self.train_val_dataset = dataset
            train_val_ratio = self.hparams.train_ratio / (
                self.hparams.train_ratio + self.hparams.val_ratio
            )
            train_dataset_size = int(len(dataset) * train_val_ratio)
            self.train_dataset, self.val_dataset = random_split(
                dataset, [train_dataset_size, len(dataset) - train_dataset_size]
            )

        # Test
        if stage == "test" or stage is None:
            dataset = NCSDataset(
                root_dir=os.path.join(self.hparams.root_dir, self.hparams.test_dir),
                transform=self.transform,
            )
            self.test_dataset = dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size)
