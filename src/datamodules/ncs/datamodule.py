from typing import Any, Dict, Optional, Tuple
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset, random_split

from src.utils.hydra import instantiate_delayed


class NCSDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size=64,
        num_workers: int = 0,
        pin_memory: bool = False,
        root_dir="data/ncs/",
        train_dir="data/ncs/train/",
        test_dir="data/ncs/validation/",
        train_ratio=0.7,
        val_ratio=0.15,
        sr=44100,
        transform=None,
        preparers={},
        datasets={},
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False, ignore=["transform", "preparers", "datasets"]
        )
        self.transform = transform
        self.preparers = preparers
        self._parse_datasets(datasets)

        # Dataset info
        self.spectrogram_size = (1, 84, 1723)
        self.num_classes = 24

    def _parse_datasets(self, datasets_config):
        self.train_datasets_configs = list(datasets_config.train.values())
        self.test_datasets_configs = list(datasets_config.test.values())

    def prepare_data(self):
        for preparer in self.preparers.values():
            preparer.prepare()

    def setup(self, stage=None):
        # Train / Validation
        if stage == "fit" or stage is None:
            instantiated_datasets = [
                instantiate_delayed(config) for config in self.train_datasets_configs
            ]

            dataset = ConcatDataset(instantiated_datasets)
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
            instantiated_datasets = [
                instantiate_delayed(config) for config in self.test_datasets_configs
            ]

            dataset = ConcatDataset(instantiated_datasets)
            self.test_dataset = dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size)
