from typing import Any, Dict, Optional, Tuple
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset, random_split

from src.utils.hydra import instantiate_delayed


class AudioDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size=64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_ratio=0.85,
        val_ratio=0.15,
        sr=44100,
        interval_length=20,
        extensions=[],
        loader_type="torch",
        transform=None,
        preparers=None,
        train_datasets=None,
        test_datasets=None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False, ignore=["preparers", "datasets", "transform"]
        )
        self.preparers = preparers if preparers is not None else []
        self._parse_datasets(train_datasets, test_datasets)

        # Dataset info
        self.num_classes = 24

        self.spectrogram_size = None
        if transform is not None:
            self.spectrogram_size = (
                1,
                84,  # transform.n_bins
                interval_length * sr // transform.hop_length + 1,
            )

    def _parse_datasets(self, train_datasets_configs, test_datasets_configs):

        self.train_datasets_configs = (
            list(train_datasets_configs.values())
            if train_datasets_configs is not None
            else []
        )
        self.test_datasets_configs = (
            list(test_datasets_configs.values())
            if test_datasets_configs is not None
            else []
        )

    def prepare_data(self):
        print("Audio data module prepare start...")
        for preparer in self.preparers.values():
            preparer.prepare()
        print("Audio data module prepare finished.")

    def setup(self, stage=None):
        print("Audio data module setup start...")

        # Train / Validation
        if stage == "fit" or stage is None:
            instantiated_datasets = [
                instantiate_delayed(config) for config in self.train_datasets_configs
            ]

            dataset = ConcatDataset(instantiated_datasets)
            print(f"Train dataset size: {len(dataset)}")

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
            print(f"Test dataset size: {len(dataset)}")

            self.test_dataset = dataset

        print("Audio data module setup finished.")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
