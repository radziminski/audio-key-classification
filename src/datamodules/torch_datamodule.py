import os

from src.datamodules.common.generic_datamodule import GenericDatamodule
from src.utils.download import download_and_unzip_parts
import torch
from torch_audiomentations import Compose
import re


class TorchDataModule(GenericDatamodule):
    def __init__(
        self,
        root_dir="./data",
        batch_size=64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_ratio=0.85,
        val_ratio=0.15,
        preparers=None,
        train_datasets=None,
        test_datasets=None,
        # ncs_images_url="",
        gs_mtg_torch_urls=[],
        ncs_torch_urls=[],
        gs_key_torch_urls=[],
        torch_dir="torch",
        download=False,
        process=False,
        torch_dir_processed="torch-processed",
        interval_length=30,
        sr=44100,
        transform=None,
        device="gpu",
        augmentations=[],
    ):
        super().__init__(
            batch_size,
            num_workers,
            pin_memory,
            train_ratio,
            val_ratio,
            train_datasets,
            test_datasets,
        )
        self.preparers = preparers if preparers is not None else []
        self.transform = transform
        self.device = "cuda" if device == "gpu" else device

    def prepare_data(self):
        if self.hparams.download:
            self._prepare_data()

        if self.hparams.process:
            self._process_data()

    def _prepare_data(self):
        if not os.path.exists(self.hparams.root_dir):
            os.mkdir(self.hparams.root_dir)

        if not os.path.exists(self.hparams.torch_dir):
            os.mkdir(self.hparams.torch_dir)

        download_and_unzip_parts(
            self.hparams.gs_key_torch_urls,
            os.path.join(self.hparams.torch_dir, "gs_key_"),
            os.path.join(self.hparams.torch_dir, "gs_key.zip"),
            self.hparams.torch_dir,
            "url",
        )

        download_and_unzip_parts(
            self.hparams.gs_mtg_torch_urls,
            os.path.join(self.hparams.torch_dir, "gs_mtg_"),
            os.path.join(self.hparams.torch_dir, "gs_mtg.zip"),
            self.hparams.torch_dir,
            "url",
        )

        download_and_unzip_parts(
            self.hparams.ncs_torch_urls,
            os.path.join(self.hparams.torch_dir, "ncs_"),
            os.path.join(self.hparams.torch_dir, "ncs.zip"),
            self.hparams.torch_dir,
            "url",
        )

    def _process_data(self):
        augmentations = self.hparams.augmentations
        interval_samples = self.hparams.interval_length * self.hparams.sr

        if not os.path.exists(self.hparams.torch_dir_processed):
            os.mkdir(self.hparams.torch_dir_processed)

        for (
            root,
            dirs,
            files,
        ) in os.walk(self.hparams.torch_dir):
            destination_dir = root.replace(
                self.hparams.torch_dir, self.hparams.torch_dir_processed
            )
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)

            for index, file in enumerate(files):
                if file.endswith(".pt"):
                    source_path = os.path.join(root, file)
                    destination_path = os.path.join(destination_dir, str(index))

                    audio = torch.load(source_path).to(self.device)
                    intervals = torch.split(
                        audio, split_size_or_sections=interval_samples
                    )

                    for index, interval in enumerate(intervals):
                        if len(interval) == interval_samples:
                            self.transform_and_save(destination_path, index, interval)
                            if not self.is_test_dir(root):
                                self.augment_interval(
                                    augmentations, destination_path, index, interval
                                )

    def augment_interval(self, augmentations, destination_path, index, interval):
        for a_index, augmentation in enumerate(augmentations):
            augment = Compose([augmentation]).to(self.device)
            interval = augment(interval.float())
            self.transform_and_save(
                destination_path + str(100 + a_index), index, interval
            )

    def transform_and_save(self, destination_path, index, interval):
        spectrogram = self.transform.to(self.device)(interval.float())
        interval_destination_path = f"{destination_path}_{index}.pt"
        torch.save(spectrogram.clone(), interval_destination_path)

    @staticmethod
    def format_augmentation_name(augmentation_raw_name):
        return re.sub(r"[^a-zA-Z0-9]", "", augmentation_raw_name).lower()

    @staticmethod
    def is_test_dir(root):
        test_dirs = ["gs_key", "ncs/validation"]
        for directory in test_dirs:
            if directory in root:
                return True
        return False

    def _get_mean_std(self, loader):
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0

        for data, _ in loader:
            channels_sum += torch.mean(data, dim=[0, 2, 3])
            channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
            num_batches += 1

        mean = channels_sum / num_batches
        std = (channels_squared_sum / num_batches - mean * 2) * 0.5
        return mean, std

    def print_train_mean_std(self):
        mean, std = self._get_mean_std(self.train_dataloader())
        print(f"Train mean={mean}; std={std};")

    def print_val_mean_std(self):
        mean, std = self._get_mean_std(self.val_dataloader())
        print(f"Val mean={mean}; std={std};")

    def print_test_mean_std(self):
        mean, std = self._get_mean_std(self.test_dataloader())
        print(f"Test mean={mean}; std={std};")
