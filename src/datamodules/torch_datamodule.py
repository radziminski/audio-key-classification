import os

from src.datamodules.common.generic_datamodule import GenericDatamodule
from src.utils.download import download_and_unzip_parts
import torch


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

                    audio = torch.load(source_path)
                    intervals = torch.split(
                        audio, split_size_or_sections=interval_samples
                    )

                    for index, interval in enumerate(intervals):
                        if len(interval) == interval_samples:
                            spectrogram = self.transform.to(self.device)(
                                interval.float()
                            )

                            interval_destination_path = f"{destination_path}_{index}.pt"
                            torch.save(spectrogram.clone(), interval_destination_path)
