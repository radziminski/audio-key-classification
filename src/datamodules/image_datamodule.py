import os

from src.datamodules.generic_datamodule import GenericDatamodule
from src.utils.download import download_and_unzip


class ImageDataModule(GenericDatamodule):
    def __init__(
            self,
            root_dir='./data',
            batch_size=64,
            num_workers: int = 0,
            pin_memory: bool = False,
            train_ratio=0.85,
            val_ratio=0.15,
            preparers=None,
            train_datasets=None,
            test_datasets=None,
            ncs_images_url='',
            gs_mtg_images_url='',
            gs_key_images_url='',
            images_dir='images',
            download=False
    ):
        super().__init__(batch_size,
                         num_workers,
                         pin_memory,
                         train_ratio,
                         val_ratio,
                         train_datasets,
                         test_datasets)
        self.preparers = preparers if preparers is not None else []

    def prepare_data(self):
        if self.hparams.download:
            self._prepare_data()

    def _prepare_data(self):
        if not os.path.exists(self.hparams.root_dir):
            os.mkdir(self.hparams.root_dir)

        if not os.path.exists(self.hparams.images_dir):
            os.mkdir(self.hparams.images_dir)

        download_and_unzip(self.hparams.ncs_images_url, os.path.join(self.hparams.images_dir, 'ncs_images.tar.xz'),
                           self.hparams.images_dir, 'url')
        download_and_unzip(self.hparams.gs_mtg_images_url, os.path.join(self.hparams.images_dir, 'gs_mtg.tar.xz'),
                           self.hparams.images_dir, 'url')
        download_and_unzip(self.hparams.gs_key_images_url, os.path.join(self.hparams.images_dir, 'gs_key.tar.xz'),
                           self.hparams.images_dir, 'url')
