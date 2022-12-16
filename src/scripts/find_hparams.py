import logging

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typing import List, Optional, Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils


@utils.task_wrapper
def find_params(cfg: DictConfig) -> Tuple[dict, dict]:
    datamodule = None
    if cfg.datamodule_type == "image":
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule.image)
    if cfg.datamodule_type == "audio":
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule.audio)

    model: LightningModule = hydra.utils.instantiate(cfg.model)

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
    )
    tuner = trainer.tuner
    lr_finder = tuner.lr_find(model, datamodule)

    lr = lr_finder.suggestion()
    print(f"Best initial learning rate: {lr}")
    lr_finder.plot()

    tuner.scale_batch_size(model=model, datamodule=datamodule)
    print(f"Best batch size: {datamodule.batch_size}")


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    find_params(cfg)


if __name__ == "__main__":
    main()
