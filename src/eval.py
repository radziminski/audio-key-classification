import logging

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typing import List, Tuple

import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils
from src.utils.eval import eval_on_full_songs

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    assert cfg.ckpt_path

    logging.getLogger("PIL").setLevel(logging.WARNING)

    datamodule = None
    if cfg.datamodule_type == "image":
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule.image)
    if cfg.datamodule_type == "audio":
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule.audio)
    if cfg.datamodule_type == "tensor":
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule.torch)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    if cfg.full:
        device = "cuda" if cfg.trainer.accelerator == "gpu" else device
        eval_on_full_songs(datamodule, model, cfg.ckpt_path, device)
    else:
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
