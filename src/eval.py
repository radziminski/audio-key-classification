import logging
import re
import subprocess
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
from src.utils.mirex import mirex_score, mirex_score_single
import torch

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    assert cfg.ckpt_path
    print(f'ckpt path: {cfg.ckpt_path}')

    logging.getLogger("PIL").setLevel(logging.WARNING)

    datamodule = None
    if cfg.datamodule_type == "image":
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule.image)
    if cfg.datamodule_type == "audio":
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule.audio)

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

    checkpoint = torch.load(cfg.ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])

    log.info("Starting testing!")
    # trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    datamodule.prepare_data()
    datamodule.setup(stage="test")

    for dataset in datamodule.instantiated_datasets:
        print(dataset.root)
        test_dataset(dataset, model)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


def test_dataset(dataset, model):
    previous_song_num = None
    current_batch_samples = []
    current_batch_labels = []
    losses_sum = 0
    songs_count = 0
    for index, sample in enumerate(dataset):
        full_sample_path = dataset.samples[index][0]
        image = sample[0]
        label = sample[1]
        logits = model(torch.stack([image], dim=0))
        print(logits.shape)
        # print(image.shape)
        print(f'label: {label}, pred:{torch.argmax(logits, dim=1)}')
        filepart = extract_filepart(full_sample_path)
        song_num = extract_song_num(filepart)
        # print(filepart)

        if previous_song_num is None:
            previous_song_num = song_num
            current_batch_samples.append(image)
            current_batch_labels.append(label)

        if song_num is previous_song_num:
            pass
            current_batch_samples.append(image)
            current_batch_labels.append(label)
        else:
            batch = torch.stack(current_batch_samples, dim=0), current_batch_labels
            # batch = current_batch_samples[0].reshape((1, 1, 120, -1)), current_batch_labels
            loss, _, _ = test_step(model, batch)
            # print(f'mirex_score: {loss}')
            losses_sum += loss
            songs_count += 1
            current_batch_samples = [image]
            current_batch_labels = [label]
            previous_song_num = song_num
    print(f'final loss: {losses_sum / songs_count}')


def extract_song_num(filename: str):
    match = re.match(r"^[^-]*", filename)
    return int(match.group(0))


def extract_filepart(full_path: str):
    match = re.search(r'^.*/([^/]*)$', full_path)
    return match.group(1)


def test_step(model, batch):
    # print(batch)
    # print(batch[0].shape)
    # print(batch)
    x, y = batch
    logits = model.forward(x)
    # print(f'logits: {logits}')
    # print(f'single logits argmax: {torch.argmax(logits, dim=1)}')
    # print(f'y: {y}')
    # logits = torch.nn.functional.softmax(logits, dim=1)
    logits = torch.sum(logits, dim=0)
    # print(f'summed logits: {logits}')
    prediction = torch.argmax(logits, dim=0)
    # print(f'prediction: {prediction.item()}, actual: {y[0]}')
    loss = mirex_score_single(prediction.item(), y[0])
    return loss, prediction, y


def _test_step(self, batch):
    x, y = batch
    logits = self.forward(x)
    preds = torch.argmax(logits, dim=1)
    loss = mirex.mirex_score(preds, y)
    return loss, preds, y


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
