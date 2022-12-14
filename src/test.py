import pyrootutils
import torch

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is an optional line at the top of each entry file
# that helps to make the environment more robust and convenient
#
# the main advantages are:
# - allows you to keep all entry files in "src/" without installing project as a package
# - makes paths and scripts always work no matter where is your current work dir
# - automatically loads environment variables from ".env" file if exists
#
# how it works:
# - the line above recursively searches for either ".git" or "pyproject.toml" in present
#   and parent dirs, to determine the project root dir
# - adds root dir to the PYTHONPATH (if `pythonpath=True`), so this file can be run from
#   any place without installing project as a package
# - sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
#   to make all paths always relative to the project root
# - loads environment variables from ".env" file in root dir (if `dotenv=True`)
#
# you can remove `pyrootutils.setup_root(...)` if you:
# 1. either install project as a package or move each entry file to the project root dir
# 2. simply remove PROJECT_ROOT variable from paths in "configs/paths/default.yaml"
# 3. always run entry files from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

import hydra
from omegaconf import DictConfig

from src import utils

log = utils.get_pylogger(__name__)


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    dm = hydra.utils.instantiate(cfg.datamodule.audio)
    dm.prepare_data()
    dm.setup()

    for dataset in dm.instantiated_datasets:
        log.debug(dataset.class_to_idx)

    print("Input (spectrogram) size: ", dm.spectrogram_size)

    print("Some train tensors: ")
    for index, entry in enumerate(dm.train_dataloader()):
        sample, label = entry
        print(sample[0].shape, label[0])
        if index > 4:
            break

    print("Some test tensors: ")
    for index, entry in enumerate(dm.test_dataloader()):
        sample, label = entry
        print(sample[0].shape, label[0])
        if index > 4:
            break


def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


# TEST
# mean = ([0.9629]) std = ([2.2975])

# TRAIN
# mean = ([0.9176]) std = ([2.1356])

# AVG
# mean = 0,94025 avg = 2,21655

if __name__ == "__main__":
    main()
