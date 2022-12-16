import subprocess

import pyrootutils
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import hydra
from omegaconf import DictConfig

from src import utils

log = utils.get_pylogger(__name__)


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    dm = hydra.utils.instantiate(cfg.datamodule.audio)
    # dm.prepare_data()
    # dm.setup()
    dm.prepare_images()
    return

    # for dataset in dm.instantiated_datasets:
    #     print(dataset.root)
    #     dataloader = build_dataloader(dataset)
    #     class_idx_map = dataset.class_to_idx
    #     save_dataloader(dataloader, class_idx_map)

    print("Some train tensors: ")
    for index, entry in enumerate(dm.train_dataloader()):
        sample, label = entry
        print(sample[0].shape, label[0])
        save_image(sample[0], f"{index}.png")
        if index > 4:
            break

    print("Some test tensors: ")
    for index, entry in enumerate(dm.test_dataloader()):
        sample, label = entry
        print(sample[0].shape, label[0])
        if index > 4:
            break


def save_dataloader(data_loader, class_idx_map):
    for index, entry in enumerate(data_loader):
        sample, label = entry
        class_name = find_class(label[0].item(), class_idx_map)
        class_path = f"data/images/ncs/validation/{class_name}"
        subprocess.run(
            f"mkdir -p {class_path}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        image_path = f"data/images/ncs/validation/{class_name}/{index}.png"
        print(image_path)
        save_image(sample[0], image_path)


def find_class(index, class_idx_map):
    for key, value in class_idx_map.items():
        if index is value:
            return key


def build_dataloader(dataset):
    return DataLoader(dataset, batch_size=1, shuffle=True)


def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5
    return mean, std


# TEST
# mean = ([0.9629]) std = ([2.2975])

# TRAIN
# mean = ([0.9176]) std = ([2.1356])

# AVG
# mean = 0,94025 avg = 2,21655

if __name__ == "__main__":
    main()
