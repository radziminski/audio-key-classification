import torchvision.datasets
import torch


class TorchDatasetFolder(torchvision.datasets.DatasetFolder):
    def __init__(
        self,
        name="",
        root_dir="./data/ncs",
        transform=None,
        device="gpu",
    ):
        self.root = root_dir

        device = "cuda" if device == "gpu" else device

        def loader(path):
            with open(path, "rb") as file:
                return torch.load(file).float()

        super(TorchDatasetFolder, self).__init__(
            root_dir,
            loader=loader,
            extensions=(".pt"),
            transform=transform,
        )
