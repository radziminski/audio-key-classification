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
        loader: lambda *args: torch.load(args).float()

        super(TorchDatasetFolder, self).__init__(
            root_dir,
            loader=torch.load,
            extensions=(".pt"),
            transform=transform,
        )
