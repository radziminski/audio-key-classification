import torchvision.datasets
import torch
import torchvision.transforms as transforms


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
                try:
                    tensor = torch.load(file).float()
                    return tensor
                except:
                    print("ERROR LOADING:")
                    print(path)

        normalize_transform = transforms.Normalize((0.78,), (0.65,))

        super(TorchDatasetFolder, self).__init__(
            root_dir,
            loader=loader,
            extensions=(".pt"),
            transform=transform if transform is not None else normalize_transform,
        )
