import torchvision.datasets
from src.utils.audio import common_audio_loader, common_audio_transform


class AudioDatasetFolder(torchvision.datasets.DatasetFolder):
    def __init__(
        self,
        name="",
        root_dir="./data/ncs",
        transform=None,
        sr=44100,
        extensions=("wav"),
        device="gpu",
        interval_length=20,
        loader_type="torch",
    ):
        self.root = root_dir

        device = "cuda" if device == "gpu" else device

        target_length = interval_length * sr
        dataset_transform = lambda x: common_audio_transform(
            x, transform, sr, target_length, device
        )
        loader = lambda file: common_audio_loader(file, loader_type, device)

        super(AudioDatasetFolder, self).__init__(
            root_dir,
            loader=loader,
            extensions=tuple(extensions),
            transform=dataset_transform,
        )
