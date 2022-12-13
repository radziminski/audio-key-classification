import torchvision.datasets
from src.utils.audio import common_audio_loader, common_audio_transform


class AudioDatasetFolder(torchvision.datasets.DatasetFolder):
    def __init__(
        self, root_dir="./data/ncs", transform=None, sr=44100, extensions=("wav")
    ):
        self.root = root_dir

        dataset_transform = lambda x: common_audio_transform(x, transform, sr)

        super(AudioDatasetFolder, self).__init__(
            root_dir,
            loader=common_audio_loader,
            extensions=tuple(extensions),
            transform=dataset_transform,
        )
