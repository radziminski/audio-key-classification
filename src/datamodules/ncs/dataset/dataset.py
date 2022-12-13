import torchvision.datasets
import audiofile
import torch
from src.utils.audio import resample


def audio_transform(sample, transform, target_sr):
    audio, sr = sample

    # Resample audio to target_sr (44100) sample rate, so that all inputs have the same size
    if sr != target_sr:
        audio = resample(audio, sr, target_sr)

    audio_tensor = torch.tensor(audio)

    if transform is not None and callable(transform):
        spectrogram = transform(audio_tensor)
        return spectrogram

    return audio


class NCSDataset(torchvision.datasets.DatasetFolder):
    def __init__(
        self, root_dir="./data/ncs", loader=audiofile.read, transform=None, sr=44100
    ):
        self.root = root_dir

        dataset_transform = lambda x: audio_transform(x, transform, sr)

        super(NCSDataset, self).__init__(
            root_dir, loader=loader, extensions=(".wav"), transform=dataset_transform
        )
