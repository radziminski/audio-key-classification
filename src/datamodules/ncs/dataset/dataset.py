import torchvision.datasets
import audiofile
import torch
import scipy.signal


def audio_transform(sample, transform):
    audio, sr = sample
    if sr != 44100:
        number_of_samples = round(len(audio) * float(44100) / sr)
        audio = scipy.signal.resample(audio, number_of_samples)

    audio_tensor = torch.tensor(audio)

    if transform is not None and callable(transform):
        spectrogram = transform(audio_tensor)
        return spectrogram

    return audio


class NCSDataset(torchvision.datasets.DatasetFolder):
    def __init__(self, root_dir="./data/ncs", loader=audiofile.read, transform=None):
        self.root = root_dir

        dataset_transform = lambda x: audio_transform(x, transform)

        super(NCSDataset, self).__init__(
            root_dir, loader=loader, extensions=(".wav"), transform=dataset_transform
        )
