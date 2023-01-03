from src.datamodules.common.generic_datamodule import GenericDatamodule
from src.utils.hydra import instantiate_delayed
import os
from torchvision.utils import save_image
import torch
from src.utils.audio import save_mp3_to_tensor


class AudioDataModule(GenericDatamodule):
    def __init__(
        self,
        batch_size=64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_ratio=0.85,
        val_ratio=0.15,
        sr=44100,
        interval_length=20,
        extensions=[],
        loader_type="torch",
        transform=None,
        preparers=None,
        train_datasets=None,
        test_datasets=None,
        images_preparers=None,
        images_dir="",
        torch_preparers=None,
        torch_dir="",
        audio_dir="",
        device="gpu",
    ):
        super().__init__(
            batch_size,
            num_workers,
            pin_memory,
            train_ratio,
            val_ratio,
            train_datasets,
            test_datasets,
        )
        self.preparers = preparers if preparers is not None else []
        self.images_preparers = images_preparers if images_preparers is not None else []
        self.images_dir = images_dir
        self.torch_preparers = torch_preparers if torch_preparers is not None else []
        self.torch_dir = torch_dir
        self.audio_dir = audio_dir
        self.sr = sr
        self.loader_type = loader_type
        self.device = "cuda" if device == "gpu" else device

    def prepare_data(self):
        print("Audio data module prepare start...")
        for preparer in self.preparers.values():
            preparer.prepare()
        print("Audio data module prepare finished.")

    def create_audio_tensors(self):
        if not os.path.exists(self.torch_dir):
            os.mkdir(self.torch_dir)

        files_num = 0
        for (
            root,
            dirs,
            files,
        ) in os.walk(self.audio_dir):
            destination_dir = root.replace(self.audio_dir, self.torch_dir)
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)

            for file in files:
                if file.endswith(".mp3") or file.endswith(".wav"):
                    source_path = os.path.join(root, file)
                    destination_path = os.path.join(destination_dir, file).replace(
                        ".mp3", ".pt"
                    )
                    save_mp3_to_tensor(
                        source_path,
                        destination_path,
                        self.sr,
                        self.loader_type,
                        self.device,
                    )

    def create_spectrograms(self):
        if not os.path.exists(self.images_dir):
            os.mkdir(self.images_dir)

        datasets = [
            *self.train_datasets_configs,
            *self.test_datasets_configs,
        ]
        for image_preparer in self.images_preparers.values():
            dataset_images_dir = image_preparer.images_dir

            if not os.path.exists(dataset_images_dir):
                os.makedirs(dataset_images_dir, exist_ok=True)

            dataset_name = image_preparer.dataset_name
            dataset_config = next(filter(lambda d: d["name"] == dataset_name, datasets))
            dataset = instantiate_delayed(dataset_config)
            idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
            for index, entry in enumerate(dataset):
                if entry is None or entry[0] is None:
                    continue

                sample, label = entry
                image = sample[0]
                key_dir = idx_to_class[label]
                filename = os.path.basename(dataset.samples[index][0][:-4]) + ".png"
                full_dir = os.path.join(dataset_images_dir, key_dir)
                full_path = os.path.join(full_dir, filename)

                if not os.path.exists(full_dir):
                    os.mkdir(full_dir)

                save_image(image, full_path)
