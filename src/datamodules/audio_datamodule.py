from src.datamodules.common.generic_datamodule import GenericDatamodule
from src.utils.hydra import instantiate_delayed
import os
from torchvision.utils import save_image
import torch


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

    def prepare_data(self):
        print("Audio data module prepare start...")
        for preparer in self.preparers.values():
            preparer.prepare()
        print("Audio data module prepare finished.")

    def create_audio_tensors(self):
        if not os.path.exists(self.torch_dir):
            os.mkdir(self.torch_dir)

        datasets = [
            *self.train_datasets_configs,
            *self.test_datasets_configs,
        ]

        for torch_preparer in self.torch_preparers.values():
            dataset_torch_dir = torch_preparer.torch_dir

            if not os.path.exists(dataset_torch_dir):
                os.makedirs(dataset_torch_dir, exist_ok=True)

            dataset_name = torch_preparer.dataset_name
            dataset_config = next(filter(lambda d: d["name"] == dataset_name, datasets))
            dataset = instantiate_delayed(dataset_config)
            idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

            for index, entry in enumerate(dataset):
                sample, label = entry
                audio_tensor = sample[0].clone()
                key_dir = idx_to_class[label]
                filename = os.path.basename(dataset.samples[index][0][:-4]) + ".pt"
                full_dir = os.path.join(dataset_torch_dir, key_dir)
                full_path = os.path.join(full_dir, filename)

                if not os.path.exists(full_dir):
                    os.mkdir(full_dir)

                torch.save(audio_tensor, full_path)

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
                sample, label = entry
                image = sample[0]
                key_dir = idx_to_class[label]
                filename = os.path.basename(dataset.samples[index][0][:-4]) + ".png"
                full_dir = os.path.join(dataset_images_dir, key_dir)
                full_path = os.path.join(full_dir, filename)

                if not os.path.exists(full_dir):
                    os.mkdir(full_dir)

                save_image(image, full_path)
