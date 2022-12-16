from src.datamodules.common.generic_datamodule import GenericDatamodule
from src.utils.hydra import instantiate_delayed
import os
from torchvision.utils import save_image


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

    def prepare_data(self):
        print("Audio data module prepare start...")
        for preparer in self.preparers.values():
            preparer.prepare()
        print("Audio data module prepare finished.")

    def prepare_images(self):
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
                print(filename)
                full_dir = os.path.join(dataset_images_dir, key_dir)
                full_path = os.path.join(full_dir, filename)

                if not os.path.exists(full_dir):
                    os.mkdir(full_dir)

                save_image(image, full_path)
