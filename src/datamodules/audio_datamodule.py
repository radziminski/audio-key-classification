from src.datamodules.common.generic_datamodule import GenericDatamodule


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

    def prepare_data(self):
        print("Audio data module prepare start...")
        for preparer in self.preparers.values():
            preparer.prepare()
        print("Audio data module prepare finished.")
