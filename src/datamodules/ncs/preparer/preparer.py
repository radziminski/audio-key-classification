from .utils.download import download_ncs_dataset
from .utils.create import create_ncs_dataset
from src.utils.audio import split_to_intervals_in_dirs
from src.datamodules.common.preparer.preparer import Preparer


class NCSPreparer(Preparer):
    def __init__(
        self,
        root_dir="data/ncs/",
        train_dir="data/ncs/train/",
        test_dir="data/ncs/validation/",
        train_ratio=0.85,
        download=False,
        google_id="",
        create=False,
        interval_length=20,
    ):
        self.root_dir = root_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_ratio = train_ratio
        self.download = download
        self.google_id = google_id
        self.create = create
        self.interval_length = interval_length

    def prepare(
        self,
    ):
        if self.download:
            print("TODO: Download NCS!")
        elif self.create:
            create_ncs_dataset(
                self.root_dir, self.train_dir, self.test_dir, self.train_ratio
            )

        if self.download or self.create:
            split_to_intervals_in_dirs(self.train_dir, self.interval_length)
            split_to_intervals_in_dirs(self.test_dir, self.interval_length)
