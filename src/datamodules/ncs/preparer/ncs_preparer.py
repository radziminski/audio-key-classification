import os

from .utils.create import create_ncs_dataset
from src.utils.audio import split_to_intervals_in_dirs
from src.datamodules.common.preparer.preparer import Preparer
from src.utils.download import download_and_unzip


class NCSPreparer(Preparer):
    def __init__(
        self,
        data_dir="data/",
        root_dir="data/ncs/",
        train_dir="data/ncs/train/",
        test_dir="data/ncs/validation/",
        train_ratio=0.85,
        download_type="google",
        download=False,
        download_id="",
        zip_filename="ncs.zip",
        create=False,
        interval_length=20,
        split=False,
        extensions=[".wav", ".mp3"],
    ):
        self.data_dir = data_dir
        self.root_dir = root_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_ratio = train_ratio
        self.download = download
        self.download_type = download_type
        self.download_id = download_id
        self.create = create
        self.interval_length = interval_length
        self.zip_filename = zip_filename
        self.split = split
        self.extensions = extensions

    def prepare(
        self,
    ):
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        if self.download:
            print(f"Downloading NCS dataset from {self.download_type}...")
            download_and_unzip(
                self.download_id,
                self.zip_filename,
                self.data_dir,
                download_type=self.download_type,
            )
        elif self.create:
            print("Creating NCS dataset from scratch...")
            create_ncs_dataset(
                self.root_dir, self.train_dir, self.test_dir, self.train_ratio
            )

        if (self.download or self.create) and not self.split:
            print(
                "Warning: you disabled splitting while creating.downloading the files. Model might not work properly."
            )

        if self.split:
            print("Splitting into intervals...")
            split_to_intervals_in_dirs(
                self.train_dir, self.interval_length, self.extensions
            )
            split_to_intervals_in_dirs(
                self.test_dir, self.interval_length, self.extensions
            )
            print("Splitting into intervals finished")
