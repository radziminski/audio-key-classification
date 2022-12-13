import os

from src.utils.audio import split_to_intervals_in_dirs
from src.datamodules.common.preparer.preparer import Preparer


class GS_MTG_Preparer(Preparer):
    def __init__(
        self,
        data_dir="data/",
        root_dir="data/ncs/",
        train_ratio=0.85,
        download=False,
        google_id="",
        zip_filename="ncs.zip",
        interval_length=20,
        split=False,
        extensions=[".wav", ".mp3"],
    ):
        self.data_dir = data_dir
        self.root_dir = root_dir
        self.train_ratio = train_ratio
        self.download = download
        self.google_id = google_id
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
            print("Downloading NCS dataset from google drive...")

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
