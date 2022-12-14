import os
import gdown

from src.utils.audio import split_to_intervals_in_dirs, try_delete_dir
from src.datamodules.common.preparer.preparer import Preparer
from src.utils.download import download_and_unzip, download
from .utils.sort import sort_files_to_dirs


class GS_MTGPreparer(Preparer):
    def __init__(
        self,
        data_dir="data/",
        root_dir="data/gs_mtg/",
        download=False,
        download_type="google",
        download_id="",
        annotations_download_id="",
        zip_filename="gs_mtg.zip",
        interval_length=20,
        annotations_filename="annotations.txt",
        split=False,
        extensions=[".wav", ".mp3"],
    ):
        self.data_dir = data_dir
        self.root_dir = root_dir
        self.download_type = download_type
        self.download = download
        self.download_id = download_id
        self.annotations_download_id = annotations_download_id
        self.interval_length = interval_length
        self.zip_filename = zip_filename
        self.annotations_filename = annotations_filename
        self.split = split
        self.extensions = extensions

    def prepare(
        self,
    ):
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        if not os.path.isdir(self.root_dir):
            os.mkdir(self.root_dir)

        if self.download:
            print(f"Downloading Giantsteps MTG dataset from {self.download_type}...")
            download_and_unzip(
                self.download_id,
                self.zip_filename,
                self.data_dir,
                download_type=self.download_type,
            )
            annotations_path = os.path.join(self.root_dir, self.annotations_filename)
            download(
                self.annotations_download_id,
                annotations_path,
                download_type=self.download_type,
            )
            audio_path = os.path.join(self.data_dir, "audio")
            sort_files_to_dirs(annotations_path, audio_path, self.root_dir)
            # cleanup
            try_delete_dir(audio_path)

        if self.download and not self.split:
            print(
                "Warning: you disabled splitting while creating.downloading the files. Model might not work properly."
            )

        if self.split:
            print("Splitting into intervals...")
            split_to_intervals_in_dirs(
                self.root_dir, self.interval_length, self.extensions
            )
            print("Splitting into intervals finished")
