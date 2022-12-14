import os
import gdown

from src.utils.audio import split_to_intervals_in_dirs, try_delete_dir
from src.datamodules.common.preparer.preparer import Preparer
from src.utils.download import gdown_and_unzip
from .utils.sort import sort_files_to_dirs


class GS_MTGPreparer(Preparer):
    def __init__(
        self,
        data_dir="data/",
        root_dir="data/gs_mtg/",
        download=False,
        google_id="",
        annotations_google_id="",
        zip_filename="gs_mtg.zip",
        interval_length=20,
        annotations_filename="annotations.txt",
        split=False,
        extensions=[".wav", ".mp3"],
    ):
        self.data_dir = data_dir
        self.root_dir = root_dir
        self.download = download
        self.google_id = google_id
        self.annotations_google_id = annotations_google_id
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
            print("Downloading Giantsteps MTG dataset from google drive...")
            gdown_and_unzip(self.google_id, self.zip_filename, self.data_dir)
            annotations_path = os.path.join(self.root_dir, self.annotations_filename)
            gdown.download(id=self.annotations_google_id, output=annotations_path)
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
