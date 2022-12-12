import os

from .download import download_ncs_dataset
from .create import create_ncs_dataset
from src.utils.audio import split_to_intervals_in_dirs


def prepare_ncs_dataset(
    root_dir="data/ncs/",
    train_dir="train/",
    test_dir="validation/",
    train_ratio=0.85,
    download=False,
    create=False,
    interval_length=20,
):
    if download:
        print("TODO: Download NCS!")
    elif create:
        create_ncs_dataset(root_dir, train_dir, test_dir, train_ratio)

    if download or create:
        full_train_dir = os.path.join(root_dir, train_dir)
        full_test_dir = os.path.join(root_dir, test_dir)
        split_to_intervals_in_dirs(full_train_dir, interval_length)
        split_to_intervals_in_dirs(full_test_dir, interval_length)
