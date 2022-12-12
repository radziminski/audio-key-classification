from .create_utils import (
    download_ncs_songs,
    split_files_to_train_test,
    zip_dataset,
)


def create_ncs_dataset(root_dir, train_dir, test_dir, train_ratio):
    download_ncs_songs(root_dir)
    split_files_to_train_test(root_dir, train_dir, test_dir, train_ratio)
