import os.path

import gdown
import zipfile
from mega import Mega
import urllib.request
import tarfile
import subprocess
import progressbar


pbar = None


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


def download(id, filename, download_type="google"):
    print(f"Downloading from {id}...")
    if download_type == "google":
        gdown.download(id=id, output=filename)

    if download_type == "mega":
        mega = Mega()
        m = mega.login()
        m.download_url(id, filename)

    if download_type == "url":
        urllib.request.urlretrieve(id, filename, show_progress)


def download_and_unzip(id, filename, destination, download_type="google"):
    if os.path.exists(filename):
        print(f"{filename} already exists, skipping download")
    else:
        download(id, filename, download_type=download_type)

    decompress(filename, destination)


def download_and_unzip_parts(
    ids, filename_prefix, zip_filename, destination, download_type="google"
):
    for index, id in enumerate(ids):
        filename = filename_prefix + str(index)
        if os.path.exists(filename):
            print(f"{filename} already exists, skipping download")
        else:
            print(id)
            download(id, filename, download_type=download_type)

    join_zip(filename_prefix, zip_filename)
    decompress(zip_filename, destination)


def join_zip(filename_prefix, destination):
    command = f"cat {filename_prefix}* > {destination} && rm {filename_prefix}*"
    subprocess.run(
        command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def decompress(filename, destination):
    print(f"Decompressing {filename}...")
    if filename.endswith("tar.xz"):
        with tarfile.open(filename, "r:xz") as tar:
            tar.extractall(destination)
    else:
        zip = zipfile.ZipFile(filename)
        zip.extractall(destination)
        zip.close()
