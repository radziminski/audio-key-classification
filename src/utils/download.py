import os.path

import gdown
import zipfile
from mega import Mega
import urllib.request
import tarfile


def download(id, filename, download_type="google"):
    if download_type == "google":
        gdown.download(id=id, output=filename)

    if download_type == "mega":
        mega = Mega()
        m = mega.login()
        m.download_url(id, filename)

    if download_type == "url":
        urllib.request.urlretrieve(id, filename)


def download_and_unzip(id, filename, destination, download_type="google"):
    if os.path.exists(filename):
        print(f'{filename} already exists, skipping download')
    else:
        download(id, filename, download_type=download_type)

    decompress(filename, destination)


def decompress(filename, destination):
    print(f'Decompressing {filename}')
    if filename.endswith('tar.xz'):
        with tarfile.open(filename, "r:xz") as tar:
            tar.extractall(destination)
    else:
        zip = zipfile.ZipFile(filename)
        zip.extractall(destination)
        zip.close()
