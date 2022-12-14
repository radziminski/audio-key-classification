import gdown
import zipfile
from mega import Mega

mega = Mega()
m = mega.login()


def download(id, filename, download_type="google"):
    if download_type == "google":
        gdown.download(id=id, output=filename)

    if download_type == "mega":
        m.download_url(id, filename)

    if download_type == "url":
        # TODO: add logic for downloading file from url to a specific destination (filename)
        pass


def download_and_unzip(id, filename, destination, download_type="google"):
    download(id, filename, download_type=download_type)

    zip = zipfile.ZipFile(filename)
    zip.extractall(destination)
    zip.close()
