import gdown
import zipfile
from mega import Mega
import urllib.request


def download(id, filename, download_type="google"):
    if download_type == "google":
        gdown.download(id=id, output=filename)

    if download_type == "mega":
        mega = Mega()
        m = mega.login()
        m.download_url(id, filename)

    if download_type == "url":
        print("urlretrive", id, filename)
        urllib.request.urlretrieve(id, filename)


def download_and_unzip(id, filename, destination, download_type="google"):
    download(id, filename, download_type=download_type)

    zip = zipfile.ZipFile(filename)
    zip.extractall(destination)
    zip.close()
