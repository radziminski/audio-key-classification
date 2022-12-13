import gdown
import zipfile


def gdown_and_unzip(id, filename, destination):
    gdown.download(id=id, output=filename)
    zip = zipfile.ZipFile(filename)
    zip.extractall(destination)
    zip.close()
