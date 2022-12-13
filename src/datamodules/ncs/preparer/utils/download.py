import gdown
import zipfile


def download_ncs_dataset(id, filename, destination):
    gdown.download(id=id, output=filename)
    zip = zipfile.ZipFile(filename)
    zip.extractall(destination)
    zip.close()
