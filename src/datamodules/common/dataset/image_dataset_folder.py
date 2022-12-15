import torchvision.datasets
import torchvision.transforms as transforms


class ImageDatasetFolder(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root_dir,
    ):
        print(root_dir)
        self.root = root_dir
        dataset_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.94025,), (2.21655,))])

        super(ImageDatasetFolder, self).__init__(
            root_dir,
            transform=dataset_transform,
        )
