
import torchvision.transforms as transforms

from pathlib import Path

from .mnist import MNISTDataset


class RotatedMNIST(MNISTDataset):
    all_domains = ["0", "15", "30", "45", "60", "75"]
    domain_map = {"0": 0, "15": 15, "30": 30, "45": 45, "60": 60, "75": 75}
    n_classes = 10

    def __init__(self, root: str | Path, download: bool = True):
        angles = list(self.domain_map.values())
        domain_transforms = [transforms.RandomRotation([angle, angle]) for angle in angles]

        super().__init__(root, download=download, domain_transforms=domain_transforms)
