import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from pathlib import Path

from ..domain_generalization import DomainGeneralizationDataset

from .transforms import IdentityTransform


class MNISTDataset(DomainGeneralizationDataset):
    def __init__(self, root: str | Path, download: bool = True, transform: callable = None, target_transform: callable = None, domain_transforms: list[callable] = None, domain_lengths: list[int] = None):
        super().__init__(root, download=download)

        if transform is None:
            transform = transforms.ToTensor()

        if target_transform is None:
            target_transform = transforms.Lambda(lambda x: F.one_hot(torch.tensor(x), num_classes=10))

        mnist_train = MNIST(root=self.root, download=False, train=True, transform=transform, target_transform=target_transform)
        mnist_test = MNIST(root=self.root, download=False, train=False, transform=transform, target_transform=target_transform)
        self.mnist = ConcatDataset((mnist_train, mnist_test))

        self.domain_transforms = domain_transforms
        self.domain_lengths = domain_lengths
        if self.domain_lengths is None:
            self.domain_lengths = [len(self.mnist) // len(self.domain_transforms)] * len(self.domain_transforms)
            self.domain_lengths[-1] = len(self.mnist) - sum(self.domain_lengths[:-1])

        if not sum(self.domain_lengths) == len(self.mnist):
            sum_string = ' + '.join([str(l) for l in self.domain_lengths])
            raise ValueError(f"Sum of domain lengths must equal length of dataset, but got {sum_string} != {len(self.mnist)}")

    def download(self) -> None:
        if self.root.is_dir():
            print(f"Found existing {self.__class__.__name__} dataset in {self.root}, skipping download.")
            return

        MNIST(root=self.root, download=True)

    def domain_indices(self, domain: str) -> list[int]:
        domain_idx = self.all_domains.index(domain)
        start = sum(self.domain_lengths[:domain_idx])
        stop = sum(self.domain_lengths[:domain_idx + 1])

        return list(range(start, stop))

    def _domain_item(self, item: int) -> (int, int):
        domain = 0
        while item >= self.domain_lengths[domain]:
            item -= self.domain_lengths[domain]
            domain += 1
        return domain, item

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, item):
        img, label = self.mnist[item]

        domain, _ = self._domain_item(item)

        if self.domain_transforms is not None:
            transform = self.domain_transforms[domain]
            if transform is not None:
                img, label = transform(img, label)

        return img, label
