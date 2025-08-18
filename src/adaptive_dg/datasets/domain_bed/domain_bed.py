import torch
import torch.nn.functional as F

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from pathlib import Path


from ..domain_generalization import DomainGeneralizationDataset
from .download import DomainBedDownloadMixin


class DomainBedDataset(DomainBedDownloadMixin, DomainGeneralizationDataset):
    name: str = NotImplemented

    def __init__(self, root: str | Path, download: bool = True):
        super().__init__(root, download=download)

        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])

        # transform = transforms.Compose([
        #     transforms.RandomResizedCrop([224, 224], scale=(0.7, 1.0), antialias=True),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        #     transforms.RandomGrayscale(),
        #     transforms.ToTensor(),
        # ])

        target_transform = transforms.Lambda(lambda x: F.one_hot(torch.tensor(x), num_classes=self.n_classes))

        self.image_folder = ImageFolder(
            root=str(self.root / self.name),
            transform=transform,
            target_transform=target_transform
        )

    def domain(self, domain: str) -> ImageFolder:
        domain = self.domain_map[domain]
        return ImageFolder(
            root=str(self.root / self.name / domain),
            transform=self.image_folder.transform,
            target_transform=self.image_folder.target_transform
        )

    def domain_indices(self, domain: str) -> list[int]:
        domain_idx = self.all_domains.index(domain)
        domain_lengths = [len(self.domain(domain)) for domain in self.all_domains]

        start = sum(domain_lengths[:domain_idx])
        stop = sum(domain_lengths[:domain_idx + 1])

        return list(range(start, stop))

    def __len__(self) -> int:
        return len(self.image_folder)

    def __getitem__(self, item) -> (torch.Tensor, torch.Tensor):
        return self.image_folder[item]
