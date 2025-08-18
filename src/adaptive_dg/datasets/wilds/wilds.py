
import torch
import torch.nn.functional as F

import torchvision.transforms as transforms

from pathlib import Path
import wilds

from ..domain_generalization import DomainGeneralizationDataset


class WILDSDataset(DomainGeneralizationDataset):
    name: str = NotImplemented

    def __init__(self, root: str | Path, download: bool = True, domain_meta_index: int = 0):
        super().__init__(root, download=download)
        self.wilds = wilds.get_dataset(self.name, root_dir=self.root, download=False)
        self.domains = self.wilds.metadata_array[:, domain_meta_index].long()

        self.transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])

        self.target_transform = transforms.Lambda(lambda x: F.one_hot(x, num_classes=self.n_classes))

    def download(self) -> None:
        wilds.get_dataset(self.name, root_dir=self.root, download=True)

    def domain_indices(self, domain: str) -> list[int]:
        domain_idx = self.all_domains.index(domain)
        return torch.nonzero(self.domains == domain_idx).squeeze().tolist()

    def __len__(self) -> int:
        return len(self.wilds)

    def __getitem__(self, item) -> (torch.Tensor, torch.Tensor):
        img, label, meta = self.wilds[item]

        img = self.transform(img)
        label = self.target_transform(label)

        return img, label
