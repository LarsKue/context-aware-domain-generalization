import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset, Subset

import torchvision.transforms as transforms

import wilds

from typing import Sequence


# TODO: make one parent class maybe


class FMoWRegionDataset(Dataset):

    num_classes = ...

    def __init__(self, root: str, in_distribution: bool, id_domains: Sequence[int], ood_domains: Sequence[int], download: bool = True, normalize: bool = True):
        super().__init__()

        self.wilds = wilds.get_dataset("fmow", root_dir=root, download=download)
        self.in_distribution = in_distribution

        self.id_domains = set(id_domains)
        self.ood_domains = set(ood_domains)

        expected_domains = set(range(6))
        assert self.id_domains ^ self.ood_domains == expected_domains, f"ID and OOD domains must be disjoint and cover all domains. Expected {expected_domains} but got {self.id_domains} and {self.ood_domains}"

        if in_distribution:
            mask = torch.isin(self.wilds.metadata_array[:, 0], torch.as_tensor(list(self.id_domains)))
            indices = torch.nonzero(mask).squeeze()
            self.wilds = Subset(self.wilds, indices)
        else:
            mask = torch.isin(self.wilds.metadata_array[:, 0], torch.as_tensor(list(self.ood_domains)))
            indices = torch.nonzero(mask).squeeze()
            self.wilds = Subset(self.wilds, indices)

        domains = self.wilds.dataset.metadata_array[mask, 0].long()
        self._domain_counts = torch.bincount(domains)

        self.augment = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, item):
        image, label, meta_data = self.wilds[item]

        image = self.augment(image)

        label = F.one_hot(label.long(), num_classes=62)

        if self.in_distribution:
            domain = meta_data[0].item()

            domain = torch.tensor(list(self.id_domains).index(domain))

            domain = F.one_hot(domain, num_classes=len(self.id_domains))
        else:
            domain = torch.zeros(len(self.id_domains), dtype=torch.long)

        return image, label, domain

    def __len__(self):
        return len(self.wilds)

    def domain_indices(self, domain: int) -> Sequence[int]:
        domains = self.id_domains if self.in_distribution else self.ood_domains

        if isinstance(domain, int):
            if domain < 0:
                domain += len(domains)

            if not 0 <= domain < len(domains):
                id_ood = "ID" if self.in_distribution else "OOD"
                raise IndexError(
                    f"Domain index {domain} out of range for {id_ood} dataset with {len(domains)} domains.")

        # compute indices from domain counts
        indices = list(range(sum(self._domain_counts[:domain]), sum(self._domain_counts[:domain + 1])))

        return indices

    def domain(self, domain: int):
        return Subset(self, self.domain_indices(domain))


class FMoWYearDataset(Dataset):

    num_classes = ...

    def __init__(self, root: str, in_distribution: bool, id_domains: Sequence[int], ood_domains: Sequence[int], download: bool = True, normalize: bool = True):
        super().__init__()

        self.wilds = wilds.get_dataset("fmow", root_dir=root, download=download)
        self.in_distribution = in_distribution

        self.id_domains = set(id_domains)
        self.ood_domains = set(ood_domains)

        expected_domains = set(range(16))
        assert self.id_domains ^ self.ood_domains == expected_domains, f"ID and OOD domains must be disjoint and cover all domains. Expected {expected_domains} but got {self.id_domains} and {self.ood_domains}"

        if in_distribution:
            mask = torch.isin(self.wilds.metadata_array[:, 1], torch.as_tensor(list(self.id_domains)))
            indices = torch.nonzero(mask).squeeze()
            self.wilds = Subset(self.wilds, indices)
        else:
            mask = torch.isin(self.wilds.metadata_array[:, 1], torch.as_tensor(list(self.ood_domains)))
            indices = torch.nonzero(mask).squeeze()
            self.wilds = Subset(self.wilds, indices)

        domains = self.wilds.dataset.metadata_array[mask, 0].long()
        self._domain_counts = torch.bincount(domains)

        self.augment = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, item):
        image, label, meta_data = self.wilds[item]

        image = self.augment(image)

        label = F.one_hot(label.long(), num_classes=62)

        if self.in_distribution:
            domain = meta_data[1].item()

            domain = torch.tensor(list(self.id_domains).index(domain))

            domain = F.one_hot(domain, num_classes=len(self.id_domains))
        else:
            domain = torch.zeros(len(self.id_domains), dtype=torch.long)

        return image, label, domain

    def __len__(self):
        return len(self.wilds)

    def domain_indices(self, domain: int) -> Sequence[int]:
        domains = self.id_domains if self.in_distribution else self.ood_domains

        if isinstance(domain, int):
            if domain < 0:
                domain += len(domains)

            if not 0 <= domain < len(domains):
                id_ood = "ID" if self.in_distribution else "OOD"
                raise IndexError(
                    f"Domain index {domain} out of range for {id_ood} dataset with {len(domains)} domains.")

        # compute indices from domain counts
        indices = list(range(sum(self._domain_counts[:domain]), sum(self._domain_counts[:domain + 1])))

        return indices

    def domain(self, domain: int):
        return Subset(self, self.domain_indices(domain))
