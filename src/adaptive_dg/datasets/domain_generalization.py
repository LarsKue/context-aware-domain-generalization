
import torch
from torch.utils.data import Dataset, Subset
from pathlib import Path

from typing import Any


class DomainGeneralizationDataset(Dataset):
    """ Base class to construct the innermost domain datasets. """
    all_domains: list[str] = NotImplemented
    domain_map: dict[str, Any] = NotImplemented
    n_classes: int = NotImplemented

    def __init__(self, root: str | Path, download: bool = True):
        super().__init__()
        self.root = Path(root)

        if download:
            self.download()

    def download(self) -> None:
        raise NotImplementedError

    def domain(self, domain: str) -> Dataset:
        return Subset(self, self.domain_indices(domain))

    def domain_indices(self, domain: str) -> list[int]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, item) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError
