
import torch
from pathlib import Path

from ..domain_generalization import DomainGeneralizationDataset


class Kang(DomainGeneralizationDataset):
    def __init__(self, root: str | Path, download: bool = True):
        super().__init__(root, download=download)
        # TODO: Initialize dataset and domain lengths

    def download(self) -> None:
        if self.root.is_dir():
            print(f"Found existing {self.__class__.__name__} at {self.root}, skipping download.")
            return

        # TODO: download and extract the dataset into self.root

        raise NotImplementedError

    def domain_indices(self, domain: str) -> list[int]:
        # TODO: return indices pertaining to the given (short) domain name
        raise NotImplementedError

    def __len__(self) -> int:
        # TODO: return length of underlying dataset
        raise NotImplementedError

    def __getitem__(self, item) -> (torch.Tensor, torch.Tensor):
        # TODO: return image tensor in size (3, 224, 224) and one-hot label tensor
        raise NotImplementedError
