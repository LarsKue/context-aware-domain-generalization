
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from torchvision.datasets.utils import download_and_extract_archive

from .domain_generalization import DomainGeneralizationDataset


class BikeSharingDataset(DomainGeneralizationDataset):

    url = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
    dirname = "bike-sharing"
    filename = "hour.csv"

    def download(self) -> None:
        path = self.root / self.dirname
        if (path / self.filename).is_file():
            print(f"Found existing dataset in {path}. Skipping download.")
            return

        download_and_extract_archive(self.url, path, path, remove_finished=True)


class BikeSharingSeason(BikeSharingDataset):
    all_domains = ["fall", "spring", "summer", "winter"]

    def __init__(self, root: str | Path, download: bool = True):
        super().__init__(root, download)
        df = pd.read_csv(self.root / self.dirname / self.filename)
        df = df.drop(columns=["instant", "dteday", "mnth", "casual", "registered"])
        df = df.dropna()
        df = df.sort_values("season")

        targets = df.pop("cnt").astype(np.float32)

        domains = df.pop("season")
        domains = pd.get_dummies(domains, dtype=np.float32)

        self.data = torch.from_numpy(df.to_numpy().astype(np.float32))
        self.targets = torch.sqrt(torch.from_numpy(targets.to_numpy()).view(-1, 1))
        self.domains = torch.from_numpy(domains.to_numpy())

    def domain_indices(self, domain: str) -> list[int]:
        domain_index = self.all_domains.index(domain)

        return torch.nonzero(self.domains[:, domain_index]).squeeze().tolist()

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return len(self.data)
