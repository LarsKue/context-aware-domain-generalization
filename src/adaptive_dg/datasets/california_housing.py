
import torch
from pathlib import Path

import numpy as np
import pandas as pd

from .domain_generalization import DomainGeneralizationDataset


class CaliforniaHousingDataset(DomainGeneralizationDataset):

    dirname = "california-housing-prices"
    filename = "housing.csv"

    all_domains = ["<1HR OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

    def __init__(self, root: str | Path, download: bool = True):
        super().__init__(root, download=download)

        df = pd.read_csv(self.root / self.dirname / self.filename)
        df = df.drop(columns=["longitude", "latitude"])
        df = df.dropna()
        df = df.sort_values("ocean_proximity")

        target = df.pop("median_house_value").astype(np.float32)

        domain = df.pop("ocean_proximity")
        domain = pd.get_dummies(domain, dtype=np.float32)

        self.data = torch.from_numpy(df.to_numpy().astype(np.float32))
        self.targets = torch.from_numpy(target.to_numpy())
        self.domains = torch.from_numpy(domain.to_numpy())

    def download(self):
        pass
        # TODO:
        # import kaggle
        # kaggle.api.datasets_download_file("camnugent", "california-housing-prices", "housing.csv")

    def domain_indices(self, domain: str) -> list[int]:
        domain_index = self.all_domains.index(domain)

        return torch.nonzero(self.domains[:, domain_index]).squeeze().tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]


