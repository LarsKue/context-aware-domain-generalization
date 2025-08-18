
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from torchvision.datasets.utils import download_and_extract_archive


from .domain_generalization import DomainGeneralizationDataset
from .census_income import convert_one_hot


class MLBookDataset(DomainGeneralizationDataset):
    url = "https://www.stats.ox.ac.uk/~snijders/mlbook2_r_dat.zip"
    dirname = "mlbook2_r_dat"
    filename = "mlbook2_r.dat"

    all_domains = [str(i) for i in range(211)]

    def __init__(self, root: Path | str, download: bool = True):
        super().__init__(root, download)
        df = pd.read_csv(self.root / self.dirname / self.filename, sep=r"\s+")
        df = df.drop(columns=["pupilNR_new", "sch_ses", "sch_iqv", "sch_min"])
        df = df.dropna()

        categorical_columns = ["denomina", "Minority", "sex"]

        for column in categorical_columns:
            df = convert_one_hot(df, column)

        domains = pd.get_dummies(df.pop("schoolnr"), dtype=np.float32).to_numpy()
        targets = df.pop("langPOST").to_numpy()

        data = df.to_numpy().astype(np.float32)

        self.data = torch.from_numpy(data)
        self.targets = torch.from_numpy(targets)
        self.domains = torch.from_numpy(domains)

    def download(self):
        path = self.root / self.dirname
        if path.is_dir() and any(path.iterdir()):
            print(f"Found existing dataset in {path}. Skipping download.")
            return

        download_and_extract_archive(self.url, path, path, remove_finished=True)

    def domain_indices(self, domain: str) -> list[int]:
        domain_index = self.all_domains.index(domain)

        return torch.nonzero(self.domains[:, domain_index]).squeeze().tolist()

    def __getitem__(self, item):
        return self.data[item], self.targets[item]
