
from torch.utils.data import Dataset

from torchvision.datasets.utils import download_url, extract_archive

import gdown

import pathlib
import pandas as pd

from typing import Sequence


class VelodromeDataset(Dataset):

    all_domains = {"D", "E", "G", "P"}
    domain_map = {
        "D": "Docetaxel",
        "E": "Erlotinib",
        "G": "Gemcitabine",
        "P": "Paclitaxel",
    }

    def __init__(self, root: str, in_distribution: bool, id_domains: Sequence[str], ood_domains: Sequence[str], download: bool = True):
        self.root = root
        self.in_distribution = in_distribution

        if download:
            self.download()

        self.id_domains = set(id_domains)
        self.ood_domains = set(ood_domains)

        assert self.id_domains ^ self.ood_domains == self.all_domains, \
            f"ID and OOD domains must be disjoint and cover all domains"

        self.data = pd.DataFrame()

        root = pathlib.Path(self.root) / "velodrome"

        for domain in self.all_domains:
            domain_name = self.domain_map[domain]
            pattern = f"{domain_name}_.."

            domain_data = pd.read_csv(domain_path / "data.csv")
            domain_data["domain"] = domain
            self.data = self.data.append(domain_data)

    def download(self):
        url = "https://drive.google.com/drive/folders/1_vAB3b4GizjffvbqC_q3VDjZti_qhSUl"
        path = pathlib.Path(self.root) / "velodrome"

        gdown.download_folder(
            url=url,
            output=str(path),
            use_cookies=False,
            quiet=False,
        )

