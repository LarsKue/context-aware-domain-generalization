
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from torchvision.datasets.utils import download_and_extract_archive

from .domain_generalization import DomainGeneralizationDataset


def convert_one_hot(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Convert a categorical column in a pd.DataFrame to one-hot encoding.
    """
    one_hot = pd.get_dummies(df[column], dtype=np.float32)
    one_hot = one_hot.add_prefix(f"{column}-")
    df = df.drop(column, axis=1)
    df = pd.concat([df, one_hot], axis=1)
    return df


class CensusIncomeDataset(DomainGeneralizationDataset):

    url = "https://archive.ics.uci.edu/static/public/2/adult.zip"
    dirname = "census-income"

    names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
             "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

    def download(self) -> None:
        path = self.root / self.dirname
        if path.is_dir() and any(path.iterdir()):
            print(f"Found existing dataset in {path}. Skipping download.")
            return

        download_and_extract_archive(self.url, path, path, remove_finished=True)

        # it is better to remove the spaces here because this allows the csv reader to use the C engine
        # which does not support multi-character separators (e.g. ", ")
        with open(path / "adult.data", "r") as f:
            content = f.read()
            content = content.replace(", ", ",")

        with open(path / "adult.data", "w") as f:
            f.write(content)

        with open(path / "adult.test", "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.replace(", ", ",")

                # the test file has a period at the end of (almost) each line
                if line.endswith(".\n"):
                    line = line[:-2] + "\n"

                lines[i] = line

        with open(path / "adult.test", "w") as f:
            f.writelines(lines)


class CensusIncomeEthnicity(CensusIncomeDataset):
    all_domains = ["Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"]

    def __init__(self, root: str | Path, download: bool = True):
        super().__init__(root, download)

        df_train = pd.read_csv(self.root / self.dirname / "adult.data", names=self.names)
        df_test = pd.read_csv(self.root / self.dirname / "adult.test", names=self.names, skiprows=1)

        df = pd.concat([df_train, df_test])
        df = df.drop(columns=["fnlwgt", "education-num"])
        df = df.dropna()
        df = df.sort_values("race")

        categorical_columns = ["education", "workclass", "marital-status", "occupation", "relationship", "sex",
                               "native-country"]
        for column in categorical_columns:
            df = convert_one_hot(df, column)

        self.df = df

        targets = pd.get_dummies(df.pop("income"), dtype=np.float32)
        domains = pd.get_dummies(df.pop("race"), dtype=np.float32)

        self.data = torch.from_numpy(df.to_numpy().astype(np.float32))
        self.targets = torch.from_numpy(targets.to_numpy())
        self.domains = torch.from_numpy(domains.to_numpy())

    def domain_indices(self, domain: str) -> list[int]:
        domain_index = self.all_domains.index(domain)

        return torch.nonzero(self.domains[:, domain_index]).squeeze().tolist()

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return len(self.data)


