
import torch
import torch.distributions as D

import pandas as pd
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt

from .domain_generalization import DomainGeneralizationDataset


class SimpsonDataset(DomainGeneralizationDataset):

    all_domains = ["0", "1", "2", "3", "4"]

    dirname = "simpson"
    filename = "data.csv"

    n_domains = 5
    n_samples = 10_000
    spacing = 2.0
    noise = 0.25
    noise_ratio = 6.0
    rotation_range = (45.0, 45.0)

    def __init__(self, root: str | Path, download: bool = True):
        super().__init__(root, download=download)
        df = pd.read_csv(self.root / self.dirname / self.filename)

        data = df["x"].astype(np.float32).to_numpy()
        data = data.reshape(-1, 1)
        self.data = torch.from_numpy(data)
        targets = df["y"].astype(np.float32).to_numpy()
        targets = targets.reshape(-1, 1)
        self.targets = torch.from_numpy(targets)
        domains = pd.get_dummies(df["domain"], dtype=np.float32)
        self.domains = torch.from_numpy(domains.to_numpy())

    def domain_indices(self, domain: str) -> list[int]:
        domain_index = self.all_domains.index(domain)

        return torch.nonzero(self.domains[:, domain_index]).squeeze().tolist()

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return len(self.data)

    def download(self):
        if (self.root / self.dirname / self.filename).is_file():
            print(f"Replacing existing dataset in {self.root}.")

        n_domains = self.n_domains
        n_samples = self.n_samples // self.n_domains
        spacing = self.spacing
        noise = self.noise

        means = torch.stack([
            torch.linspace(0.0, spacing * n_domains, n_domains),
            torch.linspace(spacing * n_domains, 0.0, n_domains),
        ], dim=1)

        stds = torch.Tensor([self.noise_ratio * noise, noise]).repeat(n_domains, 1)

        angles = torch.linspace(*self.rotation_range, n_domains)
        angles = torch.deg2rad(angles)

        rotations = torch.stack([
            torch.stack([torch.cos(angles), -torch.sin(angles)], dim=-1),
            torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)
        ], dim=-2)

        covariances = torch.matmul(torch.matmul(rotations, torch.diag_embed(stds)), rotations.transpose(-1, -2))

        distributions = [D.MultivariateNormal(means[i], covariances[i]) for i in range(n_domains)]
        samples = [d.sample(torch.Size((n_samples,))) for d in distributions]

        df = pd.DataFrame({
            "x": torch.cat(samples).numpy()[:, 0],
            "y": torch.cat(samples).numpy()[:, 1],
            "domain": torch.arange(n_domains).repeat_interleave(n_samples).numpy()
        })

        # for domain in range(n_domains):
        #     ddf = df[df["domain"] == domain]
        #     plt.scatter(ddf["x"], ddf["y"], label=f"Domain {domain}", s=5, alpha=0.5)
        #
        # plt.legend()
        # plt.show()
        path = self.root / self.dirname / self.filename
        path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(path, index=False)

