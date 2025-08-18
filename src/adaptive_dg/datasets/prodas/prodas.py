
import numpy as np
import torch
from torch.nn import functional as F
import torchvision.transforms as transforms
from pathlib import Path

from ..domain_generalization import DomainGeneralizationDataset


class ProDASCondSatisfied(DomainGeneralizationDataset):

    all_domains =['env_1', 'env_2', 'env_3', 'env_4']
    n_domains = 5
    n_classes = 1

    dirname = "prodas"

    def __init__(self, root: str | Path, download: bool = True):
        super().__init__(root, download=download)
         
        # contains the images
        self.data = []
        self.dirname = 'prodas'


        self.augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(360),
            transforms.ToTensor(),
        ])
        self.augment.transforms.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

        # contains labels
        self.targets = []

        # contains one-hot encoded domain labels
        self.domains = []

        self._domain_counts = []

        domains = ['env_1', 'env_2', 'env_3', 'env_4']

        for domain_num ,domain_name in enumerate(domains):


            images_path = Path(self.root) / self.dirname / (domain_name + "_x.npy")
            labels_path = Path(self.root) / self.dirname / (domain_name + "_y.npy")

            images_tmp = np.load(images_path)
            images_tmp = np.swapaxes(images_tmp, 2, 3)
            images_tmp = np.swapaxes(images_tmp, 1, 2)

            self.data.append(images_tmp)
            self.targets.append(np.load(labels_path))

            domain = torch.tensor(domain_num, dtype=torch.long)
            domain = F.one_hot(domain, num_classes=len(self.all_domains))
            self.domains.append(domain.expand(len(self.data[-1]), -1))

            if self._domain_counts:
                self._domain_counts.append(len(self.domains[-1]))
            else:
                self._domain_counts.append(len(self.domains[-1]))

        self.data = torch.from_numpy(np.concatenate(self.data))
        self.targets= torch.from_numpy(np.concatenate(self.targets)).view(-1,1)
        self.domains = torch.cat(self.domains)
        

    def download(self) -> None:
        if self.root.is_dir():
            print(f"Found existing {self.__class__.__name__} at {self.root}, skipping download.")
            return

        # TODO: download and extract the dataset into self.root

        raise NotImplementedError

    def domain_indices(self, domain: str) -> list[int]:
        domain_index = self.all_domains.index(domain)
        return torch.nonzero(self.domains[:, domain_index]).squeeze().tolist()

    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, item) -> (torch.Tensor, torch.Tensor):
        # We augment all the data
        return self.augment(self.data[item]), self.targets[item]
