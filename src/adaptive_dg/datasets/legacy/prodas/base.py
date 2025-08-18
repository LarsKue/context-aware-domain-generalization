import pathlib
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

import numpy as np

import torchvision.transforms as transforms

from typing import Sequence

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



class ProDASImageFolder(Dataset):

    url: str
    dirname: str
    filename: str
    md5: str
    _remove_finished: bool = True

    all_domains = {"env_1", "env_2", "env_3", "env_4"}
    domain_map = {
        "env_1": "env_1",
        "env_2": "env_2",
        "env_3": "env_3",
        "env_4": "env_4",
    }
    input_shape = (3, 64, 64)


    def __init__(self, root: str, in_distribution: bool, id_domains: Sequence[str], ood_domains: Sequence[str], download: bool = True, normalize: bool = True):
        super().__init__()

        self.root = root
        self.in_distribution = in_distribution

        self.id_domains = set(id_domains)
        self.ood_domains = set(ood_domains)

        assert self.id_domains ^ self.ood_domains == self.all_domains, \
            f"ID and OOD domains must be disjoint and cover all domains"

        self.augment = transforms.Compose([
            transforms.RandomResizedCrop([224, 224], scale=(0.7, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
        ])

        if normalize:
            # use the default normalization from DomainBed
            # this is static between datasets, so take it with a grain of salt
            # usually it should be better to use BatchNorm
            self.augment.transforms.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )

        # contains the images
        self.images = []

        # contains labels
        self.labels = []
        # contains one-hot encoded domain labels
        self.domains = []

        self._domain_counts = []

        if self.in_distribution:
            domains = self.id_domains
        else:
            domains = self.ood_domains

        for domain_name in domains:
            
            if self.domain_map is not None:
                images_path = pathlib.Path(self.root) / self.dirname / (self.domain_map[domain_name] + "_x.npy")
                labels_path = pathlib.Path(self.root) / self.dirname / (self.domain_map[domain_name] + "_y.npy")
            else:
                images_path = pathlib.Path(self.root) / self.dirname / (domain_name + "_x.npy")
                labels_path = pathlib.Path(self.root) / self.dirname / (domain_name + "_y.npy")

            images_tmp = np.load(images_path)
            print(images_tmp.shape, images_path, labels_path)
            images_tmp = np.swapaxes(images_tmp, 2, 3)
            images_tmp = np.swapaxes(images_tmp, 1, 2)
            print(images_tmp.shape)
                
            self.images.append(images_tmp)
            self.labels.append(np.load(labels_path))
            
            if self.in_distribution:
                domain = sorted(list(self.id_domains)).index(domain_name)
                domain = torch.tensor(domain, dtype=torch.long)
                domain = F.one_hot(domain, num_classes=len(self.id_domains))
            else:
                domain = torch.zeros(len(self.id_domains), dtype=torch.long)
            self.domains.append(domain.expand(len(self.images[-1]), -1))
            

            if domain_name == 'env_2':
                self.labels[-1] -= 1. 
            elif domain_name == 'env_3':
                self.labels[-1] -= 2.  
            elif domain_name == 'env_4':
                self.labels[-1] -= 3.

            if self._domain_counts:
                #self._domain_counts.append(len(self.domains) - sum(self._domain_counts))
                self._domain_counts.append(len(self.domains[-1]))
            else:
                self._domain_counts.append(len(self.domains[-1]))

        self.images = torch.from_numpy(np.concatenate(self.images))
        self.labels = torch.from_numpy(np.concatenate(self.labels)).view(-1,1)
        self.domains = torch.cat(self.domains)

        # Normalization
        #self.images = self.images.float() / 255.0
        #self.images = (self.images -self.images.mean(0) )/ self.images.std(0)
        #print(self.images.mean(0))

    def __getitem__(self, item):
        image = self.images[item]
        #image = self.augment(image)

        label = self.labels[item]
        domain = self.domains[item]

        return image, label, domain

    def __len__(self):
        return len(self.images)

    def domain_indices(self, domain: int) -> Sequence[int]:
        """ Return the indices pertaining to the subset of the dataset containing only the given domain. """
        domains = self.id_domains if self.in_distribution else self.ood_domains

        if isinstance(domain, int):
            if domain < 0:
                domain += len(domains)

            if not 0 <= domain < len(domains):
                id_ood = "ID" if self.in_distribution else "OOD"
                raise IndexError(f"Domain index {domain} out of range for {id_ood} dataset with {len(domains)} domains.")

        # compute indices from domain counts
        indices = list(range(sum(self._domain_counts[:domain]), sum(self._domain_counts[:domain + 1])))

        return indices

    def domain(self, domain: int) -> Dataset:
        """ Return a subset of the dataset containing only the given domain. """
        return Subset(self, self.domain_indices(domain))