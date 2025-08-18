import pathlib
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url, extract_archive

from typing import Sequence

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DomainBedImageFolder(Dataset):

    url: str
    dirname: str
    filename: str
    md5: str
    _remove_finished: bool = True

    all_domains: set[str]
    domain_map: dict[str]
    n_classes: int

    def __init__(self, root: str, in_distribution: bool, id_domains: Sequence[str], ood_domains: Sequence[str], download: bool = True, normalize: bool = True):
        super().__init__()

        self.root = root
        self.in_distribution = in_distribution

        if download:
            self.download()

        self.id_domains = set(id_domains)
        self.ood_domains = set(ood_domains)

        assert self.id_domains ^ self.ood_domains == self.all_domains, \
            f"ID and OOD domains must be disjoint and cover all domains"

        self.augment_eval = transforms.Compose([
            transforms.RandomResizedCrop([224, 224], scale=(0.7, 1.0), antialias=True),
            transforms.ToTensor(),
        ])
        self.eval_mode = False

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
            self.augment_eval.transforms.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )

        # contains the image paths
        self.image_paths = []

        # contains one-hot encoded labels
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
                domain_path = pathlib.Path(self.root) / self.dirname / self.domain_map[domain_name]
            else:
                domain_path = pathlib.Path(self.root) / self.dirname / domain_name
            label_paths = sorted(list(domain_path.iterdir()))

            for label_idx, label_path in enumerate(sorted(domain_path.iterdir())):
                for image in label_path.iterdir():
                    label = torch.tensor(label_idx, dtype=torch.long)
                    label = F.one_hot(label, num_classes=len(label_paths))

                    if self.in_distribution:
                        domain = sorted(list(self.id_domains)).index(domain_name)
                        domain = torch.tensor(domain, dtype=torch.long)
                        domain = F.one_hot(domain, num_classes=len(self.id_domains))
                    else:
                        domain = torch.zeros(len(self.id_domains), dtype=torch.long)

                    self.image_paths.append(image)
                    self.labels.append(label)
                    self.domains.append(domain)

            if self._domain_counts:
                self._domain_counts.append(len(self.domains) - sum(self._domain_counts))
            else:
                self._domain_counts.append(len(self.domains))

    def __getitem__(self, item):
        image = self.image_paths[item]
        image = Image.open(image).convert("RGB")

        if self.eval_mode:
            image = self.augment_eval(image)
        else:
            image = self.augment(image)

        label = self.labels[item]
        domain = self.domains[item]

        return image, label, domain

    def __len__(self):
        return len(self.image_paths)

    def download(self):
        """Download the dataset and extract it if necessary."""
        root = pathlib.Path(self.root)
        archive = root / self.filename
        target = root / self.dirname

        if target.is_dir():
            print("Found existing dataset, skipping download.")
            return

        print(f"Downloading {self.__class__.__name__} dataset from {self.url} to {self.root}...")

        download_url(
            url=self.url,
            root=self.root,
            filename=self.filename,
            md5=self.md5,
        )

        assert archive.is_file(), f"Failed to download {self.__class__.__name__} to archive file."

        print(f"Extracting {archive} to {root}...")
        extract_archive(
            from_path=str(archive),
            to_path=str(target),
            remove_finished=self._remove_finished,
        )

        assert target.is_dir(), f"Failed to extract {self.__class__.__name__} archive to directory."

        # remove intermediate folder by moving all files up one level
        intermediate_folders = list(target.glob("*"))
        assert len(intermediate_folders) == 1, f"Expected exactly one intermediate folder, " \
                                               f"found {len(intermediate_folders)}"

        # move all files up one level
        intermediate_folder = intermediate_folders[0]
        for folder in intermediate_folder.iterdir():
            if folder.is_dir():
                shutil.move(str(folder), str(target / folder.name))

        # remove intermediate folder
        shutil.rmtree(str(intermediate_folder))

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