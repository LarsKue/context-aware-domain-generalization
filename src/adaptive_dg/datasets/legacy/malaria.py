import glob

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import InterpolationMode

from .augments import RandomRotate
from .base import SplitDomainDataset


class MalariaDataset(SplitDomainDataset):
    def __init__(self, root: str, in_distribution: bool = True):
        self.root = root

        if in_distribution:
            data, labels = ...
        else:
            data, labels = ...

        self.augment = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRotate(),
        ])

        super().__init__(
            data=data,
            labels=labels,
            transforms=None,
            domains=4,
            in_distribution=in_distribution,
        )

    def __getitem__(self, index):
        data, label, domain = super().__getitem__(index)
        data = self.augment(data)
        return data, label, domain

    def get_cells_from_imgs(self, label_folder, domain):
        all_cells = [f for f in glob.glob(self.path + label_folder + "*.png", recursive=True)]

        cells_belonging_to_domain = []

        for cell in all_cells:
            if domain in cell:
                cells_belonging_to_domain.append(cell)

        cell_tensor_list = []
        for cell in cells_belonging_to_domain:
            with open(cell, 'rb') as f:
                with Image.open(f) as img:
                    img = img.convert('RGB')
            cell_tensor_list.append(self.to_tensor(self.resize(img)))

        # Concatenate
        return torch.stack(cell_tensor_list)

    def get_data(self):
        cells_per_domain_list = []
        labels_per_domain_list = []
        domain_per_domain_list = []

        for i, domain in enumerate(self.domain_list):
            cells_unifected = self.get_cells_from_imgs('Uninfected/', domain)
            label_unifected = torch.zeros(cells_unifected.size()[0]) + 0

            cells_parasitized = self.get_cells_from_imgs('Parasitized/', domain)
            label_parasitized = torch.zeros(cells_parasitized.size()[0]) + 1

            cells_per_domain_list.append(torch.cat((cells_unifected, cells_parasitized), 0))
            labels_per_domain_list.append(torch.cat((label_unifected, label_parasitized), 0))
            domain_labels = torch.zeros(label_unifected.size()[0] + label_parasitized.size()[0]) + i
            domain_per_domain_list.append(domain_labels)

        # One last cat
        train_imgs = torch.cat(cells_per_domain_list).float()
        train_labels = torch.cat(labels_per_domain_list).long()
        train_domains = torch.cat(domain_per_domain_list).long()

        # Convert to onehot
        y = torch.eye(2)
        train_labels = y[train_labels]

        #d = torch.eye(len(self.domain_list))
        d = torch.eye(10)
        train_domains = d[train_domains]

        return train_imgs, train_labels, train_domains


