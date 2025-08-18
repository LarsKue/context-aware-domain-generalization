
import torch
from torch.utils.data import Dataset, Subset

from .cached_dataset import StaticCachedDataset, DynamicCachedDataset

from typing import Sequence


class SetDataset(Dataset):
    def __init__(self, dataset, base_dataset, set_size=10, cache="dynamic", **kwargs):
        super().__init__()

        match cache:
            case "static":
                self.dataset = StaticCachedDataset(dataset, **kwargs)
            case "dynamic":
                self.dataset = DynamicCachedDataset(dataset, **kwargs)
            case "none":
                self.dataset = dataset
            case other:
                raise NotImplementedError(f"Unrecognized cache type: '{other}'")

        self.set_size = set_size
        
        if dataset.in_distribution:
            self.base_dataset = base_dataset
            self.compute_domain_indices()

    def __getitem__(self, item):
        image, label, domain = self.dataset[item]
        
        if self.dataset.in_distribution:
            domain_index = domain.argmax(0)
            domain_dataset = self.domain(domain_index)
            idx = torch.randperm(len(domain_dataset))[:self.set_size].tolist()
        else:
            domain_dataset = self.dataset
            idx = torch.randperm(len(domain_dataset))[:self.set_size].tolist()

        subset = Subset(domain_dataset, idx)
        image_set = torch.stack([x[0] for x in subset])

        return image, image_set, label, domain

    def __len__(self):
        return len(self.dataset)
    
    def compute_domain_indices(self):
        self.domain_indices = []
        for domain in range(len(self.base_dataset.id_domains)):
            self.domain_indices.append(list(set(self.base_dataset.domain_indices(domain)) & set(self.dataset.indices)))
        
    def domain(self, domain: int) -> Subset:
        """ Return a subset of the dataset containing only the given domain. """
        return Subset(self.base_dataset, self.domain_indices[domain])      
