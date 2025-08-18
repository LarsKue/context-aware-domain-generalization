import torch
from torch.utils.data import Dataset, Subset

from .multi_domain import MultiDomainDataset


class SetDataset(Dataset):
    """
    Wrapper Dataset to produce a set of images from the same domain as each sample.

    Careful: Use this as the outermost dataset, unless you know what you are doing.
    Wrapping this dataset e.g. in a Subset for train/val splitting will lead to data leakage,
    if you split the dataset within a domain.
    This is caused by the fact that __getitem__ accesses items at other indices as well to construct the set.
    """
    def __init__(self, dataset: MultiDomainDataset, set_size: int = 10):
        super().__init__()
        self.dataset = dataset
        self.set_size = set_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label, domain = self.dataset[item]
        domain_idx = torch.argmax(domain)

        domain_set = self.dataset.domains[domain_idx]

        # ensure that the item is not in the set
        idx = torch.arange(len(domain_set))
        idx = idx[idx != item]
        idx = idx[torch.randperm(len(idx))]
        idx = idx[:self.set_size].tolist()

        subset = Subset(domain_set, idx)
        image_set = torch.stack([x[0] for x in subset])

        return image, image_set, label, domain

    def domain(self, domain: int) -> Subset:
        # this is safe because we are taking the subset over the whole domain
        return Subset(self, self.domain_indices(domain))

    def domain_indices(self, domain: int):
        return self.dataset.domain_indices(domain)
