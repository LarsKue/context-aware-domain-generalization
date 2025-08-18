
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset


class MultiDomainDataset(Dataset):
    """
    Dataset that combines multiple domains into one dataset, adding a one-hot domain label to each sample.
    """
    def __init__(self, *domains: Dataset, in_distribution: bool, n_domains: int):
        super().__init__()
        self.domains = list(domains)
        self.in_distribution = in_distribution
        self.n_domains = n_domains

    def __len__(self):
        return sum(self.domain_lengths)

    def __getitem__(self, item):
        domain, item = self._domain_item(item)
        img, label = self.domains[domain][item]

        if self.in_distribution:
            domain = F.one_hot(torch.tensor(domain), num_classes=self.n_domains)
        else:
            domain = torch.zeros(self.n_domains)

        return img, label, domain

    @property
    def domain_lengths(self):
        return [len(domain) for domain in self.domains]

    def domain_indices(self, domain: int):
        return list(range(sum(self.domain_lengths[:domain]), sum(self.domain_lengths[:domain+1])))

    def domain(self, domain: int) -> Subset:
        return Subset(self, self.domain_indices(domain))

    def _domain_item(self, item: int) -> (int, int):
        domain = 0
        while item >= self.domain_lengths[domain]:
            item -= self.domain_lengths[domain]
            domain += 1

        return domain, item
