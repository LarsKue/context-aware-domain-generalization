
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

from lightning_trainable.hparams import Choice

from ..transforms import DomainTransform, IdentityTransform

from typing import Sequence


class MultiDomainDataset(Dataset):
    """
    Base Class for Multi-Domain Datasets

    Assumes you pass one tensor for each domain in data and labels.
    Environment labels are automatically computed as one-hot labels for in-distribution data.
    Out-of-distribution data uses zeros as environment labels.

    You may also pass a list of DomainTransforms, one for each domain.
    """
    def __init__(
            self,
            data: list[torch.Tensor],
            labels: list[torch.Tensor],
            transforms: list[DomainTransform] | None,
            domains: int,
            in_distribution: bool,
            normalize: Choice("none", "domain", "all") = "none",
    ):
        """
        :param data: list of tensors for each domain
        :param labels: list of tensors for each domain
        :param transforms: optional list of transforms for each domain
        :param domains: number of domains in the training data
        :param in_distribution: whether to use in-distribution or out-of-distribution domain labels
        :param normalize: how to normalize the data
            "none": no normalization
            "domain": normalize each domain separately
            "all": normalize all data together
        """
        if transforms is None:
            # this is a hack but since this is only called once we don't really care about the overhead
            transforms = [IdentityTransform() for _ in range(len(data))]

        assert len(data) == len(labels) == len(transforms), \
            f"Environment Mismatch: {len(data)}, {len(labels)}, {len(transforms)}"

        assert normalize in ("none", "domain", "all"), f"Unrecognized normalization: '{normalize}'"

        # compute one-hot encodings and apply transforms
        envs = []
        for i in range(len(data)):
            if in_distribution:
                domain = torch.full((data[i].shape[0],), fill_value=i)
                domain = F.one_hot(domain, num_classes=domains).float()
            else:
                domain = torch.zeros(data[i].shape[0], domains)

            data[i], labels[i], domain = transforms[i](data[i], labels[i], domain)

            if normalize == "domain":
                data[i] = (data[i] - data[i].mean(0)) / data[i].std(0)

            envs.append(domain)

        self.in_distribution = in_distribution

        # count number of samples per domain
        self._domain_counts = [len(d) for d in data]

        # concatenate all data
        self.data = torch.cat(data, dim=0)
        self.labels = torch.cat(labels, dim=0)
        self.domains = torch.cat(envs, dim=0)

        if normalize == "all":
            self.data = (self.data - self.data.mean(0)) / self.data.std(0)

    def domain(self, domain: int) -> Subset:
        """
        Returns a subset of the dataset for a specific domain.
        This is particularly useful when you want to access data from a specific domain, e.g.

        >>> dataset = MultiDomainDataset(...)
        >>> domain_set = dataset.domain(0)  # returns a subset of the dataset for domain 0
        >>> domain_set[0]  # returns sample 0 from domain 0
        """
        indices = self.domain_indices(domain)

        return Subset(self, indices)

    def domain_indices(self, domain: int) -> Sequence[int]:
        """ Return the indices pertaining to the subset of the dataset containing only the given domain. """
        # handle negative indices
        if domain < 0:
            domain += len(self._domain_counts)
        if not 0 <= domain < len(self._domain_counts):
            raise IndexError(f"Domain {domain} is out of range for dataset with {len(self._domain_counts)} domains")

        # compute indices from domain counts
        indices = list(range(sum(self._domain_counts[:domain]), sum(self._domain_counts[:domain + 1])))

        return indices

    def __getitem__(self, item: int) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        return self.data[item], self.labels[item], self.domains[item]

    def __len__(self) -> int:
        return len(self.data)
