import torch
import torch.nn.functional as F

from lightning_trainable.hparams import Choice

from .multi_domain import MultiDomainDataset
from ..transforms import DomainTransform


class SplitDomainDataset(MultiDomainDataset):
    """
    Utility class for easier access to MultiDomainDataset

    Makes it easier to shuffle and split the dataset into domains.

    The `splits` argument controls how the dataset is split up. You can pass a list of transforms and then
        1. Pass a list of integers specifying the length of each split
        2. Pass a list of floats specifying the fraction of data for each split
        3. Pass "random" for a random split
        4. Pass "equal" for an equal split

    Usage:
        >>> data = torch.randn(100, 28, 28)
        >>> labels = torch.randint(0, 10, (100,))
        >>> transforms = [DomainTransform(), DomainTransform()]
        >>> # split dataset into 2 domains with 50 samples each
        >>> dataset = SplitDomainDataset(data, labels, transforms, domains=2, in_distribution=True, splits="equal")
        >>> # split dataset into 2 domains with 40, 60 samples each
        >>> dataset = SplitDomainDataset(data, labels, transforms, domains=2, in_distribution=True, splits=[40, 60])
        >>> # split dataset into 2 domains with 80%, 20% samples each
        >>> dataset = SplitDomainDataset(data, labels, transforms, domains=2, in_distribution=True, splits=[0.8, 0.2])
        >>> # split dataset into 2 domains with random proportions
        >>> dataset = SplitDomainDataset(data, labels, transforms, domains=2, in_distribution=True, splits="random")
    """
    def __init__(
            self,
            data: torch.Tensor,
            labels: torch.Tensor,
            transforms: list[DomainTransform],
            splits: str | list[int | float],
            domains: int,
            in_distribution: bool,
            shuffle: bool = True,
            normalize: Choice("none", "domain", "all") = "none",
    ):
        """
        :param data: Tensor containing all data
        :param labels: Tensor containing all labels
        :param transforms: list of transforms for each domain
        :param splits: One of
            1. list of integers specifying the length of each split
            2. list of floats specifying the fraction of data for each split
            3. "random" for a random split
            4. "equal" for an equal split
        :param domains: number of domains in the training data
        :param in_distribution: whether to use in-distribution or out-of-distribution domain labels
        :param shuffle: whether to shuffle the dataset before splitting
        :param normalize: how to normalize the data
            "none": no normalization
            "domain": normalize each domain separately
            "all": normalize all data together
        """
        # handle special-cases (strings)
        if splits == "random":
            splits = F.softmax(torch.randn(len(transforms)), dim=0).tolist()
        if splits == "equal":
            splits = [1.0 / len(transforms)] * len(transforms)

        # check validity of splits list
        assert len(transforms) == len(splits), f"Split Mismatch: {len(transforms)}, {len(splits)}"
        assert all([isinstance(s, type(splits[0])) for s in splits]), f"Cannot pass mixed-type splits."
        assert isinstance(splits[0], (int, float)), f"Invalid type for splits."

        # convert ratios to lengths
        if isinstance(splits[0], float):
            splits = (torch.tensor(splits) * len(data)).int().tolist()

            # adjust last split to match dataset length (for rounding errors etc.)
            splits[-1] = len(data) - sum(splits[:-1])

        # shuffle data and labels
        if shuffle:
            idx = torch.randperm(len(data))
            data = data[idx]
            labels = labels[idx]

        # split data and labels according to splits
        data = list(torch.split(data, splits, dim=0))
        labels = list(torch.split(labels, splits, dim=0))

        super().__init__(
            data=data,
            labels=labels,
            transforms=transforms,
            domains=domains,
            in_distribution=in_distribution,
            normalize=normalize
        )
