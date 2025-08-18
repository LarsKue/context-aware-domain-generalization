import torch

from .domain_transform import DomainTransform


class FlipColor(DomainTransform):
    """
    Randomly flips (zeros) a color channel based on the label
    """
    def __init__(self, p: float = 1.0):
        self.p = p

    def __call__(self, data, labels, domains):
        assert data.shape[1] == 2, "Data must have 2 channels"
        assert labels.dtype == torch.bool, "Labels must be boolean"

        mask = torch.rand(len(data)) < self.p
        data[mask, 1 - labels[mask].int()] = 0.0

        return data, labels, domains
