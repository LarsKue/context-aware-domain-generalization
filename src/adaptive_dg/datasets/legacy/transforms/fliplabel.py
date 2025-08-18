import torch

from .domain_transform import DomainTransform


class FlipLabel(DomainTransform):
    """
    Randomly flips the binary label
    """
    def __init__(self, p: float = 1.0):
        self.p = p

    def __call__(self, data, labels, domains):
        mask = torch.rand(data.shape[0]) < self.p
        labels[mask] = ~labels[mask]

        return data, labels, domains
