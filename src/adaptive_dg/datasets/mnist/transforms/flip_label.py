
import torch

from .domain_transform import DomainTransform


class FlipLabel(DomainTransform):
    """ Randomly flips the binary one-hot label """

    def __init__(self, p: float = 1.0):
        self.p = p

    def __call__(self, img, label):
        if torch.rand(1) < self.p:
            label = 1 - label
        
        return img, label
