
import torch.nn.functional as F

from .domain_transform import DomainTransform


class OneHotLabel(DomainTransform):
    """
    Produces a one-hot label
    """
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __call__(self, data, labels, domains):
        return data, F.one_hot(labels.long(), self.num_classes), domains
