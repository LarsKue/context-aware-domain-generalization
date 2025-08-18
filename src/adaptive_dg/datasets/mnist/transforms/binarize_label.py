
import torch.nn.functional as F

from .domain_transform import DomainTransform


class BinarizeLabel(DomainTransform):
    """
    Binarizes the label based on a threshold value
    """
    def __init__(self, threshold: int):
        self.threshold = threshold

    def __call__(self, img, label):
        label = label.argmax()
        label = label < self.threshold
        label = F.one_hot(label.long(), num_classes=2)

        return img, label
