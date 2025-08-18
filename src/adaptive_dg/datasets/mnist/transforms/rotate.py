
from torchvision.transforms.functional import rotate

from .domain_transform import DomainTransform


class Rotate(DomainTransform):
    def __init__(self, angle: float):
        self.angle = angle

    def __call__(self, img, label):
        return rotate(img, self.angle), label
