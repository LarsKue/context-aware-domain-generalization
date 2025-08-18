
import torch

from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate


class Rotate:
    def __init__(self, angle: int | float, p: float = 1.0):
        self.angle = angle  # in degrees
        self.p = p

    def __call__(self, data, labels, domains):
        mask = torch.rand(data.shape[0]) < self.p
        data = data.clone()
        data[mask] = rotate(data[mask], self.angle, interpolation=InterpolationMode.BILINEAR, fill=[0])

        return data, labels, domains

