
import torch

from .domain_transform import DomainTransform


class ZeroChannel(DomainTransform):
    """ Randomly zeros a color channel based on the label """
    def __init__(self, p: float):
        self.p = p

    def __call__(self, img, label):
        channel = label.argmax()
        if torch.rand(1) < self.p:
            channel = 1 - channel

        img[channel].zero_()

        return img, label
