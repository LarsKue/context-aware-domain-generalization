
import torch
from torchvision.datasets import MNIST

from ..base import SplitDomainDataset
from ..transforms import Compose, OneHotLabel, Rotate


class RotatedMNIST(SplitDomainDataset):
    """ Rotated MNIST Dataset """

    n_classes = 10

    def __init__(self, root: str, in_distribution: bool, normalize: str = "none", **kwargs):
        self.root = root

        mnist_train = MNIST(self.root, train=True, **kwargs)
        mnist_test = MNIST(self.root, train=False, **kwargs)
        data = torch.cat((mnist_train.data, mnist_test.data), dim=0).unsqueeze(1)
        labels = torch.cat((mnist_train.targets, mnist_test.targets), dim=0)

        if in_distribution:
            transforms = [
                Rotate(0),
                Rotate(15),
                Rotate(30),
                Rotate(45),
                Rotate(60),
            ]
        else:
            transforms = [
                Rotate(75),
            ]

        transforms = [
            Compose([t, OneHotLabel(10)]) for t in transforms
        ]

        super().__init__(data, labels, transforms, splits="equal", domains=5, in_distribution=in_distribution, normalize=normalize)
