
import torch

from torchvision.datasets import MNIST

from ..base import SplitDomainDataset
from ..transforms import BinarizeLabel, Compose, FlipColor, FlipLabel, OneHotLabel


class ColoredMNIST(SplitDomainDataset):
    """ Colored MNIST Dataset """
    # domains are labeled by the probability to flip the color
    all_domains = {"L", "M", "H"}
    domain_map = {
        "L": "Low",
        "M": "Medium",
        "H": "High",
    }

    n_classes = 2

    def __init__(self, root: str, in_distribution: bool, id_domains: set, ood_domains: set, normalize: str = "none", **kwargs):
        self.root = root

        mnist_train = MNIST(self.root, train=True, **kwargs)
        mnist_test = MNIST(self.root, train=False, **kwargs)
        data = torch.cat((mnist_train.data, mnist_test.data), dim=0).unsqueeze(1).repeat(1, 2, 1, 1)
        labels = torch.cat((mnist_train.targets, mnist_test.targets), dim=0)

        self.id_domains = set(id_domains)
        self.ood_domains = set(ood_domains)
        assert self.id_domains ^ self.ood_domains == self.all_domains, "id_domains and ood_domains must be disjoint and cover all domains"

        domain_transforms = {
            "L": Compose([BinarizeLabel(), FlipLabel(0.25), FlipColor(0.1), OneHotLabel(2)]),
            "M": Compose([BinarizeLabel(), FlipLabel(0.25), FlipColor(0.2), OneHotLabel(2)]),
            "H": Compose([BinarizeLabel(), FlipLabel(0.25), FlipColor(0.9), OneHotLabel(2)]),
        }

        if in_distribution:
            transforms = [domain_transforms[domain] for domain in id_domains]
        else:
            transforms = [domain_transforms[domain] for domain in ood_domains]

        super().__init__(data, labels, transforms, splits="equal", domains=2, in_distribution=in_distribution, normalize=normalize)
