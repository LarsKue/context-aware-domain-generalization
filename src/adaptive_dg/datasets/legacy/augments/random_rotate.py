
import torch

from torchvision.transforms.functional import rotate

import numpy as np


class RandomRotate:
    """
    Rotate input tensor by degrees chosen with probability p.
    """

    def __init__(self, degrees=(0, 90, 180, 270), p=(0.25, 0.25, 0.25, 0.25), **kwargs):
        assert np.isclose(np.sum(p), 1.0), "p must sum to 1.0"
        assert len(degrees) == len(p), "degrees and p must have the same length"

        self.degrees = degrees
        self.p = p
        self.kwargs = kwargs

    def __call__(self, data):
        # select degrees
        degrees = np.random.choice(self.degrees, p=self.p, size=data.shape[0])

        # rotate each image individually
        rotated_data = [rotate(data[i], degrees[i], **self.kwargs) for i in range(data.shape[0])]
        rotated_data = torch.stack(rotated_data)

        return rotated_data
