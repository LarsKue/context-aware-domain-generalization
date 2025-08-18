
import torch


class DomainTransform:
    def __call__(self, img: torch.Tensor, label: torch.Tensor):
        raise NotImplementedError
