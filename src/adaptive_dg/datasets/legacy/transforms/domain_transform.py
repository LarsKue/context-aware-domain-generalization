
from torch import Tensor


class DomainTransform:
    """
    Transform applicable to MultiDomainDatasets
    """
    def __call__(self, data: Tensor, labels: Tensor, domains: Tensor) -> (Tensor, Tensor, Tensor):
        raise NotImplementedError
