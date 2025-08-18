
from .domain_transform import DomainTransform


class IdentityTransform(DomainTransform):
    def __call__(self, data, labels, domains):
        return data, labels, domains
