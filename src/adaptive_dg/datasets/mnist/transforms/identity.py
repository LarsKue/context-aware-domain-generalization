
from .domain_transform import DomainTransform


class IdentityTransform(DomainTransform):
    """ Does nothing """
    def __call__(self, img, label):
        return img, label
