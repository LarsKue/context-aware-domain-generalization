from .domain_transform import DomainTransform


class Compose(DomainTransform):
    """
    Compose multiple domain transforms
    """
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, data, labels, domains):
        for transform in self.transforms:
            data, labels, domains = transform(data, labels, domains)

        return data, labels, domains
