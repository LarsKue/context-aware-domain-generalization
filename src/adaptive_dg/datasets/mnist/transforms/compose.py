from .domain_transform import DomainTransform


class Compose(DomainTransform):
    """
    Compose multiple domain transforms
    """
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img, label):
        for transform in self.transforms:
            img, label = transform(img, label)

        return img, label
