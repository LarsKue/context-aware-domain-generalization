from .domain_transform import DomainTransform


class BinarizeLabel(DomainTransform):
    """
    Binarizes the label based on a threshold value
    """
    def __init__(self, threshold: int = 5):
        self.threshold = threshold

    def __call__(self, data, labels, domains):
        labels = labels < self.threshold
        return data, labels, domains
