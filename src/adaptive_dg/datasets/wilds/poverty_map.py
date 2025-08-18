
from .wilds import WILDSDataset


class PovertyMapCountry(WILDSDataset):
    name = "poverty"
    all_domains = [str(i) for i in range(22)]
    n_classes = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, domain_meta_index=2)
        self.transform = lambda x: x
        self.target_transform = lambda x: x


class PovertyMapUrbanicity(WILDSDataset):
    name = "poverty"
    all_domains = ["rural", "urban"]
    n_classes = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, domain_meta_index=0)
        self.transform = lambda x: x
        self.target_transform = lambda x: x
