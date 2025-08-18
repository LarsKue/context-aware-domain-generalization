from .wilds import WILDSDataset


class Camelyon17(WILDSDataset):
    name = "camelyon17"
    # these are anonymized hospitals
    all_domains = ["0", "1", "2", "3", "4"]
    domain_map = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
    }

    n_classes = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, domain_meta_index=0)
