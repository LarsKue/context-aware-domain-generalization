
from .wilds import WILDSDataset


class FMoWRegion(WILDSDataset):
    name = "fmow"
    all_domains = ["africa", "asia", "europe", "north-america", "south-america"]
    domain_map = {
        "africa": 0,
        "asia": 1,
        "europe": 2,
        "north-america": 3,
        "south-america": 4,
    }
    n_classes = 62

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, domain_meta_index=1)


class FMoWYear(WILDSDataset):
    name = "fmow"
    all_domains = {"2015", "2016", "2017", "2018"}
    domain_map = {
        "2015": 0,
        "2016": 1,
        "2017": 2,
        "2018": 3,
    }
    n_classes = 62

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, domain_meta_index=0)
