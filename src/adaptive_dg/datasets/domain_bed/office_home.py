from .domain_bed import DomainBedDataset
from .download import DomainBedDownloadMixin


class OfficeHome(DomainBedDataset):
    name = "office_home"

    url = "https://drive.google.com/file/d/1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC"
    filename = "office_home.zip"
    dirname = "office_home"
    md5 = "b1c14819770c4448fd5b6d931031c91c"

    all_domains = ["A", "C", "P", "R"]
    domain_map = {
        "A": "Art",
        "C": "Clipart",
        "P": "Product",
        "R": "Real World",
    }

    n_classes = 65
