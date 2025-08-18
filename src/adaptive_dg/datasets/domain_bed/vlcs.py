from .domain_bed import DomainBedDataset
from .download import DomainBedDownloadMixin


class VLCS(DomainBedDataset):
    name = "VLCS"

    url = "https://drive.google.com/file/d/1skwblH1_okBwxWxmRsp9_qi15hyPpxg8"
    filename = "VLCS.tar.gz"
    dirname = "VLCS"
    md5 = "8870171eba660927d9668f4e515928e4"

    all_domains = ["V", "L", "C", "S"]
    domain_map = {
        "V": "VOC2007",
        "L": "LabelMe",
        "C": "Caltech101",
        "S": "SUN09",
    }

    n_classes = 5
