
from .base import DomainBedImageFolder


class PACS(DomainBedImageFolder):
    url = "https://drive.google.com/file/d/1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd"
    dirname = "PACS"
    filename = "PACS.zip"
    md5 = "fd5db817f7cdafeef3bfd525aaf9b42e"

    all_domains = {"P", "A", "C", "S"}
    domain_map = {
        "P": "photo",
        "A": "art_painting",
        "C": "cartoon",
        "S": "sketch",
    }

    n_classes = 7
