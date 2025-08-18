from pathlib import Path
import numpy as np

from lightning_trainable.hparams import Range, Choice
from lightning_trainable.trainable import TrainableHParams


class DGHParams(TrainableHParams):
    data_root: Path | str = Path("data")
    dataset: Choice(
        "BikeSharingSeason",
        "CaliforniaHousing",
        "CensusIncomeEthnicity",
        "MLBook",
        "ProDASCondSatisfied",
        "ColoredMNIST", "RotatedMNIST",
        "OfficeHome", "PACS", "TerraIncognita", "VLCS",
        "Camelyon17", "FMoWYear", "FMoWRegion", "PovertyMapCountry", "PovertyMapUrbanicity",
        "Kang",
        "Simpson",
    )
    augment: bool = True

    train_split: Range(0.0, 1.0) = 0.8
    val_split: Range(0.0, 1.0) = 0.1
    test_split: Range(0.0, 1.0) = 0.1
    data_seed: int = 42
    training_seed: int = 42

    set_size: int
    id_domains: list | str
    ood_domains: list | str

    @classmethod
    def validate_parameters(cls, hparams):
        hparams = super().validate_parameters(hparams)

        hparams["data_root"] = Path(hparams["data_root"])

        assert np.isclose(sum([hparams["train_split"], hparams["val_split"], hparams["test_split"]]), 1.0), \
            "train_split, val_split, and test_split must sum to 1"

        if isinstance(hparams.id_domains, str):
            hparams.id_domains = list(hparams.id_domains)

        if isinstance(hparams.ood_domains, str):
            hparams.ood_domains = list(hparams.ood_domains)

        return hparams
