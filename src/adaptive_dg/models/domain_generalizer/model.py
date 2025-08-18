import torchvision.transforms as transforms

from lightning_trainable.trainable import Trainable
from torch.utils.data import DataLoader, random_split

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from adaptive_dg.datasets import *
from adaptive_dg.utils import temporary_seed

from .hparams import DGHParams


class DomainGeneralizer(Trainable):
    """
    Base class for domain generalization models.
    Includes data loading and splitting.
    """
    hparams: DGHParams

    def __init__(self, hparams: DGHParams | dict):
        super().__init__(hparams)

        self.train_data, self.val_data, self.test_data, self.ood_data = self.configure_datasets()

    @property
    def image_shape(self):
        image, image_set, label, domain = self.train_data[0]
        return image.shape

    @property
    def image_set_shape(self):
        image, image_set, label, domain = self.train_data[0]
        return image_set.shape

    @property
    def n_classes(self):
        image, image_set, label, domain = self.train_data[0]
        return len(label)

    @property
    def n_domains(self):
        image, image_set, label, domain = self.train_data[0]
        return len(domain)

    def configure_datasets(self):
        with temporary_seed(self.hparams.data_seed):
            train_domains, val_domains, test_domains, ood_domains = self.configure_splits()

            train_data = MultiDomainDataset(*train_domains, in_distribution=True,
                                            n_domains=len(self.hparams.id_domains))
            val_data = MultiDomainDataset(*val_domains, in_distribution=True, n_domains=len(self.hparams.id_domains))
            test_data = MultiDomainDataset(*test_domains, in_distribution=True, n_domains=len(self.hparams.id_domains))
            ood_data = MultiDomainDataset(*ood_domains, in_distribution=False, n_domains=len(self.hparams.id_domains))

            train_set_data = SetDataset(train_data, set_size=self.hparams.set_size)
            val_set_data = SetDataset(val_data, set_size=self.hparams.set_size)
            test_set_data = SetDataset(test_data, set_size=self.hparams.set_size)
            ood_set_data = SetDataset(ood_data, set_size=self.hparams.set_size)

        return train_set_data, val_set_data, test_set_data, ood_set_data

    def configure_splits(self):
        id_domains, ood_domains = self.configure_domains()

        # split id domains into train, val, and test
        splits = [self.hparams.train_split, self.hparams.val_split, self.hparams.test_split]
        id_splits = [random_split(domain, splits) for domain in id_domains]

        train_domains = [split[0] for split in id_splits]
        val_domains = [split[1] for split in id_splits]
        test_domains = [split[2] for split in id_splits]

        if self.hparams.augment:
            augment = transforms.Compose([
                transforms.RandomResizedCrop([224, 224], scale=(0.7, 1.0), antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
            ])
            train_domains = [AugmentDataset(domain, augment) for domain in train_domains]

        return train_domains, val_domains, test_domains, ood_domains

    def configure_domains(self) -> tuple[list, list]:
        # construct base datasets
        match self.hparams.dataset:
            case "BikeSharingSeason":
                dataset = BikeSharingSeason(self.hparams.data_root, download=True)
            case "CaliforniaHousing":
                dataset = CaliforniaHousingDataset(self.hparams.data_root, download=True)
            case "Camelyon17":
                dataset = Camelyon17(self.hparams.data_root, download=True)
            case "CensusIncomeEthnicity":
                dataset = CensusIncomeEthnicity(self.hparams.data_root, download=True)
            case "ColoredMNIST":
                dataset = ColoredMNIST(self.hparams.data_root, download=True)
            case "FMowRegion":
                dataset = FMoWRegion(self.hparams.data_root, download=True)
            case "FMoWYear":
                dataset = FMoWYear(self.hparams.data_root, download=True)
            case "Kang":
                dataset = Kang(self.hparams.data_root, download=True)
            case "MLBook":
                dataset = MLBookDataset(self.hparams.data_root, download=True)
            case "OfficeHome":
                dataset = OfficeHome(self.hparams.data_root, download=True)
            case "PACS":
                dataset = PACS(self.hparams.data_root, download=True)
            case "ProDASCondSatisfied":
                dataset = ProDASCondSatisfied(self.hparams.data_root, download=True)
            case "RotatedMNIST":
                dataset = RotatedMNIST(self.hparams.data_root, download=True)
            case "TerraIncognita":
                dataset = TerraIncognita(self.hparams.data_root, download=True)
            case "VLCS":
                dataset = VLCS(self.hparams.data_root, download=True)
            case "PovertyMapUrbanicity":
                dataset = PovertyMapUrbanicity(self.hparams.data_root, download=True)
            case "PovertyMapCountry":
                dataset = PovertyMapCountry(self.hparams.data_root, download=True)
            case "Simpson":
                dataset = SimpsonDataset(self.hparams.data_root, download=True)
            case other:
                raise NotImplementedError(f"Unrecognized dataset: {other}")

        if not set(self.hparams.id_domains) ^ set(self.hparams.ood_domains) == set(dataset.all_domains):
            have_domains = set(self.hparams.id_domains) | set(self.hparams.ood_domains)
            need_domains = set(dataset.all_domains)
            missing_domains = need_domains - have_domains
            extra_domains = have_domains - need_domains

            message = "ID and OOD domains must be disjoint and cover all domains in dataset."

            if missing_domains:
                message += f" Missing domains: {missing_domains}."
            if extra_domains:
                message += f" Extra domains: {extra_domains}."

            raise ValueError(message)

        # get domains
        id_domains = [dataset.domain(domain) for domain in self.hparams.id_domains]
        ood_domains = [dataset.domain(domain) for domain in self.hparams.ood_domains]

        return id_domains, ood_domains

    def val_dataloader(self):
        id_val_loader = DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
            num_workers=self.hparams.num_workers,
        )
        ood_loader = DataLoader(
            self.ood_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
            num_workers=self.hparams.num_workers,
        )
        id_test_loader = DataLoader(
            self.test_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
            num_workers=self.hparams.num_workers,
        )

        return [id_val_loader, id_test_loader, ood_loader]

    def compute_ood_metrics(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            id_metrics = self.compute_metrics(batch, batch_idx)
            for key, value in id_metrics.items():
                self.log(f"validation/{key}", value, prog_bar=key == self.hparams.loss, add_dataloader_idx=False)
        elif dataloader_idx == 1:
            id_metrics = self.compute_metrics(batch, batch_idx)
            for key, value in id_metrics.items():
                self.log(f"test/{key}", value, prog_bar=key == self.hparams.loss, add_dataloader_idx=False)
        elif dataloader_idx == 2:
            ood_metrics = self.compute_ood_metrics(batch, batch_idx)
            for key, value in ood_metrics.items():
                self.log(f"ood/{key}", value, prog_bar=key == self.hparams.loss, add_dataloader_idx=False)
