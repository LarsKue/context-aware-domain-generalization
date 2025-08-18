
from torch.utils.data import Dataset


class AugmentDataset(Dataset):
    def __init__(self, dataset, augment):
        super().__init__()
        self.dataset = dataset
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        image = self.augment(image)
        return image, label
