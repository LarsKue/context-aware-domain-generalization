import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torchvision import transforms


# Create dataset from ISIC images
class ISICDataset(Dataset):
    def __init__(self, img_path, transform,
                 csv_path="/home/tarkus/Desktop/WILDS/data/skin_bias/debiasing-skin/trap-sets/isic_annotated_train1.csv",
                 test=False):
        self.targets = pd.read_csv(csv_path)
        self.img_path = img_path
        self.transform = transform
        self.test = test

    def __getitem__(self, index):
        img_name = os.path.join(self.img_path,
                                f'{self.targets.image.iloc[index][:-4]}.jpg')
        img = Image.open(img_name)
        img = self.transform(img)
        # print(img.shape)
        if not self.test:
            targets = self.targets.label.iloc[index]

            # targets = self.targets.label.iloc[index, 1:]
            targets = np.array([targets])
            ## env_info = self.targets[['dark_corner', 'hair', 'gel_border', 'ruler', 'ink', 'patches', 'vasc']].iloc[index].to_numpy()
            env_info = self.targets[['dark_corner']].iloc[index].to_numpy()
            # targets = targets.astype('float').reshape(-1, 9)
            return img, targets, env_info  # {'image': img, 'label': targets}
        else:
            return {'image': img}

    def __len__(self):
        return len(self.targets)


def create_skin_dataset(batch_size):
    dataset_name = "Skin"

    df = pd.read_csv("/home/tarkus/Desktop/WILDS/data/skin_bias/debiasing-skin/trap-sets/isic_annotated_train1.csv")

    data_transforms = transforms.Compose([transforms.Resize([256, 256]),
                                          transforms.ToTensor()])  # ,
    # transforms.Normalize((0.6678, 0.5298, 0.5245), (0.1333, 0.1476, 0.1590))])

    train_img_path = "/home/tarkus/Desktop/WILDS/data/skin_bias/ISIC2018_Task1-2_Training_Input/"

    data_set_train = ISICDataset(
        train_img_path, transform=data_transforms
    )
    # dataloader_train = data.DataLoader(data_set_train, batch_size=batch_size, shuffle=True, num_workers=2)
    dataloader_tmp = data.DataLoader(data_set_train, batch_size=batch_size, shuffle=True, num_workers=2)
    xs, ys, es = [], [], []
    for x, y, e in dataloader_tmp:
        xs.append(x)
        ys.append(y)
        es.append(e)

    xs = torch.cat(xs)
    ys = torch.cat(ys)
    es = torch.cat(es)

    aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0),
        transforms.RandomApply(torch.nn.ModuleList([transforms.RandomResizedCrop(256)]), p=0.2)
    ])
    n_train = int(0.8 * xs.shape[0])
    # dataloader_train = data.DataLoader(TensorDataset(xs[:n_train],ys[:n_train],es[:n_train]), batch_size=batch_size, shuffle=True, num_workers=2)
    dataloader_train = data.DataLoader(
        CustomTensorDataset(tensors=(xs[:n_train], ys[:n_train], es[:n_train]), transform=aug_transform),
        batch_size=batch_size, shuffle=True, num_workers=2)
    dataloader_train_no_aug = data.DataLoader(TensorDataset(xs[:n_train], ys[:n_train], es[:n_train]),
                                              batch_size=batch_size, shuffle=True, num_workers=2)
    dataloader_val = data.DataLoader(TensorDataset(xs[n_train:], ys[n_train:], es[n_train:]), batch_size=batch_size,
                                     shuffle=True, num_workers=2)
    dataloader_ood = data.DataLoader(TensorDataset(xs[n_train:], ys[n_train:], es[n_train:]), batch_size=batch_size,
                                     shuffle=True, num_workers=2)
    '''
    dataloader_train = data.DataLoader(data_set_train, batch_size=10, shuffle=True, num_workers=2)

    x, y, e = next(iter(dataloader_train))

    del dataloader_train

    dataloader_train = data.DataLoader(TensorDataset(x,y,e), batch_size=batch_size, shuffle=True, num_workers=2)
    '''
    # dataloader_train, dataloader_train_no_aug, dataloader_val, dataloader_iid_test, dataloader_ood = create_skin_dataset(batch_size=batch_size)

    return dataloader_train, dataloader_train_no_aug, dataloader_val, dataloader_val, dataloader_ood


def create_embedded_dataloader(dataloader, model, batch_size=32):
    model.eval()
    xs, ys, es = [], [], []

    for x, y, e in dataloader:
        x = x.float().cuda()
        y = y.float()
        e = e.float()
        with torch.no_grad():
            xs.append(model(x).cpu())
        ys.append(y)
        es.append(e)

    xs = torch.cat(xs)
    ys = torch.cat(ys)
    es = torch.cat(es)

    dataloader_embedding = data.DataLoader(TensorDataset(xs, ys, es), batch_size=batch_size, shuffle=True,
                                           num_workers=2)

    return dataloader_embedding


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]
        e = self.tensors[2][index]

        return x, y, e

    def __len__(self):
        return self.tensors[0].size(0)
