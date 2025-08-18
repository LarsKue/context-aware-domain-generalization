import glob
import os
import random
from typing import Union, Iterable, Sized

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset


class SimulationDataset(Dataset):
    def __init__(self, root, noise_aug: bool = False, noise_std: float = 0.3, flip_aug: bool = True,
                 used_channels: Union[Iterable, Sized, int] = np.s_[:]
                 ):

        self.image_list = sorted(glob.glob(root))

        self.image_list_length = len(self.image_list)

        self.noise_aug = noise_aug
        self.noise_std = noise_std

        self.flip_aug = flip_aug

        self.used_channels = used_channels

    def __getitem__(self, index) -> dict:
        path = self.image_list[index % self.image_list_length]

        data = np.load(path)
        recon = data["reconstruction"]
        oxy = data["oxygenation"]
        seg = data["segmentation"]

        img = torch.from_numpy(recon)
        oxy, seg = torch.from_numpy(oxy).unsqueeze(0), torch.from_numpy(seg).unsqueeze(0)
        if len(img.size()) < 3:
            img = img.unsqueeze(0)

        if isinstance(self.used_channels, (slice, int)):
            img = img[self.used_channels, :, :].unsqueeze(0)
        elif isinstance(self.used_channels, (Iterable, Sized)):
            num_channels = len(self.used_channels)
            img_dims = img.size()
            new_img = torch.ones(num_channels, img_dims[1], img_dims[2])

            for channel_idx, used_channel in enumerate(self.used_channels):
                new_img[channel_idx, :, :] = img[used_channel, :, :]

            img = new_img

        if self.flip_aug:
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, [2])
                oxy = torch.flip(oxy, [2])
                seg = torch.flip(seg, [2])

        if self.noise_aug:
            img += torch.normal(0.0, self.noise_std, size=img.shape)

        return {"image": img.type(torch.float32),
                "oxy": oxy.type(torch.float32),
                "seg": seg.type(torch.float32)}

    def __len__(self):
        return self.image_list_length


def create_dataset_kris(batch_size=128):
    data_root_0 = "/home/tarkus/Desktop/WILDS/domain_disentanglement/data/DAS"
    data_root_1 = "/home/tarkus/Desktop/WILDS/domain_disentanglement/data/bad_simulations"

    channel_indices = list(range(0, 16))

    training_dataset_0 = SimulationDataset(root=os.path.join(data_root_0, "training/*.npz"),
                                           used_channels=channel_indices)
    training_dataset_1 = SimulationDataset(root=os.path.join(data_root_1, "training/*.npz"),
                                           used_channels=channel_indices)

    training_dataloader_0 = DataLoader(training_dataset_0, batch_size=batch_size)
    training_dataloader_1 = DataLoader(training_dataset_1, batch_size=batch_size)

    val_dataset_0 = SimulationDataset(root=os.path.join(data_root_0, "validation/*.npz"), used_channels=channel_indices)
    val_dataset_1 = SimulationDataset(root=os.path.join(data_root_1, "validation/*.npz"), used_channels=channel_indices)

    val_dataloader_0 = DataLoader(val_dataset_0, batch_size=batch_size)
    val_dataloader_1 = DataLoader(val_dataset_1, batch_size=batch_size)

    xs = []
    ys = []
    es = []
    for i, dataloader in enumerate([training_dataloader_0, training_dataloader_1]):
        for jk, (batch) in enumerate(dataloader):
            x = batch['image']  # [:,:3]
            # x = x[:,:,:,:128]
            x = torch.cat((x, x), 2)
            y = torch.zeros(x.shape[0], 2)
            e = torch.zeros(x.shape[0], 3)
            e[:, i] += 1
            if i == 1:
                for kk in range(e.shape[0]):
                    a = random.randint(0, 1)
                    e[kk, 2] += a
                    e[kk, i] -= a

            xs.append(x)
            ys.append(y)
            es.append(e)

    xs = torch.cat(xs)
    ys = torch.cat(ys)
    es = torch.cat(es)

    xs_val = []
    ys_val = []
    es_val = []
    for i, dataloader in enumerate([val_dataloader_0, val_dataloader_1]):
        for jk, (batch) in enumerate(dataloader):
            x = batch['image']  # [:,:3]
            # x = x[:,:,:,:128]
            x = torch.cat((x, x), 2)
            y = torch.zeros(x.shape[0], 2)
            e = torch.zeros(x.shape[0], 3)
            e[:, i] += 1

            if i == 1:
                for kk in range(e.shape[0]):
                    a = random.randint(0, 1)
                    e[kk, 2] += a
                    e[kk, i] -= a

            xs_val.append(x)
            ys_val.append(y)
            es_val.append(e)

    xs_val = torch.cat(xs_val)
    ys_val = torch.cat(ys_val)
    es_val = torch.cat(es_val)

    # xs = (xs-xs.mean(0))/xs.std(0)
    # xs_val = (xs_val-xs_val.mean(0))/xs_val.std(0)

    # restrict
    xs, ys, es = xs[:], ys[:], es[:]
    xs_val, ys_val, es_val = xs_val[:], ys_val[:], es_val[:]

    workers = 1
    dataloader = DataLoader(TensorDataset(xs, ys, es), batch_size=batch_size, shuffle=True, num_workers=workers,
                            drop_last=True)
    dataloader_val = DataLoader(TensorDataset(xs_val, ys_val, es_val), batch_size=batch_size, shuffle=True,
                                num_workers=workers, drop_last=True)

    return dataloader, dataloader, dataloader_val, dataloader_val, dataloader_val
