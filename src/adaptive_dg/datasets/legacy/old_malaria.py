import glob

import numpy as np
import numpy.random
import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from .augments import RandomRotate


def get_patient_ids(path, threshold):
    files = [f for f in glob.glob(path + "**/*.png", recursive=True)]
    files = [f.split('/')[-1].split('.')[0].split('_')[0].split('thin')[0].split('Thin')[0] for f in files]

    unique_file_names = list(set(files))

    patients_with_threshold_cells = []

    for file_name in unique_file_names:
        if files.count(file_name) >= threshold:
            patients_with_threshold_cells.append(file_name)

    patients_with_threshold_cells.sort()
    return patients_with_threshold_cells


class MalariaData(data_utils.Dataset):
    def __init__(self, path, domain_list=[], transform=False):
        self.path = path
        self.domain_list = domain_list
        self.transform = transform

        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((64, 64), interpolation=Image.BILINEAR)
        self.hflip = transforms.RandomHorizontalFlip()
        self.vflip = transforms.RandomVerticalFlip()
        self.rrotate = RandomRotate()
        self.to_pil = transforms.ToPILImage()

        self.train_data, self.train_labels, self.train_domain = self.get_data()

    def get_cells_from_imgs(self, label_folder, domain):
        all_cells = [f for f in glob.glob(self.path + label_folder + "*.png", recursive=True)]

        cells_belonging_to_domain = []

        for cell in all_cells:
            if domain in cell:
                cells_belonging_to_domain.append(cell)

        cell_tensor_list = []
        for cell in cells_belonging_to_domain:
            with open(cell, 'rb') as f:
                with Image.open(f) as img:
                    img = img.convert('RGB')
            cell_tensor_list.append(self.to_tensor(self.resize(img)))

        # Concatenate
        return torch.stack(cell_tensor_list)

    def get_data(self):
        cells_per_domain_list = []
        labels_per_domain_list = []
        domain_per_domain_list = []

        for i, domain in enumerate(self.domain_list):
            cells_unifected = self.get_cells_from_imgs('Uninfected/', domain)
            label_unifected = torch.zeros(cells_unifected.size()[0]) + 0

            cells_parasitized = self.get_cells_from_imgs('Parasitized/', domain)
            label_parasitized = torch.zeros(cells_parasitized.size()[0]) + 1

            cells_per_domain_list.append(torch.cat((cells_unifected, cells_parasitized), 0))
            labels_per_domain_list.append(torch.cat((label_unifected, label_parasitized), 0))
            domain_labels = torch.zeros(label_unifected.size()[0] + label_parasitized.size()[0]) + i
            domain_per_domain_list.append(domain_labels)

        # One last cat
        train_imgs = torch.cat(cells_per_domain_list).float()
        train_labels = torch.cat(labels_per_domain_list).long()
        train_domains = torch.cat(domain_per_domain_list).long()

        # Convert to onehot
        y = torch.eye(2)
        train_labels = y[train_labels]

        # d = torch.eye(len(self.domain_list))
        d = torch.eye(10)
        train_domains = d[train_domains]

        return train_imgs, train_labels, train_domains

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, index):
        x = self.train_data[index].float()
        y = self.train_labels[index]  # .float()
        d = self.train_domain[index]  # .float()

        if self.transform:
            x = self.to_tensor(self.rrotate(self.vflip(self.hflip(self.to_pil(x)))))

        return x, y, d


def create_malaria_dataloader(batch_size, transform=True, test_env=0, seed=0):
    malaria_directory = '/home/tarkus/Desktop/WILDS/prodas_exp/data_viz/'

    kwargs = {'num_workers': 1, 'pin_memory': False}

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)

    patient_ids = get_patient_ids(malaria_directory + 'malaria/cell_images/', 400)

    print(patient_ids)
    train_patient_ids = patient_ids[:]
    test_patient_ids = patient_ids[test_env]
    train_patient_ids.remove(test_patient_ids)
    print(f"Test {test_patient_ids}")

    train_dataset = MalariaData(malaria_directory + 'malaria/cell_images/', domain_list=train_patient_ids,
                                transform=True)
    train_size = int(0.80 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = data_utils.DataLoader(
        MalariaData(malaria_directory + 'malaria/cell_images/', domain_list=[test_patient_ids]),
        batch_size=batch_size,
        shuffle=False)

    aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0),
        # transforms.RandomApply(torch.nn.ModuleList([transforms.RandomResizedCrop(64)]), p=0.2)
    ])

    if transform:
        print('Transform activated ')
        xs, ys, es = [], [], []
        for x, y, e in train_loader:
            xs.append(x)
            ys.append(y)
            es.append(y)

        xs = torch.cat(xs)
        ys = torch.cat(ys)
        es = torch.cat(es)
        '''
        ys = torch.cat(ys)
        mask_1 = (ys.argmax(1)==1).view(-1)
        mask_0 = (ys.argmax(1)==0).view(-1)
        n_s  = 50
        xs_1 = torch.cat(xs)[mask_1][:n_s]
        ys_1 = ys[mask_1][:n_s]
        es_1 = torch.cat(es)[mask_1][:n_s]

        xs_0 = torch.cat(xs)[mask_0][:n_s]
        ys_0 = ys[mask_0][:n_s]
        es_0 = torch.cat(es)[mask_0][:n_s]

        xs = torch.cat((xs_1,xs_0),0)
        ys = torch.cat((ys_1,ys_0),0)
        es = torch.cat((es_1,es_0),0)
        '''
        '''
        xs = torch.cat(xs)[:50]
        ys = torch.cat(ys)[:50]
        es = torch.cat(es)[:50]
        '''
        workers = 2
        #####
        weights_0 = ys.sum(0)[0].item()
        weights_1 = ys.sum(1)[1].item()
        weights = 1 / torch.tensor([weights_0, weights_1])
        weights = weights.double()
        weights = torch.tensor([0.1159, 0.8841], dtype=torch.float64)
        print(xs.shape)

        # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size)
        # train_loader_aug= DataLoader(CustomTensorDataset(tensors=(xs, ys, es), transform=aug_transform), batch_size=batch_size, shuffle=True, num_workers=workers)
        train_loader_aug = DataLoader(TensorDataset(xs, ys, es), batch_size=batch_size, shuffle=True,
                                      num_workers=workers)
        # train_loader_aug= DataLoader(CustomTensorDataset(tensors=(xs, ys, es), transform=aug_transform), batch_size=batch_size, sampler=sampler,  num_workers=workers)

    else:
        print('no aug used')
        train_loader_aug = train_loader

    return train_loader, train_loader, val_loader, val_loader, test_loader
    # return train_loader_aug, train_loader_aug, val_loader , val_loader, test_loader
    # return train_loader_aug, train_loader, val_loader , val_loader, test_loader


'''
    xs = []

    ys = []
    es = []

    xs_test = []
    ys_test = []
    es_test = []


    for x,y, e in dataloader:
        mask = (e[:,0] != 1).view(-1)
        mask_inverse = (1- mask.float()).bool()

        xs.append(x[mask])
        ys.append(y[mask])
        es.append(e[mask])

        xs_test.append(x[mask_inverse])
        ys_test.append(y[mask_inverse])
        es_test.append(e[mask_inverse])


    xs = torch.cat(xs)
    ys = torch.cat(ys)
    es = torch.cat(es)

    xs_test = torch.cat(xs_test)
    ys_test = torch.cat(ys_test)
    es_test = torch.cat(es_test)

    xs_mean = xs.mean()
    xs_std = xs.std() + 1e-7

    #xs = (xs- xs_mean)/ xs_std
    #xs_test = (xs_test- xs_mean)/ xs_std

    idx = torch.randperm(xs.shape[0])

    xs = xs[idx]
    ys = ys[idx]
    es = es[idx]

    idx = torch.randperm(xs_test.shape[0])
    xs_test = xs_test[idx]
    ys_test = ys_test[idx]
    es_test = es_test[idx]

    n_val = int(0.8 * xs.shape[0])
    n_test = int(0.9 * xs.shape[0])

    workers = 2

    aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0),
        transforms.RandomApply(torch.nn.ModuleList([transforms.RandomResizedCrop(64)]), p=0.2)
    ])

    if transform:
        dataloader = DataLoader(CustomTensorDataset(tensors=(xs[:n_val], ys[:n_val], es[:n_val]), transform=aug_transform), batch_size=batch_size,shuffle=True, num_workers=workers)
    else:
        dataloader = DataLoader(TensorDataset(xs[:n_val], ys[:n_val], es[:n_val]), batch_size=batch_size,shuffle=True, num_workers=workers)
    dataloader_iid_val = DataLoader(TensorDataset(xs[n_val:n_test], ys[n_val:n_test], es[n_val:n_test]), batch_size=batch_size,shuffle=True, num_workers=workers)
    dataloader_iid_test = DataLoader(TensorDataset(xs[n_test:], ys[n_test:], es[n_test:]), batch_size=batch_size,shuffle=True, num_workers=workers)
    dataloader_ood = DataLoader(TensorDataset(xs_test, ys_test, es_test), batch_size=batch_size,shuffle=True, num_workers=workers)


    return dataloader, dataloader, dataloader_iid_val, dataloader_iid_test, dataloader_ood
'''
