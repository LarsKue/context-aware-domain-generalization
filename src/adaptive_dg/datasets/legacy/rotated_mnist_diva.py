"""Pytorch Dataset object that loads MNIST and SVHN. It returns x,y,s where s=0 when x,y is taken from MNIST."""

import os
import numpy as np
import torch
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms

from torchvision.utils import save_image


class MnistRotated(data_utils.Dataset):
    """
    due to: https://github.com/AMLab-Amsterdam/DIVA/blob/master/paper_experiments/rotated_mnist/dataset/mnist_loader.py
    """

    def __init__(
        self,
        list_train_domains,
        list_test_domain,
        num_supervised,
        mnist_subset,
        root,
        transform=None,
        train=True,
        download=True,
    ):
        self.list_train_domains = list_train_domains
        self.list_test_domain = list_test_domain
        self.num_supervised = num_supervised
        self.mnist_subset = mnist_subset
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        self.download = download

        if self.train:
            self.train_data, self.train_labels, self.train_domain = self._get_data()
        else:
            self.test_data, self.test_labels, self.test_domain = self._get_data()

    def load_inds(self):
        return np.load(self.root + "supervised_inds_" + str(self.mnist_subset) + ".npy")

    def _get_data(self):
        if self.train:
            train_loader = torch.utils.data.DataLoader(
                datasets.MNIST(
                    self.root,
                    train=True,
                    download=self.download,
                    transform=transforms.ToTensor(),
                ),
                batch_size=60000,
                shuffle=False,
            )

            for i, (x, y) in enumerate(train_loader):
                mnist_imgs = x
                mnist_labels = y

            # Get num_supervised number of labeled examples
            sup_inds = self.load_inds()
            mnist_labels = mnist_labels[sup_inds]
            mnist_imgs = mnist_imgs[sup_inds]

            to_pil = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()

            # Run transforms
            mnist_0_img = torch.zeros((self.num_supervised, 28, 28))
            mnist_15_img = torch.zeros((self.num_supervised, 28, 28))
            mnist_30_img = torch.zeros((self.num_supervised, 28, 28))
            mnist_45_img = torch.zeros((self.num_supervised, 28, 28))
            mnist_60_img = torch.zeros((self.num_supervised, 28, 28))
            mnist_75_img = torch.zeros((self.num_supervised, 28, 28))

            for i in range(len(mnist_imgs)):
                mnist_0_img[i] = to_tensor(to_pil(mnist_imgs[i]))

            for i in range(len(mnist_imgs)):
                mnist_15_img[i] = to_tensor(
                    transforms.functional.rotate(to_pil(mnist_imgs[i]), 15)
                )

            for i in range(len(mnist_imgs)):
                mnist_30_img[i] = to_tensor(
                    transforms.functional.rotate(to_pil(mnist_imgs[i]), 30)
                )

            for i in range(len(mnist_imgs)):
                mnist_45_img[i] = to_tensor(
                    transforms.functional.rotate(to_pil(mnist_imgs[i]), 45)
                )

            for i in range(len(mnist_imgs)):
                mnist_60_img[i] = to_tensor(
                    transforms.functional.rotate(to_pil(mnist_imgs[i]), 60)
                )

            for i in range(len(mnist_imgs)):
                mnist_75_img[i] = to_tensor(
                    transforms.functional.rotate(to_pil(mnist_imgs[i]), 75)
                )

            # Choose subsets that should be included into the training
            training_list_img = []
            training_list_labels = []

            for domain in self.list_train_domains:
                if domain == "0":
                    training_list_img.append(mnist_0_img)
                    training_list_labels.append(mnist_labels)
                if domain == "15":
                    training_list_img.append(mnist_15_img)
                    training_list_labels.append(mnist_labels)
                if domain == "30":
                    training_list_img.append(mnist_30_img)
                    training_list_labels.append(mnist_labels)
                if domain == "45":
                    training_list_img.append(mnist_45_img)
                    training_list_labels.append(mnist_labels)
                if domain == "60":
                    training_list_img.append(mnist_60_img)
                    training_list_labels.append(mnist_labels)
                if domain == "75":
                    training_list_img.append(mnist_75_img)
                    training_list_labels.append(mnist_labels)

            # Stack
            train_imgs = torch.cat(training_list_img)
            train_labels = torch.cat(training_list_labels)

            # Create domain labels
            train_domains = torch.zeros(train_labels.size())
            train_domains[0 : self.num_supervised] += 0
            train_domains[self.num_supervised : 2 * self.num_supervised] += 1
            train_domains[2 * self.num_supervised : 3 * self.num_supervised] += 2
            train_domains[3 * self.num_supervised : 4 * self.num_supervised] += 3
            train_domains[4 * self.num_supervised : 5 * self.num_supervised] += 4

            # Shuffle everything one more time
            inds = np.arange(train_labels.size()[0])
            np.random.shuffle(inds)
            train_imgs = train_imgs[inds]
            train_labels = train_labels[inds]
            train_domains = train_domains[inds].long()

            # Convert to onehot
            y = torch.eye(10)
            train_labels = y[train_labels]

            # Convert to onehot
            d = torch.eye(5)
            train_domains = d[train_domains]

            return train_imgs.unsqueeze(1), train_labels, train_domains

        else:
            train_loader = torch.utils.data.DataLoader(
                datasets.MNIST(
                    self.root,
                    train=True,
                    download=self.download,
                    transform=transforms.ToTensor(),
                ),
                batch_size=60000,
                shuffle=False,
            )

            for i, (x, y) in enumerate(train_loader):
                mnist_imgs = x
                mnist_labels = y

            # Get num_supervised number of labeled examples
            sup_inds = self.load_inds()
            mnist_labels = mnist_labels[sup_inds]
            mnist_imgs = mnist_imgs[sup_inds]

            to_pil = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()

            # Get angle
            rot_angle = int(self.list_test_domain[0])

            # Resize
            mnist_imgs_rot = torch.zeros((self.num_supervised, 28, 28))

            for i in range(len(mnist_imgs)):
                mnist_imgs_rot[i] = to_tensor(
                    transforms.functional.rotate(to_pil(mnist_imgs[i]), rot_angle)
                )

            # Create domain labels
            test_domain = torch.zeros(mnist_labels.size()).long()

            # Convert to onehot
            y = torch.eye(10)
            mnist_labels = y[mnist_labels]

            # Convert to onehot
            d = torch.eye(5)
            test_domain = d[test_domain]

            return mnist_imgs_rot.unsqueeze(1), mnist_labels, test_domain

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)

    def __getitem__(self, index):
        if self.train:
            x = self.train_data[index]
            y = self.train_labels[index]
            d = self.train_domain[index]
        else:
            x = self.test_data[index]
            y = self.test_labels[index]
            d = self.test_domain[index]

        if self.transform is not None:
            x = self.transform(x)

        return x, y, d


def get_dataloader_rmnist_diva(seed=1, test_env=0, extend_dim=False):
    """
    Loading of RMNIST data due to diva.
    Unnecesarilly complex
    """

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    list_train_domains = ["0", "15", "30", "45", "60", "75"]

    list_test_domains = list(list_train_domains[test_env : test_env + 1])
    del list_train_domains[test_env]

    num_supervised = 1000

    train_loader = data_utils.DataLoader(
        MnistRotated(
            list_train_domains,
            list_test_domains,
            num_supervised,
            seed,
            "/home/tarkus/Desktop/WILDS/DomainBed/domainbed/data/",
            train=True,
        ),
        batch_size=100,
        shuffle=False,
    )

    xs, ys, es = [], [], []

    ## Extend channels from 1 to 3
    for i, (x, y, d) in enumerate(train_loader):
        x = torch.cat(
            (
                x,
                x,
                x,
            ),
            1,
        )
        xs.append(x)
        ys.append(y)
        es.append(d)

    test_loader = data_utils.DataLoader(
        MnistRotated(
            list_train_domains,
            list_test_domains,
            num_supervised,
            seed,
            "/home/tarkus/Desktop/WILDS/DomainBed/domainbed/data/",
            train=False,
        ),
        batch_size=100,
        shuffle=False,
    )

    xs_test, ys_test, es_test = [], [], []
    ## Extend channels from 1 to 3
    for i, (x, y, d) in enumerate(test_loader):
        x = torch.cat(
            (
                x,
                x,
                x,
            ),
            1,
        )
        xs_test.append(x)
        ys_test.append(y)
        es_test.append(d)

    xs = torch.cat(xs)
    ys = torch.cat(ys)
    es = torch.cat(es)

    xs_mean, xs_std = xs.mean(0), xs.std(0)
    xs = (xs-xs_mean) #/ xs_std


    xs_test = torch.cat(xs_test)
    ys_test = torch.cat(ys_test)
    es_test = torch.cat(es_test)
    xs_test = (xs_test-xs_mean) #/ xs_std

    ## Unnecessary:
    idx = torch.randperm(xs_test.shape[0])
    xs_test, ys_test, es_test = xs_test[idx], ys_test[idx], es_test[idx]

    idx = torch.randperm(xs.shape[0])
    xs, ys, es= xs[idx], ys[idx], es[idx]

    n_train = int(0.8*xs.shape[0] )

    return (
        TensorDataset(xs, ys, es),
        TensorDataset(xs, ys, es),
        TensorDataset(xs[n_train:], ys[n_train:], es[n_train:]),
        TensorDataset(xs_test, ys_test, es_test),
    )
