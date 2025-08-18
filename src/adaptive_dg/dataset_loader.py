from adaptive_dg.datasets.rotated_mnist_diva import get_dataloader_rmnist_diva
from adaptive_dg.datasets.colored_mnist import get_dataloader_colored_mnist


def get_datasets(dataset, test_domain=0, batch_size=32, seed=0, augment=True):
    assert dataset in ["RotatedMNISTDiva",  "ColoredMNIST"]

    if dataset == "RotatedMNISTDiva":
        print(augment)
        if augment == True:
            raise NotImplementedError

        kwargs = {}
        kwargs["channels"] = 3
        kwargs["img_resolution"] = 32

        kwargs["n_envs"] = 5
        kwargs["n_classes"] = 10

        return kwargs, *get_dataloader_rmnist_diva(
            seed=seed, test_env=test_domain, extend_dim=False
        )

    elif dataset == "RotatedMNIST":
        raise NotImplementedError

    elif dataset == "ColoredMNIST":

        kwargs = {}
        kwargs["channels"] = 1
        kwargs["img_resolution"] = 28 

        kwargs["n_envs"] = 2
        kwargs["n_classes"] = 2 

        return kwargs, *get_dataloader_colored_mnist(
            test_domain=test_domain
        )


    elif dataset == "Malaria":
        raise NotImplementedError

    elif dataset == "SkinLession":
        raise NotImplementedError
