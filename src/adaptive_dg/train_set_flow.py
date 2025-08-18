"""
Module to train models
"""

import importlib
import argparse

import yaml

from adaptive_dg.dataset_loader import get_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a sweep")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directoy where the results are saved",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directoy of the data"
    )
    parser.add_argument(
        "--dataset_name", required=True, type=str, default="RotatedMNIST"
    )
    parser.add_argument("--test_domains", nargs="+", type=int, default=[0])
    parser.add_argument("--latent_dims", nargs="+", type=int, default=[32, 32, 32])

    parser.add_argument("--model_name", type=str, default="GaussianMixtureINN")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_flow", type=float, default=1e-4)
    parser.add_argument("--lr_set", type=float, default=1e-6)
    parser.add_argument("--lr_auto", type=float, default=1e-5)
    parser.add_argument("--n_layers", type=int, default=7)
    parser.add_argument("--noise_scaling", type=float, default=1e-2)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--cond_dim", type=int, default=32)
    parser.add_argument("--reconstruction_weight", type=float, default=50.)
    parser.add_argument("--joint_training", type=bool, default=True)
    parser.add_argument("--augment", type=bool, default=False)

    args = parser.parse_args()

    dataset_name = args.dataset_name

    for test_domain in args.test_domains:
        with open("config.yml", "r") as file:
            configuration = yaml.safe_load(file)

        # Details to load pre-trained autoencoder
        autoencoder_cfg = {}
        autoencoder_cfg["path_checkpoint"] = configuration[
            f"AutoEncoderPaths_{dataset_name}"
        ][f"env_{test_domain}"]["path_checkpoint"]
        autoencoder_cfg["model_class"] = configuration[
            f"AutoEncoderPaths_{dataset_name}"
        ][f"env_{test_domain}"]["model_class"]

        latent_dim_total = sum(args.latent_dims)

        G_kwargs = {
            "noise_scaling": args.noise_scaling,
            "latent_dim_total": latent_dim_total,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "joint_training": args.joint_training,
            "autoencoder_cfg": autoencoder_cfg,
            "optimizer": {
                "name": "Adam",
                "lr_flow": args.lr_flow,
                "lr_set": args.lr_set,
                "lr_auto": args.lr_auto,
            },
            "n_layers_flow": args.n_layers,
            "cond_dim": args.cond_dim,
            "reconstruction_weight": args.reconstruction_weight,
        }

        (
            kwargs,
            dataset_id_train,
            dataset_id_val,
            dataset_id_test,
            dataset_ood,
        ) = get_datasets(
            args.dataset_name,
            test_domain=test_domain,
            augment=args.augment,
        )

        module_models = importlib.import_module(
            "adaptive_dg.models.cinn_set"
        )  # may raise ImportError

        module = getattr(module_models, args.model_name)
        MyHParams = getattr(module_models, args.model_name + "HParams")
        hparams = MyHParams(**G_kwargs)

        model = module(hparams)

        model.train_data = dataset_id_train
        from torch.utils.data import IterableDataset
        print('shuffle ', not isinstance(model.train_data, IterableDataset))
        model.val_data = dataset_id_val
        model.ood_data = dataset_ood 
        model.ood_dataload = model.ood_dataloader()

        model.fit(
            logger_kwargs={
                "save_dir": args.output_dir,
                "name": f"{args.model_name}_{test_domain}",
            }
        )
