"""
Module to train models
"""

import importlib

import argparse

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
    parser.add_argument("--latent_dim_total", type=int, default=64)

    parser.add_argument("--model_name", type=str, default="GaussianMixtureINN")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--augment", type=bool, default=False)

    args = parser.parse_args()

    dataset_name = args.dataset_name


    for test_domain in args.test_domains:

        G_kwargs = {
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "latent_dim_total": args.latent_dim_total,
            "optimizer": {"name": "Adam", "lr": args.lr},
        }

        (
            kwargs,
            dataset_id_train,
            dataset_id_val,
            dataset_id_test,
            dataloader_ood,
        ) = get_datasets(
            args.dataset_name,
            test_domain=test_domain,
            augment=args.augment,
        )
        G_kwargs.update({"img_resolution": kwargs["img_resolution"]})

        module_models = importlib.import_module(
            "adaptive_dg.models.autoencoders"
        )  # may raise ImportError

        module = getattr(module_models, args.model_name)
        MyHParams = getattr(module_models, args.model_name + "HParams")
        hparams = MyHParams(**G_kwargs)

        model = module(hparams)

        model.train_data = dataset_id_train
        model.val_data = dataset_id_val
        model.ood_data = dataloader_ood
        model.ood_dataload = model.ood_dataloader()

        model.fit(
            logger_kwargs={
                "save_dir": args.output_dir,
                "name": f"{args.model_name}_{test_domain}",
            }
        )
