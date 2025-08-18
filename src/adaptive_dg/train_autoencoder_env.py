"""
Module to train models
"""

import importlib
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a sweep")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directoy where the results are saved",
    )

    parser.add_argument(
        "--dataset_name", required=True, type=str, default="VLCS"
    )
    parser.add_argument("--test_domains", nargs="+", type=str, default=["V"])
    parser.add_argument("--train_domains", nargs="+", type=str, default=["L","C","S"])

    parser.add_argument("--model_name", type=str, default="AutoEncoder")
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--input_channel", type=int, default=3)
    parser.add_argument("--set_size", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--devices",nargs="+",type=int, default=[0])
    parser.add_argument("--maxpool",type=bool,default=False)

    args = parser.parse_args()

    dataset_name = args.dataset_name

    for test_domain in args.test_domains:

        enc_hparams = {
            "latent_dim": args.latent_dim,
            "input_channel": args.input_channel,
        }
        

        G_kwargs = {
            "devices": args.devices,
            "dataset": args.dataset_name,
            "data_seed": args.data_seed,
            "batch_size": args.batch_size,
            "set_size": args.set_size,
            "max_epochs": args.max_epochs,
            "ood_domains": args.test_domains,
            "id_domains": args.train_domains,
            "enc_hparams": enc_hparams,
        }

        if "Set" in args.model_name:
            set_hparams = {
            "latent_dim": 128,
            "layer_width": 256,
            "output_dim": len(args.train_domains)-1,
            "name": "convolutional",
            "pma_args": {"num_heads": 16},
            "maxpool": args.maxpool
            }
            G_kwargs["encoder_hparams"] = set_hparams
            G_kwargs["strategy"] = "repeat"

        module_models = importlib.import_module(
            "adaptive_dg.models.reconstruction_nets"
        )  # may raise ImportError

        module = getattr(module_models, args.model_name)
        model = module(G_kwargs)

        model.fit(
            logger_kwargs={
                "save_dir": args.output_dir,
                "name": f"{args.model_name}_{test_domain}",
            }
        )
