"""
Module with which the criteria can be checked
"""

import itertools
import argparse
import importlib
from copy import deepcopy

import  yaml
import pytorch_lightning as pl

#from adaptive_dg.models.base_classes import BaseClassClassifiersHParams
#from adaptive_dg.models.criteria import PredEfromXHParams 
from adaptive_dg.models.criteria.base_model import BaseModelHParams

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a sweep")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directoy where the results are saved",
    )

    parser.add_argument(
        "--dataset_name", required=True, type=str, default="RotatedMNIST"
    )
    parser.add_argument("--test_domains", nargs="+", type=str,  default=["V"])
    parser.add_argument("--model_names", nargs="+", type=str, default=["PredictYfromX"])
    parser.add_argument("--training_seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--dataset_seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--hparams_file", type=str, default="hparams.yml")
    args = parser.parse_args()

    with open("configs/data_sets.yml", "r") as file:
        configuration_datasets = yaml.load(file, Loader=yaml.FullLoader)

    for test_domain, model_name, training_seed, dataset_seed in itertools.product(args.test_domains, args.model_names, args.training_seeds, args.dataset_seeds):

        dataset_name = args.dataset_name
        
        # Load hyperparameters
        with open(args.hparams_file, "r") as file:
            hparams_dict = yaml.load(file, Loader=yaml.FullLoader)[model_name]

        # Create Id and Ood domainTrue:#s
        id_domains = list(configuration_datasets[dataset_name]['domains'])
        id_domains.remove(test_domain)
        ood_domains = [test_domain]
        hparams_dict.update({'id_domains': id_domains, 'ood_domains':  ood_domains})
        hparams_dict.update({"n_classes": configuration_datasets[dataset_name]['n_classes']})

        # Set seeds
        hparams_dict.update({'training_seed': training_seed, 'data_seed': dataset_seed})
        hparams_dict.update({'max_epochs': args.max_epochs}) 
        
        hparams = BaseModelHParams(**deepcopy(hparams_dict))
        module = importlib.import_module("adaptive_dg.models.criteria")
        model = getattr(module, model_name)(hparams).cuda()
            
        pl.seed_everything(hparams.training_seed)

        model.fit(
            logger_kwargs={
                "save_dir": args.output_dir,
                "name": f"{hparams.dataset}/{test_domain}/{dataset_seed}/{model_name}",
            },
        )