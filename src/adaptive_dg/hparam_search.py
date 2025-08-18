
import argparse

from copy import deepcopy
import  yaml
import pytorch_lightning as pl

from adaptive_dg.models.base_classes import BaseClassClassifiersHParams

from ray import tune
from ray import air 
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import importlib

import lightning.pytorch as L

from ray.tune.schedulers import ASHAScheduler

from adaptive_dg.models import *


# Due to conflicts we need to redefine the TuneReportCallback
# (see https://github.com/ray-project/ray/issues/33426 )
class _TuneReportCallback(TuneReportCallback, L.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def create_train_function(model_name):

    def train(config):    
        config_copy = deepcopy(config)
        hparams = BaseClassClassifiersHParams(**config_copy)
                
        # load model from module
        module = importlib.import_module("adaptive_dg.models.all_in_one")
        model = getattr(module, model_name)(hparams)


        callback = _TuneReportCallback(
            {"loss": "validation/loss"},
            on="validation_end")  
        trainer_kwargs = {"callbacks": [callback]}
        model.fit(trainer_kwargs=trainer_kwargs)

    return train

if __name__ == "__main__":   
    parser = argparse.ArgumentParser(description="Run a sweep")

    parser.add_argument(
        "--dataset_name", required=True, type=str
    )
    parser.add_argument("--test_domain", required=True,  type=str)
    parser.add_argument("--model_name", type=str, default="PredictYfromXSet")
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--grace_period", type=int, default=5)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_cpus", type=int, default=6)

    
    with open("config_criteria.yml", "r") as file:
        configuration = yaml.load(file, Loader=yaml.FullLoader)
    with open("configs/data_sets.yml", "r") as file:
        configuration_dataset = yaml.load(file, Loader=yaml.FullLoader)
        
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    test_domain = int(args.test_domain) if args.test_domain.isdigit() else args.test_domain

    hparams_dic = configuration['ModelHParams']
    hparams_dic['dataset'] = dataset_name

    id_domains = list(configuration_dataset[dataset_name]['domains'])
    id_domains.remove(test_domain)
    ood_domains = [test_domain]
    hparams_dic.update({'id_domains': id_domains, 'ood_domains':  ood_domains})

    param_space = {'optimizer': {"lr": tune.loguniform(1e-6, 1e-3), 'name': "Adam", "weight_decay": tune.loguniform(1e-5, 1e-2)},
        'lr_scheduler': {'name': 'StepLR', 'step_size':  tune.choice([2000,5000, 10000]), 'gamma': 0.5},
        'batch_size': tune.choice([16, 32, 64]),
        'set_size': tune.choice([4, 8]),
        'gradient_clip': 2.0,
        'encoder_hparams': {
            'latent_dim': tune.qlograndint(256, 1024, 2, 2),
            'output_dim':  len(id_domains),
            'layer_width': tune.qlograndint(256, 1024, 2, 2),
            'p': tune.uniform(0.0, 0.5),
            'pma_args': {
                'num_heads': tune.choice([8, 16, 32, 64])
            },
            'name': 'Standard'
        }
    }

    hparams_dic.update(param_space)
    hparams_dic['max_epochs'] = args.max_epochs
    
    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='loss',
        mode='min',
        max_t=args.max_epochs,
        grace_period=args.grace_period,
        reduction_factor=3,
        brackets=1,
    )

    train = create_train_function(args.model_name)

    tuner = tune.Tuner(
        tune.with_resources(train, {"cpu": args.num_cpus, "gpu":args.num_gpus}),
        tune_config=tune.TuneConfig(metric="loss", mode="min", num_samples=args.num_runs, scheduler=tune.schedulers.ASHAScheduler(max_t=args.max_epochs)),
        param_space=hparams_dic,
    )

    results = tuner.fit()

    print(f"Best hyperparameters for model {args.model_name} found were: ", results.get_best_result().config)
    results_all_hparams = dict(BaseClassClassifiersHParams(**results.get_best_result().config))

    try: 
        with open(f"hparams_search_{dataset_name.lower()}.yml", "r") as file:
            results_dict = yaml.load(file, Loader=yaml.FullLoader)

        results_dict[args.model_name] = results_all_hparams
        
        with open(f"hparams_search_{dataset_name.lower()}.yml", 'w') as outfile:
            yaml.dump(results_dict, outfile, default_flow_style=False)
    
    except:
        results_dict = {args.model_name: results_all_hparams}
        with open(f"hparams_search_{dataset_name.lower()}.yml", 'w') as outfile:
            yaml.dump(results_dict, outfile, default_flow_style=False)