import torch
import yaml
from pathlib import Path

from adaptive_dg.models.criteria import PredEfromXSet
from adaptive_dg.models.criteria import PredEfromX, PredYfromX, PredYfromXE, PredYfromXSet 
from adaptive_dg.models.criteria.base_model import BaseModelHParams

from adaptive_dg.evaluation import find_events, events_to_dataframe



def load_losses(directory, data_set, domain, model_class, seed=42, version=0):
    
    logdir = f"{directory}/{data_set}/{str(domain)}/{str(seed)}/{model_class}/"
    path = Path(logdir)
    path = path / f"version_{version}"
    number = int(str(list(path.glob("events.out.tfevents.*.*"))[0])[-1])
    
    if number %2 == 1:
        events = list(path.glob("events.out.tfevents.*.*"))[0]
    elif number %2 == 0 and len(list(path.glob("events.out.tfevents.*.*"))) == 1 :
        events = list(path.glob("events.out.tfevents.*.*"))[0]
    else:
        events = list(path.glob("events.out.tfevents.*.*"))[1]
    events = str(events)
    df = events_to_dataframe(events)
    df = df.convert_dtypes()
    # Remove first element
    df = df[1:]

    return df 

def model_selection(directory, data_set, domain, model_class, seed=42, version=0):
    
    df = load_losses(directory, data_set, domain, model_class, seed=seed, version=version)

    losses_val = df[[ "validation/loss"]].dropna()
    model_selection = losses_val.idxmin().iloc[0]

    return model_selection

def compute_best_metrics(directory, data_set, domain, model_class, seed=42, version=0):
    df = load_losses(directory, data_set, domain, model_class, seed=seed, version=version)
    model_selection_pure = model_selection(directory, data_set, domain, model_class, seed=seed, version=version)

    metric_val = df[["validation/metric"]].dropna().loc[model_selection_pure].item()
    metric_ood = df[["ood/metric"]].dropna().loc[model_selection_pure].item()
    metric_test = df[["test/metric"]].dropna().loc[model_selection_pure].item()
    
    return metric_val,  metric_test, metric_ood

def load_model(model_class, domain, directory, seed=42, data_set="PACS", version=0):
    
    df =load_losses(directory, data_set, domain, model_class, version=version)
    step = model_selection(directory, data_set, domain, model_class, seed=seed, version=version) 
    epoch = df['epoch'].dropna()[step]

    loading_step = int(step + 1)
    checkpoint_path = f"{directory}/{data_set}/{str(domain)}/{str(seed)}/{model_class}/version_{version}/checkpoints/epoch={epoch}-step={loading_step}.ckpt"
    # load hparams path
    hparams_path = f"{directory}/{data_set}/{str(domain)}/{str(seed)}/{model_class}/version_{version}/hparams.yaml"
    # load hpaarams
    with open(hparams_path) as file:
        hparams = yaml.load(file, Loader=yaml.Loader)

    hparams = BaseModelHParams(**hparams)

    model =eval(model_class)(hparams)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict'])
    return model
    
