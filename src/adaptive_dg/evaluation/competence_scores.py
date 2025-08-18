import faiss
import torch 
import numpy as np

from adaptive_dg.evaluation.model_loading import load_model

# has no effect. TODO: remove
normalizer = lambda x: x 

def create_features(model, set_feature=True):

    ref_summaries = []
    for x, x_set, y, e in model.val_dataloader()[0]:

        with torch.no_grad():
            if set_feature:
                ref_summary = model.feature_set(x, x_set=x_set)
            else:
                ref_summary = model.feature(x)
            ref_summaries.append(ref_summary.cpu())

    summaries_id_test = []
    for x, x_set, y, e in model.val_dataloader()[1]:
        with torch.no_grad():
            if set_feature:
                summary_id_test = model.feature_set(x, x_set=x_set)
            else:
                summary_id_test = model.feature(x)
            summaries_id_test.append(summary_id_test.cpu())

    summaries_ood_test = []
    for x, x_set, y, e in model.val_dataloader()[2]:
        with torch.no_grad():
            if set_feature:
                summary_ood_test = model.feature_set(x, x_set=x_set)
            else:
                summary_ood_test = model.feature(x)
            summaries_ood_test.append(summary_ood_test.cpu())
        
    summaries_id_train = []
    for x, x_set, y, e in model.train_dataloader():
        with torch.no_grad():
            if set_feature:
                summary_id_train = model.feature_set(x, x_set=x_set)
            else:
                summary_id_train = model.feature(x)
            summaries_id_train.append(summary_id_train.cpu())

    summaries_id_train = torch.cat(summaries_id_train)
    summaries_id_test = torch.cat(summaries_id_test)
    summaries_ood_test = torch.cat(summaries_ood_test)
    summaries_id_val = torch.cat(ref_summaries)

    return summaries_id_train, summaries_id_val, summaries_id_test, summaries_ood_test

def score_function_create(features_train, features_val, K=5):
    ftrain = normalizer(features_train)
    fval = normalizer(features_val)

    index = faiss.IndexFlatL2(ftrain.shape[1])
    index.add(ftrain)

    D, _ = index.search(fval, K)
    scores_iid = -D[:, -1]

    def score_function(features):
        fest = normalizer(features)
        D, _ = index.search(fest, K)
        scores_ood_test = -D[:, -1]
        return -scores_ood_test
    return -scores_iid, score_function

def compute_model_scores(model, set_feature=True, K=5): 

    summaries_id_train, summaries_id_val, summaries_id_test, summaries_ood = create_features(model, set_feature=set_feature)
    
    scores_id, score_function = score_function_create(summaries_id_train, summaries_id_val, K=K)
    scores_ood = score_function(summaries_ood)
    scores_id_test = score_function(summaries_id_test)
    
    return scores_id, scores_id_test, scores_ood