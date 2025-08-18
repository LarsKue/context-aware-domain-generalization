from torch import nn
import torch

from adaptive_dg.models.base_classes import BaseClassClassifiers
from adaptive_dg.models.target_predict import EnvClassifier

from adaptive_dg.models.utils import ECELoss

from sklearn.decomposition import PCA
import adaptive_dg.models.utils_networks as utils_networks

import lightning_trainable
from adaptive_dg.models.domain_generalizer import DomainGeneralizer, DGHParams
from lightning_trainable.modules import FullyConnectedNetwork

from adaptive_dg.models.criteria.utils import _mmd, Predictor
from adaptive_dg.models.criteria.base_model import BaseModel, BaseModelHParams



class PredYfromXSet(BaseModel):
    """Classifiert that predicts class Y from input X and Set-Input

    Args:
        BaseClassClassifiers (_type_): _description_
    """
    hparams: BaseModelHParams 

    def __init__(self, hparams):
        super().__init__(hparams)

        # Standard predictor 
        self.feature_extractor = self.configure_feature_extractor()
        self.set_encoder = self.configure_encoder()
        self.predictor = Predictor(hparams.featurizer_hparams['output_dim'], 
                                   self.hparams.n_classes, 
                                   cond_dim=self.hparams.encoder_hparams['dimensions'][-1],
                                   layer_width=self.hparams.classifier_hparams['layer_width'],
                                   depth=self.hparams.classifier_hparams['depth'],
                                   )

        if self.hparams.task == "classification":        
            self.ece_loss = ECELoss(n_bins=15)
        
        match self.hparams.loss_fct:
            case 'mse':
                self.loss_fct = nn.MSELoss()
            case 'cross_entropy':
                self.loss_fct = nn.CrossEntropyLoss()
            case other:
                raise NotImplementedError(f"Unrecognized loss choice: '{other}'")
            
            
    def compute_metrics(self, batch, batch_idx):
        
        x, x_set, y, e = batch       

        x, x_set, e, y = x.float(), x_set.float(),  e.long(), y.float()
        batch_size, set_size, *_ = x_set.shape
        x_set = x_set.view(batch_size * set_size, *x_set.shape[2:])

        # Workaround
        x, x_set, e, y = x.cuda(), x_set.cuda(),  e.cuda(), y.cuda()

        out_dict = {}
       
        
        # Set model
        if self.hparams.feature_train:
            feature = self.feature_extractor(x)
            feature_dim = feature.shape[1]
            set_feature = self.feature_extractor(x_set)
        else:
            with torch.no_grad():
                feature = self.feature_extractor(x)
                feature_dim = feature.shape[1]
                set_feature = self.feature_extractor(x_set)
                
        set_feature = set_feature.view(batch_size, set_size,  feature_dim)
        set_summary = self.set_encoder(set_feature)
            
            
        y_pred_standard = self.predictor(feature, c=set_summary)

        loss = self.loss_fct(y_pred_standard, y)

        if self.hparams.mmd_lambda > 0:
            mmd_set_embedding = _mmd(set_summary, factor_z=2)
            loss += self.hparams.mmd_lambda * mmd_set_embedding
            out_dict['mmd_set_enhanced'] = mmd_set_embedding.item()
        
        out_dict['loss'] = loss
        out_dict['loss_set_enhanced'] = loss.item()

        if self.hparams.task == "classification":
            out_dict['acc_set_enhanced'] = (y_pred_standard.argmax(1) == y.argmax(1)).float().mean().item()
            out_dict['ece_set_enhanced'] = self.ece_loss(y_pred_standard, y.argmax(1)).item()

            out_dict['acc'] = out_dict['acc_set_enhanced']
            out_dict['ece'] = out_dict['ece_set_enhanced']
            out_dict['metric'] = out_dict['acc'] 
        else:
            out_dict['metric'] = loss.item()

        return out_dict

    def compute_ood_metrics(self, batch, batch_idx):
        return self.compute_metrics(batch, batch_idx)    


    def predict(self, x, x_set=None, e=None):

        assert e == None, "environment info must not be provided for this model"

        batch_size = x.shape[0]
        set_size = x_set.shape[1]
        # Set model
        feature = self.feature_extractor(x)
        feature_dim = feature.shape[1]
        x_set_tmp = torch.flatten(x_set, start_dim=0, end_dim=1)
        set_feature = self.feature_extractor(x_set_tmp)
        set_feature = set_feature.view(batch_size, set_size,  feature_dim)
        set_summary = self.set_encoder(set_feature)
            
            
        y_pred = self.predictor(feature, c=set_summary)
        return y_pred

    def feature_set(self, x, x_set=None, e=None):

        assert e == None, "environment info must not be provided for this model"

        batch_size = x.shape[0]
        set_size = x_set.shape[1]
        # Set model
        feature = self.feature_extractor(x)
        feature_dim = feature.shape[1]
        x_set_tmp = torch.flatten(x_set, start_dim=0, end_dim=1)
        set_feature = self.feature_extractor(x_set_tmp)
        set_feature = set_feature.view(batch_size, set_size,  feature_dim)
        set_summary = self.set_encoder(set_feature)
        
        return set_summary