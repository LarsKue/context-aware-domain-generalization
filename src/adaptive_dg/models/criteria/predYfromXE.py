from torch import nn
import torch

from adaptive_dg.models.base_classes import BaseClassClassifiers
from adaptive_dg.models.target_predict import EnvClassifier

from adaptive_dg.models.utils import ECELoss

from sklearn.decomposition import PCA
import adaptive_dg.models.utils_networks as utils_networks

from adaptive_dg.models.domain_generalizer import DomainGeneralizer, DGHParams
from lightning_trainable.modules import FullyConnectedNetwork

from adaptive_dg.models.criteria.utils import _mmd, Predictor
from adaptive_dg.models.criteria.base_model import BaseModel, BaseModelHParams



class PredYfromXE(BaseModel):
    """Classifiert that predicts class Y from input X and Set-Input

    Args:
        BaseClassClassifiers (_type_): _description_
    """
    hparams: BaseModelHParams 

    def __init__(self, hparams):
        super().__init__(hparams)

        # Standard predictor 
        self.feature_extractor = self.configure_feature_extractor()
        self.predictor = Predictor(hparams.featurizer_hparams['output_dim'], 
            self.hparams.n_classes, 
            cond_dim=len(self.hparams.id_domains),
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

        # Workaround
        x, x_set, e, y = x.cuda(), x_set.cuda(),  e.cuda(), y.cuda()

        out_dict = {}
       
        #standard model
        if self.hparams.feature_train:
            feature_standard = self.feature_extractor(x)
        else:
            with torch.no_grad():
                feature_standard = self.feature_extractor(x)
            
            
        y_pred_standard = self.predictor(feature_standard, c=e)

        loss = self.loss_fct(y_pred_standard, y)

        if self.hparams.mmd_lambda > 0:
            mmd_feature_embedding = _mmd(feature_standard, factor_z=2)
            loss += self.hparams.mmd_lambda * mmd_feature_embedding
            out_dict['mmd_env_enhanced'] = mmd_feature_embedding.item()
        
        out_dict['loss'] = loss#.item()
        out_dict['loss_env_enhanced'] = loss#.item()

        if self.hparams.task == "classification":
            out_dict['acc_env_enhanced'] = (y_pred_standard.argmax(1) == y.argmax(1)).float().mean().item()
            out_dict['ece_env_enhanced'] = self.ece_loss(y_pred_standard, y.argmax(1)).item()

            out_dict['acc'] = out_dict['acc_env_enhanced']
            out_dict['ece'] = out_dict['ece_env_enhanced']
            out_dict['metric'] = out_dict['acc']
        else:
            out_dict['metric'] = loss.item()

        return out_dict

    def compute_ood_metrics(self, batch, batch_idx):
        return self.compute_metrics(batch, batch_idx)    

    def predict(self, x, x_set=None, e=None):
        feature_standard = self.feature_extractor(x)
        y_pred = self.predictor(feature_standard, c=e)
        return y_pred