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


class PredEfromX(BaseModel):
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
            len(self.hparams.id_domains), 
            cond_dim=0,
            layer_width=self.hparams.classifier_hparams['layer_width'],
            depth=self.hparams.classifier_hparams['depth'],
        )
        
        self.ece_loss = ECELoss(n_bins=15)
        
        self.loss_fct = nn.CrossEntropyLoss()
            
            
    def compute_metrics(self, batch, batch_idx):
        
        x, x_set, y, e = batch       

        x, x_set, e, y = x.float(), x_set.float(),  e.float(), y.float()
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
            
            
        e_pred = self.predictor(feature_standard)

        loss = self.loss_fct(e_pred, e)

        if self.hparams.mmd_lambda > 0:
            mmd_feature_embedding = _mmd(feature_standard, factor_z=2)
            loss += self.hparams.mmd_lambda * mmd_feature_embedding
            out_dict['mmd_env_enhanced'] = mmd_feature_embedding#.item()
        
        out_dict['loss'] = loss

        with torch.no_grad():
            out_dict['acc_env'] = (e_pred.argmax(1) == e.argmax(1)).float().mean().item()
            out_dict['ece_env'] = self.ece_loss(e_pred, e.argmax(1)).item()

            out_dict['acc'] = out_dict['acc_env']
            out_dict['ece'] = out_dict['ece_env']

            out_dict['metric'] = out_dict['acc_env'] 

        return out_dict

    def compute_ood_metrics(self, batch, batch_idx):
        return self.compute_metrics(batch, batch_idx)  

    def feature(self, x):
        return self.feature_extractor(x)