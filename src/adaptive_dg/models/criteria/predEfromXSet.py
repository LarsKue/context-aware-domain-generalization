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



class PredEfromXSet(BaseModel):
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
            cond_dim=self.hparams.encoder_hparams['dimensions'][-1],
            layer_width=self.hparams.classifier_hparams['layer_width'],
            depth=self.hparams.classifier_hparams['depth'],
        )

        self.set_encoder = self.configure_encoder()

        self.ece_loss = ECELoss(n_bins=15)
        
        self.loss_fct = nn.CrossEntropyLoss()

    def compute_metrics(self, batch, batch_idx):
        
        x, x_set, y, e = batch       

        x, x_set, e, y = x.float(), x_set.float(),  e.float(), y.float()
        batch_size, set_size, *_ = x_set.shape

        # Workaround
        x, x_set, e, y = x.cuda(), x_set.cuda(),  e.cuda(), y.cuda()

        out_dict = {}
               
        # Set model
        if self.hparams.feature_train:
            feature = self.feature_extractor(x)
            feature_dim = feature.shape[1]
            x_set_tmp = torch.flatten(x_set, start_dim=0, end_dim=1)
            set_feature = self.feature_extractor(x_set_tmp)

        else:
            with torch.no_grad():
                feature = self.feature_extractor(x)
                feature_dim = feature.shape[1]
                x_set_tmp = torch.flatten(x_set, start_dim=0, end_dim=1)
                set_feature = self.feature_extractor(x_set_tmp)
                
        set_feature = set_feature.view(batch_size, set_size,  feature_dim)
        set_summary = self.set_encoder(set_feature)       
          
        e_pred = self.predictor(feature, c=set_summary)

        loss = self.loss_fct(e_pred, e)
        
        if self.hparams.mmd_lambda > 0:
            mmd_feature_embedding = _mmd(set_summary, factor_z=2)
            loss += self.hparams.mmd_lambda * mmd_feature_embedding
            out_dict['mmd_standard'] = mmd_feature_embedding.item()
            
        if self.hparams.mmd_lambda > 0:
            mmd_feature_embedding = _mmd(feature, factor_z=2)
            loss += self.hparams.mmd_lambda * mmd_feature_embedding
            out_dict['mmd_env_enhanced'] = mmd_feature_embedding#.item()
        
        out_dict['loss'] = loss

        with torch.no_grad():
            out_dict['acc_env'] = (e_pred.argmax(1) == e.argmax(1)).float().mean().item()
            out_dict['ece_env'] = self.ece_loss(e_pred, e.argmax(1)).item()

            out_dict['acc'] = out_dict['acc_env']
            out_dict['ece'] = out_dict['ece_env']
            out_dict['metric'] = out_dict['acc'] 

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
            
            