
from torch import nn
import torch
from lightning_trainable import utils

from adaptive_dg.models.base_classes import BaseClassClassifiers

class PredictYfromXSet(BaseClassClassifiers):
    """Classifiert that predicts class Y from input X and Set-Input

    Args:
        BaseClassClassifiers (_type_): _description_
    """

    def __init__(self, hparams):
        super().__init__(hparams)

        self.feature_extractor = self.configure_feature_extractor()
        self.feature_extractor_set = self.configure_feature_extractor()
        self.encoder = self.configure_encoder()        

        match self.hparams.loss_fct:
            case 'mse':
                self.loss_fct = nn.MSELoss()
            case 'cross_entropy':
                self.loss_fct = nn.CrossEntropyLoss()
            case other:
                raise NotImplementedError(f"Unrecognized loss choice: '{other}'")

        self.class_layer = nn.Sequential(nn.Linear(self.hparams.encoder_hparams['output_dim'], 512), nn.ReLU(), nn.Linear(512,512), nn.ReLU(), nn.Linear(512, len(self.hparams.id_domains)))
        self.loss_classifier = nn.CrossEntropyLoss()
        self.class_classifier = EnvClassifier(hparams.featurizer_hparams['output_dim'], self.n_classes, self.hparams.encoder_hparams['output_dim'])

    def forward(self, x, x_set):

        feature = self.feature_extractor(x)
        ####
        # Concat first and sceond dimension x_set
        x_set = x_set.view(x_set.shape[0] * x_set.shape[1], *x_set.shape[2:])
        if self.hparams.feature_train:
            feature_set = self.feature_extractor_set(x_set)
        else:
            with torch.no_grad():
                feature_set = self.feature_extractor_set(x_set)
                                
        feature_set = feature_set.view(x.shape[0], -1,  feature.shape[1])
        set_summary = self.encoder(feature_set)
        
        e_pred = self.class_layer(set_summary)
        
        y_pred = self.class_classifier(feature, set_summary )

        return y_pred, e_pred

    def compute_metrics(self, batch, batch_idx):
        x, x_set, y, e = batch       

        x, x_set, e, y = x.float(), x_set.float(),  e.long(), y.float()

        y_pred, e_pred = self.forward(x, x_set)

        loss= self.loss_fct(y_pred, y.float())
        loss += 1*self.loss_classifier(e_pred, e.float())

        out_dict = {'loss': loss}

        if  self.hparams.task == 'classification':
            # Accuracies for different predictions
            acc_y = (y_pred.argmax(1) == y.argmax(1)).sum()/y.shape[0]
            out_dict['acc_y'] = acc_y
                
        if self.hparams.task== 'regression':
            out_dict['mse'] = nn.MSELoss()(y_pred, y)
            
        return out_dict

    def compute_ood_metrics(self, batch, batch_idx):
        return self.compute_metrics(batch, batch_idx)


class PredictYfromX(BaseClassClassifiers):
    """Classifier that predicts class Y from input X

    Args:
        BaseClassClassifiers (_type_): _description_
    """

    def __init__(self, hparams):
        super().__init__(hparams)

        self.feature_extractor = self.configure_feature_extractor()
        self.class_classifier = EnvClassifier(hparams.featurizer_hparams['output_dim'], self.n_classes,len(hparams.id_domains))

        match self.hparams.loss_fct:
            case 'mse':
                self.loss_fct = nn.MSELoss()
            case 'cross_entropy':
                self.loss_fct = nn.CrossEntropyLoss()
            case other:
                raise NotImplementedError(f"Unrecognized loss choice: '{other}'")


    def compute_metrics(self, batch, batch_idx):
        x, _, y, e = batch       

        x, e, y = x.float(), e.long(), y.float()

        if self.hparams.feature_train:
            feature = self.feature_extractor(x)
        else: 
            with torch.no_grad():
                feature = self.feature_extractor(x)
            

        y_pred = self.class_classifier(feature)

        loss = self.loss_fct(y_pred, y.float())
    
        out_dict  = dict(loss=loss)
    
        if  self.hparams.task == 'classification':
            # Accuracies for different predictions
            acc_y = (y_pred.argmax(1) == y.argmax(1)).sum()/y.shape[0]
            out_dict['acc_y'] = acc_y
        
        if self.hparams.task== 'regression':
            out_dict['mse'] = nn.MSELoss()(y_pred, y)    
        return out_dict


    def compute_ood_metrics(self, batch, batch_idx):
        return self.compute_metrics(batch, batch_idx)



class EnvClassifier(nn.Module):
    """
    Classsifier that can use the environment information
    """
    def __init__(self, input_dim, output_dim, env_dim, layer_width=2048):
        super().__init__()
        self.env_dim = env_dim

        self.l1 = nn.Linear(input_dim+env_dim, layer_width)
        self.l2 = nn.Linear(layer_width+env_dim, layer_width)
        self.l3 = nn.Linear(layer_width+env_dim, output_dim)
        self.r = nn.ReLU()

    def forward(self, x, e=None):
        if e is None:
            e = torch.zeros(x.shape[0], self.env_dim).to(x.device)
        x = torch.cat((x, e), 1)
        x = self.l1(x)
        x = self.r(x)
        x = torch.cat((x, e), 1)
        x = self.l2(x)
        x = self.r(x)
        x = torch.cat((x, e), 1)
        x = self.l3(x)
        return x

class PredictYfromXE(BaseClassClassifiers):
    """Implements the third criterion: I([X,E]; Y) - I(X;Y)
    
    This criterion checks whether the ground-truth environments adds to the predictive performance
    """
    def __init__(self, hparams):
        super().__init__(hparams)

        self.feature_extractor = self.configure_feature_extractor()
        self.class_classifier = EnvClassifier(hparams.featurizer_hparams['output_dim'], self.n_classes, len(hparams.id_domains))

        match self.hparams.loss_fct:
            case 'mse':
                self.loss_fct = nn.MSELoss()
            case 'cross_entropy':
                self.loss_fct = nn.CrossEntropyLoss()
            case other:
                raise NotImplementedError(f"Unrecognized loss choice: '{other}'")

    def compute_metrics(self, batch, batch_idx):
        x, _, y, e = batch       
     
        x, e, y = x.float(), e.long(), y.float()

        if self.hparams.feature_train:
            feature = self.feature_extractor(x)
        else: 
            with torch.no_grad():
                feature = self.feature_extractor(x)
            
        y_pred_e = self.class_classifier(feature, e)

        loss = self.loss_fct(y_pred_e, y.float())

        out_dict  = dict(loss=loss)
    
        if  self.hparams.task == 'classification':
            # Accuracies for different predictions
            acc_y = (y_pred_e.argmax(1) == y.argmax(1)).sum()/y.shape[0]
            out_dict['acc_y'] = acc_y  
        if self.hparams.task== 'regression':
            out_dict['mse'] = nn.MSELoss()(y_pred_e, y)   
            
                 
        return out_dict

    def compute_ood_metrics(self, batch, batch_idx):
        return  dict(
            loss=0,
        ) 