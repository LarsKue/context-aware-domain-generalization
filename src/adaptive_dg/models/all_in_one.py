from torch import nn
import torch

from adaptive_dg.models.base_classes import BaseClassClassifiers
from adaptive_dg.models.target_predict import EnvClassifier

from adaptive_dg.models.utils import ECELoss


class AllInOne(BaseClassClassifiers):
    """Classifiert that predicts class Y from input X and Set-Input

    Args:
        BaseClassClassifiers (_type_): _description_
    """

    def __init__(self, hparams):
        super().__init__(hparams)

        self.feature_extractor = self.configure_feature_extractor()
        self.encoder = self.configure_encoder()
        
        self.class_classifier = EnvClassifier(hparams.featurizer_hparams['output_dim'], self.n_classes, len(self.hparams.id_domains)+3)

        self.env_classifier_set_input = self.configure_simple_classifier(self.hparams.encoder_hparams['output_dim'], len(self.hparams.id_domains))
        self.env_classifier_conv_input = self.configure_simple_classifier(self.hparams.featurizer_hparams['output_dim'] , len(self.hparams.id_domains))

        self.loss_classifier = nn.CrossEntropyLoss()

        match self.hparams.loss_fct:
            case 'mse':
                self.loss_fct = nn.MSELoss()
            case 'cross_entropy':
                self.loss_fct = nn.CrossEntropyLoss()
            case other:
                raise NotImplementedError(f"Unrecognized loss choice: '{other}'")

    def forward_from_feature(self, feature, set_summary, e,  mode='pure'):
        """Forward pass of the model (without feature computation)
        """
        
        assert mode in ['pure', 'set', 'env']
        
        if mode == 'set':
            one_hot = torch.zeros(e.shape[0], 3).to(self.device)
            one_hot[:, 0] = 1
            if set_summary.shape[1] < e.shape[1]:
                zero_padding = torch.zeros(set_summary.shape[0], e.shape[1] - set_summary.shape[1]).to(self.device)
                set_summary = torch.cat((set_summary, zero_padding), 1)
            summary_only = torch.cat((set_summary, one_hot), 1)
            ye_pred = self.class_classifier(feature, summary_only)
        
        if mode == 'pure':
            one_hot = torch.zeros(e.shape[0], 3).to(self.device)
            one_hot[:, 1] = 1
            zero_padding = torch.zeros(e.shape[0], max(e.shape[1], self.hparams.encoder_hparams['output_dim'])).to(e.device)
            summary_only = torch.cat((zero_padding, one_hot), 1)
            ye_pred = self.class_classifier(feature, summary_only)
        
        if mode == 'env':
            one_hot = torch.zeros(e.shape[0], 3).to(self.device)
            one_hot[:, 2] = 1
            if e.shape[1] < self.hparams.encoder_hparams['output_dim']:
                zero_padding = torch.zeros(e.shape[0], self.hparams.encoder_hparams['output_dim'] - e.shape[1]).to(self.device)
                e = torch.cat((e, zero_padding), 1)
            summary_only = torch.cat((e, one_hot), 1)
            ye_pred = self.class_classifier(feature, summary_only)

        return ye_pred[:, :self.n_classes]
    
    def encode_set(self, x_set):
        
        batch_size, set_size, *_ = x_set.shape
        feature_dim = self.hparams.featurizer_hparams['output_dim']

        # Concat first and second dimension x_set
        x_set = x_set.view(batch_size * set_size, *x_set.shape[2:])
        feature_set = self.feature_extractor(x_set)
        feature_set = feature_set.view(batch_size, set_size,  feature_dim)
        set_summary = self.encoder(feature_set)
        return set_summary
    
    def compute_metrics(self, batch, batch_idx):
        x, x_set, y, e = batch       

        x, x_set, e, y = x.float(), x_set.float(),  e.long(), y.float()

        batch_size, set_size, *_ = x_set.shape
        
        if self.hparams.feature_train:
            feature = self.feature_extractor(x)
            feature_dim = feature.shape[1]

            x_set = x_set.view(batch_size * set_size, *x_set.shape[2:])
            feature_set = self.feature_extractor(x_set)
            feature_set = feature_set.view(batch_size, set_size,  feature_dim)
        else:
            with torch.no_grad():
                feature = self.feature_extractor(x)
                feature_dim = feature.shape[1]

                # Concat first and second dimension x_set
                x_set = x_set.view(batch_size * set_size, *x_set.shape[2:])
                feature_set = self.feature_extractor(x_set)
                feature_set = feature_set.view(batch_size, set_size,  feature_dim)
        set_summary = self.encoder(feature_set)
        
        # Environment Prediction
        env_pred_set = self.env_classifier_set_input(set_summary)
        env_pred_conv = self.env_classifier_conv_input(feature)
        
        # Target Prediction and Environment prediction
        y_pred_set = self.forward_from_feature(feature, set_summary, e, mode='set')
        y_pred_pure= self.forward_from_feature(feature, set_summary, e, mode='pure')
        y_pred_env = self.forward_from_feature(feature, set_summary, e, mode='env')
        
        loss_set = self.loss_fct(y_pred_set, y.float())
        loss_pure = self.loss_fct(y_pred_pure, y.float())
        loss_env_info = self.loss_fct(y_pred_env, y.float())
        
        loss_env_pred_set = self.loss_classifier(env_pred_set, e.float())
        loss_env_pred_conv = self.loss_classifier(env_pred_conv, e.float())
    
        loss = loss_pure + loss_env_info + loss_set + loss_env_pred_set + loss_env_pred_conv

        with torch.no_grad():
            acc_e_pred_set = (env_pred_set.argmax(1) == e.argmax(1)).sum()/e.shape[0]
            acc_e_pred_conv = (env_pred_conv.argmax(1) == e.argmax(1)).sum()/e.shape[0]

        out_dict = dict(
            loss=loss,
            loss_set=loss_set,
            loss_pure=loss_pure,
            loss_env_pred_set=loss_env_pred_set,
            loss_env_pred_conv=loss_env_pred_conv,
            loss_env_info=loss_env_info,
            loss_set_benefit=loss_set-loss_pure,
            loss_env_info_benefit=loss_env_info-loss_pure,
            acc_e_pred_set=acc_e_pred_set,
            acc_e_pred_conv=acc_e_pred_conv,
        )
    
        if self.hparams.task == 'classification':
            # Accuracies for different predictions
            with torch.no_grad():
                acc_y_set = (y_pred_set.argmax(1) == y.argmax(1)).sum()/y.shape[0]
                acc_y_pure = (y_pred_pure.argmax(1) == y.argmax(1)).sum()/y.shape[0]
                acc_y_env_info = (y_pred_env.argmax(1) == y.argmax(1)).sum()/y.shape[0]


                # Empirical Calibration Error (ECE)
                ece = ECELoss(n_bins=10)
                ece_set = ece(y_pred_set, y.argmax(1))[0].item()
                ece_pure = ece(y_pred_pure, y.argmax(1))[0].item()
                ece_env = ece(y_pred_env, y.argmax(1))[0].item()

            out_dict.update(dict(acc_y_set=acc_y_set,
                acc_y_pure=acc_y_pure,
                acc_y_env_info=acc_y_env_info,
                acc_y_set_benefit=acc_y_set-acc_y_pure,
                acc_y_env_info_benefit=acc_y_env_info-acc_y_pure,
                #'''
                ece_set=ece_set,
                ece_pure=ece_pure,
                ece_env=ece_env,
                ece_set_benefit=ece_set-ece_pure,
                ece_env_benefit=ece_env-ece_pure,
                #'''
                ))

        return out_dict

    def compute_ood_metrics(self, batch, batch_idx):
        return self.compute_metrics(batch, batch_idx)
