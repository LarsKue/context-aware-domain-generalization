
from torch import nn
import torch
from lightning_trainable import utils

from adaptive_dg.models.base_classes import BaseClassClassifiers

class PredictEfromSet(BaseClassClassifiers):
    """Classifier that predicts domain E from Set
    """

    def __init__(self, hparams):
        super().__init__(hparams)

        self.feature_extractor_set = self.configure_feature_extractor()
        self.encoder = self.configure_encoder()
        self.env_classifier_set_input = self.configure_classifier(self.hparams.encoder_hparams['output_dim'], len(self.hparams.id_domains))

        self.loss_classifier = nn.CrossEntropyLoss()
        
    def compute_metrics(self, batch, batch_idx):
        x, x_set, y, e = batch       
        x, x_set, e, y = x.float(), x_set.float(), e.long(), y.long()

        x_set = x_set.view(x_set.shape[0] * x_set.shape[1], *x_set.shape[2:])
        feature_set = self.feature_extractor_set(x_set)
        #feature_set = self.feature_extractor_set(x)
        n_features = feature_set.shape[1]
        feature_set = feature_set.view(x.shape[0], -1, n_features)
        set_summary = self.encoder(feature_set)

        #feature_set = self.feature_extractor_set(x)
        #set_summary = self.compute_encoder_output(feature_set, e)

        e_pred = self.env_classifier_set_input(set_summary)

        loss = self.loss_classifier(e_pred, e.float())
    
        # Accuracies for different predictions
        acc_e = (e_pred.argmax(1) == e.argmax(1)).sum()/ e.shape[0]

        return dict(
            loss=loss,
            H_e_set=loss,
            acc_e=acc_e,
        )

    def compute_ood_metrics(self, batch, batch_idx):
        return dict(
            loss=0
        )

    def on_validation_epoch_end(self) ->None:
        return 

        self.eval()
        # Model-Misspecification
        set_summaries_id = []
        for x_id, y_id, e_id in self.val_dataloader()[0]:
            x_id, e_id, y_id = x_id.float(), e_id.long(), y_id.long()

            with torch.no_grad():
                feature_id = self.feature_extractor(x_id.to(self.device))
                for i in range(e_id.shape[1]):
                    mask = e_id[:,i] == 1
                    if mask.sum() >0:
                        set_summaries_id.append(self.compute_encoder_output(feature_id[mask][:1], e_id[mask][:1]))
        set_summaries_id = torch.cat(set_summaries_id)

        reference_mmds = []
        for i in range(set_summaries_id.shape[0]):
            reference_mmds.append(_mmd(set_summaries_id[i:i+1], torch.cat((set_summaries_id[:i], set_summaries_id[i+1:]),0) ).item())
        reference_mmd = np.quantile(reference_mmds, 0.95)

        true_false = []
        for x, y, e in self.val_dataloader()[1]:
            x, e, y= x.float(), e.long(), y.long()
            with torch.no_grad():
                feature_ood = self.feature_extractor(x.to(self.device))
                set_embedding = self.compute_encoder_output(feature_ood, e)

                out_mmd = _mmd(set_embedding, set_summaries_id).item()
            if reference_mmd > out_mmd:
                true_false.append(torch.ones_like(y.to(self.device))[:,0])
            else:
                true_false.append(torch.zeros_like(y.to(self.device))[:,0])
        
        true_false = torch.cat(true_false)
      
        acc_miss = true_false.sum() / true_false.shape[0]
        self.log(f"ood/miss_spec_acc", acc_miss)

        self.train()

    def compute_ood_metrics(self, batch, batch_idx):
        return dict(
            loss=0
        )

class PredictEfromX(BaseClassClassifiers):
    """Classifier that predicts domain E from input X

    Args:
        BaseClassClassifiers (_type_): _description_
    """
    def __init__(self, hparams):
        super().__init__(hparams)

        self.feature_extractor_x = self.configure_feature_extractor()
        self.env_classifier = self.configure_classifier(self.hparams.featurizer_hparams['output_dim'] , len(self.hparams.id_domains))

        self.loss_classifier = nn.CrossEntropyLoss()

    def compute_metrics(self, batch, batch_idx):
        x, _, y, e = batch       

        x, e, y = x.float(), e.long(), y.long()

        feature_x = self.feature_extractor_x(x)
        e_pred = self.env_classifier(feature_x)

        loss = self.loss_classifier(e_pred, e.float())
    
        # Accuracies for different predictions
        acc_e = (e_pred.argmax(1) == e.argmax(1)).sum()/ e.shape[0]

        return dict(
            loss=loss,
            H_e_x=loss,
            acc_e=acc_e,
        )

    def compute_ood_metrics(self, batch, batch_idx):
        return dict(
            loss=0
        )
