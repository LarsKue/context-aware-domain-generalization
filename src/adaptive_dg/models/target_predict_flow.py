
from torch import nn
import torch
from lightning_trainable import utils

from adaptive_dg.models.base_classes import BaseClassClassifiers
from models.utils_networks import MNIST_Autoencoder
from models.flows import configure_mnist_inn


class PredictYfromXSetFlow(BaseClassClassifiers):
    """Classifier that predicts class Y from input X and Set-Input using Flow

    Args:
        BaseClassClassifiers (_type_): _description_
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        n_layers = 12 #should be self.hparams.n_layers_flow but would require to make new hparams class with specific validate_parameters function
        self.feature_extractor = self.configure_feature_extractor()
        self.encoder = self.configure_encoder()
        self.class_classifier = self.configure_flow(self.hparams.featurizer_hparams['output_dim'], self.hparams.encoder_hparams['output_dim'], self.hparams.n_classes, n_layers) 
    
    def configure_feature_extractor(self):
        featurizer = MNIST_Autoencoder(self.hparams.featurizer_hparams["input_dim"], self.hparams.featurizer_hparams["output_dim"])

        try:
            featurizer.load_state_dict(torch.load("./adaptive_dg/models_utils/mnist_AE/state_dict_AE.pth"))
        except:
            raise ValueError("No such file or directory, try executing file from root folder")

        for p in featurizer.parameters():
                p.requires_grad = False
        
        featurizer.eval() #not sure if even required after disabling grad
        return featurizer
    
    def configure_flow(self, input_dim, set_dim, n_classes, n_layers):
        return configure_mnist_inn(input_dim, set_dim+n_classes, n_layers)
    
    def predict_y(self,x,cond):
        nc = self.hparams.n_classes

        y = nn.functional.one_hot(torch.arange(nc),nc)
        y = y.unsqueeze(0).repeat(x.shape[0],1,1).view(-1,nc)

        x = x.unsqueeze(1).repeat(1,nc,1).view(-1,x.shape[1])
        cond = cond.unsqueeze(1).repeat(1,nc,1).view(-1,cond.shape[1])
        
        out, jac = self.class_classifier(x,c=[torch.cat((y,cond),1)])

        pred = 0.5*torch.sum(out**2, 1) - jac
        pred = torch.argmin(pred.view(-1,nc),1)
        
        return pred

    def compute_metrics(self, batch, batch_idx):
        x, y, e = batch

        x, e, y = x.float(), e.long(), y.long()

        feature_set = self.feature_extractor.encode(x.flatten(start_dim=1))
        set_summary = self.compute_encoder_output(feature_set, e)

        feature_noise = feature_set + 2e-2*torch.randn(feature_set.shape, device=feature_set.device) 

        out, jac = self.class_classifier(feature_noise,c=[torch.cat((y,set_summary),1)])

        loss = torch.mean(0.5*torch.sum(out**2, 1) - jac) / self.hparams.featurizer_hparams['output_dim']

        y_pred = self.predict_y(feature_set,set_summary)
    
        # Accuracies for different predictions
        acc_y = (y_pred == y.argmax(1)).sum()/y.shape[0]

        return dict(
            loss=loss,
            H_y_xset=loss,
            acc_y=acc_y,
        )

    def compute_ood_metrics(self, batch, batch_idx):
        return self.compute_metrics(batch, batch_idx)

    def configure_optimizers(self):
        """
        Configure optimizers for Lightning
        """
        kwargs = dict()

        parameters = [
            {
                "params": list(self.encoder.parameters()),
                "lr": self.hparams.optimizer["lr"],
                "weight_decay": self.hparams.optimizer["weight_decay"],
            },
            {
                "params": list(self.class_classifier.parameters()),
                "lr": self.hparams.optimizer["lr"],
                "weight_decay": self.hparams.optimizer["weight_decay"],
            },
        ]


        match self.hparams.optimizer:
            case str() as name:
                optimizer = utils.get_optimizer(name)(parameters)
            case dict() as kwargs:
                name = kwargs.pop("name")
                optimizer = utils.get_optimizer(name)(parameters)
            case type(torch.optim.Optimizer) as Optimizer:
                optimizer = Optimizer(parameters)
            case torch.optim.Optimizer() as optimizer:
                pass
            case None:
                return None
            case other:
                raise NotImplementedError(f"Unrecognized Optimizer: {other}")

        lr_scheduler = self.configure_lr_schedulers(optimizer)

        if lr_scheduler is None:
            return optimizer

        return dict(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )