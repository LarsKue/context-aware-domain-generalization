from torch import nn
import torch

from adaptive_dg.models.base_classes import BaseClassClassifiers
from adaptive_dg.models.target_predict import EnvClassifier

from adaptive_dg.models.utils import ECELoss

from sklearn.decomposition import PCA
import adaptive_dg.models.utils_networks as utils_networks

from adaptive_dg.models.domain_generalizer import DomainGeneralizer, DGHParams
from lightning_trainable.modules import FullyConnectedNetwork


class BaseModelHParams(DGHParams):

    featurizer_hparams: dict
    encoder_hparams: dict = dict()
    classifier_hparams: dict

    feature_train: bool = True

    input_shape: tuple = (3, 224, 224)
    n_classes: int = 1

    gradient_clip: float = 2. 
    gradient_clip_algorithm: str = "value"
    set_size: int = 10 

    mmd_lambda: float = 0.0

    task: str = "classification"
    loss_fct: str = "cross_entropy"
    
    @classmethod
    def validate_parameters(cls, hparams):
        hparams = super().validate_parameters(hparams)

        #'''
        #assert all(x in hparams['encoder_hparams'].keys() for x in ['output_dim', 'name']), \
        #    "encoder_hparams must contain output_dim and name"
        #assert "output_dim" in hparams['featurizer_hparams'].keys(), \
        #    "output_dim must be in featurizer_hparams"
        #assert all(x in hparams['classifier_hparams'].keys() for x in ['activation']), \
        #    "activation must be in classifier_hparams"

        return hparams

class BaseModel(DomainGeneralizer):
    """Classifiert that predicts class Y from input X and Set-Input

    Args:
        BaseClassClassifiers (_type_): _description_
    """
    hparams: BaseModelHParams

    def __init__(self, hparams):
        super().__init__(hparams)

        self.loss_classifier = nn.CrossEntropyLoss()

        match self.hparams.loss_fct:
            case 'mse':
                self.loss_fct = nn.MSELoss()
            case 'cross_entropy':
                self.loss_fct = nn.CrossEntropyLoss()
            case other:
                raise NotImplementedError(f"Unrecognized loss choice: '{other}'")
            
    def configure_encoder(self):
        self.hparams.encoder_hparams['input_dim'] = self.hparams.featurizer_hparams['output_dim']
        encoder = utils_networks.EncoderConstructor(self.hparams.encoder_hparams)
        return encoder

    def configure_feature_extractor(self):
        network = utils_networks.Featurizer(self.image_shape, self.hparams.featurizer_hparams)
        if not self.hparams.feature_train:
            for p in network.parameters():
                p.requires_grad = False

        return network

    def configure_classifier(self, input_dim, output_dim):
        hparams =dict(
            input_dims=input_dim,
            output_dims=output_dim,
            layer_widths=self.hparams.classifier_hparams['layer_sizes'],
            activation="relu",
        )
        classifier = FullyConnectedNetwork(hparams)
        return classifier

    def configure_simple_classifier(self, input_dim, output_dim):
        network = nn.Sequential(nn.Linear(input_dim, output_dim)) 
        return network
    