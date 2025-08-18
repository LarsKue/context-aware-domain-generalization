import torch
from torch import nn

from lightning_trainable import  TrainableHParams
from lightning_trainable.hparams import Range

import adaptive_dg.models.utils_networks as utils_networks

from adaptive_dg.models.domain_generalizer import DomainGeneralizer, DGHParams

from lightning_trainable import utils
from lightning_trainable.modules import FullyConnectedNetwork

class BaseClassClassifiersHParams(DGHParams):

    featurizer_hparams: dict
    encoder_hparams: dict 
    classifier_hparams: dict

    feature_train: bool = False 

    gradient_clip: float = 2. 
    gradient_clip_algorithm: str = "value"
    strategy: str = "repeat"
    set_size: int = 10 

    task: str = "classification"
    loss_fct: str = "cross_entropy"
    
    @classmethod
    def validate_parameters(cls, hparams):
        hparams = super().validate_parameters(hparams)

        assert all(x in hparams['encoder_hparams'].keys() for x in ['output_dim', 'name']), \
            "encoder_hparams must contain output_dim and name"
        assert "output_dim" in hparams['featurizer_hparams'].keys(), \
            "output_dim must be in featurizer_hparams"
        assert all(x in hparams['classifier_hparams'].keys() for x in ['activation']), \
            "activation must be in classifier_hparams"

        return hparams
    

class BaseClassClassifiers(DomainGeneralizer):
    """Implements the Base Class of a classiifer that can use set-encoded input, normal input or both 
    
    This class enables to load the set-encoder, feature-extractor and classifier
    """
    def __init__(self, hparams):
        super().__init__(hparams)
        
    def configure_encoder(self):
        if self.hparams.strategy == 'double_repeat':
            self.hparams.encoder_hparams['input_dim'] = 2 * self.hparams.featurizer_hparams['output_dim']
        else:
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

    def configure_simple_classifier(self, input_dim, output_dim, layer_widths=[512, 512]):
        network = nn.Sequential(nn.Linear(input_dim, layer_widths[0]), nn.ReLU(), nn.Linear(layer_widths[1], output_dim))
        return network
    
    def compute_encoder_output(self, feature, e):
        """Implements the strategy how a usual batch is formed to a batch for the set encoder
        """
        
        condition = torch.zeros((feature.shape[0], self.hparams.encoder_hparams['output_dim'])).to(feature.device)

        if e[0,:].sum() == 1:
            for je in range(e.shape[1]):
                mask_e = (e[:, je] == 1).view(-1).to(feature.device)
                if mask_e.sum() > 0:

                    if self.hparams.strategy == 'repeat':
                        set_input = feature[mask_e][None, :, :]
                        condition[mask_e] = self.encoder(set_input).repeat(mask_e.sum(), 1)
                        
                    elif self.hparams.strategy == 'double_repeat':
                        # Concatenate each input element with set_input
                        set_input_1 = feature[mask_e][None, :, :].repeat(mask_e.sum(), 1, 1)
                        set_input_2 = feature[mask_e][:, None, :].repeat(1, mask_e.sum(), 1)
                        set_input = torch.cat([set_input_1, set_input_2], dim=2)
                        condition[mask_e] = self.encoder(set_input)#.repeat(mask_e.sum(), 1)
                    
                    # Currently not a useful strategy
                    elif self.hparams.strategy == 'repeat2':
                        for i in range(mask_e.sum()):
                            if self.training:
                                mask_all  = self.train_data[:][2][:,je] == 1 
                                data = self.train_data[:][0][mask_all]#[None,:,:]
                                idx = torch.randperm(mask_all.sum())[:500]

                                data = self.feature_extractor(data[idx].to(e.device))[None,:,:]
                                condition[mask_e][i:i+1] = self.encoder(data)
                            else:
                                mask_all  = self.val_data[:][2][:, je] == 1 
                                data = self.val_data[:][0][mask_all]#[None,:,:]
                                idx = torch.randperm(mask_all.sum())[:500]
                                data = self.feature_extractor(data[idx].to(e.device))
                                data = data[None,:,:]
                                condition[mask_e][i:i+1] = self.encoder(data)
                        """
                        set_input = feature[mask_e][None, :, :]
                        set_length = int(0.8* mask_e.sum())
                        idx  = torch.randperm(mask_e.sum())[:set_length]
                        for i in range(mask_e.sum()):
                            condition[mask_e][i:i+1] = self.encoder(set_input[:,idx,:])
                        """                        
        else:
            if self.hparams.strategy == 'repeat':
                set_input = feature[None, :, :]
                condition = self.encoder(set_input).repeat(feature.shape[0], 1)
            elif self.hparams.strategy == 'double_repeat':
                set_input_1 = feature[None, :, :].repeat(feature.shape[0], 1, 1)
                set_input_2 = feature[:, None, :].repeat(1, feature.shape[0], 1)
                set_input = torch.cat([set_input_1, set_input_2], dim=2)
                condition = self.encoder(set_input)
            else:
                return NotImplementedError

        return condition 
