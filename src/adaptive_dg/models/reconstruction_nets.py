from torch import nn
import torch
from lightning_trainable import utils

from adaptive_dg.models.base_classes import BaseClassClassifiers
from .domain_generalizer import DGHParams

from adaptive_dg.models.autoencoders_models.helper_functions import Decoder, Encoder
from models.set_encoder import SetEncoderConv

class AutoencoderHParams(DGHParams):
    enc_hparams: dict
    devices: list
    
    @classmethod
    def validate_parameters(cls, hparams):
        hparams = super().validate_parameters(hparams)

        assert all(x in hparams['enc_hparams'].keys() for x in ['latent_dim', 'input_channel']), \
            "encoder_hparams must contain output_dim and name"

        return hparams

class VarAutoencoderHParams(AutoencoderHParams):
    beta: float = 1e-3

class AutoencoderSetHParams(DGHParams):
    enc_hparams: dict
    encoder_hparams: dict
    devices: list
    strategy: str = "repeat"
    
    @classmethod
    def validate_parameters(cls, hparams):
        hparams = super().validate_parameters(hparams)

        assert all(x in hparams['enc_hparams'].keys() for x in ['latent_dim', 'input_channel']), \
            "encoder_hparams must contain output_dim and name"
        
        assert all(x in hparams['encoder_hparams'].keys() for x in ['latent_dim', 'output_dim']), \
            "encoder_hparams must contain output_dim and name"

        return hparams

class VarAutoencoderSetHParams(AutoencoderSetHParams):
    beta: float = 1e-3

class ResNetWrapper(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        n = x.shape[0]
        return self.network(x).view(n, -1)

class AutoEncoder(BaseClassClassifiers):
    hparams: AutoencoderHParams

    def __init__(self, hparams):
        super().__init__(hparams)

        self.rec_loss = nn.MSELoss()

        self.configure_autoencoder(hparams["enc_hparams"]["input_channel"],hparams["enc_hparams"]["latent_dim"])

        
    def configure_autoencoder(self,ch_inp,latent_dim=1024):
        self.encoder = Encoder(ch_inp,enc_size=latent_dim)
        self.decoder = Decoder(ch_inp,enc_size=latent_dim)
        
    def forward(self, x):
        return self.decoder(self.encoder(x))
        
    def compute_metrics(self, batch, batch_idx):
        x, _, y, e = batch       
        x, e, y = x.float(), e.long(), y.long()

        x_rec = self(x)
        loss = self.rec_loss(x_rec, x)
    
        return dict(
            loss=loss,
            l2_loss=loss,
        )

    def compute_ood_metrics(self, batch, batch_idx):
        return self.compute_metrics(batch, batch_idx)
    
class AutoEncoderEnv(BaseClassClassifiers):
    hparams: AutoencoderHParams

    def __init__(self, hparams):
        super().__init__(hparams)

        self.rec_loss = nn.MSELoss()

        self.configure_autoencoder(hparams["enc_hparams"]["input_channel"],hparams["enc_hparams"]["latent_dim"],len(hparams["id_domains"]))

        
    def configure_autoencoder(self,ch_inp, latent_dim=1024, n_envs=2):
        self.encoder = Encoder(ch_inp, enc_size=latent_dim)
        self.decoder = Decoder(ch_inp, enc_size=latent_dim+n_envs)
        
    def forward(self, x, e):
        z = self.encoder(x)
        z = torch.cat((z, e), 1)
        return self.decoder(z)
        
    def compute_metrics(self, batch, batch_idx):
        x, _, y, e = batch       
        x, e, y = x.float(), e.long(), y.long()

        x_rec = self(x, e)
        loss = self.rec_loss(x_rec, x)
    
        return dict(
            loss=loss,
            l2_loss=loss,
        )

    def compute_ood_metrics(self, batch, batch_idx):
        return self.compute_metrics(batch, batch_idx)

class AutoEncoderSet(BaseClassClassifiers):
    """Autoencoder that uses additional Set info for reconstruction
    """
    hparams: AutoencoderSetHParams

    def __init__(self, hparams):
        super().__init__(hparams)

        self.rec_loss = nn.MSELoss()

        self.configure_autoencoder(self.hparams.enc_hparams["input_channel"],self.hparams.enc_hparams["latent_dim"],self.hparams.encoder_hparams["output_dim"])

        self.set_encoder = self.configure_encoder(self.hparams)

    def forward(self, x, x_set):
        enc = self.encoder(x)
        set_summary = self.compute_encoder_output(x_set)

        enc_set = torch.cat((enc, set_summary), 1)
        
        x_rec = self.decoder(enc_set)
        
        return x_rec

    def compute_encoder_output(self, x_set):
        condition = torch.zeros((x_set.shape[0], self.hparams.encoder_hparams['output_dim']),device=x_set.device)#.to(x_set.device)
        for i in range(x_set.shape[0]):
            condition[i] = self.set_encoder(x_set[i])
        return condition

    def configure_autoencoder(self,ch_inp, latent_dim=1024, n_envs=2):
        self.encoder = Encoder(ch_inp, enc_size=latent_dim)
        self.decoder = Decoder(ch_inp, enc_size=latent_dim+n_envs)
    
    def configure_encoder(self,hparams):
        return SetEncoderConv(hparams.enc_hparams["input_channel"],hparams.encoder_hparams["latent_dim"],hparams.encoder_hparams["output_dim"],hparams.encoder_hparams["layer_width"],pma_args=hparams.encoder_hparams["pma_args"],maxpool=hparams.encoder_hparams["maxpool"])
     
    def compute_metrics(self, batch, batch_idx):
        x, x_set, y, e = batch       
        x, x_set, e, y = x.float(), x_set.float(), e.long(), y.long()

        x_rec = self(x, x_set)
    
        loss = self.rec_loss(x_rec, x)

        return dict(
            loss=loss,
            l2_loss=loss,
        )


    def compute_ood_metrics(self, batch, batch_idx):
        return self.compute_metrics(batch, batch_idx)

class VarAutoEncoderEnv(BaseClassClassifiers):
    hparams: VarAutoencoderHParams

    def __init__(self, hparams):
        super().__init__(hparams)

        self.rec_loss = nn.MSELoss()

        self.configure_autoencoder(hparams["enc_hparams"]["input_channel"],hparams["enc_hparams"]["latent_dim"],len(hparams["id_domains"]))

        
    def configure_autoencoder(self,ch_inp, latent_dim=1024, n_envs=2):
        self.encoder = Encoder(ch_inp, enc_size=2*latent_dim)
        self.decoder = Decoder(ch_inp, enc_size=latent_dim+n_envs)
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        sample = torch.randn_like(logvar)*std + mu
        return sample
        
    def forward(self, x, e):
        mu, logvar = torch.chunk(self.encoder(x),2,1)

        if self.training:
            z = self.reparametrize(mu, logvar)
        else:
            z = mu
        
        z = torch.cat((z, e), 1)

        x_rec = self.decoder(z)

        return x_rec, mu, logvar
        
    def compute_metrics(self, batch, batch_idx):
        x, _, y, e = batch       
        x, e, y = x.float(), e.long(), y.long()

        x_rec, mu, logvar = self(x, e)
        l2_loss = self.rec_loss(x_rec, x)
    
        KLD = 0.5 * torch.sum(torch.pow(mu,2) + torch.exp(logvar) - 1.0 - logvar, dim=1)
        KLD = torch.mean(KLD)

        loss = l2_loss + self.hparams.beta*KLD

        return dict(
            loss=loss,
            l2_loss=l2_loss,
            kld_loss=KLD
        )

    def compute_ood_metrics(self, batch, batch_idx):
        return self.compute_metrics(batch, batch_idx)

class VarAutoEncoderSet(BaseClassClassifiers):
    """Autoencoder that uses additional Set info for reconstruction
    """
    hparams: VarAutoencoderSetHParams

    def __init__(self, hparams):
        super().__init__(hparams)

        self.rec_loss = nn.MSELoss()

        self.configure_autoencoder(self.hparams.enc_hparams["input_channel"],self.hparams.enc_hparams["latent_dim"],self.hparams.encoder_hparams["output_dim"])

        self.set_encoder = self.configure_encoder(self.hparams)
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        sample = torch.randn_like(logvar)*std + mu
        return sample

    def forward(self, x, x_set):
        mu, logvar = torch.chunk(self.encoder(x),2,1)

        if self.training:
            z = self.reparametrize(mu, logvar)
        else:
            z = mu
        
        set_summary = self.compute_encoder_output(x_set)

        enc_set = torch.cat((z, set_summary), 1)
        
        x_rec = self.decoder(enc_set)
        
        return x_rec, mu, logvar

    def compute_encoder_output(self, x_set):
        condition = torch.zeros((x_set.shape[0], self.hparams.encoder_hparams['output_dim']),device=x_set.device)#.to(x_set.device)
        for i in range(x_set.shape[0]):
            condition[i] = self.set_encoder(x_set[i])
        return condition

    def configure_autoencoder(self,ch_inp, latent_dim=1024, n_envs=2):
        self.encoder = Encoder(ch_inp, enc_size=2*latent_dim)
        self.decoder = Decoder(ch_inp, enc_size=latent_dim+n_envs)
    
    def configure_encoder(self,hparams):
        return SetEncoderConv(hparams.enc_hparams["input_channel"],hparams.encoder_hparams["latent_dim"],hparams.encoder_hparams["output_dim"],hparams.encoder_hparams["layer_width"],pma_args=hparams.encoder_hparams["pma_args"],maxpool=hparams.encoder_hparams["maxpool"])
     
    def compute_metrics(self, batch, batch_idx):
        x, x_set, y, e = batch       
        x, x_set, e, y = x.float(), x_set.float(), e.long(), y.long()

        x_rec, mu, logvar = self(x, x_set)

        l2_loss = self.rec_loss(x_rec, x)
    
        KLD = 0.5 * torch.sum(torch.pow(mu,2) + torch.exp(logvar) - 1.0 - logvar, dim=1)
        KLD = torch.mean(KLD)

        loss = l2_loss + self.hparams.beta*KLD

        return dict(
            loss=loss,
            l2_loss=l2_loss,
            kld_loss=KLD
        )


    def compute_ood_metrics(self, batch, batch_idx):
        return self.compute_metrics(batch, batch_idx)