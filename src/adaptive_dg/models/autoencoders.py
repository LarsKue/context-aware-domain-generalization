from abc import ABC, abstractmethod

import os
import json
import math

import PIL

import numpy as np
import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, TensorDataset

from adaptive_dg.models_utils.autoencoder import (
    texture_layers,
    Encoder,
    Decoder,
    FastGanDecoder,
)
from adaptive_dg.models_utils.autoencoder_net2net import ResnetEncoder
from adaptive_dg.models_utils.network_fastgan import Generator

# from utils import *

from adaptive_dg.models_utils.losses import LPIPS  # LPIPS loss

from adaptive_dg.plots.sampling import plot_reconstruction

from lightning_trainable import TrainableHParams, Trainable


class AutoEncoderStandardHParams(TrainableHParams):
    latent_dim_total: int = 32
    img_resolution: int = 32
    optimizer: dict = ({"name": "Adam", "lr": 1e-4},)


class AutoEncoderStandard(Trainable):
    hparams: AutoEncoderStandardHParams

    def __init__(self, hparams):
        super().__init__(hparams)

        if self.hparams.img_resolution == 32 or True:
            self.encoder = Encoder_Diva(latent_dim=self.hparams.latent_dim_total)
            self.decoder = Decoder_Diva(latent_dim=self.hparams.latent_dim_total)

        if False:
            # encoder_t = ResnetEncoder(self.latent_dim//2, max(img_resolution, 64), in_channels=channels)
            encoder_t = ResnetEncoder(
                self.hparams.latent_dim_total // 2,
                max(self.hparams.img_resolution, 64),
                in_channels=channels,
                type="resnet18",
            )
            encoder_t.use_preprocess = False
            self.encoder = ResNetWrapper(encoder_t)
            self.decoder = FastGanDecoder(
                z_dim=self.hparams.latent_dim_total,
                img_resolution=img_resolution_out,
                nc=channels,
                lite=lite,
            )

        # Customizable
        self.loss_fct = nn.MSELoss()

    def forward_decoder(self, z):
        """
        Computes Decoding from latent space input z
        """
        return self.decoder(z)

    def forward_encoder(self, x):
        """
        Computes Encoding from input x
        """
        return self.encoder(x)

    def forward_autoencoder(self, x):
        """
        Computes reconstruction: Decoder(Encoder(x))
        """
        return self.forward_decoder(self.forward_encoder(x))

    def reconstruction_loss_fct(self, x, x_rec):
        """
        Reconstruction Loss
        """
        return self.loss_fct(x, x_rec)

    def compute_metrics(self, batch, batch_idx):
        """
        Computes metrics, here only reconstruction loss
        """

        x, y, d = batch

        z = self.encoder(x)
        x_rec = self.decoder(z)

        loss = self.reconstruction_loss_fct(x, x_rec) #+ 1e-3 * (z**2).mean()

        return dict(loss=loss)

    def ood_dataloader(self):
        """
        Configure and return the ood dataloader. Has to be invorked activelyt
        """
        if self.ood_data is None:
            return None
        return DataLoader(
            dataset=self.ood_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
            num_workers=self.hparams.num_workers,
        )

    def on_validation_epoch_end(self) -> None:
        """
        Plots Vaidation Epoch:
            - Plot: reconstrucion on id and ood data
        """
        # Reconstruction
        fig = plot_reconstruction(self, self.val_data)
        self.logger.experiment.add_figure(
            "Reconstruction_id",
            fig,
            global_step=int(
                self.trainer.global_step / self.trainer.num_training_batches
            ),
        )

        fig = plot_reconstruction(self, self.ood_data)
        self.logger.experiment.add_figure(
            "Reconstruction_ood",
            fig,
            global_step=int(
                self.trainer.global_step / self.trainer.num_training_batches
            ),
        )


class ResNetWrapper(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        n = x.shape[0]
        return self.network(x).view(n, -1)


class Encoder_Diva(nn.Module):
    """
    From Diva Repo. URL missing
    """

    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc11 = nn.Sequential(nn.Linear(1024, latent_dim))
        # self.fc12 = nn.Sequential(nn.Linear(1024, zd_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.encoder[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder[4].weight)
        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        # $torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        # self.fc12[0].bias.data.zero_()

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(-1, 1024)
        zd_loc = self.fc11(h)
        # zd_scale = self.fc12(h) + 1e-7

        return zd_loc  # , zd_scale


class Decoder_Diva(nn.Module):
    """
    From Diva Repo. URL missing
    """

    def __init__(self, latent_dim=16):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim, 1024, bias=False), nn.BatchNorm1d(1024), nn.ReLU()
        )
        self.up1 = nn.Upsample(8)
        self.de1 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.up2 = nn.Upsample(24)
        self.de2 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 256, kernel_size=5, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.de3 = nn.Sequential(nn.Conv2d(256, 3, kernel_size=1, stride=1))

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.de1[0].weight)
        torch.nn.init.xavier_uniform_(self.de2[0].weight)
        torch.nn.init.xavier_uniform_(self.de3[0].weight)
        self.de3[0].bias.data.zero_()

    def forward(self, z):
        h = self.fc1(z)
        h = h.view(-1, 64, 4, 4)
        h = self.up1(h)
        h = self.de1(h)
        h = self.up2(h)
        h = self.de2(h)
        loc_img = self.de3(h)
        return loc_img
