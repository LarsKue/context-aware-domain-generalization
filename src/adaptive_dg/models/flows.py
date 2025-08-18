
import importlib

import os
import json
import math

import PIL

import numpy as np
import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F


import FrEIA.framework as Ff
import FrEIA.modules as Fm

from adaptive_dg.plots.sampling import (
    plot_reconstruction,
    plot_data_sampling,
    plot_latent_space_exchange,
    plot_latent_space_exchange_generic,
)

from lightning_trainable import Trainable, utils, TrainableHParams



def subnet_fc(dims_in, dims_out):
    #n = 2 * 1024
    n = 1 * 1024
    p = 0.25
    return nn.Sequential(
        nn.Linear(dims_in, n),
        nn.SELU(),
        nn.Dropout(p=p),
        nn.Linear(n, n),
        nn.SELU(),
        nn.Dropout(p=p),
        nn.Linear(n, dims_out),
    )

class GaussianMixtureINNHParams(TrainableHParams):
    # Model specific hyperparameters
    latent_dims: list= [32, 16, 16] 
    latent_dim_total: int = 64
    n_layers_flow: int = 8
    optimizer: dict = {"name": "Adam", "lr": 1e-4}
    autoencoder_cfg: dict
    noise_scaling: float = 1e-2
    reconstruction_weight: float = 100.0
    joint_training: bool = True

    # Dataset specific hyperparameters
    n_envs: int = 5
    n_classes: int = 10



class GaussianMixtureINN(Trainable):
    hparams: GaussianMixtureINNHParams

    def __init__(self, hparams):
        super().__init__(hparams)

        self.latent_dims = self.hparams["latent_dims"]

        self.flow = Ff.SequenceINN(self.hparams.latent_dim_total)
        for k in range(self.hparams.n_layers_flow):
            self.flow.append(
                Fm.AllInOneBlock,
                subnet_constructor=subnet_fc,
                permute_soft=True,
                gin_block=False,
            )

        # Loading Autoencoder
        module_models = importlib.import_module(
            "adaptive_dg.models.autoencoders"
        )  # may raise ImportError
        module = getattr(module_models, self.hparams.autoencoder_cfg["model_class"])
        try:
            self.autoencoder = module.load_from_checkpoint(
                self.hparams.autoencoder_cfg["path_checkpoint"]
            )
        except:
            raise Exception("No AutoEncoder specified")
        assert (
            self.autoencoder.hparams.latent_dim_total == self.hparams.latent_dim_total
            )


        n = 128
        self.mu_env = nn.Sequential(
            nn.Linear(self.hparams.n_envs, n),
            nn.PReLU(),
            nn.Linear(n, n),
            nn.PReLU(),
            nn.Linear(n, self.latent_dims[0]),
        )
        self.mu_class = nn.Sequential(
            nn.Linear(self.hparams.n_classes, n),
            nn.PReLU(),
            nn.Linear(n, n),
            nn.PReLU(),
            nn.Linear(n, self.latent_dims[1]),
        )

        self.joint_training = False  # joint_training
        if not self.joint_training:
            for p in self.autoencoder.parameters():
                p.requires_grad = False

    def on_validation_epoch_end(self) -> None:
        # plot code

        fig = plot_data_sampling(self)
        self.logger.experiment.add_figure(
            "Sampling",
            fig,
            global_step=int(
                self.trainer.global_step / self.trainer.num_training_batches
            ),
        )
        fig = plot_latent_space_exchange(self, self.train_data)
        self.logger.experiment.add_figure(
            "Latent Space Ex.",
            fig,
            global_step=int(
                self.trainer.global_step / self.trainer.num_training_batches
            ),
        )

        out = self.predict(self.ood_data[:][0].float().to(self.device))
        y = self.ood_data[:][1].float().to(self.device)
        acc_class_ood = (out.argmax(1) == y.argmax(1)).sum() / y.shape[0]

        self.log(f"ood/acc_class", acc_class_ood)

    def ood_dataloader(self):
        """
        Configure and return the ood dataloader. Has to be invorked actively
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

    def compute_metrics(self, batch, batch_idx):
        x, y, e = batch

        if self.joint_training:
            with torch.no_grad():
                emb = self.forward_encoder(x)
        else:
            emb = self.forward_encoder(x)

        emb += self.hparams.noise_scaling * torch.randn_like(emb)

        mu_env = self.mu_env(e)
        mu_class = self.mu_class(y)
        mu_residual = torch.zeros(y.shape[0], self.hparams.latent_dims[2]).to(
            x.device
        )
        
        mu_all = torch.cat((mu_env, mu_class, mu_residual), 1)

        loss = 0
        z_inn, log_jac_det = self.flow(emb)
        loss_flow = (
            0.5 * torch.sum((z_inn - mu_all) ** 2, 1) - log_jac_det
        ).mean() / self.hparams.latent_dim_total
        loss += loss_flow

        if self.joint_training:
            x_rec = self.forward_decoder(emb)  # / 1e10
            loss_rec = self.autoencoder.reconstruction_loss_fct(x, x_rec)
            loss += self.hparams.reconstruction_weight * loss_rec
        else:
            loss_rec = 0

        with torch.no_grad():
            acc_class = (
                self.predict(x, location="class").argmax(1) == y.argmax(1)
            ).sum() / y.shape[0]
            acc_env = (
                self.predict(x, location="env").argmax(1) == e.argmax(1)
            ).sum() / y.shape[0]

        return dict(
            loss=loss,
            loss_flow=loss_flow,
            loss_rec=loss_rec,
            acc_class=acc_class,
            acc_env=acc_env,
        )

    def forward_autoencoder(self, x):
        return self.forward_decoder(self.forward_encoder(x))

    def forward_encoder(self, x):
        return self.autoencoder.forward_encoder(x)

    def forward_decoder(self, x):
        return self.autoencoder.forward_decoder(x)

    def forward_latent(self, x):
        return self.flow(self.forward_encoder(x))[0]

    def log_density(
        self, x, embedded_input=False, density_type="latent", location="env"
    ):
        """
        Compute Density of latent variable under the assumption it follows a
        standarad normal distribution
        """

        assert density_type in ["latent"]
        assert location in ["env", "class"]

        if embedded_input:
            print("Warning: Input embedded")
            x = self.forward_autoencoder(x)

        z = self.forward_latent(x)
        # zz = self.cluster_distances_classes(z)
        zz = self.cluster_distances(z, location=location)

        # Not exactly density (Constant term missing)
        log_density = -0.5 * zz  # .mean(1)
        return log_density

    def predict(self, x, location="env", density_type="latent", embedded_input=False):
        """
        ...
        """

        return self.log_density(
            x,
            location=location,
            density_type=density_type,
            embedded_input=embedded_input,
        )

    def cluster_distances(self, z, location="class"):
        """
        Compute quadratic l2-distance of z to each class mixture component

        Return:
            out: Of shape (z.shape[0], self.n_classes)
        """
        assert location in ["class", "env"]

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_clusters = (
            self.hparams.n_classes if location == "class" else self.hparams.n_envs
        )

        if location == "class":
            mu = torch.zeros(n_clusters, self.latent_dims[1]).to(z.device)
        elif location == "env":
            mu = torch.zeros(n_clusters, self.latent_dims[0]).to(z.device)

        x = torch.arange(0, n_clusters).to(z.device)
        cond = F.one_hot(x, num_classes=n_clusters).float()

        with torch.no_grad():
            if location == "class":
                mu = self.mu_class(cond)
            elif location == "env":
                mu = self.mu_env(cond)

        out = torch.zeros(z.shape[0], n_clusters).to(z.device)
        for i in range(n_clusters):
            if location == "class":
                out[:, i] = (
                    (
                        z[
                            :,
                            self.latent_dims[0] : self.latent_dims[1]
                            + self.latent_dims[0],
                        ]
                        - mu[i]
                    )
                    ** 2
                ).mean(1)
            elif location == "env":
                out[:, i] = ((z[:, : self.latent_dims[0]] - mu[i]) ** 2).mean(1)
        return out

    def generate_samples(self, n_samples=16, y=None, e=None):
        self.mu_class.eval()
        self.mu_env.eval()
        self.autoencoder.eval()
        self.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if e == None:
            e = torch.randint(0, self.hparams.n_envs, (n_samples,)).to(device)
            e = F.one_hot(e, num_classes=self.hparams.n_envs).float()
        if y == None:
            y = torch.randint(0, self.hparams.n_classes, (n_samples,)).to(device)
            y = F.one_hot(y, num_classes=self.hparams.n_classes).float()

        assert e.shape[1] == self.hparams.n_envs
        assert y.shape[1] == self.hparams.n_classes

        with torch.no_grad():
            mu_env = self.mu_env(e)
            mu_class = self.mu_class(y)
            mu_residual = torch.zeros(n_samples, self.hparams.latent_dims[2]).to(device)
            mu_all = torch.cat((mu_env, mu_class, mu_residual), 1)

        l = torch.randn(n_samples, self.hparams.latent_dim_total).to(device) + mu_all
        z, _ = self.flow(l, rev=True)

        return self.forward_decoder(z)

    def latent_exchange_generic(self, x, y=None, e=None, location="env"):
        assert location in ["env", "class"]

        latent_code = self.forward_latent(x)

        latent_dims = self.latent_dims

        if location == "env":
            mu_e = self.mu_env(e)

            exchange_env = torch.cat(
                (
                    mu_e,
                    latent_code[:, latent_dims[0] :],
                ),
                1,
            )
            z_env = self.flow(exchange_env, rev=True)[0]
            x_env = self.forward_decoder(z_env)

            return x_env

        if location == "class":
            exchange_class = torch.cat(
                (
                    latent_code[:, : latent_dims[0]],
                    mu_y,
                    latent_code[:, latent_dims[0] + latent_dims[1] :],
                ),
                1,
            )
            mu_y = self.mu_class(y)

            z_class = self.flow(exchange_class, rev=True)[0]

            x_class = self.forward_decoder(z_class)

            return x_class

    def latent_exchange(self, x_1, x_2):
        latent_code_1 = self.forward_latent(x_1)
        latent_code_2 = self.forward_latent(x_2)

        latent_dims = self.latent_dims

        exchange_env = torch.cat(
            (
                latent_code_2[:, : latent_dims[0]],
                latent_code_1[:, latent_dims[0] : latent_dims[0] + latent_dims[1]],
                latent_code_1[:, latent_dims[0] + latent_dims[1] :],
            ),
            1,
        )

        exchange_class = torch.cat(
            (
                latent_code_1[:, : latent_dims[0]],
                latent_code_2[:, latent_dims[0] : latent_dims[0] + latent_dims[1]],
                latent_code_1[:, latent_dims[0] + latent_dims[1] :],
            ),
            1,
        )

        exchange_residual = torch.cat(
            (
                latent_code_1[:, : latent_dims[0]],
                latent_code_1[:, latent_dims[0] : latent_dims[0] + latent_dims[1]],
                latent_code_2[:, latent_dims[0] + latent_dims[1] :],
            ),
            1,
        )

        z_res = self.flow(exchange_residual, rev=True)[0]
        z_class = self.flow(exchange_class, rev=True)[0]
        z_env = self.flow(exchange_env, rev=True)[0]

        x_res = self.forward_decoder(z_res)
        x_class = self.forward_decoder(z_class)
        x_env = self.forward_decoder(z_env)

        return x_env, x_class, x_res

def subnet_fc_mnist(dim_in, dim_out):
    dim_latent = 1024
    subnet = nn.Sequential(nn.Linear(dim_in, dim_latent), nn.BatchNorm1d(dim_latent),
                        nn.ReLU(), nn.Dropout(p=0.5),
                        nn.Linear(dim_latent,  dim_latent), nn.BatchNorm1d(dim_latent),
                        nn.ReLU(), nn.Dropout(p=0.5),
                        nn.Linear(dim_latent,  dim_out))

    for l in subnet:
        if isinstance(l, nn.Linear):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet

def configure_mnist_inn(inp_dim, cond_dim, n_layers):
    model = Ff.SequenceINN(inp_dim)
    for k in range(n_layers):
        model.append(Fm.AllInOneBlock, cond=0, cond_shape=[cond_dim], subnet_constructor=subnet_fc_mnist)
    return model
