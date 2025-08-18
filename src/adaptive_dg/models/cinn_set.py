import warnings
import importlib

from torch.utils.data import DataLoader  # , Dataset, IterableDataset
import torch

from adaptive_dg.models.flows import (
    subnet_fc,
)
from adaptive_dg.models.set_encoder import SetEncoder

import FrEIA.framework as Ff
import FrEIA.modules as Fm


from torch import nn

from lightning_trainable import Trainable, utils, TrainableHParams


from adaptive_dg.plots.sampling import (
    plot_data_sampling,
    # plot_latent_space_exchange,
    plot_latent_space_exchange_generic,
)


class CondINNSetHParams(TrainableHParams):
    # Model specific hyperparameters
    latent_dim_total: int = 64
    n_layers_flow: int = 8
    optimizer: dict = {"name": "Adam", "lr_flow": 1e-4, "lr_set": 1e-7, "lr_auto": 1e-5}
    lr_scheduler: dict  =  {"name": "StepLR", "step_size":  1000, "gamma": 0.5}
    autoencoder_cfg: dict
    noise_scaling: float = 1e-2
    reconstruction_weight: float = 100.0
    joint_training: bool  = True
    cond_dim: int = 32
    set_encoder_dim: int = 32 
    set_encoder_heads: int = 32

    # Dataset specific hyperparameters
    n_envs: int = 5
    n_classes: int = 10


class CondINNSet(Trainable):
    """
    A  Trainable-class for an invertible neural network (INN) which  gets a set a set encoder output as condition.
    Inputs to the set encoder and INN are encoding of a Auto-Encoder. The Auto-Encoder can be pre-trained or/and can be trained during draining
    """

    hparams: CondINNSetHParams

    def __init__(self, hparams):
        super().__init__(hparams)

        self.set_encoder = SetEncoder(
            self.hparams.latent_dim_total,
            self.hparams.cond_dim,
            dim_encoder=self.hparams.set_encoder_dim,
            dim_decoder=self.hparams.cond_dim,
            #pma_args={"num_heads": self.hparams.set_encoder_heads},
        )
        self.flow = Ff.SequenceINN(self.hparams.latent_dim_total)
        for k in range(self.hparams.n_layers_flow):
            self.flow.append(
                Fm.AllInOneBlock,
                subnet_constructor=subnet_fc,
                cond=0,
                #cond_shape=(self.hparams.cond_dim + self.hparams.n_classes,),
                cond_shape=(self.hparams.cond_dim + self.hparams.n_classes + 1,),
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

        self.joint_training = self.hparams.joint_training
        if not self.joint_training:
            print("No joint training")
            for p in self.autoencoder.parameters():
                p.requires_grad = False

        ##### Environment Classifier
        self.env_classifier = nn.Sequential( nn.Linear(self.hparams.cond_dim, 128), nn.ReLU(), nn.Linear(128,self.hparams.n_envs))
        self.loss_classifier = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        """
        Configure optimizers for Lightning
        """
        kwargs = dict()

        if self.hparams.joint_training:
            parameters = [
                {
                    "params": list(self.flow.parameters()),
                    "lr": self.hparams.optimizer["lr_flow"],
                },
                {
                    "params": list(self.set_encoder.parameters()),
                    "lr": self.hparams.optimizer["lr_set"],
                },
                {
                    "params": list(self.env_classifier.parameters()),
                    "lr": self.hparams.optimizer["lr_flow"],
                },
            ]
        else:
            parameters = [
                {
                    "params": list(self.flow.parameters()),
                    "lr": self.hparams.optimizer["lr_flow"],
                },
                {
                    "params": list(self.set_encoder.parameters()),
                    "lr": self.hparams.optimizer["lr_set"],
                },
                {
                    "params": list(self.autoencoder.parameters()),
                    "lr": self.hparams.optimizer["lr_auto"],
                },
                {
                    "params": list(self.env_classifier.parameters()),
                    "lr": self.hparams.optimizer["lr_flow"],
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

    def compute_metrics(self, batch, batch_idx):
        """
        Compute different Metrics. Here Reconstruction loss, Log-Likelihood losses, accuracy-class
        """

        x, y, e = batch

        x += self.hparams.noise_scaling * torch.randn_like(x)
        if not self.joint_training:
            with torch.no_grad():
                emb = self.forward_encoder(x)
        else:
            emb = self.forward_encoder(x)

        #emb += self.hparams.noise_scaling * torch.randn_like(emb)

        #emb += self.hparams.noise_scaling * torch.randn_like(emb)
        condition = torch.zeros((emb.shape[0], self.hparams.cond_dim)).to(x.device)

        for je in range(e.shape[1]):
            mask_e = (e[:, je] == 1).view(-1).to(x.device)

            #idx = torch.randperm(mask_e.sum())

            if mask_e.sum() > 0:

                set_input = emb[mask_e][None, :, :]#.repeat(mask_e.sum(), 1, 1)
                condition[mask_e] = self.set_encoder(set_input).repeat(mask_e.sum(), 1)
                #for i  in  range(mask_e.sum()):
                #    idx  = torch.randperm(mask_e.sum())[:25]
                #    condition[mask_e] = self.set_encoder(set_input[:,idx,:])

                #idx = torch.cat([torch.randperm(mask_e.sum()).view(-1,1)[:10] for i in range(mask_e.sum().item())], 1)
                #import pdb
                #set_input = emb[mask_e][None, :, :].repeat(mask_e.sum(),1 ,1)[idx]
                #condition[mask_e] = self.set_encoder(set_input)
        
        env_pred  = self.env_classifier(condition)
        loss_classifier = self.loss_classifier(env_pred, e) 

        #z, log_jac_det = self.flow(emb, c=[torch.cat((y, condition), 1)])
        #z, log_jac_det = self.flow(emb[:16], c=[torch.cat((y[:16], condition[:16], env_pred.argmax(1).view(-1,1)[:16]), 1)])
        z, log_jac_det = self.flow(emb, c=[torch.cat((y, condition, env_pred.argmax(1).view(-1,1)), 1)])
        loss = (
            (0.5 * torch.sum((z) ** 2, 1) - log_jac_det)
        ).mean() / self.hparams.latent_dim_total

        loss += loss_classifier
        # Uncommented for debigguiung
        '''
        z, log_jac_det = self.flow(emb, c=[torch.cat((0 * y, condition), 1)])
        loss += (
            (0.5 * torch.sum((z) ** 2, 1) - log_jac_det)
        ).mean() / self.hparams.latent_dim_total
        '''

        if self.joint_training:
            x_rec = self.forward_decoder(emb)
            loss_rec = self.autoencoder.reconstruction_loss_fct(x, x_rec)
            loss += self.hparams.reconstruction_weight * loss_rec
        else:
            loss_rec = 0

        

        # Accuracy
        with torch.no_grad():
            acc_class = (
                self.predict(x, domains=e, location="class").argmax(1) == y.argmax(1)
                #self.predict(x, domains=e, location="class").argmax(1) == y[:16].argmax(1)
            ).sum() / y.shape[0]

            acc_env = (
                self.env_classifier(condition).argmax(1) == e.argmax(1)
                #self.env_classifier(condition[:16]).argmax(1) == e[:16].argmax(1)
            ).sum() / e.shape[0]

        return dict(
            loss=loss,
            loss_rec=loss_rec,
            loss_classifier=loss_classifier,
            acc_class=acc_class,
            acc_env=acc_env,
        )

    def ood_dataloader(self):
        """
        Configure and return the ood dataloader. Has to be invorked actively
        """
        if self.ood_data is None:
            return None
        return DataLoader(
            dataset=self.ood_data,
            batch_size=self.hparams.batch_size // self.hparams.n_envs,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
            num_workers=self.hparams.num_workers,
        )

    def on_validation_epoch_end(self) -> None:
        """
        Plots/Metrics after Vaidation Epoch:
            - Plot: Sampling from model
            - Plot: Latent Space Exchange
            - Metric: Accuracy on ood data
        """

        # Save refernece data that characterize the environments
        self.save_reference_data(self.val_data)

        # Sampling from model
        fig = plot_data_sampling(self)
        self.logger.experiment.add_figure(
            "Sampling",
            fig,
            global_step=int(
                self.trainer.global_step / self.trainer.num_training_batches
            ),
        )

        # Latent Space Exchange (swicht domains)
        fig = plot_latent_space_exchange_generic(self, self.train_data)
        self.logger.experiment.add_figure(
            "Latent Space Ex.",
            fig,
            global_step=int(
                self.trainer.global_step / self.trainer.num_training_batches
            ),
        )

        # Compute Accuracy on OOD dataset (TODO: current implementation needs improvement)
        acc_class_ood = 0
        n_iterations = 0
        self.eval()
        for x, y, _ in self.ood_dataload:
            #y = y[:16]
            with torch.no_grad():
                out = self.predict(x.float().to(self.device), location="class")
            acc_class_ood += ((
                out.argmax(1) == y.to(self.device).argmax(1)
            ).sum() / y.shape[0]) * (y.shape[0] / self.ood_dataload.batch_size) 
            n_iterations += (y.shape[0] / self.ood_dataload.batch_size) 
        self.train()

        self.log(f"ood/acc_class", acc_class_ood / n_iterations)

    def predict(self, x, domains=None, location="class", density_type="feature"):
        """
        Predict class (if location=='class') or environment (if location=='env') from  input x
        Currently: Only location='class' and density_type='feature' implemented
        """

        assert density_type in ["feature"]
        assert location in ["class"]

        if location == "class" and density_type == "feature":
            return self.log_likelihood_feature_class(x, domains=domains)
        return

    def log_likelihood_feature_class(self, x, domains=None):
        """
        Currently we use an uninformed prior p(y). But it should be an informed one, estimated from the validation/training set
        Also: We only compute the log-likelihood up to a constant
        """
        enc = self.forward_encoder(x)
        if domains == None:
            set_input = enc[None, :, :]
            set_condition = self.set_encoder(set_input).repeat(x.shape[0], 1 )
        else:
            set_condition = torch.zeros((x.shape[0], self.hparams.cond_dim)).to(
                x.device
            )
            for je in range(domains.shape[1]):
                mask_e = (domains[:, je] == 1).view(-1)
                if mask_e.sum() > 0:
                    set_input = enc[mask_e][None, :, :]
                    set_condition[mask_e] = self.set_encoder(set_input).repeat(
                        mask_e.sum(), 1
                    )

        env_pred  = self.env_classifier(set_condition).argmax(1).view(-1,1)

        densities = torch.zeros(x.shape[0], self.hparams.n_classes).to(x.device)
        #densities = torch.zeros(16, self.hparams.n_classes).to(x.device)
        for d in range(self.hparams.n_classes):
            cond = torch.zeros(enc.shape[0], self.hparams.n_classes).to(x.device)
            cond[:, d] = 1
            #z, log_jac_det = self.flow(enc, c=[torch.cat((cond, set_condition), 1)])
            z, log_jac_det = self.flow(enc, c=[torch.cat((cond, set_condition, env_pred), 1)])

            log_density = (
                -(0.5 * torch.sum((z) ** 2, 1) - log_jac_det)  # - prior[d] (prior)
                / self.hparams.latent_dim_total
            )
            densities[:, d] = log_density  # torch.exp(-0.5 * z_c**2).mean(1)
        return densities

    def save_reference_data(self, dataset):
        """
        Create refernece conditions for the flow model.
        This means we compute for each domain the output of the set_encoder.
        This output then represents the corresponding domain.
        As input sets we consider all elements that belong to a domain for a batch
        of size self.hparams.batch_size
        """

        device = self.device
        x, y, d = dataset[:]

        batch_size = self.hparams.batch_size
        x, y, d = (
            x[:batch_size].to(device).float(),
            y[:batch_size].to(device).float(),
            d[:batch_size].to(device).float(),
        )
        with torch.no_grad():
            emb = self.forward_encoder(x)

        self.reference_data = []

        condition = torch.zeros((emb.shape[0], self.hparams.cond_dim)).to(device)
        for jd in range(self.hparams.n_envs):
            mask_d = (d[:, jd] == 1).view(-1)

            if mask_d.sum() > 0:
                set_input = emb[mask_d][None, :, :]

                with torch.no_grad():
                    condition = self.set_encoder(set_input)
                    self.reference_data.append(condition.cpu())
            else:
                warnings.warn("Reference data can be build for all classes")

        self.reference_data = torch.cat(self.reference_data)

    def forward_autoencoder(self, x):
        """
        Compute encoding and decoding of input x (aka reconstruction)
        """
        return self.forward_decoder(self.forward_encoder(x))

    def forward_encoder(self, x):
        """
        Compute Encoding of input x
        """
        return self.autoencoder.forward_encoder(x)

    def forward_decoder(self, enc):
        """
        Compute Decoding  of input enc
        """
        return self.autoencoder.forward_decoder(enc)

    def forward_latent(self, x, d=0, y=0):
        """
        Compute Latent Space encoding of Flow [Not to confuse with latent space of autoencoder]
        """
        emb = self.forward_encoder(x)
        own_condition = self.set_encoder(emb[:, None, :])  # .repeat(x.shape[0],1)

        z = self.flow(
            emb,
            c=[
                torch.cat(
                    (
                        torch.zeros(x.shape[0], self.hparams.n_classes).to(x.device),
                        own_condition,
                    ),
                    1,
                )
            ],
            rev=False,
        )[0]
        return z

    def latent_exchange_generic(self, x, y=None, d=None, location="env"):
        """
        Domain- and class-transfer for x to environment d or class y (Currently only domain-transfer is working)
        """
        assert location in ["env"]

        emb = self.forward_encoder(x)

        if location == "env":
            assert d is not None

            d = d.argmax(1)

            condition = self.reference_data.to(x.device)[d]  # .repeat(n_samples, 1)
            env_pred  = self.env_classifier(condition).argmax(1).view(-1,1)

            own_condition = self.set_encoder(emb[:, None, :])
            env_pred_own = self.env_classifier(own_condition).argmax(1).view(-1,1)

            z = self.flow(
                emb,
                c=[
                    torch.cat(
                        (
                            torch.zeros(x.shape[0], self.hparams.n_classes).to(
                                x.device
                            ),
                            own_condition, env_pred_own
                        ),
                        1,
                    )
                ],
                rev=False,
            )[0]

            z_env = self.flow(
                z,
                c=[
                    torch.cat(
                        (
                            torch.zeros(x.shape[0], self.hparams.n_classes).to(
                                x.device
                            ),
                            condition, env_pred
                        ),
                        1,
                    )
                ],
                rev=True,
            )[0]
            x_env = self.forward_decoder(z_env)

            return x_env

    def latent_exchange(self, x1, x2):
        latent_code = self.forward_latent(x)

        condition = torch.zeros((x.shape[0], self.cond_dim)).cuda()
        for je in range(self.n_envs):
            mask_e = (e[:, je] == 1).view(-1)

            if mask_e.sum() > 0:
                set_input = emb[mask_e][None, :, :]

            condition[mask_e] = self.set_encoder(set_input).repeat(mask_e.sum(), 1)

        return x_env

    def generate_samples(self, y, d, n_samples=16):
        """
        Generate samples according to class y and domain d
        """
        device = self.device

        l = torch.randn(n_samples, self.hparams.latent_dim_total).to(device)

        d = d.argmax(1)

        condition = self.reference_data.to(device)[d]
        env_pred  = self.env_classifier(condition).argmax(1).view(-1,1)
        z = self.flow(l, c=[torch.cat((y, condition, env_pred), 1)], rev=True)[0]

        return self.forward_decoder(z)
