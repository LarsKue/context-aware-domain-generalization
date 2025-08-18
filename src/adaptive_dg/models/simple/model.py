
import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning_trainable.hparams import Choice, HParams
from lightning_trainable.modules import FullyConnectedNetwork

from ..domain_generalizer import DomainGeneralizer, DGHParams
from ..set_encoders import SingleBlockSetEncoder, SingleBlockSetEncoderHParams, SimpleSetEncoder, SimpleSetEncoderHParams


class FeatureExtractorHParams(HParams):
    name: str = "resnet50"
    features: int = 256
    frozen: bool = True


class SetEncoderHParams(HParams):
    name: str = "mean8"
    features: int = 256
    frozen: bool = False


class FCHParams(HParams):
    layer_widths: list[int] = [256, 256]
    activation: str = "relu"
    dropout: float = 0.0


class SimpleModelHParams(DGHParams):
    feature_extractor: FeatureExtractorHParams = FeatureExtractorHParams.defaults()
    set_encoder: SetEncoderHParams | None = SetEncoderHParams.defaults()
    fc: FCHParams = FCHParams.defaults()
    mode: Choice("classification", "regression") = "classification"
    target: Choice("y", "e") = "e"


class SimpleModel(DomainGeneralizer):
    hparams: SimpleModelHParams

    def __init__(self, hparams):
        super().__init__(hparams)

        self.feature_extractor = self.configure_feature_extractor()
        self.fc = self.configure_fc()
        self.set_encoder = self.configure_set_encoder()

    def compute_metrics(self, batch, batch_idx) -> dict:
        x, x_set, y, e = batch

        batch_size, set_size, *_ = x_set.shape

        x = self.feature_extractor(x)

        if self.set_encoder is not None:
            x_set = x_set.flatten(0, 1)
            x_set = self.feature_extractor(x_set)
            x_set = x_set.unflatten(0, (batch_size, set_size))

            x_set = self.set_encoder(x_set)
            x_set = x_set.mean(dim=1)

            x = torch.cat([x, x_set], dim=1)

        output = self.fc(x)

        match self.hparams.target:
            case "y":
                target = y
            case "e":
                target = e.float()
            case other:
                raise NotImplementedError(f"Unreachable")

        match self.hparams.mode:
            case "classification":
                cross_entropy = F.cross_entropy(output, target)
                predictions = output.argmax(1)
                targets = target.argmax(1)

                accuracy = (predictions == targets).float().mean(0)

                metrics = dict(
                    loss=cross_entropy,
                    cross_entropy=cross_entropy,
                    accuracy=accuracy,
                )
            case "regression":
                mse = F.mse_loss(output, target)

                metrics = dict(
                    loss=mse,
                    mse=mse,
                )
            case other:
                raise NotImplementedError(f"Unreachable")

        return metrics

    def compute_ood_metrics(self, batch, batch_idx):
        return self.compute_metrics(batch, batch_idx)

    def configure_feature_extractor(self):
        match self.hparams.feature_extractor.name:
            case None:
                return None
            case str() as name:
                network = torch.hub.load("pytorch/vision:v0.10.0", name, pretrained=True)
                network.conv1 = nn.Conv2d(self.image_shape[0], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                network.fc = nn.Linear(network.fc.in_features, self.hparams.feature_extractor.features)
            case other:
                raise NotImplementedError(f"Unrecognized feature extractor: '{other}'")

        if self.hparams.feature_extractor.frozen:
            network.requires_grad_(False)

        return network

    def configure_set_encoder(self):
        if self.hparams.set_encoder is None:
            return None

        match self.hparams.set_encoder.name:
            case "mean8":
                hparams = dict(
                    input_dims=self.hparams.feature_extractor.features,
                    output_dims=self.hparams.set_encoder.features,
                    layer_widths=[256] * 8,
                    activation="relu",
                    dropout=0.25,
                    input_set_size=self.hparams.set_size,
                    output_set_size=1,
                    pooling_method="mean",
                    pooling_kwargs=dict(),
                    flatten=False,
                )
                network = SingleBlockSetEncoder(hparams)
            case "attention16":
                hparams = dict(
                    dimensions=[self.hparams.feature_extractor.features, 256, 256, 256, self.hparams.set_encoder.features],
                    set_sizes=[self.hparams.set_size // (2 ** i) for i in range(5)],
                    layer_widths=[[256] * 4] * 4,
                    activation="relu",
                    dropout=0.25,
                    pooling_methods="attention",
                    pooling_kwargs=dict(heads=4),
                    flatten=False,
                )
                network = SimpleSetEncoder(hparams)
            case other:
                raise NotImplementedError(f"Unrecognized set encoder: '{other}'")

        if self.hparams.set_encoder.frozen:
            network.requires_grad_(False)

        return network

    def configure_fc(self):
        input_dims = self.hparams.feature_extractor.features

        if self.hparams.set_encoder is not None:
            input_dims += self.hparams.set_encoder.features

        match self.hparams.target:
            case "y":
                output_dims = self.n_classes
            case "e":
                output_dims = self.n_domains
            case other:
                raise NotImplementedError(f"Unreachable")

        hparams = dict(
            input_dims=input_dims,
            output_dims=output_dims,
            **self.hparams.fc
        )

        return FullyConnectedNetwork(hparams)

