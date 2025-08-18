
import torch
import torch.nn as nn

from lightning_trainable.modules import HParamsModule
from lightning_trainable.modules import FullyConnectedNetwork

from ..attention import *
from ..mean import Mean
from ..sum import Sum
from ..topk import TopK

from .block_hparams import SetEncoderBlockHParams


class SetEncoderBlock(HParamsModule):
    hparams: SetEncoderBlockHParams

    def __init__(self, hparams: dict | SetEncoderBlockHParams):
        super().__init__(hparams)

        self.network = self.configure_network()
        self.pooling = self.configure_pooling()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        x = self.pooling(x)

        return x

    def configure_network(self) -> nn.Module:
        fc_hparams = dict(
            input_dims=self.hparams.input_dims,
            output_dims=self.hparams.output_dims,
            layer_widths=self.hparams.layer_widths,
            activation=self.hparams.activation,
            dropout=self.hparams.dropout,
        )

        if len(self.hparams.layer_widths) > 0:
            network = FullyConnectedNetwork(fc_hparams)
            if self.hparams.last_layer_dropout:
                network.network.insert(-1, network.configure_dropout())

            if self.hparams.last_layer_activation:
                network.network.append(network.configure_activation())
        else:
            network = nn.Sequential(nn.Linear(self.hparams.input_dims, self.hparams.output_dims))

        return network

    def configure_pooling(self) -> nn.Module:
        pooling = nn.Sequential()

        match self.hparams.pooling_method:
            case "mean":
                pooling.append(Mean(**self.hparams.pooling_kwargs))
            case "sum":
                pooling.append(Sum(**self.hparams.pooling_kwargs))
            case "topk":
                pooling.append(TopK(k=self.hparams.output_set_size, **self.hparams.pooling_kwargs))
            case "attention":
                kwargs = dict(
                    features=self.hparams.output_dims,
                    seeds=self.hparams.output_set_size,
                    heads=self.hparams.pooling_kwargs.get("heads", 4),
                )
                pooling.append(PoolingByMultiheadAttention(**kwargs))
            case "none":
                pass
            case other:
                raise ValueError(f"Unrecognized pooling method: '{other}'")

        if self.hparams.flatten:
            pooling.append(nn.Flatten())

        return pooling
