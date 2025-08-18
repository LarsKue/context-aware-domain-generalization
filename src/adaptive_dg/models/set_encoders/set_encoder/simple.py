
import torch
import torch.nn as nn

from lightning_trainable.modules import HParamsModule

from .block import SetEncoderBlock
from .simple_hparams import SimpleSetEncoderHParams


class SimpleSetEncoder(HParamsModule):
    hparams: SimpleSetEncoderHParams

    def __init__(self, hparams: dict | SimpleSetEncoderHParams):
        super().__init__(hparams)

        self.blocks = self.configure_blocks()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

    def configure_blocks(self) -> nn.Module:
        blocks = nn.Sequential()

        for i in range(self.hparams.n_blocks):

            if isinstance(self.hparams.pooling_methods, str):
                pooling_method = self.hparams.pooling_methods
            else:
                pooling_method = self.hparams.pooling_methods[i]

            if isinstance(self.hparams.pooling_kwargs, dict):
                pooling_kwargs = self.hparams.pooling_kwargs
            else:
                pooling_kwargs = self.hparams.pooling_kwargs[i]

            block_hparams = dict(
                input_dims=self.hparams.dimensions[i],
                output_dims=self.hparams.dimensions[i + 1],
                layer_widths=self.hparams.layer_widths[i],
                activation=self.hparams.activation,
                dropout=self.hparams.dropout,
                input_set_size=self.hparams.set_sizes[i],
                output_set_size=self.hparams.set_sizes[i + 1],
                pooling_method=pooling_method,
                pooling_kwargs=pooling_kwargs,
                flatten=False,
            )
            blocks.append(SetEncoderBlock(block_hparams))

        if self.hparams.flatten:
            blocks.append(nn.Flatten())

        return blocks
