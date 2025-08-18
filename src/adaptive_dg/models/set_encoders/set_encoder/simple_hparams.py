
from lightning_trainable.hparams import HParams


class SimpleSetEncoderHParams(HParams):
    dimensions: list[int]
    set_sizes: list[int]
    layer_widths: list[list]
    pooling_methods: str | list[str] = "attention"
    pooling_kwargs: dict | list[dict] = {}

    activation: str = "relu"
    dropout: float = 0.0

    flatten: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_blocks = len(self.layer_widths)

    @classmethod
    def validate_parameters(cls, hparams):
        hparams = super().validate_parameters(hparams)

        n_blocks = len(hparams.layer_widths)

        if n_blocks < 1:
            raise ValueError(f"Expected at least 1 block, got {n_blocks}.")

        if not len(hparams.dimensions) == n_blocks + 1:
            # +1 for input and output dimensions
            raise ValueError(f"Expected {n_blocks} + 1 dimensions, got {len(hparams.dimensions)}.")

        if not len(hparams.set_sizes) == n_blocks + 1:
            # +1 for input and output set sizes
            raise ValueError(f"Expected {n_blocks} + 1 set_sizes, got {len(hparams.set_sizes)}.")

        if isinstance(hparams.pooling_methods, list):
            if not len(hparams.pooling_methods) == n_blocks:
                raise ValueError(f"Expected {n_blocks} pooling_methods, got {len(hparams.pooling_methods)}.")

        if isinstance(hparams.pooling_kwargs, list):
            if not len(hparams.pooling_kwargs) == n_blocks:
                raise ValueError(f"Expected {n_blocks} pooling_kwargs, got {len(hparams.pooling_kwargs)}.")

        return hparams
