
from lightning_trainable.hparams import Choice, HParams


class SetEncoderBlockHParams(HParams):
    input_dims: int
    output_dims: int
    layer_widths: list[int]
    activation: str = "relu"
    dropout: float = 0.0

    last_layer_activation: bool = False
    last_layer_dropout: bool = False

    input_set_size: int
    output_set_size: int = 1
    pooling_method: Choice("attention", "topk", "mean", "sum", "none")
    pooling_kwargs: dict = {}

    flatten: bool = True

    @classmethod
    def validate_parameters(cls, hparams):
        hparams = super().validate_parameters(hparams)

        match hparams.pooling_method:
            case "mean" | "sum":
                if hparams.output_set_size != 1:
                    raise ValueError(f"Pooling method {hparams.pooling_method} requires output_set_size=1")
            case "none":
                if hparams.input_set_size != hparams.output_set_size:
                    raise ValueError(f"Pooling method {hparams.pooling_method} requires input_set_size=output_set_size")
            case "topk":
                if hparams.input_set_size < hparams.output_set_size:
                    raise ValueError(f"Pooling method {hparams.pooling_method} requires input_set_size >= output_set_size")

        return hparams
