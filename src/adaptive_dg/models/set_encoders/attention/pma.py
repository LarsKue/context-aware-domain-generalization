
import torch
import torch.nn as nn

from torch import Tensor


from .mab import MultiheadAttentionBlock


class PoolingByMultiheadAttention(nn.Module):
    """
    Pooling by Multi-Head Attention as described in arXiv:1810.00825
    This block applies Multi-Head Attention to the input set with a learnable set of seed vectors.
    For an input tensor of shape (batch_size, set_size, features),
    the output tensor will have shape (batch_size, seeds, features).
    """
    def __init__(self, features: int, heads: int = 4, seeds: int = 32):
        super().__init__()
        self.register_buffer("features", torch.tensor(features, dtype=torch.long))
        self.register_buffer("heads", torch.tensor(heads, dtype=torch.long))

        self.seeds = nn.Parameter(torch.empty(1, seeds, features))
        nn.init.xavier_uniform_(self.seeds)

        self.mab = MultiheadAttentionBlock(features, features, features, heads=heads)
        self.rFF = nn.Linear(features, features)
        self.rFF.weight.data[:] = torch.eye(features)
        self.rFF.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        # Eq. 11
        seeds = self.seeds.repeat(x.shape[0], 1, 1)
        return self.mab(seeds, self.rFF(x))

    def extra_repr(self) -> str:
        return f"features={self.features.item()}, heads={self.heads.item()}, seeds={self.seeds.size(0)}"
