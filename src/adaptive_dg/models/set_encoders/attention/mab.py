
import torch
import torch.nn as nn

from torch import Tensor


class MultiheadAttentionBlock(nn.Module):
    """
    Multi-Head Attention Block as described in arXiv:1810.00825
    This is a variant of a transformer block without positional encoding or dropout.
    """

    def __init__(self, query_features: int, key_features: int, value_features: int, heads: int = 2):
        super().__init__()
        self.register_buffer("query_features", torch.tensor(query_features))
        self.register_buffer("key_features", torch.tensor(key_features))
        self.register_buffer("value_features", torch.tensor(value_features))
        self.register_buffer("heads", torch.tensor(heads))

        self.att = nn.MultiheadAttention(embed_dim=query_features, kdim=key_features, vdim=value_features, num_heads=heads, batch_first=True)
        self.rFF = nn.Linear(value_features, value_features)
        self.rFF.weight.data[:] = torch.eye(value_features)
        self.rFF.bias.data.zero_()

        self.ln1 = nn.LayerNorm(value_features)
        self.ln2 = nn.LayerNorm(value_features)

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        # Eq. 6 and 7
        h = self.ln1(query + self.att(query, key, key)[0])

        return self.ln2(h + self.rFF(h))

    def extra_repr(self) -> str:
        return f"query_features={self.query_features.item()}, key_features={self.key_features.item()}, value_features={self.value_features.item()}, heads={self.heads.item()}"
