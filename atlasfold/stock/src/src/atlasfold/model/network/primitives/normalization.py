import torch
import torch.nn as nn

from .linear import Linear, LinearNoBias


class LayerNorm(nn.Module):
    """Basic LayerNorm layer with learnable scale and offset."""

    def __init__(
        self,
        normalized_shape: int,
        create_scale: bool = True,
        create_offset: bool = True,
        eps=1e-5,
    ):
        super().__init__()
        self.normalized_shape: int = normalized_shape
        self.eps: float = eps
        if create_scale:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.weight = None
        if create_offset:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.dtype
        x = x.float()
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        out = nn.functional.layer_norm(
            input=x,
            normalized_shape=(self.normalized_shape,),
            weight=weight,
            bias=bias,
            eps=self.eps,
        )
        return out.to(d)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization
    See Section 3.7 Algorithm 26 Adaptive LayerNorm
    """

    def __init__(self, channel: int, channel_cond: int):
        """Initialize the adaptive layer normalization.

        Parameters
        ----------
        channel : int
            The input dimension.
        channel_cond : int
            The condition dimension.

        """
        super().__init__()
        self.layernorm = LayerNorm(channel, create_scale=False, create_offset=False)
        self.layernorm_cond = LayerNorm(
            channel_cond, create_scale=True, create_offset=False
        )
        self.linear_g = Linear(channel_cond, channel, init="gating")
        self.linear_bias = LinearNoBias(channel_cond, channel, init="final")
        self.sigmoid = nn.Sigmoid()

    def forward(self, a: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """see Section 3.7 Algorithm 26 Adaptive LayerNorm"""
        # Line 1
        a = self.layernorm(a)
        # Line 2
        cond = self.layernorm_cond(cond)
        # Line 3
        a = self.sigmoid(self.linear_g(cond)) * a + self.linear_bias(cond)
        return a
