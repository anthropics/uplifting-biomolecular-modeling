import torch

from .linear import LinearNoBias


class SwiGLU(torch.nn.Module):
    """SiLU Gated Linear Unit (SwiGLU) activation function."""

    def __init__(self, channel_in: int, channel_out: int):
        super().__init__()
        self.linear = LinearNoBias(channel_in, channel_out * 2, init="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.linear(x).chunk(2, dim=-1)
        return torch.nn.functional.silu(a) * b
