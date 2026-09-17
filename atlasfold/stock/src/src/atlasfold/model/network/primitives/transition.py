import torch

from .activation import SwiGLU
from .linear import LinearNoBias
from .normalization import LayerNorm


class Transition(torch.nn.Sequential):
    """Perform a two-layer MLP.
    See Section 3.3 Algorithm 11 Transition layer
    """

    def __init__(self, channel: int, expansion_factor: int) -> None:
        """Initialize the Transition module.

        Parameters
        ----------
        channel: int
            The dimension of the input
        expansion_factor: int
            The expansion factor for the hidden dimension
        """
        hidden_dim = channel * expansion_factor
        super().__init__(
            LayerNorm(channel),
            SwiGLU(channel, hidden_dim),
            LinearNoBias(hidden_dim, channel, init="final"),
        )
