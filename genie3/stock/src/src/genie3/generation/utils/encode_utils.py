"""
Encoding utilities.

Functions for encoding categorical and continuous features.
"""

import math
import torch


def sinusoidal_encoding(v: torch.Tensor, N: int, D: int) -> torch.Tensor:
    """
    Generate sinusoidal positional encodings.

    Args:
        v: Input values to encode [*]
        N: Maximum value for normalization
        D: Encoding dimension (must be even)

    Returns:
        torch.Tensor: Sinusoidal encodings [*, D]
    """
    # v: [*]

    # [D]
    k = torch.arange(1, D + 1).to(v.device)

    # [*, D]
    sin_div_term = N ** (2 * k / D)
    sin_div_term = sin_div_term.view(*((1,) * len(v.shape) + (len(sin_div_term),)))
    sin_enc = torch.sin(v.unsqueeze(-1) * math.pi / sin_div_term)

    # [*, D]
    cos_div_term = N ** (2 * (k - 1) / D)
    cos_div_term = cos_div_term.view(*((1,) * len(v.shape) + (len(cos_div_term),)))
    cos_enc = torch.cos(v.unsqueeze(-1) * math.pi / cos_div_term)

    # [*, D]
    enc = torch.zeros_like(sin_enc).to(v.device)
    enc[..., 0::2] = cos_enc[..., 0::2]
    enc[..., 1::2] = sin_enc[..., 1::2]

    return enc
