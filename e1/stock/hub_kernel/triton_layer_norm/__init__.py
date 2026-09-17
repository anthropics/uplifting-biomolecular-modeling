"""Triton layer normalization kernels

This kernel implements layers normalization using Triton. This kernel is from
the `flash-attention <https://github.com/Dao-AILab/flash-attention>`_ project.
"""

from typing import Optional

import torch

from . import layers
from .layer_norm import layer_norm_fn, layer_norm_linear_fn, rms_norm_fn


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    x1: Optional[torch.Tensor] = None,
    weight1: Optional[torch.Tensor] = None,
    bias1: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    dropout_p: float = 0.0,
    rowscale=None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    zero_centered_weight: bool = False,
    is_rms_norm: bool = False,
    return_dropout_mask: bool = False,
    out: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
):
    """
    Apply layer normalization to the input tensor with Triton acceleration.

    Args:
        x (`torch.Tensor`):
            Input tensor to normalize.
        weight (`torch.Tensor`):
            Scale parameter for normalization.
        bias (`torch.Tensor`):
            Shift parameter for normalization.
        residual (`torch.Tensor`, *optional*):
            Optional residual tensor to add to the input before normalization.
        x1 (`torch.Tensor`, *optional*):
            Optional second input tensor to combine with `x`. When provided, the function
            first adds `x1` to `x` and then applies normalization.
        weight1 (`torch.Tensor`, *optional*):
            Scale parameter for the second normalization.
        bias1 (`torch.Tensor`, *optional*):
            Shift parameter for the second normalization.
        eps (`float`, *optional*, defaults to 1e-6):
            Small constant added for numerical stability in normalization.
        dropout_p (`float`, *optional*, defaults to 0.0):
            Dropout probability. If greater than 0, applies dropout to the input before
            normalization and residual addition.
        rowscale (`torch.Tensor`, *optional*):
            Optional scaling factor applied to each row of the input tensor.
            Not compatible with the use of `x1`.
        prenorm (`bool`, *optional*, defaults to False):
            If True, returns both the normalized output and the unnormalized input+residual.
        residual_in_fp32 (`bool`, *optional*, defaults to False):
            If True, performs the residual connection in FP32 precision.
        zero_centered_weight (`bool`, *optional*, defaults to False):
            When set to true, 1.0 is added to the weight before applying it.
        is_rms_norm (`bool`, *optional*, defaults to False):
            If True, uses RMS normalization instead of layer normalization.
        return_dropout_mask (`bool`, *optional*, defaults to False):
            If True, returns the dropout mask used for the computation.
        out (`torch.Tensor`, *optional*):
            Output tensor for the normalized result. If `None`, a new tensor is allocated.
        residual_out (`torch.Tensor`, *optional*):
            Output tensor for the residual result when using prenorm. If `None`, a new tensor
            is allocated when needed.

    Returns:
        `torch.Tensor` or tuple of `torch.Tensor`:
            - The normalized input.
            - The second normalization of the input if `weight1` is provided.
            - The residual tensor if `prenorm` is set.
            - The dropout mask if `return_dropout_mask` is set.
            - The dropout mask for `x1` if `x1` is provided and `return_dropout_mask` is set.
    """
    return layer_norm_fn(
        x,
        weight,
        bias,
        residual,
        x1,
        weight1,
        bias1,
        eps,
        dropout_p,
        rowscale,
        prenorm,
        residual_in_fp32,
        is_rms_norm,
        return_dropout_mask,
        out=out,
        residual_out=residual_out,
    )


__kernel_metadata__ = {
    "license": "bsd-3-clause",
}


__all__ = [
    "__kernel_metadata__",
    "layers",
    "layer_norm",
    "layer_norm_fn",
    "layer_norm_linear_fn",
    "rms_norm_fn",
]
