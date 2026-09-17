import numbers
from typing import List, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Size

import triton
import triton.language as tl

from xfold.fastnn import config as fastnn_config

_shape_t = Union[int, List[int], Size]


# modified from triton/python/tutorials/05-layer-norm.py
@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
    USE_WEIGHTS: tl.constexpr = True,
    USE_BIAS: tl.constexpr = True,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0).to(tl.int64)   # int64 BEFORE `row * N`: N*N rows of 267 (the diffusion pair conditioning) pass 2**31 - 1 past 2836 tokens, of 128 past 4096 (upstream issue XFOLD-003)
    Y += row * N
    X += row * N
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        if USE_WEIGHTS is True:
            w = tl.load(W + cols, mask=mask)
            if USE_BIAS is True:
                b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        if USE_WEIGHTS is True:
            y = x_hat * w
            if USE_BIAS is True:
                y += b
        else:
            y = x_hat
        # Write output
        tl.store(Y + cols, y, mask=mask)


class LayerNormTritonFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        x = x.contiguous()
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        N = x.shape[-1]
        M = x.numel() // N
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError(
                "This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel
        use_weight = weight is not None
        use_bias = bias is not None
        _layer_norm_fwd_fused[(M, )](  #
            x, y, weight, bias,    #
            N, eps,   #
            BLOCK_SIZE=BLOCK_SIZE, USE_WEIGHTS=use_weight, USE_BIAS=use_bias, #
            num_warps=num_warps, num_ctas=1)
        return y


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(
            normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)
            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if fastnn_config.layer_norm_implementation == "torch":
            return F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif fastnn_config.layer_norm_implementation == "triton":
            fast_out = LayerNormTritonFunc.apply(
                input, self.normalized_shape, self.weight, self.bias, self.eps
            )
            return fast_out
        else:
            raise ValueError(
                f"fastnn_config must be 'torch' or 'triton', got {fastnn_config}"
            )

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )
