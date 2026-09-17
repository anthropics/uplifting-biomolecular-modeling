"""
HCS -- Hyena Cascade Short.

Triton depthwise causal 1D conv for the short-filter (fir_length < 128)
gated branch of HyenaInferenceEngine.parallel_fir. A depthwise filter of
fir_length taps (7 in evo2_7b) applied per channel; the only time-mixing
op in an HCS layer. Exposes the @triton.jit kernel and the hcs_conv
adapter wired behind use_hcs_kernel.
"""

import torch
import triton
import triton.language as tl

from vortex.ops.triton_common import BDL_TILE_CONFIGS, bdl_grid_3d


@triton.autotune(configs=BDL_TILE_CONFIGS, key=["D", "L", "FIR_LEN"])
@triton.jit
def _hcs_depthwise_conv_kernel(
    u_ptr,
    w_ptr,
    z_ptr,
    D,
    L,
    stride_ub,
    stride_ud,
    stride_ul,
    stride_wd,
    stride_wk,
    FIR_LEN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """
    Depthwise causal conv: z[b, d, t] = sum_k w[d, k] * u[b, d, t - FIR_LEN + 1 + k].

    One program covers a (BLOCK_D, BLOCK_L) tile of one batch element. The
    FIR_LEN tap loop is unrolled at compile time. Input positions before 0
    are masked to zero, giving a causal (left-padded) convolution.
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_l = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_d = offs_d < D
    mask_l = offs_l < L
    tile_mask = mask_d[:, None] & mask_l[None, :]

    u_base = u_ptr + pid_b * stride_ub + offs_d[:, None] * stride_ud
    acc = tl.zeros((BLOCK_D, BLOCK_L), dtype=tl.float32)

    # hcs_conv forces fp32 before launch, so loaded tiles are already fp32 --
    # the .to(tl.float32) calls are no-ops at runtime.
    for k in tl.static_range(FIR_LEN):
        w_k = tl.load(
            w_ptr + offs_d * stride_wd + k * stride_wk, mask=mask_d, other=0.0
        )
        pos = offs_l - (FIR_LEN - 1) + k
        mask_pos = tile_mask & (pos[None, :] >= 0) & (pos[None, :] < L)
        u_tile = tl.load(u_base + pos[None, :] * stride_ul, mask=mask_pos, other=0.0)
        acc += w_k[:, None] * u_tile

    z_ptrs = (
        z_ptr
        + pid_b * stride_ub
        + offs_d[:, None] * stride_ud
        + offs_l[None, :] * stride_ul
    )
    tl.store(z_ptrs, acc, mask=tile_mask)


def hcs_depthwise_conv(u: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Depthwise causal 1D convolution, the HCS short-filter time-mixing op.

    Equivalent to F.conv1d(u, weight, padding=fir_length - 1, groups=D)
    trimmed to length L, but in a single fused Triton launch.

    Args:
        u (torch.Tensor): Input activations, shape (B, D, L), contiguous.
        weight (torch.Tensor): Depthwise filter, shape (D, 1, fir_length),
                               contiguous. Every channel has its own filter.

    Returns:
        torch.Tensor: Convolved output, shape (B, D, L), same dtype as u.
    """
    u = u.contiguous()
    weight = weight.contiguous()
    if u.dim() != 3 or weight.dim() != 3:
        raise ValueError(f"expected 3-D u and weight, got {u.shape} and {weight.shape}")

    B, D, L = u.shape
    Dw, in_per_group, fir_length = weight.shape
    if Dw != D or in_per_group != 1:
        raise ValueError(f"weight {tuple(weight.shape)} is not depthwise for D={D}")

    z: torch.Tensor = torch.empty_like(u)
    _hcs_depthwise_conv_kernel[bdl_grid_3d(B, D, L)](
        u,
        weight,
        z,
        D,
        L,
        u.stride(0),
        u.stride(1),
        u.stride(2),
        weight.stride(0),
        weight.stride(2),
        FIR_LEN=fir_length,
    )
    return z


def hcs_conv(
    x1: torch.Tensor,
    x2: torch.Tensor,
    v: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    gated_bias: bool = False,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fully-gated HCS short conv: z = x2 * (conv(x1 * v, weight) + bias).

    Drop-in for the gated fir_length < 128 branch of
    HyenaInferenceEngine.parallel_fir. Conv runs in fp32 for parity with
    F.conv1d, then casts back to x1.dtype.

    Args:
        x1 (torch.Tensor): Pre-gate "key" stream, shape (B, D, L).
        x2 (torch.Tensor): Post-gate stream, shape (B, D, L).
        v (torch.Tensor): "Value" stream, shape (B, D, L).
        weight (torch.Tensor): Depthwise filter, shape (D, 1, fir_length).
        bias (torch.Tensor | None): Per-channel skip-gain, shape (D,).
        gated_bias (bool): If True, bias is applied multiplicatively
                           (bias * x1 * v); HCS uses additive (False).
        padding_mask (torch.Tensor | None): If set, zeros masked positions
                                            after the conv, shape (B, L).

    Returns:
        torch.Tensor: Gated HCS output, shape (B, D, L), x1's dtype.
    """
    u: torch.Tensor = x1 * v
    z: torch.Tensor = hcs_depthwise_conv(u=u.float(), weight=weight.float())
    z = z.to(u.dtype)

    if bias is not None:
        if gated_bias:
            z = z + bias[None, :, None] * u
        else:
            z = z + bias[None, :, None]

    if isinstance(padding_mask, torch.Tensor):
        z = z * padding_mask[:, None]

    return x2 * z
