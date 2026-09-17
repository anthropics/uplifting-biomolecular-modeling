"""
HCM -- Hyena Cascade Medium.

Fused Triton epilogues for the FFT-convolution path of
HyenaInferenceEngine.parallel_fir (the fir_length >= 128 branch). At a
128-tap filter Triton cannot out-write cuFFT for the transforms themselves,
so the win is launch-count: the elementwise glue around the three cuFFT
calls is fused into Triton kernels.

It provides hcm_fft_conv -- a drop-in for fftconv_func -- built on two fused
stage kernels: _hcm_complex_mul (stages 1 + 3, the scaled spectral product
u_f * k_f) and _hcm_bias_residual (stage 5, the skip-residual add y + u*bias).
"""

from collections.abc import Callable

import torch
import triton
import triton.language as tl

from vortex.ops.triton_common import BDL_TILE_CONFIGS, bdl_grid_3d

# 1-D BLOCK sweep for the complex multiply -- the only HC kernel that flattens
# (D, F) to a single axis, so it can't share BDL_TILE_CONFIGS.
_COMPLEX_MUL_CONFIGS: list[triton.Config] = [
    triton.Config({"BLOCK": 256}, num_warps=2),
    triton.Config({"BLOCK": 512}, num_warps=4),
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 2048}, num_warps=8),
]


@triton.autotune(configs=_COMPLEX_MUL_CONFIGS, key=["DF"])
@triton.jit
def _hcm_complex_mul_kernel(
    u_ptr,
    k_ptr,
    y_ptr,
    DF,
    inv_fft_size,
    stride_batch,
    BLOCK: tl.constexpr,
):
    """
    Broadcast complex multiply: y[b] = u_f[b] * k_f[0] * inv_fft_size.

    One program covers a BLOCK-element slice of the flattened (D, F) plane
    of one batch element. Complex values are stored interleaved (real, imag)
    -- view_as_real's layout -- so element n's real part is at offset 2n and
    its imag part at 2n + 1. The filter k_f carries no batch stride: every
    batch element multiplies against the same spectrum.

    The flat (D, F) tile is grid axis 0: cdiv(DF, BLOCK) overruns the 65535
    cap on axes 1 and 2 at long context, so the small batch sits on axis 1.
    """
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < DF

    u_base = u_ptr + pid_b * stride_batch
    y_base = y_ptr + pid_b * stride_batch

    u_re = tl.load(u_base + 2 * offs, mask=mask, other=0.0)
    u_im = tl.load(u_base + 2 * offs + 1, mask=mask, other=0.0)
    k_re = tl.load(k_ptr + 2 * offs, mask=mask, other=0.0)
    k_im = tl.load(k_ptr + 2 * offs + 1, mask=mask, other=0.0)

    y_re = (u_re * k_re - u_im * k_im) * inv_fft_size
    y_im = (u_re * k_im + u_im * k_re) * inv_fft_size

    tl.store(y_base + 2 * offs, y_re, mask=mask)
    tl.store(y_base + 2 * offs + 1, y_im, mask=mask)


def _hcm_complex_mul(
    u_f: torch.Tensor, k_f: torch.Tensor, fft_size: int
) -> torch.Tensor:
    """
    Fused broadcast complex multiply with the 1/fft_size filter scale.

    Computes u_f * k_f / fft_size -- stage 3 of fftconv_func, with stage 1's
    filter normalisation folded in. u_f is the activation spectrum; k_f is
    the *unscaled* filter spectrum, already shaped for broadcast over the
    batch by adjust_filter_shape_for_broadcast.

    Args:
        u_f (torch.Tensor): Activation spectrum, complex, shape (B, D, F).
        k_f (torch.Tensor): Filter spectrum, complex, shape (1, D, F), shared
                            across the batch and not yet scaled by 1/fft_size.
        fft_size (int): The FFT length n = 2 * seqlen; its reciprocal folds
                        in as the filter normalisation.

    Returns:
        torch.Tensor: The scaled product u_f * k_f / fft_size, complex, shape
                      (B, D, F), u_f's dtype.

    Raises:
        ValueError: If the tensors are not 3-D complex, or k_f is not
                    broadcastable over the batch of u_f.
    """
    if u_f.dim() != 3 or k_f.dim() != 3:
        raise ValueError(
            f"expected 3-D u_f and k_f, got {tuple(u_f.shape)} and {tuple(k_f.shape)}"
        )
    if not u_f.is_complex() or not k_f.is_complex():
        raise ValueError("u_f and k_f must be complex tensors")

    B, D, F = u_f.shape
    if tuple(k_f.shape) != (1, D, F):
        raise ValueError(
            f"k_f {tuple(k_f.shape)} is not broadcastable over u_f {tuple(u_f.shape)}"
        )

    u_f = u_f.contiguous()
    k_f = k_f.contiguous()
    y_f: torch.Tensor = torch.empty_like(u_f)

    # Triton has no complex dtype: operate on the (..., 2) real/imag view.
    u_r = torch.view_as_real(u_f)
    k_r = torch.view_as_real(k_f)
    y_r = torch.view_as_real(y_f)

    DF = D * F
    grid: Callable[[dict], tuple[int, int]] = lambda meta: (
        triton.cdiv(DF, meta["BLOCK"]),
        B,
    )
    _hcm_complex_mul_kernel[grid](
        u_r,
        k_r,
        y_r,
        DF,
        1.0 / fft_size,
        u_r.stride(0),
    )
    return y_f


@triton.autotune(configs=BDL_TILE_CONFIGS, key=["D", "L"])
@triton.jit
def _hcm_bias_residual_kernel(
    y_ptr,
    u_ptr,
    bias_ptr,
    out_ptr,
    D,
    L,
    stride_b,
    stride_d,
    stride_l,
    BLOCK_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """
    Skip-residual add: out[b, d, l] = y[b, d, l] + u[b, d, l] * bias[d].

    One program covers a (BLOCK_D, BLOCK_L) tile of one batch element. y, u
    and out share a contiguous (B, D, L) layout; bias is per-channel, shape
    (D,), broadcast over batch and length. The fp32 accumulator is cast to
    out's dtype on the store -- fftconv_func's stage-6 .to(u.dtype) cast.
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_l = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_d = offs_d < D
    mask_l = offs_l < L
    mask = mask_d[:, None] & mask_l[None, :]

    offs = pid_b * stride_b + offs_d[:, None] * stride_d + offs_l[None, :] * stride_l
    y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    acc = y + u * bias[:, None]
    tl.store(out_ptr + offs, acc, mask=mask)


def _hcm_bias_residual(
    y: torch.Tensor, u: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """
    Fused skip-residual add -- stage 5 of fftconv_func.

    Computes y + u * bias[:, None] and writes it at u's dtype, fusing
    fftconv_func's broadcast multiply and residual add (and its stage-6
    dtype cast) into a single Triton launch.

    Args:
        y (torch.Tensor): The irfft output, shape (B, D, L).
        u (torch.Tensor): The activations, shape (B, D, L); its dtype is the
                          output dtype -- fftconv_func's stage-6 cast target.
        bias (torch.Tensor): Per-channel skip gain, shape (D,), broadcast
                             over batch and length.

    Returns:
        torch.Tensor: y + u * bias[:, None], shape (B, D, L), u's dtype.

    Raises:
        ValueError: If y and u are not matching 3-D tensors, or bias is not
                    1-D of length D.
    """
    if y.dim() != 3 or u.dim() != 3:
        raise ValueError(
            f"expected 3-D y and u, got {tuple(y.shape)} and {tuple(u.shape)}"
        )
    if y.shape != u.shape:
        raise ValueError(f"y {tuple(y.shape)} and u {tuple(u.shape)} must match")

    B, D, L = u.shape
    if bias.dim() != 1 or bias.shape[0] != D:
        raise ValueError(f"bias {tuple(bias.shape)} must be 1-D of length D={D}")

    y = y.contiguous()
    u = u.contiguous()
    bias = bias.contiguous()
    out: torch.Tensor = torch.empty_like(u)
    _hcm_bias_residual_kernel[bdl_grid_3d(B, D, L)](
        y,
        u,
        bias,
        out,
        D,
        L,
        u.stride(0),
        u.stride(1),
        u.stride(2),
    )
    return out


def hcm_fft_conv(
    u: torch.Tensor,
    k: torch.Tensor,
    D: torch.Tensor,
    dropout_mask: torch.Tensor | None,
    gelu: bool = False,
    k_rev: torch.Tensor | None = None,
    bidirectional: bool = False,
    print_activations: bool = False,
    layer_idx: int | None = None,
    **kwargs,
) -> torch.Tensor:
    """
    Fused HCM FFT-convolution -- drop-in for fftconv_func.

    cuFFT keeps the three transforms; _hcm_complex_mul does stage 3 (scaled
    spectral product), _hcm_bias_residual does stage 5 (skip-residual add).
    Trailing kwargs exist only for signature parity with fftconv_func so the
    engine dispatch is a one-line swap. gelu and dropout_mask are part of that
    parity surface but unsupported -- the kernel has no activation or dropout
    stage, so a set value raises instead of being silently dropped.

    Args:
        u (torch.Tensor): Input activations, shape (B, D, L).
        k (torch.Tensor): Filter, shape (D, 1, K).
        D (torch.Tensor): Per-channel skip-connection bias, shape (D,).
        dropout_mask (torch.Tensor | None): Unsupported; must be None (parity only).
        gelu (bool): Unsupported; must be False; the kernel has no activation stage.

    Returns:
        torch.Tensor: y + u * D[:, None], shape (B, D, L), u's dtype.

    Raises:
        NotImplementedError: If bidirectional is True, k_rev is set, gelu is
                             True, or dropout_mask is not None -- the HCM
                             kernel implements none of these.
    """
    if bidirectional or k_rev is not None:
        raise NotImplementedError(
            "hcm_fft_conv handles only the causal, non-reverse path"
        )
    if gelu or dropout_mask is not None:
        raise NotImplementedError(
            "hcm_fft_conv implements only the gelu=False, dropout_mask=None "
            "path used by evo2; the kernel has no activation or dropout stage."
        )

    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    # rfft(k) reshaped to (1, D, F) for the batch broadcast -- inlined to avoid
    # the adjust_filter_shape_for_broadcast import cycle. squeeze(1) drops only
    # the channel-group axis; .squeeze() with no arg would also collapse a D=1
    # case and break the (1, D, F) broadcast contract.
    k_f = torch.fft.rfft(k, n=fft_size).squeeze(1).unsqueeze(0)
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    prod = _hcm_complex_mul(u_f, k_f, fft_size)
    y = torch.fft.irfft(prod, n=fft_size, norm="forward")[..., :seqlen]
    return _hcm_bias_residual(y, u, D)
