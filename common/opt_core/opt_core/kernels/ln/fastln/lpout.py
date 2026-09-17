"""fastln, low-precision OUTPUT form — the carried ``fastln.fast_layer_norm_triton`` row LayerNorm (one program per BLOCK_M rows, the
whole row in registers, two-pass statistics) with the arithmetic pinned to fp32 whatever the storage dtypes are, and the result stored in
the OUTPUT dtype the caller names (bf16 by default for a bf16 input): the contract of a trunk's compiled ``fast_layernorm`` extension under
bf16 autocast — bf16 activations in, fp32 gamma / beta, fp32 mean / variance / normalisation, bf16 out — which the carried module does not
offer (it stores the input's dtype after statistics in the input's dtype, and the provider only admits it for fp32 rows).

    y = layer_norm_lp(x, (C,), weight, bias, eps, out_dtype=torch.bfloat16)      # x bf16 | fp16 | fp32 rows, weight / bias fp32 | bf16 | None

Numerics class: tolerance (fp32 statistics as ATen's autocast path; sum order = Triton's tree reduction, not ATen's Welford; one final
round-to-nearest-even into the output dtype).  Row offsets are int64.  House-written (this tree); the tiling rule is the carried module's.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _ln_rows_lp_kernel(X, W, B, Y, M, eps,
                       N: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
                       HAS_W: tl.constexpr, HAS_B: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    rmask = rows < M
    cmask = cols < N
    mask = rmask[:, None] & cmask[None, :]
    offs = rows.to(tl.int64)[:, None] * N + cols[None, :]
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, 1) / N
    d = tl.where(mask, x - mean[:, None], 0.0)
    var = tl.sum(d * d, 1) / N
    rstd = tl.div_rn(1.0, tl.sqrt_rn(var + eps))
    y = d * rstd[:, None]
    if HAS_W:
        w = tl.load(W + cols, mask=cmask, other=0.0).to(tl.float32)
        y = y * w[None, :]
    if HAS_B:
        b = tl.load(B + cols, mask=cmask, other=0.0).to(tl.float32)
        y = y + b[None, :]
    tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)


def _cfg(N):
    BLOCK_N = max(16, triton.next_power_of_2(N))
    if BLOCK_N <= 64:
        BLOCK_M = 4096 // BLOCK_N; nw = 8
    elif BLOCK_N <= 1024:
        BLOCK_M = max(1, 4096 // BLOCK_N); nw = 4
    elif BLOCK_N <= 4096:
        BLOCK_M = 1; nw = 8
    else:
        BLOCK_M = 1; nw = 16
    return BLOCK_N, BLOCK_M, nw


OUT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def layer_norm_lp(x, normalized_shape, weight=None, bias=None, eps=1e-5, out_dtype=None):
    """F.layer_norm(x.float(), normalized_shape, weight.float(), bias.float(), eps).to(out_dtype) semantics over the last dim(s), in one
    pass: fp32 statistics and affine, ``out_dtype`` store (default: x's dtype).  x contiguous rows (copied contiguous otherwise)."""
    N = 1
    for s in normalized_shape:
        N *= int(s)
    out_dtype = x.dtype if out_dtype is None else out_dtype
    if out_dtype not in OUT_DTYPES:
        raise ValueError("layer_norm_lp: out_dtype %s (served: bf16 | fp16 | fp32)" % out_dtype)
    xc = x.contiguous()
    M = xc.numel() // N
    y = torch.empty(xc.shape, dtype=out_dtype, device=xc.device)
    w = weight.contiguous() if weight is not None else xc
    b = bias.contiguous() if bias is not None else xc
    BLOCK_N, BLOCK_M, nw = _cfg(N)
    grid = (triton.cdiv(M, BLOCK_M),)
    _ln_rows_lp_kernel[grid](xc, w, b, y, M, float(eps), N=N, BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M,
                             HAS_W=weight is not None, HAS_B=bias is not None, num_warps=nw)
    return y.view(x.shape)
