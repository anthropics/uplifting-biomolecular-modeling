"""pf_triton.py — fused pair-bias PRODUCER for AttentionPairBias in the Pairformer (trunk 48 blocks x N_cycle, confidence head 4 blocks):
    bias[h, i, j] = Linear_{c_z -> H}( LayerNorm_{c_z}(z[i, j, :]) )[h]          (stock: linear_nobias_z(layernorm_z(z)) then permute [2,0,1])
in ONE pass over z (z read once: fp32 or bf16 [N, N, c_z]; LN statistics fp32; the normalized row cast to bf16 and multiplied by the bf16 weight
with fp32 accumulation = the roundings of stock's autocast Linear; output bf16 [H, N, N] stored with a row pitch rounded up to 8 elements so the
attention core's TMA bias path applies at any N — the [.., :N] view is returned).
Numerics class: TOLERANCE (same roundings as stock; the c_z-term dot is accumulated in a different order than cuBLAS).
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["NN", "szr", "eps"])
def _pf_bias_kernel(Z, LNW, LNB, W, OUT, NN, szr, eps,
                    C: tl.constexpr, H: tl.constexpr, R: tl.constexpr, HAS_LNB: tl.constexpr, OUT_PITCH_NN: tl.constexpr, OUT_F32: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * R + tl.arange(0, R)                       # flattened (i, j) index in [0, N*N)
    rmask = rows < NN
    offs_c = tl.arange(0, C)
    z = tl.load(Z + rows.to(tl.int64)[:, None] * szr + offs_c[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)   # [R, C]
    mean = tl.sum(z, 1) / C
    zc = z - mean[:, None]
    var = tl.sum(zc * zc, 1) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(LNW + offs_c).to(tl.float32)
    xh = zc * rstd[:, None] * w[None, :]
    if HAS_LNB:
        xh = xh + tl.load(LNB + offs_c).to(tl.float32)[None, :]
    xb = xh.to(tl.bfloat16)                                # stock: autocast casts the LN output to bf16 for the Linear
    wt = tl.load(W + tl.arange(0, H)[None, :] * C + offs_c[:, None]).to(tl.bfloat16)      # W [H, C] row-major -> [C, H] operand
    acc = tl.dot(xb, wt)                                   # [R, H] fp32 accumulate
    if OUT_F32:
        o = acc.to(tl.bfloat16).to(tl.float32)              # stock: bf16 Linear output, then _attention's .float() -> same values, fp32 container
    else:
        o = acc.to(tl.bfloat16)
    # OUT [H, pitch-padded N*N plane]: element (h, row) at OUT + h * OUT_PITCH_NN + out_index(row)
    tl.store(OUT + tl.arange(0, H)[None, :].to(tl.int64) * OUT_PITCH_NN + rows.to(tl.int64)[:, None], o, mask=rmask[:, None])


def pf_bias(z, ln_weight, ln_bias, lin_weight, eps: float = 1e-5, *, R: int | None = None, num_warps: int = 4, out=None, out_dtype=torch.bfloat16):
    """z [N, N, C] (fp32|bf16, last dim contiguous, uniform row stride) -> bias view [H, N, N] bf16 of a [H, N, P8] buffer (P8 = ceil8(N)).
    The kernel writes the dense [H, N*N] product plane when N % 8 == 0 (pitch == N); otherwise it writes a dense scratch plane and one strided
    copy lays it into the pitch-8 buffer (N % 8 != 0 costs one extra 2-byte pass over the bias)."""
    N, N2, C = z.shape
    R = R or max(16, 16384 // C)                     # rows of (i, j) per program: a [R, C] fp32 tile in registers (128 @ c_z=128, 64 @ c_z=256)
    assert N == N2 and z.stride(2) == 1 and z.stride(0) == N * z.stride(1), f"pf_bias: z layout {tuple(z.shape)} {z.stride()}"
    H = lin_weight.shape[0]
    assert lin_weight.shape == (H, C) and lin_weight.stride() == (C, 1)
    NN = N * N
    P8 = (N + 7) // 8 * 8
    if out is None:
        out = torch.empty((H, N, P8), device=z.device, dtype=out_dtype)
    grid = (triton.cdiv(NN, R),)
    if P8 == N:
        _pf_bias_kernel[grid](z, ln_weight, ln_bias if ln_bias is not None else ln_weight, lin_weight, out, NN, z.stride(1), eps,
                              C=C, H=H, R=R, HAS_LNB=ln_bias is not None, OUT_PITCH_NN=NN, OUT_F32=out_dtype == torch.float32, num_warps=num_warps)
        return out
    tmp = torch.empty((H, N, N), device=z.device, dtype=out_dtype)
    _pf_bias_kernel[grid](z, ln_weight, ln_bias if ln_bias is not None else ln_weight, lin_weight, tmp, NN, z.stride(1), eps,
                          C=C, H=H, R=R, HAS_LNB=ln_bias is not None, OUT_PITCH_NN=NN, OUT_F32=out_dtype == torch.float32, num_warps=num_warps)
    out[:, :, :N].copy_(tmp)
    return out[:, :, :N]
