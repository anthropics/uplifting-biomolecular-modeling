"""E64 (featurizer half): the 3-tap featurizer FIR over a projection output whose channels are contiguous along L (the transposed GEMM output
of fp8emit): E7's per-element expression (p_k = w_k * x[t-2+k] where the tap is in range, else +0.0; acc = ((0.0 + p0) + p1) + p2;
bf16(acc)) on (BLOCK_D, BLOCK_L) tiles with L fastest, writing the same (B, 3D, L) contiguous bf16 featurizer output E7 writes."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from evo2_opt.kit import i32 as _I32

LEVER = "E64_channels_first_projection"
GEOMETRY = (8, 512, 4)           # BLOCK_D, BLOCK_L, num_warps: launch geometry only


@triton.jit
def _fir3_acc(row_ptr, offs_l, sl, L, w0, w1, w2, mrow):
    t0 = offs_l[None, :] - 2
    t1 = offs_l[None, :] - 1
    t2 = offs_l[None, :]
    ml = t2 < L
    m0 = (t0 >= 0) & ml & mrow
    m1 = (t1 >= 0) & ml & mrow
    m2 = ml & mrow
    x0 = tl.load(row_ptr + t0 * sl, mask=m0, other=0.0).to(tl.float32)
    x1 = tl.load(row_ptr + t1 * sl, mask=m1, other=0.0).to(tl.float32)
    x2 = tl.load(row_ptr + t2 * sl, mask=m2, other=0.0).to(tl.float32)
    p0 = tl.where(m0, w0[:, None] * x0, 0.0)
    p1 = tl.where(m1, w1[:, None] * x1, 0.0)
    p2 = tl.where(m2, w2[:, None] * x2, 0.0)
    return ((0.0 + p0) + p1) + p2                                       # ascending tap order from +0.0, as E7 / ATen's depthwise conv


@triton.jit
def _fir3_rows_kernel(x_ptr, w_ptr, o_ptr, D3, L, sxb, sxd, sxl, sob, sod, sol, BLOCK_D: tl.constexpr, BLOCK_L: tl.constexpr):
    pb = tl.program_id(0)
    pd = tl.program_id(1)
    pl = tl.program_id(2)
    od = pd * BLOCK_D + tl.arange(0, BLOCK_D)
    ol = pl * BLOCK_L + tl.arange(0, BLOCK_L)
    md = od < D3
    w0 = tl.load(w_ptr + od * 3 + 0, mask=md, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + od * 3 + 1, mask=md, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + od * 3 + 2, mask=md, other=0.0).to(tl.float32)
    rows = x_ptr + pb * sxb + od[:, None] * sxd
    acc = _fir3_acc(rows, ol, sxl, L, w0, w1, w2, md[:, None])
    tl.store(o_ptr + pb * sob + od[:, None] * sod + ol[None, :] * sol, acc.to(tl.bfloat16), mask=md[:, None] & (ol[None, :] < L))


def takes(u, weight) -> bool:
    """The featurizer input is the channels-first projection output: (B, L, 3D) bf16 whose L stride is 1, a (3D, 1, 3) filter."""
    return u.dim() == 3 and u.dtype == torch.bfloat16 and u.stride(1) == 1 and weight.shape[-1] == 3 and int(weight.shape[0]) == int(u.shape[2])


def fir3_rows(u_bld, weight, geometry=GEOMETRY):
    """u_bld: (B, L, 3D) view with L contiguous; weight: (3D, 1, 3) bf16 -> (B, 3D, L) contiguous bf16 = E7's featurizer output."""
    x = u_bld.permute(0, 2, 1)                                          # (B, 3D, L), no copy
    Bn, D3, Ln = (int(s) for s in x.shape)
    _I32.check(LEVER, f"the channels-first projection output (B={Bn}, 3D={D3}, L={Ln})", (_I32.strided_extent(x.shape, x.stride()), Bn * D3 * Ln),
               f"batch x padded length <= {_I32.EXTENT // max(1, D3):,} at width {D3}")
    w2 = weight.reshape(D3, 3).contiguous()
    out = torch.empty((Bn, D3, Ln), dtype=torch.bfloat16, device=x.device)
    bd, bl, nw = geometry
    _fir3_rows_kernel[(Bn, triton.cdiv(D3, bd), triton.cdiv(Ln, bl))](x, w2, out, D3, Ln, x.stride(0), x.stride(1), x.stride(2),
                                                                         out.stride(0), out.stride(1), out.stride(2), BLOCK_D=bd, BLOCK_L=bl, num_warps=nw)
    return out
