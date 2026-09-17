# SPDX-License-Identifier: Apache-2.0
# BlockFuse research track (FlashPairformer v0.6/v0.7), 2026-08-25.
"""blockfuse.biasln — single-z-pass triangle-BIAS producer: LayerNorm in registers + the 8-channel bias projection, nothing else.

    bias_ln(module, z, ending) -> bias [H, NI, NJ] fp32  ( == permute_final_dims(module.linear(fast_layernorm(x)), (2,0,1)).float(), x = z | z^T )

This is the F1 prologue of the MK-PF track (fpf_mkpf.kernels._f1_prologue_ln_kernel) with the q/k/v/g projection loops removed: the z tile read,
the in-register LayerNorm (`_ln_rows`, LN_ARITH=2 = the integrator's fast_layernorm Welford emulation from fpf_triatt_pro, EXACT on cc9.0|triton3.7 per
MKPF cell sm90_t37) and the bias MMA (one full-K chain, bf16 rounding then .float()) are imported / copied verbatim, so the bias bits are the tested
ones GIVEN the same LN bits (checked by torch.equal against F1's own bias output and against the stock LN+linear in bench.py — never assumed).
Purpose: pass 1 of the row-chunked BlockFuse statement needs the FULL [H,N,N] bias before any attention chunk can run; producing it this way reads z
once (N^2*C*2 B) and writes N^2*H*4 B, instead of stock LN (read z, write x_ln [+ the transposed .contiguous() copy on the ending node]) + linear (read x_ln).
Credits: kernel mathematics = MK-PF track (F1) / FPF Fusion track (prologue v3) / integrator (welford emulation); this file only deletes code.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl

from fpf_mkpf.kernels import _ln_rows, HAS_WELFORD, _LN_ARITH


@triton.jit
def _bias_ln_kernel(Z, LNW, LNB, WB, BIAS,
                    NI, NJ, s_zi, s_zj, eps,
                    C: tl.constexpr, H: tl.constexpr, HB: tl.constexpr,
                    BI: tl.constexpr, BJ: tl.constexpr,
                    LN_ARITH: tl.constexpr, FMA_M2: tl.constexpr, FMA_MERGE: tl.constexpr, FMA_AFF: tl.constexpr):
    BM: tl.constexpr = BI * BJ
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    r = tl.arange(0, BM)
    ii = pid_i * BI + r // BJ
    jj = pid_j * BJ + r % BJ
    rmask = (ii < NI) & (jj < NJ)
    rc = tl.arange(0, C)
    ii64 = ii.to(tl.int64)
    jj64 = jj.to(tl.int64)
    zoff = ii64 * s_zi + jj64 * s_zj
    zt = tl.load(Z + zoff[:, None] + rc[None, :], mask=rmask[:, None], other=0.0)
    wln = tl.load(LNW + rc).to(tl.float32)
    bln = tl.load(LNB + rc).to(tl.float32)
    x = _ln_rows(zt, wln, bln, rmask, eps, C, BM, LN_ARITH, FMA_M2, FMA_MERGE, FMA_AFF)          # bf16 [BM, C], identical helper to F1
    rb = tl.arange(0, HB)
    wbT = tl.load(WB + rb[None, :].to(tl.int64) * C + rc[:, None])                                   # [C, HB]
    accb = tl.dot(x, wbT)                                                                            # one full-K fp32 chain (== F1 / prologue v3 / cuBLAS bits)
    b32 = accb.to(tl.bfloat16).to(tl.float32)                                                        # stock: bf16 GEMM output, then .float()
    NJ64 = NJ.to(tl.int64)
    boff = (rb[None, :].to(tl.int64) * NI + ii64[:, None]) * NJ64 + jj64[:, None]
    tl.store(BIAS + boff, b32, mask=rmask[:, None] & (rb[None, :] < H))


def bias_ln(module, z, cch, ending: bool, ln_arith="welford", fma_flags=(True, True, True), cfg=None, out=None):
    """z [N0,N1,C] bf16 (native block layout, stride(-1)==1); ending selects the x-frame by address math.  cch = fpf_triatt_pro.prologue.get_cache(module, dev).
    Returns bias [H, NI, NJ] fp32 contiguous."""
    assert z.dim() == 3 and z.dtype == torch.bfloat16 and z.stride(-1) == 1
    C, H, HB = cch["C"], cch["H"], cch["HB"]
    if ending:
        NI, NJ, s_zi, s_zj = int(z.shape[1]), int(z.shape[0]), z.stride(1), z.stride(0)
    else:
        NI, NJ, s_zi, s_zj = int(z.shape[0]), int(z.shape[1]), z.stride(0), z.stride(1)
    la = _LN_ARITH[ln_arith]
    if la == 2:
        assert HAS_WELFORD and C == 256
    cfg = cfg or dict(BI=8, BJ=16, num_warps=8, num_stages=2)
    BI, BJ = int(cfg["BI"]), int(cfg["BJ"])
    bias = torch.empty((H, NI, NJ), dtype=torch.float32, device=z.device) if out is None else out
    assert tuple(bias.shape) == (H, NI, NJ) and bias.is_contiguous() and bias.dtype == torch.float32
    grid = (triton.cdiv(NI, BI), triton.cdiv(NJ, BJ))
    _bias_ln_kernel[grid](z, cch["lnw"], cch["lnb"], cch["wb"], bias, NI, NJ, s_zi, s_zj, float(cch["eps"]),
                          C=C, H=H, HB=HB, BI=BI, BJ=BJ, LN_ARITH=la,
                          FMA_M2=bool(fma_flags[0]), FMA_MERGE=bool(fma_flags[1]), FMA_AFF=bool(fma_flags[2]),
                          num_warps=int(cfg["num_warps"]), num_stages=int(cfg.get("num_stages", 2)))
    return bias
