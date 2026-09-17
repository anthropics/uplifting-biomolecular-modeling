# SPDX-License-Identifier: Apache-2.0
# BlockFuse XL-prologue add-on for FlashPairformer v0 (XL-patched v3 candidate bundle), 2026-08-25.
"""blockfuse_xl — drop-in replacement of the FPF XL 'lean prologue' tri-attention statement (PTX_FPF_CHUNK_MODE=prologue, N_token > PTX_FPF_CHUNK_TOK).

What runs instead of the bundle's two-pass v3 loop (ptx_trunk2_levers._triatt_block_pro_epi, lean-prologue branch), per TriangleAttention statement:
    pass 1  bias  = blockfuse.biasln.bias_ln(z)                      ONE launch: LayerNorm in registers (welford emulation) + 8-channel bias projection; z read once,
                                                                     writes only the [H,N,N] fp32 triangle bias (no x_ln, no q/k/v/g, no torch.cat)
    pass 2  for each stock row chunk (chunk_size rows): fpf_mkpf F1 prologue_ln on the z rows/columns of the chunk (strided read, LN in registers)
            -> the block core's attention on the chunk rows with the full bias: the stock cuequivariance triangle_attention under ARM E (== stock
               chunk_layer call; exact) or, kit-local change 0.1.1 (Big-path P1, 2026-09-11), the arm's K2B under ARM T (ptx_trunk2_levers._blk_core_att;
               K2B batches pair rows in its grid, so the per-chunk calls are bitwise == fast's one full-N K2B call) -> fpf_triatt_epi epilogue, residual in place.
Everything is an unchanged, separately tested kernel (F1 = EXACT cell sm90_t37; epilogue/cuEq = ARM E) except bias_ln, which is F1 with the q/k/v/g loops
deleted (torch.equal-checked against F1's and prologue-v3's bias in selftest.py).  Peak transient per statement: bias (4*H*N^2 B) + one chunk's q/k/v/g/o
instead of the bundle's full-N x_ln (stock-LN mode) and the XL loop's R x full-prologue pass-1 launches.
Engages ONLY when: the lean-prologue branch would engage (chunk_size given and ptx_trunk2_levers._xl_lean_prologue(N)), z is 3-D bf16 with unit channel stride,
module dims are c=256 / H*D=256, and fpf_mkpf resolves an EXACT welford cell for this (cc, triton) — otherwise the bundle's code path runs untouched (counted).
Env: PTX_BLOCKFUSE_XL=1 (read by the add-on sitecustomize) ; PTX_BLOCKFUSE_XL_ALLOW_TIER2=1 would accept a non-EXACT F1 cell (then the statement is TIER-2; default refuses).
Counters: blockfuse_xl.stats(); one '[blockfuse_xl] ...' stderr line at install and at exit.
Credits: F1 prologue_ln = MK-PF track; prologue v3 / epilogue = FPF Fusion track; welford LN emulation = integrator; XL lean path + policy = XL track; cuEq = NVIDIA cuEquivariance (as installed).
"""
from __future__ import annotations
import os, sys, math, atexit
import torch

__version__ = "0.1.1"
_STATE = {"installed": False, "why": "", "cell": None, "calls": {"bf_stmts": 0, "bf_chunks": 0, "k2b_chunks": 0, "cueq_chunks": 0, "passthru": 0, "refused_ineligible": 0, "max_ntok": 0}, "orig": None}


def stats():
    return dict(_STATE["calls"], installed=_STATE["installed"], why=_STATE["why"], cell=_STATE["cell"], version=__version__)


def _resolve_f1():
    import fpf_mkpf as MKP
    import fpf_mkpf.kernels as MK
    cells = MKP.resolve_cells()
    if not cells or "f1" not in cells:
        raise RuntimeError(f"no fpf_mkpf F1 cell for this (cc, triton): {MKP._STATE.get('why')}")
    cls = str(cells.get("class", "?"))
    arith = cells.get("ln_arith", "fused")
    if not (cls.upper().startswith("EXACT") and arith == "welford") and os.environ.get("PTX_BLOCKFUSE_XL_ALLOW_TIER2", "0") != "1":
        raise RuntimeError(f"fpf_mkpf F1 cell is not EXACT/welford on this stack (class={cls!r}, ln_arith={arith!r}); refusing (PTX_BLOCKFUSE_XL_ALLOW_TIER2=1 to accept as TIER-2)")
    fma = tuple(bool(x) for x in cells.get("fma_flags", [1, 1, 1]))
    return MK, dict(cells["f1"]["cfg"]), arith, fma, cls, MKP._STATE.get("why", "")


def statement(module, z, ending, chunk_size, epi_block_kernel=None, get_cache=None, L=None):
    """The BlockFuse lean statement (block mode: z updated in place with the stock residual semantics). Returns z."""
    from blockfuse.biasln import bias_ln
    import protenix.model.triangular.layers as TL
    MK, f1cfg, arith, fma = _STATE["f1"][:4]
    if get_cache is None or epi_block_kernel is None:
        from fpf_triatt_pro.prologue import get_cache as _gc
        from fpf_triatt_epi.epilogue import triatt_epilogue as _ep
        get_cache = get_cache or _gc; epi_block_kernel = epi_block_kernel or _ep
    x = z.transpose(-2, -3) if ending else z
    NI = int(x.shape[0]); R = int(chunk_size)
    cch = get_cache(module, z.device)
    ns = getattr(module, "_fpf_cache", None)
    if ns is None:
        ns = {}; module._fpf_cache = ns
    t2 = ns.setdefault("trunk2", {})
    wo16 = t2.get("wo16")
    if wo16 is None or wo16.device != z.device:
        wo16 = module.mha.linear_o.weight.detach().to(torch.bfloat16).contiguous(); t2["wo16"] = wo16
    ns.setdefault("wo16", wo16)
    scale = 1.0 / math.sqrt(module.mha.c_hidden)
    bias = bias_ln(module, z, cch, bool(ending), ln_arith=arith, fma_flags=fma)          # pass 1 (one launch, z read once)
    b4 = bias.unsqueeze(0)
    sm100f = getattr(L, "_cueq_sm100f_exposed", None) if L is not None else None
    tmask = getattr(L, "_stock_true_mask", None) if L is not None else None
    att = L._blk_core_att(int(x.shape[-2])) if (L is not None and hasattr(L, "_blk_core_att")) else None   # 0.1.1: the arm's block-core attention (K2B under ARM T), None = stock cuEq (ARM E)
    nchunks = 0
    for i0 in range(0, NI, R):                                                            # pass 2
        i1 = min(NI, i0 + R)
        zc = z[:, i0:i1] if ending else z[i0:i1]
        q, k, v, g, _b = MK.prologue_ln(module, zc, cch, f1cfg, ending=bool(ending), ln_arith=arith, fma_flags=fma); del _b
        if att is None:
            mask = tmask(q, i1 - i0, k.shape[-2]) if (sm100f is not None and tmask is not None and sm100f(k.shape[-2])) else None
            o = TL.cuequivariance_triangular_attn(q, k, v, b4, mask, scale); _STATE["calls"]["cueq_chunks"] += 1
        else:                                                                             # ARM T: K2B on this row-chunk, full fp32 bias (bitwise == the full-N K2B statement)
            o = att(*(t if t.dim() == 5 else t.unsqueeze(0) for t in (q, k, v)), b4, mask=None, scale=scale); _STATE["calls"]["k2b_chunks"] += 1
            L._STATS["blk_att_k2b_routed"] = L._STATS.get("blk_att_k2b_routed", 0) + 1; L._STATS["blk_att_k2b_calls"] = L._STATS["blk_att_k2b_routed"]
        o = o[0] if isinstance(o, (tuple, list)) else o
        if o.dim() == 5:
            assert o.shape[0] == 1, f"attention output leading dim {tuple(o.shape)}"; o = o[0]
        assert o.dim() == 4 and o.shape[0] == (i1 - i0) and o.shape[1] == q.shape[-3], f"blockfuse_xl: attention output layout {tuple(o.shape)} rows {i0}:{i1}"
        epi_block_kernel(o, g, wo16, zc, ending=bool(ending), residual=True)
        del o, q, k, v, g; nchunks += 1
    del bias, b4
    return z, nchunks


def eligible(module, z):
    try:
        return (z.dim() == 3 and z.dtype == torch.bfloat16 and z.stride(-1) == 1 and z.is_cuda and int(module.c_in) == 256
                and int(module.mha.linear_q.weight.shape[0]) == 256 and int(module.linear.weight.shape[0]) == int(module.mha.no_heads))
    except Exception:
        return False


def install(L=None, verbose=True):
    """Patch ptx_trunk2_levers._triatt_block_pro_epi (idempotent). Raises if F1 is not EXACT on this stack (unless PTX_BLOCKFUSE_XL_ALLOW_TIER2=1)."""
    if _STATE["installed"]:
        return stats()
    if L is None:
        import ptx_trunk2_levers as L
    MK, f1cfg, arith, fma, cls, why = _resolve_f1()
    _STATE["f1"] = (MK, f1cfg, arith, fma); _STATE["cell"] = {"cfg": f1cfg, "ln_arith": arith, "fma_flags": list(fma), "class": cls, "cells": why}
    orig = L._triatt_block_pro_epi
    _STATE["orig"] = orig

    def _triatt_block_pro_epi_blockfuse(module, z, ending, epi_block_kernel, prologue, get_cache, chunk_size=None):
        x = z.transpose(-2, -3) if ending else z
        n = int(x.shape[-2])
        c = _STATE["calls"]
        if n > c["max_ntok"]: c["max_ntok"] = n
        if chunk_size is not None and L._xl_lean_prologue(n):
            if eligible(module, z):
                _, nch = statement(module, z, ending, chunk_size, epi_block_kernel=epi_block_kernel, get_cache=get_cache, L=L)
                c["bf_stmts"] += 1; c["bf_chunks"] += nch
                L._XL_MEM["lean_prologue_calls"] = L._XL_MEM.get("lean_prologue_calls", 0) + 1
                L._STATS["blk2_tri_chunked_calls"] = L._STATS.get("blk2_tri_chunked_calls", 0) + 1
                L._STATS["blk2_tri_chunks"] = L._STATS.get("blk2_tri_chunks", 0) + nch
                L._STATS["blockfuse_xl_stmts"] = c["bf_stmts"]
                return z
            c["refused_ineligible"] += 1
        c["passthru"] += 1
        return orig(module, z, ending, epi_block_kernel, prologue, get_cache, chunk_size=chunk_size)

    _triatt_block_pro_epi_blockfuse._blockfuse_xl = True
    _triatt_block_pro_epi_blockfuse._fpf_orig = orig
    L._triatt_block_pro_epi = _triatt_block_pro_epi_blockfuse
    _STATE["installed"] = True
    _STATE["why"] = f"patched ptx_trunk2_levers._triatt_block_pro_epi; F1 cell {why} class={cls.split(' ')[0]} arith={arith}"
    try:
        L._STATS.setdefault("applied", []).append(f"BLOCKFUSE_XL:{__version__}(F1 {why},{arith},{cls.split(' ')[0]})")
    except Exception:
        pass
    if verbose:
        print(f"[blockfuse_xl] {__version__} installed: {_STATE['why']} (engages in the XL lean-prologue branch: PTX_FPF_CHUNK_MODE=prologue and N_token > PTX_FPF_CHUNK_TOK={getattr(L, '_XL_CHUNK_TOK', '?')}; attention core per row-chunk = the arm's block core, resolved per call: K2B under ARM T past its gates, else the stock cuEq kernel)", file=sys.stderr, flush=True)

    def _bye():
        try:
            print("[blockfuse_xl] EXIT " + " ".join(f"{k}={v}" for k, v in stats().items() if k != "cell"), file=sys.stderr, flush=True)
            rp = os.environ.get("PTX_BLOCKFUSE_XL_REPORT")
            if rp:
                import json
                with open(rp, "a") as fh: fh.write(json.dumps(stats(), default=str) + "\n")
        except Exception:
            pass
    atexit.register(_bye)
    return stats()
