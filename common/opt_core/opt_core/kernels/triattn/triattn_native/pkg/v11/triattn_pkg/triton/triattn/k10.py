"""k10 -- triangle-attention forward (inference) in Triton for H100, the successor of opt_core's flash_triattn v9 kernel
(opt_core/kernels/flash_triattn.py).

    out[b,i,h,q,:] = softmax_k( scale * q[b,i,h,q,:] . k[b,i,h,k,:] + bias[b,0,h,q,k] (+ -1e9 where mask[b,i,0,0,k]==0) ) @ v[b,i,h,k,:]

q, k, v [B, N, H, S, D] (bf16 / fp16, any strides with stride(-1) == 1), bias [B, 1, H, S_q, S_k] (fp32, bf16 or fp16; shared by
all N rows), mask [B, N, 1, 1, S_k] bool (True = keep), out [B, N, H, S_q, D] contiguous in q.dtype.

One program owns ROWS pair rows x one BLOCK_M query tile of one (b, h) and loops over key tiles: per key tile it loads ONE bias
tile (shared by its ROWS rows; converted to fp32 and scaled by log2(e) once) and ROWS K/V tiles, and keeps ROWS running
softmax states (m, l, acc) in registers.  Logits live in the base-2 domain (scale*log2e folded into the QK^T scale, log2e into
the bias tile) so the exponential is a bare ex2.  The kernel source is GENERATED per ROWS value (Triton has no arrays of
register tiles): see _render().  Numerics otherwise follow v9 / FlashAttention: bf16 tensor-core products with fp32
accumulation, fp32 online softmax, P rounded to v.dtype for the PV product, -1e9 select for masked keys (cuEquivariance
semantics: a fully-masked row yields the uniform average of v).  No atomics; fixed reduction order per config -> run-to-run
bitwise reproducible.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import sys
from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

from .errors import Unsupported
from .masking import census, mask_tables

_LOG2E = 1.4426950408889634
_GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_gen")


# --------------------------------------------------------------------------------------------------------------------------
# bias preparation (fp32 bias only): fp32 copy with unit key stride + 16-bit copy + per-program lossless flags (from v9)
# --------------------------------------------------------------------------------------------------------------------------
@triton.jit
def _bias_prep(Bias, Out32, Out16, Flags, OrPat, sbb, sbh, sbq, sbk, H, SQ, SK, SKp, NQB, msent,
               PB: tl.constexpr, PK: tl.constexpr, MAKE16: tl.constexpr, FOLD_MASK: tl.constexpr):
    """Stage bias[b,0,h] as contiguous fp32 (Out32) and, if MAKE16, 16-bit (Out16) [SQ, SKp] tiles; Flags[bh, qblock] = 1 iff the
    16-bit copy is lossless.  FOLD_MASK: keys kept by no pair row of the batch element (OrPat[b, k] == 0, masking.py) get the mask
    logit `msent` (-2**30 for bf16 / fp32 tiles; -32768 when the 16-bit tiles are fp16, which cannot hold -2**30)."""
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = (pid_bh // H).to(tl.int64)
    h = (pid_bh % H).to(tl.int64)
    rows = pid_q * PB + tl.arange(0, PB)
    rmask = rows < SQ
    src = Bias + b * sbb + h * sbh
    dst = pid_bh.to(tl.int64) * SQ * SKp
    nbad = tl.zeros([PB], dtype=tl.int32)
    for k0 in range(0, SK, PK):
        cols = k0 + tl.arange(0, PK)
        m2 = rmask[:, None] & (cols < SK)[None, :]
        w2 = rmask[:, None] & (cols < SKp)[None, :]
        x = tl.load(src + rows[:, None].to(tl.int64) * sbq + cols[None, :] * sbk, mask=m2, other=0.0).to(tl.float32)
        if FOLD_MASK:
            keep = tl.load(OrPat + b * SK + cols, mask=cols < SK, other=0)
            x = tl.where(keep[None, :] != 0, x, msent)
        o = dst + rows[:, None] * SKp + cols[None, :]
        tl.store(Out32 + o, x, mask=w2)
        if MAKE16:
            x16 = x.to(Out16.dtype.element_ty)
            tl.store(Out16 + o, x16, mask=w2)
            nbad += tl.sum((x16.to(tl.float32) != x).to(tl.int32), 1)
    if MAKE16:
        ok = (tl.sum(nbad, 0) == 0).to(tl.int32)
        tl.store(Flags + pid_bh * NQB + pid_q, ok)



# --------------------------------------------------------------------------------------------------------------------------
# generated attention kernel
# --------------------------------------------------------------------------------------------------------------------------
_HEADER = """
import triton
import triton.language as tl


@triton.jit
def _qk(q, k_base, k_lo, k_off, kvm, MASK_N: tl.constexpr):
    if MASK_N:
        k = tl.load(k_base + k_lo + k_off, mask=kvm[:, None], other=0.0)
    else:
        k = tl.load(k_base + k_lo + k_off)
    return tl.dot(q, tl.trans(k))


@triton.jit
def _finish(acc, m_i, l_i, lacc, s, v_base, m_base, v_lo, m_lo, v_off, m_off, s_bias, kvm, qk_scale2, ones_kn,
            MASK_N: tl.constexpr, APPLY_MASK: tl.constexpr, EXP2: tl.constexpr, LSUM: tl.constexpr):
    if MASK_N:
        v = tl.load(v_base + v_lo + v_off, mask=kvm[:, None], other=0.0)
    else:
        v = tl.load(v_base + v_lo + v_off)
    s = s * qk_scale2 + s_bias
    if APPLY_MASK:
        if MASK_N:
            mk = tl.load(m_base + m_lo + m_off, mask=kvm, other=1)
        else:
            mk = tl.load(m_base + m_lo + m_off)
        if EXP2:
            s = tl.where(mk[None, :] != 0, s, -1.4426950408889634e9)
        else:
            s = tl.where(mk[None, :] != 0, s, -1.0e9)
    if MASK_N:
        s = tl.where(kvm[None, :], s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, 1))
    if EXP2:
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
    else:
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
    if LSUM:
        l_new = l_i
    else:
        l_new = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None]
    p16 = p.to(v.dtype)
    if LSUM:
        lacc = lacc * alpha[:, None]
        lacc = tl.dot(p16, ones_kn, lacc)
    acc = tl.dot(p16, v, acc)
    return acc, m_new, l_new, lacc


@triton.jit
def _fwd(
    Q, K, V, Bias32, Bias16, Flags, Mask, Out,
    sqb, sqi, sqh, sqq,
    skb, ski, skh, skk,
    svb, svi, svh, svk,
    sbb, sbh, sbq,
    smb, smi, smk,
    sob, soi, soh, soq,
    N_ROWS, SEQ_Q, SEQ_K, H, n_flags, RowInfo,
    qk_scale2, bias_scale,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HAS_MASK: tl.constexpr, EXP2: tl.constexpr, BIAS_MODE: tl.constexpr, ORDER: tl.constexpr,
    NFLAG: tl.constexpr, LSUM: tl.constexpr,
):
    # BIAS_MODE: 0 = fp32 tiles (Bias32), 1 = 16-bit tiles (Bias16), 2 = 16-bit tiles iff every Flags entry is nonzero else fp32
    if ORDER == 0:
        pid_q = tl.program_id(0)
        pid_r = tl.program_id(1)
    else:
        pid_r = tl.program_id(0)
        pid_q = tl.program_id(1)
    pid_bh = tl.program_id(2)
    b = (pid_bh // H).to(tl.int64)
    h = (pid_bh % H).to(tl.int64)
    offs_m = pid_q * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_mc = tl.minimum(offs_m, SEQ_Q - 1)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)
    k_off = offs_n[:, None] * skk + offs_d[None, :]
    v_off = offs_n[:, None] * svk + offs_d[None, :]
    m_off = offs_n * smk
    b_off = offs_mc[:, None] * sbq + offs_n[None, :]
    q_off = offs_mc[:, None] * sqq + offs_d[None, :]
    bias32_bh = Bias32 + b * sbb + h * sbh
    bias16_bh = Bias16 + b * sbb + h * sbh
    kv_all = offs_n < BLOCK_N
    ones_kn = tl.full([BLOCK_N, 16], 1.0, dtype=V.dtype.element_ty)
    r0 = pid_r * ROWS_LIT
    hi = (SEQ_K // BLOCK_N) * BLOCK_N
    n_kt = hi // BLOCK_N                  # full key tiles to visit
    tail_ok = True                        # and then the ragged tail tile, if SEQ_K has one
"""



def _render(rows: int, pipe: bool, ws: bool = False) -> str:
    """The kernel source for ROWS = rows (per-row register state unrolled by name).  pipe: issue row r+1's QK^T dot before
    row r's softmax so the tensor core overlaps the FMA/MUFU work (the K/V/bias tiles are prefetched by Triton's pipeliner
    either way)."""
    R = range(rows)
    L = []
    ap = L.append
    src = _HEADER.replace("ROWS_LIT", str(rows))
    ind = "    "
    for r in R:
        ap(f"{ind}i{r} = tl.minimum(r0 + {r}, N_ROWS - 1).to(tl.int64)")
        ap(f"{ind}q{r} = tl.load(Q + b * sqb + i{r} * sqi + h * sqh + q_off)")
        ap(f"{ind}kb{r} = K + b * skb + i{r} * ski + h * skh")
        ap(f"{ind}vb{r} = V + b * svb + i{r} * svi + h * svh")
        ap(f"{ind}mb{r} = Mask + b * smb + i{r} * smi")
        ap(f"{ind}acc{r} = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)")
        ap(f"{ind}lacc{r} = tl.zeros([BLOCK_M, 16], dtype=tl.float32)")
        ap(f"{ind}m{r} = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')")
        ap(f"{ind}l{r} = tl.zeros([BLOCK_M], dtype=tl.float32)")
    ap(f"{ind}use16 = BIAS_MODE == 1")
    ap(f"{ind}if BIAS_MODE == 2:")
    ap(f"{ind}    foffs = tl.arange(0, NFLAG)")
    ap(f"{ind}    fmin = tl.min(tl.load(Flags + foffs, mask=foffs < n_flags, other=1), 0)")
    ap(f"{ind}    use16 = fmin != 0")
    ap(f"{ind}all_unmasked = True")
    ap(f"{ind}if HAS_MASK:")
    ap(f"{ind}    # per-row key-tile table (masking.py): tiles to visit / first tile needing the per-key select; keys masked for every")
    ap(f"{ind}    # row of the batch element already carry the sentinel in the staged bias, so rows equal to that pattern need no select")
    ap(f"{ind}    nt = tl.load(RowInfo + (b * N_ROWS + i0) * 2)")
    ap(f"{ind}    ns = tl.load(RowInfo + (b * N_ROWS + i0) * 2 + 1)")
    for r in R:
        if r == 0:
            continue
        ap(f"{ind}    nt = tl.maximum(nt, tl.load(RowInfo + (b * N_ROWS + i{r}) * 2))")
        ap(f"{ind}    ns = tl.minimum(ns, tl.load(RowInfo + (b * N_ROWS + i{r}) * 2 + 1))")
    ap(f"{ind}    tail_ok = nt * BLOCK_N > hi")
    ap(f"{ind}    n_kt = tl.minimum(n_kt, nt)")
    ap(f"{ind}    all_unmasked = ns >= nt")

    def tile(indent: str, apply_mask: str, bias_ptr: str, tail: bool):
        o = []
        w = o.append
        mask_n = "True" if tail else "False"
        kvm = "kvm" if tail else "kv_all"
        base = "hi" if tail else "start_n"
        if tail:
            w(f"{indent}kvm = (hi + offs_n) < SEQ_K")
            w(f"{indent}s_bias = tl.load({bias_ptr} + hi + b_off, mask=kvm[None, :], other=0.0).to(tl.float32) * bias_scale")
        else:
            w(f"{indent}s_bias = tl.load({bias_ptr} + start_n + b_off).to(tl.float32) * bias_scale")
        w(f"{indent}k_lo = {base} * skk")
        w(f"{indent}v_lo = {base} * svk")
        w(f"{indent}m_lo = {base} * smk")
        fin = (f"vb{{r}}, mb{{r}}, v_lo, m_lo, v_off, m_off, s_bias, {kvm}, qk_scale2, ones_kn, "
               f"MASK_N={mask_n}, APPLY_MASK={apply_mask}, EXP2=EXP2, LSUM=LSUM)")
        if pipe:
            w(f"{indent}s_nx = _qk(q0, kb0, k_lo, k_off, {kvm}, MASK_N={mask_n})")
            for r in R:
                w(f"{indent}s_cu = s_nx")
                if r + 1 < rows:
                    w(f"{indent}s_nx = _qk(q{r+1}, kb{r+1}, k_lo, k_off, {kvm}, MASK_N={mask_n})")
                w(f"{indent}acc{r}, m{r}, l{r}, lacc{r} = _finish(acc{r}, m{r}, l{r}, lacc{r}, s_cu, " + fin.format(r=r))
        else:
            for r in R:
                w(f"{indent}s_cu = _qk(q{r}, kb{r}, k_lo, k_off, {kvm}, MASK_N={mask_n})")
                w(f"{indent}acc{r}, m{r}, l{r}, lacc{r} = _finish(acc{r}, m{r}, l{r}, lacc{r}, s_cu, " + fin.format(r=r))
        return o

    def attend(indent: str, apply_mask: str, bias_ptr: str):
        loop = "tl.range(0, n_kt, 1, warp_specialize=True)" if ws else "range(0, n_kt)"
        o = [f"{indent}for kt in {loop}:", f"{indent}    start_n = kt * BLOCK_N"]
        o += tile(indent + "    ", apply_mask, bias_ptr, tail=False)
        o.append(f"{indent}if tail_ok & (hi < SEQ_K):")
        o += tile(indent + "    ", apply_mask, bias_ptr, tail=True)
        return o

    ap(f"{ind}if all_unmasked:")
    ap(f"{ind}    if use16:")
    L += attend(ind + "        ", "False", "bias16_bh")
    ap(f"{ind}    else:")
    L += attend(ind + "        ", "False", "bias32_bh")
    ap(f"{ind}else:")
    ap(f"{ind}    if use16:")
    L += attend(ind + "        ", "True", "bias16_bh")
    ap(f"{ind}    else:")
    L += attend(ind + "        ", "True", "bias32_bh")
    ap(f"{ind}row_ok = (offs_m < SEQ_Q)[:, None]")
    ap(f"{ind}o_off = offs_mc[:, None] * soq + offs_d[None, :]")
    ap(f"{ind}o_bh = Out + b * sob + h * soh")
    for r in R:
        cond = "row_ok" if r == 0 else f"row_ok & ((r0 + {r}) < N_ROWS)"
        ap(f"{ind}if LSUM:")
        ap(f"{ind}    l{r} = tl.max(lacc{r}, 1)")
        ap(f"{ind}tl.store(o_bh + i{r} * soi + o_off, (acc{r} / l{r}[:, None]).to(Out.dtype.element_ty), mask={cond})")
    return src + "\n".join(L) + "\n"


_MODULES: Dict[Tuple[int, bool, bool], object] = {}


def kernel_for(rows: int, pipe: bool = False, ws: bool = False):
    """The generated Triton kernel (JITFunction) for this (ROWS, PIPE, WS) (source written under _gen/, imported once)."""
    key = (rows, bool(pipe), bool(ws))
    mod = _MODULES.get(key)
    if mod is None:
        src = _render(rows, bool(pipe), bool(ws))
        tag = hashlib.sha1(src.encode()).hexdigest()[:10]
        # the generated module: <this dir>/_gen/<name> on a writable install (unchanged); under a
        # read-only tree _gen_file writes it under the kit's JIT root, else the temp dir; a file of
        # that name already there is compared with src, never imported: _gen_exec runs src itself
        path = _gen_file(
            f"k10_rows{rows}_pipe{int(bool(pipe))}_ws{int(bool(ws))}_{tag}.py",
            src,
        )
        name = f"triattn_k10_gen_rows{rows}_pipe{int(bool(pipe))}_ws{int(bool(ws))}_{tag}"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        _gen_exec(mod, src, path)
        _MODULES[key] = mod
    return mod._fwd


# --------------------------------------------------------------------------------------------------------------------------
# Python wrapper
# --------------------------------------------------------------------------------------------------------------------------
DEFAULT_CONFIG = dict(BLOCK_M=64, BLOCK_N=32, ROWS=2, num_warps=4, num_stages=3, ORDER=0, PIPE=0, LSUM=0, maxnreg=0, WS=0)
_SUPPORTED_D = (16, 32, 64, 128)
INFO: Dict = {}          # per-call notes of the last call (mask mode, verification flags)
KERNEL_INFO: Dict[Tuple, Dict] = {}     # (rows, const tuple, warps, stages) -> {n_regs, n_spills, shared}


def _ensure_dims(t: torch.Tensor, n: int) -> torch.Tensor:
    while t.dim() < n:
        t = t.unsqueeze(0)
    return t


def triattn_k10(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor,
    mask: Optional[torch.Tensor] = None, scale: Optional[float] = None, *,
    config: Optional[Dict[str, int]] = None,
    exp2: bool = True,
    bias16: Optional[bool] = None,      # None: 16-bit tiles when the bias is 16-bit or losslessly castable (checked on device); True: force (lossy); False: fp32 tiles
    out_dtype: Optional[torch.dtype] = None,
    fold_mask: bool = True,             # fold the batch OR-mask into the staged bias (tile skipping for padding masks); False = one launch fewer
) -> torch.Tensor:
    q = _ensure_dims(q, 5); k = _ensure_dims(k, 5); v = _ensure_dims(v, 5); bias = _ensure_dims(bias, 5)
    if mask is not None:
        mask = _ensure_dims(mask, 5)
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
    B, N, H, SQ, D = q.shape
    SK = k.shape[3]
    if k.shape != (B, N, H, SK, D) or v.shape != (B, N, H, SK, D):
        raise Unsupported(f"k/v must be (B,N,H,S_kv,D); got {tuple(k.shape)} / {tuple(v.shape)}")
    if bias.shape != (B, 1, H, SQ, SK):
        raise Unsupported(f"bias must be (B,1,H,S_q,S_kv); got {tuple(bias.shape)}")
    if mask is not None and mask.shape != (B, N, 1, 1, SK):
        raise Unsupported(f"mask must be (B,N,1,1,S_kv); got {tuple(mask.shape)}")
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise Unsupported(f"q/k/v must all be bf16 or fp16; got {q.dtype}/{k.dtype}/{v.dtype}")
    if D not in _SUPPORTED_D:
        raise Unsupported(f"head_dim {D} unsupported")
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    out = torch.empty((B, N, H, SQ, D), dtype=out_dtype or q.dtype, device=q.device)
    if out.numel() == 0:
        return out
    if SK == 0:
        return out.fill_(float("nan"))
    if q.stride(-1) != 1:
        q = q.contiguous()
    if k.stride(-1) != 1:
        k = k.contiguous()
    if v.stride(-1) != 1:
        v = v.contiguous()
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    bm, bn, rows, order = int(cfg["BLOCK_M"]), int(cfg["BLOCK_N"]), int(cfg["ROWS"]), int(cfg.get("ORDER", 0))
    nw, ns = int(cfg["num_warps"]), int(cfg["num_stages"])
    pipe, lsum, maxnreg, ws = bool(cfg.get("PIPE", 0)), bool(cfg.get("LSUM", 0)), int(cfg.get("maxnreg", 0) or 0), bool(cfg.get("WS", 0))

    # ---- mask tables + bias staging -------------------------------------------------------------------------------------
    dev = q.device
    has_mask = mask is not None
    fold = has_mask and fold_mask
    if has_mask:
        mask_u8 = mask.view(torch.uint8)
        if mask_u8.stride(-1) != 1:
            mask_u8 = mask_u8.contiguous()
        orpat, rowinfo, n_all = mask_tables(mask_u8.view(B, N, SK) if mask_u8.is_contiguous() else mask_u8[:, :, 0, 0, :], bn, fold=fold_mask)
        INFO["mask_census"] = lambda: census(rowinfo, n_all)
        smb, smi, smk = mask_u8.stride(0), mask_u8.stride(1), mask_u8.stride(4)
    else:
        mask_u8, rowinfo, smb, smi, smk = out, None, 0, 0, 0
    if not fold and bias.dtype in (torch.bfloat16, torch.float16) and bias16 is not False:
        b16 = bias if bias.stride(-1) == 1 or SK == 1 else bias.contiguous()
        bias32_t, bias16_t, flags, n_flags, bias_mode = b16, b16, _ones(dev), 1, 1
        sbb, sbh, sbq = b16.stride(0), b16.stride(2), b16.stride(3)
    elif not fold and bias.dtype == torch.float32 and bias16 is False and bias.stride(-1) == 1:
        bias32_t, bias16_t, flags, n_flags, bias_mode = bias, bias, _ones(dev), 1, 0
        sbb, sbh, sbq = bias.stride(0), bias.stride(2), bias.stride(3)
    else:
        biasf = bias
        SKp = -(-SK // 16) * 16
        PB, PK = 32, 128
        nqb = triton.cdiv(SQ, PB)
        make16 = bias16 is not False
        bias32_t = torch.empty((B, 1, H, SQ, SKp), dtype=torch.float32, device=dev)
        if make16:
            bias16_t = torch.empty((B, 1, H, SQ, SKp), dtype=q.dtype, device=dev)
            n_flags = B * H * nqb
            flags = torch.empty((n_flags,), dtype=torch.int32, device=dev)
            bias_mode = 1 if (bias16 is True or bias.dtype == q.dtype) else 2
        else:
            bias16_t, flags, n_flags, bias_mode = bias32_t, _ones(dev), 1, 0
        if SQ * SKp >= 2 ** 31:
            raise ValueError("bias plane too large for int32 offsets in the prep kernel")
        _bias_prep[(nqb, B * H)](biasf, bias32_t, bias16_t, flags, orpat if fold else biasf,
                                 biasf.stride(0), biasf.stride(2), biasf.stride(3), biasf.stride(4),
                                 H, SQ, SK, SKp, nqb, -32768.0 if (make16 and q.dtype == torch.float16) else -1073741824.0,
                                 PB=PB, PK=PK, MAKE16=make16, FOLD_MASK=fold, num_warps=4, num_stages=2)
        sbb, sbh, sbq = bias32_t.stride(0), bias32_t.stride(2), bias32_t.stride(3)

    # int32 in-tile offset guards (row / batch / head bases are int64 in the kernel)
    if bn * max(k.stride(3), v.stride(3), 1) + D >= 2 ** 31 or bm * max(q.stride(3), out.stride(3), sbq, 1) + max(bn, D) >= 2 ** 31:
        raise ValueError("tile offsets exceed int32")
    n_q, n_r = triton.cdiv(SQ, bm), triton.cdiv(N, rows)
    grid = (n_q, n_r, B * H) if order == 0 else (n_r, n_q, B * H)
    kern = kernel_for(rows, pipe, ws)
    extra = dict(maxnreg=maxnreg) if maxnreg else {}
    qk_scale2 = float(scale) * (_LOG2E if exp2 else 1.0)
    bias_scale = _LOG2E if exp2 else 1.0
    ck = kern[grid](
        q, k, v, bias32_t, bias16_t, flags, mask_u8, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        sbb, sbh, sbq,
        smb, smi, smk,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        N, SQ, SK, H, n_flags, rowinfo if has_mask else flags,
        qk_scale2, bias_scale,
        HEAD_DIM=D, BLOCK_M=bm, BLOCK_N=bn, HAS_MASK=has_mask, EXP2=bool(exp2), BIAS_MODE=bias_mode, ORDER=order,
        NFLAG=max(1024, triton.next_power_of_2(n_flags)), LSUM=lsum,
        num_warps=nw, num_stages=ns, **extra,
    )
    key = (rows, bm, bn, D, has_mask, bool(exp2), bias_mode, order, nw, ns, str(q.dtype), pipe, lsum, maxnreg, ws)
    if key not in KERNEL_INFO and ck is not None:
        KERNEL_INFO[key] = dict(n_regs=getattr(ck, "n_regs", None), n_spills=getattr(ck, "n_spills", None),
                                shared=getattr(getattr(ck, "metadata", None), "shared", None))
    return out


_ONES_CACHE: Dict = {}


def _ones(dev):
    t = _ONES_CACHE.get(dev)
    if t is None:
        t = torch.ones((1,), dtype=torch.int32, device=dev)
        _ONES_CACHE[dev] = t
    return t


def _gen_dirs():
    """Where a generated kernel module is written, in order: ``_GEN_DIR`` (<this directory>/_gen — a writable install: unchanged); under a
    read-only tree (a container image's file system, a shared read-only checkout) the kit's JIT root
    ``<MODEL_OPT_JIT_ROOT>[/<MODEL_OPT_STACK_KEY>]/opt_core_gen/triattn_pkg/triton/triattn/_gen``, then ``<tempdir>/opt_core_gen-uid<uid>/triattn_pkg/triton/triattn/_gen``.  The generated
    source is byte-identical wherever it lands (the Triton cache keys on the source text, not the path)."""
    rel = os.path.join("triattn_pkg", "triton", "triattn", "_gen")
    dirs = [_GEN_DIR]
    root = os.environ.get("MODEL_OPT_JIT_ROOT")
    if root:
        key = os.environ.get("MODEL_OPT_STACK_KEY")
        dirs.append(os.path.join(root, key, "opt_core_gen", rel) if key else os.path.join(root, "opt_core_gen", rel))
    import tempfile
    dirs.append(os.path.join(tempfile.gettempdir(), "opt_core_gen-uid%d" % os.getuid(), rel))
    return dirs


def _gen_file(fname: str, src: str) -> str:
    """The generated module's path: the first directory of ``_gen_dirs()`` that may hold generated source (``_gen_why_not``) and that holds
    ``src`` under ``fname`` byte for byte when this returns.  A file already there is compared with ``src``: equal bytes are kept, anything
    else is refused by name and replaced (atomically); a directory where it cannot be replaced is skipped as if absent.  The bytes on disk
    are never imported -- ``kernel_for`` runs ``src`` itself (``_gen_exec``)."""
    data = src.encode()
    err = None
    for d in _gen_dirs():
        path = os.path.join(d, fname)
        try:
            os.makedirs(d, mode=0o755, exist_ok=True)
            why = _gen_why_not(d)
            if why:
                _gen_say(f"directory {d}", f"{why}; skipped as if absent, the next location serves")
                err = err or PermissionError(f"{d}: {why}")
                continue
            found = _gen_found(path, data)
            if found is not True:
                tmp = path + f".tmp{os.getpid()}"
                if os.path.lexists(tmp):
                    os.unlink(tmp)
                with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644), "wb") as f:   # O_EXCL: never through a link left at that name
                    f.write(data)
                os.replace(tmp, path)
                if found is False:
                    _gen_say(f"file {path}", "its bytes were not the generated source; replaced by the generated source (nothing to fix)")
            return path
        except OSError as e:
            err = e
            found = _gen_found(path, data)
            if found is True:
                return path
            if found is False:
                _gen_say(f"file {path}", f"its bytes are not the generated source and it cannot be replaced ({e.strerror}); "
                                         "skipped as if absent, the next location serves; to use this location delete that file")
    raise err


def _gen_why_not(d: str):
    """Why directory ``d`` may not hold generated source, else None -- the cache directories' rule: group or others can write it, or its
    owner is neither this user nor root (uid 0 accepts any owner: in a container the bound directories belong to the host user)."""
    st = os.stat(d)
    if st.st_mode & 0o022:
        return f"group or others can write it (mode {st.st_mode & 0o7777:04o}; to use it: chmod go-w)"
    uid = os.getuid()
    if uid != 0 and st.st_uid not in (0, uid):
        return f"it belongs to uid {st.st_uid}, not to this user (uid {uid}) or root (to use it: make it a directory of this user)"
    return None


def _gen_found(path: str, data: bytes):
    """What is at ``path``: None -- nothing; True -- a regular file (not a link) whose bytes are exactly ``data``; False -- anything else."""
    import stat
    try:
        mode = os.lstat(path).st_mode
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError:
        return False
    try:
        if not stat.S_ISREG(mode):
            return False
        with open(path, "rb") as f:
            return f.read(len(data) + 1) == data
    except OSError:
        return False


_GEN_SAID = set()


def _gen_say(what: str, why: str) -> None:
    """One stderr line per refused directory or file, once per process."""
    if what not in _GEN_SAID:
        _GEN_SAID.add(what)
        sys.stderr.write(f"[opt_core] GENERATED-SOURCE refused {what}: {why}\n")
        sys.stderr.flush()


def _gen_exec(mod, src: str, path: str) -> None:
    """Run the generated source in ``mod``.  The code object is compiled from ``src`` -- the text this process generated -- with ``path`` as
    its file name: neither the bytes at ``path`` nor a cached .pyc beside it are read.  The line cache is primed with ``src`` first, so
    inspect.getsourcelines (how the Triton JIT reads a kernel's text) returns it without reading ``path`` either."""
    import linecache
    linecache.cache[path] = (len(src), None, src.splitlines(True), path)
    exec(compile(src, path, "exec", dont_inherit=True), mod.__dict__)
