"""attbwd_dkdv.py — the triangle-attention BACKWARD kernels re-cut (a contribution to the kit's lever `triatt`, whose owner keeps triatt_attn.py and
integrates). Same mathematics, operands and rounding points as triatt_attn's K1/K2 (P and dS rounded to the input dtype before their products,
fp32 accumulation, exp2 statistics from the forward's lse, dead-row dS = 0, masked keys p = 0, no atomics, run-to-run bitwise); what moves is
WHICH kernel forms d(pair bias):

  K1' (grid over key blocks)   dK, dV only — no dS partial sums, no partial store — computed KEY-MAJOR (_bwd_dkdv_kernel_t, FlashAttention-2's
                               dK/dV orientation: Sᵀ = K·Qᵀ, Pᵀ, dV += Pᵀ·dO, dPᵀ = V·dOᵀ, dK += dSᵀ·Q): no register tile is ever transposed
                               into an MMA (triatt's query-major K1 transposes P and dS every step).
  K2' (grid over query blocks) dQ AND the G-summed dS tile of every (query block, key block) → d(bias) partials [ceil(B/G), H, S_q, S_kp].
                               dQ's accumulator is ONE [bq, D] fp32 tile per row (16 registers at bq=64, D=32, 4 warps), so G = 8 rows per
                               program fit where K1 could only hold 2: the partial-sum traffic (the write + XLA's fp32 group reduction), which
                               was 2 x 2.2 GB per call at 800 tokens, drops 4x.
`_bwd(...)` has triatt_attn._bwd's signature and returns (dq, dk, dv, dbias); `k1` / `k2` dicts carry the tiles (bq, bk, G, num_warps, num_stages).
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from . import triatt_attn as T

_load, _store, _dot, _pids, _grid, _full, _tiles, _scaled, _dyn = T._load, T._store, T._dot, T._pids, T._grid, T._full, T._tiles, T._scaled, T._dyn
LOG2E, NEG, _CP = T.LOG2E, T.NEG, T._CP
DEFAULTS_BWD = dict(k1=dict(bq=64, bk=64, G=2, num_warps=4, num_stages=3),  # K1': dK/dV, key-major (triatt's query-major orientation measured 11 % slower: tools/attbwd_dev.py)
                    k2=dict(bq=64, bk=32, G=4, num_warps=4, num_stages=3),      # K2': dQ + d(bias) partials (219 regs, no spill; G=8 spills)
                    resident=1)                                                 # K2' keeps its rows' q / dO tiles in registers across the key loop


# =============================================================================================================== K1': dK, dV
# =============================================================================================================== K1'ᵀ: dK, dV in the key-major domain
def _bwd_dkdv_kernel_t(q_ref, k_ref, v_ref, b_ref, m_ref, d_ref, do_ref, lse_ref, dl_ref, dk_ref, dv_ref, *, scale2, scale, bq, bk, G, sq, sk, nb, D,
                       order, f32p, has_mask, bias_mma):
    """K1' computed TRANSPOSED (key-major tiles, the FlashAttention-2 dK/dV form): Sᵀ = K·Qᵀ, Pᵀ = exp2(Sᵀ·log2e − lseᵀ), dV += Pᵀ·dO,
    dPᵀ = V·dOᵀ, dSᵀ = Pᵀ⊙(dPᵀ − δᵀ), dK += dSᵀ·Q — every MMA's A operand is a tile in its natural register/SMEM layout and every transposed
    operand is a LOADED tile (an SMEM descriptor), so no register tile is ever transposed. Same products and fp32 sums as _bwd_dkdv_kernel."""
    pid = _pids(order)
    jb, h, bg = pid["i"], pid["h"], pid["b"]
    if sk >= bk:
        c0 = jnp.minimum(jb * bk, sk - bk); cvec = cmask = None
    else:
        c0 = 0; cvec = jnp.arange(bk) < sk; cmask = cvec[:, None]
    b0 = jnp.minimum(bg * G, nb - G)
    cols = pl.dslice(c0, bk)
    nfull, rem = sq // bq, sq % bq
    ks, vs, madds = [], [], []
    for g in range(G):
        ks.append(_load(k_ref, (b0 + g, h, cols, slice(None)), mask=cmask, other=0.0))
        vs.append(_load(v_ref, (b0 + g, h, cols, slice(None)), mask=cmask, other=0.0))
        if has_mask:
            live = 1.0 - _load(d_ref, (b0 + g,)).astype(jnp.float32)
            mcol = _load(m_ref, (b0 + g, cols))[:, None]                                  # [bk, 1]: the additive key mask along the KEY (row) axis here
            pad_neg = None if cvec is None else (jnp.where(cvec, 0.0, NEG)[:, None] * (1.0 - live))
            madds.append((live, mcol, pad_neg))
    kneg = None if (cvec is None or has_mask) else jnp.where(cvec, 0.0, NEG)[:, None]
    assert bias_mma, "the key-major K1 is written for bias_mma=1 (the lever's default)"
    eye = (jax.lax.broadcasted_iota(jnp.int32, (bk, bk), 0) == jax.lax.broadcasted_iota(jnp.int32, (bk, bk), 1)).astype(b_ref.dtype)

    def qblock(r0, carry, rin):
        rows = pl.dslice(r0, bq)
        rl = None if rin is None else rin[:, None]
        rt = None if rin is None else rin[None, :]
        btile = _load(b_ref, (h, rows, cols), mask=rl, other=0.0)                          # [bq, bk] as stored
        bias_t = _dot(eye, btile.T, f32p)                                                  # biasᵀ [bk, bq] in the accumulator layout (I·Bᵀ, Bᵀ an SMEM view; a plain
        #                                                                                    transposed add costs a layout conversion = +12 %, and the I-MMA is free here)
        if kneg is not None:
            bias_t = bias_t + kneg
        out = []
        for g in range(G):
            dk, dv = carry[g]
            q = _load(q_ref, (b0 + g, h, rows, slice(None)), mask=rl, other=0.0)            # [bq, D]
            do = _load(do_ref, (b0 + g, h, rows, slice(None)), mask=rl, other=0.0)
            lse = _load(lse_ref, (b0 + g, h, rows), mask=rin, other=0.0)[None, :]           # [1, bq]
            dl = _load(dl_ref, (b0 + g, h, rows), mask=rin, other=0.0)[None, :]
            kq = _dot(ks[g], q.T, f32p)                                                    # Sᵀ [bk, bq]
            if scale != 1.0:
                kq = kq * scale
            st = kq + bias_t
            if has_mask:
                st = (st + madds[g][1]) * madds[g][0]
                if madds[g][2] is not None:
                    st = st + madds[g][2]
            pt = jnp.exp2(st * LOG2E - lse)                                                # Pᵀ
            if rt is not None:
                pt = jnp.where(rt, pt, 0.0)
            dv = dv + _dot(pt.astype(do.dtype), do, f32p)                                  # [bk, D]
            dpt = _dot(vs[g], do.T, f32p)                                                  # dPᵀ [bk, bq]
            dst = pt * (dpt - dl)
            if has_mask:
                dst = dst * madds[g][0]
            dk = dk + _dot(dst.astype(q.dtype), q, f32p)                                   # [bk, D]
            out.append((dk, dv))
        return tuple(out)

    carry = tuple((jnp.zeros((bk, D), jnp.float32), jnp.zeros((bk, D), jnp.float32)) for _ in range(G))
    if nfull > 0:
        carry = jax.lax.fori_loop(0, nfull, lambda ib, cr: qblock(ib * bq, cr, None), carry)
    if rem:
        carry = qblock(_dyn(nfull * bq), carry, jnp.arange(bq) < rem)
    for g in range(G):
        dk, dv = carry[g]
        _store(dk_ref, (b0 + g, h, cols, slice(None)), (dk if scale == 1.0 else dk * scale).astype(dk_ref.dtype), mask=cmask)
        _store(dv_ref, (b0 + g, h, cols, slice(None)), dv.astype(dv_ref.dtype), mask=cmask)


# =============================================================================================================== K2': dQ + d(bias) partials
def _bwd_dqdb_kernel(q_ref, k_ref, v_ref, b_ref, m_ref, d_ref, do_ref, lse_ref, dl_ref, dq_ref, dbp_ref, *, scale2, scale, bq, bk, G, sq, sk, nb, D,
                     order, f32p, has_mask, bias_mma, resident):
    """One program = G batch rows x one head x bq query rows; loops over key blocks; dQ of its rows and the G-summed dS tile of every
    (its query block, key block) into its group's d(bias) partial [ceil(B/G), H, Sq, Skp]. `resident`: the rows' q / dO tiles stay in
    registers across the key loop (1) or are re-read per key block (0: fewer live registers for large G; the re-reads are L1/L2 hits)."""
    pid = _pids(order)
    i, h, bg = pid["i"], pid["h"], pid["b"]
    if sq >= bq:
        r0 = jnp.minimum(i * bq, sq - bq); rvec = rmask = None
    else:
        r0 = 0; rvec = jnp.arange(bq) < sq; rmask = rvec[:, None]
    b0 = jnp.minimum(bg * G, nb - G)
    rows = pl.dslice(r0, bq)
    nfull, rem = sk // bk, sk % bk
    w = [None if nb % G == 0 else ((b0 + g) >= bg * G).astype(jnp.float32) for g in range(G)]   # dbias ownership of a clamped group's rows
    lses, dls = [], []
    for g in range(G):
        lses.append(_load(lse_ref, (b0 + g, h, rows), mask=rvec, other=0.0)[:, None])
        dls.append(_load(dl_ref, (b0 + g, h, rows), mask=rvec, other=0.0)[:, None])
    live = [1.0 - _load(d_ref, (b0 + g,)).astype(jnp.float32) for g in range(G)] if has_mask else None
    if bias_mma:
        eye = (jax.lax.broadcasted_iota(jnp.int32, (bq, bq), 0) == jax.lax.broadcasted_iota(jnp.int32, (bq, bq), 1)).astype(b_ref.dtype)
        e2, smul = LOG2E, 1.0
    else:
        e2, smul = 1.0, scale2

    def q_do(g):
        return _load(q_ref, (b0 + g, h, rows, slice(None)), mask=rmask, other=0.0), _load(do_ref, (b0 + g, h, rows, slice(None)), mask=rmask, other=0.0)
    qd = [q_do(g) for g in range(G)] if resident else None

    def kblock(c0, carry, kin):
        cols = pl.dslice(c0, bk)
        if bias_mma:
            bias = _dot(eye, _load(b_ref, (h, rows, cols), mask=rmask, other=0.0), f32p)
        else:
            bias = _load(b_ref, (h, rows, cols), mask=rmask, other=0.0)
        if kin is not None and not has_mask:
            bias = bias + jnp.where(kin, 0.0, NEG)[None, :]
        kl = None if kin is None else kin[:, None]
        out = []; dsum = None
        for g in range(G):
            dq = carry[g]
            q, do = qd[g] if resident else q_do(g)
            k = _load(k_ref, (b0 + g, h, cols, slice(None)), mask=kl, other=0.0)
            v = _load(v_ref, (b0 + g, h, cols, slice(None)), mask=kl, other=0.0)
            qk = _dot(q, k.T, f32p)
            if bias_mma and scale != 1.0:
                qk = qk * scale
            s = (qk + bias) if bias_mma else (qk * smul + bias)
            if has_mask:
                s = (s + _load(m_ref, (b0 + g, cols))[None, :]) * live[g]
                if kin is not None:
                    s = s + jnp.where(kin, 0.0, NEG)[None, :] * (1.0 - live[g])
            p = jnp.exp2(s * e2 - lses[g]) if bias_mma else jnp.exp2(s - lses[g])
            dp = _dot(do, v.T, f32p)
            ds = p * (dp - dls[g])
            if has_mask:
                ds = ds * live[g]                                                 # dead row: dQ = 0 and no d(bias) (stock's clip)
            if rmask is not None:
                ds = jnp.where(rmask, ds, 0.0)                                    # rows beyond S_q (only when S_q < bq) carry no d(bias)
            dq = dq + _dot(ds.astype(k.dtype), k, f32p)
            c = ds if w[g] is None else ds * w[g]
            dsum = c if dsum is None else dsum + c
            out.append(dq)
        _store(dbp_ref, (bg, h, rows, cols), dsum.astype(dbp_ref.dtype), mask=rmask)       # padded / out-of-range key columns hold exact zeros (p = 0 there)
        return tuple(out)

    carry = tuple(jnp.zeros((bq, D), jnp.float32) for _ in range(G))
    if nfull > 0:
        carry = jax.lax.fori_loop(0, nfull, lambda j, c: kblock(j * bk, c, None), carry)
    if rem:
        carry = kblock(_dyn(nfull * bk), carry, jnp.arange(bk) < rem)
    for g in range(G):
        _store(dq_ref, (b0 + g, h, rows, slice(None)), (carry[g] * scale).astype(dq_ref.dtype) if scale != 1.0 else carry[g].astype(dq_ref.dtype), mask=rmask)


def k1_call(q, k, v, bias2, madd, dead, do, lse, delta, *, scale, k1, order, f32p, has_mask, bias_mma, kernel=None):
    """K1' launch: (dk, dv). `kernel`: an alternative kernel body with _bwd_dkdv_kernel_t's signature (dev A/B)."""
    B, H, Sq, D = q.shape; Sk = k.shape[2]
    ins = [_full(q), _full(k), _full(v), _full(bias2), _full(madd), _full(dead), _full(do), _full(lse), _full(delta)]
    base = dict(scale2=float(scale) * LOG2E, scale=float(scale), sq=Sq, sk=Sk, nb=B, D=D, order=order, f32p=f32p, has_mask=has_mask, bias_mma=bool(bias_mma))
    bq, bk, G = _tiles(Sq, Sk, B, H, k1["bq"], k1["bk"], k1["G"], "k")
    tag = f"{'m' if has_mask else 'u'}_t_G{G}_bq{bq}_bk{bk}_w{k1['num_warps']}_s{k1['num_stages']}"
    return pl.pallas_call(
        functools.partial(kernel or _bwd_dkdv_kernel_t, bq=bq, bk=bk, G=G, **base), grid=_grid(order, pl.cdiv(Sk, bk), H, pl.cdiv(B, G)),
        in_specs=ins,
        out_specs=[pl.BlockSpec((B, H, Sk, D), lambda *_: (0, 0, 0, 0)), pl.BlockSpec((B, H, Sk, D), lambda *_: (0, 0, 0, 0))],
        out_shape=[jax.ShapeDtypeStruct((B, H, Sk, D), k.dtype), jax.ShapeDtypeStruct((B, H, Sk, D), v.dtype)],
        compiler_params=_CP(num_warps=k1["num_warps"], num_stages=k1["num_stages"]),
        name=f"attbwd_dkdv_{tag}",
    )(q, k, v, bias2, madd, dead, do, lse, delta)


def k2_call(q, k, v, bias2, madd, dead, do, lse, delta, *, scale, k2, order, f32p, has_mask, bias_mma, pdt, resident=0, kernel=None):
    """K2' launch: (dq, dbias partials [ceil(B/G), H, Sq, Skp])."""
    B, H, Sq, D = q.shape; Sk = k.shape[2]; Skp = bias2.shape[-1]
    ins = [_full(q), _full(k), _full(v), _full(bias2), _full(madd), _full(dead), _full(do), _full(lse), _full(delta)]
    base = dict(scale2=float(scale) * LOG2E, scale=float(scale), sq=Sq, sk=Sk, nb=B, D=D, order=order, f32p=f32p, has_mask=has_mask, bias_mma=bool(bias_mma))
    bq, bk, G = _tiles(Sq, Sk, B, H, k2["bq"], k2["bk"], k2["G"], "q")
    ngrp = pl.cdiv(B, G)
    tag = f"{'m' if has_mask else 'u'}_G{G}_bq{bq}_bk{bk}_w{k2['num_warps']}_s{k2['num_stages']}"
    return pl.pallas_call(
        functools.partial(kernel or _bwd_dqdb_kernel, bq=bq, bk=bk, G=G, resident=int(resident), **base), grid=_grid(order, pl.cdiv(Sq, bq), H, ngrp),
        in_specs=ins,
        out_specs=[pl.BlockSpec((B, H, Sq, D), lambda *_: (0, 0, 0, 0)), pl.BlockSpec((ngrp, H, Sq, Skp), lambda *_: (0, 0, 0, 0))],
        out_shape=[jax.ShapeDtypeStruct((B, H, Sq, D), q.dtype), jax.ShapeDtypeStruct((ngrp, H, Sq, Skp), pdt)],
        compiler_params=_CP(num_warps=k2["num_warps"], num_stages=k2["num_stages"]),
        name=f"attbwd_dqdb_{tag}",
    )(q, k, v, bias2, madd, dead, do, lse, delta)


def _bwd(q, k, v, bias2, madd, dead, o, lse, do, *, scale, k1, k2, order, f32p, has_mask, bias_mma, pdt, skip=(), resident=1):
    """triatt_attn._bwd's signature and results: (dq, dk, dv, dbias fp32 [H, Sq, Sk]). skip: ('k1',) / ('k2',) = that kernel's outputs as zeros (diagnostic)."""
    B, H, Sq, D = q.shape; Sk = k.shape[2]; Skp = bias2.shape[-1]
    delta = jnp.einsum("bhqd,bhqd->bhq", o.astype(jnp.float32), do.astype(jnp.float32))
    if "k1" in skip:
        dk, dv = jnp.zeros((B, H, Sk, D), k.dtype), jnp.zeros((B, H, Sk, D), v.dtype)
    else:
        dk, dv = k1_call(q, k, v, bias2, madd, dead, do, lse, delta, scale=scale, k1=k1, order=order, f32p=f32p, has_mask=has_mask, bias_mma=bias_mma)
    if "k2" in skip:
        G2 = _tiles(Sq, Sk, B, H, k2["bq"], k2["bk"], k2["G"], "q")[2]
        dq, dbp = jnp.zeros((B, H, Sq, D), q.dtype), jnp.zeros((pl.cdiv(B, G2), H, Sq, Skp), pdt)
    else:
        dq, dbp = k2_call(q, k, v, bias2, madd, dead, do, lse, delta, scale=scale, k2=k2, order=order, f32p=f32p, has_mask=has_mask, bias_mma=bias_mma, pdt=pdt, resident=resident)
    dbias = jnp.sum(dbp[:, :, :, :Sk], axis=0, dtype=jnp.float32)
    return dq, dk, dv, dbias
