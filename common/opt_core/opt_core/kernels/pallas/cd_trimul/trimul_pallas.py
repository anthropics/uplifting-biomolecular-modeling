"""trimul_pallas.py — fused GPU kernels (Pallas, Triton lowering) for AF2-Multimer's `TriangleMultiplication` in its fused-projection
form (`modules.TriangleMultiplication._fused_triangle_multiplication`, `fuse_projection_weights=True` — the form every multimer_v3 config
uses), FORWARD and BACKWARD (`jax.custom_vjp`), any N, pair channels CZ / intermediate channels C powers of two (128/128 Evoformer pair
stack, 64/64 template pair stack).

Stock, per call, on pair activations x (N,N,CZ) [bf16 under the multimer bf16 getter] with pair mask m (N,N):
    xn  = LayerNorm_left_norm_input(x)                                  f32 statistics, output rounded to x.dtype
    pg  = m * (xn @ Wproj + bproj) * sigmoid(xn @ Wgate + bgate)        (N,N,2C); a = [..., :C], b = [..., C:]
    t   = einsum(equation, a, b)                                        'ikc,jkc->ijc' (outgoing) | 'kjc,kic->ijc' (incoming)
    y   = (LayerNorm_center_norm(t) @ Wout + bout) * sigmoid(xn @ Wgating + bgating)
XLA runs this as ~15 kernels forward / ~45 in the remat'd backward, moving ~36 / ~120 × the bytes of one (N,N,CZ) tensor per call: the
einsum's operands and result are transposed through HBM to/from the channel-major layout cuBLAS needs, padded, and every elementwise stage
is its own pass. Here:
    forward   K1 prologue: LN + 4 projection/gate products + mask for a tile of T pixels, written DIRECTLY as channel-major planes
                 P, Q of shape (C, N·N) with t[c,i,j] = Σ_k P[c,i·N+k] Q[c,j·N+k] for BOTH equations (incoming: the tile is a run of plane
                 positions and the pixels are gathered transposed);
              EIN: ONE cuBLAS batched GEMM (XLA einsum 'cik,cjk->cij' on the planes viewed (C,Np,Np)) — no transposes, no pads: the planes
              are written ZERO-PADDED to Np = ceil8(N) per side (_pad8) because cuBLAS' bf16 GEMMs want multiples of 8 and XLA would
              otherwise copy every operand into a padded buffer and slice every result back (2+1 pair-tensor passes per GEMM; measured
              31 ms/step @500 tokens, nothing @800); K3/B1/B3 read T3/dP/dQ with the padded stride, the pixels stay (N·N, CZ);
              K3 epilogue: centre LN over c (channel-major [C,T] tile) + output projection + output gate (LN(x) recomputed) → y (N,N,CZ).
    backward  B1: from dy, x, T3 → dT3 (channel-major, for the GEMMs) and dxh = cotangent reaching xn through the output gate;
              two cuBLAS batched GEMMs: dP = dT3·Q, dQ = dT3ᵀ·P;
              B3: from x, m, dP, dQ, dxh: recompute the prologue, back through gates / projections / LN → dx.
              Residuals: x, mask, P, Q, T3. The mask and parameter cotangents are plain-jnp recomputations from the residuals that XLA
              removes as dead code unless a caller differentiates w.r.t. them (the design step differentiates the sequence only).
Every product inside the Pallas kernels is oriented so that tiles come out of the MMA in the layout they are stored in ([C,T] plane side,
[T,CZ] row side): transposes are shared-memory operand views, never register shuffles or HBM passes.
The padded layout is BITWISE equal (y and dx, every N, both equations) to the
previous, unpadded generation of these kernels: same per-pixel arithmetic, same cuBLAS shapes as XLA's own padding produced, and the T3-tile
accesses of K3/B1 keep the old buffers' provable alignment class (_acls / _t3_cols) so the compiler lays the channel-LayerNorm reductions
out — and orders their sums — exactly as before.

Numerics (class: precision — NOT bitwise vs stock): every product keeps stock's operand class (bf16 operands / f32 accumulation when x is
bf16; f32 operands at XLA-DEFAULT = tf32 class when x is f32); LayerNorm statistics in f32 as stock, sigmoids in f32 via tanh.approx (_sigmoid); tensors that reach HBM are
rounded to x.dtype where stock's are (the LN output feeding the MMAs, the einsum operands/result, y, dT, dxh, dx) while in-register intermediates
(Linear outputs, gates) stay f32 — fewer rounding points than stock, measured closer to an f32 reference. Differences vs stock are
bf16 re-association / rounding-point moves: measured in tests/test_trimul_pallas.py (vs stock and vs an f32-HIGHEST yardstick) and by the
step-check replays. Run-to-run bitwise (no atomics).

LayerNorm statistics inside the kernels use the centred two-pass variance (see _ln_rows) — a policy, not a knob: it is free on a
register-resident tile and never less accurate than stock's E[x²]−E[x]² (haiku use_fast_variance) form; the pure-jnp `reference` keeps
stock's form so it stays stock's math op by op.
"""
from __future__ import annotations

import functools
from typing import Dict, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

_CP = getattr(plgpu, "TritonCompilerParams", None) or getattr(plgpu, "CompilerParams")
_PL_LOAD, _PL_STORE = getattr(pl, "load", None), getattr(pl, "store", None)
F32 = jnp.float32


def _sigmoid(z):
    """sigmoid(z) = ½ + ½·tanh(z/2) through the GPU's tanh.approx.f32 (one MUFU op instead of exp + reciprocal; |rel err| ≲ 2⁻¹¹, an order
    below the bf16 resolution the gates are rounded to). The kernels are elementwise-throughput bound (profiled), so this is worth ~14 % of K1."""
    return 0.5 + 0.5 * plgpu.approx_tanh(0.5 * z)
LN_EPS = 1e-5
EQ_OUT, EQ_IN = "ikc,jkc->ijc", "kjc,kic->ijc"
EQUATIONS = (EQ_OUT, EQ_IN)
PRECISION_WORDS = {"bf16": "bf16", "f32": "tf32"}       # LEVER-line precision word per activation dtype (product operand class; f32 accumulate)

# launch configuration: t = pixels per program (a run of T consecutive positions of the flattened N·N pixel / plane index), w = num_warps.
DEFAULT_CFG = dict(t1=64, w1=4, t3=64, w3=4, tb1=64, wb1=4, tb3=64, wb3=8)
_A = slice(None)
_ZERO = np.zeros((1,), np.int32)                                   # a run-time zero (see _t3_cols)


def _load(ref, idx, mask=None, other=None):
    if _PL_LOAD is not None:
        return _PL_LOAD(ref, idx, mask=mask, other=other)
    return plgpu.load(ref.at[idx], mask=mask, other=other)


def _store(ref, idx, val, mask=None):
    if _PL_STORE is not None:
        return _PL_STORE(ref, idx, val, mask=mask)
    return plgpu.store(ref.at[idx], val, mask=mask)


def _mm(a, b, ca, cb):
    """Σ over a's axis `ca` and b's axis `cb`, f32 accumulation (bf16 operands → bf16 tensor-core products; f32 → XLA-DEFAULT class).
    Contracting a leading axis is a shared-memory operand view in the Triton lowering, not a data transpose."""
    return jax.lax.dot_general(a, b, (((ca,), (cb,)), ((), ())), preferred_element_type=F32)


def _ln_rows(x):
    """LayerNorm statistics over axis 1 of a [T, C] f32 tile → (xhat, rstd [T,1]). The variance is the CENTRED two-pass form E[(x-mean)²]
    (the tile is register-resident, so the second pass is free): never negative and free of the E[x²]−E[x]² cancellation of stock's
    use_fast_variance form — equal or closer to exact arithmetic on every row; same eps placement (rsqrt(var + 1e-5))."""
    mean = jnp.mean(x, axis=1, keepdims=True)
    xc = x - mean
    var = jnp.mean(xc * xc, axis=1, keepdims=True)
    rstd = jax.lax.rsqrt(var + LN_EPS)
    return xc * rstd, rstd


def _ln_cols(x):
    """The same over axis 0 of a channel-major [C, T] tile → (xhat, rstd [1,T])."""
    mean = jnp.mean(x, axis=0, keepdims=True)
    xc = x - mean
    var = jnp.mean(xc * xc, axis=0, keepdims=True)
    rstd = jax.lax.rsqrt(var + LN_EPS)
    return xc * rstd, rstd


def _pad8(n):
    """Plane edge: N rounded up to a multiple of 8 — cuBLAS' bf16 GEMMs take the (C, Np, Np) planes as they are (XLA pads any other size with
    two extra passes per operand and slices the result back)."""
    return -(-n // 8) * 8


def _tile(n, npad, t, transposed):
    """The T plane positions this program owns — flat g = i·Np + k over the zero-padded (Np, Np) plane —, which are inside the buffer (g < Np²),
    which are real (i, k < N), and the pixel row of x viewed (N·N, CZ) each maps to: (i, k) for the outgoing equation, the transposed pixel
    (k, i) for the incoming one (0 for padding positions; their loads are masked)."""
    g0 = pl.program_id(0) * t
    g = g0 + jnp.arange(t)
    inside = g < npad * npad
    if npad == n:                                                                           # no padding (N % 8 == 0): the pre-padding kernels' addressing —
        if not transposed:                                                                  # plane position == pixel: contiguous pixel tiles,
            return g0, inside, inside, pl.ds(g0, t)
        return g0, inside, inside, (g % n) * n + g // n                                     # or the transposed pixel, gathered
    i = g // npad
    k = g - i * npad
    valid = inside & (i < n) & (k < n)
    rows = (k * n + i) if transposed else (i * n + k)
    return g0, inside, valid, jnp.where(valid, rows, 0)


def _rows_load(ref, rows, valid):
    """rows: a pl.ds (contiguous pixels) or an index vector (gathered pixels) → the [T, CZ] tile of ref viewed (N·N, CZ)."""
    return _load(ref, (rows, _A), mask=valid[:, None], other=0.0)


def _acls(n):
    """Alignment class the compiler could prove for the einsum-result tile loads of the pre-padding kernels, whose planes were (C, N·N): the
    power of two dividing N² capped at a 16-byte vector — 3 (N % 4 == 0), 2 (N % 4 == 2) or 0 (N odd). The padded planes (C, Np²) are always
    16-byte aligned; K3/B1 re-impose the old class on their t3 tile loads (a column start of provable divisibility 2^class) so the channel
    LayerNorm's reduction runs in the same register layout, hence the same summation order, as before: y and dx stay BITWISE equal to the
    previous generation's kernels at every N (measured: without this, 1-ulp flips at ~1e-4 of the outputs when N % 4 != 0)."""
    v = 0
    while (n * n) % (2 ** (v + 1)) == 0 and v < 3:
        v += 1
    return v


def _t3_cols(g0, t, acls, z_ref):
    """Column slice of the (C, Np²) einsum-result tile starting at g0 (a multiple of t), with provable divisibility 2^acls (z_ref holds a run-time 0)."""
    if acls >= 3:
        return pl.ds(g0, t)
    z = jnp.sum(z_ref[...])                                                                  # 0, of unknown divisibility to the compiler
    if acls == 2:
        return pl.ds((pl.program_id(0) * (t // 4) + z) * 4, t)
    return pl.ds(g0 + z, t)


# =============================================================================================================== K1: forward prologue
def _k1_kernel(x_ref, m_ref, s_ref, o_ref, wpp_ref, bpp_ref, wgp_ref, bgp_ref, wpq_ref, bpq_ref, wgq_ref, bgq_ref, p_ref, q_ref, *, n, npad, t, transposed):
    g0, inside, valid, rows = _tile(n, npad, t, transposed)
    io = p_ref.dtype
    x = _rows_load(x_ref, rows, valid).astype(F32)                                          # [T, CZ]
    xhat, _ = _ln_rows(x)
    xn = (xhat * s_ref[...][None, :] + o_ref[...][None, :]).astype(io)                     # stock: LN output in x.dtype
    m = _load(m_ref, (rows,), mask=valid, other=0.0).astype(F32)[None, :]                   # [1, T]; 0 at padding positions → the planes' padding is 0
    for wp_ref, bp_ref, wg_ref, bg_ref, out_ref in ((wpp_ref, bpp_ref, wgp_ref, bgp_ref, p_ref), (wpq_ref, bpq_ref, wgq_ref, bgq_ref, q_ref)):
        gT = _mm(wg_ref[...], xn, 0, 1) + bg_ref[...].astype(F32)[:, None]                 # [C, T] = (xn @ Wg + bg)ᵀ
        sg = _sigmoid(gT)
        pT = (_mm(wp_ref[...], xn, 0, 1) + bp_ref[...].astype(F32)[:, None])
        _store(out_ref, (_A, pl.ds(g0, t)), ((m * pT) * sg).astype(io), mask=inside[None, :])


def prologue(xf, mf, n, s1, o1, wp_p, bp_p, wg_p, bg_p, wp_q, bp_q, wg_q, bg_q, *, transposed, t, num_warps):
    """xf (N·N, CZ) pixels, mf (N·N,) mask → planes P, Q (C, Np·Np) in xf.dtype, zero outside (N, N)."""
    NP, CZ = xf.shape
    C = wp_p.shape[1]
    G = _pad8(n) ** 2
    full = lambda *shape: pl.BlockSpec(tuple(shape), lambda i: (0,) * len(shape))
    plane = jax.ShapeDtypeStruct((C, G), xf.dtype)
    return pl.pallas_call(
        functools.partial(_k1_kernel, n=n, npad=_pad8(n), t=t, transposed=transposed), grid=(pl.cdiv(G, t),),
        in_specs=[full(NP, CZ), full(NP), full(CZ), full(CZ)] + [full(CZ, C), full(C)] * 4,
        out_specs=[full(C, G)] * 2, out_shape=[plane, plane],
        compiler_params=_CP(num_warps=num_warps), name="trimul_k1_prologue",
    )(xf, mf, s1, o1, wp_p, bp_p, wg_p, bg_p, wp_q, bp_q, wg_q, bg_q)


# =============================================================================================================== K3: forward epilogue
def _k3_kernel(t_ref, x_ref, s1_ref, o1_ref, s2_ref, o2_ref, wo_ref, bo_ref, wg_ref, bg_ref, z_ref, y_ref, *, n, npad, t, acls):
    g0, inside, valid, rows = _tile(n, npad, t, False)
    io = y_ref.dtype
    x = _rows_load(x_ref, rows, valid).astype(F32)                                          # [T, CZ]
    xhat, _ = _ln_rows(x)
    xn = (xhat * s1_ref[...][None, :] + o1_ref[...][None, :]).astype(io)                   # left_norm_input recomputed (never stored)
    sg = _sigmoid((_mm(xn, wg_ref[...], 1, 0) + bg_ref[...].astype(F32)[None, :]))   # [T, CZ]
    tt = _load(t_ref, (_A, _t3_cols(g0, t, acls, z_ref)), mask=inside[None, :], other=0.0).astype(F32)   # [C, T] channel-major einsum result
    that, _ = _ln_cols(tt)
    tnT = (that * s2_ref[...][:, None] + o2_ref[...][:, None]).astype(io)                  # [C, T]
    o = (_mm(tnT, wo_ref[...], 0, 0) + bo_ref[...].astype(F32)[None, :])   # [T, CZ] = tn @ Wo + bo
    _store(y_ref, (rows, _A), (o * sg).astype(io), mask=valid[:, None])


def epilogue(t3f, xf, n, s1, o1, s2, o2, wo, bo, wg, bg, *, t, num_warps):
    """t3f (C, Np·Np) + xf (N·N, CZ) → y (N·N, CZ)."""
    NP, CZ = xf.shape
    C, G = t3f.shape
    full = lambda *shape: pl.BlockSpec(tuple(shape), lambda i: (0,) * len(shape))
    return pl.pallas_call(
        functools.partial(_k3_kernel, n=n, npad=_pad8(n), t=t, acls=_acls(n)), grid=(pl.cdiv(G, t),),
        in_specs=[full(C, G), full(NP, CZ), full(CZ), full(CZ), full(C), full(C), full(C, CZ), full(CZ), full(CZ, CZ), full(CZ), full(1)],
        out_specs=full(NP, CZ), out_shape=jax.ShapeDtypeStruct((NP, CZ), xf.dtype),
        compiler_params=_CP(num_warps=num_warps), name="trimul_k3_epilogue",
    )(t3f, xf, s1, o1, s2, o2, wo, bo, wg, bg, _ZERO)


# =============================================================================================================== B1: backward epilogue
def _b1_kernel(dy_ref, t_ref, x_ref, s1_ref, o1_ref, s2_ref, o2_ref, wo_ref, bo_ref, wg_ref, bg_ref, z_ref, dt_ref, dxh_ref, *, n, npad, t, acls):
    """Channel-major: every tile is [channels, T]; the one register transpose is the dy tile."""
    g0, inside, valid, rows = _tile(n, npad, t, False)
    io = dt_ref.dtype
    # the output gate, recomputed: sgᵀ = sigmoid(Wgᵀ·xnᵀ + bg)   [CZ, T]
    x = _rows_load(x_ref, rows, valid).astype(F32)                                          # [T, CZ]
    xhat, _ = _ln_rows(x)
    xn = (xhat * s1_ref[...][None, :] + o1_ref[...][None, :]).astype(io)
    wg = wg_ref[...]
    sgT = _sigmoid((_mm(wg, xn, 0, 1) + bg_ref[...].astype(F32)[:, None]))    # [CZ, T]
    dyT = _rows_load(dy_ref, rows, valid).T.astype(F32)                                     # [CZ, T] (0 at padding positions → dT's padding is 0)
    d_oT = (dyT * sgT).astype(io)                                                           # cotangent of the output_projection output
    dsgT = dyT * sgT * (1.0 - sgT)                                                          # d_h / o
    # centre LN + output projection, recomputed: oᵀ = Woᵀ·tnᵀ + bo   [CZ, T]
    tt = _load(t_ref, (_A, _t3_cols(g0, t, acls, z_ref)), mask=inside[None, :], other=0.0).astype(F32)   # [C, T]
    that, rstd2 = _ln_cols(tt)
    s2 = s2_ref[...][:, None]
    tnT = (that * s2 + o2_ref[...][:, None]).astype(io)
    wo = wo_ref[...]                                                                        # [C, CZ]
    oT = (_mm(wo, tnT, 0, 0) + bo_ref[...].astype(F32)[:, None])    # [CZ, T]
    d_hT = (dsgT * oT).astype(io)                                                           # cotangent of the gating_linear output
    dxhT = _mm(wg, d_hT, 1, 0).astype(io)                                                   # (d_h @ Wgᵀ)ᵀ = Wg·d_hᵀ  [CZ, T]
    _store(dxh_ref, (rows, _A), dxhT.T, mask=valid[:, None])                                # stored pixel-major (N·N, CZ): B3 gathers pixel rows
    d_tnT = _mm(wo, d_oT, 1, 0)                                      # (d_o @ Woᵀ)ᵀ = Wo·d_oᵀ  [C, T] (stock: bf16 dot output)
    dxhat = d_tnT * s2
    d_tT = rstd2 * (dxhat - jnp.mean(dxhat, axis=0, keepdims=True) - that * jnp.mean(dxhat * that, axis=0, keepdims=True))
    _store(dt_ref, (_A, _t3_cols(g0, t, acls, z_ref)), d_tT.astype(io), mask=inside[None, :])   # same class as the load: the store anchors the LN-backward's layout


def bwd_epilogue(dyf, t3f, xf, n, s1, o1, s2, o2, wo, bo, wg, bg, *, t, num_warps):
    """→ dT3 (C, Np·Np) channel-major (zero outside (N, N)) and dxh (N·N, CZ) pixel-major (the cotangent reaching LN(x)'s output through the output gate)."""
    NP, CZ = xf.shape
    C, G = t3f.shape
    full = lambda *shape: pl.BlockSpec(tuple(shape), lambda i: (0,) * len(shape))
    return pl.pallas_call(
        functools.partial(_b1_kernel, n=n, npad=_pad8(n), t=t, acls=_acls(n)), grid=(pl.cdiv(G, t),),
        in_specs=[full(NP, CZ), full(C, G), full(NP, CZ), full(CZ), full(CZ), full(C), full(C), full(C, CZ), full(CZ), full(CZ, CZ), full(CZ), full(1)],
        out_specs=[full(C, G), full(NP, CZ)],
        out_shape=[jax.ShapeDtypeStruct((C, G), xf.dtype), jax.ShapeDtypeStruct((NP, CZ), xf.dtype)],
        compiler_params=_CP(num_warps=num_warps), name="trimul_b1_bwd_epilogue",
    )(dyf, t3f, xf, s1, o1, s2, o2, wo, bo, wg, bg, _ZERO)


# =============================================================================================================== B3: backward prologue
def _b3_kernel(x_ref, m_ref, dp_ref, dq_ref, dxh_ref, s_ref, o_ref, wpp_ref, bpp_ref, wgp_ref, bgp_ref, wpq_ref, bpq_ref, wgq_ref, bgq_ref,
               dx_ref, *, n, npad, t, transposed):
    """Channel-major: the x tile is transposed once on load ([CZ, T]); d(xn) accumulates as [CZ, T]; dx is transposed once on store."""
    g0, inside, valid, rows = _tile(n, npad, t, transposed)
    io = dx_ref.dtype
    xT = _rows_load(x_ref, rows, valid).T.astype(F32)                                       # [CZ, T]
    xhat, rstd1 = _ln_cols(xT)
    s1 = s_ref[...][:, None]
    xnT = (xhat * s1 + o_ref[...][:, None]).astype(io)                                      # [CZ, T]
    m = _load(m_ref, (rows,), mask=valid, other=0.0).astype(F32)[None, :]                   # [1, T]
    d_xnT = _rows_load(dxh_ref, rows, valid).T.astype(F32)                                  # [CZ, T] (the output-gate path, from B1; same pixels as x)
    for wp_ref, bp_ref, wg_ref, bg_ref, dv_ref in ((wpp_ref, bpp_ref, wgp_ref, bgp_ref, dp_ref), (wpq_ref, bpq_ref, wgq_ref, bgq_ref, dq_ref)):
        wg = wg_ref[...]                                                                    # [CZ, C]
        sg = _sigmoid((_mm(wg, xnT, 0, 0) + bg_ref[...].astype(F32)[:, None]))   # [C, T] = (xn@Wg+bg)ᵀ
        dv = _load(dv_ref, (_A, pl.ds(g0, t)), mask=inside[None, :], other=0.0).astype(F32)                  # [C, T]
        d_mp = dv * sg                                                                      # cotangent of (m · proj)
        wp = wp_ref[...]
        d_xnT = d_xnT + _mm(wp, (d_mp * m).astype(io), 1, 0)                               # (d_proj @ Wpᵀ)ᵀ = Wp·d_projᵀ  [CZ, T]
        pT = (_mm(wp, xnT, 0, 0) + bp_ref[...].astype(F32)[:, None])                  # [C, T] proj recomputed
        d_gate = (d_mp * (m * pT) * (1.0 - sg)).astype(io)                                 # = dv · v · (1 - sg)
        d_xnT = d_xnT + _mm(wg, d_gate, 1, 0)                                               # Wg·d_gateᵀ
    d_xnT = d_xnT                                                   # stock: bf16 sum of bf16 dot outputs
    dxhat = d_xnT * s1
    d_xT = rstd1 * (dxhat - jnp.mean(dxhat, axis=0, keepdims=True) - xhat * jnp.mean(dxhat * xhat, axis=0, keepdims=True))
    _store(dx_ref, (rows, _A), d_xT.astype(io).T, mask=valid[:, None])


def bwd_prologue(xf, mf, n, dPf, dQf, dxhf, s1, o1, wp_p, bp_p, wg_p, bg_p, wp_q, bp_q, wg_q, bg_q, *, transposed, t, num_warps):
    """→ dx (N·N, CZ)."""
    NP, CZ = xf.shape
    C, G = dPf.shape
    full = lambda *shape: pl.BlockSpec(tuple(shape), lambda i: (0,) * len(shape))
    return pl.pallas_call(
        functools.partial(_b3_kernel, n=n, npad=_pad8(n), t=t, transposed=transposed), grid=(pl.cdiv(G, t),),
        in_specs=[full(NP, CZ), full(NP), full(C, G), full(C, G), full(NP, CZ), full(CZ), full(CZ)] + [full(CZ, C), full(C)] * 4,
        out_specs=full(NP, CZ), out_shape=jax.ShapeDtypeStruct((NP, CZ), xf.dtype),
        compiler_params=_CP(num_warps=num_warps), name="trimul_b3_bwd_prologue",
    )(xf, mf, dPf, dQf, dxhf, s1, o1, wp_p, bp_p, wg_p, bg_p, wp_q, bp_q, wg_q, bg_q)


# =============================================================================================================== parameters
PARAM_KEYS = ("left_norm_input", "projection", "gate", "center_norm", "output_projection", "gating_linear")


def split_params(p: Dict, equation: str, dt):
    """Stock parameter tree (module-relative names) → kernel operands. Linear weights/biases in the activation dtype (that IS the model under
    the bf16 getter); LayerNorm scale/offset f32. Plane 'p' = the operand indexed by the OUTPUT ROW i: outgoing → left (columns :C),
    incoming → right (columns C:); plane 'q' the other."""
    C = p["projection"]["weights"].shape[1] // 2
    W, B, G, Gb = p["projection"]["weights"], p["projection"]["bias"], p["gate"]["weights"], p["gate"]["bias"]
    left = (W[:, :C].astype(dt), B[:C].astype(dt), G[:, :C].astype(dt), Gb[:C].astype(dt))
    right = (W[:, C:].astype(dt), B[C:].astype(dt), G[:, C:].astype(dt), Gb[C:].astype(dt))
    pp, qq = (left, right) if equation == EQ_OUT else (right, left)
    return dict(s1=p["left_norm_input"]["scale"].astype(F32), o1=p["left_norm_input"]["offset"].astype(F32), p=pp, q=qq,
                s2=p["center_norm"]["scale"].astype(F32), o2=p["center_norm"]["offset"].astype(F32),
                wo=p["output_projection"]["weights"].astype(dt), bo=p["output_projection"]["bias"].astype(dt),
                wg=p["gating_linear"]["weights"].astype(dt), bg=p["gating_linear"]["bias"].astype(dt))


# =============================================================================================================== the op
def _forward(x, mask, w, *, transposed, cfg):
    """→ y and the residuals (P, Q planes, T3)."""
    N, _, CZ = x.shape
    xf, mf = x.reshape(N * N, CZ), mask.reshape(N * N)
    Pf, Qf = prologue(xf, mf, N, w["s1"], w["o1"], *w["p"], *w["q"], transposed=transposed, t=cfg["t1"], num_warps=cfg["w1"])
    C, M = Pf.shape[0], _pad8(N)
    t3 = jnp.einsum("cik,cjk->cij", Pf.reshape(C, M, M), Qf.reshape(C, M, M))               # ONE cuBLAS batched GEMM on the padded planes, x.dtype out (f32 accumulate)
    yf = epilogue(t3.reshape(C, M * M), xf, N, w["s1"], w["o1"], w["s2"], w["o2"], w["wo"], w["bo"], w["wg"], w["bg"], t=cfg["t3"], num_warps=cfg["w3"])
    return yf.reshape(N, N, CZ), (Pf, Qf, t3)


def _backward(x, mask, w, Pf, Qf, t3, dy, *, transposed, cfg):
    """dy = the module-output cotangent → (dx, dP, dQ)."""
    N, _, CZ = x.shape
    C, M = Pf.shape[0], _pad8(N)
    xf, mf = x.reshape(N * N, CZ), mask.reshape(N * N)
    dyf = dy.reshape(N * N, CZ)
    dTf, dxhf = bwd_epilogue(dyf, t3.reshape(C, M * M), xf, N, w["s1"], w["o1"], w["s2"], w["o2"], w["wo"], w["bo"], w["wg"], w["bg"],
                             t=cfg["tb1"], num_warps=cfg["wb1"])
    dT = dTf.reshape(C, M, M)
    dP = jnp.einsum("cij,cjk->cik", dT, Qf.reshape(C, M, M))
    dQ = jnp.einsum("cij,cik->cjk", dT, Pf.reshape(C, M, M))
    dxf = bwd_prologue(xf, mf, N, dP.reshape(C, M * M), dQ.reshape(C, M * M), dxhf, w["s1"], w["o1"], *w["p"], *w["q"],
                       transposed=transposed, t=cfg["tb3"], num_warps=cfg["wb3"])
    return dxf.reshape(N, N, CZ), dP, dQ


def _mask_cotangent(x, mask, w, dP, dQ, *, transposed):
    """d/d(mask) by plain-jnp recomputation of the prologue (dead code — removed by XLA — unless the caller differentiates w.r.t. the mask;
    the design step never does: the pair mask is a constant feature)."""
    N = x.shape[0]
    dt = x.dtype
    x32 = x.astype(F32)
    mean = jnp.mean(x32, -1, keepdims=True)
    var = jnp.mean(x32 * x32, -1, keepdims=True) - jnp.square(mean)
    xn = ((x32 - mean) * jax.lax.rsqrt(var + LN_EPS) * w["s1"] + w["o1"]).astype(dt)
    dm = jnp.zeros((N, N), F32)
    for (wp, bp, wg, bg), dV in ((w["p"], dP), (w["q"], dQ)):
        proj = (jnp.einsum("ikz,zc->ikc", xn, wp, preferred_element_type=F32) + bp.astype(F32)).astype(dt).astype(F32)
        sg = jax.nn.sigmoid((jnp.einsum("ikz,zc->ikc", xn, wg, preferred_element_type=F32) + bg.astype(F32)).astype(dt).astype(F32))
        dv = dV[:, :N, :N].astype(F32).transpose(1, 2, 0)                                  # [c, i, k] (padding dropped) → pixel-major (i, k, c) of the PLANE position
        if transposed:
            dv = dv.transpose(1, 0, 2)                                                      # plane position (i,k) is pixel (k,i)
        dm = dm + jnp.sum(dv * sg * proj, axis=-1)
    return dm


def _param_cotangents(x, mask, p, equation, dy):
    """Parameter cotangents by plain-jnp recomputation (dead code — removed by XLA — unless a caller differentiates w.r.t. the parameters)."""
    _, vjp = jax.vjp(lambda pp: reference(x, mask, pp, equation=equation), p)
    return vjp(dy.astype(x.dtype))[0]


@functools.lru_cache(maxsize=None)
def make_triangle_multiplication(equation: str, cfg_items: Optional[tuple] = None):
    """The custom_vjp op for one equation and launch configuration: op(x, mask, params) → y."""
    if equation not in EQUATIONS:
        raise ValueError(f"equation {equation!r} is not one of {EQUATIONS}")
    cfg = {**DEFAULT_CFG, **dict(cfg_items or ())}
    transposed = equation == EQ_IN

    @jax.custom_vjp
    def op(x, mask, params):
        w = split_params(params, equation, x.dtype)
        return _forward(x, mask.astype(x.dtype), w, transposed=transposed, cfg=cfg)[0]

    def op_fwd(x, mask, params):
        w = split_params(params, equation, x.dtype)
        y, (Pf, Qf, t3) = _forward(x, mask.astype(x.dtype), w, transposed=transposed, cfg=cfg)
        return y, (x, mask, params, Pf, Qf, t3)

    def op_bwd(res, dy):
        x, mask, params, Pf, Qf, t3 = res
        w = split_params(params, equation, x.dtype)
        dy = dy.astype(x.dtype)
        dx, dP, dQ = _backward(x, mask.astype(x.dtype), w, Pf, Qf, t3, dy, transposed=transposed, cfg=cfg)
        if jnp.issubdtype(mask.dtype, jnp.floating):
            dmask = _mask_cotangent(x, mask, w, dP, dQ, transposed=transposed).astype(mask.dtype)
        else:                                                                               # integer / bool masks: the structural zero
            dmask = np.zeros(mask.shape, dtype=jax.float0)
        return dx, dmask, _param_cotangents(x, mask, params, equation, dy)

    op.defvjp(op_fwd, op_bwd)
    return op


def triangle_multiplication(x, mask, params, *, equation, cfg: Optional[dict] = None):
    """x (N,N,CZ) pair activations, mask (N,N) pair mask, params = the stock module's parameter tree (module-relative:
    left_norm_input/{scale,offset}, projection/{weights,bias}, gate/…, center_norm/…, output_projection/…, gating_linear/…)."""
    op = make_triangle_multiplication(equation, tuple(sorted((cfg or {}).items())))
    return op(x, mask, params)


# =============================================================================================================== pure-jnp reference
def reference(x, mask, p, *, equation, compute_dtype=None, precision=None):
    """Op-by-op replica of stock `_fused_triangle_multiplication` (no haiku). compute_dtype=None: stock's dtypes (x.dtype activations, f32
    LayerNorm statistics); compute_dtype=f32 + precision=HIGHEST: the accurate yardstick both stock and the kernels are measured against."""
    dt = x.dtype
    cd = compute_dtype or dt

    def ln(v, s, o):
        v32 = v.astype(F32)
        mean = jnp.mean(v32, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(v32), axis=-1, keepdims=True) - jnp.square(mean)
        return ((v32 - mean) * jax.lax.rsqrt(var + LN_EPS) * s.astype(F32) + o.astype(F32)).astype(cd)

    def linear(v, name):
        wgt = p[name]["weights"].astype(dt).astype(cd); b = p[name]["bias"].astype(dt).astype(cd)
        return jnp.einsum("...c,cd->...d", v, wgt, precision=precision) + b
    C = p["projection"]["weights"].shape[1] // 2
    m = mask.astype(cd)[..., None]
    left_act = ln(x.astype(cd), p["left_norm_input"]["scale"], p["left_norm_input"]["offset"])
    proj_act = m * linear(left_act, "projection")
    proj_act = proj_act * jax.nn.sigmoid(linear(left_act, "gate"))
    left, right = proj_act[:, :, :C], proj_act[:, :, C:]
    act = jnp.einsum(equation, left, right, precision=precision)
    act = ln(act, p["center_norm"]["scale"], p["center_norm"]["offset"])
    act = linear(act, "output_projection")
    gate = linear(left_act, "gating_linear")
    return (act * jax.nn.sigmoid(gate)).astype(cd)
