"""triatt_attn.py — lever `triatt`: flash attention with per-head pair bias and key mask for EVERY haiku `Attention` call of the AF-Multimer
design model (TriangleAttention starting/ending node, MSARowAttentionWithPairBias, MSAColumnAttention, the template pair stack's triangle
attention, the extra-MSA row attention), FORWARD + BACKWARD (jax.custom_vjp), Pallas (Triton lowering), written for H100 at AF2's shapes:
few heads (4-8), small head dim (8/16/32), many batch rows (B = N_res) that SHARE one pair bias [H, S_q, S_k].

Design (D=32: per S^2 element the two products cost 128 FLOP while the softmax costs one exp + ~6 fp32 ops — issue/MUFU/L2-bound,
not tensor-bound, so the kernel spends its effort on instructions per logit and bytes per logit):
  * one XLA prep pass per call pads the key stride of the pair bias (kept in the 16-bit input dtype) and writes the key mask as an fp32
    additive row [B, S_kp] (0 | NEG) plus a per-row `dead` flag; in-kernel the bias tile enters through the tensor core — dot(I, bias_tile),
    exact, already in the MMA register layout — so the logits cost no layout conversion and one fused multiply-add each: p = exp2(s*log2e - m*log2e).
  * q * key_dim**-0.5 is rounded to bf16 once in XLA (stock's own rounding point, modules.py); the kernels are scale-free.
  * G batch rows per program share each bias tile (bias traffic / G) and, in the backward, sum their dS tiles in registers: d(pair bias) =
    sum over B of dS is written as ceil(B/G) fp32 partials [ceil(B/G), H, S_q, S_kp], reduced over groups by XLA (not B per-row partials).
  * no per-logit bounds predicates in the steady state: the ragged last key block is a peeled iteration; ragged query blocks and batch
    groups are CLAMPED (overlapping) blocks recomputing identical rows (a clamped group zero-weights rows the previous group owns in dbias).
  * key mask: a call whose mask is all-true (the Evoformer's, in the binder-design protocol) runs the mask-free kernels — chosen per call at
    run time by an XLA conditional on all(mask); identical arithmetic for such calls, one add + one load per logit fewer.
  * backward = two kernels (kernels/attbwd_dkdv.py — TRIMUL's re-cut of this module's original pair), no atomics, run-to-run bitwise:
    K1' (grid over key blocks) forms dK, dV; K2' (grid over query blocks) forms dQ and the dS partial summed over G=4 batch rows.
    delta = rowsum(O*dO) and the group reduction of d(bias) are small XLA ops.
  * S_q or S_k below 16 (MSA column attention over N_seq = 2 sequences): plain XLA ops (a [B,H,2,2] softmax is not kernel work) — counted
    as served-small in the op's census, same numerics class.
Numerics (class `fast`, Tier 2 — a different kernel than stock's XLA ops, never bitwise): bf16 (or f16/f32) inputs, fp32 products/accumulation
and fp32 softmax statistics, P rounded to the input dtype before P.V (and dS before dS^T.Q / dS.K), exact (not approximate) online softmax.
float32 inputs: products at `f32_precision` (default 'tf32' = the class of XLA's DEFAULT-precision f32 dot on sm_80+, i.e. stock's; 'ieee').
Head dims below 16 (extra-MSA row attention, 8 per head) are zero-padded to 16 columns: exact (zero products add exact zeros).

Public API (layout heads-major, the layout opt_core's pallas_attn_serve feeds an `op`):
    attention(q, k, v, bias, key_mask, scale)            q,k,v [B,H,S,D]; bias [H,S_q,S_k]; key_mask [B,S_k] bool -> [B,H,S_q,D] (q's dtype)
    make_attention(**knobs) -> such an op                knobs: see DEFAULTS
    reference_attention(q, k, v, bias, key_mask, scale)  pure-JAX (XLA) reference in fp32 math (for tests)
    describe_op(op) -> dict                              the knobs an op was built with (the kit's LEVER-line evidence: precision=, tiles)
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

_CP = getattr(plgpu, "TritonCompilerParams", None) or getattr(plgpu, "CompilerParams")
LOG2E = 1.4426950408889634
NEG = -1e30                      # masked-key logit in log2 units (finite: a fully masked row degrades to stock's uniform average, never NaN)
MIN_HEAD_DIM = 16                # dot operands need K >= 16: smaller head dims are zero-padded to it (exact)
MIN_SEQ = 16                     # below this many queries or keys the call is plain XLA ops (MSA column attention over 2 sequences)
KPAD = 128                       # key padding of the bias / mask / dbias-partial rows: aligned rows AND the peeled ragged key block reads a full tile in range
F32_PRECISIONS = ("tf32", "ieee")
_F32_PREC = {"ieee": jax.lax.Precision.HIGHEST, "tf32": jax.lax.Precision.DEFAULT}
DEFAULTS = dict(bq=64, bk=32, G=1, num_warps=4, num_stages=3,                        # forward: query tile, key tile, batch rows per program
                f32_precision="tf32", grid_order="ihb", mask_free="auto", dbias_partials="input")
# The backward is kernels/attbwd_dkdv.py (TRIMUL's re-cut of this module's original K1/K2: K1' = dK/dV, grid over key blocks; K2' = dQ plus
# the d(pair-bias) partials summed over G=4 batch rows, grid over query blocks; its tiles live in attbwd_dkdv.DEFAULTS_BWD).
# Scale placement (fixed): the op computes bf16(q * key_dim**-0.5) once in XLA — stock's own rounding point
# (modules.py multiplies the bf16 q by the scale) and the placement class of a caller that folds the scale into its q projection and passes
# scale=1.0 — and the kernels are scale-free. The measured alternative (the scale multiplying the fp32 logits in-kernel, F1's placement) has a
# strictly smaller forward error against an fp32 reference but, measured against stock's reference design states, lands on the other side of the
# direction coin at 1gpbA_N600@75 / pdl1_hotspot@25; it is not selectable code here
#
# The pair bias enters the kernels through the tensor core: dot(I, bias_tile) — exact for a 16-bit bias, and it lands in the MMA register
# layout with no shared-memory layout conversion (the measured alternative, an fp32 tile added on the fp32 pipe, is slower).
# dbias_partials: dtype of the per-group dS partial sums ('input' = the bias/q dtype, what F1 writes per ROW; 'f32').
# grid_order: roles (i = q/k block, h = head, b = batch group) by grid axis, first letter = axis 0 (fastest-varying at launch).
# mask_free: 'auto' = run the mask-free kernels when all(key_mask) (XLA conditional per call); 'never' = always the masked kernels.

_PL_LOAD, _PL_STORE = getattr(pl, "load", None), getattr(pl, "store", None)


def _load(ref, idx, mask=None, other=None):
    if mask is None:
        other = None
    if _PL_LOAD is not None:
        try:
            return _PL_LOAD(ref, idx, mask=mask, other=other)
        except TypeError:
            pass
    return plgpu.load(ref.at[idx], mask=mask, other=other)


def _store(ref, idx, val, mask=None):
    if _PL_STORE is not None:
        try:
            return _PL_STORE(ref, idx, val, mask=mask)
        except TypeError:
            pass
    return plgpu.store(ref.at[idx], val, mask=mask)


def _dot(a, b, f32p):
    """a @ b with fp32 accumulation: bf16/f16 operands on tensor cores; f32 operands at the precision word."""
    if a.dtype == jnp.float32:
        return jnp.dot(a, b, preferred_element_type=jnp.float32, precision=_F32_PREC[f32p])
    return jnp.dot(a, b, preferred_element_type=jnp.float32)


def _scaled(x, scale):
    """x * scale rounded to x's dtype — no op at all when scale == 1.0 (the op pre-scales q once in XLA, fused into q's producer)."""
    return x if scale == 1.0 else (x.astype(jnp.float32) * scale).astype(x.dtype)


def _dyn(x: int):
    """A traced int32 equal to the static x: Pallas refuses STATIC slice starts beyond the ref bounds, and the peeled ragged block
    deliberately starts a full-width (masked) tile inside the last partial block."""
    return pl.program_id(0) * 0 + x


def _pids(order):
    return {c: pl.program_id(n) for n, c in enumerate(order)}


def _grid(order, nblk, H, ngrp):
    size = {"i": nblk, "h": H, "b": ngrp}
    return tuple(size[c] for c in order)


def _full(x):
    return pl.BlockSpec(tuple(x.shape), lambda *_: (0,) * x.ndim)


# =============================================================================================================== forward
def _fwd_kernel(q_ref, k_ref, v_ref, b_ref, m_ref, d_ref, o_ref, lse_ref, *, bq, bk, G, sq, sk, nb, D, order, f32p, has_mask):
    """One program = G batch rows x one head x bq query rows. Whole-array refs: q,k,v,o [B,H,S,D]; b [H,Sq,Skp] f32 (bias*log2e);
    m [B,Skp] f32 additive (0 | NEG); lse [B,H,Sq] f32 (log2 units: m2 + log2(l))."""
    pid = _pids(order)
    i, h, bg = pid["i"], pid["h"], pid["b"]
    if sq >= bq:
        r0 = jnp.minimum(i * bq, sq - bq); rvec = rmask = None             # clamped block: no row predicates
    else:
        r0 = 0; rvec = jnp.arange(bq) < sq; rmask = rvec[:, None]
    b0 = jnp.minimum(bg * G, nb - G)                                        # clamped group (nb >= G by construction)
    rows = pl.dslice(r0, bq)
    nfull, rem = sk // bk, sk % bk
    qs = [_load(q_ref, (b0 + g, h, rows, slice(None)), mask=rmask, other=0.0) for g in range(G)]
    live = [1.0 - _load(d_ref, (b0 + g,)).astype(jnp.float32) for g in range(G)] if has_mask else None   # scalar per row: 0.0 when every key is masked
    # logits in natural units: s = dot(q, k^T) + dot(I, bias) (q arrives pre-scaled); p = exp2(s*log2e - m*log2e), one FFMA per logit
    eye = (jax.lax.broadcasted_iota(jnp.int32, (bq, bq), 0) == jax.lax.broadcasted_iota(jnp.int32, (bq, bq), 1)).astype(b_ref.dtype)
    e2 = LOG2E

    def kblock(c0, carry, kin):
        cols = pl.dslice(c0, bk)
        bias = _dot(eye, _load(b_ref, (h, rows, cols), mask=rmask, other=0.0), f32p)       # [bq,bk] fp32 in the MMA layout, exact
        out = []
        for g in range(G):
            acc, m_i, l_i = carry[g]
            kl = None if kin is None else kin[:, None]
            k = _load(k_ref, (b0 + g, h, cols, slice(None)), mask=kl, other=0.0)
            v = _load(v_ref, (b0 + g, h, cols, slice(None)), mask=kl, other=0.0)
            s = _dot(qs[g], k.T, f32p) + bias
            if has_mask:
                s = (s + _load(m_ref, (b0 + g, cols))[None, :]) * live[g]        # a dead row (every key masked): all logits 0 = uniform, as stock
                if kin is not None:
                    s = jnp.where(kin[None, :], s, NEG)                        # (padded columns of a dead row must still not count)
            elif kin is not None:
                s = jnp.where(kin[None, :], s, NEG)
            m_new = jnp.maximum(m_i, jnp.max(s, axis=1))                       # running max in the logits' units
            alpha = jnp.exp2((m_i - m_new) * e2); p = jnp.exp2(s * e2 - (m_new * e2)[:, None])       # one FFMA per logit
            if kin is not None:
                p = jnp.where(kin[None, :], p, 0.0)                            # out-of-range keys never count (even for a fully masked row)
            l_new = l_i * alpha + jnp.sum(p, axis=1)
            acc = acc * alpha[:, None] + _dot(p.astype(v.dtype), v, f32p)
            out.append((acc, m_new, l_new))
        return tuple(out)

    carry = tuple((jnp.zeros((bq, D), jnp.float32), jnp.full((bq,), -jnp.inf, jnp.float32), jnp.zeros((bq,), jnp.float32)) for _ in range(G))
    if nfull > 0:
        carry = jax.lax.fori_loop(0, nfull, lambda j, c: kblock(j * bk, c, None), carry)
    if rem:
        carry = kblock(_dyn(nfull * bk), carry, jnp.arange(bk) < rem)
    for g in range(G):
        acc, m_i, l_i = carry[g]
        _store(o_ref, (b0 + g, h, rows, slice(None)), (acc / l_i[:, None]).astype(o_ref.dtype), mask=rmask)
        _store(lse_ref, (b0 + g, h, rows), m_i * e2 + jnp.log(l_i) * LOG2E, mask=rvec)     # log2-units logsumexp


MIN_PROGRAMS = 528             # 4 programs per H100 SM: below this a call cannot fill the card (stock's subbatch-4 chunks: B = 4)


def _tiles(Sq, Sk, B, H, bq, bk, G, grid_axis="q"):
    """Clamp the tiles to the problem, then — for calls too small to fill the card (few batch rows: stock's subbatch-4 chunks) — trade rows
    per program for programs (G -> 1). Tiles stay: below 64 rows the products leave the wgmma path (measured slower even at 4x the programs)."""
    G = max(1, min(G, B))
    if Sq < bq:
        bq = max(16, pl.next_power_of_2(Sq))
    if Sk < bk:
        bk = max(16, pl.next_power_of_2(Sk))
    n_axis = Sq if grid_axis == "q" else Sk

    def nprog(t, g):
        return pl.cdiv(n_axis, t) * H * pl.cdiv(B, g)
    t = bq if grid_axis == "q" else bk
    while nprog(t, G) < MIN_PROGRAMS and G > 1:
        G //= 2
    return bq, bk, G


def _fwd(q, k, v, bias2, madd, dead, *, bq, bk, G, num_warps, num_stages, order, f32p, has_mask):
    B, H, Sq, D = q.shape; Sk = k.shape[2]
    bq, bk, G = _tiles(Sq, Sk, B, H, bq, bk, G, "q")
    kern = functools.partial(_fwd_kernel, bq=bq, bk=bk, G=G, sq=Sq, sk=Sk, nb=B, D=D, order=order, f32p=f32p, has_mask=has_mask)
    return pl.pallas_call(
        kern, grid=_grid(order, pl.cdiv(Sq, bq), H, pl.cdiv(B, G)),
        in_specs=[_full(q), _full(k), _full(v), _full(bias2), _full(madd), _full(dead)],
        out_specs=[pl.BlockSpec((B, H, Sq, D), lambda *_: (0, 0, 0, 0)), pl.BlockSpec((B, H, Sq), lambda *_: (0, 0, 0))],
        out_shape=[jax.ShapeDtypeStruct((B, H, Sq, D), q.dtype), jax.ShapeDtypeStruct((B, H, Sq), jnp.float32)],
        compiler_params=_CP(num_warps=num_warps, num_stages=num_stages),
        name=f"triatt_fwd_{'m' if has_mask else 'u'}_G{G}_bq{bq}_bk{bk}_w{num_warps}_s{num_stages}",
    )(q, k, v, bias2, madd, dead)


# =============================================================================================================== backward
def _bwd(q, k, v, bias2, madd, dead, o, lse, do, *, order, f32p, has_mask, pdt):
    """(dq, dk, dv, d bias fp32 [H, Sq, Sk]) by kernels/attbwd_dkdv.py: K1' (grid over key blocks: dK, dV) and K2' (grid over query blocks:
    dQ and the d(pair-bias) partials summed over G batch rows [ceil(B/G), H, Sq, Skp] in `pdt`, reduced over groups by XLA in fp32);
    delta = rowsum(o*do) is a small XLA op. Deterministic (no atomics). Same operands and mask/dead-row semantics as the forward."""
    from . import attbwd_dkdv as A                                          # imports this module's helpers at its import: bind lazily
    return A._bwd(q, k, v, bias2, madd, dead, o, lse, do, scale=1.0, k1=dict(A.DEFAULTS_BWD["k1"]), k2=dict(A.DEFAULTS_BWD["k2"]), order=order,
                  f32p=f32p, has_mask=has_mask, bias_mma=True, pdt=pdt, resident=int(A.DEFAULTS_BWD.get("resident", 1)))


# =============================================================================================================== the op
def _prep(bias, key_mask, Sk, cdt):
    """The per-call XLA prep pass: the bias in the compute dtype `cdt` (exact when it already is; an f32 bias with 16-bit q is rounded to it)
    with the key stride padded to KPAD [H,Sq,Skp] — it enters the kernels through the tensor core in natural units.
    The key mask becomes an fp32 additive row [B,Skp] (0 | NEG; padded columns NEG) plus a per-row flag `dead` [B] (every key masked): such a
    row is served as stock serves it — its bf16 logits all absorb into the mask's -1e9 and ColabDesign's clip(-1e8, 1e8) pins them: the
    softmax is uniform over every key (o = mean of v, dV = P^T dO at uniform P) and the clipped logits pass NO gradient (dQ = dK = d bias = 0
    from that row). The kernels give such a row zero logits on real keys and zero dS."""
    Skp = -(-Sk // KPAD) * KPAD
    bias2 = jnp.pad(bias.astype(cdt), ((0, 0), (0, 0), (0, Skp - Sk)))
    madd = jnp.pad(jnp.where(key_mask, 0.0, NEG).astype(jnp.float32), ((0, 0), (0, Skp - Sk)), constant_values=NEG)
    dead = jnp.logical_not(jnp.any(key_mask, axis=-1)).astype(jnp.int32)                # [B]: rows with EVERY key masked (stock: uniform average)
    return bias2, madd, dead


def _small_attention(q, k, v, bias, key_mask, scale):
    """S_q or S_k below MIN_SEQ: plain XLA ops in fp32 (the MSA column attention over N_seq sequences)."""
    logits = jnp.einsum("bhqd,bhkd->bhqk", q.astype(jnp.float32) * scale, k.astype(jnp.float32)) + bias.astype(jnp.float32)[None]
    logits = jnp.where(key_mask[:, None, None, :], logits, NEG)
    w = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", w.astype(v.dtype), v, preferred_element_type=jnp.float32).astype(q.dtype)


def make_attention(**knobs):
    """Build op(q, k, v, bias, key_mask, scale) -> o with the given knobs (DEFAULTS for the rest). Unknown knobs or words raise."""
    cfg = dict(DEFAULTS)
    for kname, val in knobs.items():
        if kname not in cfg:
            raise ValueError(f"triatt_attn: unknown knob {kname!r} (known: {sorted(cfg)})")
        cfg[kname] = val
    if cfg["f32_precision"] not in F32_PRECISIONS:
        raise ValueError(f"triatt_attn: f32_precision={cfg['f32_precision']!r} not in {F32_PRECISIONS}")
    if sorted(cfg["grid_order"]) != ["b", "h", "i"]:
        raise ValueError(f"triatt_attn: grid_order={cfg['grid_order']!r} must be a permutation of 'ihb'")
    if cfg["mask_free"] not in ("auto", "never"):
        raise ValueError(f"triatt_attn: mask_free={cfg['mask_free']!r} not in ('auto', 'never')")
    fk = dict(bq=int(cfg["bq"]), bk=int(cfg["bk"]), G=int(cfg["G"]), num_warps=int(cfg["num_warps"]), num_stages=int(cfg["num_stages"]),
              order=cfg["grid_order"], f32p=cfg["f32_precision"])
    bk_ = dict(order=cfg["grid_order"], f32p=cfg["f32_precision"])
    if cfg["dbias_partials"] not in ("input", "f32"):
        raise ValueError(f"triatt_attn: dbias_partials={cfg['dbias_partials']!r} not in ('input', 'f32')")
    auto = cfg["mask_free"] == "auto"

    def fwd_both(q, k, v, bias2, madd, dead, unmasked):
        if not auto:
            return _fwd(q, k, v, bias2, madd, dead, has_mask=True, **fk)
        return jax.lax.cond(unmasked, lambda: _fwd(q, k, v, bias2, madd, dead, has_mask=False, **fk),
                            lambda: _fwd(q, k, v, bias2, madd, dead, has_mask=True, **fk))

    def bwd_both(q, k, v, bias2, madd, dead, o, lse, do, unmasked, pdt):
        if not auto:
            return _bwd(q, k, v, bias2, madd, dead, o, lse, do, has_mask=True, pdt=pdt, **bk_)
        return jax.lax.cond(unmasked, lambda: _bwd(q, k, v, bias2, madd, dead, o, lse, do, has_mask=False, pdt=pdt, **bk_),
                            lambda: _bwd(q, k, v, bias2, madd, dead, o, lse, do, has_mask=True, pdt=pdt, **bk_))

    @jax.custom_vjp
    def attn(q, k, v, bias, key_mask):
        biasf, madd, dead = _prep(bias, key_mask, k.shape[2], q.dtype)
        return fwd_both(q, k, v, biasf, madd, dead, jnp.all(key_mask))[0]

    def attn_fwd(q, k, v, bias, key_mask):
        biasf, madd, dead = _prep(bias, key_mask, k.shape[2], q.dtype)
        unmasked = jnp.all(key_mask)
        o, lse = fwd_both(q, k, v, biasf, madd, dead, unmasked)
        return o, (q, k, v, biasf, madd, dead, unmasked, o, lse, jnp.zeros((0,), bias.dtype))

    def attn_bwd(res, do):
        q, k, v, biasf, madd, dead, unmasked, o, lse, bdt = res
        pdt = jnp.float32 if (cfg["dbias_partials"] == "f32" or q.dtype == jnp.float32) else q.dtype
        dq, dk, dv, dbias = bwd_both(q, k, v, biasf, madd, dead, o, lse, do, unmasked, pdt)
        return dq, dk, dv, dbias.astype(bdt.dtype), None

    attn.defvjp(attn_fwd, attn_bwd)

    def op(q, k, v, bias, key_mask, scale):
        """q,k,v [B,H,S,D]; bias [H,Sq,Sk]; key_mask [B,Sk] bool; scale float -> o [B,H,Sq,Dv] in q's dtype."""
        Sq, Sk, D, Dv = int(q.shape[2]), int(k.shape[2]), int(q.shape[-1]), int(v.shape[-1])
        if Sq < MIN_SEQ or Sk < MIN_SEQ:
            return _small_attention(q, k, v, bias, key_mask, float(scale))
        if D < MIN_HEAD_DIM or Dv < MIN_HEAD_DIM:
            padq = [(0, 0)] * 3 + [(0, MIN_HEAD_DIM - D)]; padv = [(0, 0)] * 3 + [(0, MIN_HEAD_DIM - Dv)]
            q, k, v = jnp.pad(q, padq), jnp.pad(k, padq), jnp.pad(v, padv)
        if float(scale) != 1.0:                                                # stock's rounding point: bf16(q * key_dim**-0.5) once (XLA fuses it into q's
            q = (q.astype(jnp.float32) * float(scale)).astype(q.dtype)          # producer); scale = 1.0 => no multiply; the kernels are scale-free
        out = attn(q, k, v, bias, key_mask)
        return out if int(out.shape[-1]) == Dv else out[..., :Dv]

    op.knobs = dict(cfg)
    return op


def describe_op(op) -> dict:
    return dict(getattr(op, "knobs", {}))


attention = make_attention()


def reference_attention(q, k, v, bias, key_mask, scale, precision=jax.lax.Precision.HIGHEST):
    """Pure-JAX reference in fp32 math: softmax(q k^T * scale + bias + mask) v, heads-major -> q's dtype."""
    qf, kf, vf = (x.astype(jnp.float32) for x in (q, k, v))
    logits = jnp.einsum("bhqd,bhkd->bhqk", qf * scale, kf, precision=precision) + bias.astype(jnp.float32)[None]
    logits = jnp.where(key_mask[:, None, None, :], logits, -1e9)
    w = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", w, vf, precision=precision).astype(q.dtype)
