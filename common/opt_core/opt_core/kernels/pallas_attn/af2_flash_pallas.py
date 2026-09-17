"""af2_flash_pallas.py — flash attention with per-head pair bias and key mask for AF2/AF-Multimer Evoformer attention
(TriangleAttention starting/ending node, MSARowAttentionWithPairBias, MSAColumnAttention), FORWARD + BACKWARD, as jax.custom_vjp.
Layout (heads-major): q,k,v [B,H,S,D]; bias [H,Sq,Sk] shared over B (nonbatched_bias); kmask [B,Sk] bool. bf16/f16/f32 inputs, fp32 accumulation,
exact online softmax (fwd) and exact recomputation (bwd, FlashAttention-2 algorithm; dbias = sum over B of dS, deterministic: per-batch partials reduced by XLA, or — dbias='kernel' — summed over B inside a third kernel with no partials buffer;
written by the kernel + XLA sum; dQ via a separate kernel => no atomics, bit-exact run-to-run).
API written against jax 0.5.x Pallas/Triton (pl.load(ref, idx, mask=, other=), plgpu.TritonCompilerParams); on jax lines without pl.load / pl.store
the shims below use the Triton backend's plgpu.load / plgpu.store on ref.at[idx] (same masks, same arithmetic), and plgpu.CompilerParams.
"""
import functools, math, os
import numpy as np
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

_CP = getattr(plgpu, "TritonCompilerParams", None) or getattr(plgpu, "CompilerParams")
LOG2E = math.log2(math.e)
LN2 = math.log(2.0)
NEG = -1e30   # applied to fp32 logits


_PL_LOAD, _PL_STORE = getattr(pl, "load", None), getattr(pl, "store", None)   # jax lines that still carry pl.load / pl.store (0.5-0.7)


def _load(ref, idx, mask=None, other=None):
    if _PL_LOAD is not None:
        try:
            return _PL_LOAD(ref, idx, mask=mask, other=other)
        except TypeError:  # a pl.load without the (ref, idx) form
            pass
    return plgpu.load(ref.at[idx], mask=mask, other=other)       # jax lines without pl.load: the Triton backend's masked load on a ref view


def _store(ref, idx, val, mask=None):
    if _PL_STORE is not None:
        try:
            return _PL_STORE(ref, idx, val, mask=mask)
        except TypeError:
            pass
    return plgpu.store(ref.at[idx], val, mask=mask)


F32_PRECISIONS = ("tf32", "ieee", "bf16")                # forward products on float32 inputs: ieee fp32 (lax HIGHEST; F32_F32_F32 FMA; the DEFAULT = the v2 statement)
                                                          # | TF32 (lax DEFAULT: the class of XLA's default-precision f32 einsum on sm_80+, i.e. of the stock attention it replaces; opt-in) | bf16 operands.
                                                          # No 3-pass word: lax.Precision.HIGH lowers to plain tf32 in Pallas-Triton, so 'bf16x3' / 'tf32x3' / 'high' refuse by name.
F32_PRECISION_ALIASES = {"highest": "ieee"}
F32_PRECISION_DEFAULT = "ieee"
DQ_MODES = ("kernel", "from_ds")                          # how the backward forms dQ: a second Pallas kernel that recomputes S/P per q block (the DEFAULT = the v2 statement),
                                                          # or one batched GEMM dS·K over the per-batch fp32 dS buffer the dK/dV kernel writes for dbias (make_flash_attention(dq=))
DQ_MODE_DEFAULT = "kernel"
DBIAS_MODES = ("xla", "kernel")                          # how the backward forms d(pair bias) = sum over the batch of dS: 'xla' (the DEFAULT = the v2 statement) = the dK/dV
                                                          # kernel writes per-batch dS partials [B,H,Sq,Sk] and XLA reduces them over B; 'kernel' = a third Pallas kernel, grid
                                                          # (q_block, k_block, h), loops over the batch INSIDE the program recomputing S/P/dP/dS and accumulating one fp32 [bq,bk]
                                                          # tile, written once — no [B,H,Sq,Sk] buffer at all (make_flash_attention(dbias=)); both deterministic (no atomics)
DBIAS_MODE_DEFAULT = "xla"
BWD_F32_PRECISIONS = ("tf32", "ieee")                    # backward products on float32 inputs (make_flash_attention(bwd_f32_precision=)): ieee (the DEFAULT = the v2 statement) | tf32
                                                          # (tensor cores: the class of the XLA default-precision einsums of the stock backward it replaces). The precise_bwd path is ieee fp32 by construction.
_F32_PREC = {"ieee": jax.lax.Precision.HIGHEST, "tf32": jax.lax.Precision.DEFAULT}


def _dot(a, b, f32p="ieee"):  # fp32 accumulate
    # f32 inputs: products at `f32p` (ieee = full-precision fp32, the backward's class; tf32 = tensor cores; bf16 = operands rounded to bf16);
    # bf16 inputs: tensor-core bf16 x bf16 -> fp32 accumulate (f32p ignored)
    if a.dtype == jnp.float32:
        if f32p == "bf16":
            return jnp.dot(a.astype(jnp.bfloat16), b.astype(jnp.bfloat16), preferred_element_type=jnp.float32)
        return jnp.dot(a, b, preferred_element_type=jnp.float32, precision=_F32_PREC[f32p])
    return jnp.dot(a, b, preferred_element_type=jnp.float32, precision=None)


# ----------------------------------------------------------------------------------------------- forward
def _fwd_kernel(q_ref, k_ref, v_ref, b_ref, m_ref, o_ref, lse_ref, *, sm_scale, bq, bk, sq, sk, f32p="ieee"):
    D = q_ref.shape[-1]
    iq = pl.program_id(0)
    q_valid = (iq * bq + jnp.arange(bq)) < sq
    q = _load(q_ref, (slice(None), slice(None)), mask=q_valid[:, None], other=0.0)          # [bq, D]
    m_i = jnp.full((bq,), -jnp.inf, jnp.float32); l_i = jnp.zeros((bq,), jnp.float32); acc = jnp.zeros((bq, D), jnp.float32)

    def body(j, carry):
        acc, m_i, l_i = carry
        kidx = j * bk + jnp.arange(bk); kv = kidx < sk
        ks = pl.dslice(j * bk, bk)
        k = _load(k_ref, (ks, slice(None)), mask=kv[:, None], other=0.0)                    # [bk, D]
        v = _load(v_ref, (ks, slice(None)), mask=kv[:, None], other=0.0)
        b = _load(b_ref, (slice(None), ks), mask=q_valid[:, None] & kv[None, :], other=0.0).astype(jnp.float32)   # [bq, bk]
        km = _load(m_ref, (ks,), mask=kv, other=False)
        s = _dot(q, k.T, f32p) * sm_scale + b
        s = jnp.where(km[None, :], s, NEG)
        m_new = jnp.maximum(m_i, jnp.max(s, axis=1))
        alpha = jnp.exp2((m_i - m_new) * LOG2E)
        p = jnp.exp2((s - m_new[:, None]) * LOG2E)
        l_new = alpha * l_i + p.sum(axis=1)
        acc = acc * alpha[:, None] + _dot(p.astype(v.dtype), v, f32p)
        return acc, m_new, l_new

    acc, m_i, l_i = jax.lax.fori_loop(0, pl.cdiv(sk, bk), body, (acc, m_i, l_i))
    o = acc / l_i[:, None]
    _store(o_ref, (slice(None), slice(None)), o.astype(o_ref.dtype), mask=q_valid[:, None])
    _store(lse_ref, (slice(None),), m_i + jnp.log(l_i), mask=q_valid)


def _fwd(q, k, v, bias, kmask, *, sm_scale, bq, bk, num_warps, num_stages, f32p="ieee"):
    B, H, Sq, D = q.shape; Sk = k.shape[2]
    grid = (pl.cdiv(Sq, bq), H, B)   # batch innermost: bias[h, qblock, :] stays hot in L2 across the batch sweep
    o, lse = pl.pallas_call(
        functools.partial(_fwd_kernel, sm_scale=sm_scale, bq=bq, bk=bk, sq=Sq, sk=Sk, f32p=f32p),
        grid=grid,
        in_specs=[pl.BlockSpec((None, None, bq, D), lambda i, h, b: (b, h, i, 0)),
                  pl.BlockSpec((None, None, Sk, D), lambda i, h, b: (b, h, 0, 0)),
                  pl.BlockSpec((None, None, Sk, D), lambda i, h, b: (b, h, 0, 0)),
                  pl.BlockSpec((None, bq, Sk), lambda i, h, b: (h, i, 0)),
                  pl.BlockSpec((None, Sk), lambda i, h, b: (b, 0))],
        out_specs=[pl.BlockSpec((None, None, bq, D), lambda i, h, b: (b, h, i, 0)),
                   pl.BlockSpec((None, None, bq), lambda i, h, b: (b, h, i))],
        out_shape=[jax.ShapeDtypeStruct((B, H, Sq, D), q.dtype), jax.ShapeDtypeStruct((B, H, Sq), jnp.float32)],
        compiler_params=_CP(num_warps=num_warps, num_stages=num_stages),
        name="af2_flash_fwd",
    )(q, k, v, bias, kmask)
    return o, lse


# ----------------------------------------------------------------------------------------------- backward
def _bwd_dkdv_kernel(q_ref, k_ref, v_ref, b_ref, m_ref, do_ref, lse_ref, delta_ref, dk_ref, dv_ref, dbias_ref=None, *, sm_scale, bq, bk, sq, sk, precise=False, f32p="ieee"):
    """grid (k_block, h, b): dK, dV for one key block (loop over q blocks); also writes dS (per batch) into dbias partial [b,h,Sq,bk-slice]
    when a dbias_ref is given (dbias='xla'); with dbias='kernel' the call has two outputs and dS stays in registers."""
    D = q_ref.shape[-1]
    jk = pl.program_id(0)
    kidx = jk * bk + jnp.arange(bk); kv = kidx < sk
    k = _load(k_ref, (slice(None), slice(None)), mask=kv[:, None], other=0.0)   # [bk, D]  (block already selected by BlockSpec)
    v = _load(v_ref, (slice(None), slice(None)), mask=kv[:, None], other=0.0)
    km = _load(m_ref, (slice(None),), mask=kv, other=False)                     # [bk]
    dk = jnp.zeros((bk, D), jnp.float32); dv = jnp.zeros((bk, D), jnp.float32)

    def body(i, carry):
        dk, dv = carry
        qidx = i * bq + jnp.arange(bq); qv = qidx < sq
        qs = pl.dslice(i * bq, bq)
        q = _load(q_ref, (qs, slice(None)), mask=qv[:, None], other=0.0)         # [bq, D]
        do = _load(do_ref, (qs, slice(None)), mask=qv[:, None], other=0.0)
        lse = _load(lse_ref, (qs,), mask=qv, other=0.0)
        dl = _load(delta_ref, (qs,), mask=qv, other=0.0)
        b = _load(b_ref, (qs, slice(None)), mask=qv[:, None] & kv[None, :], other=0.0).astype(jnp.float32)   # [bq, bk]
        s = _dot(q, k.T, "ieee" if precise else f32p) * sm_scale + b
        s = jnp.where(km[None, :], s, NEG)
        p = jnp.exp2((s - lse[:, None]) * LOG2E)
        p = jnp.where(qv[:, None] & km[None, :], p, 0.0)
        if precise:   # dS/P kept in fp32 for the gradient products (inputs upcast; ~2x slower backward, error == fp32 reference class)
            dv = dv + _dot(p.T, do.astype(jnp.float32))
            dp = _dot(do.astype(jnp.float32), v.astype(jnp.float32).T)
            ds = p * (dp - dl[:, None])
            dk = dk + _dot(ds.T, q.astype(jnp.float32)) * sm_scale
        else:
            dv = dv + _dot(p.astype(do.dtype).T, do, f32p)
            dp = _dot(do, v.T, f32p)
            ds = p * (dp - dl[:, None])                                              # fp32 [bq, bk]
            dk = dk + _dot(ds.astype(q.dtype).T, q, f32p) * sm_scale
        if dbias_ref is not None:
            _store(dbias_ref, (qs, slice(None)), ds.astype(dbias_ref.dtype), mask=qv[:, None] & kv[None, :])
        return dk, dv

    dk, dv = jax.lax.fori_loop(0, pl.cdiv(sq, bq), body, (dk, dv))
    _store(dk_ref, (slice(None), slice(None)), dk.astype(dk_ref.dtype), mask=kv[:, None])
    _store(dv_ref, (slice(None), slice(None)), dv.astype(dv_ref.dtype), mask=kv[:, None])


def _bwd_dq_kernel(q_ref, k_ref, v_ref, b_ref, m_ref, do_ref, lse_ref, delta_ref, dq_ref, *, sm_scale, bq, bk, sq, sk, precise=False, f32p="ieee"):
    D = q_ref.shape[-1]
    iq = pl.program_id(0)
    qv = (iq * bq + jnp.arange(bq)) < sq
    q = _load(q_ref, (slice(None), slice(None)), mask=qv[:, None], other=0.0)
    do = _load(do_ref, (slice(None), slice(None)), mask=qv[:, None], other=0.0)
    lse = _load(lse_ref, (slice(None),), mask=qv, other=0.0)
    dl = _load(delta_ref, (slice(None),), mask=qv, other=0.0)
    dq = jnp.zeros((bq, D), jnp.float32)

    def body(j, dq):
        kidx = j * bk + jnp.arange(bk); kvv = kidx < sk
        ks = pl.dslice(j * bk, bk)
        k = _load(k_ref, (ks, slice(None)), mask=kvv[:, None], other=0.0)
        v = _load(v_ref, (ks, slice(None)), mask=kvv[:, None], other=0.0)
        km = _load(m_ref, (ks,), mask=kvv, other=False)
        b = _load(b_ref, (slice(None), ks), mask=qv[:, None] & kvv[None, :], other=0.0).astype(jnp.float32)
        s = _dot(q, k.T, "ieee" if precise else f32p) * sm_scale + b
        s = jnp.where(km[None, :], s, NEG)
        p = jnp.exp2((s - lse[:, None]) * LOG2E)
        p = jnp.where(km[None, :], p, 0.0)
        if precise:
            dp = _dot(do.astype(jnp.float32), v.astype(jnp.float32).T)
            ds = p * (dp - dl[:, None])
            return dq + _dot(ds, k.astype(jnp.float32)) * sm_scale
        dp = _dot(do, v.T, f32p)
        ds = p * (dp - dl[:, None])
        return dq + _dot(ds.astype(k.dtype), k, f32p) * sm_scale

    dq = jax.lax.fori_loop(0, pl.cdiv(sk, bk), body, dq)
    _store(dq_ref, (slice(None), slice(None)), dq.astype(dq_ref.dtype), mask=qv[:, None])


def _bwd_dbias_kernel(q_ref, k_ref, v_ref, b_ref, m_ref, do_ref, lse_ref, delta_ref, dbias_ref, *, sm_scale, bq, bk, sq, sk, nb, precise=False, f32p="ieee"):
    """grid (q_block, k_block, h): d(pair bias) for one [bq, bk] tile = the sum over the batch of dS, accumulated in fp32 registers across a
    loop over the nb batch rows INSIDE the program (S, P, dP, dS recomputed per row exactly as the dK/dV kernel forms them) and written once:
    no per-batch [B,H,Sq,Sk] partials, no reduction outside the kernel, no atomics (deterministic; the summation order over the batch is
    sequential, the 'xla' mode's is XLA's reduction tree — same values to fp32 re-association)."""
    i = pl.program_id(0); j = pl.program_id(1)
    qidx = i * bq + jnp.arange(bq); qv = qidx < sq
    kidx = j * bk + jnp.arange(bk); kv = kidx < sk
    bias = _load(b_ref, (slice(None), slice(None)), mask=qv[:, None] & kv[None, :], other=0.0).astype(jnp.float32)   # [bq, bk] (block selected by BlockSpec)
    acc = jnp.zeros((bq, bk), jnp.float32)

    def body(b, acc):
        q = _load(q_ref, (b, slice(None), slice(None)), mask=qv[:, None], other=0.0)      # [bq, D]
        do = _load(do_ref, (b, slice(None), slice(None)), mask=qv[:, None], other=0.0)
        lse = _load(lse_ref, (b, slice(None)), mask=qv, other=0.0)
        dl = _load(delta_ref, (b, slice(None)), mask=qv, other=0.0)
        k = _load(k_ref, (b, slice(None), slice(None)), mask=kv[:, None], other=0.0)       # [bk, D]
        v = _load(v_ref, (b, slice(None), slice(None)), mask=kv[:, None], other=0.0)
        km = _load(m_ref, (b, slice(None)), mask=kv, other=False)                            # [bk]
        s = _dot(q, k.T, "ieee" if precise else f32p) * sm_scale + bias
        s = jnp.where(km[None, :], s, NEG)
        p = jnp.exp2((s - lse[:, None]) * LOG2E)
        p = jnp.where(qv[:, None] & km[None, :], p, 0.0)
        if precise:
            dp = _dot(do.astype(jnp.float32), v.astype(jnp.float32).T)
        else:
            dp = _dot(do, v.T, f32p)
        return acc + p * (dp - dl[:, None])                                                   # dS of row b, fp32

    acc = jax.lax.fori_loop(0, nb, body, acc)
    _store(dbias_ref, (slice(None), slice(None)), acc.astype(dbias_ref.dtype), mask=qv[:, None] & kv[None, :])


def _bwd(q, k, v, bias, kmask, o, lse, do, *, sm_scale, bq, bk, num_warps, num_stages, dbias_dtype, precise=False, f32p="ieee", dq_mode="kernel", dbias_mode="xla"):
    B, H, Sq, D = q.shape; Sk = k.shape[2]
    delta = jnp.einsum("bhqd,bhqd->bhq", o.astype(jnp.float32), do.astype(jnp.float32))   # rowsum(O*dO)
    common = dict(sm_scale=sm_scale, bq=bq, bk=bk, sq=Sq, sk=Sk, precise=precise, f32p=f32p)
    full_q = pl.BlockSpec((None, None, Sq, D), lambda j, h, b: (b, h, 0, 0))
    blk_k = pl.BlockSpec((None, None, bk, D), lambda j, h, b: (b, h, j, 0))
    dkdv_in_specs = [full_q, blk_k, blk_k,
                     pl.BlockSpec((None, Sq, bk), lambda j, h, b: (h, 0, j)),
                     pl.BlockSpec((None, bk), lambda j, h, b: (b, j)),
                     full_q,
                     pl.BlockSpec((None, None, Sq), lambda j, h, b: (b, h, 0)),
                     pl.BlockSpec((None, None, Sq), lambda j, h, b: (b, h, 0))]
    dkdv_shapes = [jax.ShapeDtypeStruct((B, H, Sk, D), k.dtype), jax.ShapeDtypeStruct((B, H, Sk, D), v.dtype)]
    if dbias_mode == "kernel":        # dK/dV without the dS partials; d(pair bias) by the third kernel (batch loop inside the program); dQ by its kernel
        dk, dv = pl.pallas_call(
            functools.partial(_bwd_dkdv_kernel, **common),
            grid=(pl.cdiv(Sk, bk), H, B), in_specs=dkdv_in_specs, out_specs=[blk_k, blk_k], out_shape=dkdv_shapes,
            compiler_params=_CP(num_warps=num_warps, num_stages=num_stages),
            name="af2_flash_bwd_dkdv_nods",
        )(q, k, v, bias, kmask, do, lse, delta)
        rows_q = pl.BlockSpec((B, None, bq, D), lambda i, j, h: (0, h, i, 0))          # every batch row of one (head, q block): the program loops over rows
        rows_k = pl.BlockSpec((B, None, bk, D), lambda i, j, h: (0, h, j, 0))
        rows_stat = pl.BlockSpec((B, None, bq), lambda i, j, h: (0, h, i))
        dbias = pl.pallas_call(
            functools.partial(_bwd_dbias_kernel, nb=B, **common),
            grid=(pl.cdiv(Sq, bq), pl.cdiv(Sk, bk), H),
            in_specs=[rows_q, rows_k, rows_k,
                      pl.BlockSpec((None, bq, bk), lambda i, j, h: (h, i, j)),
                      pl.BlockSpec((B, bk), lambda i, j, h: (0, j)),
                      rows_q, rows_stat, rows_stat],
            out_specs=pl.BlockSpec((None, bq, bk), lambda i, j, h: (h, i, j)),
            out_shape=jax.ShapeDtypeStruct((H, Sq, Sk), jnp.float32),
            compiler_params=_CP(num_warps=num_warps, num_stages=num_stages),
            name="af2_flash_bwd_dbias",
        )(q, k, v, bias, kmask, do, lse, delta)
        dbias_part = None
    else:
        dk, dv, dbias_part = pl.pallas_call(
            functools.partial(_bwd_dkdv_kernel, **common),
            grid=(pl.cdiv(Sk, bk), H, B), in_specs=dkdv_in_specs,
            out_specs=[blk_k, blk_k, pl.BlockSpec((None, None, Sq, bk), lambda j, h, b: (b, h, 0, j))],
            out_shape=dkdv_shapes + [jax.ShapeDtypeStruct((B, H, Sq, Sk), jnp.float32 if dq_mode == "from_ds" else dbias_dtype)],
            compiler_params=_CP(num_warps=num_warps, num_stages=num_stages),
            name="af2_flash_bwd_dkdv",
        )(q, k, v, bias, kmask, do, lse, delta)
    if dq_mode == "from_ds":      # dQ = dS·K·scale from the fp32 dS the dkdv kernel wrote per batch row: one batched GEMM over that buffer instead of a
                                  # second kernel that recomputes S, P and dP per tile (same values; products at the backward's class)
        prec = jax.lax.Precision.HIGHEST if (precise or f32p == "ieee" or q.dtype != jnp.float32) else jax.lax.Precision.DEFAULT
        dq = (jnp.einsum("bhqk,bhkd->bhqd", dbias_part, k.astype(jnp.float32), precision=prec, preferred_element_type=jnp.float32) * sm_scale).astype(q.dtype)
        dbias = dbias_part.sum(axis=0).astype(bias.dtype)                    # [H,Sq,Sk]; deterministic XLA reduction over batch
        return dq, dk, dv, dbias
    blk_q = pl.BlockSpec((None, None, bq, D), lambda i, h, b: (b, h, i, 0))
    full_k = pl.BlockSpec((None, None, Sk, D), lambda i, h, b: (b, h, 0, 0))
    dq = pl.pallas_call(
        functools.partial(_bwd_dq_kernel, **common),
        grid=(pl.cdiv(Sq, bq), H, B),
        in_specs=[blk_q, full_k, full_k,
                  pl.BlockSpec((None, bq, Sk), lambda i, h, b: (h, i, 0)),
                  pl.BlockSpec((None, Sk), lambda i, h, b: (b, 0)),
                  blk_q,
                  pl.BlockSpec((None, None, bq), lambda i, h, b: (b, h, i)),
                  pl.BlockSpec((None, None, bq), lambda i, h, b: (b, h, i))],
        out_specs=blk_q,
        out_shape=jax.ShapeDtypeStruct((B, H, Sq, D), q.dtype),
        compiler_params=_CP(num_warps=num_warps, num_stages=num_stages),
        name="af2_flash_bwd_dq",
    )(q, k, v, bias, kmask, do, lse, delta)
    if dbias_mode == "kernel":
        return dq, dk, dv, dbias.astype(bias.dtype)                          # [H,Sq,Sk] summed over the batch inside the kernel
    dbias = dbias_part.astype(jnp.float32).sum(axis=0).astype(bias.dtype)   # [H,Sq,Sk]; deterministic XLA reduction over batch
    return dq, dk, dv, dbias


# ----------------------------------------------------------------------------------------------- public op
def _bwd_chunked(q, k, v, bias, kmask, o, lse, do, *, batch_chunk, **kw):
    """Backward with the batch processed in chunks of `batch_chunk` rows (lax.map): the per-batch dS partial buffer for dbias is
    [batch_chunk,H,Sq,Sk] instead of [B,H,Sq,Sk] (B x smaller peak), dbias accumulated in fp32 across chunks. Same math, same kernels."""
    B = q.shape[0]
    nb = -(-B // batch_chunk); Bp = nb * batch_chunk
    pad = lambda x: jnp.pad(x, [(0, Bp - B)] + [(0, 0)] * (x.ndim - 1)) if Bp != B else x
    rs = lambda x: pad(x).reshape((nb, batch_chunk) + x.shape[1:])
    xs = tuple(rs(x) for x in (q, k, v, kmask, o, lse, do))
    H, Sq, Sk = bias.shape[0], q.shape[2], k.shape[2]

    def body(carry, xc):
        qc, kc, vc, mc, oc, lc, dc = xc
        dq, dk, dv, dbias = _bwd(qc, kc, vc, bias, mc, oc, lc, dc, **kw)
        return carry + dbias.astype(jnp.float32), (dq, dk, dv)

    dbias_acc, (dq, dk, dv) = jax.lax.scan(body, jnp.zeros((H, Sq, Sk), jnp.float32), xs)
    us = lambda x: x.reshape((Bp,) + x.shape[2:])[:B]
    return us(dq), us(dk), us(dv), dbias_acc.astype(bias.dtype)


_BUILT = {}                                            # id(op) -> what make_flash_attention built


def make_flash_attention(bq=64, bk=64, num_warps=4, num_stages=2, dbias_dtype=None, bq_bwd=64, bk_bwd=64, batch_chunk=None, precise_bwd=None, f32_precision=None,
                         bwd_f32_precision=None, dq=None, dbias=None):
    """Build the differentiable op. Forward precision knob for float32 inputs (deterministic; bf16 inputs ignore it):
       f32_precision: one of F32_PRECISIONS — 'ieee' (full fp32 products: the v2 statement and the default; alias 'highest'), 'tf32' (tensor
                      cores; the class of XLA's default-precision float32 einsum on sm_80+, i.e. of the stock attention this op replaces — a kit
                      that wants it passes it explicitly and prints precision=tf32),
                      'bf16' (q/k/p/v operands rounded to bf16, fp32 accumulate, f32 out; the pair bias stays fp32). Any other word (incl. 'bf16x3',
                      'tf32x3', 'high': lax.Precision.HIGH lowers to plain tf32 here) raises ValueError naming the valid set.
                      None -> env AF_PALLAS_ATTN_F32_PRECISION or 'ieee'.
       bwd_f32_precision: one of BWD_F32_PRECISIONS for the backward's float32 products — 'ieee' (the default, the v2 statement) or 'tf32';
                      None -> env AF_PALLAS_ATTN_BWD_F32_PRECISION or 'ieee'. `precise_bwd` keeps dS/P and its products in ieee fp32 regardless.
       dq:            one of DQ_MODES — 'kernel' (the default, the v2 statement: a second kernel recomputes S/P per q block) or 'from_ds' (dQ = dS·K, one
                      batched GEMM over the fp32 per-batch dS buffer written for dbias; fewer FLOPs, the buffer held in fp32). None -> env AF_PALLAS_ATTN_DQ or 'kernel'.
       dbias:         one of DBIAS_MODES — 'xla' (the default, the v2 statement: per-batch dS partials [B,H,Sq,Sk] reduced over the batch by XLA) or 'kernel'
                      (a third kernel accumulates the batch sum of dS per [bq,bk] tile in registers — no [B,H,Sq,Sk] buffer, one fp32 [H,Sq,Sk] write;
                      recomputes S/P/dP per row, i.e. more FLOPs for far fewer bytes). None -> env AF_PALLAS_ATTN_DBIAS or 'xla'. dq='from_ds' needs
                      the dS buffer and therefore refuses dbias='kernel' (ValueError).
    Backward precision knobs (all deterministic):
       dbias_dtype : dtype of the per-batch dS partials that are summed over the batch into d(pair bias). None -> env AF_PALLAS_ATTN_DBIAS_F32=1 selects
                     float32, else the bias dtype (bf16 under AF2's policy). float32 costs B*H*S*S*4 bytes of scratch (use batch_chunk to bound it).
       precise_bwd : keep P/dS in float32 for the dV/dK/dQ products (inputs upcast). None -> env AF_PALLAS_ATTN_PRECISE_BWD=1. ~2x slower backward.
       batch_chunk : process the batch in chunks in the backward (bounds the dS scratch); None -> env AF_PALLAS_ATTN_BWD_BATCH_CHUNK (0/unset = off)."""
    if precise_bwd is None: precise_bwd = os.environ.get("AF_PALLAS_ATTN_PRECISE_BWD", "0") == "1"
    f32p = f32_precision if f32_precision is not None else (os.environ.get("AF_PALLAS_ATTN_F32_PRECISION", "") or F32_PRECISION_DEFAULT)
    f32p = F32_PRECISION_ALIASES.get(f32p, f32p)
    if f32p not in F32_PRECISIONS:
        raise ValueError(f"f32_precision={f32p!r} not in {F32_PRECISIONS}")
    f32pb = bwd_f32_precision if bwd_f32_precision is not None else (os.environ.get("AF_PALLAS_ATTN_BWD_F32_PRECISION", "") or F32_PRECISION_DEFAULT)
    f32pb = F32_PRECISION_ALIASES.get(f32pb, f32pb)
    if f32pb not in BWD_F32_PRECISIONS:
        raise ValueError(f"bwd_f32_precision={f32pb!r} not in {BWD_F32_PRECISIONS}")
    dq_mode = dq if dq is not None else (os.environ.get("AF_PALLAS_ATTN_DQ", "") or DQ_MODE_DEFAULT)
    if dq_mode not in DQ_MODES:
        raise ValueError(f"dq={dq_mode!r} not in {DQ_MODES}")
    dbias_mode = dbias if dbias is not None else (os.environ.get("AF_PALLAS_ATTN_DBIAS", "") or DBIAS_MODE_DEFAULT)
    if dbias_mode not in DBIAS_MODES:
        raise ValueError(f"dbias={dbias_mode!r} not in {DBIAS_MODES}")
    if dbias_mode == "kernel" and dq_mode == "from_ds":
        raise ValueError("dq='from_ds' forms dQ from the per-batch dS buffer, which dbias='kernel' does not write: use dq='kernel' with dbias='kernel'")
    dbias_f32 = os.environ.get("AF_PALLAS_ATTN_DBIAS_F32", "0") == "1"
    @functools.partial(jax.custom_vjp, nondiff_argnums=(5,))
    def flash_attn(q, k, v, bias, kmask, sm_scale):
        return _fwd(q, k, v, bias, kmask, sm_scale=sm_scale, bq=bq, bk=bk, num_warps=num_warps, num_stages=num_stages, f32p=f32p)[0]

    def f_fwd(q, k, v, bias, kmask, sm_scale):
        o, lse = _fwd(q, k, v, bias, kmask, sm_scale=sm_scale, bq=bq, bk=bk, num_warps=num_warps, num_stages=num_stages, f32p=f32p)
        return o, (q, k, v, bias, kmask, o, lse)

    def f_bwd(sm_scale, res, do):
        q, k, v, bias, kmask, o, lse = res
        kw = dict(sm_scale=sm_scale, bq=bq_bwd, bk=bk_bwd, num_warps=num_warps, num_stages=num_stages,
                  dbias_dtype=(dbias_dtype or (jnp.float32 if dbias_f32 else bias.dtype)), precise=precise_bwd, f32p=f32pb, dq_mode=dq_mode, dbias_mode=dbias_mode)
        bc = batch_chunk if batch_chunk is not None else int(os.environ.get("AF_PALLAS_ATTN_BWD_BATCH_CHUNK", "0") or 0)
        if bc and q.shape[0] > bc:
            dq, dk, dv, dbias = _bwd_chunked(q, k, v, bias, kmask, o, lse, do, batch_chunk=bc, **kw)
        else:
            dq, dk, dv, dbias = _bwd(q, k, v, bias, kmask, o, lse, do, **kw)
        return dq, dk, dv, dbias, None

    flash_attn.defvjp(f_fwd, f_bwd)
    meta = dict(f32_precision=f32p, bwd_f32_precision=f32pb, dq=dq_mode, dbias=dbias_mode, bq=bq, bk=bk, num_warps=num_warps, num_stages=num_stages, bq_bwd=bq_bwd,
                bk_bwd=bk_bwd, batch_chunk=batch_chunk)
    _BUILT[id(flash_attn)] = meta                       # describe_op(op): what was built (the kit records precision=<f32_precision> on its LEVER line)
    try:
        flash_attn.f32_precision = f32p; flash_attn.bwd_f32_precision = f32pb; flash_attn.tiles = dict(meta)
    except Exception:  # noqa: BLE001 — an op object that refuses attributes is still described by describe_op
        pass
    return flash_attn


def describe_op(op):
    """{'f32_precision', 'bwd_f32_precision', 'dq', 'dbias', 'bq', 'bk', 'num_warps', 'num_stages', ...} of an op built by make_flash_attention (the module default included)."""
    return dict(_BUILT.get(id(op), {}))


flash_attention = make_flash_attention()


def attention_af2(q, k, v, mask_bias, nonbatched_bias, scale, impl=flash_attention):
    """Drop-in for alphafold.model.modules.Attention core: q,k,v [b,h,S,c] (heads-major), mask_bias additive [b,1,1,Sk] (-1e9 = masked),
    nonbatched_bias [h,Sq,Sk] or None -> [b,h,Sq,c]."""
    b, h, sq, c = q.shape; sk = k.shape[2]
    kmask = mask_bias[:, 0, 0, :] > -1e8
    bias = jnp.zeros((h, sq, sk), q.dtype) if nonbatched_bias is None else nonbatched_bias.astype(q.dtype)
    return impl(q, k, v, bias, kmask, float(scale))


def reference_attention(q, k, v, bias, kmask, sm_scale, precision=None):
    """XLA reference in the dtype of the inputs (AF2's own math: logits in input dtype, softmax via jax.nn.softmax)."""
    logits = jnp.einsum("bhqd,bhkd->bhqk", q * sm_scale, k, precision=precision) + bias[None]
    logits = jnp.where(kmask[:, None, None, :], logits, -1e9)
    w = jax.nn.softmax(logits)
    return jnp.einsum("bhqk,bhkd->bhqd", w.astype(v.dtype), v, precision=precision)
