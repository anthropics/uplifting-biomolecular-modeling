"""rowshared_flash_pallas.py — FORWARD flash attention with a pair bias SHARED by every batch row (AF2 / AF-Multimer
MSARowAttentionWithPairBias, TriangleAttention starting / ending node, extra-MSA row attention): the carried kernel
``pallas_attn/af2_flash_pallas._fwd_kernel`` re-expressed with two independent launch levers (lever KOPT_ROWATTN):

  rows=R    one program serves R consecutive batch rows of one (q block, head): the bias tile [bq, bk] is loaded and converted to fp32
            ONCE per key block and applied to the R rows; each row keeps its own (acc, m, l) online-softmax state in registers and walks
            the key blocks in the same order with the same tiles as the carried kernel, so the per-row arithmetic (products, fp32
            accumulation order, exp2 arguments) is the carried kernel's: bitwise equality per row is EXPECTED and is a MEASURED property
            (selftest under interpret=True on CPU; the GPU microbench re-checks it per shape and per tile row).
  order     'legacy'  = the carried launch order grid=(q_block, head, row_block): CUDA x = q block is the fastest axis, rows the slowest
                        (between two uses of one bias tile the whole [H, Sq, Sk] bias passes through L2);
            'grouped' = grid=(row_block inside a group of `group_rows` rows, q_block, head x group): the rows of a group are the fastest
                        axis, so one bias row-block [bq, Sk] is read from HBM once per group and the K / V of the group's rows
                        (group_rows x 2 x Sk x D per head) stay L2-resident across the q-block sweep. Same program body: bitwise.

Layout (the carried kernel's): q, k, v [B, H, S, D] (bf16 / f16 / f32, D >= 16), bias [H, Sq, Sk] shared over B, kmask [B, Sk] bool ->
o [B, H, Sq, D] (input dtype), lse [B, H, Sq] fp32. fp32 logits, fp32 online-softmax statistics, fp32 accumulators always.
Index safety: Pallas-Triton forms element offsets in int32; ``make_rowshared_attention`` splits the batch into row chunks whose largest
operand stays below 2**31 elements (static Python split) and REFUSES BY NAME (``RowAttnRefusal('index_ge_2p31', ...)``) when one row or the
bias alone would pass it. Written against jax 0.5.3 Pallas/Triton with the carried kernel's load/store shims for newer jax lines.
Backward: not built here. The op is a ``jax.custom_vjp`` whose residuals (q, k, v, bias, kmask, o, lse) are the carried kernel's, so the
backward seam calls the carried ``af2_flash_pallas._bwd`` unchanged (``differentiable=True``) or raises by name.
"""
import functools, math, os, sys
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

_CP = getattr(plgpu, "TritonCompilerParams", None) or getattr(plgpu, "CompilerParams")
LOG2E = math.log2(math.e)
NEG = -1e30          # applied to fp32 logits (the carried kernel's constant)
I32_LIMIT = 2 ** 31  # element-offset ceiling of one pallas_call operand
ORDERS = ("legacy", "grouped")
F32_PRECISIONS = ("tf32", "ieee", "bf16")
_F32_PREC = {"ieee": jax.lax.Precision.HIGHEST, "tf32": jax.lax.Precision.DEFAULT}
_PL_LOAD, _PL_STORE = getattr(pl, "load", None), getattr(pl, "store", None)


class RowAttnRefusal(RuntimeError):
    """A named refusal (``kind``, ``detail``): index_ge_2p31 · bad_rows · bad_order · bad_tiles · head_dim_lt_16 · backward_not_built."""

    def __init__(self, kind, detail=""):
        self.kind, self.detail = kind, detail
        super().__init__(f"kopt_rowattn: {kind}" + (f" — {detail}" if detail else ""))


def _load(ref, idx, mask=None, other=None):
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


def _dot(a, b, f32p="ieee"):  # fp32 accumulate; the carried kernel's rule
    if a.dtype == jnp.float32:
        if f32p == "bf16":
            return jnp.dot(a.astype(jnp.bfloat16), b.astype(jnp.bfloat16), preferred_element_type=jnp.float32)
        return jnp.dot(a, b, preferred_element_type=jnp.float32, precision=_F32_PREC[f32p])
    return jnp.dot(a, b, preferred_element_type=jnp.float32, precision=None)


# ----------------------------------------------------------------------------------------------- kernel
def _fwd_kernel(q_ref, k_ref, v_ref, b_ref, m_ref, o_ref, lse_ref, *, sm_scale, bq, bk, sq, sk, rows, nb, nrb, order, gb, ngroups, f32p):
    """One program: R = `rows` batch rows [rb*R, rb*R+R) of one (q block iq, head). Refs (blocks picked by the BlockSpecs):
    q_ref/o_ref [R, bq, D], k_ref/v_ref [R, Sk, D], b_ref [bq, Sk], m_ref [R, Sk], lse_ref [R, bq]."""
    D = q_ref.shape[-1]
    if order == "legacy":
        iq = pl.program_id(0); rb = pl.program_id(2); live = None
    else:
        iq = pl.program_id(1)
        rb_raw = (pl.program_id(2) % ngroups) * gb + pl.program_id(0)
        live = rb_raw < nrb                                   # the last group may be short: its surplus programs do nothing
        rb = jnp.minimum(rb_raw, nrb - 1)
    q_valid = (iq * bq + jnp.arange(bq)) < sq
    partial_rows = (nb % rows) != 0                            # static: only then a row of the last row block can be out of range
    rv = [((rb * rows + r) < nb) if partial_rows else None for r in range(rows)]

    def _and(mask, r):
        return mask if rv[r] is None else (mask & rv[r])

    ri = [jnp.asarray(r, jnp.int32) for r in range(rows)]      # row index inside the block as an int32 scalar (a Python int is not an index the
                                                               # jax 0.5.3 interpret-mode load rule accepts; the Triton lowering takes either)

    def run():
        qs = [_load(q_ref, (ri[r], slice(None), slice(None)), mask=_and(q_valid[:, None], r), other=0.0) for r in range(rows)]     # R x [bq, D]
        init = tuple((jnp.zeros((bq, D), jnp.float32), jnp.full((bq,), -jnp.inf, jnp.float32), jnp.zeros((bq,), jnp.float32)) for _ in range(rows))

        def body(j, carry):
            kidx = j * bk + jnp.arange(bk); kv = kidx < sk
            ks = pl.dslice(j * bk, bk)
            b = _load(b_ref, (slice(None), ks), mask=q_valid[:, None] & kv[None, :], other=0.0).astype(jnp.float32)          # [bq, bk] ONCE for the R rows
            out = []
            for r in range(rows):                                                                                             # static unroll: R independent softmax states
                acc, m_i, l_i = carry[r]
                k = _load(k_ref, (ri[r], ks, slice(None)), mask=_and(kv[:, None], r), other=0.0)                                  # [bk, D]
                v = _load(v_ref, (ri[r], ks, slice(None)), mask=_and(kv[:, None], r), other=0.0)
                km = _load(m_ref, (ri[r], ks), mask=_and(kv, r), other=False)
                s = _dot(qs[r], k.T, f32p) * sm_scale + b
                s = jnp.where(km[None, :], s, NEG)
                m_new = jnp.maximum(m_i, jnp.max(s, axis=1))
                alpha = jnp.exp2((m_i - m_new) * LOG2E)
                p = jnp.exp2((s - m_new[:, None]) * LOG2E)
                l_new = alpha * l_i + p.sum(axis=1)
                acc = acc * alpha[:, None] + _dot(p.astype(v.dtype), v, f32p)
                out.append((acc, m_new, l_new))
            return tuple(out)

        fin = jax.lax.fori_loop(0, pl.cdiv(sk, bk), body, init)
        for r in range(rows):
            acc, m_i, l_i = fin[r]
            o = acc / l_i[:, None]
            _store(o_ref, (ri[r], slice(None), slice(None)), o.astype(o_ref.dtype), mask=_and(q_valid[:, None], r))
            _store(lse_ref, (ri[r], slice(None)), m_i + jnp.log(l_i), mask=_and(q_valid, r))

    if live is None:
        run()
    else:
        pl.when(live)(run)


def _fwd_call(q, k, v, bias, kmask, *, sm_scale, bq, bk, num_warps, num_stages, rows, order, group_rows, f32p, interpret):
    B, H, Sq, D = q.shape; Sk = k.shape[2]
    R = int(rows)
    nrb = -(-B // R)                                           # row blocks
    nq = pl.cdiv(Sq, bq)
    if order == "legacy":
        gb, ngroups = nrb, 1
        grid = (nq, H, nrb)
        rowmap = lambda i, h, b: (b, h, i)                     # (row block, head, q block)
    else:
        gb = max(1, min(nrb, int(group_rows) // R))            # row blocks per group
        ngroups = -(-nrb // gb)
        grid = (gb, nq, H * ngroups)
        rowmap = lambda r0, i, hg: (jnp.minimum((hg % ngroups) * gb + r0, nrb - 1), hg // ngroups, i)
    if grid[1] > 65535 or grid[2] > 65535:
        raise RowAttnRefusal("bad_tiles", f"launch grid {grid} exceeds the CUDA y/z limit 65535")

    def spec_q(*g):
        rb, h, i = rowmap(*g); return (rb, h, i, 0)

    def spec_kv(*g):
        rb, h, i = rowmap(*g); return (rb, h, 0, 0)

    def spec_bias(*g):
        rb, h, i = rowmap(*g); return (h, i, 0)

    def spec_mask(*g):
        rb, h, i = rowmap(*g); return (rb, 0)

    def spec_lse(*g):
        rb, h, i = rowmap(*g); return (rb, h, i)

    kern = functools.partial(_fwd_kernel, sm_scale=sm_scale, bq=bq, bk=bk, sq=Sq, sk=Sk, rows=R, nb=B, nrb=nrb, order=order, gb=gb, ngroups=ngroups, f32p=f32p)
    kw = dict(interpret=True) if interpret else dict(compiler_params=_CP(num_warps=num_warps, num_stages=num_stages))
    o, lse = pl.pallas_call(
        kern, grid=grid,
        in_specs=[pl.BlockSpec((R, None, bq, D), spec_q),
                  pl.BlockSpec((R, None, Sk, D), spec_kv),
                  pl.BlockSpec((R, None, Sk, D), spec_kv),
                  pl.BlockSpec((None, bq, Sk), spec_bias),
                  pl.BlockSpec((R, Sk), spec_mask)],
        out_specs=[pl.BlockSpec((R, None, bq, D), spec_q),
                   pl.BlockSpec((R, None, bq), spec_lse)],
        out_shape=[jax.ShapeDtypeStruct((B, H, Sq, D), q.dtype), jax.ShapeDtypeStruct((B, H, Sq), jnp.float32)],
        name=f"kopt_rowattn_fwd_r{R}_{order}", **kw,
    )(q, k, v, bias, kmask)
    return o, lse


# ----------------------------------------------------------------------------------------------- index safety
def max_rows_i32(H, Sq, Sk, D):
    """Largest batch-row count of ONE pallas_call whose largest operand (q / o [B,H,Sq,D], k / v [B,H,Sk,D]) stays below 2**31 elements."""
    per_row = H * max(Sq, Sk) * D
    return (I32_LIMIT - 1) // per_row


def check_index_space(B, H, Sq, Sk, D):
    """Raises RowAttnRefusal('index_ge_2p31') when the bias or a single row alone passes 2**31 elements; returns the row chunk (<= B)."""
    if H * Sq * Sk >= I32_LIMIT:
        raise RowAttnRefusal("index_ge_2p31", f"pair bias [H={H},Sq={Sq},Sk={Sk}] has {H*Sq*Sk} elements >= 2**31 (int32 element offsets)")
    mr = max_rows_i32(H, Sq, Sk, D)
    if mr < 1:
        raise RowAttnRefusal("index_ge_2p31", f"one batch row [H={H},S={max(Sq,Sk)},D={D}] has >= 2**31 elements")
    return min(B, mr)


def _fwd(q, k, v, bias, kmask, **kw):
    B, H, Sq, D = q.shape; Sk = k.shape[2]
    chunk = check_index_space(B, H, Sq, Sk, D)
    R = int(kw["rows"])
    if chunk < B:
        chunk = max(R, (chunk // R) * R)                        # whole row blocks per chunk
    if chunk >= B:
        return _fwd_call(q, k, v, bias, kmask, **kw)
    outs = [_fwd_call(q[s:s + chunk], k[s:s + chunk], v[s:s + chunk], bias, kmask[s:s + chunk], **kw) for s in range(0, B, chunk)]
    return jnp.concatenate([o for o, _ in outs], axis=0), jnp.concatenate([l for _, l in outs], axis=0)


# ----------------------------------------------------------------------------------------------- public op
def _carried_kernel():
    """The carried kernel module of this process (the routed top-level ``af2_flash_pallas`` when a kit bound it, else the core's copy)."""
    m = sys.modules.get("af2_flash_pallas")
    if m is not None:
        return m
    try:
        from ..pallas_attn import af2_flash_pallas as m          # noqa: WPS433
        return m
    except Exception:  # noqa: BLE001 — loaded as a loose file (tests / the microbench)
        import importlib
        return importlib.import_module("af2_flash_pallas")


_BUILT = {}


def make_rowshared_attention(bq=64, bk=64, num_warps=4, num_stages=2, rows=1, order="legacy", group_rows=32, f32_precision=None,
                             differentiable=True, interpret=None):
    """Build the op ``f(q, k, v, bias, kmask, sm_scale) -> o`` (the carried op's signature; ``f.with_lse`` returns ``(o, lse)``).
    rows ∈ {1, 2, 4, 8}; order ∈ ORDERS; group_rows = rows per launch group of order='grouped' (a multiple of `rows`).
    differentiable=True wires the backward seam to the carried kernel's backward (its default backward tiles); False raises by name."""
    if int(rows) not in (1, 2, 4, 8):
        raise RowAttnRefusal("bad_rows", f"rows={rows!r} not in (1, 2, 4, 8)")
    if order not in ORDERS:
        raise RowAttnRefusal("bad_order", f"order={order!r} not in {ORDERS}")
    if order == "grouped" and (int(group_rows) < int(rows) or int(group_rows) % int(rows)):
        raise RowAttnRefusal("bad_order", f"group_rows={group_rows} must be a positive multiple of rows={rows}")
    for name, t in (("bq", bq), ("bk", bk)):
        if int(t) < 16 or (int(t) & (int(t) - 1)):
            raise RowAttnRefusal("bad_tiles", f"{name}={t} must be a power of two >= 16")
    f32p = f32_precision if f32_precision is not None else (os.environ.get("AF_PALLAS_ATTN_F32_PRECISION", "") or "ieee")
    f32p = {"highest": "ieee"}.get(f32p, f32p)
    if f32p not in F32_PRECISIONS:
        raise ValueError(f"f32_precision={f32p!r} not in {F32_PRECISIONS}")
    interp = (os.environ.get("KOPT_ROWATTN_INTERPRET", "0") == "1") if interpret is None else bool(interpret)
    kw = dict(bq=int(bq), bk=int(bk), num_warps=int(num_warps), num_stages=int(num_stages), rows=int(rows), order=order, group_rows=int(group_rows),
              f32p=f32p, interpret=interp)

    def _check(q):
        if q.shape[-1] < 16:
            raise RowAttnRefusal("head_dim_lt_16", f"D={q.shape[-1]} (Triton dot operands need >= 16; zero-pad as pallas_attn_serve.pad_head_dim does)")

    @functools.partial(jax.custom_vjp, nondiff_argnums=(5,))
    def rowshared_attn(q, k, v, bias, kmask, sm_scale):
        _check(q)
        return _fwd(q, k, v, bias, kmask, sm_scale=sm_scale, **kw)[0]

    def f_fwd(q, k, v, bias, kmask, sm_scale):
        _check(q)
        o, lse = _fwd(q, k, v, bias, kmask, sm_scale=sm_scale, **kw)
        return o, (q, k, v, bias, kmask, o, lse)

    def f_bwd(sm_scale, res, do):
        if not differentiable:
            raise RowAttnRefusal("backward_not_built", "built with differentiable=False; the design loop uses af2_flash_pallas.make_flash_attention")
        K0 = _carried_kernel()
        q, k, v, bias, kmask, o, lse = res
        dq, dk, dv, dbias = K0._bwd(q, k, v, bias, kmask, o, lse, do, sm_scale=sm_scale, bq=64, bk=64, num_warps=4, num_stages=2,
                                    dbias_dtype=bias.dtype, precise=False, f32p="ieee", dq_mode="kernel", dbias_mode="xla")
        return dq, dk, dv, dbias, None

    rowshared_attn.defvjp(f_fwd, f_bwd)

    def with_lse(q, k, v, bias, kmask, sm_scale):
        _check(q)
        return _fwd(q, k, v, bias, kmask, sm_scale=sm_scale, **kw)

    meta = dict(kw, differentiable=bool(differentiable))
    _BUILT[id(rowshared_attn)] = meta
    try:
        rowshared_attn.with_lse = with_lse; rowshared_attn.tiles = dict(meta)
    except Exception:  # noqa: BLE001
        pass
    return rowshared_attn


def describe_op(op):
    return dict(_BUILT.get(id(op), {}))


def finite_sentinel(o):
    """NaN / non-finite sentinel of the writer: a scalar bool (True = every element finite). jit-safe; the caller decides what to do with it."""
    return jnp.all(jnp.isfinite(o.astype(jnp.float32)))


def reference_attention_f64(q, k, v, bias, kmask, sm_scale):
    """float64 reference on the host (numpy): softmax(q·kᵀ·scale + bias, keys masked)·v from the SAME (rounded) inputs."""
    import numpy as np
    q64, k64, v64, b64 = (np.asarray(x.astype(jnp.float32)).astype(np.float64) for x in (q, k, v, bias))
    m = np.asarray(kmask)
    s = np.einsum("bhqd,bhkd->bhqk", q64, k64) * float(sm_scale) + b64[None]
    s = np.where(m[:, None, None, :], s, -np.inf)
    s = s - s.max(axis=-1, keepdims=True)
    p = np.exp(s); p = p / p.sum(axis=-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", p, v64)
