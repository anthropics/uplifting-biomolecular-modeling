"""opm_pallas.py - OuterProductMean (AlphaFold 2 / AlphaFold-Multimer Evoformer, Suppl. Alg. 10) without the transposed [N, N, C1*C2]
intermediate: JAX, Pallas (Triton lowering) + XLA, INFERENCE ONLY (forward; the op is a pure function of arrays - a jax.custom_vjp seam).

The stock op (alphafold/model/modules.py::OuterProductMean, monomer and multimer share the class; act = LayerNorm(msa), C1 = C2 = 32, F = c_z = 128):
    left  = mask[..., None] * Linear(C1)(act)            [S, N, C1]      right = mask[..., None] * Linear(C2)(act)      [S, N, C2]
    per 128-column chunk of `left` (mapping.inference_subbatch(chunk_size = config.outer_product_mean.chunk_size = 128; NOT global_config.subbatch_size)):
        O   = einsum('acb,ade->dceb', left_chunk^T, right)   [N, C1, C2, 128]   GEMM1 (K = S)     -> written, then PHYSICALLY TRANSPOSED (read + write)
        out = einsum('dceb,cef->dbf', O, output_w) + output_b [N, 128, F]        GEMM2 (K = C1*C2) -> reads it again; transposed into out[chunk]
    norm = einsum('abc,adc->bdc', mask, mask)            [N, N, 1]  (in act.dtype: bf16 in a bf16 model - integers above 256 are rounded)
    out  = out / (1e-3 + norm)                            the bias is INSIDE the division; the mean's normaliser is applied AFTER the projection
On an H100 the optimised HLO of the bf16 model (stored audit, N = 705, S = 512) is, per chunk: transpose fusion (left slice) -> cuBLAS GEMM1
bf16[N*C2, C1*128] -> input_transpose_fusion bf16[N,128,C1,C2] (a full read + write of the intermediate) -> cuBLAS GEMM2 -> bias + transpose +
dynamic-update-slice fusion; then the divide fusion over [N,N,F]. The [N, N, C1*C2] intermediate (2 GiB in bf16 at N = 1,024) crosses HBM FOUR times.

This file provides three routes with the same signature `(left, right, norm, w, b) -> [N, N, F]`:
  * `opm_two_launch`  ("pallas2"): GEMM1 as ONE XLA/cuBLAS GEMM per row chunk in its NATURAL output layout O[(i,c),(j,e)] (no transpose exists to be
        materialised), then ONE Pallas kernel that reads each O tile [bi, C1, bj, C2] straight from that layout, contracts it with output_w
        (C1 dots of K = C2, or C1/cc dots of K = cc*C2, f32 accumulator [bi*bj, F]), adds the bias, divides by (eps + norm) and writes [bi, bj, F].
        The intermediate crosses HBM twice (GEMM1 write, kernel read); the bias / transpose / dynamic-update-slice / divide passes over [N,N,F] are gone.
  * `opm_reassoc`     ("reassoc", pure XLA, differentiable as is): for SHORT stacks (`reassoc_wins`: S < C1*F/(F - C1) = 42.7) contract left with
        output_w first: T[s,i,e,f] = sum_c left[s,i,c] w[c,e,f]; out[i,j,f] = sum_{s,e} right[s,j,e] T[s,i,e,f].  FLOPs 2*S*N^2*C2*F instead of
        2*N^2*C1*C2*(S + F) and NO [N,N,C1*C2] intermediate at all (af2ig runs S = 5; the design loop runs S = 1..few).
  * `opm_stock`       the stock body op for op (chunked scan, same einsum strings, same rounding points) - the A/B arm and the fallback.
Numerics: bf16 operands where stock is bf16; every accumulator f32; O is rounded to the input dtype exactly where stock rounds it (GEMM1's output);
bias add, normaliser and division in f32 with ONE final rounding (stock rounds after GEMM2, after the bias add, and after the divide, and rounds the
normaliser itself to bf16). `norm` is passed in f32 (exact integer counts up to 2^24).  Not bit-identical to stock.
Index width: Pallas-Triton on this jax line switches its pointer offsets to 64 bit by itself when an operand exceeds 2^32 BYTES
(`_compute_offsets_from_indices`); the per-call operands here are bounded by `chunk_bytes` (default 512 MiB) anyway, and `check_index_width`
refuses BY NAME (`index_over_int32`) any call whose per-launch element count could pass 2^31 - 1.
"""
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

_CP = getattr(plgpu, "CompilerParams", None) or getattr(plgpu, "TritonCompilerParams")
F32, BF16 = jnp.float32, jnp.bfloat16
EPS = 1e-3                                   # stock: epsilon = 1e-3 in OuterProductMean.__call__
STOCK_CHUNK = 128                            # config.model.embeddings_and_evoformer.evoformer.outer_product_mean.chunk_size (monomer and multimer)
INT32_MAX = 2 ** 31 - 1
INDEX_OVER_INT32 = "index_over_int32"
DEFAULT_CHUNK_BYTES = 512 * 2 ** 20          # cap of the per-chunk intermediate O[ci*C1, Nj*C2] (the stock chunk holds N*C1*C2*128 elements: 256 MiB at N = 1,024 in bf16)
DEFAULT_CFG = dict(bi=2, bj=64, cc=1, warps=4, stages=3)     # the SAFE row: cc = 1 needs no in-kernel transpose; every tile a power of two; bi*bj >= 16
PRECISIONS = ("tf32", "tf32x3", "ieee")      # f32 operands only: the MMA operand class of the Pallas projection kernel (bf16 operands ignore it)


class IndexWidthError(ValueError):
    """`index_over_int32`: a per-launch operand of this call could be indexed past 2^31 - 1 elements (refused by name, never wrapped around)."""
    kind = INDEX_OVER_INT32


def check_index_width(**elements) -> None:
    for name, n in elements.items():
        if int(n) > INT32_MAX:
            raise IndexWidthError(f"{INDEX_OVER_INT32}: {name} has {int(n)} elements > 2^31-1 in one launch; lower chunk_bytes")


def _dot_precision(dtype, precision):
    """The `precision=` of the in-kernel dots. bf16 / f16 operands: None (bf16 x bf16 -> f32).  f32 operands: a NAMED class - 'tf32' (Precision.DEFAULT:
    tensor-core tf32 operands; on the jax 0.5 line Pallas-Triton TRUNCATES them - the class recorded in fpf_pallas_serve.F32_LINES), 'tf32x3', 'ieee'."""
    if jnp.dtype(dtype) != jnp.dtype(F32):
        return None
    if precision == "tf32":
        return lax.Precision.DEFAULT
    if precision == "ieee":
        return lax.Precision.HIGHEST
    if precision == "tf32x3":
        preset = getattr(getattr(lax, "DotAlgorithmPreset", None), "TF32_TF32_F32_X3", None)
        if preset is None:
            raise ValueError("precision 'tf32x3' needs lax.DotAlgorithmPreset.TF32_TF32_F32_X3 (this jax has none)")
        return preset
    raise ValueError(f"f32 operands need a named precision in {PRECISIONS}, got {precision!r}")


# ------------------------------------------------------------------------------------------------------------- Pallas: projection + epilogue
def _proj_kernel(o_ref, w_ref, b_ref, n_ref, out_ref, *, bi, bj, c1, c2, f, cc, precision):
    """One (i-block, j-block) tile: acc[(i,j), f] = sum_c sum_e O[i, c, j, e] * w[c, e, f]; out = (acc + b) / (eps + norm)."""
    steps = c1 // cc

    def body(t, acc):
        cs = pl.ds(t * cc, cc)
        x = o_ref[:, cs, :, :]                                        # [bi, cc, bj, c2]   (for each i, c: bj*c2 contiguous elements of O)
        w = w_ref[cs, :, :]                                           # [cc, c2, f]
        if cc == 1:
            x = x.reshape(bi * bj, c2)                                # row-major merge of (i, j): no data movement between lanes
            w = w.reshape(c2, f)
        else:
            x = jnp.transpose(x, (0, 2, 1, 3)).reshape(bi * bj, cc * c2)   # [(i,j), (c,e)]
            w = w.reshape(cc * c2, f)
        return acc + jnp.dot(x, w, preferred_element_type=F32, precision=precision)

    acc = lax.fori_loop(0, steps, body, jnp.zeros((bi * bj, f), F32))
    acc = acc + b_ref[...].astype(F32)[None, :]
    den = EPS + n_ref[...].astype(F32).reshape(bi * bj, 1)
    out_ref[...] = (acc / den).reshape(bi, bj, f).astype(out_ref.dtype)


def opm_project_pallas(o4, w, b, norm, *, out_dtype=None, bi=2, bj=64, cc=1, warps=4, stages=3, precision=None, interpret=False):
    """o4 [Ni, C1, Nj, C2] (GEMM1's natural layout, viewed 4-D), w [C1, C2, F], b [F], norm [Ni, Nj] f32 -> [Ni, Nj, F] in `out_dtype` (o4.dtype).
    Ni % bi == 0, Nj % bj == 0, C1 % cc == 0; bi, bj, cc, C2, F powers of two; bi*bj >= 16 (the MMA's M)."""
    ni, c1, nj, c2 = o4.shape
    f = w.shape[-1]
    assert w.shape == (c1, c2, f) and b.shape == (f,) and norm.shape == (ni, nj), (o4.shape, w.shape, b.shape, norm.shape)
    assert ni % bi == 0 and nj % bj == 0 and c1 % cc == 0 and bi * bj >= 16, (o4.shape, bi, bj, cc)
    for v in (bi, bj, cc, c2, f):
        assert v & (v - 1) == 0, f"tile sizes must be powers of two, got {(bi, bj, cc, c2, f)}"
    check_index_width(o4=o4.size, out=ni * nj * f)
    out_dtype = out_dtype or o4.dtype
    kern = functools.partial(_proj_kernel, bi=bi, bj=bj, c1=c1, c2=c2, f=f, cc=cc, precision=_dot_precision(o4.dtype, precision))
    kw = dict(interpret=True) if interpret else dict(compiler_params=_CP(num_warps=warps, num_stages=stages))
    return pl.pallas_call(
        kern, grid=(ni // bi, nj // bj),
        in_specs=[pl.BlockSpec((bi, c1, bj, c2), lambda i, j: (i, 0, j, 0)),
                  pl.BlockSpec((c1, c2, f), lambda i, j: (0, 0, 0)),
                  pl.BlockSpec((f,), lambda i, j: (0,)),
                  pl.BlockSpec((bi, bj), lambda i, j: (i, j))],
        out_specs=pl.BlockSpec((bi, bj, f), lambda i, j: (i, j, 0)),
        out_shape=jax.ShapeDtypeStruct((ni, nj, f), out_dtype),
        name="opm_project", **kw,
    )(o4, w.astype(o4.dtype), b, norm)


# ------------------------------------------------------------------------------------------------------------- route "pallas2"
def plan_rows(n, c1, c2, itemsize, bi, bj, chunk_bytes=DEFAULT_CHUNK_BYTES):
    """(ni_pad, nj_pad, ci, n_chunks): columns padded to a multiple of bj; rows split into n_chunks chunks of ci rows (ci a multiple of bi) so that one
    chunk's intermediate O[ci*C1, nj_pad*C2] is <= chunk_bytes (at least one row tile)."""
    nj = -(-n // bj) * bj
    row_bytes = c1 * nj * c2 * itemsize
    max_rows = max(bi, (chunk_bytes // row_bytes) // bi * bi)
    n_chunks = -(-n // max_rows)
    ci = -(-(-(-n // n_chunks)) // bi) * bi
    return ci * n_chunks, nj, ci, n_chunks


def mask_norm(mask):
    """norm[i, j] = sum_s mask[s, i] mask[s, j] in f32 at HIGHEST precision (0/1 products, exact integer counts below 2^24). mask [S, N]."""
    m = mask.astype(F32)
    return lax.dot_general(m, m, (((0,), (0,)), ((), ())), precision=lax.Precision.HIGHEST, preferred_element_type=F32)


def opm_two_launch(left, right, norm, w, b, *, cfg=None, chunk_bytes=DEFAULT_CHUNK_BYTES, precision=None, interpret=False):
    """left [S, N, C1], right [S, N, C2] (already masked, model dtype), norm [N, N] f32 (`mask_norm`), w [C1, C2, F], b [F] -> [N, N, F] in left.dtype."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    bi, bj = int(cfg["bi"]), int(cfg["bj"])
    s, n, c1 = left.shape
    c2 = right.shape[-1]
    f = w.shape[-1]
    dt = left.dtype
    ni, nj, ci, n_chunks = plan_rows(n, c1, c2, jnp.dtype(dt).itemsize, bi, bj, chunk_bytes)
    lp = jnp.pad(left, ((0, 0), (0, ni - n), (0, 0))) if ni != n else left
    rp = jnp.pad(right, ((0, 0), (0, nj - n), (0, 0))) if nj != n else right
    npad = jnp.pad(norm.astype(F32), ((0, ni - n), (0, nj - n))) if (ni != n or nj != n) else norm.astype(F32)
    r2 = rp.reshape(s, nj * c2)
    lc = lp.reshape(s, n_chunks, ci * c1)
    nc = npad.reshape(n_chunks, ci, nj)

    def one(args):
        l2, nrm = args                                                  # [S, ci*C1], [ci, nj]
        o = lax.dot_general(l2, r2, (((0,), (0,)), ((), ())), preferred_element_type=dt)     # [(i,c), (j,e)] - the GEMM's own output order
        return opm_project_pallas(o.reshape(ci, c1, nj, c2), w, b, nrm, out_dtype=dt, bi=bi, bj=bj, cc=int(cfg["cc"]),
                                  warps=int(cfg["warps"]), stages=int(cfg["stages"]), precision=precision, interpret=interpret)

    if n_chunks == 1:
        out = one((lc[:, 0], nc[0]))
    else:
        out = lax.map(one, (jnp.moveaxis(lc, 1, 0), nc)).reshape(ni, nj, f)
    return out[:n, :n] if (ni != n or nj != n) else out


# ------------------------------------------------------------------------------------------------------------- route "reassoc" (pure XLA)
def reassoc_wins(s, c1=32, c2=32, f=128) -> bool:
    """FLOPs of the re-associated order 2*S*N^2*C2*F (+ the small T GEMM) against the stock order 2*N^2*C1*C2*(S + F): wins when S*F < C1*(S + F),
    i.e. S < C1*F/(F - C1) (42.7 at C1 = 32, F = 128) - and it never forms the [N, N, C1*C2] intermediate."""
    return s * f < c1 * (s + f)


def opm_reassoc(left, right, norm, w, b, *, chunk_bytes=DEFAULT_CHUNK_BYTES, precision=None):
    """Same signature as `opm_two_launch`; pure jnp (jax.grad works through it as is). T is rounded to the model dtype (the one intermediate rounding,
    where stock rounds O); accumulators, bias, normaliser and division in f32; one final rounding."""
    s, n, c1 = left.shape
    c2 = right.shape[-1]
    f = w.shape[-1]
    dt = left.dtype
    t = jnp.einsum("sic,cef->sief", left, w.astype(dt), preferred_element_type=F32, precision=precision).astype(dt)      # [S, N, C2, F]
    bf = b.astype(F32)
    rows = max(1, min(n, chunk_bytes // (n * f * 4)))                    # rows of the f32 accumulator [rows, N, F] per chunk
    n_chunks = -(-n // rows)
    rows = -(-n // n_chunks)
    npad = rows * n_chunks

    def one(args):
        tc, nrm = args                                                   # [S, rows, C2, F], [rows, N]
        acc = jnp.einsum("sje,sief->ijf", right, tc, preferred_element_type=F32, precision=precision)
        return ((acc + bf) / (EPS + nrm)[..., None]).astype(dt)

    nf = norm.astype(F32)
    if n_chunks == 1:
        return one((t, nf))
    tp = jnp.pad(t, ((0, 0), (0, npad - n), (0, 0), (0, 0))) if npad != n else t
    nfp = jnp.pad(nf, ((0, npad - n), (0, 0))) if npad != n else nf
    out = lax.map(one, (jnp.moveaxis(tp.reshape(s, n_chunks, rows, c2, f), 1, 0), nfp.reshape(n_chunks, rows, n))).reshape(npad, n, f)
    return out[:n]


# ------------------------------------------------------------------------------------------------------------- stock body + references
def opm_stock(left, right, mask, w, b, *, chunk_size=STOCK_CHUNK):
    """The stock body op for op (alphafold-colabfold 2.3.13 modules.py:1649-1672 + mapping.sharded_apply's scan + remainder), for arrays already
    projected and masked: left/right [S, N, C], mask [S, N] in the model dtype, w [C1, C2, F], b [F] in the model dtype -> [N, N, F]."""
    dt = left.dtype
    s, n, c1 = left.shape
    f = w.shape[-1]

    def compute_chunk(left_chunk):
        lt = jnp.transpose(left_chunk, [0, 2, 1])
        act = jnp.einsum("acb,ade->dceb", lt, right)
        act = jnp.einsum("dceb,cef->dbf", act, w.astype(dt)) + b.astype(dt)
        return jnp.transpose(act, [1, 0, 2])

    n_full, rem = divmod(n, chunk_size)
    if rem == 0 and n_full > 0:
        n_scan, last = n_full - 1, chunk_size                            # sharded_apply: num_extra_shards = (in_size - 1) // shard_size; the last shard is the remainder call
    else:
        n_scan, last = n_full, rem
    out = jnp.zeros((n, n, f), dt)

    def step(o, start):
        blk = compute_chunk(lax.dynamic_slice_in_dim(left, start, chunk_size, axis=1))
        return lax.dynamic_update_slice_in_dim(o, blk, start, axis=0), ()

    if n_scan > 0:
        out, _ = lax.scan(step, out, jnp.arange(n_scan) * chunk_size)
    blk = compute_chunk(lax.dynamic_slice_in_dim(left, n - last, last, axis=1))
    out = lax.dynamic_update_slice_in_dim(out, blk, n - last, axis=0)
    m = mask.astype(dt)[..., None]
    nrm = jnp.einsum("abc,adc->bdc", m, m)
    return out / (EPS + nrm)


def opm_reference(left, right, mask, w, b, *, dtype=jnp.float64):
    """The op in `dtype` (float64 under jax_enable_x64 = the reference; float32 at HIGHEST otherwise) from the SAME rounded inputs (left, right, w, b as the
    model holds them): out[i,j,f] = (sum_{s,c,e} left[s,i,c] right[s,j,e] w[c,e,f] + b[f]) / (eps + sum_s mask[s,i] mask[s,j])."""
    hp = lax.Precision.HIGHEST
    l, r, m = left.astype(dtype), right.astype(dtype), mask.astype(dtype)
    o = jnp.einsum("sic,sje->icje", l, r, precision=hp)
    acc = jnp.einsum("icje,cef->ijf", o, w.astype(dtype), precision=hp) + b.astype(dtype)
    nrm = jnp.einsum("si,sj->ij", m, m, precision=hp)
    return acc / (EPS + nrm)[..., None]


# ------------------------------------------------------------------------------------------------------------- static cost model
def cost_model(n, s, *, c1=32, c2=32, f=128, itemsize=2, chunk=STOCK_CHUNK):
    """HBM bytes and FLOPs of ONE OuterProductMean call per route (static; what each lowering materialises - stock rows from the stored H100 HLO audit).
    Returns {route: {flops, bytes, x_passes, detail}}; X = the [N, N, C1*C2] intermediate in the model dtype."""
    x = n * n * c1 * c2 * itemsize
    out = n * n * f * itemsize
    lr = s * n * (c1 + c2) * itemsize
    n_chunks = -(-n // chunk)
    g1, g2 = 2.0 * s * n * n * c1 * c2, 2.0 * n * n * c1 * c2 * f
    stock = dict(flops=g1 + g2, x_passes=4,
                 bytes=4 * x                       # GEMM1 write, transpose fusion read + write, GEMM2 read
                       + n_chunks * s * n * c2 * itemsize + s * n * c1 * itemsize * 2      # right re-read per chunk; left slice + its transpose
                       + out * (1 + 2 + 2) + 2 * n * n * itemsize,   # GEMM2 write; bias/transpose/DUS fusion r+w; divide fusion r+w; norm r/w
                 detail="GEMM1 -> transpose fusion -> GEMM2 -> bias+transpose+DUS fusion -> divide fusion (H100 HLO audit)")
    p2_chunks = max(1, -(-x // DEFAULT_CHUNK_BYTES))
    pallas2 = dict(flops=g1 + g2, x_passes=2,
                   bytes=2 * x + p2_chunks * s * n * c2 * itemsize + s * n * c1 * itemsize + out + n * n * 4 + p2_chunks * c1 * c2 * f * itemsize,
                   detail="GEMM1 (natural layout) -> Pallas projection+bias+normalise (reads O once, writes [N,N,F] once)")
    fused = dict(flops=g1 + g2, x_passes=0, bytes=lr + out + n * n * 4,
                 detail="ideal single kernel: reads left/right, writes [N,N,F] (operand re-reads served by L2)")
    t_bytes = s * n * c2 * f * itemsize
    reassoc = dict(flops=2.0 * s * n * c1 * c2 * f + 2.0 * s * n * n * c2 * f, x_passes=0,
                   bytes=lr + 2 * t_bytes + n * n * f * 4 * 2 + out + n * n * 4,
                   detail="T = left.w (S*N*C2*F), out = right.T batched over i; f32 accumulator written + read by the epilogue fusion")
    return dict(stock=stock, pallas2=pallas2, fused_ideal=fused, reassoc=reassoc)
