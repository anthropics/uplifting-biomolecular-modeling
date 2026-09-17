"""mlp_transition_pallas.py — the Evoformer / Pairformer *Transition* (LayerNorm -> Linear c->F -> activation -> Linear F->c) as ONE Pallas
(Triton lowering) GPU kernel per row tile, inference forward. Engine-free pure functions over arrays; jax + pallas at import, nothing else.

It GENERALIZES the carried AlphaFold-3-family kernel ``fpf_pallas/transition_pallas.py`` (LN -> SwiGLU -> W2, bias-free, bf16, rows % T == 0),
whose bytes are untouched (the AF3-JAX fast mode keeps importing that file); this module adds what the AlphaFold-2 / AF-Multimer stack needs:

    activation = "relu"     out = relu(LN(x) @ W1 + b1) @ W2 + b2          (alphafold.model.modules.Transition: Jumper 2021 Alg. 9 / 15)
    activation = "swiglu"   out = (swish(LN(x) @ Wa) * (LN(x) @ Wb)) @ W2   (+ b2)   (the AF3 family; same math as the carried kernel)

    * Linear BIASES (b1 [F], b2 [C]) — AF2's Linears carry them; ``None`` = bias-free (AF3).
    * LayerNorm variance in the stock form of the engine: ``fast_variance=False`` -> mean((x-mean)^2) (AF2: hk.LayerNorm default),
      ``True`` -> E[x^2]-E[x]^2 (AF3: use_fast_variance=True). Statistics are ALWAYS float32.
    * ANY number of rows M (masked loads / stores on the ragged last tile: N^2 rows of an odd N need no pad copy).
    * bfloat16 AND float32 activations. bf16: bf16 x bf16 -> f32 MMAs, LN output / hidden activation rounded to bf16 at the stock rounding
      points. f32 (the af2ig stack): f32 operands at a NAMED precision word — "tf32" (tensor cores; Pallas-Triton on the jax 0.5 line converts
      by TRUNCATION), "tf32r" (operands pre-rounded to NEAREST tf32 in-kernel, so the truncating conversion is exact: the class of an XLA
      default-precision f32 GEMM on sm_80+ on every jax line), "ieee" (lax HIGHEST: CUDA-core fp32, validation only).
    * NaN sentinel: ``sentinel=True`` adds a second output, int32 [num_tiles] = the count of non-finite values each program was about to write.
    * int32 guard: element offsets are int32 inside the lowering; ``check_index_range`` refuses BY NAME (``elements_ge_2p31``) when
      M * C >= 2**31; the serve layer splits such a call into row slabs (pair N = 4,096 at C = 128 is exactly 2**31 elements).

What never reaches HBM: the [M, F] hidden activation (F = 4C: [N, N, 512] pair / [S, N, 1024] MSA), the LN output, the GEMM1 output before
its bias + ReLU. HBM traffic per call: read x [M, C] once, write out [M, C] once, weights once (L2 afterwards).  Accumulators: float32.

Rounding points vs the stock bf16 body (XLA): stock rounds LN -> bf16, GEMM1 -> bf16, +b1 -> bf16, GEMM2 -> bf16, +b2 -> bf16; the kernel rounds
LN -> bf16, relu(a + b1) -> bf16 (a, b1 in f32), and the final acc + b2 -> bf16: a strict subset of stock's rounding points, f32 in between.
Not bit-exact to the XLA body (different accumulation order): Tier 2 — error vs an f64 reference at or below the stock body's.

Backward: not built here. ``make_transition_op(bwd=...)`` is the custom_vjp seam: ``bwd=None`` (default) = forward only, differentiation raises
by name; ``bwd="reference"`` = the VJP of the op-by-op XLA replica evaluated at the same inputs (correct gradients of the reference math, no
fused backward kernel, materializes [M, F] in the backward only) — the design loop's stop-gap; a fused backward is a separate kernel pair
(dX/dLN and dW reductions) the design captain would own.
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

_CP = getattr(plgpu, "TritonCompilerParams", None) or getattr(plgpu, "CompilerParams")
F32, BF16 = jnp.float32, jnp.bfloat16
LN_EPS = 1e-5
ACTIVATIONS = ("relu", "swiglu")
F32_PRECISIONS = ("tf32", "tf32r", "ieee")
INT32_LIMIT = 2 ** 31
DEFAULT_CFG = dict(t=64, tf=64, num_warps=4, num_stages=2)     # provisional: the carried AF3 kernel's H100 default; the autotune job writes the tile table
ELEMENTS_GE_2P31 = "elements_ge_2p31"

_PL_LOAD, _PL_STORE = getattr(pl, "load", None), getattr(pl, "store", None)   # jax 0.5-0.7 carry pl.load / pl.store; newer lines: plgpu.load / store on a ref view


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


class NotServed(ValueError):
    """A shape / dtype / index range the kernel does not serve: ``kind`` is the refusal's NAME (the caller routes to stock by it)."""
    def __init__(self, kind, detail=""):
        self.kind, self.detail = kind, detail
        super().__init__(f"mlp_transition: {kind}" + (f" — {detail}" if detail else ""))


def round_to_tf32(x):
    """f32 -> the nearest tf32-representable f32 (10 explicit mantissa bits; ties away from zero in magnitude; inf / nan pass through):
    add half an ulp of the 13 dropped bits to the magnitude bits, clear them. A later TRUNCATING f32->tf32 conversion is then exact."""
    u = jax.lax.bitcast_convert_type(x.astype(F32), jnp.uint32)
    is_special = (u & jnp.uint32(0x7F800000)) == jnp.uint32(0x7F800000)
    r = (u + jnp.uint32(0x00001000)) & jnp.uint32(0xFFFFE000)
    return jax.lax.bitcast_convert_type(jnp.where(is_special, u, r), F32)


def _dot(a, b, f32p):
    """f32 accumulate. bf16 operands: tensor-core bf16 x bf16 -> f32. f32 operands: the precision word."""
    if a.dtype == F32:
        if f32p == "ieee":
            return jnp.dot(a, b, preferred_element_type=F32, precision=jax.lax.Precision.HIGHEST)
        if f32p == "tf32r":
            a, b = round_to_tf32(a), round_to_tf32(b)
        return jnp.dot(a, b, preferred_element_type=F32, precision=jax.lax.Precision.DEFAULT)
    return jnp.dot(a, b, preferred_element_type=F32)


def _layer_norm(x, scale, offset, eps, fast_variance):
    """x [T, C] f32 -> LN(x) f32; hk.LayerNorm's own expression: inv = scale * rsqrt(var + eps); inv * (x - mean) + offset."""
    mean = jnp.mean(x, axis=1, keepdims=True)
    if fast_variance:
        var = jnp.mean(x * x, axis=1, keepdims=True) - mean * mean
    else:
        xc = x - mean
        var = jnp.mean(xc * xc, axis=1, keepdims=True)
    inv = scale[None, :] * jax.lax.rsqrt(var + eps)
    return inv * (x - mean) + offset[None, :]


def _kernel(*refs, nf, tf, t, m_rows, eps, activation, fast_variance, has_b1, has_b2, sentinel, f32p, cdt):
    it = iter(refs)
    x_ref, s_ref, o_ref, w1_ref = next(it), next(it), next(it), next(it)
    wb_ref = next(it) if activation == "swiglu" else None
    b1_ref = next(it) if has_b1 else None
    w2_ref = next(it)
    b2_ref = next(it) if has_b2 else None
    out_ref = next(it)
    flag_ref = next(it) if sentinel else None

    rows = pl.program_id(0) * t + jnp.arange(t, dtype=jnp.int32)
    valid = rows < m_rows                                                     # the ragged last tile: masked loads (0) and masked stores
    x = _load(x_ref, (slice(None), slice(None)), mask=valid[:, None], other=0.0).astype(F32)     # [T, C]
    xn = _layer_norm(x, s_ref[...], o_ref[...], eps, fast_variance).astype(cdt)                  # the stock rounding point (bf16 route); f32 route: no rounding
    T, C = x.shape

    def body(f, acc):
        fs = pl.ds(f * tf, tf)
        a = _dot(xn, w1_ref[:, fs], f32p)                                     # [T, tf] f32
        if has_b1:
            a = a + b1_ref[fs][None, :]
        if activation == "relu":
            h = jnp.maximum(a, 0.0)
        else:                                                                 # swiglu: swish(a) * b, the carried AF3 kernel's expression
            h = a * jax.nn.sigmoid(a) * _dot(xn, wb_ref[:, fs], f32p)
        return acc + _dot(h.astype(cdt), w2_ref[fs, :], f32p)                 # hidden rounded once (bf16 route) = the stock GEMM2 operand dtype

    acc = jax.lax.fori_loop(0, nf, body, jnp.zeros((T, C), F32))
    if has_b2:
        acc = acc + b2_ref[...][None, :]
    if sentinel:                                                              # NaN / non-finite check AT THE WRITER: what this program is about to store
        bad = jnp.logical_and(jnp.logical_not(jnp.isfinite(acc)), valid[:, None])
        flag_ref[...] = jnp.sum(bad.astype(jnp.int32), dtype=jnp.int32).reshape((1,))
    _store(out_ref, (slice(None), slice(None)), acc.astype(out_ref.dtype), mask=valid[:, None])


def _is_pow2(n):
    return n >= 1 and (n & (n - 1)) == 0


def check_index_range(m_rows, c):
    """Raises NotServed(elements_ge_2p31) when a flat element offset of the call can reach 2**31 (int32 offsets inside the lowering)."""
    biggest = int(m_rows) * int(c)
    if biggest >= INT32_LIMIT:
        raise NotServed(ELEMENTS_GE_2P31, f"rows {m_rows} x C {c} = {biggest} >= 2**31 elements: split the rows (pallas_transition_serve.transition_block does) or run stock")


def supported(x_shape, x_dtype, c_hidden, cfg=None, activation="relu"):
    """(ok, reason) — never raises. reason words: activation:<a> · dtype:<t> · c:<C> · f:<F>%<tf> · tile:<t|tf> · elements_ge_2p31."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    C = int(x_shape[-1])
    if activation not in ACTIVATIONS:
        return False, f"activation:{activation}"
    if jnp.dtype(x_dtype) not in (jnp.dtype(BF16), jnp.dtype(F32)):
        return False, f"dtype:{jnp.dtype(x_dtype).name}"
    if not _is_pow2(C) or not (16 <= C <= 256):                              # Triton block shapes are powers of two; one program holds the [T, C] f32 accumulator
        return False, f"c:{C}"
    if not (_is_pow2(cfg["t"]) and _is_pow2(cfg["tf"]) and cfg["t"] >= 16 and cfg["tf"] >= 16):
        return False, f"tile:{cfg['t']}|{cfg['tf']}"
    if int(c_hidden) % cfg["tf"]:
        return False, f"f:{c_hidden}%{cfg['tf']}"
    M = 1
    for d in x_shape[:-1]:
        M *= int(d)
    if M * C >= INT32_LIMIT:
        return False, ELEMENTS_GE_2P31
    return True, "ok"


def fused_transition(x, ln_scale, ln_offset, w1, w2, b1=None, b2=None, *, activation="relu", eps=LN_EPS, fast_variance=False, cfg=None,
                     f32_precision=None, sentinel=False, interpret=False):
    """x [..., C] bf16 | f32; ln_scale / ln_offset [C] (None = ones / zeros); relu: w1 [C, F], b1 [F] | None; swiglu: w1 [C, 2F] (columns
    [:F] = a (swish), [F:] = b) or (C, 2, F), b1 must be None; w2 [F, C]; b2 [C] | None. Returns [..., C] in x.dtype (and, with
    ``sentinel=True``, an int32 [num_tiles] array of per-program non-finite counts). f32 inputs REQUIRE ``f32_precision`` in F32_PRECISIONS.
    Raises NotServed(kind) for anything ``supported`` refuses."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    C = int(x.shape[-1])
    if activation == "swiglu":
        if w1.ndim == 3:
            wa, wb = w1[:, 0, :], w1[:, 1, :]
        else:
            F2 = w1.shape[1]
            wa, wb = w1[:, : F2 // 2], w1[:, F2 // 2:]
        if b1 is not None:
            raise NotServed("swiglu_b1", "the SwiGLU form carries no first-layer bias")
    else:
        wa, wb = w1, None
    F = int(wa.shape[1])
    ok, why = supported(x.shape, x.dtype, F, cfg, activation)
    if not ok:
        raise NotServed(why.split(":")[0] if why != ELEMENTS_GE_2P31 else why, why)
    dt = x.dtype
    is_f32 = jnp.dtype(dt) == jnp.dtype(F32)
    if is_f32 and f32_precision not in F32_PRECISIONS:
        raise NotServed("precision_required", f"float32 activations name their MMA operand precision: {F32_PRECISIONS}")
    t, tf = int(cfg["t"]), int(cfg["tf"])
    lead = x.shape[:-1]
    M = 1
    for d in lead:
        M *= int(d)
    x2 = x.reshape(M, C)
    s = jnp.ones((C,), F32) if ln_scale is None else ln_scale.astype(F32)
    o = jnp.zeros((C,), F32) if ln_offset is None else ln_offset.astype(F32)
    full = lambda *shape: pl.BlockSpec(shape, lambda i: (0,) * len(shape))
    args, in_specs = [x2, s, o, wa.astype(dt)], [pl.BlockSpec((t, C), lambda i: (i, 0)), full(C), full(C), full(C, F)]
    if wb is not None:
        args.append(wb.astype(dt)); in_specs.append(full(C, F))
    if b1 is not None:
        args.append(b1.astype(F32)); in_specs.append(full(F))
    args.append(w2.astype(dt)); in_specs.append(full(F, C))
    if b2 is not None:
        args.append(b2.astype(F32)); in_specs.append(full(C))
    n_tiles = pl.cdiv(M, t)
    out_shape = [jax.ShapeDtypeStruct((M, C), dt)]
    out_specs = [pl.BlockSpec((t, C), lambda i: (i, 0))]
    if sentinel:
        out_shape.append(jax.ShapeDtypeStruct((n_tiles,), jnp.int32)); out_specs.append(pl.BlockSpec((1,), lambda i: (i,)))
    kern = functools.partial(_kernel, nf=F // tf, tf=tf, t=t, m_rows=M, eps=eps, activation=activation, fast_variance=bool(fast_variance),
                             has_b1=b1 is not None, has_b2=b2 is not None, sentinel=bool(sentinel), f32p=f32_precision, cdt=dt)
    res = pl.pallas_call(kern, grid=(n_tiles,), in_specs=in_specs, out_specs=out_specs, out_shape=out_shape,
                         compiler_params=_CP(num_warps=int(cfg["num_warps"]), num_stages=int(cfg["num_stages"])),
                         interpret=bool(interpret), name=f"mlp_transition_{activation}")(*args)
    out = res[0].reshape(*lead, C)
    return (out, res[1]) if sentinel else out


def transition_reference(x, ln_scale, ln_offset, w1, w2, b1=None, b2=None, *, activation="relu", eps=LN_EPS, fast_variance=False,
                         compute_dtype=F32, precision=None):
    """Op-by-op XLA replica. ``compute_dtype=float64`` (+ jax_enable_x64) = the float64 reference; ``compute_dtype=bfloat16`` = the stock
    bf16 body's rounding points (LN -> bf16, GEMM1 -> bf16, +b1 -> bf16, act, GEMM2 -> bf16, +b2 -> bf16: alphafold common_modules.Linear on
    bf16 activations); ``compute_dtype=float32`` + ``precision=HIGHEST`` = an f32 reference. Weights are first rounded to x.dtype (that IS
    the model under a bf16 context), then upcast."""
    cd, dt = compute_dtype, x.dtype
    stat = jnp.promote_types(cd, F32)
    x32 = x.astype(stat)
    C = x.shape[-1]
    s = (jnp.ones((C,)) if ln_scale is None else ln_scale).astype(stat)
    o = (jnp.zeros((C,)) if ln_offset is None else ln_offset).astype(stat)
    mean = jnp.mean(x32, -1, keepdims=True)
    var = (jnp.mean(jnp.square(x32), -1, keepdims=True) - jnp.square(mean)) if fast_variance else jnp.mean(jnp.square(x32 - mean), -1, keepdims=True)
    xn = ((s * jax.lax.rsqrt(var + jnp.asarray(eps, stat))) * (x32 - mean) + o).astype(cd)
    up = lambda w: w.astype(dt).astype(cd)
    kw = dict(precision=precision) if jnp.dtype(cd) != jnp.dtype(BF16) else {}
    ein = lambda a, b: jnp.einsum("...c,cf->...f", a, b, **kw).astype(cd)
    if activation == "swiglu":
        if w1.ndim == 3:
            wa, wb = w1[:, 0, :], w1[:, 1, :]
        else:
            F2 = w1.shape[1]; wa, wb = w1[:, : F2 // 2], w1[:, F2 // 2:]
        h = (jax.nn.swish(ein(xn, up(wa))) * ein(xn, up(wb))).astype(cd)
    else:
        a = ein(xn, up(w1))
        if b1 is not None:
            a = (a + b1.astype(cd)).astype(cd)
        h = jax.nn.relu(a)
    out = ein(h, up(w2))
    if b2 is not None:
        out = (out + b2.astype(cd)).astype(cd)
    return out


def make_transition_op(*, activation="relu", eps=LN_EPS, fast_variance=False, cfg=None, f32_precision=None, interpret=False, bwd=None):
    """The differentiable-wrappable seam: ``op(x, ln_scale, ln_offset, w1, w2, b1, b2) -> out`` as a ``jax.custom_vjp`` whose forward is the
    fused kernel. ``bwd=None``: forward only — asking for a gradient raises NotServed('backward_not_built'). ``bwd="reference"``: the VJP of
    ``transition_reference`` (float32 compute) at the same inputs; b1 / b2 may be None (their cotangents are None)."""
    kw = dict(activation=activation, eps=eps, fast_variance=fast_variance)

    @jax.custom_vjp
    def op(x, ln_scale, ln_offset, w1, w2, b1, b2):
        return fused_transition(x, ln_scale, ln_offset, w1, w2, b1, b2, cfg=cfg, f32_precision=f32_precision, interpret=interpret, **kw)

    def fwd(x, ln_scale, ln_offset, w1, w2, b1, b2):
        return op(x, ln_scale, ln_offset, w1, w2, b1, b2), (x, ln_scale, ln_offset, w1, w2, b1, b2)

    def bwd_fn(res, g):
        if bwd is None:
            raise NotServed("backward_not_built", "the fused transition is forward-only; make_transition_op(bwd='reference') differentiates the XLA replica")
        x, ln_scale, ln_offset, w1, w2, b1, b2 = res
        ref = lambda *a: transition_reference(*a, compute_dtype=F32, **kw).astype(x.dtype)
        _, vjp = jax.vjp(ref, x, ln_scale, ln_offset, w1, w2, b1, b2)
        return vjp(g)

    op.defvjp(fwd, bwd_fn)
    return op


def traffic_model(m_rows, c, f, itemsize=2):
    """Static HBM bytes / FLOPs of one call: the fused kernel vs the stock XLA lowering read off the GPU-optimized HLO (jax 0.5.3: LN fusion,
    slice copy, cuBLAS GEMM1, bias+ReLU loop fusion, cuBLAS GEMM2, bias + dynamic-update-slice fusion)."""
    xc, hf = m_rows * c * itemsize, m_rows * f * itemsize
    stock = dict(ln=2 * xc, slice=2 * xc, gemm1=xc + hf, bias_relu=2 * hf, gemm2=hf + xc, bias_dus=2 * xc)
    fused = 2 * xc + 2 * c * f * itemsize
    return dict(stock_bytes=sum(stock.values()), stock_parts=stock, fused_bytes=fused, flops=4 * m_rows * c * f, hidden_bytes=hf)
