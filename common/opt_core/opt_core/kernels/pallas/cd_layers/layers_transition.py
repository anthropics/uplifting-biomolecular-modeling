"""Lever `transition` — AlphaFold's Transition (modules.Transition, Suppl. Alg. 9 / 15: LayerNorm -> Linear(C -> 4C) -> relu -> Linear(4C -> C);
pair_transition on the [N, N, 128] pair activation and msa_transition on the [S, N, 256] MSA activation of every Evoformer block, the extra-MSA
stack's two, the template pair stack's one) as ONE Pallas/Triton kernel forward and ONE backward, so the 4C-wide intermediate never touches HBM.

Why (PROFILE M2, ROOT #7): the transition's GEMMs are K=128-class projections at ~64 flop/byte — ON the HBM roofline — and the glue around them
(LN statistics + normalise, +bias/relu on the [M, 4C] intermediate, relu-backward select, the LN backward) runs at >= 90 % of HBM bandwidth already:
nothing here gets faster per byte, the only lever is moving fewer bytes. Stock moves, per [M, C] call (M = N^2 rows for the pair transition), about
x + LN out + h (4C wide, written, read) + out forward, the same again for remat's re-forward, and dy + dh (4C, written, read) + relu mask source + dxn
+ LN-backward passes backward: ~16 M*C-sized bf16 passes + ~6 M*4C-sized ones. This kernel: forward reads x once and writes out once; backward reads
x and dy once and writes dx once (the re-forward under remat is dead code: the VJP's residuals are the kernel's INPUTS, so XLA drops the recompute).

Math, per row tile (bm rows), everything on-chip; the rounding points of stock's bf16 path are KEPT where they are free to keep (marked R):
    fwd:  xn = LN(x) (f32 stats, the instance's eps; -> bf16 R)                      [bm, C]
          for each 4C-chunk j (bh wide):  h = xn @ W1[:, j] (f32 acc; -> bf16 R) + b1[j] (-> bf16 R); h = relu(h)
                                          acc += h @ W2[j, :] (f32 acc)
          out = (acc -> bf16 R) + b2 (-> bf16 R)
    bwd:  xn recomputed; for each chunk j: h as fwd; dh = dy @ W2[j, :]^T (f32 acc; -> bf16 R); dh = where(h > 0, dh, 0)
                                          dxn += dh @ W1[:, j]^T (f32 acc)
          dxn -> bf16 R; dx = LN_bwd(x, dxn) (f32, row-local) -> bf16
          dW1, db1, dW2, db2, dscale, doffset: plain-JAX expressions in the VJP (dead code in the design step, which differentiates the sequence only;
          right when somebody asks).
Numerics: the same products in the same dtypes as stock (bf16 operands, f32 accumulation) with a different accumulation order inside the tiles and
one fewer rounding (stock rounds each K=4C GEMM's result once; here the 4C contraction is accumulated over chunks in f32 and rounded once — at least
as accurate) -> class `precision`, never bitwise; `precision=bf16` on the LEVER line.

Served: act.dtype bf16, C = act.shape[-1] in {64, 128, 256}, intermediate width C*num_intermediate_factor a multiple of 128 (factor 4 everywhere,
factor 2 in the template pair stack: every Transition of the model), LayerNorm over the last axis; anything else runs the stock class BY NAME
(census `fallback_by=<reason>`; EXPECTED_FALLBACKS = (): none expected at the pinned settings). The in-kernel LayerNorm forms its two row sums as
sums of channel pairs — one of two exact-math-equivalent f32 associations, fixed (layers.md 'transition localisation': summation order alone
moves chaotic step-check units, so the choice is declared, not tunable). Stock's `subbatch_size` chunking of
this module (mapping.inference_subbatch) bounds nothing here — the kernel's working set is a row tile — so a served call is one launch at every size.

Install: `install()` rebinds `modules.Transition` to a same-name subclass (haiku's metaclass wraps its __call__: parameter names and scopes are
stock's — `input_layer_norm/{scale,offset}`, `transition1/{weights,bias}`, `transition2/{weights,bias}`); idempotent; LEVER line via lever_line():
    [colabdesign-opt] LEVER name=transition state=on impl=transition_pallas@kit origin=kit numerics=precision precision=bf16 served=<n> fallback=<n> fallback_by=<reason:n|none> shapes=<C<c>:n,..>
"""
from __future__ import annotations

import collections
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

NAME = "transition"
IMPL = "transition_pallas@kit"
MARKER = "_colabdesign_opt_layers_transition"
NUMERICS = "precision" 
PRECISION = "bf16"
EXPECTED_FALLBACKS = ()
SERVED_C = (64, 128, 256)
HIDDEN_CHUNK = 128                       # the 4C (or 2C) intermediate is processed in on-chip chunks of this width; H must be a multiple of it (every Transition of the model is)

_CP = getattr(plgpu, "TritonCompilerParams", None) or getattr(plgpu, "CompilerParams")
_PL_LOAD, _PL_STORE = getattr(pl, "load", None), getattr(pl, "store", None)

_CENSUS = {"served": collections.Counter(), "fallback": collections.Counter()}
_STATE = {"installed": False, "stock_cls": None}


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


def _tiles(C: int, H: int):
    """(bm rows per program, bh hidden columns per chunk, num_warps)."""
    if C <= 64:
        return 128, HIDDEN_CHUNK, 4
    if C <= 128:
        return 64, HIDDEN_CHUNK, 4
    return 64, HIDDEN_CHUNK, 8


def _bf16_round(v):
    return v.astype(jnp.bfloat16).astype(jnp.float32)


# ─────────────────────────────────────────── kernels ───────────────────────────────────────────
def _ln_rows(x, sc, of, eps, C):
    """Row LayerNorm in f32 (haiku's formula: two-pass mean / centred variance, eps inside the rsqrt). The two row sums are formed as sums of
    channel PAIRS then a flat sum — one of two exact-math-equivalent associations, chosen once and fixed (no knob); layers.md 'transition
    localisation' records why the association is worth a sentence (f32 summation order alone moves chaotic step-check units)."""
    xp = x.reshape(x.shape[0], C // 2, 2)
    mu = jnp.sum(jnp.sum(xp, axis=2), axis=1, keepdims=True) * (1.0 / C)
    xc = x - mu
    cp = (xc * xc).reshape(x.shape[0], C // 2, 2)
    var = jnp.sum(jnp.sum(cp, axis=2), axis=1, keepdims=True) * (1.0 / C)
    r = jax.lax.rsqrt(var + eps)
    return (sc * r) * xc + of, xc, r


def _fwd_kernel(x_ref, sc_ref, of_ref, w1_ref, b1_ref, w2_ref, b2_ref, o_ref, *, eps, bm, bh, C, H, M):
    i = pl.program_id(0)
    rows = i * bm + jnp.arange(bm)
    rmask = (rows < M)[:, None]
    ridx = (pl.dslice(i * bm, bm), pl.dslice(0, C))
    x = _load(x_ref, ridx, mask=rmask, other=0.0).astype(jnp.float32)
    sc = _load(sc_ref, (pl.dslice(0, C),))[None, :]
    of = _load(of_ref, (pl.dslice(0, C),))[None, :]
    xn, _, _ = _ln_rows(x, sc, of, eps, C)
    xn16 = xn.astype(jnp.bfloat16)                                              # R: stock's LN output is bf16
    acc = jnp.zeros((bm, C), jnp.float32)
    for j in range(H // bh):
        w1 = _load(w1_ref, (pl.dslice(0, C), pl.dslice(j * bh, bh)))             # bf16 [C, bh]
        b1 = _load(b1_ref, (pl.dslice(j * bh, bh),)).astype(jnp.float32)[None, :]
        h = pl.dot(xn16, w1).astype(jnp.float32)                                # f32 accumulation
        h = _bf16_round(_bf16_round(h) + b1)                                    # R, R: stock rounds the GEMM result, then the bf16 bias add
        h = jnp.maximum(h, 0.0)
        w2 = _load(w2_ref, (pl.dslice(j * bh, bh), pl.dslice(0, C)))             # bf16 [bh, C]
        acc += pl.dot(h.astype(jnp.bfloat16), w2).astype(jnp.float32)
    b2 = _load(b2_ref, (pl.dslice(0, C),)).astype(jnp.float32)[None, :]
    out = _bf16_round(acc) + b2                                                 # R (+ the store's rounding = stock's bf16 bias add)
    _store(o_ref, ridx, out.astype(o_ref.dtype), mask=rmask)


def _bwd_kernel(x_ref, dy_ref, sc_ref, of_ref, w1_ref, b1_ref, w2_ref, dx_ref, *, eps, bm, bh, C, H, M):
    i = pl.program_id(0)
    rows = i * bm + jnp.arange(bm)
    rmask = (rows < M)[:, None]
    ridx = (pl.dslice(i * bm, bm), pl.dslice(0, C))
    x = _load(x_ref, ridx, mask=rmask, other=0.0).astype(jnp.float32)
    dy16 = _load(dy_ref, ridx, mask=rmask, other=0.0).astype(jnp.bfloat16)
    sc = _load(sc_ref, (pl.dslice(0, C),))[None, :]
    of = _load(of_ref, (pl.dslice(0, C),))[None, :]
    xn, xc, r = _ln_rows(x, sc, of, eps, C)
    xn16 = xn.astype(jnp.bfloat16)
    dxn = jnp.zeros((bm, C), jnp.float32)
    for j in range(H // bh):
        w1 = _load(w1_ref, (pl.dslice(0, C), pl.dslice(j * bh, bh)))             # bf16 [C, bh]
        b1 = _load(b1_ref, (pl.dslice(j * bh, bh),)).astype(jnp.float32)[None, :]
        h = pl.dot(xn16, w1).astype(jnp.float32)
        h = _bf16_round(_bf16_round(h) + b1)
        w2 = _load(w2_ref, (pl.dslice(j * bh, bh), pl.dslice(0, C)))             # bf16 [bh, C]
        dh = pl.dot(dy16, w2, trans_b=True).astype(jnp.float32)                 # [bm, bh] = dy @ W2[j]^T
        dh = jnp.where(h > 0.0, _bf16_round(dh), 0.0)                           # R: stock's dh is a bf16 GEMM result; relu backward on stock's (rounded) h
        dxn += pl.dot(dh.astype(jnp.bfloat16), w1, trans_b=True).astype(jnp.float32)   # [bm, C] += dh @ W1[:, j]^T
    g = _bf16_round(dxn) * sc                                                   # R: stock's d(LN out) is bf16; LN backward in f32
    xhat = xc * r
    mg = jnp.sum(g, axis=1, keepdims=True) * (1.0 / C)
    mgx = jnp.sum(g * xhat, axis=1, keepdims=True) * (1.0 / C)
    dx = r * (g - mg - xhat * mgx)
    _store(dx_ref, ridx, dx.astype(dx_ref.dtype), mask=rmask)


def _call_fwd(x2, scale, offset, w1, b1, w2, b2, eps):
    M, C = x2.shape
    H = w1.shape[1]
    bm, bh, nw = _tiles(C, H)
    kern = functools.partial(_fwd_kernel, eps=eps, bm=bm, bh=bh, C=C, H=H, M=M)
    return pl.pallas_call(kern, grid=(pl.cdiv(M, bm),), out_shape=jax.ShapeDtypeStruct((M, C), x2.dtype),
                          compiler_params=_CP(num_warps=nw, num_stages=2), name=f"transition_fwd_c{C}")(x2, scale, offset, w1, b1, w2, b2)


def _call_bwd(x2, dy, scale, offset, w1, b1, w2, eps):
    M, C = x2.shape
    H = w1.shape[1]
    bm, bh, nw = _tiles(C, H)
    kern = functools.partial(_bwd_kernel, eps=eps, bm=bm, bh=bh, C=C, H=H, M=M)
    return pl.pallas_call(kern, grid=(pl.cdiv(M, bm),), out_shape=jax.ShapeDtypeStruct((M, C), x2.dtype),
                          compiler_params=_CP(num_warps=nw, num_stages=2), name=f"transition_bwd_c{C}")(x2, dy, scale, offset, w1, b1, w2)


@functools.partial(jax.custom_vjp, nondiff_argnums=(7,))
def transition_2d(x2, scale, offset, w1, b1, w2, b2, eps):
    """out = Linear2(relu(Linear1(LN(x2)))) row-wise on a 2-D [M, C] bf16 array; scale/offset f32 [C]; w1 [C,4C], b1 [4C], w2 [4C,C], b2 [C] bf16."""
    return _call_fwd(x2, scale, offset, w1, b1, w2, b2, eps)


def _t_fwd(x2, scale, offset, w1, b1, w2, b2, eps):
    return _call_fwd(x2, scale, offset, w1, b1, w2, b2, eps), (x2, scale, offset, w1, b1, w2, b2)


def _t_bwd(eps, res, dy):
    x2, scale, offset, w1, b1, w2, b2 = res
    dx = _call_bwd(x2, dy, scale, offset, w1, b1, w2, eps)
    # parameter cotangents: plain JAX (stock's formula and dtypes); dead code in the design step (sequence-only gradient), eliminated by XLA there
    xn = reference_layer_norm(x2, scale, offset, eps)
    xf = x2.astype(jnp.float32); mu = jnp.mean(xf, axis=1, keepdims=True); xhat = (xf - mu) * jax.lax.rsqrt(jnp.var(xf, axis=1, keepdims=True) + eps)
    h = jnp.dot(xn, w1, preferred_element_type=jnp.float32).astype(x2.dtype) + b1
    hr = jax.nn.relu(h)
    dyf = dy
    dW2 = jnp.dot(hr.T, dyf, preferred_element_type=jnp.float32).astype(w2.dtype)
    db2 = jnp.sum(dyf.astype(jnp.float32), axis=0).astype(b2.dtype)
    dh = jnp.dot(dyf, w2.T, preferred_element_type=jnp.float32).astype(x2.dtype)
    dh = jnp.where(h > 0, dh, jnp.zeros_like(dh))
    dW1 = jnp.dot(xn.T, dh, preferred_element_type=jnp.float32).astype(w1.dtype)
    db1 = jnp.sum(dh.astype(jnp.float32), axis=0).astype(b1.dtype)
    dxn = jnp.dot(dh, w1.T, preferred_element_type=jnp.float32).astype(x2.dtype).astype(jnp.float32)
    dscale = jnp.sum(dxn * xhat, axis=0)
    doffset = jnp.sum(dxn, axis=0)
    return dx, dscale, doffset, dW1, db1, dW2, db2


transition_2d.defvjp(_t_fwd, _t_bwd)


def transition(x, scale, offset, w1, b1, w2, b2, eps=1e-5):
    """The fused Transition for x of any rank >= 2 (rows = all leading axes); parameters as modules.Transition holds them (see transition_2d)."""
    C = x.shape[-1]
    return transition_2d(x.reshape(-1, C), scale, offset, w1, b1, w2, b2, eps).reshape(x.shape)


def reference_layer_norm(x, scale, offset, eps=1e-5):
    xf = x.astype(jnp.float32)
    mu = jnp.mean(xf, axis=-1, keepdims=True)
    var = jnp.var(xf, axis=-1, keepdims=True)
    return (scale * jax.lax.rsqrt(var + eps) * (xf - mu) + offset).astype(x.dtype)


def reference_transition(x, scale, offset, w1, b1, w2, b2, eps=1e-5):
    """Stock's Transition on arrays in stock's dtypes (bf16 operands, XLA's f32 accumulation): the unit tests' A side without haiku."""
    xn = reference_layer_norm(x, scale, offset, eps)
    h = jax.nn.relu(jnp.dot(xn, w1, preferred_element_type=jnp.float32).astype(x.dtype) + b1)
    return jnp.dot(h, w2, preferred_element_type=jnp.float32).astype(x.dtype) + b2


# ─────────────────────────────────────────── serving rule ───────────────────────────────────────────
def _eligibility(act, C, H):
    if jax.default_backend() != "gpu":
        return "platform"
    if act.dtype != jnp.bfloat16:
        return "dtype_not_bf16"
    if C not in SERVED_C:
        return "channels"
    if H < HIDDEN_CHUNK or H % HIDDEN_CHUNK:
        return "intermediate_width"
    if act.ndim < 2:
        return "rank"
    return None


def _name_scope(name):
    import haiku as hk
    ns = getattr(hk, "name_scope", None) or hk.experimental.name_scope
    return ns(name)


def _transition_call(self, act, mask, stock_call):
    """modules.Transition.__call__ (colabdesign_opt lever transition): stock's parameters and names, one fused kernel per direction."""
    import haiku as hk
    c, gc = self.config, self.global_config
    C = int(act.shape[-1])
    H = int(C * c.num_intermediate_factor)
    why = _eligibility(act, C, H)
    if why is not None:
        _CENSUS["fallback"][why] += 1
        return stock_call(self, act, mask)
    # parameters under stock's names and scopes (<this module>/input_layer_norm/{scale,offset} f32; transition1/{weights [C,H], bias [H]};
    # transition2/{weights [H,C], bias [C]} in act.dtype through the bf16 custom getter, exactly as common_modules.LayerNorm / Linear request them)
    with _name_scope("input_layer_norm"):
        scale = hk.get_parameter("scale", (C,), jnp.float32, init=jnp.ones)
        offset = hk.get_parameter("offset", (C,), jnp.float32, init=jnp.zeros)
    w_init1 = hk.initializers.TruncatedNormal(stddev=float((2.0 / C) ** 0.5))                    # stock initializer='relu' (values matter at init only)
    w_init2 = hk.initializers.Constant(0.0) if gc.zero_init else hk.initializers.TruncatedNormal(stddev=float((1.0 / H) ** 0.5))   # 'final_init'
    with _name_scope("transition1"):
        w1 = hk.get_parameter("weights", (C, H), act.dtype, init=w_init1)
        b1 = hk.get_parameter("bias", (H,), act.dtype, init=hk.initializers.Constant(0.0))
    with _name_scope("transition2"):
        w2 = hk.get_parameter("weights", (H, C), act.dtype, init=w_init2)
        b2 = hk.get_parameter("bias", (C,), act.dtype, init=hk.initializers.Constant(0.0))
    _CENSUS["served"][f"C{C}"] += 1
    return transition(act, scale.astype(jnp.float32), offset.astype(jnp.float32), w1.astype(act.dtype), b1.astype(act.dtype),
                      w2.astype(act.dtype), b2.astype(act.dtype), 1e-5)


def install() -> dict:
    _require()
    from colabdesign.af.alphafold.model import modules
    A = modules.Transition
    if not getattr(A, MARKER, False):
        stock_call = A.__call__

        class Transition(A):                    # noqa: D101
            _stock_cls = A

            def __call__(self, act, mask, *args, **kwargs):
                return _transition_call(self, act, mask, lambda s_, a_, m_: stock_call(s_, a_, m_, *args, **kwargs))
        setattr(Transition, MARKER, True)
        Transition.__qualname__ = "Transition"; Transition.__module__ = A.__module__
        _STATE["stock_cls"] = A
        modules.Transition = Transition
    _STATE["installed"] = True
    _register_exit_line()
    return {"lever": NAME, "impl": IMPL, "numerics": NUMERICS, "precision": PRECISION}


def uninstall() -> None:
    from colabdesign.af.alphafold.model import modules
    A = modules.Transition
    if getattr(A, MARKER, False):
        modules.Transition = A._stock_cls
    _STATE["installed"] = False


def installed() -> bool:
    return bool(_STATE["installed"])


def stock_class():
    from colabdesign.af.alphafold.model import modules
    A = modules.Transition
    return A._stock_cls if getattr(A, MARKER, False) else A


def census() -> dict:
    return {"served": int(sum(_CENSUS["served"].values())), "fallback": int(sum(_CENSUS["fallback"].values())),
            "shapes": dict(_CENSUS["served"]), "fallback_by": dict(_CENSUS["fallback"])}


# ─────────────────────────────────────────── kit lever protocol (registry.py; kit 0.4.3) ───────────────────────────────────────────
class Refusal(RuntimeError):
    """install() raises this when the lever cannot run here (levers.install turns it into the mode's refusal by name)."""


REFUSALS = (Refusal,)


def _require() -> None:
    """The lever's floor: a GPU backend and jax's Pallas/Triton lowering (the pinned stack: jax 0.6.0)."""
    problems = []
    try:
        if jax.default_backend() != "gpu":
            problems.append(f"backend={jax.default_backend()} (gpu required)")
    except Exception as e:                                            # noqa: BLE001
        problems.append(f"backend: {e!r}")
    if not hasattr(pl, "pallas_call") or _CP is None:
        problems.append("jax.experimental.pallas triton lowering absent")
    if problems:
        raise Refusal(f"lever {NAME}: " + "; ".join(problems))


def evidence() -> dict:
    c = census()
    return {"lever": NAME, "line_name": NAME, "installed": bool(_STATE["installed"]), "impl": IMPL, "numerics": NUMERICS, "precision": PRECISION, **c}


def off_line(reason: str) -> str:
    from opt_core import report as _r
    from ..names import TAG
    return _r.lever_line(TAG, NAME, "off", reason=reason, impl=IMPL, origin="kit")


def lever_line() -> str:
    """This lever's ONE LEVER line (state=on with its census; the kit's exit printer calls it at process exit)."""
    from opt_core import report as _r
    from ..names import TAG
    if not _STATE["installed"]:
        return off_line("not_installed")
    c = census()
    shapes = ",".join(f"{k}:{v}" for k, v in sorted(c["shapes"].items())) or "none"
    fb = ",".join(f"{k}:{v}" for k, v in sorted(c["fallback_by"].items())) or "none"
    return _r.lever_line(TAG, NAME, "on", ("numerics", NUMERICS), ("precision", PRECISION), ("served", c["served"]), ("fallback", c["fallback"]),
                         ("fallback_by", fb), ("shapes", shapes), impl=IMPL, origin="kit")


def _register_exit_line() -> None:
    """Hand lever_line to the package's one exit printer (kernels/__init__.py) — a no-op until the registry names this lever."""
    try:
        from . import register_exit_line
        register_exit_line(NAME, lever_line)
    except (ImportError, ValueError):
        pass
