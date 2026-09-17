"""transition_pallas.py — fused pair Transition (LayerNorm -> SwiGLU -> W_out) as ONE Pallas (Triton lowering) GPU kernel, inference only.
Module-free pure function `fused_transition` + a Haiku installer that rebinds the model's modules.TransitionBlock class-wide.

Stock math (the model's modules.py::TransitionBlock, bf16 activations; the standard pairformer transition block):
    xn  = LayerNorm(x)                          [.., C]    f32 statistics, cast back to bf16
    c   = swish(xn @ Wa) * (xn @ Wb)            [.., F=4C] tokamax gated_linear_unit (Pallas-Triton), bf16 out  <- the [.., 4C] intermediate is
    out = c @ W2                                [.., C]    XLA GEMM                                                       written to and read from HBM
This cell: per row tile (T rows of the flattened [M, C] input): LN in registers -> loop over F in tiles of tf: a = xn@Wa[:,f], b = xn@Wb[:,f]
(f32 acc), c_f = (swish(a)*b) -> bf16 (the stock rounding point), acc += c_f @ W2[f,:] (f32) -> ONE bf16 store. The 4C intermediate never
reaches HBM; weights stream from L2. Numerics: same rounding points as stock except the F-reduction of the last GEMM is accumulated in f32
tile by tile (stock: one cuBLAS/XLA GEMM) -> not bit-exact, tier-2 (rel-rms vs an f32 reference <= stock's; measured on the tile sweep).
Served: x.dtype == bf16, C % 16 == 0, 16 <= C <= 256, F % tf == 0, M % T == 0 (M = prod(leading dims)); anything else -> the caller keeps
the stock path BY NAME (the installer counts fused|fallback per traced call site, like the kit's SERVED census).
"""
import functools, collections, logging
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

_CP = getattr(plgpu, "CompilerParams", None) or getattr(plgpu, "TritonCompilerParams")
F32, BF16 = jnp.float32, jnp.bfloat16
LN_EPS = 1e-5
log = logging.getLogger("cov_ttr")
DEFAULT_CFG = dict(t=64, tf=64, num_warps=4, num_stages=2)                  # H100 sweep ([N,N,128] N=256..1536 + [512,512,64]): 1.1-1.8x the stock TransitionBlock
STATE = {"modules": None, "stock": None, "sites": collections.Counter(), "cfg": dict(DEFAULT_CFG), "installed": False}


def _ln(x, scale, offset, eps):
    mean = jnp.mean(x, axis=1, keepdims=True)
    var = jnp.mean(x * x, axis=1, keepdims=True) - mean * mean          # haiku use_fast_variance=True
    return (x - mean) * (jax.lax.rsqrt(var + eps) * scale[None, :]) + offset[None, :]


def _dot(a, b):
    return jnp.dot(a, b, preferred_element_type=F32)


def _ttr_kernel(x_ref, s_ref, o_ref, wa_ref, wb_ref, w2_ref, *rest, nf, tf, eps, has_bias):
    if has_bias:
        b2_ref, out_ref = rest
    else:
        (out_ref,) = rest
    x = x_ref[...].astype(F32)                                          # [T, C]
    xn = _ln(x, s_ref[...], o_ref[...], eps).astype(BF16)               # stock casts the LN output back to bf16
    T, C = x.shape

    def body(f, acc):
        fs = pl.ds(f * tf, tf)
        a = _dot(xn, wa_ref[:, fs])                                     # [T, tf] f32
        b = _dot(xn, wb_ref[:, fs])
        c = (a * jax.nn.sigmoid(a) * b).astype(BF16)                    # swish(a) * b, rounded to bf16 = the stock GLU kernel's output dtype
        return acc + _dot(c, w2_ref[fs, :])                             # [T, C] f32

    acc = jax.lax.fori_loop(0, nf, body, jnp.zeros((T, C), F32))
    if has_bias:
        acc = acc + b2_ref[...][None, :]
    out_ref[...] = acc.astype(out_ref.dtype)


def supported(x, C=None, F=None, cfg=None):
    """(ok, reason) — never raises. reason words: dtype:<t>, c:<C>, f:<F>, rows:<M>%<T>."""
    cfg = {**STATE["cfg"], **(cfg or {})}
    C = x.shape[-1] if C is None else C
    if x.dtype != BF16: return False, f"dtype:{x.dtype}"
    if C % 16 or not (16 <= C <= 256): return False, f"c:{C}"
    if F is not None and F % cfg["tf"]: return False, f"f:{F}"
    M = 1
    for d in x.shape[:-1]: M *= int(d)
    if M % cfg["t"]: return False, f"rows:{M}%{cfg['t']}"
    return True, "ok"


def fused_transition(x, ln_scale, ln_offset, w1, w2, b2=None, *, eps=LN_EPS, cfg=None):
    """x [..., C] bf16; ln_scale/ln_offset [C] (None = ones/zeros); w1 [C, 2F] (columns [:F] = a (swish), [F:] = b) or (C, 2, F); w2 [F, C];
    b2 [C] or None. Returns [..., C] in x.dtype = (swish(LN(x)@Wa) * (LN(x)@Wb)) @ W2 (+ b2)."""
    cfg = {**STATE["cfg"], **(cfg or {})}
    C = x.shape[-1]
    if w1.ndim == 3:                       # stock storage (C, 2, F)
        wa, wb = w1[:, 0, :], w1[:, 1, :]
    else:
        F2 = w1.shape[1]; wa, wb = w1[:, : F2 // 2], w1[:, F2 // 2:]
    F = wa.shape[1]
    ok, why = supported(x, C, F, cfg)
    if not ok:
        raise ValueError(f"fused_transition: shape not served ({why}); the caller routes to stock by name")
    t, tf = cfg["t"], cfg["tf"]
    lead = x.shape[:-1]; M = 1
    for d in lead: M *= int(d)
    x2 = x.reshape(M, C)
    dt = x.dtype
    s = (jnp.ones((C,), F32) if ln_scale is None else ln_scale.astype(F32)); o = (jnp.zeros((C,), F32) if ln_offset is None else ln_offset.astype(F32))
    args = [x2, s, o, wa.astype(dt), wb.astype(dt), w2.astype(dt)]
    in_specs = [pl.BlockSpec((t, C), lambda i: (i, 0)), pl.BlockSpec((C,), lambda i: (0,)), pl.BlockSpec((C,), lambda i: (0,)),
                pl.BlockSpec((C, F), lambda i: (0, 0)), pl.BlockSpec((C, F), lambda i: (0, 0)), pl.BlockSpec((F, C), lambda i: (0, 0))]
    if b2 is not None:
        args.append(b2.astype(F32)); in_specs.append(pl.BlockSpec((C,), lambda i: (0,)))
    out = pl.pallas_call(
        functools.partial(_ttr_kernel, nf=F // tf, tf=tf, eps=eps, has_bias=b2 is not None), grid=(M // t,),
        in_specs=in_specs, out_specs=pl.BlockSpec((t, C), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((M, C), dt),
        compiler_params=_CP(num_warps=cfg["num_warps"], num_stages=cfg["num_stages"]),
        name="fused_transition",
    )(*args)
    return out.reshape(*lead, C)


def transition_reference(x, ln_scale, ln_offset, w1, w2, b2=None, *, eps=LN_EPS, compute_dtype=F32, precision=None):
    """Op-by-op replica of the stock math. compute_dtype=bf16 mimics stock rounding points; f32 + precision=HIGHEST is the reference.
    Weights are first rounded to x.dtype (that IS the model), then upcast."""
    cd = compute_dtype; dt = x.dtype
    hi = dict(precision=precision, preferred_element_type=cd if cd != BF16 else None)
    x32 = x.astype(jnp.promote_types(cd, F32))
    mean = jnp.mean(x32, -1, keepdims=True); var = jnp.mean(jnp.square(x32), -1, keepdims=True) - jnp.square(mean)
    s = (jnp.ones((x.shape[-1],)) if ln_scale is None else ln_scale).astype(x32.dtype); o = (jnp.zeros((x.shape[-1],)) if ln_offset is None else ln_offset).astype(x32.dtype)
    xn = ((x32 - mean) * jax.lax.rsqrt(var + eps) * s + o).astype(cd)
    if w1.ndim == 3: wa, wb = w1[:, 0, :], w1[:, 1, :]
    else: F2 = w1.shape[1]; wa, wb = w1[:, : F2 // 2], w1[:, F2 // 2:]
    wa, wb, w2c = (w.astype(dt).astype(cd) for w in (wa, wb, w2))
    a = jnp.einsum("...c,cf->...f", xn, wa, **hi).astype(cd); b = jnp.einsum("...c,cf->...f", xn, wb, **hi).astype(cd)
    c = (jax.nn.swish(a) * b).astype(cd)
    out = jnp.einsum("...f,fc->...c", c, w2c, **hi).astype(cd)
    if b2 is not None: out = out + b2.astype(cd)
    return out


# ----------------------------------------------------------------------------------------------------------- Haiku installer (the model's modules)
def make_class(stock_tb):
    import haiku as hk
    from alphafold3.model.components import haiku_modules as hm

    class FusedTransitionBlock(stock_tb):
        """TransitionBlock with the fused LN+SwiGLU+W_out Pallas kernel; same parameters as stock (read in the stock name scopes)."""

        def __call__(self, act, broadcast_dim=0):
            c = act.shape[-1]; f = int(c * self.config.num_intermediate_factor)
            ok, why = supported(act, c, f)
            if not (ok and self.config.use_glu_kernel):
                STATE["sites"][("transition", "fallback:" + (why if not ok else "no_glu_kernel"), tuple(act.shape), str(act.dtype))] += 1
                return super().__call__(act, broadcast_dim=broadcast_dim)
            STATE["sites"][("transition", "fused", tuple(act.shape), str(act.dtype))] += 1
            with hk.name_scope("input_layer_norm"):
                s = hk.get_parameter("scale", (c,), jnp.float32, init=jnp.ones); o = hk.get_parameter("offset", (c,), jnp.float32, init=jnp.zeros)
            act_n_dummy = act                                                  # haiku_linear_get_params reads dtype/shape from its input
            w1, _ = hm.haiku_linear_get_params(act_n_dummy, num_output=2 * f, initializer="relu", name="transition1")
            w1 = jnp.reshape(w1, (c, 2, f))
            with hk.name_scope("transition2"):
                w2 = hk.get_parameter("weights", (f, c), act.dtype, init=jnp.zeros)
            return fused_transition(act, s, o, w1, w2, None, cfg=STATE["cfg"]).astype(act.dtype)

    return FusedTransitionBlock


def install(modules=None, cfg=None):
    if modules is None:
        import alphafold3.model.network.modules as modules
    if STATE["modules"] is None:
        STATE.update(modules=modules, stock=modules.TransitionBlock)
    if cfg: STATE["cfg"].update(cfg)
    modules.TransitionBlock = make_class(STATE["stock"])
    STATE["installed"] = True; STATE["sites"].clear()
    log.warning("cov_ttr: TransitionBlock=%s cfg=%s jax %s", modules.TransitionBlock.__name__, STATE["cfg"], jax.__version__)
    return True


def uninstall():
    if STATE["modules"] is not None:
        STATE["modules"].TransitionBlock = STATE["stock"]
    STATE["installed"] = False


def report():
    sites = {" ".join(str(x) for x in k): v for k, v in STATE["sites"].items()}
    return {"installed": STATE["installed"], "cfg": STATE["cfg"], "sites": sites,
            "fused_sites": sum(v for k, v in STATE["sites"].items() if k[1] == "fused"), "fallback_sites": sum(v for k, v in STATE["sites"].items() if k[1] != "fused")}
