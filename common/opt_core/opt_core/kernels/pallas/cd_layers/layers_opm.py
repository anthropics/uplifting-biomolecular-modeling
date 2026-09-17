"""Lever `opm_fold` — AlphaFold's OuterProductMean (modules.OuterProductMean, Suppl. Alg. 10) with its two contractions RE-ASSOCIATED so the
[N_res, c, c, chunk] outer-product intermediate is never formed.

Stock (colabdesign/af/alphafold/model/modules.py:1100-1184), per Evoformer block (48) and extra-MSA block (4), forward:

    left  = mask * Linear_c(LN(act))            [S, N, c]      S = MSA depth (2 in BindCraft's binder design: the sequence row + one template row; 1 in the extra-MSA stack), c = 32
    right = mask * Linear_c(LN(act))            [S, N, c]
    for each 128-row chunk of `left` (hk.scan, chunk_size=128):
        P[d,c,e,b] = sum_a left[a,b,c] right[a,d,e]          bf16 [N, c, c, 128]   <- 2*S*N*128*c*c flop, an N*c*c*128 bf16 intermediate written + read
        out[b,d,f] = sum_{c,e} P[d,c,e,b] W[c,e,f] + bias   bf16 GEMM, K = c*c = 1024: 2*N*128*1024*F flop per chunk  (F = pair channels, 128)
    out /= (1e-3 + sum_a mask[a,b] mask[a,d])

    = 2*N^2*c^2*F flop per call in the second GEMM alone (94 GFLOP at N=600; x(48+4) blocks x ~3 passes per design step), plus the transposes
    XLA inserts around the chunked scan ('acb' / 'dceb' layouts) and the scan's dynamic-update-slices.

This lever (same parameters, same names — `layer_norm_input`, `left_projection`, `right_projection`, `output_w`, `output_b`):

    T[a,d,c,f] = sum_e right[a,d,e] W[c,e,f]                 [S, N, c, F]   2*S*N*c*c*F flop (0.3 GFLOP at N=600) — W folded into the RIGHT operand
    out[b,d,f] = sum_{a,c} left[a,b,c] T[a,d,c,f] + bias      ONE GEMM  [N x S*c] . [S*c x N*F]: K = S*c (64), 2*N^2*S*c*F flop (5.9 GFLOP at N=600),
                                                              whose row-major result IS [b,d,f] — no output transpose, no chunk loop, no scan
    out /= (1e-3 + norm)

Identical in exact arithmetic (sum re-association: sum_a sum_c sum_e l*r*W = sum_a sum_c l * (sum_e r*W)); the rounding points move (stock rounds
the S-term outer products P to bf16 and accumulates K=1024 in fp32 inside cuBLAS; here T is formed and KEPT in f32 and the K=S*c GEMM runs f32 x f32
at XLA's DEFAULT precision — TF32-class products, above stock's bf16 — so the lever's error against an fp32 reference is AT PARITY with stock's own:
out 1.00-1.02x, d_act 0.99-1.04x of stock's error over S in {1,2,4,5}; HIGHEST instead of DEFAULT on these two products changes nothing, the
residual is the alignment of the bf16 roundings both paths share, not lost precision; T in bf16 instead measured 1.10x) —
NOT bitwise: numerics class `precision` (re-associated accumulation), tier `fast`; `precision=tf32` on the LEVER line (the ONE product class this
lever introduces; nothing here changes XLA's global f32 dot precision). The backward pass is XLA's own autodiff of these two einsums (d_left:
[N x N*F].[N*F x S*c]; d_T: [S*c x N].[N x N*F]; d_right through W) — no custom VJP. By-name gate: S > 32 (= c) would make the fold cost more
flop than stock's association — such a call runs the stock class, census `fallback_by=gated_s_gt_32` (unreachable at the pinned settings).

Memory: the largest live intermediate is T (S*N*c*F f32, 20 MB at N=600) instead of stock's per-chunk P (N*c*c*128 bf16, 157 MB at N=600) — the
chunking stock needs (`chunk_size=128`) has nothing left to bound, so the whole call is one pass at every size.

Install: `install()` rebinds `modules.OuterProductMean` to a same-name subclass (marker attribute MARKER; haiku's metaclass wraps its __call__ so the
parameter scope is the module's own) before the design model is traced; idempotent; every traced call is counted by (S, N, c, F) for the LEVER
line, printed once at exit:

    [colabdesign-opt] LEVER name=opm_fold state=on impl=opm_fold@kit origin=kit precision=tf32 numerics=precision served=<traced calls> fallback=<n> fallback_by=<gated_s_gt_32:n|none> shapes=S2xN600xc32xF128:<n>,…
(`lever_line()` renders it; the kit's exit multiplexer (kernels/__init__.py, KIT track) prints it once at process exit, after the calls were traced.)
"""
from __future__ import annotations

import collections

try:                                                    # the stack (jax, haiku); on a host without it the module stays importable — the registry protocol, the dry run —
    import jax                                          # and install() refuses by name
    import jax.numpy as jnp
    import haiku as hk
except Exception:                                       # pragma: no cover - exercised on the stand-in stack (ImportError; AttributeError when a real haiku meets a stand-in jax)
    jax = jnp = hk = None

NAME = "opm_fold"
IMPL = "opm_fold@kit"
MARKER = "_colabdesign_opt_layers_opm_fold"
NUMERICS = "precision"             # re-associated accumulation (exact in real arithmetic; bf16 rounding points move) — never bitwise
PRECISION = "tf32"                      # T is formed and kept in f32 and the one GEMM runs f32 x f32 at XLA's DEFAULT precision (TF32 class on sm_80+): ABOVE stock's bf16 products, so the lever's error vs an fp32 reference is <= stock's (the kit's class rule for re-associations); the price is an f32 [N,N,F] GEMM result (+64 MB written @500) that XLA's epilogue fusion (bias, /norm, cast, residual) reads once
MAX_S = 32                              # by-name gate: the fold costs 2*N^2*S*c*F flop vs stock's 2*N^2*c^2*F — a pessimisation above S = c = 32 (unreachable at the pinned settings: S = 2 / 1); such a call runs the stock class, counted `gated_s_gt_32`

_CENSUS: "collections.Counter[str]" = collections.Counter()
_GATED: "collections.Counter[str]" = collections.Counter()
EXPECTED_FALLBACKS = ("gated_s_gt_32",)       # the ONLY fallback word this lever may print (KIT's evidence classification stays fail-closed on any other)
_STATE = {"installed": False, "stock_cls": None}


def outer_product_mean(act, mask, left_lin, right_lin, output_w, output_b):
    """The re-associated OuterProductMean on arrays. act [S,N,C_m] (ALREADY layer-normed); mask [S,N]; left_lin/right_lin: callables (the two
    Linear projections); output_w [c,c,F]; output_b [F]. Returns [N,N,F] in the projections' dtype. ONE dtype policy (no knob): the projections run
    in the activations' dtype exactly as stock (bf16 under the Evoformer); T and the K=S*c GEMM are f32 (TF32-class products), the result is cast back
    to the activations' dtype after +bias — before the division by (1e-3 + norm), as stock."""
    mask = mask[..., None].astype(act.dtype)
    left_act = mask * left_lin(act)                                   # [S, N, c]   mask on BOTH operands, as stock
    right_act = mask * right_lin(act)                                 # [S, N, c]
    f32 = jnp.float32
    # T[a,d,c,f] = sum_e right[a,d,e] * W[c,e,f]   (W folded into the right operand; S*N*c*F f32, 20 MB @600)
    t = jnp.einsum("ade,cef->adcf", right_act.astype(f32), output_w.astype(f32), preferred_element_type=f32)
    # out[b,d,f] = sum_{a,c} left[a,b,c] * T[a,d,c,f]   (one f32 GEMM, K = S*c; its row-major result IS [b,d,f])
    out = jnp.einsum("abc,adcf->bdf", left_act.astype(f32), t, preferred_element_type=f32)
    out = (out + output_b.astype(f32)).astype(left_act.dtype)         # + bias BEFORE the normalisation, then stock's dtype
    epsilon = 1e-3
    norm = jnp.einsum("abc,adc->bdc", mask, mask)
    return out / (epsilon + norm)


def _opm_call(self, act, mask, stock_call):
    """modules.OuterProductMean.__call__ (colabdesign_opt lever opm_fold): same parameters and names as stock, contractions re-associated."""
    from colabdesign.af.alphafold.model import common_modules
    gc, c = self.global_config, self.config
    S, N = int(act.shape[0]), int(act.shape[1])
    if S > MAX_S:                                                      # the by-name gate: stock's association is cheaper above S = c
        _GATED[f"gated_s_gt_{MAX_S}"] += 1
        return stock_call(self, act, mask)
    act = common_modules.LayerNorm([-1], True, True, name="layer_norm_input")(act)
    left_lin = common_modules.Linear(c.num_outer_channel, initializer="linear", name="left_projection")
    right_lin = common_modules.Linear(c.num_outer_channel, initializer="linear", name="right_projection")
    init_w = hk.initializers.Constant(0.0) if gc.zero_init else hk.initializers.VarianceScaling(scale=2.0, mode="fan_in")
    output_w = hk.get_parameter("output_w", shape=(c.num_outer_channel, c.num_outer_channel, self.num_output_channel), dtype=act.dtype, init=init_w)
    output_b = hk.get_parameter("output_b", shape=(self.num_output_channel,), dtype=act.dtype, init=hk.initializers.Constant(0.0))
    _CENSUS[f"S{S}xN{N}xc{int(c.num_outer_channel)}xF{int(self.num_output_channel)}"] += 1
    return outer_product_mean(act, mask, left_lin, right_lin, output_w, output_b)


def install() -> dict:
    """Rebind `colabdesign.af.alphafold.model.modules.OuterProductMean` to a same-name subclass whose __call__ is the re-associated form
    (idempotent). haiku detail (as opt_core's pallas_attn_serve): the subclass is created in a class body through haiku's metaclass with the SAME
    class name, so `hk.get_parameter` runs in the module's own scope and the parameter tree is unchanged. Call before the model is traced."""
    _require()
    from colabdesign.af.alphafold.model import modules
    A = modules.OuterProductMean
    if not getattr(A, MARKER, False):
        stock_call = A.__call__

        class OuterProductMean(A):                      # noqa: D101
            _stock_cls = A

            def __call__(self, act, mask):
                return _opm_call(self, act, mask, stock_call)
        setattr(OuterProductMean, MARKER, True)
        OuterProductMean.__qualname__ = "OuterProductMean"
        OuterProductMean.__module__ = A.__module__
        _STATE["stock_cls"] = A
        modules.OuterProductMean = OuterProductMean
    _STATE["installed"] = True
    _register_exit_line()
    return {"lever": NAME, "impl": IMPL, "numerics": NUMERICS, "precision": PRECISION}


def uninstall() -> None:
    import sys
    modules = sys.modules.get("colabdesign.af.alphafold.model.modules")          # nothing to undo in a process that never imported the model
    A = getattr(modules, "OuterProductMean", None) if modules else None
    if A is not None and getattr(A, MARKER, False):
        modules.OuterProductMean = A._stock_cls
    _STATE["installed"] = False


def stock_class():
    """The stock OuterProductMean class (whether or not the lever is installed)."""
    from colabdesign.af.alphafold.model import modules
    A = modules.OuterProductMean
    return A._stock_cls if getattr(A, MARKER, False) else A


def installed() -> bool:
    import sys
    mods = sys.modules.get("colabdesign.af.alphafold.model.modules")
    return bool(mods and getattr(getattr(mods, "OuterProductMean", None), MARKER, False))


def census() -> dict:
    return {"served": int(sum(_CENSUS.values())), "shapes": dict(_CENSUS), "fallback": int(sum(_GATED.values())), "fallback_by": dict(_GATED)}


# ─────────────────────────────────────────── kit lever protocol (registry.py; kit 0.4.3) ───────────────────────────────────────────
class Refusal(RuntimeError):
    """install() raises this when the lever cannot run here (levers.install turns it into the mode's refusal by name)."""


REFUSALS = (Refusal,)


def _require() -> None:
    """The lever's floor: nothing beyond jax itself (plain jnp einsums on any backend); kept for the protocol's shape."""
    problems = []
    if jax is None or hk is None:
        raise Refusal(f"lever {NAME}: jax / haiku are not importable here")
    try:
        jax.default_backend()
    except Exception as e:                                            # noqa: BLE001
        problems.append(f"backend: {e!r}")
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
