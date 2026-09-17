"""Lever `ln` — every `common_modules.LayerNorm` call of the design model (AlphaFold-Multimer's LayerNorm over the channel axis: 8 pair-sized
and 3 MSA-sized calls per Evoformer block, the extra-MSA and template stacks, recycling norms, structure module, heads) served by ONE Pallas/Triton
row kernel forward and ONE backward that live in the SHARED CORE — the provider `opt_core.kernels.pallas` serves them (row `cd_ln` =
`opt_core/kernels/pallas/cd_layers/layers_ln.py`), reached through `kernels/provider.py` by the TIER word `fast`: the provider's measured cell
table names the row per (jax line, card, dtype, family, size, direction) and the LEVER line censuses what it served (`provider= word= row= tier=
[uncovered=]`). This module is the ADAPTER only — the haiku class rebinding, each instance's parameters / eps / variance form, the by-name
serving rule, the census and the LEVER line; it carries no kernel source (activation gradient d_x only — the design step differentiates w.r.t. the sequence, never the weights;
d_scale / d_offset are plain-JAX reductions that XLA dead-code-eliminates when nobody asks for them, and are right when somebody does).

Why: XLA lowers each LayerNorm of a [N, N, C] pair activation as a row-statistics kernel (an `input_transpose_fusion` / `input_reduce_fusion`
computing mean and rsqrt(var+eps) per row, ~190 us at N=500 for a 64 MB read = ~1/7 of an H100's bandwidth, measured) plus the normalisation
folded into a consumer fusion; the backward is two more row reductions of the same class. This kernel reads the rows once, keeps the statistics in
registers and writes the normalised rows once (fwd), and reads x, dy once / writes dx once (bwd): the op at HBM bandwidth.

Math (haiku LayerNorm, f32 statistics; bf16 in -> f32 math -> bf16 out exactly as common_modules.LayerNorm casts; EACH INSTANCE's eps, axis, param_axis,
scale/offset creation and use_fast_variance are read and honoured — the fused multimer TriangleMultiplication's LNs (modules._layer_norm) use
use_fast_variance=True, var = E[x^2]-E[x]^2, everything else the default var = E[(x-mu)^2]; the kernel carries the form as a static flag):
    fwd:  mu = mean(x); var = <form>; inv = scale*rsqrt(var+eps); y = inv*(x-mu) + offset       (haiku's operation order)
    bwd:  xhat = (x-mu)*r; g = dy*scale; dx = r*(g - mean(g) - xhat*mean(g*xhat))          (row-local; mu, r recomputed from x in the bwd kernel)
          d_scale = sum_rows(dy*xhat); d_offset = sum_rows(dy)                                (plain JAX; DCE'd in the design step)
Numerics: same formula, f32 statistics, different reduction order/rounding than XLA's fusions -> NOT bitwise: numerics class `precision` (different
kernel and reduction order, same formula), tier `fast`. No matrix products: `precision=none` on the LEVER line.

Served: x.ndim >= 2, LayerNorm axis == the last axis (every call in the model), C = x.shape[-1] a power of two in [32, 1024], dtype bf16 or f32,
GPU backend. Anything else runs the stock method BY NAME, counted `fallback_by=<reason>` (expected in this model: `channels_not_pow2` — the
384-channel single-representation norms of the structure module / heads, [N, 384]-sized, negligible).

Install: `install()` rebinds `colabdesign.af.alphafold.model.common_modules.LayerNorm` to a same-name subclass (marker MARKER; every call site in the
model looks the class up on the module object at call time) before the model is traced; the LEVER line prints once at exit:
    [colabdesign-opt] LEVER name=ln state=on impl=layers_ln@<sha8> origin=core numerics=precision precision=none served=<n> fallback=<n> fallback_by=<reason:n,..> shapes=<CxDTYPE:n,...>
                      provider=opt_core.kernels.pallas@<core> word=fast row=<row served>:<n> tier=<arm:cells,…> [uncovered=<n>:<family/direction>,…]
(`impl=` = the row module @ the first 8 hex digits of its source's sha256; a cell the tier word resolves to the stock statement runs stock's method for
that call, counted `fallback_by=cell_xla:<n>` — named, never silent; `lever_line()` renders the line and the kit's exit multiplexer prints it once at exit.)
"""
from __future__ import annotations

import collections

try:                                                    # the stack; on a host without jax the module stays importable (registry protocol, dry run) and install() refuses by name
    import jax
    import jax.numpy as jnp
except Exception:                                       # pragma: no cover - exercised on the stand-in stack (ImportError; AttributeError when a real haiku meets a stand-in jax)
    jax = jnp = None
try:                                                    # the row kernels' floor, probed for the lever's by-name refusal (_require): jax's Pallas/Triton lowering
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu
except Exception:                                       # noqa: BLE001 - no jax, or a jax without the Pallas/Triton lowering
    pl = plgpu = None

from . import provider                                 # noqa: E402 - the kit's binding of this lever to the shared core's provider row (pure Python)

NAME = "ln"
IMPL = provider.kernel_word(NAME)                       # `layers_ln@<sha8>`: the provider's row module (located, not imported: the same word on a GPU-less host)
ORIGIN = "core"                                         # the implementation lives in the shared core; this module is the adapter (class rebinding, gates, census)
MARKER = "_colabdesign_opt_layers_ln"
NUMERICS = "precision"              # KIT's registry words are exact|precision|approx: a different kernel and reduction order for the same formula (rounding-level deviations, dx relerr ~1e-5 vs haiku in f32), never bitwise
EXPECTED_FALLBACKS = ("channels_not_pow2", "channels_range", "rank", "axis", "param_axis", "dtype")   # every structural per-call step-aside word (stock's LayerNorm serves the call BY NAME); `platform` stays undeclared = fail-closed;   # the ONLY fallback word expected in this model (the 384-channel single-representation norms); any other word must fail KIT's evidence classification
PRECISION = "none"
MIN_C, MAX_C = 32, 1024

_CP = (getattr(plgpu, "TritonCompilerParams", None) or getattr(plgpu, "CompilerParams", None)) if plgpu is not None else None   # presence only: the floor _require() names

_CENSUS = {"served": collections.Counter(), "fallback": collections.Counter()}
_STATE = {"installed": False, "stock_cls": None, "K": None}          # K: the row module the word resolved to at install (provider.admit)


def _resolve(x):
    """(row module | None, its fallback word) for this call's cell under the bound word (provider.resolve): pair [N,N,C] / template [N,N,64] /
    MSA [S,N,C] / single [N,C] families by the activation's shape; None = a row this adapter does not bind for the cell (stock's method, by name)."""
    C = int(x.shape[-1])
    n = int(x.shape[-2]) if x.ndim >= 3 else int(x.shape[0])
    dt = "bf16" if x.dtype == jnp.bfloat16 else "f32"
    if x.ndim >= 3 and int(x.shape[-3]) == int(x.shape[-2]):
        shape = dict(unit=("tmpl" if C == 64 else "pair"), c=C)
    elif x.ndim >= 3:
        shape = dict(unit="msa", c=C, n_seq=int(x.shape[-3]))
    else:
        shape = dict(unit="single", c=C)
    K, _cfg = provider.resolve(NAME, n, dt, **shape)
    return K, (None if K is not None else provider.fallback_word(NAME, n, dt, **shape))


# ─────────────────────────────────────────── serving rule ───────────────────────────────────────────
def _eligibility(self, x):
    """None when served, else the fallback reason word."""
    if jax.default_backend() != "gpu":
        return "platform"
    if x.ndim < 2:
        return "rank"
    axis = tuple(a % x.ndim for a in (self.axis if isinstance(self.axis, (tuple, list)) else (self.axis,)))
    if axis != (x.ndim - 1,):
        return "axis"
    pa = self.param_axis[0] % x.ndim if self.param_axis else x.ndim - 1
    if pa != x.ndim - 1:
        return "param_axis"
    C = int(x.shape[-1])
    if C < MIN_C or C > MAX_C or (C & (C - 1)):
        return "channels_not_pow2" if (C & (C - 1)) else "channels_range"
    if x.dtype not in (jnp.bfloat16, jnp.float32):
        return "dtype"
    return None


def _ln_call(self, x, stock_call):
    """common_modules.LayerNorm.__call__ (colabdesign_opt lever ln): the instance's own parameters ('scale', 'offset', f32 [C], created only when
    the instance creates them), eps, and variance form; Pallas row kernels forward / backward; stock's method by name when not eligible."""
    import haiku as hk
    why = _eligibility(self, x)
    if why is not None:
        _CENSUS["fallback"][why] += 1
        return stock_call(self, x)
    C = int(x.shape[-1])
    K, cell_word = _resolve(x)
    if K is None:                                                     # the word named a row this adapter does not bind for this cell: stock's method, by the cell's name
        _CENSUS["fallback"][cell_word] += 1
        return stock_call(self, x)
    pdt = jnp.float32                                                 # stock: params are created against the f32-cast input -> f32
    scale = hk.get_parameter("scale", (C,), pdt, init=self.scale_init) if self._temp_create_scale else jnp.ones((C,), pdt)
    offset = hk.get_parameter("offset", (C,), pdt, init=self.offset_init) if self._temp_create_offset else jnp.zeros((C,), pdt)
    fast_var = bool(getattr(self, "use_fast_variance", False))       # honour each instance's variance form (the fused TriangleMultiplication's LNs use E[x^2]-E[x]^2)
    _CENSUS["served"][f"C{C}x{'bf16' if x.dtype == jnp.bfloat16 else 'f32'}"] += 1
    return K.layer_norm(x, scale.astype(jnp.float32), offset.astype(jnp.float32), float(self.eps), fast_var)


def install() -> dict:
    """Rebind `colabdesign.af.alphafold.model.common_modules.LayerNorm` to a same-name subclass (idempotent; haiku's metaclass wraps the
    subclass __call__, so parameters keep the module's own scope and names). Call before the design model is traced."""
    _require()
    try:
        _STATE["K"] = provider.admit(NAME)                            # the word -> the row module (cd_ln: the carried layers_ln), by the provider's selection for the main cell
    except provider.ProviderRefusal as e:
        raise Refusal(f"lever {NAME}: {e}") from None
    from colabdesign.af.alphafold.model import common_modules
    A = common_modules.LayerNorm
    if not getattr(A, MARKER, False):
        stock_call = A.__call__

        class LayerNorm(A):                             # noqa: D101
            _stock_cls = A

            def __call__(self, x):
                return _ln_call(self, x, stock_call)
        setattr(LayerNorm, MARKER, True)
        LayerNorm.__qualname__ = "LayerNorm"
        LayerNorm.__module__ = A.__module__
        _STATE["stock_cls"] = A
        common_modules.LayerNorm = LayerNorm
    _STATE["installed"] = True
    _register_exit_line()
    return {"lever": NAME, "impl": IMPL, "origin": ORIGIN, "numerics": NUMERICS, "precision": PRECISION, "row": provider.bound_row(NAME),
            "word": provider.BINDINGS[NAME].word, "provider": provider.provider_word()}


def uninstall() -> None:
    import sys
    common_modules = sys.modules.get("colabdesign.af.alphafold.model.common_modules")   # nothing to undo in a process that never imported the model
    A = getattr(common_modules, "LayerNorm", None) if common_modules else None
    if A is not None and getattr(A, MARKER, False):
        common_modules.LayerNorm = A._stock_cls
    _STATE["installed"] = False
    _STATE["K"] = None


def installed() -> bool:
    return bool(_STATE["installed"])


def stock_class():
    from colabdesign.af.alphafold.model import common_modules
    A = common_modules.LayerNorm
    return A._stock_cls if getattr(A, MARKER, False) else A


def census() -> dict:
    return {"served": int(sum(_CENSUS["served"].values())), "fallback": int(sum(_CENSUS["fallback"].values())),
            "shapes": dict(_CENSUS["served"]), "fallback_by": dict(_CENSUS["fallback"])}


# ─────────────────────────────────────────── kit lever protocol (registry.py; kit 0.4.3) ───────────────────────────────────────────
class Refusal(RuntimeError):
    """install() raises this when the lever cannot run here (levers.install turns it into the mode's refusal by name)."""


REFUSALS = (Refusal,)


def _require() -> None:
    """The lever's floor: a GPU backend and jax's Pallas/Triton lowering (tested: jax 0.6.0)."""
    problems = []
    if jax is None:
        raise Refusal(f"lever {NAME}: jax is not importable here")
    try:
        if jax.default_backend() != "gpu":
            problems.append(f"backend={jax.default_backend()} (gpu required)")
    except Exception as e:                                            # noqa: BLE001
        problems.append(f"backend: {e!r}")
    if pl is None or not hasattr(pl, "pallas_call") or _CP is None:
        problems.append("jax.experimental.pallas triton lowering absent")
    if problems:
        raise Refusal(f"lever {NAME}: " + "; ".join(problems))


def evidence() -> dict:
    c = census()
    return {"lever": NAME, "line_name": NAME, "installed": bool(_STATE["installed"]), "impl": IMPL, "origin": ORIGIN, "numerics": NUMERICS, "precision": PRECISION, **c,
            "binding": provider.report(NAME)}


def off_line(reason: str) -> str:
    from opt_core import report as _r
    from ..names import TAG
    return _r.lever_line(TAG, NAME, "off", reason=reason, impl=IMPL, origin=ORIGIN)


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
                         ("fallback_by", fb), ("shapes", shapes), *provider.facts(NAME).items(), impl=IMPL, origin=ORIGIN)


def _register_exit_line() -> None:
    """Hand lever_line to the package's one exit printer (kernels/__init__.py) — a no-op until the registry names this lever."""
    try:
        from . import register_exit_line
        register_exit_line(NAME, lever_line)
    except (ImportError, ValueError):
        pass
