"""Lever `trimul` — AF2-Multimer `TriangleMultiplication` (outgoing AND incoming; Jumper et al. Suppl. Alg. 11/12) in its fused-projection
form served by this kit's fused Pallas kernels AS THE SHARED CORE CARRIES THEM — the provider row `cd_trimul` of `opt_core.kernels.pallas`
(`opt_core/kernels/pallas/cd_trimul/trimul_pallas.py`, byte for byte the kit's former private module `kernels/trimul_pallas.py`, which left the
tree in 0.10.3), reached through `kernels/provider.py` by the provider's TIER word `fast` narrowed to the rows this adapter binds: per (jax line,
card, family, size bucket, direction) cell the provider serves its measured winner — `cd_trimul` wherever it measured ahead of the stock
multiplication — and a cell it has not measured (or measured slower) runs stock's own method for that call BY NAME (`fallback_by=cell_xla:<n>`);
the word, the row served per cell and the tier census are printed on the LEVER line (`provider= word= row= tier= [uncovered=]`). No kernel
source lives in this module: it is the adapter (the haiku class rebinding, the parameter tree, the by-name gates, the census, the LEVER line).
Forward = prologue kernel (input LayerNorm + the four projection /
gate products + mask, written directly as channel-major operand planes) → ONE cuBLAS batched GEMM (no transposes, no pads) → epilogue kernel
(centre LayerNorm + output projection + output gate); backward = `jax.custom_vjp` (two Pallas kernels + two cuBLAS GEMMs, recomputing the cheap
stages instead of saving them). Per traced design program: 48 Evoformer blocks × {outgoing, incoming} on the (N,N,128) pair activations + 2
template-pair-stack blocks × 2 on (N,N,64) — every call is served (no size gate: the kernels take any N; channel counts must be powers of two,
which every multimer_v3 config has). A config WITHOUT `fuse_projection_weights` (monomer AF2 — never built by BindCraft's multimer design model)
runs stock's own program, counted `not_fused` (declared).

Numerics class: precision. Every product keeps stock's operand class — bf16 operands / f32 accumulation under the multimer bf16 getter
(`precision=bf16` on the LEVER line; an f32 model would print `tf32`, XLA's DEFAULT f32 class = stock's); LayerNorm statistics (fast-variance
form, as stock's `_layer_norm`: modules.py:913-923) and sigmoids in f32; results rounded to the activation dtype at stock's tensor boundaries.
What moves: bf16 rounding / summation order inside the fused stages → within the per-step band, never bitwise (fast tier only).

The PATCH: `install()` rebinds `colabdesign.af.alphafold.model.modules.TriangleMultiplication` to a same-named subclass whose `__call__` creates
the SAME parameter tree (names, shapes, dtypes, initialisers: `<name>/{left_norm_input,projection,gate,center_norm,output_projection,
gating_linear}/…`) and calls the fused op; `EvoformerIteration` / the template stack look the class up at trace time, so install before the
model function is traced. Evidence: ONE LEVER line per process through the package's exit printer (kernels.register_exit_line):

    [colabdesign-opt] LEVER name=trimul_pallas state=on impl=trimul_pallas@<sha8> origin=core served=100 fallback=0 fallback_by=none
                      shapes=N600xC128xE0:48,N600xC128xE1:48,N600xC64xE0:2,N600xC64xE1:2 provider=opt_core.kernels.pallas@<core> row=cd_trimul:100
                      tier=<arm:cells,…> [uncovered=<n>:<family/direction>,…] word=fast numerics=precision precision=bf16 source=exit pid=<pid>

(`E0` = outgoing 'ikc,jkc->ijc', `E1` = incoming 'kjc,kic->ijc'; served/fallback count TRACED calls, like the attention lever's census; `impl=` = the
row module that ran @ the first 8 hex digits of its source's sha256; a call whose cell the tier word hands to the stock statement (`xla`: a cell
the provider has not measured on this card / jax line, or one where the stock multiplication measured ahead) runs stock's program, counted
`fallback_by=cell_xla`.)
"""
from __future__ import annotations

from typing import Optional

from opt_core import report as _core_report
from opt_core.counters import Ledger

from ..names import TAG
from . import provider

LEVER = "trimul"                                   # registry id (registry.LEVERS row added by KIT)
LINE_NAME = "trimul_pallas"                        # the name= token of the LEVER line
KERNEL = "trimul_pallas"                           # the row module's name (opt_core/kernels/pallas/cd_trimul/trimul_pallas.py == kernels/trimul_pallas.py)
ORIGIN = "core"                                    # the implementation lives in the shared core (provider row); the adapter is this module
EQ_OUT, EQ_IN = "ikc,jkc->ijc", "kjc,kic->ijc"     # the two equations (the row module's EQ_OUT / EQ_IN; E0 / E1 in the shapes census)
NUMERICS = "precision"
NOT_FUSED = "not_fused"                            # a config without fuse_projection_weights: stock's own program, by design
EXPECTED_FALLBACKS = (NOT_FUSED, "shape", "dtype", "equation")   # every structural per-call step-aside word _eligible can return: stock's own program serves such a call BY NAME (non-fused monomer config, an odd shape / dtype / einsum variant)
MARKER = "_colabdesign_opt_trimul"
LEDGER: Optional[Ledger] = None
_IMPL: Optional[str] = None
_K = None                                          # the row module the word resolved to at install (provider.admit)
_PRECISIONS: set = set()                            # product operand classes of the served calls (bf16 under the multimer getter)


class Refusal(RuntimeError):
    """The kernels cannot run here (no GPU backend / no Pallas-Triton lowering on this jax): the mode refuses by name, never degrades."""


REFUSALS = (Refusal,)


def kernel_impl() -> str:
    """`trimul_pallas@<sha8>` — the row module's name @ the first 8 hex digits of its source's sha256 (what ran, byte for byte; read off the
    provider's row module file without importing it, so the off / skipped lines name the same implementation)."""
    global _IMPL
    if _IMPL is None:
        _IMPL = provider.kernel_word(LEVER)
    return _IMPL


def require() -> dict:
    """The kernels' floor: jax with the Pallas Triton lowering and a GPU backend, and the provider's word for the model's main cell resolved to a
    row this adapter binds (its module imported). Raises `Refusal` (named) otherwise."""
    global _K
    try:
        import jax
        from jax.experimental import pallas as pl  # noqa: F401
        from jax.experimental.pallas import triton as plgpu
    except Exception as e:                                                    # pragma: no cover
        raise Refusal(f"trimul: jax Pallas/Triton is not importable here ({type(e).__name__}: {e})") from None
    if not (hasattr(plgpu, "TritonCompilerParams") or hasattr(plgpu, "CompilerParams")):
        raise Refusal("trimul: this jax has no Pallas Triton compiler-params API (need jax >= 0.5)")
    if jax.default_backend() != "gpu":
        raise Refusal(f"trimul: the default jax backend is {jax.default_backend()!r}, the kernels need the GPU backend")
    try:
        _K = provider.admit(LEVER)                                            # the word -> the row module (cd_trimul: the carried trimul_pallas), by the provider's selection
    except provider.ProviderRefusal as e:
        raise Refusal(f"trimul: {e}") from None
    return {"jax": jax.__version__, "backend": jax.default_backend(), "row": provider.BINDINGS[LEVER].admitted.row, "word": provider.BINDINGS[LEVER].word}


def _precision_word(dtype) -> str:
    import jax.numpy as jnp
    return "bf16" if dtype == jnp.bfloat16 else ("tf32" if dtype == jnp.float32 else str(dtype))


def precision_word() -> str:
    """`bf16` | `tf32` | `bf16+tf32` — the product operand class(es) of the calls served so far (f32 accumulation always); `bf16` before any."""
    return "+".join(sorted(_PRECISIONS)) if _PRECISIONS else "bf16"


def _eligible(module, act) -> Optional[str]:
    """None when the fused kernels serve this call, else the fallback reason word."""
    c = module.config
    if not c.get("fuse_projection_weights", False):
        return NOT_FUSED
    cz, ci = int(act.shape[-1]), int(c.num_intermediate_channel)
    pow2 = lambda v: v >= 16 and (v & (v - 1)) == 0
    if act.ndim != 3 or act.shape[0] != act.shape[1] or not pow2(cz) or not pow2(ci):
        return "shape"
    if str(act.dtype) not in ("bfloat16", "float32"):
        return "dtype"
    if c.equation not in ("ikc,jkc->ijc", "kjc,kic->ijc"):
        return "equation"
    return None


def _resolve(module, left_act):
    """(row module | None, launch config) for this call's cell under the bound word (provider.resolve; None = a row this adapter does not bind)."""
    c = module.config
    cz, ci, n = int(left_act.shape[-1]), int(c.num_intermediate_channel), int(left_act.shape[0])
    return provider.resolve(LEVER, n, _precision_word(left_act.dtype).replace("tf32", "f32"), form="af2", unit=("tmpl" if cz == 64 else "pair"),
                            c=cz, c_hidden=ci, equation=("outgoing" if c.equation == EQ_OUT else "incoming"))


def _served_call(module, left_act, left_mask, K, cfg):
    """Stock `_fused_triangle_multiplication`'s parameter tree, created through haiku exactly as stock's submodules would (same names, shapes,
    dtypes, initialisers — so loaded AlphaFold parameters and `init` both match), then the fused op of the row module `K` (launch config `cfg`:
    the row's own defaults when empty)."""
    import haiku as hk
    from colabdesign.af.alphafold.model import common_modules, utils

    c, gc = module.config, module.global_config
    cz = int(left_act.shape[-1]); ci = int(c.num_intermediate_channel)
    dt = left_act.dtype
    f32 = "float32"

    _P = _holder_class()

    def linear(name, n_in, n_out, initializer, bias_init=0.):
        w_init = common_modules.get_initializer_scale(initializer, (n_in,))
        got = _P(name=name)({"weights": ((n_in, n_out), dt, w_init), "bias": ((n_out,), dt, hk.initializers.Constant(bias_init))})
        return {"weights": got["weights"], "bias": got["bias"]}

    def layer_norm(name, n):
        got = _P(name=name)({"scale": ((n,), f32, hk.initializers.Constant(1.)), "offset": ((n,), f32, hk.initializers.Constant(0.))})
        return {"scale": got["scale"], "offset": got["offset"]}

    params = {
        "left_norm_input": layer_norm("left_norm_input", cz),
        "projection": linear("projection", cz, 2 * ci, "linear"),
        "gate": linear("gate", cz, 2 * ci, utils.final_init(gc), bias_init=1.),
        "center_norm": layer_norm("center_norm", ci),
        "output_projection": linear("output_projection", ci, cz, utils.final_init(gc)),
        "gating_linear": linear("gating_linear", cz, cz, utils.final_init(gc), bias_init=1.),
    }
    if LEDGER is not None:
        LEDGER.serve(f"N{int(left_act.shape[0])}xC{cz}xE{0 if c.equation == EQ_OUT else 1}")
    _PRECISIONS.add(_precision_word(dt))
    return K.triangle_multiplication(left_act, left_mask, params, equation=c.equation, cfg=(cfg or None))


_HOLDER = None


def _holder_class():
    """`_P(name)(specs)`: an hk.Module that only creates/reads parameters under the given submodule name (no compute) — defined once."""
    global _HOLDER
    if _HOLDER is None:
        import haiku as hk

        class _P(hk.Module):
            def __call__(self, specs):
                return {k: hk.get_parameter(k, shape, dtype, init=init) for k, (shape, dtype, init) in specs.items()}
        _HOLDER = _P
    return _HOLDER


def _rebound_class(A: type) -> type:
    stock_call = A.__call__

    class TriangleMultiplication(A):                # noqa: D101 — created in a class body so haiku's metaclass wraps __call__ (module name unchanged)
        _stock_cls = A

        def __call__(self, left_act, left_mask):
            reason = _eligible(self, left_act)
            if reason is None:
                K, cfg = _resolve(self, left_act)
                if K is not None:
                    return _served_call(self, left_act, left_mask, K, cfg)
                c = self.config                                                 # the word named a row this adapter does not bind for this cell: stock's program, by the cell's name
                reason = provider.fallback_word(LEVER, int(left_act.shape[0]), _precision_word(left_act.dtype).replace("tf32", "f32"), form="af2",
                                                unit=("tmpl" if int(left_act.shape[-1]) == 64 else "pair"), c=int(left_act.shape[-1]),
                                                c_hidden=int(c.num_intermediate_channel), equation=("outgoing" if c.equation == EQ_OUT else "incoming"))
            if LEDGER is not None:
                LEDGER.fallback(reason)             # stock's own program; `not_fused` is declared, anything else refuses the gate (fail-closed)
            return stock_call(self, left_act, left_mask)

    setattr(TriangleMultiplication, MARKER, True)
    TriangleMultiplication.__qualname__ = "TriangleMultiplication"
    TriangleMultiplication.__module__ = A.__module__
    return TriangleMultiplication


def install() -> dict:
    """Require the kernels, rebind `colabdesign.af.alphafold.model.modules.TriangleMultiplication`, register the exit line. Idempotent; call
    before the model function is traced."""
    global LEDGER
    facts = require()
    from colabdesign.af.alphafold.model import modules as CDM
    if LEDGER is None:
        LEDGER = Ledger(LINE_NAME, impl=kernel_impl(), origin=ORIGIN, expected=EXPECTED_FALLBACKS)
    A = CDM.TriangleMultiplication
    if not getattr(A, MARKER, False):
        CDM.TriangleMultiplication = _rebound_class(A)
    from . import register_exit_line
    try:
        register_exit_line(LEVER, exit_line)                                  # the package's one exit printer (registry order)
    except ValueError:                                                        # a tree whose registry does not list this lever yet: print the line itself
        import atexit
        atexit.register(lambda: _core_report.emit(exit_line()))
    return {"impl": kernel_impl(), "origin": ORIGIN, "numerics": NUMERICS, "expected_fallbacks": list(EXPECTED_FALLBACKS), **facts, "provider": provider.provider_word()}


def uninstall() -> None:
    """Restore the stock class (unit tests, in-process A/B; built models keep their programs). The Ledger stays."""
    import sys
    mods = sys.modules.get("colabdesign.af.alphafold.model.modules")
    A = getattr(mods, "TriangleMultiplication", None) if mods else None
    if A is not None and getattr(A, MARKER, False):
        mods.TriangleMultiplication = A._stock_cls


def installed() -> bool:
    import sys
    mods = sys.modules.get("colabdesign.af.alphafold.model.modules")
    return bool(mods and getattr(getattr(mods, "TriangleMultiplication", None), MARKER, False))


def off_line(reason: str) -> str:
    """The lever's line in a kit-route mode that does not select it: state=off (nothing imported beyond this module)."""
    return _core_report.lever_line(TAG, LINE_NAME, "off", reason=reason, impl=kernel_impl(), origin=ORIGIN)


def exit_line() -> str:
    """The lever's ONE evidence line: the Ledger's census (served / fallback by reason / shapes) + numerics class + product precision class."""
    if LEDGER is None:
        return off_line("not_installed")
    for k, v in provider.facts(LEVER).items():                                    # provider= row= tier= [uncovered=] word= (the Ledger prints its facts sorted, before numerics=)
        LEDGER.set(k, v)
    return LEDGER.line(TAG, numerics=NUMERICS, precision=precision_word(), source="exit")


def evidence() -> dict:
    if LEDGER is None:
        return {}
    g = LEDGER.gate()
    return {**LEDGER.fields(), "numerics": NUMERICS, "precision": precision_word(), "gate_ok": bool(g.ok), "gate_reason": g.reason, "binding": provider.report(LEVER)}
