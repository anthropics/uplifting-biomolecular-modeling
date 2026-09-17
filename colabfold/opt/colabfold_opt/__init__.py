"""colabfold_opt — explicit interface to the ColabFold (AlphaFold-Multimer v3) inference optimization.

    import colabfold_opt
    report = colabfold_opt.enable("fast", queries=queries)   # the run's queries (their size: the sub-batch lever, the tokens= census); idempotent, once per process
    colabfold_opt.enable("fast")                              # no queries: the hook on colabfold.batch.run activates at the run call
    colabfold_opt.status()                                    # the last activation report
    colabfold_opt.check("fast")                               # the dry run: resolves and gates on this machine, applies nothing

or, without code changes, `COLABFOLD_OPT=fast colabfold_batch <args>`: the package's .pth installs a lazy finder that hooks
`colabfold.batch.run` on the first import of `colabfold.batch` (_autoload.py). Nothing is imported at interpreter start beyond this
module and _autoload (the package's other modules load at the first call; PEP 562 `__getattr__` below); jax and colabfold load only
when the caller imports them.

The levers (modes.py names each with its module): DEVICE_RESIDENT (device_resident.py: alphafold.model.model.RunModel rebound so the
Python recycle loop keeps its operands on the device and fetches the float16 outputs once — placement only), SUBBATCH (subbatch.py),
TRIMUL_PALLAS (trimul_pallas.py over the shared core's provider face by the mode's tier word,
opt_core.kernels.pallas.serve), the carried Pallas flash-attention adapter patched over `alphafold.model.modules.Attention` by its own
`af2_pallas_attn.enable()` (opt/forward/af2_pallas_flash/af2_pallas_flash/af2_pallas_attn.py:69-103; the activation follows its
README.md:23-25: the kit directory on the path, `AF_PALLAS_ATTN=1`, the module imported), PALLAS_MSA (msa_attn.py), TRIATTN_XLA (triattn_xla.py), MSA_COL_CUDNN (msa_col_cudnn.py), TRANSITION (transition.py), TEMPL_DEDUP (templ_dedup.py: identical template rows embedded once in the
multimer template scan), ROWPAIR (big.py) and
the deployment lever XLA_CACHE (xla_cache.py: JAX's persistent compilation cache keyed by the stack and the numerics recipe) — all applied in the process that calls
`colabfold.batch.run`, before the model runners are built (colabfold/batch.py:1537). Modes (modes.TABLE): off = stock, exact =
DEVICE_RESIDENT (Tier 1 expected), fast = DEVICE_RESIDENT + SUBBATCH + TRIMUL_PALLAS + AF_PALLAS_ATTN + PALLAS_MSA + TRIATTN_XLA + MSA_COL_CUDNN + TEMPL_DEDUP
+ TRANSITION (Tier 2), big = fast's set + ROWPAIR at --n_gpu P > 1. Default mode modes.DEFAULT_MODE = fast. Prediction settings
are colabfold_batch's own command line, passed through verbatim in every mode (settings.py: stock's names and defaults). The stock route (mode off) is stock_pred.py: `colabfold_batch` in a cleaned
subprocess that proves its environment. CLI: `colabfold-opt pred|check|warm` (cli.py).

The activation report (enable / status / check; `pred`'s launch record): active, mode, levers, levers_applied, levers_unavailable, partial,
reason (when inactive), tokens_min, tokens_max, queries, gpu, key (<cc>|<jax>), stack_key,
colabfold_version, alphafold_colabfold_version, jax_version, package_version, kit, kit_env, kit_state (the kit's _STATE), pins, weights,
notes, route, trigger, data_dir, patch_marker, switches_not_wired.
"""
from __future__ import annotations

from typing import Optional

__version__ = "0.2.23"
__all__ = ["enable", "status", "check", "ActivationError", "UnsupportedMode", "__version__"]
_LAZY = {"UnsupportedMode": "modes", "ActivationError": "stack"}


def __getattr__(name: str):
    """PEP 562: the exception classes resolve on first use (their modules are not imported at interpreter start)."""
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(f"{__name__}.{_LAZY[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _package():
    """The package's modules behind the ONE guard of the in-process route: the core pin gate first (an absent opt_core, or another one than
    opt/pyproject.toml pins, is `NOT ACTIVE: reason=core_missing:opt_core …` / `reason=core_mismatch: …` and SystemExit(3)), then the imports
    (a module of the pinned core missing is `NOT ACTIVE: core_missing:<module>` and SystemExit(3)) — the words and the code of the command
    line and of the environment route, never a traceback out of `enable()` / `status()` / `check()`."""
    from ._autoload import core_missing, gate_core
    gate_core()                                                            # the pinned core, or NOT ACTIVE and SystemExit(3) — before any opt_core import
    try:
        from . import modes as _modes, stack as _stack
    except ImportError as e:
        raise SystemExit(core_missing(e)) from None
    return _modes, _stack


def enable(mode: Optional[str] = None, queries=None, strict: bool = False) -> dict:
    """Activate ``mode`` (default modes.DEFAULT_MODE) in this process and return the activation report. With ``queries`` (the list
    colabfold.batch.run receives) the levers are applied now, sized by the run; without them the hook is installed on
    ``colabfold.batch.run`` and the report says so (active False, the activation happens at the run call). ``off`` applies nothing. Raises
    UnsupportedMode for an unknown name; under ``strict`` a refused activation raises ActivationError after its NOT ACTIVE line."""
    _modes, _stack = _package()
    mode = _modes.DEFAULT_MODE if mode is None else mode
    if mode not in _modes.MODES:
        _modes.resolve(mode)                                              # raises UnsupportedMode by name
    if mode == "off" or queries is not None:
        return _stack.activate(mode, queries=queries, strict=strict, route="in-process")
    _stack.hook_run(mode, strict=strict, trigger=None)
    rep = _stack.status()
    if rep.get("active") or rep.get("mode") == mode:
        return rep
    return {"active": False, "mode": mode, "hooked": True,
            "reason": f"hooked: the gates are decided at the {_stack.TRIGGER_MODULE}.{_stack.RUN_FUNCTION} call"}


def status() -> dict:
    _modes, _stack = _package()
    return _stack.status()


def check(mode: Optional[str] = None, data_dir: Optional[str] = None, n_gpu: Optional[int] = None) -> dict:
    _modes, _stack = _package()
    return _stack.check(_modes.DEFAULT_MODE if mode is None else mode, data_dir=data_dir, n_gpu=n_gpu)
