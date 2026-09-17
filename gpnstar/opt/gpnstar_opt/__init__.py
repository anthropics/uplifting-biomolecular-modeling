"""gpnstar_opt — inference optimizations for GPN-Star (upstream ``gpn`` 0.9.0 at its pinned commit), one mode: ``exact``.

The kit is a drop-in: with it enabled, every model upstream's inference wrappers build in the process (``gpn star vep | logits |
embedding``, or ``gpn.star.inference.MLMforVEPModel`` and friends in your own code) carries the levers from the moment it sits on the GPU and
returns the same bytes the stock forward returns under the same numerics flags, faster. Nothing about the model's interface changes.

    GPNSTAR_OPT=exact gpn star vep ...              # no code change (the package's .pth arms it at interpreter start)
    import gpnstar_opt; gpnstar_opt.enable()         # or explicitly, BEFORE the model is built
    gpnstar_opt.apply(model)                         # a GPNStarForMaskedLM / GPNStarModel you loaded yourself, ALREADY on the GPU

Levers (all eight, always; ``gpnstar_opt.accel``): ``devconst`` (phylogenetic constants resident on the GPU), ``constcache`` (per-shape
constants computed once), ``srcgather`` (device-side species gather), ``unifiedkv`` + ``dedup`` (the column cross-attention's clade K/V
projected once per layer over the distinct alignment rows), ``colattn`` (that attention run as the same op sequence on operands built
once in the layout its matmuls read), ``fusedattn`` (the same attention read straight from the distinct rows in cuBLAS's own summation order where
the batch shape's first forward proves it equal) and ``graph`` (small batch shapes replayed from one whole-forward CUDA graph per shape). The levers set no numerics flag: they follow the
caller's (``gpn star ... --tf32`` runs TF32 matmuls in stock and kit alike). Every line the package prints starts with ``[gpnstar-opt]`` on
stderr. ``enable()`` refuses by name (``NOT ACTIVE``, nothing applied, stock untouched) when no CUDA device is visible, when the installed
``gpn`` is not the pinned stock, or when ``transformers`` is off upstream's pin; a dependency merely off its pin or a card outside the
tested classes is named on the ACTIVE line and served. ``python -m gpnstar_opt check`` (``run.sh check``) is the dry run: the same
resolution, nothing applied.
"""
__version__ = "0.2.0"
__all__ = ["enable", "disable", "apply", "check", "status", "ActivationError", "PREFIX", "__version__"]

from ._core_gate import gate as _gate
from ._names import MODE as _MODE, TAG as _TAG

PREFIX = f"[{_TAG}]"


class ActivationError(RuntimeError):
    """The mode could not be activated (a gate refused, the levers refused at engage, or another mode word was asked)."""


def _stack():
    _gate(__file__, _TAG)                                 # the core pin gate first: [gpnstar-opt] NOT ACTIVE: reason=core_missing|core_mismatch … -> exit 3
    from . import stack
    stack.ActivationError = ActivationError               # one refusal type across the package
    return stack


def _mode(mode) -> None:
    """The one mode word is ``exact`` (None = it); anything else is refused by name."""
    w = _MODE if mode is None or not str(mode).strip() else str(mode).strip().lower()
    if w != _MODE:
        raise ActivationError(f"unknown mode {w!r}: this kit has one mode, {_MODE}")


def enable(mode: str = None, *, strict: bool = False, trigger: str = None) -> dict:
    """Activate the kit in this process: gate the machine and arm upstream's inference wrappers so the levers apply to every model they build,
    once it is on the GPU. Returns the activation report; ``strict=True`` raises ActivationError instead of returning an inactive report."""
    try:
        _mode(mode)
    except ActivationError as e:
        from . import report
        report.emit(report.not_active_line(str(e)))
        if strict:
            raise
        return {"active": False, "reason": str(e)}
    return _stack().activate(strict=strict, trigger=trigger, arm=True)


def apply(model, mode: str = None) -> dict:
    """Apply the levers to a GPN-Star model YOU loaded, ALREADY ON THE GPU (``model.cuda()`` first), in place; prints ACTIVE + LEVER and
    returns the engaged record. Activates the kit when the process has not (without arming the wrappers). A refusal raises ActivationError."""
    _mode(mode)
    stack = _stack()
    rep = stack.activate(strict=True, arm=False)
    return stack.apply(model, rep=rep)


def disable() -> dict:
    """Withdraw the kit from this process: the wrapper hooks are removed (models built afterwards are stock) and the kit's reporters come off
    the models it engaged; a model the levers were applied to keeps them (its outputs are the stock's either way). Prints REMOVED."""
    return _stack().disable()


def status() -> dict:
    """The last activation report (+ the K/V route tallies once forwards have run), or {"active": False, "reason": ...} before enable()."""
    return _stack().status()


def check(mode: str = None) -> dict:
    """Dry run: resolve and gate the kit on this machine and apply nothing; the report ``run.sh check`` prints (report["would_refuse"])."""
    _mode(mode)
    return _stack().activate(dry_run=True)


def __getattr__(name):                                   # PEP 562: keep `import gpnstar_opt` (the .pth) free of any other import
    if name in ("registry", "stack", "report", "weights", "_autoload"):
        import importlib
        _gate(__file__, _TAG)
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
