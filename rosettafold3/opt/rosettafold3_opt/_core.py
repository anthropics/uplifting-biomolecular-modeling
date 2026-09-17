"""The pinned core (``opt_core``, ``common/opt_core/``): the shared mechanisms this package stands on.

``opt/pyproject.toml`` ``[tool.opt_core]`` pins the core by path (relative to ``opt/``) and by a MINIMUM version (a floor: a newer core
passes, an older one is refused). ``load(name)`` returns ``opt_core.<name>``: from the interpreter that carries the core (``pip install
-e <tree>/common/opt_core``), else from the pin's path (a tree checkout whose interpreter has only this package on ``PYTHONPATH`` — the
``install`` command's own case); a core importable from neither is a named error. Which one was imported is ``origin()``; ``stack.activate``'s
core gate (``_core_gate.gate``, via ``stack.core_gate``) proves the imported core's version against the pin before anything is applied, so
the two routes cannot differ silently.
"""
import importlib
import importlib.util
import os
import sys

PYPROJECT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pyproject.toml")
_STATE = {"origin": None}


class CoreError(ImportError):
    """opt_core is importable neither from this interpreter nor from the pin's path."""


def pin_path() -> str:
    """The pinned core's directory (``[tool.opt_core].path`` resolved against ``opt/``)."""
    import tomllib
    with open(PYPROJECT, "rb") as fh:
        table = tomllib.load(fh).get("tool", {}).get("opt_core", {})
    if not table.get("path"):
        raise CoreError(f"{PYPROJECT}: [tool.opt_core] lacks path")
    return os.path.normpath(os.path.join(os.path.dirname(PYPROJECT), table["path"]))


MIN_CORE = "0.5.10.4"                               # the oldest core that carries every module in REQUIRED (named in the producer_missing / core_missing reason)
REQUIRED = (                                        # every ``opt_core`` module this package loads (``load(...)`` call sites), checked BEFORE any command resolves anything
    "opt_core", "opt_core.gates", "opt_core.report", "opt_core.home", "opt_core.kernels", "opt_core.process",
    "opt_core.jit_cache", "opt_core.modes", "opt_core.stock_proof", "opt_core.attn.size_gate",
    "opt_core.mem", "opt_core.mem.mode", "opt_core.mem.compose", "opt_core.mem.record", "opt_core.mem.torch_alloc", "opt_core.mem.registry", "opt_core.mem.patchset",
    "opt_core.mem.chunk", "opt_core.mem.offload",
    "opt_core.mem.ngpu", "opt_core.mem.rowpair", "opt_core.mem.rowpair.launch", "opt_core.attn.pair_fused", "opt_core.kernels.transition", "opt_core.oom",
    "opt_core.attn.sdpa_bias",                          # the --n_gpu P>1 line's attention-pair-bias on local query rows (rowpair._apb_fn)
)


class ProducersMissing(CoreError):
    """The pinned core is importable but lacks modules this package loads (an older core), or is absent altogether. ``str()`` is the
    ``reason=`` word and sentence of the NOT ACTIVE line (``producer_missing:<m>,<m> — …`` / ``core_missing:opt_core — …``)."""


def pin_version() -> str:
    """``[tool.opt_core].version`` of ``opt/pyproject.toml`` (the pinned core's version)."""
    import tomllib
    with open(PYPROJECT, "rb") as fh:
        return str(tomllib.load(fh).get("tool", {}).get("opt_core", {}).get("version") or "?")


def missing_producers() -> list:
    """The members of :data:`REQUIRED` that cannot be found on this interpreter (or on the pin's path, the ``install`` command's case) —
    ``["opt_core"]`` alone when the core itself is absent. Finds specs; executes no producer body beyond the packages on the way."""
    import importlib.util
    try:
        _import()
    except (ImportError, OSError):                          # not installed, and no readable pin table / pinned checkout beside this package: absent
        return ["opt_core"]
    missing = []
    for name in REQUIRED[1:]:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):                  # a missing parent package raises instead of returning None
            missing.append(name)
    return missing


def producers_reason(missing) -> str:
    """The NOT ACTIVE reason for a core that lacks producers: names every missing module and the remedy (one line)."""
    try:
        pinned = pin_version()
    except Exception:                                       # noqa: BLE001 — the pin table unreadable is its own named error elsewhere; the refusal still names the modules
        pinned = "?"
    if list(missing) == ["opt_core"]:
        head = "core_missing:opt_core"
        why = f"opt_core is not importable on {sys.executable}"
    else:
        head = "producer_missing:" + ",".join(missing)
        why = f"the importable opt_core lacks {len(missing)} module(s) this package loads"
    return (f"{head} — {why}; this package imports opt_core >= {MIN_CORE} (pinned: [tool.opt_core] version {pinned}): "
            f"pip install -e <tree>/common/opt_core into this interpreter")


def require() -> None:
    """Refuse BY NAME before anything resolves when the core or a module this package loads is absent (an older core): raises
    :class:`ProducersMissing` BEFORE any producer is used (the CLI, the hook, ``run.sh`` and the config print
    ``[rosettafold3-opt] NOT ACTIVE: reason=<…>`` and exit 3 on it). The pinned-version comparison is the activation's core gate
    (``stack.core_gate``, via ``_core_gate.gate``)."""
    missing = missing_producers()
    if missing:
        raise ProducersMissing(producers_reason(missing))


def _import():
    try:
        mod = importlib.import_module("opt_core")
        _STATE["origin"] = _STATE["origin"] or "installed"
        return mod
    except ImportError:
        pass
    root = pin_path()
    if not os.path.isdir(os.path.join(root, "opt_core")):
        raise CoreError(f"opt_core is not installed on {sys.executable} and the pinned path {root} carries no opt_core/ package: "
                        f"pip install -e {root}")
    if root not in sys.path:
        sys.path.append(root)                                   # after every installed entry: an installed core would have won above
    mod = importlib.import_module("opt_core")
    _STATE["origin"] = "pin_path"
    return mod


def load(name: str):
    """``opt_core.<name>`` (module contract)."""
    _import()
    return importlib.import_module(f"opt_core.{name}")


def origin():
    """``installed`` (the interpreter carries the core) · ``pin_path`` (loaded from the pinned directory) · None (not imported yet)."""
    return _STATE["origin"]


def not_active_reason(exc: BaseException) -> str:
    """The ``reason=`` word of the NOT ACTIVE line for a failure to IMPORT what activation needs — ``core_missing:<module>`` when the
    pinned core (``opt_core`` or one of its modules) is not importable, ``import_error:<module>`` for any other missing module,
    ``activation_error:<Type>`` otherwise — followed by the exception's own words. One line, never a traceback (the hook and
    ``python -m rosettafold3_opt`` exit 3 on it: stock never runs silently under ROSETTAFOLD3_OPT)."""
    if isinstance(exc, ProducersMissing):
        return f"reason={exc}"
    if isinstance(exc, ImportError):
        name = getattr(exc, "name", None) or ""
        if isinstance(exc, CoreError) or name.startswith("opt_core") or "opt_core" in str(exc):
            return f"reason=core_missing:{name or 'opt_core'} ({exc})"
        return f"reason=import_error:{name or 'unknown'} ({exc})"
    return f"reason=activation_error:{type(exc).__name__} ({exc})"


NOT_ACTIVE_PREFIX = "[rosettafold3-opt] NOT ACTIVE:"     # the one refusal-line prefix of the entry routes below (report.PREFIX + " NOT ACTIVE:" once the core resolves)
EXIT_NOT_ACTIVE = 3                                       # the kit's refusal exit code (cli.EXIT_NOT_ACTIVE, _core_gate.EXIT_NOT_ACTIVE: the same value)


def require_or_exit() -> None:
    """``require()`` as EVERY entry route runs it, right after the pin gate (``python -m rosettafold3_opt``, ``_require`` — run.sh's and the
    config env's probe —, the CLI's ``main``, ``enable()`` and through it the ``.pth`` hook): a core lacking a module this package loads — an
    older core, or one whose ``__version__`` claims the pin while its tree does not carry the modules this package loads — is ONE line
    ``[rosettafold3-opt] NOT ACTIVE: reason=producer_missing:<module>,… — <sentence>`` on stderr and ``SystemExit(3)``; never a traceback,
    nothing resolved. Silent when every producer is present."""
    try:
        require()
    except CoreError as e:
        sys.stderr.write(f"{NOT_ACTIVE_PREFIX} {not_active_reason(e)}\n")
        sys.stderr.flush()
        raise SystemExit(EXIT_NOT_ACTIVE) from None



def expose_pin_path() -> bool:
    """The tree-checkout route: when nothing is importable as ``opt_core`` and the pinned checkout (``[tool.opt_core] path``) carries the
    package, put that directory on ``sys.path`` — after every installed entry, so an installed core always wins — so the pin gate
    (``_core_gate.gate``) and every later import see the pinned core. True when the path was added."""
    try:
        if importlib.util.find_spec("opt_core") is not None:
            return False
        root = pin_path()
    except Exception:                                       # noqa: BLE001 — an unreadable pin is the gate's line to print (core_pin_unreadable), not this helper's
        return False
    if os.path.isdir(os.path.join(root, "opt_core")) and root not in sys.path:
        sys.path.append(root)
        return True
    return False


expose_pin_path()

