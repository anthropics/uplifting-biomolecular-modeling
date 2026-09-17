"""The shared core (``common/opt_core``) this package stands on: pinned in ``opt/pyproject.toml`` ``[tool.opt_core]`` (path, MINIMUM
version — a floor), installed beside the kit (``pip install -e <tree>/common/opt_core -e <tree>/chai1/opt``) or reached through the tree
layout — ``<tree>/common/opt_core`` beside the engine directory — when it is not installed (``ensure_importable``: one ``sys.path`` entry,
appended, only when ``opt_core`` does not resolve). ``core_gate`` is statement one of every entry route (the CLI's verbs, the driver, the ``.pth`` hook's activation,
``stack.activate`` — the in-process API): THE pin gate, the shared template ``_core_gate.py`` carried byte-identical inside this
package (``common/opt_core/kit_template/_core_gate.py``) — it locates ``opt_core`` without importing it and reads its ``__version__``
literal, never a byte manifest: an absent core is ``NOT ACTIVE: reason=core_missing:opt_core (pinned >= v<want> at <path>; nothing
importable as opt_core on sys.path)``, one older than the pin ``reason=core_mismatch: opt_core pinned >= v<want> at <path>, installed
v<have> at <dir>``, exit 3, before any ``opt_core`` import, never a traceback. ``producers_refusal`` is the second, finer statement: a
matching (or newer) core that still lacks a module this package imports is ``reason=producer_missing:<module>,… (…)``.
"""
import importlib.util
import os
import sys

PYPROJECT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pyproject.toml")

# Every ``opt_core`` module this package imports (each import is inside the function that needs it; tests/test_core_adoption.py locks this
# tuple against the package's source). A core without one of them is an older core: refused by name before anything resolves.
REQUIRED_PRODUCERS = (
    "opt_core.counters",
    "opt_core.gates",
    "opt_core.kernels.ln",
    "opt_core.kernels.triattn",
    "opt_core.mem",
    "opt_core.mem.chunk",
    "opt_core.mem.ckpt",
    "opt_core.mem.compose",
    "opt_core.mem.ngpu",
    "opt_core.mem.record",
    "opt_core.mem.registry",
    "opt_core.mem.torch_alloc",
    "opt_core.modes",
    "opt_core.oom",
    "opt_core.precision.policy",
    "opt_core.report",
    "opt_core.stock_proof",
    "opt_core.trimul",
)


def tree_core_dir() -> str:
    """``<tree>/common/opt_core`` for the tree this package sits in (``opt/chai1_opt`` → ``../../../common/opt_core``): the pin's path
    resolved against the pyproject that carries it."""
    ensure_importable()
    from opt_core.gates import core_pin
    return core_pin(PYPROJECT)["abs_path"]


def _pin_path_import_free() -> str:
    for ln in open(PYPROJECT, "r", encoding="utf-8"):
        if ln.strip().startswith("path") and "=" in ln:
            return os.path.normpath(os.path.join(os.path.dirname(PYPROJECT), ln.split("=", 1)[1].strip().strip('"').strip("'")))
    raise ValueError(f"{PYPROJECT}: [tool.opt_core] path missing")


def ensure_importable() -> str:
    """Make ``opt_core`` importable: the installed distribution when there is one, else the tree's ``common/opt_core`` appended to
    ``sys.path`` (never in front of anything). Returns how it resolves; raises ImportError naming both places when it does not."""
    if importlib.util.find_spec("opt_core") is not None:
        return "installed"
    d = _pin_path_import_free()
    if os.path.isdir(os.path.join(d, "opt_core")):
        if d not in sys.path:
            sys.path.append(d)
        importlib.invalidate_caches()
        if importlib.util.find_spec("opt_core") is not None:
            return f"tree:{d}"
    raise ImportError(f"opt_core is neither installed nor at the pin's path {d} (install the core beside this kit: pip install -e <tree>/common/opt_core)")


TAG = "chai1-opt"                                       # the tag of every line this package prints ([chai1-opt] …; report.PREFIX)


def core_gate(stream=None) -> dict:
    """THE pin gate (``_core_gate.gate``): the facts ``{"pinned", "installed", "tag"}`` when the ``opt_core`` this interpreter
    would import is the one ``[tool.opt_core]`` pins; otherwise one ``NOT ACTIVE`` line on ``stream`` (default stderr) and
    ``_core_gate.CoreGateRefused`` (``SystemExit(3)``). The tree layout is honoured first (``ensure_importable``: the pin's path appended to
    ``sys.path`` when no ``opt_core`` is installed — it imports nothing of the core), so the gate sees the core this package would use."""
    try:
        ensure_importable()
    except ImportError:
        pass                                            # nothing importable: the gate names it (core_missing)
    from ._core_gate import gate
    return gate(PYPROJECT, tag=TAG, stream=stream)


def entry_gate() -> dict:
    """Statement one of the process entries (``cli.main``, ``driver.main``, the ``.pth`` hook's activation): ``core_gate`` (exit 3 by
    name on an absent or stale core), then the finer ``producers_refusal`` (exit 3 by name on a core that lacks a module this package
    imports). Returns the gate's facts."""
    facts = core_gate()
    refusal = producers_refusal()
    if refusal is not None:
        sys.stderr.write(f"[{TAG}] NOT ACTIVE: {refusal}\n"); sys.stderr.flush()
        raise SystemExit(3)                             # EXIT_NOT_ACTIVE (cli.py)
    return facts


def missing_producers() -> list:
    """The names in :data:`REQUIRED_PRODUCERS` the importable core does not carry, read from the core's package directory (no core module
    is executed: a module ``a.b.c`` is present when ``a/b/c.py`` or ``a/b/c/__init__.py`` exists under the directory ``opt_core`` resolves
    to). Raises ImportError (``ensure_importable``) when there is no core at all."""
    ensure_importable()
    spec = importlib.util.find_spec("opt_core")
    roots = list(spec.submodule_search_locations or []) if spec is not None else []
    if not roots:
        return list(REQUIRED_PRODUCERS)
    missing = []
    for name in REQUIRED_PRODUCERS:
        parts = name.split(".")[1:]
        found = False
        for root in roots:
            base = os.path.join(root, *parts) if parts else root
            if (parts and os.path.isfile(base + ".py")) or os.path.isfile(os.path.join(base, "__init__.py")):
                found = True
                break
        if not found:
            missing.append(name)
    return missing


def producers_refusal():
    """None when the core is importable and carries every module this package imports; else the reason text of the NOT ACTIVE line:
    ``reason=core_missing:opt_core (…)`` for an absent core, ``reason=producer_missing:<m>,<m> (…)`` for a core that lacks modules."""
    try:
        missing = missing_producers()
    except ImportError as e:
        return f"reason=core_missing:opt_core ({e})"
    if not missing:
        return None
    spec = importlib.util.find_spec("opt_core")
    where = (list(spec.submodule_search_locations or [None]) or [None])[0] if spec is not None else None
    version = "?"
    try:
        for ln in open(os.path.join(where, "__init__.py"), "r", encoding="utf-8"):
            if ln.startswith("__version__"):
                version = ln.split("=", 1)[1].strip().strip('"').strip("'"); break
    except (OSError, TypeError):
        pass
    return (f"reason=producer_missing:{','.join(missing)} (the importable opt_core {version} at {where} lacks modules this package imports: "
            f"an older core; install the core this kit pins, opt/pyproject.toml [tool.opt_core], beside it: pip install -e <tree>/common/opt_core)")
