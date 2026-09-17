"""The shared core (``common/opt_core``) this package stands on: pinned in ``opt/pyproject.toml`` ``[tool.opt_core]`` (path, version — a
floor: the installed core must be at least this version), installed beside the kit (``pip install -e <tree>/common/opt_core -e
<tree>/genie3/opt``) or reached through the tree layout — ``<tree>/common/opt_core`` beside the engine directory — when it is not
installed (``reach``: one ``sys.path`` entry, appended, only when ``opt_core`` does not resolve). ``core_gate`` is statement one of every
entry of the package (``python -m genie3_opt`` / the console script, ``enable()``, ``check()``, ``run_design()``, the ``.pth`` finder's
trigger, ``run.sh``, ``configs/<gpu>.env``): the core pin gate (``_core_gate.gate`` — ``common/opt_core/kit_template/_core_gate.py``
byte for byte) locates the installed core WITHOUT importing it, reads its ``__version__`` literal and compares against the pin, refusing
by name — one ``[genie3-opt] NOT ACTIVE: reason=core_missing | core_mismatch | core_pin_unreadable …`` line, exit 3 — on an absent or
older-than-pinned core; a newer core passes. This module imports nothing of the core at module level (the standard library, ``codes`` and
``_core_gate`` only).
"""
import importlib
import importlib.util
import os
import sys

from ._core_gate import gate, read_table
from .codes import TAG

PYPROJECT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pyproject.toml")


def tree_core_dir() -> str:
    """``<tree>/common/opt_core`` for the tree this package sits in: the pin's ``path`` resolved against the pyproject that carries it
    (``opt/genie3_opt`` → ``../../../common/opt_core``)."""
    path = read_table(PYPROJECT, "tool.opt_core").get("path")
    if not path:
        raise ValueError(f"{PYPROJECT}: [tool.opt_core] path missing")
    return os.path.normpath(os.path.join(os.path.dirname(PYPROJECT), path))


def reach() -> None:
    """Append the tree's ``common/opt_core`` to ``sys.path`` (never in front of anything) when ``opt_core`` does not resolve and the tree
    has it. Import-free and silent: whether the core then resolves, and whether it is the pinned one, is ``core_gate``'s to say."""
    if importlib.util.find_spec("opt_core") is not None:
        return
    try:
        d = tree_core_dir()
    except (OSError, ValueError):
        return
    if os.path.isdir(os.path.join(d, "opt_core")) and d not in sys.path:
        sys.path.append(d)
        importlib.invalidate_caches()


def core_gate() -> dict:
    """Statement one of every entry: ``reach()``, then the core pin gate under this package's tag. Returns the gate's facts
    (``{"pinned": {...}, "installed": {...}, "tag"}``); an absent / mismatched / pre-manifest core or an unreadable pin is its one
    NOT ACTIVE line on stderr and ``SystemExit(3)`` (``_core_gate.CoreGateRefused``) — never a traceback, never a silent stock run."""
    reach()
    return gate(__file__, tag=TAG)


def ensure_importable() -> str:
    """``reach()`` for the package's internals that import the core function by function: returns how ``opt_core`` resolves
    (``installed`` | ``tree:<dir>``); raises ImportError naming both places when it does not (an entry's ``core_gate`` has refused
    before any internal runs, so this is the child-process and test path)."""
    if importlib.util.find_spec("opt_core") is not None:
        d = tree_core_dir()
        return f"tree:{d}" if d in sys.path else "installed"
    reach()
    if importlib.util.find_spec("opt_core") is not None:
        return f"tree:{tree_core_dir()}"
    raise ImportError(f"opt_core is neither installed nor at the pin's path {tree_core_dir()} (install the core beside this kit: pip install -e <tree>/common/opt_core)")
