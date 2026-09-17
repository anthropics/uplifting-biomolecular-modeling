"""The shared core (``opt_core``, common/opt_core) made importable before the package's first ``import opt_core``.

The installed distribution when there is one (``pip install -e common/opt_core -e protenix_v2/opt``); otherwise the copy at the path the
kit pins in ``pyproject.toml`` ``[tool.opt_core]`` (a run from ``PYTHONPATH=<tree>/opt`` without an install, as a driver may run the
package). The pin itself is gated at activation (:func:`gate`, the shared ``_core_gate`` stub: the imported core's version is at least the
pinned one). Every module of the package that imports the core imports this module first (``from . import _core``); it imports nothing of the
core itself, so a stock process (``stock_pred``) holds no core module beyond the proof machinery.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from typing import Optional

PYPROJECT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pyproject.toml")   # opt/pyproject.toml
PIN_TABLE = "[tool.opt_core]"
_PATH_LINE = re.compile(r'^\s*path\s*=\s*"([^"]*)"', re.M)


def pinned_path(pyproject: str = PYPROJECT) -> str:
    """The absolute directory ``[tool.opt_core] path`` names (relative to the pyproject)."""
    with open(pyproject, encoding="utf-8") as fh:
        text = fh.read()
    i = text.find(PIN_TABLE)
    if i < 0:
        raise ImportError(f"{pyproject}: no {PIN_TABLE} table (the kit's core pin)")
    table = re.split(r"^\[", text[i + len(PIN_TABLE):], maxsplit=1, flags=re.M)[0]
    m = _PATH_LINE.search(table)
    if not m:
        raise ImportError(f"{pyproject}: {PIN_TABLE} has no path")
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(pyproject)), m.group(1)))


def place() -> Optional[str]:
    """Append the pinned path to sys.path when ``opt_core`` is not importable; returns the entry added (None when nothing was)."""
    if importlib.util.find_spec("opt_core") is not None:
        return None
    try:
        p = pinned_path()
    except (ImportError, OSError):
        return None                                          # no readable pin: the gate (_core_gate) names it (core_pin_unreadable)
    if not os.path.isfile(os.path.join(p, "opt_core", "__init__.py")):
        return None                                          # nothing importable, no pinned copy: the gate names it (core_missing:opt_core)
    sys.path.append(p)
    return p


def gate():
    """The kit's pre-import core pin gate (``_core_gate.gate``, the shared template's bytes) after :func:`place`: statement one of every
    entry route before any module of the core is imported; refuses by name with exit 3 (core_missing / core_mismatch / core_pin_unreadable),
    returns the pinned / installed facts on a match."""
    from ._core_gate import gate as _gate
    return _gate(__file__, tag="protenix-opt")


def producer_refusal() -> Optional[str]:
    """After :func:`place`: the NOT ACTIVE reason when the importable core (installed, or the pinned copy) lacks a module this tree imports,
    else None (``_autoload.REQUIRED_PRODUCERS`` is the table; the absent core is :func:`place`'s ImportError = ``core_missing:opt_core``)."""
    from ._autoload import producer_refusal as _refusal, _core_version
    found = importlib.util.find_spec("opt_core")
    if found is None or not found.submodule_search_locations:
        return None                                          # the gate's refusal (it runs first on every route)
    root = list(found.submodule_search_locations)[0]
    return _refusal(root, _core_version(root))


PLACED = place()
