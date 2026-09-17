"""The kit's first statement on every documented entry: the opt_core pin gate (``_core_gate``, a byte-identical copy of opt_core's
kit_template/_core_gate.py); only after it passes is opt_core importable — ``gate()`` returns its facts, ``facts()`` repeats them."""
import os
import sys

from . import _core_gate

TAG = "atlasfold-opt"
_FACTS = None


def pyproject_path() -> str:
    return _core_gate.find_pyproject(__file__)


def gate(stream=None) -> dict:
    """Run the pin gate once per process (exit 3 with the NOT ACTIVE line on refusal); afterwards put the pinned core's root on sys.path
    when it is not importable already (a checkout layout: <release>/common/opt_core beside <release>/atlasfold)."""
    global _FACTS
    if _FACTS is None:
        _FACTS = _core_gate.gate(__file__, tag=TAG, stream=stream)
        root = _FACTS.get("installed", {}).get("root")
        if root and root not in sys.path:
            sys.path.insert(0, root)
    return _FACTS


def facts() -> dict:
    return dict(_FACTS or {})
