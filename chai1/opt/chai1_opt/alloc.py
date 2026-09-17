"""The package's opt-in allocator lever ``alloc`` (modes.OPTIN_LEVERS; strategy F7.expandable_segments): torch's CUDA caching allocator with
expandable segments (``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``), exported through the shared core's ``opt_core.mem.torch_alloc``
— its named refusals (the variable already names another configuration; CUDA already initialised in this process) and its read-back of
the allocator's own record.

Placement only: no kernel input changes, so outputs are unchanged by construction. Every kit row carries it (``modes.KitMode.implied_optin``:
exact, fast, big; never ``off``, which is stock with no environment set): the memory gate of exact / fast engages at crop 2048 and, in a
process that folds several such items, the caching allocator's fixed segments fragment (reserved-but-unallocated blocks a large request
cannot use); expandable segments are read at CUDA init, before any crop is known, so the policy rides the row, not the gate. The fast line
keeps its CUDA graph of the hoisted denoiser step (one graph per item, dropped at the next item's trunk boundary); the core's graph-pool
hazard refusal is answered with ``allow_with_graphs`` because this composition is the intended one — the policy governs the eager
allocations around the one live graph pool.

Routes. Driver route: the ``pred`` process writes the core's environment row into the driver child's environment (``env_row``) and the
driver, which imports no torch before the kit does, confirms it at start (``export``: present); in-process route: ``export`` into this
process before the eager stack installs, refused by name when CUDA is already initialised. Evidence: ``alloc_conf=`` on the activation
report, ``alloc_effective=true|false|pending alloc_source=<where the core read it>`` (the core's ``facts``: the allocator's own record) on the exit tally.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

NAME = "alloc"
POLICY = "expandable"
CORE_MODULE = "opt_core.mem.torch_alloc"


class LeverUnavailable(RuntimeError):
    """The lever cannot be applied; ``str()`` is the refusal wording."""


def core():
    """``opt_core.mem.torch_alloc`` (lazy); LeverUnavailable naming the core version when the installed core has none."""
    from ._core import ensure_importable
    ensure_importable()
    try:
        import importlib
        return importlib.import_module(CORE_MODULE)
    except ImportError as e:
        import opt_core
        raise LeverUnavailable(f"opt-in lever {NAME} needs {CORE_MODULE}, absent from the installed core (opt_core "
                               f"{getattr(opt_core, '__version__', '?')}): {e}") from None


def env_row() -> Dict[str, str]:
    """The environment row for a child process (``{PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True}``)."""
    return core().env_row(POLICY)


def export(environ=None) -> Dict[str, object]:
    """Export the policy into ``environ`` (default this process) through the core; returns its facts (``alloc``, ``alloc_conf``,
    ``alloc_export`` = exported | present). LeverUnavailable carrying the core's refusal wording otherwise."""
    ta = core()
    from opt_core.oom import is_oom
    try:
        return ta.export(POLICY, environ, lever=NAME, graphs_on=True, allow_with_graphs=True)
    except Exception as e:  # noqa: BLE001 — the core's MemLeverRefused (or any failure) becomes this lever's refusal by name
        if is_oom(e): raise                                       # noqa: E701 — except an out-of-memory error, which propagates
        raise LeverUnavailable(f"opt-in lever {NAME}: {e}") from None


def applied(environ=None) -> bool:
    """True iff the environment carries exactly the policy's configuration (what torch reads at its first CUDA allocation)."""
    env = os.environ if environ is None else environ
    try:
        want = core().conf_for(POLICY)
    except LeverUnavailable:
        return False
    return env.get("PYTORCH_CUDA_ALLOC_CONF") == want


def tally_fields() -> list:
    """``alloc_effective=true|false|pending alloc_source=<source>`` — the core's own ``facts`` (its read of the allocator's record in this
    process), nothing re-derived here; [] when the variable is not set; ``alloc_effective=unreadable alloc_source=core_missing`` when the
    installed core has no allocator module."""
    if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        return []
    try:
        f = core().facts(POLICY)
    except LeverUnavailable:
        return ["alloc_effective=unreadable", "alloc_source=core_missing"]
    return [f"alloc_effective={f['alloc_effective']}", f"alloc_source={f['alloc_source']}"]
