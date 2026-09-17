"""The kit's kernel levers — one module per kernel lever under this package (each registered in registry.LEVERS with its
numerics class), and the ONE exit printer they share.

A kernel lever's evidence is a census the process can only close at exit (calls served / fallen back by reason / shapes: an
`opt_core.counters.Ledger`), printed as its LEVER line by `opt_core.report.register_exit_tally` — which prints one registered text per tag.
`register_exit_line` multiplexes that slot: each kernel lever registers its own line producer at install, and the one printer registered
under the package's tag writes every registered lever's line, registry order, one line each. With one kernel lever installed the output is
that lever's line alone, byte for byte.
"""
from __future__ import annotations

from typing import Callable, Dict, List

from opt_core import report as _core_report

from .. import registry
from ..names import TAG

_PRINTERS: Dict[str, Callable[[], str]] = {}


def register_exit_line(lever: str, line_of: Callable[[], str]) -> bool:
    """Print `line_of()` (the lever's whole LEVER line) at interpreter exit through the package's one exit printer. Idempotent per lever;
    returns True when the lever was registered now."""
    if lever not in registry.LEVERS:
        raise ValueError(f"lever {lever!r} is not registered (registry.LEVERS)")
    new = lever not in _PRINTERS
    _PRINTERS[lever] = line_of
    _core_report.register_exit_tally(TAG, exit_lines)                        # once per process per tag (the core keeps the first registration)
    return new


def exit_lines() -> str:
    """Every registered kernel lever's exit line, registry order, newline-joined (the exit printer appends the last newline)."""
    return "\n".join(_PRINTERS[l]() for l in registry.ORDER if l in _PRINTERS)


def registered() -> List[str]:
    return [l for l in registry.ORDER if l in _PRINTERS]


def reset_for_tests() -> None:
    _PRINTERS.clear()
