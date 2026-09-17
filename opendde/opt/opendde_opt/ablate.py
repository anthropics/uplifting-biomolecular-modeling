"""MODEL_OPT_LEVERS_OFF — ablation by name: levers composed OUT of the resolved line before anything is exported or installed.

One switch for every lever of every mode (one variable name across the model-optimization kits):
``MODEL_OPT_LEVERS_OFF=<lever>[,<lever>…]`` names registry levers to leave out of THIS call's line. The composition is the kit's own
(`modes.line_without`, the small-input floor's mechanism): the named levers' switches absent from the exports and in the must-unset pool,
their rows off the lever list (a house lever is then simply not installed), the FPF in-process enable off with ``fpf_trimul_exact``, the XL
unit's directory off the path with ``xl_tri_ln``. The rest of the line runs as shipped; the left-out levers' LEVER rows read
``state=off reason=ablated`` and the ACTIVE / PRED lines carry ``ablated=<lever,…>``. Unset or empty: nothing changes (the shipped lines,
byte for byte).

Refused by name (``modes.OpenModeError`` -> NOT ACTIVE, exit 3, nothing applied): a name that is not a registry lever; a lever the resolved
line's static row does not carry (ablating what is not there is a caller's mistake, named); a lever the line carries but cannot shed
individually (no rule in `modes.LEVER_SWITCHES`: the served-levers hook, the shipped tile cache, the DITFAST pair `dit_hoist,dit_align`,
the big memory levers, which the memory mode sizes by its own gates); ``off`` with any name. A lever a size gate already composed out
of this call (the small-input floor, the chunk lever's ceiling) is accepted and stays out. Read at resolution (`modes.resolve`), so `check`
refuses a bad value before any `pred`.
"""
from __future__ import annotations

import os
import re

ENV = "MODEL_OPT_LEVERS_OFF"
REASON = "ablated"                                   # the LEVER row's reason word and the ACTIVE / PRED lines' token name
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_ST = {"names": (), "dropped": (), "line": None}


def _reset() -> None:
    _ST.update(names=(), dropped=(), line=None)


def requested(environ=None) -> tuple:
    """The lever names the variable lists (order kept, duplicates dropped); () when unset / empty. A malformed token is refused by name."""
    from . import modes
    env = os.environ if environ is None else environ
    raw = (env.get(ENV) or "").strip()
    if not raw:
        return ()
    out = []
    for tok in raw.split(","):
        t = tok.strip()
        if not t:
            continue
        if not _NAME.match(t):
            raise modes.OpenModeError(f"{ENV}={raw!r}: {t!r} is not a lever name (comma-separated registry lever names)")
        if t not in out:
            out.append(t)
    return tuple(out)


def compose(line, environ=None):
    """``line`` without the levers ``MODEL_OPT_LEVERS_OFF`` names (modes.line_without); the line itself when the variable is unset. Raises
    modes.OpenModeError (a named refusal) for an unknown name, a lever the line's static row does not carry, a lever without a composition
    rule, or any name under ``off``."""
    from . import modes, registry
    names = requested(environ)
    _ST.update(names=names, dropped=(), line=(line.name if line is not None else None))
    if not names:
        return line
    if line is None:
        raise modes.OpenModeError(f"{ENV}={','.join(names)}: mode off runs stock and carries no lever to leave out")
    unknown = [n for n in names if n not in registry.LEVERS]
    if unknown:
        raise modes.OpenModeError(f"{ENV}: unknown lever(s) {','.join(unknown)} (registry levers: {','.join(sorted(registry.LEVERS))})")
    static = modes.LINES[line.name].levers if line.name in modes.LINES else ()
    carried = set(line.levers) | set(static)
    absent = [n for n in names if n not in carried]
    if absent:
        raise modes.OpenModeError(f"{ENV}: line {line.name} does not carry {','.join(absent)} (its levers: {','.join(line.levers)})")
    fixed = [n for n in names if n in line.levers and (n in modes.LEVER_NOT_SHEDDABLE or n not in modes.LEVER_SWITCHES)]
    if fixed:
        raise modes.OpenModeError(f"{ENV}: " + "; ".join(f"{n} cannot be left out of line {line.name} alone: {modes.LEVER_NOT_SHEDDABLE.get(n, 'no composition rule in modes.LEVER_SWITCHES')}" for n in fixed)
                                  + f" (removable on this line: {','.join(x for x in line.levers if x in modes.LEVER_SWITCHES and x not in modes.LEVER_NOT_SHEDDABLE) or 'none'})")
    drop = tuple(n for n in names if n in line.levers)
    _ST["dropped"] = tuple(n for n in names if n in carried)          # a lever a size gate already left out stays named as ablated too
    if not drop:
        return line
    try:
        out = modes.line_without(line, drop)
    except ValueError as e:                                            # line_without's own refusal (kept as the named refusal of this switch)
        raise modes.OpenModeError(f"{ENV}: {e}") from None
    gone = [x for x in line.levers if x not in out.levers and x not in _ST["dropped"]]   # the levers that went with the named ones (modes.LEVER_DEPENDENTS)
    _ST["dropped"] = tuple(_ST["dropped"]) + tuple(gone)
    return out


def dropped() -> tuple:
    """The levers this process's resolution left out by name (() when none)."""
    return tuple(_ST["dropped"])


def word():
    """``<lever>[,<lever>…]`` for the ACTIVE / PRED lines' ``ablated=`` token, None when nothing was ablated."""
    return ",".join(_ST["dropped"]) or None


def gated_off(line) -> dict:
    """``{lever: "ablated"}`` for the levers left out of ``line`` by name; {} otherwise."""
    if line is None or not _ST["dropped"] or _ST["line"] != line.name:
        return {}
    return {n: REASON for n in _ST["dropped"] if n not in line.levers}
