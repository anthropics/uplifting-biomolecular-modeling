"""The ablation switch: ``MODEL_OPT_LEVERS_OFF=<lever>[,<lever>...]`` runs a mode WITHOUT the named levers of its row — for attributing an
effect to single levers; such a run is never one of the kit's modes.

Semantics (the same on every route the package activates: ``run.sh pred|check``, ``atlasfold-opt``, the drop-in ``ATLASFOLD_OPT=<mode>`` route):

* names are registry lever ids (``registry.LEVERS``), comma-separated, whitespace ignored, duplicates folded; unset or blank = no ablation and
  the run is byte for byte the mode's (no ``ablated=`` token anywhere);
* every name must be a lever of the mode's row (``modes.MODES[mode]``); an unknown name, a lever outside the row, ``lever_report`` (the report
  itself, not a patch), a list that empties the row, and any name under ``--mode off`` are refused BY NAME
  (``NOT ACTIVE: reason=levers_off_refused: ...``, exit 3) — nothing runs under a name it does not have;
* applied before activation: the named levers leave the row ``stack.activate`` installs (they are never imported, never patched), the ACTIVE /
  DRY-RUN lines gain ``ablated=<names>`` (request order), and each ablated lever prints ``LEVER name=<strategy> state=off reason=levers_off``
  at exit beside the levers kept (``check`` prints ``PLAN lever=<x> state=off reason=levers_off``);
* the kit's per-lever words (``AFO_PAIR_TRANSITION_CHUNK=0``, ``AFO_DENOISER_GRAPH_MAX_TOKENS=0``,
  ``AFO_LM_SDPA_MIN_TOKENS``, ...) are unchanged and compose with it.
Standard library + the registry at import (``check`` and the kit's unit tests import it without torch; ``lever_lines`` reaches opt_core.report when called)."""
import os
from typing import Dict, List, Mapping, Optional, Tuple

ENV = "MODEL_OPT_LEVERS_OFF"
REASON = "levers_off"                                   # the LEVER / PLAN line's reason token for an ablated lever
TOKEN = "ablated"                                       # the ACTIVE / DRY-RUN lines' key (ablated=<names>)
NOT_REMOVABLE = ("lever_report",)                       # the report itself, not a patch: removing it would remove the LEVER lines, not a forward statement


class AblationError(ValueError):
    """A MODEL_OPT_LEVERS_OFF request the kit refuses by name (the activation's NOT ACTIVE reason)."""


def requested(environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """The lever names ``MODEL_OPT_LEVERS_OFF`` lists, in order, duplicates folded; [] when unset or blank."""
    raw = (os.environ if environ is None else environ).get(ENV) or ""
    out: List[str] = []
    for w in raw.replace(" ", "").split(","):
        if w and w not in out:
            out.append(w)
    return out


def validate(mode: str, names: List[str]) -> List[str]:
    """``names`` checked against ``mode``'s row: raises AblationError naming every unknown lever, every lever outside the row, every
    non-removable name and a request that leaves the row empty; returns ``names`` unchanged otherwise ([] passes through)."""
    from .modes import MODES
    from .registry import LEVERS
    if not names:
        return []
    if mode == "off":
        raise AblationError(f"{ENV}={','.join(names)}: mode off applies no lever, there is nothing to ablate (unset {ENV} for the stock route)")
    members = list(MODES.get(mode) or [])
    unknown = [n for n in names if n not in LEVERS]
    outside = [n for n in names if n in LEVERS and n not in members]
    pinned = [n for n in names if n in members and n in NOT_REMOVABLE]
    problems = []
    if unknown:
        problems.append(f"{','.join(unknown)}: not a lever of this kit (registry levers: {','.join(LEVERS)})")
    if outside:
        problems.append(f"{','.join(outside)}: not in mode {mode}'s row ({'+'.join(members)})")
    if pinned:
        problems.append(f"{','.join(pinned)}: the lever report itself, not removable by {ENV}")
    if not problems and not [m for m in members if m not in names]:
        problems.append(f"the request removes every lever of mode {mode}: that is --mode off, not an ablation")
    if problems:
        raise AblationError(f"{ENV} refused: " + "; ".join(problems))
    return list(names)


def dropped(mode: str, names: List[str]) -> List[Tuple[str, str]]:
    """The dependency closure of a validated request: every lever of ``mode``'s row whose registry ``requires`` names a lever that is
    switched off (requested, or itself dropped here — transitive, row order) steps aside BY NAME with it:
    ``[(lever, 'requires:<prerequisite>(levers_off)'), ...]``.  A lever OF another lever's served path (triattn_core of triatt_block;
    pair_block_residual of triatt_block + pair_transition) has nothing to engage on once its prerequisite is ablated — that is a named
    consequence of the request, never a partial activation."""
    from .modes import levers_of
    from .registry import LEVERS
    off = set(validate(mode, names))
    out: List[Tuple[str, str]] = []
    for lever in levers_of(mode):
        if lever in off:
            continue
        missing = [r for r in (LEVERS.get(lever, {}).get("requires") or ()) if r in off]
        if missing:
            out.append((lever, f"requires:{missing[0]}({REASON})"))
            off.add(lever)
    return out


def reasons(mode: str, names: List[str]) -> Dict[str, str]:
    """lever -> the reason it is off under this request: the requested names first, as given (``levers_off``), then their dependency
    closure in row order (``requires:<lever>(levers_off)``, ``dropped``).  {} when nothing is requested."""
    if not names:
        return {}
    out = {n: REASON for n in validate(mode, names)}          # the request first, as given (the ACTIVE / DRY-RUN token lists the requested names first)
    out.update(dropped(mode, names))                          # then its closure, row order
    return out


def row_without(mode: str, names: List[str]) -> List[str]:
    """``mode``'s row (modes.levers_of) less the validated ``names`` AND their dependency closure (``dropped``), order kept."""
    from .modes import levers_of
    off = set(reasons(mode, names))
    return [l for l in levers_of(mode) if l not in off]


def token(names: List[str]) -> str:
    """``ablated=<a,b>`` value for the ACTIVE / DRY-RUN lines' pairs; '' when nothing is ablated (the caller adds no key then)."""
    return ",".join(names)


def lever_lines(tag: str, names: List[str], why: Optional[Mapping[str, str]] = None) -> List[str]:
    """One ``LEVER name=<strategy> state=off reason=<levers_off | requires:<lever>(levers_off)> impl=<module|kit> origin=<core|kit>`` line per lever
    that is off under the request (opt_core.report grammar); ``why`` maps lever -> reason (``reasons()``), levers_off when absent."""
    from opt_core import report as R
    from .registry import LEVERS
    out = []
    for n in names:
        d = LEVERS.get(n, {})
        impl = d.get("module") or d.get("provider") or f"atlasfold_opt:{n}"
        origin = "core" if str(d.get("strategy", "")).startswith(("F1.", "F2.")) or "opt_core" in str(d.get("provider", "")) else "kit"
        out.append(R.lever_line(tag, d.get("strategy", f"LOCAL.atlasfold.{n}"), "off", reason=(why or {}).get(n, REASON), impl=str(impl).replace(" ", "_"), origin=origin, lever=n))
    return out
