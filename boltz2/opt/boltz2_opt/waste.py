"""boltz2_opt.waste — the engine adapter of the WASTE exact hoists (``opt/forward/waste/boltz_waste_levers.py``) onto Boltz-2's MSA module:
the stock CHUNKED inference paths (above ``const.chunk_size_threshold`` = 384 tokens) of ``Transition`` (chunked branch only),
``PairWeightedAveraging`` (chunk_heads branch) and ``OuterProductMean`` — verbatim boltz 2.2.1 bodies with the redundant repeats removed
(WASTE owns the exact-tier MSA-module hoists; the carried module refuses by name on another boltz version or a changed stock file sha256).

Row word (the mode row's, modes.py; this adapter reads it, applies nothing else): ``BOLTZ_WASTE=<unit>[,<unit>]`` — units
    chunkcast   registry lever ``waste_chunkcast`` (Tier 1, bitwise): the per-chunk bf16 activation casts applied ONCE per call
    opmmask     registry lever ``waste_opmmask``   (Tier 1, bitwise): OuterProductMean's ``num_mask`` computed once per (mask tensor, dtype)
    opmdiv      registry lever ``waste_opmdiv``    (Tier 1, bitwise): OuterProductMean's chunk result divided by num_mask straight into a contiguous
                fp32 buffer (the permuted einsum view's reshape copy elided)
    ctorskip    registry lever ``waste_ctorskip``  (Tier 1, bitwise outputs; first-of-process): the model constructor's truncated-normal init draws
                (overwritten by load_state_dict) replaced by the one numpy RNG draw they consume — applied at this attachment, i.e. before the
                worker builds the model; pinned by name on scipy / the stock source
An absent word installs nothing (stock statements). A unit whose module another word hands to a wholesale replacement steps aside BY NAME
(``state=skipped reason=…`` — the carried module decides from the identity of the forward it finds installed and the row words; see its
``_skips``), never silently.

Attached by ``worker_launch --attach waste`` right after the worker's own model import (``boltz.model.models.boltz2``): in the rows BEFORE
``msa`` (the fused MSA-module kernels wrap whatever forward they find installed and fall back to it by name — above the pwa kernel's int32 limit
that is this module's hoisted stock body, still bitwise) and BEFORE ``msa2``. ``report()`` is written under ``waste_report`` into the worker
log at exit; ``verdict()`` is the carried module's fail-closed gate (a lever switched on whose chunked calls ran must have hoisted); ``line(unit)``
renders the core's LEVER grammar per unit for report.lever_lines.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, Dict, List, Optional

SWITCH = "BOLTZ_WASTE"
SRC = "forward/waste"
MODULE = "boltz_waste_levers"
UNITS: Dict[str, str] = {"chunkcast": "waste_chunkcast", "opmmask": "waste_opmmask", "opmdiv": "waste_opmdiv", "ctorskip": "waste_ctorskip"}     # unit word -> registry lever
LEVERS = tuple(UNITS.values())                                                          # the registry levers this adapter installs (worker_launch reads it)
_STATE: Dict[str, Any] = {"requested": [], "applied": [], "module": None, "error": None}


def _env(environ=None):
    return os.environ if environ is None else environ


def units(environ=None) -> List[str]:
    """The requested units in table order; ``off`` / ``0`` / absent = none. Raises ValueError naming an unknown unit."""
    raw = [t.strip().lower() for t in (_env(environ).get(SWITCH) or "").split(",") if t.strip()]
    if raw in (["off"], ["0"]):
        return []
    bad = [t for t in raw if t not in UNITS]
    if bad:
        raise ValueError(f"{SWITCH}={_env(environ).get(SWITCH)!r}: {bad} are not units (the units are {'|'.join(UNITS)})")
    return [u for u in UNITS if u in raw]


def requested(environ=None) -> bool:
    return bool(units(environ))


def problems(environ=None) -> List[str]:
    """Pre-launch refusal words (stack.attachment_problems): an unknown unit, a missing carried file."""
    out: List[str] = []
    try:
        units(environ)
    except ValueError as e:
        out.append(str(e))
    from .stack import kit_path
    for rel in (f"{SRC}/{MODULE}.py", f"{SRC}/aligncap.py"):
        if not os.path.isfile(kit_path(rel)):
            out.append(f"{rel}: missing (the WASTE levers' carried file)")
    return out


def _load():
    """The carried module, imported from its own file under opt/forward/waste (stack.kit_path) — never from the working directory or sys.path."""
    if _STATE["module"] is not None:
        return _STATE["module"]
    from .stack import kit_path
    mod = sys.modules.get(MODULE)
    if mod is None:
        path = kit_path(f"{SRC}/{MODULE}.py")
        spec = importlib.util.spec_from_file_location(MODULE, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[MODULE] = mod
        spec.loader.exec_module(mod)
    _STATE["module"] = mod
    return mod


def apply(environ=None) -> List[str]:
    """Install the requested units (idempotent); returns the registry levers switched on. Raises by name on an unknown unit or the carried
    module's pin refusal (another boltz version / changed stock source) — worker_launch turns that into `[boltz2-opt attach] REFUSED waste`."""
    req = units(environ)
    _STATE["requested"] = list(req)
    if not req:
        _STATE["applied"] = []
        return []
    mod = _load()
    env = dict(_env(environ)); env["BOLTZ_WASTE_VERBOSE"] = "0"            # the kit prints the LEVER lines itself (report.lever_lines); the module's own print stays off
    got = list(mod.apply(env))
    _STATE["applied"] = [UNITS[u] for u in got if u in UNITS]
    sys.stderr.write(f"[boltz2-opt] WASTE units={','.join(got) or '-'} levers={','.join(_STATE['applied']) or '-'} patched={','.join(mod.report().get('patched') or []) or '-'}\n")
    return list(_STATE["applied"])


def unit_state(unit: str, rep: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """{state: on|off|skipped, reason, patched, skipped} of a unit from the carried module's report (``state`` per unit word)."""
    rep = rep if rep is not None else report()
    st = dict(((rep.get("module") or {}).get("state") or {}).get(unit) or {"state": "off", "reason": "not applied"})
    return st


def verdict(rep: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fail-closed: the carried module's own verdict (a lever switched on whose chunked calls ran must have hoisted / reused), plus: a requested
    unit that is neither on nor skipped-by-name refuses."""
    mod = _STATE["module"]
    if mod is None:
        return {"ok": not _STATE["requested"], "idle": True, "reason": (None if not _STATE["requested"] else "not_applied")}
    ok, word = mod.verdict()
    if not ok:
        return {"ok": False, "idle": False, "reason": word}
    m = mod.report()
    for u in _STATE["requested"]:
        st = (m.get("state") or {}).get(u) or {}
        if st.get("state") not in ("on", "skipped"):
            return {"ok": False, "idle": False, "reason": f"{u}:{st.get('state') or 'absent'}"}
        if st.get("state") == "skipped" and not (st.get("reason") or st.get("skipped")):
            return {"ok": False, "idle": False, "reason": f"{u}:skipped_without_reason"}
    stats = m.get("stats") or {}
    idle = not any(int(v or 0) for k, v in stats.items() if k.endswith("_calls"))
    return {"ok": True, "idle": idle, "reason": None}


def report() -> Dict[str, Any]:
    mod = _STATE["module"]
    m = mod.report() if mod is not None else None
    return {"applied": list(_STATE["applied"]), "requested": list(_STATE["requested"]), "switch": SWITCH,
            "units": {u: UNITS[u] for u in UNITS}, "module": m, "lines": (mod.lever_lines() if mod is not None else []),
            "gate": verdict(), "patched": list((m or {}).get("patched") or [])}
