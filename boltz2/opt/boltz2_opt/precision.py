"""boltz2_opt.precision — the engine adapter of the fast tier's precision units (``opt/forward/precision/boltz2_precision.py``) onto
Boltz-2: the sites where stock runs IEEE fp32 (the diffusion score model's fp32 island, the Pairformer sequence attention) demoted to bf16
tensor-core operands / TF32 products, ONE UNIT PER SITE, each its own registry lever
(Tier 2: declared reduced precision where upstream runs fp32; never fewer statements, steps, recycles or rows).

Switch ``BOLTZ_PRECISION=<token>[,<token>...]`` — the row's ONE word for every unit (the kit's mode rows assign the word, modes.py):
    <unit>        install the unit (``dit_bf16`` ``dit_tf32`` ``dit_attn_bf16`` ``seq_bf16`` — UNITS; the bundle
                  module's docstring says what each demotes and what it pins to fp32)
    off:<unit>    the row's ablation entry: the unit is NOT installed — its stock statement runs untouched — and its LEVER line
                  reads ``state=off reason=ablation``; the ACTIVE line's ``off=`` census carries it (stack.evidence). An ablated run can never be
                  mistaken for the row as composed.
An unknown token or a unit named both on and off is refused BY NAME at apply (the worker exits 3,
``[boltz2-opt attach] REFUSED precision: …``). Attached by ``boltz2_opt.worker_launch --attach precision`` right after the worker's model import
(``boltz.model.models.boltz2``: every patched class is imported by then). The units install on ``__call__`` / a module-level function, never
on a ``forward`` the kit's other levers replace (the DiT hoist, the graph sampler, the trunk levers), so the attach order among them is free.

Report (``report()``, written as ``precision_report`` into the worker log by the attach hook): ``applied`` (registry names), ``variant`` (the
row word verbatim), per unit ``state`` / ``census`` (calls served, fp32 pins served, dtype facts) / ``gate``, ``off`` (the ablation entries),
and ONE LEVER line per unit named in the row (on or off) in the core's grammar (``lines``). Gate (fail-closed, ``verdict``): every installed
unit must have served at least one call of its site over the run (every predict run calls the score model and the Pairformer sequence
attention), else the gate refuses naming the unit.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

TAG = "boltz2-opt"
SWITCH = "BOLTZ_PRECISION"
BUNDLE_FILE = "forward/precision/boltz2_precision.py"                 # opt/-relative (stack.kit_path resolves it); imported from its own path, nothing enters sys.path
MODULE_NAME = "boltz2_precision"
UNITS: Dict[str, dict] = {   # unit token -> registry lever, kit-local strategy id, impl word (file:function of the bundle)
    "dit_bf16": {"lever": "dit_bf16", "strategy": "LOCAL.boltz2.precision_dit_bf16", "impl": f"{BUNDLE_FILE}:_dm_call"},
    "dit_tf32": {"lever": "dit_tf32", "strategy": "LOCAL.boltz2.precision_dit_tf32", "impl": f"{BUNDLE_FILE}:_dm_call"},
    "dit_attn_bf16": {"lever": "dit_attn_bf16", "strategy": "LOCAL.boltz2.precision_dit_attn_bf16", "impl": f"{BUNDLE_FILE}:_apb_token_bf16"},
    "seq_bf16": {"lever": "seq_bf16", "strategy": "LOCAL.boltz2.precision_seq_bf16", "impl": f"{BUNDLE_FILE}:_apb_call"},
}
LEVERS = tuple(u["lever"] for u in UNITS.values())                    # the registry levers this adapter can install (worker_launch reads it)
SERVED_KEY = {"dit_bf16": "calls", "dit_tf32": "calls", "dit_attn_bf16": "calls", "seq_bf16": "attn_calls"}
_STATE: Dict[str, Any] = {"mod": None, "on": [], "off": [], "applied": [], "variant": None, "error": None}


def _env(environ):
    return os.environ if environ is None else environ


def parse(environ=None) -> Tuple[List[str], List[str]]:
    """(units on, units off) from the row word, in order, each once; ``ValueError`` naming a bad token."""
    raw = [t.strip().lower() for t in (_env(environ).get(SWITCH) or "").split(",") if t.strip()]
    on: List[str] = []; off: List[str] = []
    for t in raw:
        neg = t.startswith("off:")
        u = t[4:] if neg else t
        if u not in UNITS:
            raise ValueError(f"{SWITCH}={_env(environ).get(SWITCH)!r}: '{t}' is not a unit (the units are {'|'.join(UNITS)}; ablation = off:<unit>)")
        bucket = off if neg else on
        if u not in bucket:
            bucket.append(u)
    both = [u for u in on if u in off]
    if both:
        raise ValueError(f"{SWITCH}={_env(environ).get(SWITCH)!r}: {both} named both on and off")
    return on, off


def requested(environ=None) -> bool:
    on, off = parse(environ)
    return bool(on or off)


def levers_of(environ=None) -> List[str]:
    """The registry levers the row's word installs (the `on` units)."""
    return [UNITS[u]["lever"] for u in parse(environ)[0]]


def off_levers(environ=None) -> Dict[str, str]:
    """The row's ablation entries: {lever: 'ablation'} (stack.evidence / the ACTIVE line's off= census read it)."""
    return {UNITS[u]["lever"]: "ablation" for u in parse(environ)[1]}


def _load_bundle():
    if _STATE["mod"] is not None:
        return _STATE["mod"]
    from . import stack
    path = stack.kit_path(BUNDLE_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{BUNDLE_FILE} is not in this kit tree ({path})")
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    _STATE["mod"] = mod
    return mod


def apply(spec: Optional[str] = None) -> List[str]:
    """Install the row's units (idempotent). Returns the registry levers installed ([] when the row names none or only ablations)."""
    if _STATE["applied"]:
        return list(_STATE["applied"])
    env = None if spec is None else {SWITCH: spec}
    on, off = parse(env)
    _STATE["variant"] = _env(env).get(SWITCH, "")
    _STATE["on"], _STATE["off"] = list(on), list(off)
    if not on:
        for u in off:
            sys.stderr.write(line(u) + "\n")
        return []
    mod = _load_bundle()
    mod.enable(on)
    _STATE["applied"] = [UNITS[u]["lever"] for u in on]
    try:
        from opt_core import report as _report
        _report.register_exit_tally(TAG + "/precision", exit_lines)
    except Exception:  # noqa: BLE001
        pass
    for u in on + off:
        sys.stderr.write(line(u) + "\n")
    return list(_STATE["applied"])


def dispositions() -> Dict[str, str]:
    """Levers the row names as disposed of rather than installed: the ablation entries (worker_launch accepts an attach that installed
    nothing when every named unit is an `off:` entry)."""
    return {UNITS[u]["lever"]: "ablation" for u in _STATE["off"]}


def _census(u: str) -> Dict[str, Any]:
    mod = _STATE["mod"]
    if mod is None:
        return {}
    return dict((mod.report().get("census") or {}).get(u) or {})


def line(u: str) -> str:
    """The ONE LEVER line of unit `u` in the core's grammar."""
    from opt_core.report import lever_line
    row = UNITS[u]
    if u in _STATE["off"]:
        return lever_line(TAG, row["lever"], "off", reason="ablation", impl=row["impl"], origin="kit", strategy=row["strategy"], switch=f"{SWITCH}=off:{u}")
    c = _census(u)
    served = int(c.get(SERVED_KEY[u], 0))
    fields = {"served": served}
    for k in sorted(c):
        if k in (SERVED_KEY[u], "facts"):
            continue
        fields[k] = c[k]
    for k, v in sorted((c.get("facts") or {}).items()):
        fields[k] = str(v).replace(" ", "")
    return lever_line(TAG, row["lever"], "on", impl=row["impl"], origin="kit", strategy=row["strategy"], **fields)


def exit_lines() -> str:
    return " | ".join(line(u) for u in _STATE["on"] + _STATE["off"])


def verdict() -> Dict[str, Any]:
    """Fail-closed: refused when the bundle raised at install or an installed unit served no call of its site."""
    if _STATE["error"]:
        return {"ok": False, "idle": False, "reason": _STATE["error"]}
    if not _STATE["on"] and not _STATE["off"]:
        return {"ok": False, "idle": False, "reason": "not applied"}
    starving = [u for u in _STATE["on"] if int(_census(u).get(SERVED_KEY[u], 0)) == 0]
    if starving:
        return {"ok": False, "idle": False, "reason": f"installed unit(s) served no call: {','.join(starving)}"}
    rep = _STATE["mod"].report() if _STATE["mod"] is not None else {}
    if rep.get("errors"):
        return {"ok": False, "idle": False, "reason": f"bundle errors: {','.join(map(str, rep['errors']))[:200]}"}
    if rep.get("tf32_flag_now") is not False or rep.get("matmul_precision") != "highest":   # the process policy at report time is stock's: no unit leaves TF32 on
        return {"ok": False, "idle": False, "reason": f"process matmul policy not stock's at exit (allow_tf32={rep.get('tf32_flag_now')}, precision={rep.get('matmul_precision')})"}
    if "dit_tf32" in _STATE["on"] and int(_census("dit_tf32").get("flag_restored", 0)) != int(_census("dit_tf32").get("calls", 0)):
        return {"ok": False, "idle": False, "reason": "dit_tf32: flag_restored != calls"}
    return {"ok": True, "idle": False, "idle_units": [], "reason": None}


def report() -> Dict[str, Any]:
    mod = _STATE["mod"]
    units = {}
    for u in _STATE["on"]:
        units[u] = {"state": "on", "lever": UNITS[u]["lever"], "census": _census(u)}
    for u in _STATE["off"]:
        units[u] = {"state": "off", "reason": "ablation", "lever": UNITS[u]["lever"]}
    try:
        import json as _json
        print("[boltz2-opt] PRECISION_STATE " + _json.dumps((_STATE["mod"].process_state() if _STATE.get("mod") and hasattr(_STATE["mod"], "process_state") else {})), flush=True)   # the process-wide numerics/dispatch state at report time (what a unit could leak)
    except Exception as _e:
        print(f"[boltz2-opt] PRECISION_STATE unavailable: {_e!r}", flush=True)
    return {"applied": list(_STATE["applied"]), "disabled": dispositions(), "variant": _STATE["variant"], "units": units,
            "lines": [line(u) for u in _STATE["on"] + _STATE["off"]], "line": exit_lines() if (_STATE["on"] or _STATE["off"]) else None,
            "gate": verdict(), "patched": ["DiffusionModule.__call__", "AttentionPairBias.__call__", "FourierEmbedding.__call__", "SingleConditioning.__call__", "DiffusionTransformerLayer.__call__"] if _STATE["on"] else [],
            "bundle": (mod.report() if mod is not None else None)}
