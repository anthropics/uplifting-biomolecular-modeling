"""boltz2_opt.atom — the engine adapter of the atom-attention levers this tree carries (``opt/forward/atom/src/boltz_atom.py`` +
``boltz_atom_kernels.py``) onto Boltz-2's diffusion module: the AtomAttentionEncoder / AtomAttentionDecoder of ``structure_module.score_model``
(3 + 3 windowed atom-transformer layers run inside every denoiser step) and the atom<->token glue around them.

Row words (the mode rows'; this adapter reads them, applies nothing else):
    BOLTZ_ATOM=<unit>[,<unit>]      units keys | glue | fused  (registry levers atom_keys_gather | atom_glue_hoist | atom_fused)
    BOLTZ_ATOM_GEMM=ieee|tf32|tf32x3|bf16   the fused kernels' dot operand precision (registry lever atom_gemm; absent = ieee; fast tier only)
    BOLTZ_ATOM_RELEASE=graph|sample  the static buffers' lifetime: the sampler group's token-threshold rule (absent) | released at every sample() return
Exact row: ``keys,glue`` (bitwise class: one-hot GEMM -> gather, step-invariant hoists).  Fast row: ``keys,fused`` + ``BOLTZ_ATOM_GEMM=bf16``.

Attached by ``worker_launch --attach atom`` after the worker's own model-module import (``boltz.model.models.boltz2``) and BEFORE the model is
built: ``apply()`` arms ``boltz_atom`` (it hooks ``AtomDiffusion.__init__`` so the structure module the checkpoint loader builds is installed at
instance level — the class-level sampler levers boltz_graph_patch / boltz_dit_hoist compose around it unchanged). ``report()`` is written under
``atom_report`` into the worker log at exit; ``line(unit)`` renders the core's LEVER grammar per unit; ``verdict()`` is the fail-closed gate:
refused on a kernel error, on a fallback word outside EXPECTED, or when a requested unit served no call over a pass that ran items.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, Dict, List, Optional

TAG = "BZ2ATOM"
SWITCH = "BOLTZ_ATOM"
GEMM_SWITCH = "BOLTZ_ATOM_GEMM"
RELEASE_SWITCH = "BOLTZ_ATOM_RELEASE"          # graph (absent) | sample: the fused kernels' static buffers released at every sample() return (boltz_atom.set_release) — the memory row's placement word
SRC = "forward/atom/src"
PACKAGE_FILES = (f"{SRC}/boltz_atom.py", f"{SRC}/boltz_atom_kernels.py")
GEMM_WORDS = ("ieee", "tf32", "tf32x3", "bf16")
UNITS: Dict[str, Dict[str, Any]] = {
    "keys": {"lever": "atom_keys_gather", "strategy": "LOCAL.boltz2.onehot_gather", "impl": "boltz_atom:GatherToKeys", "tier": 1,
             "calls": ("keys",), "what": "single_to_keys one-hot einsum -> gather kernel (exact)"},
    "glue": {"lever": "atom_glue_hoist", "strategy": "LOCAL.step_invariant_hoist", "impl": "boltz_atom:_encoder_glue+_decoder_glue", "tier": 1,
             "calls": ("enc_glue", "dec_glue"), "what": "atom_to_token casts / mean hoisted per sample(), decoder one-hot bmm -> row gather (exact)"},
    "fused": {"lever": "atom_fused", "strategy": "LOCAL.dit_fused_kernels", "impl": "boltz_atom_kernels@1", "tier": 2,
              "calls": ("enc_fused", "dec_fused"), "what": "fused atom encoder/decoder Triton kernels (fast)"},
}
GEMM_LEVER = {"lever": "atom_gemm", "strategy": {"bf16": "F4.bf16_weight_precast", "tf32": "F4.tf32_matmul", "tf32x3": "F4.tf32x3_compensated", "ieee": None}}
LEVERS = tuple(u["lever"] for u in UNITS.values()) + (GEMM_LEVER["lever"],)   # the registry levers this adapter installs (worker_launch reads it)
EXPECTED: tuple = ()                      # fallback words a healthy run may show: none (keys_shape / keys_window_* / a2t_not_onehot never occur on Boltz-2 inputs)
_STATE: Dict[str, Any] = {"units": (), "gemm": "ieee", "applied": [], "module": None, "error": None}


def _env(environ=None):
    return os.environ if environ is None else environ


def units(environ=None) -> List[str]:
    raw = [t.strip().lower() for t in (_env(environ).get(SWITCH) or "").split(",") if t.strip()]
    if raw in (["off"], ["0"]):
        return []
    bad = [t for t in raw if t not in UNITS]
    if bad:
        raise ValueError(f"{SWITCH}={_env(environ).get(SWITCH)!r}: {bad} are not units (the units are {'|'.join(UNITS)})")
    return [u for u in UNITS if u in raw]


def gemm_word(environ=None) -> str:
    w = (_env(environ).get(GEMM_SWITCH) or "ieee").strip().lower()
    if w not in GEMM_WORDS:
        raise ValueError(f"{GEMM_SWITCH}={w!r}: not one of {'|'.join(GEMM_WORDS)}")
    return w


def carried_problems() -> List[str]:
    from . import stack
    problems, _ = stack.check_kit_files(list(PACKAGE_FILES))
    return list(problems)


def problems(environ=None) -> List[str]:
    """Pre-launch words (torch-free): unknown unit / precision word, carried files missing, a precision word without the fused unit."""
    out: List[str] = []
    try:
        us = units(environ)
    except ValueError as e:
        return [str(e)]
    if not us:
        return out
    try:
        g = gemm_word(environ)
    except ValueError as e:
        out.append(str(e)); g = "ieee"
    if g != "ieee" and "fused" not in us:
        out.append(f"{GEMM_SWITCH}={g} names the fused kernels' dot precision but {SWITCH} does not request the fused unit ({','.join(us)})")
    out += carried_problems()
    return out


def _load():
    """Import boltz_atom_kernels then boltz_atom from the kit's carried files (registered in sys.modules under their own names)."""
    if _STATE["module"] is not None:
        return _STATE["module"]
    from .stack import kit_path
    for name in ("boltz_atom_kernels", "boltz_atom"):
        if name in sys.modules:
            continue
        path = kit_path(f"{SRC}/{name}.py")
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    _STATE["module"] = sys.modules["boltz_atom"]
    return _STATE["module"]


def apply(environ=None) -> List[str]:
    """Arm the requested units (refuses by name on a problem word); returns the lever names applied."""
    env = _env(environ)
    us = units(env)
    if not us:
        raise RuntimeError(f"boltz2_opt.atom: {SWITCH} is absent from the row — nothing to apply")
    words = problems(env)
    if words:
        raise RuntimeError("boltz2_opt.atom refused: " + "; ".join(words))
    g = gemm_word(env)
    BA = _load()
    rel = (env.get(RELEASE_SWITCH) or "").strip().lower() or "graph"
    BA.set_release(rel)                                                   # raises by name on a bad word
    BA.apply(",".join(us), gemm=g)
    _STATE["units"] = tuple(us); _STATE["gemm"] = g; _STATE["release"] = rel
    _STATE["applied"] = [UNITS[u]["lever"] for u in us] + ([GEMM_LEVER["lever"]] if g != "ieee" else [])
    for u in UNITS:
        print(line(u), flush=True)
    print(gemm_line(), flush=True)
    return list(_STATE["applied"])


def _census(unit: str, rep: Dict[str, Any]) -> Dict[str, Any]:
    calls = rep.get("calls") or {}
    served = int(sum(int(calls.get(k, 0)) for k in UNITS[unit]["calls"]))
    fb = {k: v for k, v in (rep.get("fallback") or {}).items()}
    return {"served": served, "fallback_by": fb, "errors": dict(rep.get("errors") or {}), "installed": int(rep.get("installed") or 0)}


def verdict(rep: Optional[Dict[str, Any]] = None, ran_items: bool = True) -> Dict[str, Any]:
    """Fail-closed: an error, a fallback word outside EXPECTED, no instance installed, or a requested unit that served nothing over items = refused."""
    BA = _STATE["module"]
    if rep is None:
        rep = BA.report() if BA is not None else {}
    reasons = []
    if not _STATE["units"]:
        return {"ok": True, "idle": True, "reason": "not_requested"}
    if int(rep.get("installed") or 0) < 1:
        reasons.append("no AtomDiffusion instance installed (the model was built before apply() or not at all)")
    if rep.get("errors"):
        reasons.append(f"errors {rep.get('errors')}")
    unexpected = {k: v for k, v in (rep.get("fallback") or {}).items() if k not in EXPECTED and v}
    if unexpected:
        reasons.append(f"fallback words outside EXPECTED: {unexpected}")
    if ran_items:
        for u in _STATE["units"]:
            if _census(u, rep)["served"] == 0:
                reasons.append(f"unit {u} ({UNITS[u]['lever']}) served no call")
    return {"ok": not reasons, "idle": False, "reason": "; ".join(reasons) or None}


def line(unit: str, state: Optional[str] = None, reason: Optional[str] = None) -> str:
    from opt_core.report import lever_line
    U = UNITS[unit]
    BA = _STATE["module"]
    rep = BA.report() if BA is not None else {}
    if unit not in _STATE["units"]:
        return lever_line(TAG, U["lever"], state or "off", ("impl", U["impl"]), ("origin", "kit"), reason=reason or "not_requested", strategy=U["strategy"])
    c = _census(unit, rep)
    fields = [("impl", U["impl"]), ("origin", "kit"), ("tier", U["tier"]), ("served", c["served"]), ("fallback", int(sum(c["fallback_by"].values()))),
              ("errors", int(sum(c["errors"].values())) if c["errors"] else 0), ("installed", c["installed"])]
    if unit == "fused":
        fields += [("gemm", _STATE["gemm"]), ("release", _STATE.get("release", "graph")), ("releases", rep.get("releases", 0))]
    return lever_line(TAG, U["lever"], state or "on", *fields, reason=reason, strategy=U["strategy"])


def gemm_line(state: Optional[str] = None, reason: Optional[str] = None) -> str:
    from opt_core.report import lever_line
    g = _STATE["gemm"]
    if "fused" not in _STATE["units"] or g == "ieee":
        return lever_line(TAG, GEMM_LEVER["lever"], "off", ("impl", "boltz_atom_kernels:_mm"), ("origin", "kit"), ("word", g),
                          reason=reason or ("not_requested" if g == "ieee" else "fused_unit_absent"))
    return lever_line(TAG, GEMM_LEVER["lever"], state or "on", ("impl", "boltz_atom_kernels:_mm"), ("origin", "kit"), ("word", g), reason=reason,
                      strategy=GEMM_LEVER["strategy"][g])


def report() -> Dict[str, Any]:
    BA = _STATE["module"]
    rep = BA.report() if BA is not None else {}
    per = {u: {"lever": UNITS[u]["lever"], "census": _census(u, rep), "line": line(u)} for u in UNITS}
    return {"applied": list(_STATE["applied"]), "variant": ",".join(_STATE["units"]), "gemm": _STATE["gemm"], "release": _STATE.get("release", "graph"), "units": per,
            "lines": [per[u]["line"] for u in UNITS] + [gemm_line()], "gate": verdict(rep), "expected": list(EXPECTED),
            "module": {k: rep.get(k) for k in ("refreshes", "shape_drops", "releases", "hoist_c_m", "cfg", "last", "triton")}}
