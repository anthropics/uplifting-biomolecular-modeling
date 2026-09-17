"""The environment proof the design script runs in the arm's own process, before torch, transformers or esm is imported.

`prove(pins, allowed, kit_paths)` checks, from the process's own view: every variable with a prefix in stock/PINS.json
`stock_environment.must_be_absent_prefixes` is absent (a kit arm's allowed exceptions: `EF2_FAST_KIT=<value>`, exactly the mode's, and the ablation word `MODEL_OPT_LEVERS_OFF` as the operator set it);
no kit module (`ef2_*`) and no upstream module (`transformers`, `esm`, `torch`) is loaded yet; on `off` no kit directory (the design kit's `k/`)
is on `sys.path`; on every arm NO `modal` stubs directory is on `sys.path` and `sys.modules["modal"]` IS the kit's no-op stand-in
(`ef2inv_opt/_absent_sdk_stub.py`, registered by `stock_design.prepare_arm_process` — the cookbook's `import modal` route in every arm
with, minus a directory that would make `modal` importable to anything else). The record is written to `stock_env_proof.json` on `off` and carried into `run.json` (`launch`) by every arm; the
line `[ef2inv-opt] ENV-CLEAN ok|FAIL: ... arm=<stock|exact|fast>` goes to stderr. A failed proof stops the arm (exit 3) before any design runs.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, List, Optional

from .modes import KIT_MODULE_PREFIXES, KIT_PATH_MARKER, SDK_STANDIN, STUBS_MARKER

UPSTREAM_ROOTS = ("transformers", "esm", "torch")


def forbidden(environ: Dict[str, str], prefixes: Iterable[str], allowed: Optional[Dict[str, str]] = None) -> List[str]:
    """Variables of `environ` with a forbidden prefix (entries ending in '_' are prefixes, others exact names), minus the allowed
    exceptions when their value is exactly the allowed one (a wrong value is a hit: `NAME=value!=allowed`)."""
    allowed = allowed or {}
    hits = []
    for k, v in environ.items():
        if any((k.startswith(p) if p.endswith("_") else k == p) for p in prefixes):
            if k in allowed:
                if v != allowed[k]:
                    hits.append(f"{k}={v}!={allowed[k]}")
                continue
            hits.append(k)
    return sorted(set(hits))


def loaded(modules, roots: Iterable[str], prefixes: bool = False) -> List[str]:
    out = []
    for m in modules:
        top = m.split(".")[0]
        if prefixes:
            if any(top.startswith(r) for r in roots):
                out.append(m)
        elif top in roots:
            out.append(m)
    return sorted(out)


def kit_dirs(path: Iterable[str]) -> List[str]:
    return sorted(p for p in path if p and os.path.isfile(os.path.join(p, KIT_PATH_MARKER)))


def stub_dirs(path: Iterable[str]) -> List[str]:
    return sorted(p for p in path if p and os.path.isfile(os.path.join(p, STUBS_MARKER)))


def prove(prefixes: Iterable[str], allowed: Optional[Dict[str, str]] = None, arm: str = "off", environ=None, modules=None, path=None) -> dict:
    """The proof record; `clean` False lists every violation. On `off`: forbidden variables absent, no kit module loaded, no kit dir on
    sys.path. On a kit arm: the allowed exception exactly, no kit module loaded YET (the script imports the kit when fastkit.install / the kit-enable step run
    after the proof), the kit dir present exactly once. On every arm: no `modal` stubs directory on sys.path and ``modules["modal"]`` is the
    kit's stand-in (``SDK_STANDIN``) — absent or any other file is a violation."""
    environ = os.environ if environ is None else environ
    modules = sys.modules if modules is None else modules
    path = sys.path if path is None else path
    hits = forbidden(environ, prefixes, allowed if arm != "off" else None)
    kit_mods = loaded(modules, KIT_MODULE_PREFIXES, prefixes=True)
    up = loaded(modules, UPSTREAM_ROOTS)
    kd = kit_dirs(path)
    sd = stub_dirs(path)
    problems = []
    if hits:
        problems.append(f"forbidden env {hits}")
    if kit_mods:
        problems.append(f"kit modules loaded {kit_mods}")
    if up:
        problems.append(f"upstream loaded before the proof {up}")
    if arm == "off" and kd:
        problems.append(f"kit dir on sys.path {kd}")
    if arm != "off" and len(kd) != 1:
        problems.append(f"kit dir on sys.path expected exactly once, found {kd}")
    if sd:
        problems.append(f"modal stubs dir on sys.path {sd}")
    modal_file = getattr(modules.get("modal"), "__file__", None) if modules.get("modal") is not None else None
    if modules.get("modal") is None:
        problems.append("modal stand-in not registered (stock_design.prepare_arm_process)")
    elif os.path.abspath(str(modal_file)) != os.path.abspath(SDK_STANDIN):
        problems.append(f"modal is {modal_file!r}, not the kit's stand-in {SDK_STANDIN}")
    rec = {"arm": arm, "prefixes_must_be_absent": sorted(prefixes), "allowed": dict(allowed or {}) if arm != "off" else {},
           "forbidden_present": hits, "kit_modules_loaded": kit_mods, "upstream_loaded": up, "kit_dirs_on_path": kd, "stub_dirs_on_path": sd, "modal": modal_file,
           "isolated": bool(sys.flags.isolated), "no_user_site": bool(sys.flags.no_user_site), "clean": not problems, "problems": problems}
    return rec


def line(rec: dict) -> str:
    """``ENV-CLEAN ok|FAIL: ... arm=<stock|exact|fast>`` (printed under the ``[ef2inv-opt]`` prefix by report.log)."""
    arm = "stock" if rec["arm"] == "off" else rec["arm"]
    return "ENV-CLEAN " + ("ok" if rec["clean"] else "FAIL: " + "; ".join(rec["problems"])) + f" arm={arm} absent={','.join(rec['prefixes_must_be_absent'])}"
