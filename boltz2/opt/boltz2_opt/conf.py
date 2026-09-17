"""boltz2_opt.conf — the engine adapter of the CONF levers (``opt/forward/conf/``: the template module, the confidence module's glue and heads,
the distogram head, the input front of boltz 2.2.1), applied in the worker process by ``boltz2_opt.worker_launch --attach conf`` right after the
worker's own model import (``boltz.model.models.boltz2``: every patched class is imported by then). The levers are class-level replacements of
stock methods (nothing of the installed ``boltz`` is edited); each is its own word of the switch (one row word per lever).

Switch ``BOLTZ_CONF=<word>[,<word>…]`` (the mode table sets it; an unknown word refuses by name, nothing applied):
    condproj Tier 2 — DiffusionConditioning's 24 token-layer + 3+3 atom-layer LayerNorm+Linear pair-bias projections as one shared-statistics
             normalisation + one folded GEMM per list (cond_levers.py; the fast row's CONF word)
    tfeat    Tier 1 — recycle-invariant template featurization computed once per prediction (template_levers.py; exact, ~20 ms/item @1400: parked, no row)
    tdummy   Tier 1 by algebra, input-conditional — the all-dummy template call replaced by its RNG draws + a zero update
             (template_levers.py; measured −41/−269/−598 ms/item; parked, in no row)

Contract (read by worker_launch and stack.evidence, the shape of boltz2_opt.trimul): ``LEVERS``; ``apply(spec=None)`` returns the levers
applied ([] when the switch is absent: the attach refuses by name); ``report()`` returns ``{applied, disabled: {}, line, census, gate}``. One
activation-evidence line per applied lever is printed on stderr at interpreter exit:
    [boltz2-opt] LEVER name=conf.<word> state=on impl=forward/conf/<module>.py origin=kit <counters…>
"""
from __future__ import annotations

import atexit
import importlib.util
import os
import sys
from typing import Any, Dict, List, Optional

TAG = "boltz2-opt"
SWITCH = "BOLTZ_CONF"
BUNDLE = "forward"                       # opt/-relative: the directory carrying the conf package (stack.kit_path resolves it)
PACKAGE = "conf"
LEVERS = ("condproj", "tfeat", "tdummy")   # every lever word this adapter installs, in canonical order
MODULE_OF = {"condproj": "cond_levers", "tfeat": "template_levers", "tdummy": "template_levers"}
TIER_OF = {"condproj": 2, "tfeat": 1, "tdummy": 1}
PATCHED_OF = {"condproj": ("DiffusionConditioning.forward",), "tfeat": ("TemplateV2Module.forward", "Boltz2.forward"), "tdummy": ("TemplateV2Module.forward", "Boltz2.forward")}
PACKAGE_FILES = tuple(f"{BUNDLE}/{PACKAGE}/{f}" for f in ("__init__.py", "template_levers.py", "cond_levers.py"))
_STATE: Dict[str, Any] = {"applied": [], "mods": {}, "pkg": None, "words": None}


def parse_words(spec: Optional[str]) -> List[str]:
    """``BOLTZ_CONF`` -> the lever words in canonical order (duplicates dropped); an unknown word raises by name."""
    spec = (spec or "").strip().lower()
    if not spec:
        return []
    seen = []
    for tok in spec.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in LEVERS:
            raise ValueError(f"{SWITCH}: unknown lever word {tok!r} (known: {', '.join(LEVERS)})")
        if tok not in seen:
            seen.append(tok)
    return [l for l in LEVERS if l in seen]


def requested(environ=None) -> bool:
    env = os.environ if environ is None else environ
    return bool((env.get(SWITCH) or "").strip())


def _package_dir() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # opt/
    return os.path.join(here, BUNDLE, PACKAGE)


def _load_package():
    """Import ``conf`` (opt/forward/conf) as the package ``bz2conf`` from its own directory (the bundle root never enters sys.path). Idempotent."""
    if _STATE["pkg"] is not None:
        return _STATE["pkg"]
    pkg_dir = _package_dir()
    missing = [f for f in PACKAGE_FILES if not os.path.isfile(os.path.join(os.path.dirname(pkg_dir), os.path.basename(pkg_dir), os.path.basename(f)))]
    if missing:
        raise RuntimeError(f"conf: kit files missing under opt/: {missing}")
    name = "bz2conf"
    spec = importlib.util.spec_from_file_location(name, os.path.join(pkg_dir, "__init__.py"), submodule_search_locations=[pkg_dir])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"conf: no importable package at {pkg_dir}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _STATE["pkg"] = mod
    return mod


def _module(name: str):
    if name not in _STATE["mods"]:
        _load_package()
        import importlib as _il
        _STATE["mods"][name] = _il.import_module(f"bz2conf.{name}")
    return _STATE["mods"][name]


def apply(spec: Optional[str] = None) -> List[str]:
    """Install the levers named by ``BOLTZ_CONF`` (or ``spec``). Idempotent; returns the applied lever words ([] when none requested)."""
    if _STATE["applied"]:
        return list(_STATE["applied"])
    words = parse_words(os.environ.get(SWITCH) if spec is None else spec)
    _STATE["words"] = words
    if not words:
        return []
    import boltz.model.models.boltz2  # noqa: F401  (the trigger: every patched class is importable)
    by_mod: Dict[str, List[str]] = {}
    for w in words:
        by_mod.setdefault(MODULE_OF[w], []).append(w)
    applied: List[str] = []
    for mname, ws in by_mod.items():
        got = _module(mname).install(ws)
        applied += [w for w in ws if w in got]
    _STATE["applied"] = [l for l in LEVERS if l in applied]
    atexit.register(_exit_lines)
    sys.stderr.write(f"[{TAG}] conf: applied {','.join(_STATE['applied'])} ({SWITCH}={','.join(words)})\n")
    return list(_STATE["applied"])


def lines() -> List[str]:
    out = []
    for w in _STATE["applied"]:
        m = _STATE["mods"].get(MODULE_OF[w])
        st = (m.report().get("stats") if m is not None else {}) or {}
        counters = " ".join(f"{k}={v}" for k, v in sorted(st.items()))
        out.append(f"[{TAG}] LEVER name=conf.{w} state=on tier={TIER_OF[w]} impl={BUNDLE}/{PACKAGE}/{MODULE_OF[w]}.py origin=kit {counters}")
    return out


def _exit_lines() -> None:
    try:
        for ln in lines():
            sys.stderr.write(ln + "\n")
    except Exception as e:   # never mask the worker's own exit path
        sys.stderr.write(f"[{TAG}] conf: exit line failed: {type(e).__name__}: {e}\n")


def census() -> Dict[str, Any]:
    return {MODULE_OF[w]: (_STATE["mods"][MODULE_OF[w]].report() if MODULE_OF[w] in _STATE["mods"] else None) for w in _STATE["applied"]}


def verdict() -> Dict[str, Any]:
    """Fail-closed: every requested word must be installed; a template lever that saw forwards must have seen template calls."""
    words = _STATE.get("words") or []
    if not words:
        return {"ok": False, "idle": False, "reason": "not applied"}
    missing = [w for w in words if w not in _STATE["applied"]]
    if missing:
        return {"ok": False, "idle": False, "reason": f"requested but not installed: {missing}"}
    return {"ok": True, "idle": False, "reason": None}


def report() -> Dict[str, Any]:
    return {"applied": list(_STATE["applied"]), "disabled": {}, "line": lines(), "census": census(), "gate": verdict(),
            "patched": sorted({p for w in _STATE["applied"] for p in PATCHED_OF[w]}), "switch": SWITCH, "words": list(_STATE.get("words") or [])}
