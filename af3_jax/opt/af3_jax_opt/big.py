"""The ``big`` release mode of this kit (the memory mode beside off/exact/fast): its line, its selection and its evidence — the package side.
``big`` is ONE composition on the FAST base: fast's mode with the FlashPairformer kernels and diffusion-conditioning hoist disengaged, exact's
bitwise Pallas levers at the vacated sites, and the kit's five memory levers; parametric on the base. The package never imports the model —
levers are applied in the model process by ``big_launch.py``. This module resolves the same selection the launcher will make so the wrapper can
name the levers in force before the run and check the transcript's ACTIVE/census lines against it afterward."""
from __future__ import annotations

import hashlib
import re
from typing import List, Optional

from . import modes as _modes

MODE = _modes.BIG
LINE = _modes.BIG                                                    # the line the launcher runs: the mode's own composition (modes.BIG_LINES names its base)
PREFIX = "AF3_JAX"                                                     # the stem the core's registry keys the memory levers' switches by (opt_core.mem.registry); the kit sets no such switch in any
                                                                       # environment: a --n_gpu P > 1 composition names the levers it switches off on the launcher's --off argument (modes.with_n_gpu)
CORE_MODULES = ("opt_core.mem.registry", "opt_core.mem.record", "opt_core.mem.compose", "opt_core.mem.ckpt", "opt_core.mem.jax_mem")   # the core producers of this mode (big_levers / big_launch import them): gate() names the first absent one as core_missing:<module> — the kit's producer table (modes) lists the same tuple
LAUNCHER = _modes.BIG_LAUNCHER                                       # beside this module: the base's installation, then the levers, then the script
TAG_RX = r"\[af3-jax-opt\]"
# the launcher's lines in the model process's transcript (opt_core.mem.record.AppliedRecord.active_line + big_launch's census line at exit):
#   [af3-jax-opt] ACTIVE mode=big:<lever,…> base=<mode> drop=<…|none> exact=<label> refused=<…|none> off=<…|none> on=<…|none> allocator=<…|none> line=big n_gpu=<P> [sharding=rowpair]
#   [af3-jax-opt] BIG census units=<n> applied=<lever,…|none> refused=<lever,…|none> traced=<lever=n,…> verdict=ok|partial n_gpu=<P> [sharding=rowpair]
ACTIVE_RX = re.compile(TAG_RX + r" ACTIVE mode=big:(?P<levers>[a-z0-9_,]*)(?P<fields>(?: [a-z_]+=\S*)*)\s*$")
CENSUS_RX = re.compile(TAG_RX + r" BIG census units=(?P<units>\S+) applied=(?P<applied>\S+) refused=(?P<refused>\S+) traced=(?P<traced>\S*) verdict=(?P<verdict>\w+)")
FAILED_RX = re.compile(TAG_RX + r" BIG FAILED\b.*")


def gate() -> Optional[str]:
    """The reason the mode cannot run here, or None: the core carries ``opt_core.mem.registry`` (nothing is vendored into the kit) and the
    kit's lever module registers every lever of its line."""
    import importlib
    for name in CORE_MODULES:
        try:
            importlib.import_module(name)
        except ImportError as e:
            return f"big: core_missing:{name} ({type(e).__name__}: {e}) — the memory mode lands with core >= 0.4.0; every other mode is unaffected"
    _r = importlib.import_module("opt_core.mem.registry")
    try:
        line = levers()
    except Exception as e:  # noqa: BLE001
        from opt_core.oom import is_oom
        if is_oom(e): raise                                               # OOM propagates: no fallback applied (opt_core.oom.is_oom)
        return f"big: the kit's lever module did not register ({type(e).__name__}: {e})"
    missing = [lv for lv in line if lv not in _r.LEVERS]
    if missing:
        return f"big: levers not registered: {', '.join(missing)}"
    return None


def levers() -> tuple:
    """The ordered memory-lever line of this kit (big_levers.LINE_LEVERS[LINE]). Importing the module registers the levers in this
    interpreter — stdlib + opt_core at module level, the model stack only inside apply."""
    from . import big_levers
    return tuple(big_levers.LINE_LEVERS[LINE])


def selection() -> dict:
    """The mode's one lever set: the registry's resolution of the line with no switch given (opt_core.mem.registry.selection — the resolution
    the launcher's apply makes in a one-device model process): every memory lever of the line, in order, at its registered settings —
    ``{'line', 'base', 'levers', 'off', 'on', 'flags', 'refusals', 'allow_partial'}`` (off/on/flags empty, allow_partial False: nothing selects a sub-line)."""
    from opt_core.mem import registry as _r
    sel = _r.selection(levers())
    return {"line": LINE, "base": _modes.BIG_LINES[LINE], "levers": tuple(sel.levers), "off": tuple(sel.off_by_flag), "on": tuple(sel.on_by_flag),
            "flags": dict(sel.flags), "refusals": [str(r) for r in sel.refusals], "allow_partial": bool(sel.allow_partial)}


def cache_suffix() -> str:
    """``-<line>-<sha8>`` of the lever set (the levers in force and their settings): the mode's cache class suffix (stack.cache_dir)."""
    sel = selection()
    key = ",".join(sel["levers"]) + "|" + ",".join(f"{k}={v}" for k, v in sorted(sel["flags"].items()))   # flags: always empty (no switch is ever set) — the key keeps its form, the class its name
    return f"-{sel['line']}-{hashlib.sha256(key.encode()).hexdigest()[:8]}"


def parse_active(line: str) -> Optional[dict]:
    """The launcher's ACTIVE line as ``{'levers': (...), 'base', 'drop', 'exact', 'refused', 'off', 'on', 'allocator', 'line', ...}`` (every
    ``key=value`` field verbatim; ``levers`` split), or None when the line is not one."""
    m = ACTIVE_RX.search(line)
    if not m:
        return None
    fields = dict(kv.split("=", 1) for kv in m.group("fields").split() if "=" in kv)
    fields["levers"] = tuple(x for x in m.group("levers").split(",") if x)
    return fields


def parse_census(line: str) -> Optional[dict]:
    """big_launch's census line at exit as ``{'units', 'applied': (...), 'refused': (...), 'traced': {lever: n}, 'verdict'}`` or None."""
    m = CENSUS_RX.search(line)
    if not m:
        return None
    tup = lambda s: tuple(x for x in s.split(",") if x and x != "none")  # noqa: E731
    traced = {}
    for kv in m.group("traced").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            traced[k] = int(v) if v.isdigit() else v
    return {"units": m.group("units"), "applied": tup(m.group("applied")), "refused": tup(m.group("refused")), "traced": traced, "verdict": m.group("verdict")}


def lever_evidence(lines: List[str], expected: Optional[List[str]] = None) -> dict:
    """The launcher's evidence in a model process's transcript against the levers the wrapper composed (``expected``: the memory levers of
    the composition in order — modes.lever_evidence passes them from the resolved composition; under --n_gpu P > 1 the levers the
    row-sharded pair stack supersedes are switched off by the wrapper, modes.with_n_gpu; default = the mode's lever set, selection): the
    ACTIVE line present, its line the mode's, nothing refused, its levers exactly ``expected``, AND the census line at exit with verdict=ok
    (the launcher's fail-closed gate; a partial names its levers) — ``{'ok', 'reported', 'reason', 'active', 'census'}``. A missing census =
    the launcher did not reach its exit: named, never assumed ok."""
    sel = selection()
    exp = expected if expected is not None else [lv for lv in _modes.KIT_MODES[_modes.BIG]["levers"]]   # default: the mode's one-GPU composition (the registered line minus modes.BIG_NOT_COMPOSED); the wrapper passes the resolved composition
    sel = {**sel, "levers": tuple(x.lower() for x in exp)}
    seen = [d for ln in lines for d in [parse_active(ln)] if d]
    reported = [ln.strip() for ln in lines if ACTIVE_RX.search(ln) or CENSUS_RX.search(ln) or FAILED_RX.search(ln)]
    if not seen:
        return {"ok": False, "reported": reported, "reason": "no big ACTIVE line: the launcher did not run", "active": None, "census": None}
    a = seen[-1]
    census = ([c for ln in lines for c in [parse_census(ln)] if c] or [None])[-1]
    ev = {"ok": False, "reported": reported, "reason": None, "active": a, "census": census}
    if a.get("line") != sel["line"]:
        return {**ev, "reason": f"line={a.get('line')}, the selection is {sel['line']}"}
    if a.get("refused", "none") != "none":
        return {**ev, "reason": f"refused={a['refused']}"}
    if a["levers"] != tuple(sel["levers"]):
        return {**ev, "reason": f"levers active={','.join(a['levers']) or 'none'}: the selection is {','.join(sel['levers']) or 'none'}"}
    if census is None:
        return {**ev, "reason": "no BIG census line: the launcher did not reach its exit gate"}
    if census["verdict"] != "ok":
        partial = [lv for lv in sel["levers"] if lv not in census["applied"] or lv in census["refused"]]
        return {**ev, "reason": f"census verdict={census['verdict']} applied={','.join(census['applied']) or 'none'} refused={','.join(census['refused']) or 'none'}"
                               + (f" partial={','.join(partial)}" if partial else "")}
    return {**ev, "ok": True}
