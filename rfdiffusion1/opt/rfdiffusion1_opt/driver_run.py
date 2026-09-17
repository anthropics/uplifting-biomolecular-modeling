"""The driver process: ``python -m rfdiffusion1_opt.driver_run --fixed K=V [--fixed K=V ...] --cases <cases.json> [--log <json>] [--compose K=V ...] <script> [args...]``.

Runs the kit line (the base kit's resident driver ``opt/forward/fast_inference/drivers/rfd_bench.py``) as ``__main__`` (``runpy``, the
script's directory first on ``sys.path`` as ``python <script>`` would put it) with ``hydra.compose`` wrapped, so that every per-case
config the driver composes carries the launch's values on the keys the driver hard-codes (``modes.DRIVER_FIXED``:
``inference.write_trajectory`` and ``inference.cautious`` — ``rfd_bench.py:103``), the row's
``inference.design_startnum`` token (the resident path composes without it, ``rfd_bench.py:99-107``, while its loop numbers the designs from
the row's ``startnum``, ``:345``; the stock caller composes it, ``design.py`` ``stock_overrides``) and the launch's carried typed keys
(``--compose``: every typed hydra key outside the driver's constants and the target — the same strings the stock caller appends). The driver's
strings on those keys are replaced by the ``--fixed`` values (typed or upstream's defaults, ``modes.settings_of``), every other override passes through
verbatim in the driver's order (``merge_overrides``). The kit bytes are untouched: the substitution is the package's, in the process the
package launches (``stack.driver_command``); the composed config is what upstream's sampler reads and what the ``.trb`` record carries,
so both arms run the same values on those keys. Every composition is appended to ``--log`` (JSON lines: the case, the dropped
strings, the appended strings) — ``opt_manifest.json`` ``driver_passes[].compose`` reads it back.

The SE(3) add-on (registry lever T2): when the mode's environment row carries ``RFD_SE3FAST=t2`` (``modes.MODES["base"]["fast"]``;
``stack.driver_environment`` exports the row and drops every other RFD_* name), ``arm_se3fast`` runs the add-on's documented in-driver
route in this process before the driver imports the model — ``opt/forward/se3fast_addon`` first on ``sys.path``, ``rfd_se3fast.apply(mode)``
(the add-on's rule, rfd_se3fast/__init__.py: apply after ``rfdiffusion`` is importable, before the first design; the kit driver's own levers patch after
it, the order of the add-on's shim ``drivers/rfd_bench_se3fast.py``). It refuses by name, exit 3, when ``triton`` does not import or the
add-on reports anything but its Triton line (the add-on's own t2 -> t2torch degradation, ``rfd_se3fast/__init__.py:44-51``, never runs
silently here), and prints the add-on's call counters at exit (``SE3FAST_FINAL``).

Lever IO1 (``pdbio``): when the row carries ``RFD_PDBIO=1``, ``pdbio.arm`` replaces ``rfdiffusion.util.writepdb`` / ``writepdb_multi`` with
the package's numpy writers before the driver imports them — the same bytes (the first call per argument signature is compared with
upstream's own writer in this process; a difference keeps upstream's file, is printed, and makes this process exit 5 after the pass), a
fraction of the CPU time inside each design's completion window. Its counters print at exit (``PDBIO_FINAL``).

Lever K2's kernel (``route_core_kernels``): the driver's ``import rfd_layernorm`` (``rfd_bench.py:133`` under ``--triton-ln 1``) resolves to
the shared core's carried Triton row LayerNorm ``opt_core/kernels/rfd_layernorm.py`` — the core's one route for a carried kernel
(``opt_core.kernels.route`` / ``route_check``), installed in this process before the driver runs; the tree carries no copy of the kernel,
so a driver started bare (``python drivers/rfd_bench.py … --triton-ln 1``, outside this module) has no ``rfd_layernorm`` to import.
A route whose name would resolve to other bytes is refused by name, exit 3.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from typing import Dict, List, Optional, Sequence

from . import pdbio
from .modes import DRIVER_FIXED, UPSTREAM_DEFAULTS
from .registry import KIT_SE3, SE3FAST_ENV, SE3FAST_MODE

STARTNUM_KEY = "inference.design_startnum"
TAG = "[rfdiffusion1-opt]"
PREFIX_KEY = "inference.output_prefix"
CORE_KERNELS = {"rfd_layernorm": "K2"}                                  # top-level name the driver imports -> the lever it serves, for every kernel the shared core carries (opt_core/kernels/<name>.py): K2's Triton row LayerNorm


STARTNUM_TOKEN = "startnum_compose"                              # a row's own `inference.design_startnum` token (design.py: the typed value verbatim in the hydra form, None when untyped; absent = the row's startnum, the cases form)


def case_startnums(cases_path: Optional[str]) -> Dict[str, Optional[str]]:
    """{absolute output prefix: the `inference.design_startnum` value to compose (None = compose none)} of the driver's cases file. Every row
    carries its `prefix` (design.write_cases); the value is the row's STARTNUM_TOKEN when it has one, else its startnum (0 when absent, as
    the driver's loop starts: rfd_bench.py:144) — the same token the stock command line of the request carries (design.stock_overrides)."""
    if not cases_path:
        return {}
    with open(cases_path, encoding="utf-8") as fh:
        rows = json.load(fh)
    out: Dict[str, Optional[str]] = {}
    for c in rows:
        tok = c[STARTNUM_TOKEN] if STARTNUM_TOKEN in c else str(int(c.get("startnum", 0)))
        out[os.path.abspath(c["prefix"])] = None if tok is None else str(tok)
    return out


def case_of(overrides: Sequence[str], startnums: Dict[str, Optional[str]]) -> Optional[str]:
    """The row a composed override list belongs to, by its ``inference.output_prefix`` (the driver composes the row's own prefix: rfd_bench.py:100,113)."""
    for o in overrides or []:
        k, _, v = str(o).partition("=")
        if k == PREFIX_KEY:
            p = os.path.abspath(v)
            return p if p in startnums else None
    return None


def merge_overrides(overrides: Optional[Sequence[str]], preset_ov: Sequence[str], startnum: Optional[str], fixed: Optional[Sequence[str]] = None,
                    extra: Optional[Sequence[str]] = None) -> dict:
    """``overrides`` with the entries on the fixed keys (``DRIVER_FIXED`` + ``inference.design_startnum`` when a startnum token is given + the
    keys of ``extra``) dropped and the ``--fixed`` values + ``inference.design_startnum=<startnum>`` + ``extra`` appended; every other
    entry verbatim, in order. Returns {"overrides": [...], "dropped": [...], "appended": [...]}."""
    extra = [str(e) for e in (extra or [])]
    fixed_keys = set(DRIVER_FIXED if fixed is None else fixed) | {e.split("=", 1)[0] for e in extra}
    if startnum is not None:
        fixed_keys.add(STARTNUM_KEY)
    kept, dropped = [], []
    for o in overrides or []:
        (dropped if str(o).split("=", 1)[0] in fixed_keys else kept).append(str(o))
    appended = list(preset_ov) + ([f"{STARTNUM_KEY}={startnum}"] if startnum is not None else []) + extra
    return {"overrides": kept + appended, "dropped": dropped, "appended": appended}


def fixed_overrides(fixed: Optional[Sequence[str]] = None) -> List[str]:
    """The values composed on the driver's hard-coded keys: ``fixed`` (``KEY=VALUE`` on DRIVER_FIXED keys, as the launch typed or defaulted
    them, modes.settings_of) completed with upstream's defaults for any DRIVER_FIXED key not given, in DRIVER_FIXED order."""
    given = {str(f).split("=", 1)[0]: str(f) for f in (fixed or [])}
    unknown = [k for k in given if k not in DRIVER_FIXED]
    if unknown:
        raise SystemExit(f"driver_run: --fixed takes {' / '.join(DRIVER_FIXED)} only, got {unknown}")
    return [given.get(k, f"{k}={UPSTREAM_DEFAULTS[k]}") for k in DRIVER_FIXED]


def install(fixed: Optional[Sequence[str]] = None, cases_path: Optional[str] = None, log_path: Optional[str] = None, extra: Optional[Sequence[str]] = None) -> List[str]:
    """Wrap ``hydra.compose`` for this process: the launch's values replace the driver's constants on DRIVER_FIXED, the case's startnum and
    the recipe's ``extra`` strings are composed."""
    import hydra
    preset_ov = fixed_overrides(fixed)
    startnums = case_startnums(cases_path)
    original = hydra.compose

    def compose(config_name=None, overrides=None, *args, **kwargs):
        case = case_of(overrides or [], startnums)                                   # a compose outside the rows (none in the driver) is left as typed but for the fixed keys
        m = merge_overrides(overrides, preset_ov, startnums.get(case) if case is not None else None, extra=extra)
        if log_path:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"case": case, "fixed": preset_ov, "dropped": m["dropped"], "appended": m["appended"]}) + "\n")
        return original(config_name, m["overrides"], *args, **kwargs)

    compose.__wrapped__ = original  # type: ignore[attr-defined]
    hydra.compose = compose
    return preset_ov


def _refuse(fact: str) -> None:
    print(f"{TAG} REFUSED: {fact}", flush=True)
    raise SystemExit(3)


def route_core_kernels() -> Dict[str, str]:
    """Route every CORE_KERNELS name to the shared core's carried copy in this process (``opt_core.kernels.route``: a meta-path entry for
    exactly that top-level name; nothing is imported until a driver imports it) and gate it (``opt_core.kernels.route_check``: the name
    resolves to the core copy's own bytes) — refused by name, exit 3, otherwise. Returns ``{name: the core copy's path}``."""
    from opt_core import kernels
    routes = {}
    for name, lever in CORE_KERNELS.items():
        routes[name] = kernels.route(name)
        gate = kernels.route_check(name)
        if not gate.ok:
            _refuse(f"kernel {name} (lever {lever}): {gate.reason}")
    return routes


def arm_se3fast(value: Optional[str], addon_dir: Optional[str] = None) -> Optional[dict]:
    """The SE(3) add-on's in-driver route for ``RFD_SE3FAST=<value>``: None when the variable is unset / ``0`` (nothing imported); for
    ``t2`` the add-on directory first on ``sys.path``, ``import triton`` (refused by name when it fails), ``rfd_se3fast.apply(mode="t2")``,
    and a refusal unless the add-on reports its Triton line applied with no fallback; any other value is refused by name (``t2torch`` and
    ``exact`` are the add-on's reference / reserved settings, not lines of this tree). Returns the add-on's stats dict."""
    if value is None or not str(value).strip() or str(value).strip() == "0":
        return None
    value = str(value).strip().lower()
    if value != SE3FAST_MODE:
        _refuse(f"{SE3FAST_ENV}={value!r}: the one value a mode of this tree sets is {SE3FAST_MODE!r} (registry lever T2)")
    if addon_dir is None:
        from .stack import kit_dir
        addon_dir = kit_dir(KIT_SE3)
    if not os.path.isfile(os.path.join(addon_dir, "rfd_se3fast", "__init__.py")):
        _refuse(f"{SE3FAST_ENV}={value}: the add-on directory {addon_dir} does not hold rfd_se3fast/")
    if addon_dir not in sys.path:
        sys.path.insert(0, addon_dir)
    try:
        import triton  # noqa: F401  (the Triton line's requirement; the add-on would otherwise degrade to its torch reference path)
    except ImportError as e:
        _refuse(f"{SE3FAST_ENV}={value} needs triton in this process: {e!r}")
    import rfd_se3fast  # noqa: E402  (the add-on, byte-equal: opt/forward/se3fast_addon/rfd_se3fast)
    st = rfd_se3fast.apply(mode=value)
    if not st.get("applied") or st.get("fallback_reason") or not st.get("triton"):
        _refuse(f"{SE3FAST_ENV}={value}: rfd_se3fast did not apply its Triton line: {st}")
    import atexit

    def _final() -> None:
        s = rfd_se3fast.stats()
        print(f"{TAG} SE3FAST_FINAL " + " ".join(f"{k}={s.get(k)}" for k in ("mode", "n_calls", "n_full", "n_topk", "triton", "fallback_reason", "verify_fail")), flush=True)
        if s.get("fallback_reason"):
            print(f"{TAG} REFUSED: rfd_se3fast reports a fallback at exit: {s.get('fallback_reason')}", flush=True)
    atexit.register(_final)
    return st


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rfdiffusion1_opt.driver_run")
    ap.add_argument("--fixed", action="append", default=[], metavar="KEY=VALUE", help=f"the value composed on a driver-hard-coded key ({' / '.join(DRIVER_FIXED)}); upstream's default when absent")
    ap.add_argument("--cases", default=None, help="the driver's cases file: the per-case startnum composed as inference.design_startnum")
    ap.add_argument("--log", default=None, help="JSON lines, one per composed config (the case, the dropped and the appended override strings)")
    ap.add_argument("--compose", action="append", default=[], metavar="KEY=VALUE", help="a recipe override composed on every config (repeatable)")
    ap.add_argument("script")
    ap.add_argument("args", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    install(a.fixed, a.cases, a.log, a.compose)
    route_core_kernels()
    arm_se3fast(os.environ.get(SE3FAST_ENV))
    io1 = pdbio.arm(os.environ.get(pdbio.ENV))
    script = os.path.abspath(a.script)
    sys.path.insert(0, os.path.dirname(script))
    sys.argv = [script] + list(a.args)
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        if io1 is not None:
            print(f"{TAG} {pdbio.final_line()}", flush=True)
    if io1 is not None and pdbio.stats()["n_mismatch"]:                  # lever IO1 wrote upstream's bytes where its own differed; the pass must not read as clean
        print(f"{TAG} REFUSED: {pdbio.TAG} produced bytes unlike rfdiffusion.util's on {pdbio.stats()['mismatches']}; exit 5", flush=True)
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
