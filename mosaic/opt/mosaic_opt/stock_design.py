
"""The stock caller — mode ``off``: the kit driver with every lever off, in a clean process, the environment proved before jax loads.

Upstream mosaic has no command line; the one scriptable form of the notebook's calls is the kit driver `tools/public_design_run.py` with
`--weights torch` and no P1 environment — the stock arms `A_stock1` (modes.ROWS; featurizes in-process, `--features-out` optional) and
`D_stock2` (the frozen features of arm A: the deterministic recipe's "identical inputs"). `design --mode off` runs
`python -s -m mosaic_opt.stock_design ...` in a subprocess whose environment the parent has stripped of every prefix in
`stock/PINS.json` "stock_environment.must_be_absent_prefixes" (the package's own switches included); this module then, in that process
and before any upstream or kit module is imported, proves the environment (no forbidden variable, no kit or lever module loaded, no
`mosaic/fast` directory on sys.path) and records the install layout it runs on — whether `run.sh install` has placed the kit's lever
files into `site-packages/mosaic/fast/` (found without importing `mosaic`: `stack.installed_mosaic_dir`) and whether those files are
the kit's own lever files (`stack.installed_fast_check`; a copy missing kit files, or carrying extra ones, is refused). The stock install
has no `mosaic/fast/` (`STOCK.md`); once `run.sh install` has placed the lever files the driver's two observers import from the installed copy instead of the kit's own
files and the proof says so (`install_layout`, `stock_install`). It prints the ENV-CLEAN line and hands the process to the driver
(`runpy.run_path(..., run_name="__main__")`), so the design is the driver's own — the block marked ``# --- the stock call`` below. Kit
code on this path, none of it a lever: the driver itself and the two observer modules it imports on every arm (`fastload.fingerprint`,
`numstate.snapshot/diff`, loaded through `recipe.kit_modules()` and called around the design in `tools/public_design_run.py`), recorded in the proof by name.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time

from . import TAG                                       # the kit's one tag spelling (the package root is import-free: nothing of the kit or the core loads with it)

PREFIX = f"[{TAG} stock]"
OBSERVER_MODULES = ("mosaic.fast.fastload (fingerprint)", "mosaic.fast.numstate (snapshot/diff)")


STOCK_LAYOUT = "the pinned stack without the kit's lever files"      # STOCK.md: the stock install has no mosaic/fast/ sub-package


def install_layout(fc: dict) -> str:
    """One sentence for the layout the proof found (`stack.installed_fast_check`)."""
    if not fc.get("installed"):
        return f"{STOCK_LAYOUT} (no mosaic/fast/ installed: the driver's observers import the kit's own files)"
    if fc.get("clean"):
        return f"the pinned stack with the kit's lever files at {fc.get('path')} (placed by run.sh install; not the stock install)"
    return f"mosaic/fast/ at {fc.get('path')} is missing kit files ({', '.join(k for k, v in (fc.get('files') or {}).items() if v == 'absent')}{' +' + ','.join(fc['extra']) if fc.get('extra') else ''})"


def env_proof(pins: dict, environ=None, modules=None, path=None, kit_dir=None) -> dict:
    """The stock environment, proved: every variable with a forbidden prefix (allowed exceptions aside) is absent; no kit lever module
    and no upstream module is loaded yet; no `mosaic/fast` directory on sys.path; and, when the kit directory is given, the installed
    `mosaic/fast/` (if any) carries the kit's own lever files — the layout is recorded either way."""
    environ = os.environ if environ is None else environ
    modules = sys.modules if modules is None else modules
    path = sys.path if path is None else path
    se = pins.get("stock_environment") or {}
    prefixes, exceptions = list(se.get("must_be_absent_prefixes") or []), list(se.get("allowed_exceptions") or [])
    present = sorted(k for k in environ if any(k.startswith(p) for p in prefixes) and k not in exceptions)
    unset_bad = sorted(k for k in (se.get("must_be_unset_every_arm") or []) if environ.get(k))
    loaded = sorted(m for m in modules if m == "mosaic" or m.startswith("mosaic.") or m.startswith("joltz") or m.startswith("boltz") or m == "jax" or m.startswith("jax."))
    on_path = sorted(p for p in path if p and os.path.basename(os.path.normpath(p)) == "fast" and "mosaic" in p)
    fc = None
    if kit_dir:
        from .stack import installed_fast_check                                  # package-internal (stdlib only); nothing upstream is imported
        fc = installed_fast_check(kit_dir)
    return {"env_prefixes_must_be_absent": prefixes, "env_allowed_exceptions": exceptions, "env_present_forbidden": present,
            "env_precision_must_be_unset": list(se.get("must_be_unset_every_arm") or []), "env_precision_set": unset_bad,
            "kit_modules_loaded_before_call": loaded, "mosaic_fast_on_sys_path": on_path,
            "mosaic_fast_installed": fc, "install_layout": install_layout(fc) if fc is not None else "not checked (no kit directory given)",
            "stock_install": (not fc.get("installed")) if fc is not None else None,
            "kit_code_on_this_path": ["tools/public_design_run.py (the driver)", *OBSERVER_MODULES],
            "clean": not present and not unset_bad and not loaded and not on_path and (fc is None or fc["clean"]),
            "reads": list(se.get("reads") or []), "values": {k: environ.get(k) for k in (se.get("reads") or []) + exceptions if k in environ}}


def env_line(proof: dict) -> str:
    fc = proof.get("mosaic_fast_installed")
    if fc is None:
        fast = "unchecked"
    elif not fc.get("installed"):
        fast = "not-installed"
    else:
        fast = f"{'kit-bytes' if fc.get('clean') else 'ABSENT'}@{fc.get('path')}"
    return (f"{PREFIX} ENV-CLEAN {'ok' if proof['clean'] else 'FAIL'}: absent={','.join(proof['env_prefixes_must_be_absent'])} "
            f"present_forbidden={','.join(proof['env_present_forbidden']) or 'none'} kit_modules_loaded={','.join(proof['kit_modules_loaded_before_call']) or 'none'} "
            f"kit_code_on_path=driver+{len(OBSERVER_MODULES)}_observers mosaic.fast={fast} stock_install={proof.get('stock_install')}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -s -m mosaic_opt.stock_design", description=__doc__.split("\n")[0])
    ap.add_argument("--driver", required=True, help="the kit driver (tools/public_design_run.py)")
    ap.add_argument("--pins", required=True, help="stock/PINS.json")
    ap.add_argument("driver_args", nargs=argparse.REMAINDER, help="the driver's own arguments (after --)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    with open(a.pins, "r", encoding="utf-8") as fh:
        pins = json.load(fh)
    kit_dir = os.path.dirname(os.path.dirname(os.path.abspath(a.driver)))    # the kit directory is found by its driver (tools/public_design_run.py)
    proof = env_proof(pins, kit_dir=kit_dir)
    proof.update(driver=a.driver, argv=list(a.driver_args), started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print(env_line(proof), file=sys.stderr, flush=True)

    if not proof["clean"]:
        print(f"{PREFIX} REFUSED: the stock environment is not clean", file=sys.stderr, flush=True)
        return 3
    args = [x for x in a.driver_args if x != "--"] if a.driver_args[:1] == ["--"] else list(a.driver_args)
    os.chdir(os.path.dirname(os.path.abspath(a.driver)))                 # the driver runs from its tools/ directory (driver.py)
    sys.argv = [a.driver] + args
    rc = 0
    try:
        # --- the stock call: the kit driver's own process body, every lever off (--weights torch, no --features-in unless the caller froze
        # the features, no P1 environment)
        runpy.run_path(a.driver, run_name="__main__")
    except SystemExit as e:
        rc = int(e.code or 0) if isinstance(e.code, int) or e.code is None else 1
    return rc


if __name__ == "__main__":
    sys.exit(main())

