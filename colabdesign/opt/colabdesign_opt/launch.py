"""The arm launcher — one body for both arms; `stock_launch.py` and `kit_launch.py` are its two entry points.

    python -s -m colabdesign_opt.stock_launch --pins <stock/PINS.json> -- <stock_design.py args>
    python -s -m colabdesign_opt.kit_launch   --pins <stock/PINS.json> --mode fast -- <stock_design.py args>

In the arm's own process, before jax or colabdesign is imported: the environment proof (envproof.py: no kit or lever variable may be set in
either arm — a mode is its lever set, nothing else selects a lever), then for the kit arm the levers of the mode through the package's ONE
installer (`levers.install(mode)`: the mode's set in registry order — each lever prints its LEVER line; a lever that cannot engage here steps
aside BY NAME (`state=skipped reason=cannot_run|no_attention_kernel`) and the arm runs the rest of the set — the kit rule). Then `runpy.run_path(stock_design.py, run_name="__main__")` — the same script bytes for both arms. Nothing
here writes a file: the proof, the levers and the lever evidence are the printed lines. Exit 3 when the proof fails or the mode word is
not one the kit route serves (configuration, never a lever); else the script's exit code.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time

from ._core_gate import gate as _core_gate               # the tree's kit_template copy (standard library only): the core pin before any opt_core import

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_design.py")
T_START = time.perf_counter()                          # the arm's clock origin: the launcher's start (stock_design.py reads it; `ready t=` counts from here in both arms)
PINS_PATH = None                                       # set from --pins before the script runs (stock_design.py checks the installed stock files against the pins)
from .names import KIT_ROUTE_MODES

ARMS = ("stock", "kit")


def parse_args(arm: str, argv=None):
    ap = argparse.ArgumentParser(prog=f"python -s -m colabdesign_opt.{arm}_launch", description=__doc__.split("\n")[0])
    ap.add_argument("--pins", required=True, help="stock/PINS.json")
    if arm == "kit":
        ap.add_argument("--mode", required=True, metavar="|".join(KIT_ROUTE_MODES),
                        help="the kit-route mode word whose levers this arm installs (modes.py; its subtractive form <mode>-no-<lever> drops levers) - validated by the installer, refused by name")
    ap.add_argument("script_args", nargs=argparse.REMAINDER, help="stock_design.py's own arguments (after --)")
    return ap.parse_args(argv)


def main(arm: str, argv=None) -> int:
    _core_gate(__file__, tag="colabdesign-opt")                                  # NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: … -> SystemExit(3), before the proof imports opt_core.stock_proof
    from . import envproof                                                         # the stock arm imports nothing else of the package or the core before its proof (opt_core.stock_proof's core-module rule)
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, not {arm!r}")
    a = parse_args(arm, argv)
    prefix = f"[colabdesign-opt {arm}]"
    with open(a.pins, "r", encoding="utf-8") as fh:
        pins = json.load(fh)
    global PINS_PATH
    PINS_PATH = os.path.abspath(a.pins)
    proof = envproof.prove(pins, arm, os.path.dirname(os.path.dirname(PINS_PATH)))   # neither arm may hold a kit or lever variable or module; the stock arm no core module beyond the proof
    print(envproof.line(prefix, proof), file=sys.stderr, flush=True)
    if not proof["clean"]:
        print(f"{prefix} REFUSED: the arm's environment is not clean", file=sys.stderr, flush=True)
        return 3
    if arm == "kit":
        from . import levers
        try:
            info = levers.install(a.mode)                                       # imports colabdesign's model modules and patches before any model exists
        except levers.LeverError as e:
            print(f"{prefix} REFUSED: {e}", file=sys.stderr, flush=True)
            return 3
        skipped = info.get("skipped") or {}                                        # the levers that stepped aside by name (their skipped LEVER line is printed): the arm runs the rest
        missing = [l for l in info["levers"] if l not in info["levers_installed"] and l not in skipped]
        if missing:                                                                  # neither installed nor stepped aside: an installer defect, refused rather than run unnamed
            print(f"{prefix} REFUSED: mode {a.mode}: lever(s) {','.join(missing)} neither installed nor stepped aside (installed={','.join(info['levers_installed']) or 'none'} skipped={','.join(skipped) or 'none'})", file=sys.stderr, flush=True)
            return 3
    args = a.script_args[1:] if a.script_args[:1] == ["--"] else list(a.script_args)
    sys.argv = [SCRIPT] + args
    rc = 0
    try:
        # --- the design: the script's own process body (both arms)
        runpy.run_path(SCRIPT, run_name="__main__")
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return rc
