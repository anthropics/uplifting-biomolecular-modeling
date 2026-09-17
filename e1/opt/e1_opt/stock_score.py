"""The stock runner — mode ``off``: the upstream CLI (`python -m E1.tools.score ...`) in a clean subprocess, nothing from the kit on the path.

`e1-opt score --mode off` runs this module as the tool's process: `python -s -m e1_opt.stock_score --variant <v> --det <0|1> -- <the documented
E1.tools.score options>`, with every kit variable stripped from the environment (`stack.MUST_BE_ABSENT_PREFIXES`: E1_OPT*, E1_KIT*,
E1_VARIANT, MODEL_OPT*) and PYTHONPATH cleaned of any kit directory. Before importing torch or the upstream package, the process
proves its environment (`env_proof`): no forbidden variable is set, no kit module (`engines.*`) is loaded, no kit directory
is on sys.path — and prints the two contract lines:

    [e1-opt stock] ENV-CLEAN ok: absent=<vars> kit_modules=none kit_dirs=none
    STOCK_ENV_CHECK:                      followed by the `env | grep -E '^(E1_OPT|E1_KIT|E1_VARIANT|MODEL_OPT)'` output — empty on stock

then runs the upstream CLI's own `__main__` block verbatim (`runpy.run_module("E1.tools.score", run_name="__main__")` with the
documented argv), so the process IS the documented command: the same logging setup, `dist.setup_dist()`, the click command, the
tool's own scores.csv. The exit code is the tool's.

The KERNELS proof (accel.py) is armed before the tool runs and imports nothing of upstream's: at the tool's first scorer construction
(the model on the device, no forward yet) it reads the accelerators the process BOUND at upstream's dispatch sites and prints

    [e1-opt stock] KERNELS route=stock flash_attn=<word> hub_layernorm=<word> flex_attention=<word> site=stock upstream_says=<bool>

an accelerator the pinned
stock engages that this process did not engage (absent, or upstream's silent fallback taken) is NAMED there by its word — the stock runs
exactly the path upstream itself takes on this host; nothing is refused.

`--det 1` applies the deterministic recipe (det.py) in this process before the tool runs — the recipe's environment (already exported
by the calling process, set here again for a standalone run), the seeds and torch switches, and the kit's own autotune pin at `E1Scorer`
construction (a wrap of `E1Scorer.__init__` that calls `pins.apply_autotune_pin(size)` once, then the original). The pin function comes
from the kit's pins module loaded BY FILE PATH (`--pins-path`; the module has only standard-library imports) under the package's own
module name, so no `engines.*` package is importable or loaded here; the environment proof runs before that load and `--det 0`
never loads it. What this process sees is on the ENV-CLEAN line (`absent=` the forbidden variables proved absent, `kit_modules` /
`kit_dirs` = none); the variables the calling process stripped are printed once (`[e1-opt] stripped from the item environment: …`).
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from typing import List, Optional

PREFIX = "[e1-opt stock]"
from ._names import MUST_BE_ABSENT_PREFIXES, ABSENT_STARRED, KIT_MODULE_PREFIXES, KIT_DIR_MARKERS, PINS_MODULE_NAME, EXIT_NOT_CLEAN   # the one names module (no re-typing)
ENV_CHECK_HEADER = "STOCK_ENV_CHECK:"


# -------------------------------------------------------------------------------------------------------- the launch (parent side)
def clean_env(environ=None, extra_unset=(), keep=()) -> tuple:
    """(env, unset): a copy of `environ` without every variable under a forbidden prefix (and `extra_unset`), PYTHONPATH without kit
    directories. `unset` lists the variable names removed (the calling score prints them). `keep` names variables under a forbidden prefix to leave
    in place on purpose — the stock route passes none: its environment stays clean of every such name."""
    environ = os.environ if environ is None else environ
    env, unset = {}, []
    for k, v in environ.items():
        if (k.startswith(MUST_BE_ABSENT_PREFIXES) and k not in keep) or k in extra_unset:
            unset.append(k)
            continue
        env[k] = v
    pp = env.get("PYTHONPATH")
    if pp:
        keep = [p for p in pp.split(os.pathsep) if p and not _is_kit_dir(p)]
        if keep:
            env["PYTHONPATH"] = os.pathsep.join(keep)
        else:
            env.pop("PYTHONPATH", None)
    return env, sorted(unset)


def command(python: str, variant: str, det: int, pins_path: str, tool_args: List[str]) -> List[str]:
    """The stock runner's argv: `python -s -m e1_opt.stock_score --variant V --det D --pins-path P -- <E1.tools.score options>`."""
    return [python, "-s", "-m", "e1_opt.stock_score", "--variant", variant, "--det", str(int(det)), "--pins-path", pins_path, "--", *tool_args]


# ------------------------------------------------------------------------------------------------------------- the proof (child side)
def _is_kit_dir(p: str) -> bool:
    q = os.path.abspath(p) if p else ""
    return any(m in q for m in KIT_DIR_MARKERS)


def env_proof(environ=None, modules=None, path=None) -> dict:
    """{ok, present, absent, kit_modules, kit_dirs}: what the process finds in its own environment, sys.modules and sys.path."""
    environ = os.environ if environ is None else environ
    modules = sys.modules if modules is None else modules
    path = sys.path if path is None else path
    present = sorted(k for k in environ if k.startswith(MUST_BE_ABSENT_PREFIXES))
    kit_modules = sorted(m for m in modules if m.split(".")[0] in KIT_MODULE_PREFIXES)
    kit_dirs = sorted({p for p in path if p and _is_kit_dir(p)})
    absent = list(ABSENT_STARRED)                                                                          # the grep's names: prefixes starred
    return {"ok": not present and not kit_modules and not kit_dirs, "present": present, "absent": absent,
            "kit_modules": kit_modules, "kit_dirs": kit_dirs}


def env_check_lines(environ=None) -> List[str]:
    """The lines `env | grep -E '^(E1_OPT|E1_KIT|E1_VARIANT|MODEL_OPT)'` would print (KEY=VALUE, sorted); empty on a clean process."""
    environ = os.environ if environ is None else environ
    return [f"{k}={environ[k]}" for k in sorted(environ) if k.startswith(MUST_BE_ABSENT_PREFIXES)]


def env_clean_line(proof: dict) -> str:
    return f"{PREFIX} ENV-CLEAN ok: absent={','.join(proof['absent'])} kit_modules=none kit_dirs=none"


def env_failed_line(proof: dict) -> str:
    return (f"{PREFIX} ENV-CLEAN FAILED: present={','.join(proof['present']) or 'none'} kit_modules={','.join(proof['kit_modules']) or 'none'} "
            f"kit_dirs={','.join(proof['kit_dirs']) or 'none'}")


def load_pins_by_path(path: str):
    """The kit's pins module from its file, under the package's own module name (never `engines.*`)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(PINS_MODULE_NAME, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[PINS_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


def arm_pin(pins, size: str) -> None:
    """Wrap E1Scorer.__init__ so the kit's own autotune pin is applied once, before the tool's first forward (the model is loaded and
    on the device when E1Scorer is constructed)."""
    import E1.scorer as SC
    orig = SC.E1Scorer.__init__
    state = {"done": False}

    def __init__(self, model, *args, **kwargs):
        if not state["done"]:
            state["done"] = True
            pins.apply_autotune_pin(size)
        orig(self, model, *args, **kwargs)
    __init__.__wrapped__ = orig
    SC.E1Scorer.__init__ = __init__


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -s -m e1_opt.stock_score", description="the stock runner (mode off)")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--det", type=int, default=0, choices=(0, 1))
    ap.add_argument("--pins-path", default=None, help="the kit's pins module file (the KERNELS proof's expected versions; the autotune pin at --det 1)")
    ap.add_argument("tool_args", nargs=argparse.REMAINDER, help="-- then the E1.tools.score options")
    a = ap.parse_args(argv)
    if a.tool_args and a.tool_args[0] == "--":
        a.tool_args = a.tool_args[1:]
    return a


def main(argv: Optional[List[str]] = None) -> int:
    a = parse_args(argv)
    proof = env_proof()
    if not proof["ok"]:
        print(env_failed_line(proof), flush=True)
        print(ENV_CHECK_HEADER, flush=True)
        for ln in env_check_lines():
            print(ln, flush=True)
        return EXIT_NOT_CLEAN
    print(env_clean_line(proof), flush=True)
    print(ENV_CHECK_HEADER, flush=True)
    for ln in env_check_lines():                       # empty on a clean process
        print(ln, flush=True)
    if not a.pins_path or not os.path.isfile(a.pins_path):
        print(f"{PREFIX} NOT ACTIVE: --pins-path (the kit's pins module file) is needed for the KERNELS proof, got {a.pins_path!r}", flush=True)
        return EXIT_NOT_CLEAN
    from . import accel                                # stdlib-only module of this package (the KERNELS proof)
    route = accel.route_name("off")
    pins = load_pins_by_path(a.pins_path)             # under the package's own module name (no engines.* name is bound)
    if a.det:
        from . import det                               # stdlib-only module of this package
        det.apply_env(pins, 1)                         # before torch (the calling process exported the same entries)
        det.apply_torch(pins, 1)                       # imports torch: seeds + switches
        arm_pin(pins, a.variant)                       # imports E1.scorer (the tool's own import chain)
    accel.install("off", route, pins)                  # imports nothing of upstream's: fires at the tool's first scorer construction
    sys.argv = ["E1.tools.score", *a.tool_args]
    try:
        runpy.run_module("E1.tools.score", run_name="__main__", alter_sys=True)
    except SystemExit as e:
        code = e.code
        return 0 if code is None else (code if isinstance(code, int) else 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
