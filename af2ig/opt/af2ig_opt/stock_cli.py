"""The stock caller — the tree's one way of running the stock line, in a clean subprocess that proves its environment first.

    python -I -m af2ig_opt.stock_cli --pins <tree>/stock/PINS.json --det <0|1> \\
           --python <interpreter> --driver <AF2IG_DIR>/predict_pdb.py -- <the driver's own arguments>

Run by ``af2ig-opt pred --mode off`` (cli.py) with an environment it has already cleaned; this module is the child's own check of that
environment, written down before anything of the model is imported, then ``os.execv`` into the driver — the process becomes
``python -I -u predict_pdb.py <args>`` with no module of this package loaded (a fresh interpreter image; only the environment is
inherited). The proof (held in memory, printed as one line): the must-be-absent names present (``stock/PINS.json`` "stock_environment": the package's
switch, the kit's two environment levers, ``CUDA_MPS_*``), the recipe variable seen (``XLA_FLAGS``, allowed with ``--det``, recorded
otherwise), the ``sys.path`` entries inside the tree or the kit, the modules of this package loaded (this module and the package root
only), the interpreter and the argv handed to ``execv``; beside them the shared core's clean-process census (``opt_core.stock_proof.env_proof``:
kit directories on ``sys.path``, kit modules loaded, autoload finders armed, kit sitecustomize, torch, core modules beyond the proof).
A clean process prints ``[af2ig-opt stock] ENV-CLEAN ok: absent=… kit_modules=none kit_dirs=none no_user_site=… autoload=none
package_modules=… det=…`` (the core's ``clean_sentence``) and execs the driver; any forbidden name present, any tree entry on ``sys.path``,
any package module beyond the two, an armed autoload finder or any other violation of the core proof prints ``[af2ig-opt] stock_cli
REFUSED: …`` and ``[af2ig-opt stock] NOT STOCK: <the core's violations_sentence>; …; nothing ran`` and exits 3 — the stock arm never runs
on a doubtful environment.

The stock line's own arguments are the driver's I/O switches (``-pdbdir -outpdbdir -scorefilename -checkpoint_name -af2_dir -timers``);
no lever flag is ever on this route (cli.py builds the list; ``FORBIDDEN_FLAGS`` refuses one here).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

from opt_core import stock_proof as _core_proof                # the only core module a stock process may hold beside opt_core itself

PREFIX = "[af2ig-opt stock]"                                     # this caller's own two lines (ENV-CLEAN / NOT STOCK); the package's lines are [af2ig-opt]
FORBIDDEN_FLAGS = ("-fast", "-precompile", "-host_outputs", "-device_params", "-sort_by_length", "-flash_attn", "-trimul_chunk", "-subbatch", "-fused_triattn", "-opm_reassoc", "-fused_trimul")   # the lever flags, and the driver's optional npz dump (-dump_npz: an extra output no arm writes)
ALLOWED_MODULES = ("af2ig_opt", "af2ig_opt.stock_cli")
EXIT_REFUSED = 3


def stock_environment(pins: dict) -> Tuple[List[str], List[str], List[str]]:
    """(must-be-absent prefixes, recipe exceptions, pass-through names) from stock/PINS.json."""
    se = pins["stock_environment"]
    return list(se["must_be_absent_prefixes"]), list(se["recipe_exceptions"]), list(se["pass_through"])


def tree_sys_path(tree_dirs: List[str]) -> List[str]:
    """sys.path entries that are paths inside the tree or the kit (an editable install's finder hook is a name, not a path: ignored)."""
    hits = []
    for e in sys.path:
        if not e or not os.path.exists(e):
            continue
        ae = os.path.abspath(e)
        if any(ae == d or ae.startswith(d + os.sep) for d in tree_dirs):
            hits.append(e)
    return hits


def package_modules_loaded() -> List[str]:
    return sorted(m for m in sys.modules if m == "af2ig_opt" or m.startswith("af2ig_opt."))


def proof(pins: dict, det: int, python: str, argv_exec: List[str], tree_dirs: List[str]) -> dict:
    prefixes, exceptions, _ = stock_environment(pins)
    present = sorted(k for k in os.environ if any(k.startswith(p) for p in prefixes))
    recipe = {k: os.environ.get(k) for k in exceptions}
    loaded = package_modules_loaded()
    beyond = [m for m in loaded if m not in ALLOWED_MODULES]
    on_path = tree_sys_path(tree_dirs)
    lever_flags = [a for a in argv_exec if a in FORBIDDEN_FLAGS]
    # the core's clean-process proof beside this module's own checks (opt_core.stock_proof: the carried kit's directories not on sys.path and
    # none of its modules loaded, no autoload finder armed, no kit sitecustomize, no core module beyond the proof itself); the recipe
    # variable is not under a must-be-absent prefix, so the proof takes no carve-out
    core = _core_proof.env_proof(env_absent=prefixes, kit_dirs=[d for d in tree_dirs[1:]], module_prefixes=("af2ig_opt.",))
    return {"forbidden_present": present, "recipe": recipe, "det": det, "tree_on_sys_path": on_path, "package_modules_loaded": loaded,
            "package_modules_beyond_allowed": beyond, "lever_flags_in_argv": lever_flags, "python": python, "argv_exec": argv_exec, "pid": os.getpid(),
            "core": core, "ok": not present and not beyond and not on_path and not lever_flags and bool(core["ok"])}


def census_line(P: dict) -> str:
    """The clean-process census as one line, in the core's words (opt_core.stock_proof.clean_sentence over the core proof: the must-be-absent
    prefixes, kit modules loaded, kit directories on sys.path, the interpreter's no-user-site flag) plus the armed autoload finders, this
    package's modules held by the caller and the recipe switch — printed once, before ``execv`` into the driver."""
    core = P["core"]
    return (f"{PREFIX} ENV-CLEAN ok: {_core_proof.clean_sentence(core)} autoload={','.join(core['autoload_armed']) or 'none'} "
            f"package_modules={','.join(P['package_modules_loaded']) or 'none'} det={P['det']}")


def refusal_line(P: dict) -> str:
    """The refusal by name (exit 3, nothing runs): every violation class of the core proof (opt_core.stock_proof.violations_sentence) and
    this caller's own three checks."""
    return (f"{PREFIX} NOT STOCK: {_core_proof.violations_sentence(P['core'])}; "
            f"package modules beyond the caller {P['package_modules_beyond_allowed']}, tree dirs on sys.path {P['tree_on_sys_path']}, "
            f"lever flags in argv {P['lever_flags_in_argv']}; nothing ran")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="af2ig_opt.stock_cli")
    ap.add_argument("--pins", required=True); ap.add_argument("--det", type=int, default=0)
    ap.add_argument("--python", required=True); ap.add_argument("--driver", required=True)
    ap.add_argument("driver_args", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    args = a.driver_args[1:] if a.driver_args[:1] == ["--"] else a.driver_args
    with open(a.pins, encoding="utf-8") as fh:
        pins = json.load(fh)
    tree = os.path.dirname(os.path.dirname(os.path.abspath(a.pins)))
    kit_dir = os.path.abspath(os.environ.get("AF2IG_OPT_KIT") or os.path.join(tree, pins["kit"]["dir"]))
    argv_exec = [a.python, "-I", "-u", a.driver] + list(args)
    P = proof(pins, a.det, a.python, argv_exec, [os.path.abspath(tree), kit_dir])
    if not P["ok"]:
        print(f"[af2ig-opt] stock_cli REFUSED: forbidden={P['forbidden_present']} tree_on_sys_path={P['tree_on_sys_path']} "
              f"core_proof={_core_proof.violations_sentence(P['core']) if not P['core']['ok'] else 'ok'} "
              f"package_modules_beyond_allowed={P['package_modules_beyond_allowed']} lever_flags_in_argv={P['lever_flags_in_argv']}", file=sys.stderr, flush=True)
        print(refusal_line(P), file=sys.stderr, flush=True)
        return EXIT_REFUSED
    print(census_line(P), file=sys.stderr)
    sys.stdout.flush(); sys.stderr.flush()
    os.execv(a.python, argv_exec)
    return 1                                                   # not reached


if __name__ == "__main__":
    sys.exit(main())
