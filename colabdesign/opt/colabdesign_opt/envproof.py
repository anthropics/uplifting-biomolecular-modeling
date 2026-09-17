"""The environment proof both launchers run in the arm's own process, before jax or colabdesign is imported — the tree's ONE clean-process
proof (`opt_core.stock_proof.env_proof`: forbidden names under the must-be-absent prefixes, the carried kit's directories on `sys.path`, its
modules and the lever modules loaded, an armed autoload finder, a kit `sitecustomize`, core modules beyond the proof itself) beside this
engine's own two facts: no upstream module (`colabdesign`, `jax`, `jaxlib`, `haiku`) is loaded yet, and the variables `stock/PINS.json`
`stock_environment.reads` lists, with their values.

`prove(pins, arm)`: the stock arm is clean when the core proof is `ok` and no upstream module is loaded; the kit arm holds the package and the
core by construction (its levers are installed next), so its cleanliness is the core proof's facts minus the core-module rule — no forbidden
name, no carried-kit directory or module, no lever module before the installer runs, no upstream module. The record is the line
`[colabdesign-opt <arm>] ENV-CLEAN ok|FAIL: ...` on stderr; a failed proof stops the arm (exit 3) before any design runs.
"""
from __future__ import annotations

import os
import sys
from typing import Iterable

from opt_core import stock_proof as _core_proof

from .names import KIT_MODULES

UPSTREAM_ROOTS = ("colabdesign", "jax", "jaxlib", "haiku")
ARMS = ("stock", "kit")


def _upstream_loaded(modules: Iterable[str]) -> list:
    return sorted(m for m in modules if m.split(".")[0] in UPSTREAM_ROOTS)


def kit_dirs(tree: str) -> list:
    """The tree's directories no arm may have on `sys.path`: none — the kit is the installed package; the proof keeps the field (empty)."""
    return []


def prove(pins: dict, arm: str, tree: str, environ=None, modules=None, path=None) -> dict:
    """The proof record of this process for `arm` (stock | kit); `tree` = the colabdesign/ directory the pins file belongs to."""
    if arm not in ARMS:
        raise ValueError(f"arm {arm!r} is not one of {ARMS}")
    environ = os.environ if environ is None else environ
    modules = sys.modules if modules is None else modules
    path = sys.path if path is None else path
    se = pins.get("stock_environment") or {}
    prefixes = list(se.get("must_be_absent_prefixes") or [])
    if arm == "stock":                                       # jax's persistent-cache variables: the kit arm's own (lever compilecache sets them there), absent from stock's
        prefixes += [x for x in (se.get("stock_arm_absent_prefixes") or []) if x not in prefixes]
    core = _core_proof.env_proof(env_absent=prefixes, kit_dirs=kit_dirs(tree), module_prefixes=KIT_MODULES, environ=environ, modules=modules, path=path)
    upstream = _upstream_loaded(modules)
    if arm == "stock":
        clean = bool(core["ok"]) and not upstream
    else:                                                    # the kit arm: the package and the core are this process's own code; every other fact of the proof holds
        clean = not (core["forbidden_present"] or core["kit_modules_loaded"] or core["kit_dirs_on_path"] or core["kit_sitecustomize"] or upstream)
    return {"arm": arm, "prefixes_must_be_absent": prefixes, "forbidden_present": list(core["forbidden_present"]), "kit_modules_loaded": list(core["kit_modules_loaded"]),
            "upstream_modules_loaded": upstream, "kit_dirs_on_sys_path": list(core["kit_dirs_on_path"]), "core": core,
            "reads": sorted((se.get("reads") or {}).keys()), "values": {k: environ[k] for k in (se.get("reads") or {}) if k in environ},
            "python": sys.executable, "sys_path": list(path), "clean": clean}


def line(prefix: str, proof: dict) -> str:
    core = proof["core"]
    return (f"{prefix} ENV-CLEAN {'ok' if proof['clean'] else 'FAIL'}: {_core_proof.clean_sentence(core)} "
            f"forbidden_present={','.join(proof['forbidden_present']) or 'none'} upstream_loaded={','.join(proof['upstream_modules_loaded']) or 'none'} "
            f"kit_dirs_on_path={','.join(proof['kit_dirs_on_sys_path']) or 'none'} core_modules={','.join(core['core_modules_loaded']) or 'none'} "
            f"autoload_armed={','.join(core['autoload_armed']) or 'none'}")
