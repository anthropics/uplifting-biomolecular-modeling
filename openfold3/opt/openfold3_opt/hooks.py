"""The add-ons' hook files: how they are executed in-process and how their installation is evidenced.

Each add-on is a `sitecustomize.py` (modes.HOOK_DIR) that installs a meta-path finder on `sys.meta_path` when its switches are set:
the fast-inference kit's ``_LeverFinder`` (targets ``openfold3.core.model.primitives.linear`` and
``openfold3.core.model.structure.diffusion_module``; fast_inference/of3_levers/sitecustomize.py),
the trunk-kernels add-on's ``_Finder`` (``openfold3.projects.of3_all_atom.model``; trunk_kernels/of3t_hook/sitecustomize.py:59-104), the offload and tp
ports' ``_PostImportFinder`` / ``_TPFinder``, and the package's own hook (modes.PACKAGE_HOOKS: hooks/confhead/sitecustomize.py, a
``_PostImportFinder`` on ``openfold3.core.model.heads.prediction_heads`` that chains ``OPENFOLD3_OPT_CONFHEAD_CHAIN``). The add-ons' own route
is `PYTHONPATH`: Python imports the first `sitecustomize` it finds at start-up, and the hooks chain — the FlashPairformer hook executes the
next `sitecustomize.py` on `sys.path` (:11-24), the trunk-kernels hook executes the one under `OF3T_KIT_LEVERS` (:11-18). `run()` reproduces
that route inside a process that already started: the hook directories go on `sys.path` in the line's order and the FIRST hook file is
executed with `runpy`; the chain does the rest exactly as on the add-ons' own route.

`installed()` lists the finders present with the hook directory each was defined in (the class's source file), so the activation report
can say which hooks are in place; `expected()` says which the line should have produced; `missing()` compares the two.
"""
from __future__ import annotations

import inspect
import os
import runpy
import sys
from typing import Dict, List, Optional

from . import env as _env
from . import modes

FINDER_CLASSES = _env.HOOK_FINDER_CLASSES


def put_on_sys_path(dirs: List[str]) -> List[str]:
    """Insert the hook directories at the front of sys.path in order (first dir ends at index 0); returns sys.path's leading entries."""
    for d in reversed(dirs):
        if d in sys.path:
            sys.path.remove(d)
        sys.path.insert(0, d)
    return sys.path[:len(dirs)]


def run(entry_hook: str, kit: str) -> dict:
    """Execute the entry hook file (the first of the line) as the interpreter would at start-up; returns {file, run_name}."""
    run_name = f"openfold3_opt_hook_{kit}"
    runpy.run_path(entry_hook, run_name=run_name)
    return {"file": entry_hook, "run_name": run_name}


def _class_file(obj) -> Optional[str]:
    """The file the finder class was defined in: its find_spec's code object (the hook files are executed with runpy / exec, so the
    class's module name is not in sys.modules and inspect.getfile cannot resolve it)."""
    fn = getattr(type(obj), "find_spec", None)
    code = getattr(fn, "__code__", None)
    if code is not None and code.co_filename and os.path.isfile(code.co_filename):
        return os.path.abspath(code.co_filename)
    try:
        return os.path.abspath(inspect.getfile(type(obj)))
    except Exception:
        return None


def installed(home: Optional[str] = None) -> List[Dict[str, object]]:
    """Every add-on finder on sys.meta_path: {cls, file, kit, hook_dir, targets}."""
    home = home or _env.tree_home()
    out = []
    for f in sys.meta_path:
        cls = type(f).__name__
        if cls not in FINDER_CLASSES:
            continue
        file = _class_file(f)
        kit = None
        for k in list(modes.KITS) + list(modes.PACKAGE_HOOKS):
            dirs = modes.hook_dirs_of(home, k)                                  # the kit's hook directory (a package hook: its one directory)
            if file and any(file == os.path.join(hd, modes.HOOK_FILE) or file.startswith(hd + os.sep) for hd in dirs):
                kit = k
                break
        targets = getattr(type(f), "_targets", None) or getattr(type(f), "_target", None)
        out.append({"cls": cls, "file": file, "kit": kit, "hook_dir": os.path.dirname(file) if file else None,
                    "targets": sorted(targets) if isinstance(targets, (set, frozenset, list, tuple)) else targets})
    return out


def expected(res: modes.Resolution) -> List[str]:
    """The add-ons whose finder the line installs: every hook of the line (each hook installs its finder when one of its switches is set;
    the trunk-kernels hook installs its finder whenever its model timer is on, which it is by default)."""
    return list(res.hooks)


def missing(res: modes.Resolution, home: Optional[str] = None) -> List[str]:
    have = {h["kit"] for h in installed(home)}
    return [k for k in expected(res) if k not in have]


def targets_imported() -> List[str]:
    """The kit hook targets already in sys.modules (once any is, the hook route is impossible: late activation is refused by name)."""
    names = ("openfold3.core.model.primitives.linear", "openfold3.core.model.structure.diffusion_module", "openfold3.core.model.latent.pairformer",
             "openfold3.core.model.heads.prediction_heads", "openfold3.projects.of3_all_atom.model")
    return [n for n in names if n in sys.modules]
