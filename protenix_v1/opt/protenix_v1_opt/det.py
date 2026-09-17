"""The deterministic recipe (`pred --det 1`): the kit's DET recipe applied in-process, nothing edited on disk.

Three things, applied to the running process (nothing in the installed package is edited): the two detpatch files
`lib/detpatch/det_segment_reduce.py` and `lib/detpatch/scatter_utils.py` are loaded, byte for byte, as
the modules `protenix.utils.det_segment_reduce` and `protenix.utils.scatter_utils` before the model package imports either (the stock
`protenix/utils/__init__.py` is empty; `protenix.model.utils` is the importer), `CUBLAS_WORKSPACE_CONFIG=:4096:8` and
`PROTENIX_DET_SCATTER=1` are exported, and the warn-only deterministic-algorithms switch is set
(`torch.use_deterministic_algorithms(True, warn_only=True)`). Refused by name when `protenix.utils.scatter_utils` is already imported (the swap would be silent
otherwise) — activate before any protenix model import, which the `runner` trigger of the .pth guarantees for the stock CLI.
Tier 1 (`exact`) equals stock bit for bit ONLY under this recipe, and stock under this recipe is not stock's default numerics: a
comparison runs both arms under it.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import Optional

from . import kit as K

ENV = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PROTENIX_DET_SCATTER": "1"}
MODULES = ("protenix.utils.det_segment_reduce", "protenix.utils.scatter_utils")        # in load order; K.DETPATCH_RELPATHS holds the files
STOCK_MODULE = "protenix.utils.scatter_utils"
_REPORT: Optional[dict] = None


def _load_as(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    pkg = sys.modules.get(name.rpartition(".")[0])
    if pkg is not None:
        setattr(pkg, name.rpartition(".")[2], mod)
    return mod


def installed() -> Optional[dict]:
    return _REPORT


def install(kit_dir: Optional[str] = None) -> dict:
    """Apply the recipe to this process (idempotent). Returns {"installed", "files": {relpath: sha256}, "env": {...}, "det_scatter_active", "reason"?}."""
    global _REPORT
    if _REPORT is not None:
        return _REPORT
    kit_dir = kit_dir or K.kit_home()
    if STOCK_MODULE in sys.modules:
        raise RuntimeError(f"det refused: {STOCK_MODULE} is already imported (the recipe must precede the model package)")
    files = {rel: K.sha256_file(os.path.join(kit_dir, rel)) for rel in K.DETPATCH_RELPATHS}
    for k, v in ENV.items():
        os.environ[k] = v
    importlib.import_module("protenix.utils")           # the empty package; its two submodules are bound below
    for name, rel in zip(MODULES, K.DETPATCH_RELPATHS):
        _load_as(name, os.path.join(kit_dir, rel))
    active = bool(getattr(sys.modules[STOCK_MODULE], "_DET_SCATTER", False))
    if not active:
        raise RuntimeError("det refused: the detpatch scatter_utils reports _DET_SCATTER=False after the swap")
    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    _REPORT = {"installed": True, "files": files, "env": dict(ENV), "det_scatter_active": active, "warn_only": True}
    return _REPORT
