"""The vendored units under opt/forward/ — names, labels and paths. ``flashpairformer`` (the trunk levers and the levers add-on), ``PTX_TP`` (the multi-GPU line's unit) and ``DIT_FUSE`` (the diffusion transformer's fused elementwise kernels, lever
``sampler_fuse``). The identity of every file is the tree's git commit; this module locates the units and lists their files (presence only).
Shared kernels the FlashPairformer levers import by top-level name are served from the core (``stack.KERNEL_ROUTES``), not from the unit."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import _core  # noqa: F401
from opt_core import kernels as _core_kernels

FORWARD_DIRNAME = "forward"                                                   # opt/forward/: where the units live
CORE_SERVED_KERNELS = ("fpf_triatt_pro", "fpf_triatt_epi", "fpf_transition", "fpf_glue_v2", "fpf_mkpf", "fpf_triatt_k2b")   # imported by top-level name, served from opt_core/kernels
CORE_KERNELS_DIR = os.path.dirname(os.path.abspath(_core_kernels.__file__))   # opt_core/kernels: where the core-served packages live
CORE_RELEASED_KERNELS: Dict[str, Tuple[str, ...]] = {}   # package directory -> the per-kit data files kept under it (the pair-stack TriMul is opt_core.kernels.trimul by tier word: its cells are the core's, none are kept here)


@dataclass(frozen=True)
class Kit:
    name: str                                   # key and directory name under opt/forward/
    label: str                                  # the unit's own version label (FlashPairformer's is also the `kit=` token of env.sh's KIT_SPEC line)


KITS: Dict[str, Kit] = {
    "flashpairformer": Kit("flashpairformer", "FLASHPAIRFORMER_v0.6.3"),
    "PTX_TP": Kit("PTX_TP", "PTX_TP_ADDON_v0.1.13"),        # the multi-GPU line's unit (tp.py)
    "DIT_FUSE": Kit("DIT_FUSE", "DIT_FUSE_ADDON_v0"),      # the diffusion transformer's fused elementwise kernels (sampler_fuse.py, lever sampler_fuse)
}


def labels() -> Dict[str, str]:
    """{unit name: its version label}."""
    return {name: k.label for name, k in KITS.items()}


# ---------------------------------------------------------------------------------------------------------------- paths ----
def opt_dir() -> str:
    from . import stack
    return stack.opt_home()


def forward_dir() -> str:
    return os.path.join(opt_dir(), FORWARD_DIRNAME)


def kit_dir(name: str) -> str:
    return os.path.join(forward_dir(), KITS[name].name)


def kit_rel(name: str, path: str = "") -> str:
    """opt-relative path of a file inside a unit."""
    rel = f"{FORWARD_DIRNAME}/{KITS[name].name}"
    return f"{rel}/{path}" if path else rel


BYTECODE_DIRNAME = "__pycache__"                                              # the interpreter's cache beside imported kit files: not kit content


def tree_files(root: str) -> List[str]:
    """Every file under `root`, as sorted paths relative to it ('/'-separated), without the interpreter's bytecode caches
    (``__pycache__/``: written beside any kit file python imports; not kit content)."""
    out = []
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d != BYTECODE_DIRNAME)
        for f in fns:
            out.append(os.path.relpath(os.path.join(dp, f), root).replace(os.sep, "/"))
    return sorted(out)


# ------------------------------------------------------------------------------------------------------------ presence ----
def missing_kit_files(rels: Optional[List[str]] = None) -> List[str]:
    """Named presence problems for the given opt-relative paths (default: every file on disk under opt/forward/): a path outside
    opt/forward/, or a path not on disk. Presence only, never a hash."""
    if rels is None:
        rels = [f"{FORWARD_DIRNAME}/{p}" for p in tree_files(forward_dir())]
    problems: List[str] = []
    for rel in rels:
        p = os.path.join(opt_dir(), rel)
        if not rel.startswith(FORWARD_DIRNAME + "/"):
            problems.append(f"{rel}: outside opt/{FORWARD_DIRNAME}/")
        if not os.path.isfile(p):
            problems.append(f"{rel}: missing")
    return problems
