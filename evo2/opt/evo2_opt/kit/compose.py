"""The torch-conv chain's composition: the device-guarded base levers, then the GEMM-side levers."""
from __future__ import annotations

import torch

from evo2_opt.kit import gemm as AG
from evo2_opt.kit import multidev as M40
from evo2_opt.kit import gemm_apply as LG

CTR = M40.CTR                                            # kit.base's counter (the base levers + E50/E56 count here)
K7 = M40.K7                                              # kit.base (kernels, caches, per-device cuFFT plans)
BASE = "v5"                                              # the kit.multidev lever set composed (multidev.VERSIONS)
GEMM_RUNG = "r5"                                         # the kit.gemm rung composed (gemm.RUNGS)
GEMM_LEVERS = tuple(AG.RUNGS[GEMM_RUNG])
VERSIONS = {"v0": list(M40.VERSIONS[BASE]) + list(GEMM_LEVERS)}
N_HYENA = AG.N_HYENA
_applied = {"version": None}


def expected_counts(version):
    """Per forward AFTER the cache-fill forward: kit.multidev's and kit.gemm's tables (disjoint keys)."""
    assert version in VERSIONS, version
    mine, theirs = M40.expected_counts(BASE), AG.expected_counts_static(GEMM_LEVERS)
    assert not set(mine) & set(theirs), set(mine) & set(theirs)
    return {**mine, **theirs}


def fill_counts(version):
    """The cache-fill (first) forward: W1 fills instead of hits; everything else as per forward."""
    exp = expected_counts(version)
    if "w1_weight_cache_hit" in exp:
        exp = {**{k: v for k, v in exp.items() if k != "w1_weight_cache_hit"}, "w1_weight_cache_fill": N_HYENA}
    return exp


def counters() -> dict:
    return LG.counters()


def apply(model, version="v0"):
    assert _applied["version"] is None, f"kit already applied ({_applied['version']})"
    assert version in VERSIONS, version
    assert LG.is_unpatched(model), "kit v40_gemm_1 is already applied in this process"
    info = M40.apply(model, BASE)                        # 1. kit.multidev: conditions on the pristine model + class patches (no forward)
    ginfo = LG.apply(model, GEMM_LEVERS)                 # 2. kit.gemm_apply: conditions (interleave True, workspaces empty) + fold/W1/E56/E61
    assert model.config.interleave is False and tuple(ginfo["levers"]) == GEMM_LEVERS, ginfo
    _applied["version"] = version; LG.reset_counters()
    info["gemm"] = ginfo; info["version"] = version; info["levers"] = list(VERSIONS[version]); info["apply_order"] = ["v40:" + BASE, "v40_gemm_1:" + GEMM_RUNG]
    return info


PROCESS_WIDE_PATCHES = M40.PROCESS_WIDE_PATCHES + LG.PROCESS_WIDE_PATCHES


def is_unpatched(model=None):
    return M40.is_unpatched() and LG.is_unpatched(model) and _applied["version"] is None


