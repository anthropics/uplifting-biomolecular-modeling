"""Installs the c2r-output-outside-the-io-buffer chains through the gate wrappers."""
from __future__ import annotations

from evo2_opt.kit import gate as GT
from evo2_opt.kit import compact_apply as M3
from evo2_opt.kit import hooks_apply as M4
from evo2_opt.kit import c2r as CO

M2, MF, LG, K7 = M4.M2, M4.MF, M4.LG, M4.K7
CTR = M4.CTR
VERSIONS = {"v0": list(M4.VERSIONS["v0"]) + ["C2R_OUT"]}       # C2R_OUT = the c2r output over the full-row spectrum storage (c2r.py)
BASE_KIT = "v40_full_4"
GATE = M4.GATE
N_HCL, N_HCM = M3.N_HCL, M3.N_HCM
_applied = {"version": None}
_installed = M4._installed
_vmm, _ORIG_MODEL_FORWARD = M4._vmm, M4._ORIG_MODEL_FORWARD
SHAPES = None
DEPS8 = None
_saved_wrappers = {}
counters = M4.counters
_stores, _model_stores = M4._stores, M4._model_stores


def expected_counts(version):
    assert version in VERSIONS, version
    return CO.expected8(M4.expected_counts("v0"), N_HCL, N_HCM)


def _parallel_iir_c8(self, *args, **kwargs):
    return CO.parallel_iir_c8(DEPS8, self, *args, **kwargs)


def _parallel_fir_c8(self, *args, **kwargs):
    return CO.parallel_fir_c8(DEPS8, self, *args, **kwargs)


_parallel_iir_c8_guarded = M3.M40.on_device(_parallel_iir_c8)
_parallel_fir_c8_guarded = M3.M40.on_device(_parallel_fir_c8)


def apply(model, version="v0"):
    """kit.hooks_apply's apply, then the compact chains re-pointed to kit.c2r's through the gate wrappers (the gate, passthrough and route
    table untouched); the out buffers live in Deps8 over the SAME base Deps (stores, caches, cuFFT object, counters)."""
    global SHAPES, DEPS8
    assert _applied["version"] is None, f"kit already applied ({_applied['version']})"
    assert version in VERSIONS, version
    info = M4.apply(model, "v0")
    SHAPES = M4.SHAPES
    DEPS8 = CO.Deps8(M3._DEPS)
    for name, kit_fn in (("HyenaInferenceEngine.parallel_iir", _parallel_iir_c8_guarded), ("HyenaInferenceEngine.parallel_fir", _parallel_fir_c8_guarded)):
        owner, attr, old_kit, w = _installed[name]
        _saved_wrappers[name] = (owner, attr, old_kit, w)
        w2 = GT.gated(GATE, kit_fn, w.__wrapped_stock__ if hasattr(w, "__wrapped_stock__") else M3._stock_of(w), name)
        setattr(owner, attr, w2)
        _installed[name] = (owner, attr, kit_fn, w2)
    M3._DEPS = DEPS8                                             # the shape manager's store release (clear_stores) releases the base Deps' stores
    _applied["version"] = version
    info["c2r_out"] = {"lever": "C2R_OUT: the compact chains' c2r into the full-row spectrum buffer's storage (an fp32 (B, D, n) view; that spectrum is consumed by then); the io buffer's tail zero by allocation, never rewritten (no per-call tail fill)",
                       "counters": ["c2r_dedicated_out"], "never": "c_tail_memset", "base_kit": BASE_KIT, "memory": "no bytes of its own (the view aliases the E60 full-row buffer)"}
    info["gate"]["c2r_out"] = True
    return info


def ensure_shape(B, L):
    return M4.ensure_shape(B, L)


PROCESS_WIDE_PATCHES = M4.PROCESS_WIDE_PATCHES


def is_unpatched(model=None):
    return M4.is_unpatched(model) and _applied["version"] is None and DEPS8 is None and not _saved_wrappers


