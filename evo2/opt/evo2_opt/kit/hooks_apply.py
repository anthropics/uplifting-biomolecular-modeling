"""Re-creates the model forward's gate with the hook rule first."""
from __future__ import annotations

from evo2_opt.kit import gate as GT
from evo2_opt.kit import compact_apply as M3
from evo2_opt.kit import hooks as HK

M2, MF, LG, K7 = M3.M2, M3.MF, M3.LG, M3.K7
CTR = M3.CTR
VERSIONS = {"v0": list(M3.VERSIONS["v0"]) + ["HOOKS_PT"]}   # HOOKS_PT = the forward-hook passthrough / refusal (hooks.py)
BASE_KIT = "v40_full_3c"
GATE = M3.GATE
_applied = {"version": None}
_installed = M3._installed
_vmm, _ORIG_MODEL_FORWARD = M3._vmm, M3._ORIG_MODEL_FORWARD
SHAPES = None

expected_counts, fill_counts, counters = M3.expected_counts, M3.fill_counts, M3.counters
_stores, _model_stores = M3._stores, M3._model_stores


def apply(model, version="v0"):
    """kit.compact_apply's apply (the composed chain, the gate, the compact chains), then the model forward's gate re-created with the
    hook rule first (hooks.forward_gate: registered forward hooks pass through or are refused by name), then the gate's own rule."""
    global SHAPES
    assert _applied["version"] is None, f"kit already applied ({_applied['version']})"
    assert version in VERSIONS, version
    info = M3.apply(model, "v0")
    SHAPES = M3.SHAPES
    _vmm.StripedHyena.forward = HK.forward_gate(_ORIG_MODEL_FORWARD, GATE, SHAPES)   # hooks passthrough / refusal first, then the route table
    _applied["version"] = version
    info["gate"]["hooks"] = "forward hooks registered on the model (Evo2.forward(return_embeddings=True)) -> passthrough counted passthrough:hooks; a hook on a folded projection -> KitRefusedHooks (stock-only)"
    return info


def ensure_shape(B, L):
    return M3.ensure_shape(B, L)


PROCESS_WIDE_PATCHES = M3.PROCESS_WIDE_PATCHES


def is_unpatched(model=None):
    return M3.is_unpatched(model) and _applied["version"] is None


