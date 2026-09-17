"""The torch-conv chain behind the gate: every patched class / module attribute becomes gated(kit_fn, stock_fn); the shape-bound and model-bound stores the shape manager releases."""
from __future__ import annotations

import torch

from evo2_opt.kit import compose as MF
from evo2_opt.kit import gate as GT

M40, LG, K7 = MF.M40, MF.LG, MF.M40.K7
HyenaInferenceEngine, HyenaCascade, RMSNorm = K7.HyenaInferenceEngine, K7.HyenaCascade, K7.RMSNorm
ParallelGatedConvBlock, te, te_linear = LG.ParallelGatedConvBlock, LG.te, LG.te_linear
_eng = K7._eng
import vortex.model.model as _vmm                         # the gated model forward (a class attribute of vortex's StripedHyena)

CTR = MF.CTR
VERSIONS = dict(MF.VERSIONS)
BASE, GEMM_RUNG, GEMM_LEVERS, N_HYENA = MF.BASE, MF.GEMM_RUNG, MF.GEMM_LEVERS, MF.N_HYENA
_applied = {"version": None}
GATE = GT.Gate(CTR)
_ORIG_MODEL_FORWARD = _vmm.StripedHyena.forward
_installed: dict = {}                                     # attr name -> (owner, attr, kit_fn, wrapper)
SHAPES: GT.ShapeManager | None = None

expected_counts, fill_counts, counters = MF.expected_counts, MF.fill_counts, MF.counters


def _destroy_cufft_plans():
    """Free every cuFFT plan (handle + work area) of every device: the plans are keyed (kind, batch = B*D, n) — shape-bound."""
    cf = K7._CF
    if cf is None:
        return
    per = getattr(cf, "per_device", None)
    objs = list(per.values()) if per is not None else [cf]
    for o in objs:
        for key, (h, work, ws) in list(o.plans.items()):
            try:
                o.lib.cufftDestroy(int(h))
            except Exception:
                pass
        o.plans.clear()


def _release_e61_buffers():
    g = te_linear.general_gemm
    kit = getattr(g, "__kit_fn__", g)
    bufs = getattr(kit, "buffers", None)
    if bufs is not None:
        bufs.clear()


def _stores():
    return {"v6_pad_bufs": K7._pad_bufs.clear, "v6_spec_bufs": K7._spec_bufs.clear, "cufft_plans": _destroy_cufft_plans, "e61_out_buffers": _release_e61_buffers}


def _model_stores(model):
    """Model-bound kit state (not shape-bound): kit.base's L-keyed filter / spectrum caches and the W1 fp8 weight workspaces of the
    Hyena projections. Released before a PASSTHROUGH only, so the stock path has the device's memory; the next kit forward re-fills
    them exactly as the first forward of a process does (W1 fills, HCL filter-cache misses)."""
    blocks = LG.hyena_blocks(model)

    def clear_w1():
        for b in blocks:
            b.projections._fp8_workspaces.clear()
    return {"v6_filter_cache": K7._filter_cache.clear, "v6_H_cache": K7._H_cache.clear, "v6_kf_cache": K7._kf_cache.clear, "w1_fp8_workspaces": clear_w1}


def _devices(model):
    return sorted({p.device.index for p in model.parameters() if p.device.type == "cuda"})


def _device_name():
    return torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else ""


def _install(owner, attr, stock_fn, name):
    kit_fn = getattr(owner, attr)
    assert kit_fn is not stock_fn, f"{name}: the constituent kit did not patch this attribute"
    w = GT.gated(GATE, kit_fn, stock_fn, name)
    setattr(owner, attr, w)
    _installed[name] = (owner, attr, kit_fn, w)


def apply(model, version="v0"):
    global SHAPES
    assert _applied["version"] is None, f"kit already applied ({_applied['version']})"
    assert version in VERSIONS, version
    assert _vmm.StripedHyena.forward is _ORIG_MODEL_FORWARD, "StripedHyena.forward is already patched in this process"
    assert K7._ORIGINALS["hyena_forward"] is LG._ORIG["hyena_forward"], "the two constituents captured different block-forward originals"
    info = MF.apply(model, version)                       # the composed chain: kit.multidev (class patches) then kit.gemm_apply (fold/W1/E56/E61)
    _install(HyenaInferenceEngine, "parallel_fir", K7._ORIGINALS["parallel_fir"], "HyenaInferenceEngine.parallel_fir")
    _install(HyenaInferenceEngine, "parallel_iir", K7._ORIGINALS["parallel_iir"], "HyenaInferenceEngine.parallel_iir")
    _install(HyenaCascade, "compute_filter", K7._ORIGINALS["compute_filter"], "HyenaCascade.compute_filter")
    _install(_eng, "fftconv_func", K7._ORIGINALS["fftconv_func"], "vortex.model.engine.fftconv_func")
    _install(RMSNorm, "forward", K7._ORIGINALS["rms_fwd"], "RMSNorm.forward")
    _install(K7.AttentionBlock, "forward", K7._ORIGINALS["attn_fwd"], "AttentionBlock.forward")
    _install(ParallelGatedConvBlock, "forward", LG._ORIG["hyena_forward"], "ParallelGatedConvBlock.forward")
    _install(_vmm.StripedHyena, "stateless_forward", K7._ORIGINALS["stateless_forward"], "StripedHyena.stateless_forward")
    _install(te.Linear, "forward", LG._ORIG["te_linear_forward"], "transformer_engine.pytorch.Linear.forward")
    _install(te_linear, "general_gemm", LG._ORIG["general_gemm"], "transformer_engine.pytorch.module.linear.general_gemm")
    SHAPES = GT.ShapeManager(_stores(), CTR, devices=_devices(model), model_stores=_model_stores(model))
    _vmm.StripedHyena.forward = GT.forward_gate(_ORIG_MODEL_FORWARD, GATE, SHAPES)
    _applied["version"] = version
    info["gate"] = {"gated": sorted(_installed), "model_forward": "StripedHyena.forward (forward_gate)",
                    "shape_stores": sorted(_stores()), "model_stores": sorted(_model_stores(model)), "devices": _devices(model)}
    return info


def ensure_shape(B, L):
    """For callers that bypass the model forward (the pipelined 2-device forward, kit.pipelined, calls the blocks directly)."""
    assert SHAPES is not None, "ensure_shape before apply"
    return SHAPES.ensure(B, L)


PROCESS_WIDE_PATCHES = MF.PROCESS_WIDE_PATCHES + ("vortex.model.model.StripedHyena.forward",)


def is_unpatched(model=None):
    return MF.is_unpatched(model) and _vmm.StripedHyena.forward is _ORIG_MODEL_FORWARD and not _installed and _applied["version"] is None


