"""Route selection from the constructed model (vortex-kernels = Evo2(..., use_kernels=True); torch-conv = the constructor's defaults), the census the levers read, and the two install chains; the model forward's gate takes every (batch, length) on the kit path."""
from __future__ import annotations

import inspect

import torch
import vortex.model.engine as _eng
import vortex.model.model as _vmm
from vortex.model.engine import HyenaInferenceEngine
from vortex.model.layers import RMSNorm
from vortex.model.model import HyenaCascade, ParallelGatedConvBlock

import transformer_engine.pytorch as te
from transformer_engine.pytorch.module import linear as te_linear

from evo2_opt.kit import fold as M23      # the torch-conv route's installed chain (kit.fold -> kit.c2r_apply -> ... -> kit.base)
from evo2_opt.kit import gate as GT
from evo2_opt.kit import hooks as HK
from evo2_opt.kit import kfft as KF
from evo2_opt.kit import multidev as M40
from evo2_opt.kit import gemm_apply as LG
from evo2_opt.kit import hcs as HCS
from evo2_opt.kit import fp8emit as FP8E
from evo2_opt.kit import fir3rows as F3R
from evo2_opt.kit import rotary as ROT
from evo2_opt.kit import ltsel as LT
from evo2_opt.kit import graph as GR

A = LG.A                      # kit.gemm: lever table, W1 / E61 wrappers, census constants (read at call time)
K7 = M23.K7                   # kit.base: the kernels, caches and the vortex originals captured at import
M2 = M23.M2                   # kit.stores: gate wrappers, shape / model stores, the model forward's original
KitRefused = A.KitRefused
CTR = K7.CTR
GATE = M23.GATE               # ONE gate object for the process (created in kit.stores), whichever route is applied
VERSIONS = {"v0": ("route-resolved",)}
KERNEL_FLAGS = ("use_hcs_kernel", "use_hcm_kernel", "use_hcl_kernel")
KERNEL_SYMBOLS = {"use_hcs_kernel": "hcs_conv", "use_hcm_kernel": "hcm_fft_conv", "use_hcl_kernel": "hcl_fft_conv"}   # vortex/model/engine.py:15-17
MODEL_KEYS = {(4096, 32): "evo2_7b", (8192, 50): "evo2_40b"}   # (hidden_size, blocks) -> the model's name for the line (configs/evo2-7b-1m.yml L4-10, evo2-40b-1m.yml L4-10); another census is named by its numbers


from evo2_opt.kit.table import LEVERS_BY_ROUTE, NOT_APPLICABLE, ROUTES   # noqa: E402 — the lever names per route

GEMM_LEVERS = A.RUNGS["r5"]   # L2, W1, E50, E56, E61

_applied = {"version": None, "route": None, "facts": None}
_installed_k: dict = {}       # vortex-kernels route: name -> (owner, attr, kit_fn, gate wrapper)
FIR3_GEOMETRY = (64, 128)       # E7 fir3 (BLOCK_D, BLOCK_L) on this route: launch geometry only; per-element arithmetic unchanged


def _hcs_gate_follows(engine) -> bool:
    """The featurizer call belongs to an HCS block whose gate call E62 takes: vortex's hcs_conv dispatch (use_hcs_kernel, the kernel
    imported) on a layer in the model's hcs set."""
    return bool(getattr(engine, "use_hcs_kernel", False)) and _eng.hcs_conv is not None and getattr(engine, "layer_idx", None) in _state_k["hcs_layers"]

_state_k = {"shapes": None, "hcs_layers": frozenset()}


# ---------------------------------------------------------------------------------------------------------------- model facts / route
def facts(model) -> dict:
    """The route-deciding facts of a constructed vortex StripedHyena, read from the object: block census, hidden size, the three kernel flags
    (uniform across the Hyena engines or refused), the kernel symbols bound in vortex.model.engine, TE presence and the fp8 flag."""
    cfg = model.config
    hb, ab = LG.hyena_blocks(model), LG.attn_blocks(model)
    engines = [b.filter.engine for b in hb]
    flags = {}
    for k in KERNEL_FLAGS:
        vals = {bool(getattr(e, k, False)) for e in engines}
        if len(vals) != 1:
            raise KitRefused(f"engine flag {k} is not uniform across the {len(engines)} Hyena blocks: {sorted(vals)}")
        flags[k] = vals.pop()
    if len(set(flags.values())) != 1:
        raise KitRefused(f"mixed kernel flags {flags}: the exact chain knows the two stock routes {ROUTES} (all three on = use_kernels=True, all off = the constructor's defaults)")
    route = "vortex-kernels" if flags["use_hcs_kernel"] else "torch-conv"
    unbound = [KERNEL_SYMBOLS[k] for k, v in flags.items() if v and getattr(_eng, KERNEL_SYMBOLS[k], None) is None]
    if unbound:
        raise KitRefused(f"use_kernels=True but vortex.model.engine binds no {unbound} (vortex.ops import failed): the stock itself would fall back — not the STOCK route")
    n_blocks = len(list(model.blocks))
    key = MODEL_KEYS.get((int(cfg.hidden_size), n_blocks))
    n = {"hcl": len(cfg.get("hcl_layer_idxs", [])), "hcm": len(cfg.get("hcm_layer_idxs", [])), "hcs": len(cfg.get("hcs_layer_idxs", [])), "attn": len(cfg.get("attn_layer_idxs", []))}
    if n["hcl"] + n["hcm"] + n["hcs"] != len(hb) or n["attn"] != len(ab):
        raise KitRefused(f"block census {len(hb)} hyena / {len(ab)} attention != config layer lists {n}")
    devices = sorted({str(p.device) for p in model.parameters()})
    return {"route": route, "kernel_flags": flags, "model_key": key, "hidden": int(cfg.hidden_size), "n_blocks": n_blocks, "n_hyena": len(hb), "n_attn": len(ab), **{f"n_{k}": v for k, v in n.items()},
            "n_rms": 2 * n_blocks + 1, "fp8_input_projections": bool(cfg.get("use_fp8_input_projections", False)), "interleave": bool(cfg.get("interleave", False)), "devices": devices}


def configure(model) -> dict:
    """Read the facts once per process and set the GEMM-side adapter's census constants from them (kit.gemm reads N_HYENA / N_ATTN / HIDDEN /
    WQKV_STRIDES at check time, kit.multidev its block census; their module-level defaults are the 40B's). Idempotent for one model object."""
    f = _applied["facts"]
    if f is not None:
        if f["model_id"] != id(model):
            raise KitRefused("configure on a second model object in one process")
        return f
    f = dict(facts(model), model_id=id(model))
    A.N_HYENA, A.N_ATTN, A.HIDDEN = f["n_hyena"], f["n_attn"], f["hidden"]
    A.WQKV_STRIDES = (1, 3 * f["hidden"])
    M40.HIDDEN, M40.N_HCS, M40.N_HCM, M40.N_HCL, M40.N_ATTN = f["hidden"], f["n_hcs"], f["n_hcm"], f["n_hcl"], f["n_attn"]   # the torch-conv chain's launch conditions read the census of THIS model (7b: 9/9/9/5 over 4096; 40b: 14/14/14/8 over 8192)
    M40.N_BLOCKS = f["n_blocks"]; M40.N_RMS = 2 * f["n_blocks"] + 1; M40.WQKV_STRIDES = (1, 3 * f["hidden"])
    _applied["facts"] = f
    return f


def route_of_facts() -> str | None:
    f = _applied["facts"]
    return None if f is None else f["route"]


def configured_facts() -> dict | None:
    f = _applied["facts"]
    return None if f is None else {k: v for k, v in f.items() if k != "model_id"}


def levers(route_name=None) -> list:
    return list(LEVERS_BY_ROUTE[route_name or _applied["route"] or "torch-conv"])


def not_applicable(route_name=None) -> dict:
    return dict(NOT_APPLICABLE[route_name or _applied["route"] or "torch-conv"])


# ---------------------------------------------------------------------------------------------------------------- tables
def counters() -> dict:
    d = LG.counters()                          # kit.gemm's CTR + kit.base's CTR (fir3 / rms / filter cache / e50 / e56 count there)
    d.update(LT.counters())                    # T9's selected / kept_torch / rechecked / call tallies
    return d


# ---------------------------------------------------------------------------------------------------------------- vortex-kernels route: patches
def _parallel_fir_featurizer(self, fir_fn, u, weight, bias, L, dims, groups=None, gated_bias=False, column_split_hyena=False, dim_last=True,
                             fir_length=3, gate=False, inference_params=None, prefill_mode=None, padding_mask=None):
    """E7-fir3 on the vortex-kernels route: the un-gated 3-tap featurizer FIR (HyenaCascade.forward's first parallel_fir call: dim_last, no bias,
    F.conv1d, no generation state, no padding mask) in Triton under the current-device guard — handed raw to the HCL / HCM chains (kfft) or to an HCS block's gate call (E62) when they apply it
    inline, kit.fir3rows on a channels-first projection output (E64), else ``fir3_triton``; the gated inner FIRs go to kfft (hcm_fft_conv) or hcs
    (hcs_conv) when they take the call, and EVERY other call is vortex's original ``parallel_fir``."""
    if (not gate and dim_last and fir_length == 3 and bias is None and inference_params is None and padding_mask is None
            and fir_fn is torch.nn.functional.conv1d and u.dim() == 3 and weight.shape[-1] == 3 and not column_split_hyena):
        if self.layer_idx in KF.DEFERRED:
            return KF.defer_featurizer(u, weight), None                                       # the HCL / HCM chains apply the featurizer FIR inline (kfft)
        with M40._guard(u.device):
            if F3R.takes(u, weight):                                                            # E64: the channels-first projection output, FIR along its rows
                if _hcs_gate_follows(self):                                                     # E62 fold: an HCS block's featurizer FIR runs inside its gate call
                    CTR["hcs_featurizer_deferred"] += 1
                    return HCS.defer_featurizer(u, weight), None
                CTR["fir3_rows_fwd"] += 1
                return F3R.fir3_rows(u, weight), None
            CTR["fir3_triton_fwd"] += 1
            return K7.fir3_triton(u, weight, 0, *FIR3_GEOMETRY), None                          # (B, 3D, L) bf16, ascending tap order
    taps = HCS.deferred_taps(u)
    if KF.serves_fir(self, fir_fn, weight, bias, dim_last, fir_length, gate, inference_params, padding_mask, column_split_hyena):
        with M40._guard(u.device):
            return KF.parallel_fir_k(self, fir_fn, u, weight, bias, L, dims, groups=groups, gated_bias=gated_bias, column_split_hyena=column_split_hyena,
                                     dim_last=dim_last, fir_length=fir_length, gate=gate, inference_params=inference_params, prefill_mode=prefill_mode,
                                     padding_mask=padding_mask)                                # the hcm_fft_conv call: cached filter spectrum, persistent plans and buffers
    if HCS.takes(self, _eng.hcs_conv, gate, dim_last, fir_length, groups, bias, inference_params, padding_mask, column_split_hyena, u, weight):
        hidden_size = dims[0]                                                                   # E62: vortex's hcs_conv branch of parallel_fir, one kernel
        x2, x1, v = u.split([hidden_size, hidden_size, hidden_size], dim=1)
        if self.hyena_flip_x1x2:
            x1, x2 = x2, x1
        with M40._guard(u.device):
            if taps is not None:
                CTR["hcs_gate_conv_fold"] += 1
                return HCS.gate_conv_fold(x2, x1, v, taps, hidden_size, bool(self.hyena_flip_x1x2), weight), None
            CTR["hcs_gate_conv_fused"] += 1
            return HCS.gate_conv(x1, x2, v, weight), None
    assert taps is None, "a deferred featurizer input reached a parallel_fir call the kit chains do not serve"
    return K7._ORIGINALS["parallel_fir"](self, fir_fn, u, weight, bias, L, dims, groups=groups, gated_bias=gated_bias, column_split_hyena=column_split_hyena,
                                         dim_last=dim_last, fir_length=fir_length, gate=gate, inference_params=inference_params, prefill_mode=prefill_mode,
                                         padding_mask=padding_mask)


def check_conditions_kernels(model, f: dict) -> dict:
    """The vortex-kernels stock route's conditions, read from the live model (refused by name on drift); kit.gemm_apply's own check_conditions
    (TE fp8 projections, interleave True, empty fp8 workspaces, F.conv1d featurizer, nn.Linear out_filter_dense with bias, Wqkv layout, fp8
    accumulation mode, te 2.3.0 / vtx 1.1.0) runs inside its apply right after this."""
    import importlib.metadata as md
    if f["route"] != "vortex-kernels":
        raise KitRefused(f"route {f['route']} is not the vortex-kernels route")
    if not f["fp8_input_projections"]:
        raise KitRefused("use_fp8_input_projections is off: not the STOCK route (Transformer Engine FP8 input projections)")
    if HyenaInferenceEngine.parallel_fir is not K7._ORIGINALS["parallel_fir"] or HyenaCascade.compute_filter is not K7._ORIGINALS["compute_filter"] \
            or HyenaInferenceEngine.parallel_iir is not K7._ORIGINALS["parallel_iir"] or RMSNorm.forward is not K7._ORIGINALS["rms_fwd"] \
            or _vmm.StripedHyena.forward is not M2._ORIG_MODEL_FORWARD or _vmm.StripedHyena.stateless_forward is not K7._ORIGINALS["stateless_forward"] or not KF.is_unpatched() or not LT.is_unpatched():
        raise KitRefused("a vortex class attribute is already patched in this process (parallel_fir / parallel_iir / compute_filter / RMSNorm.forward / StripedHyena.forward / stateless_forward / T9 linears)")
    for b in LG.hyena_blocks(model):
        flt = b.filter
        if int(getattr(flt, "short_filter_length", 0)) != 3 or flt.short_filter_bias is not None or flt.fir_fn is not torch.nn.functional.conv1d:
            raise KitRefused(f"featurizer is not the 3-tap bias-free F.conv1d FIR (short_filter_length={getattr(flt, 'short_filter_length', None)})")
        if bool(getattr(flt, "column_split_hyena", False)):
            raise KitRefused("column_split_hyena is on")
    for m in model.modules():
        if isinstance(m, RMSNorm) and (not hasattr(m, "scale") or m.scale.dtype != torch.bfloat16):
            raise KitRefused("an RMSNorm has no bf16 scale parameter")
    pins = {"torch": torch.__version__.split("+")[0], "vtx": md.version("vtx"), "evo2": md.version("evo2")}   # recorded; a version off stock/PINS.json is named by the package's line, never refused here
    return {"pins": pins, "kernel_flags": f["kernel_flags"], "kernel_impls": {s: f"{getattr(_eng, s).__module__}.{getattr(_eng, s).__qualname__}" for s in KERNEL_SYMBOLS.values()}}


def _install_k(owner, attr, stock_fn, name):
    kit_fn = getattr(owner, attr)
    assert kit_fn is not stock_fn, f"{name}: not patched"
    w = GT.gated(GATE, kit_fn, stock_fn, name)
    setattr(owner, attr, w)
    _installed_k[name] = (owner, attr, kit_fn, w)


def _apply_kernels_route(model, f: dict) -> dict:
    assert LG.is_unpatched(model), "kit v40_gemm_1 is already applied in this process"
    info = {"route": f["route"], "conditions": check_conditions_kernels(model, f)}
    # 1. class patches of the Hyena side (no forward): L1b / L1d, E7-fir3 + the HCM chain, the HCL chain, E11
    KF.DEFERRED = frozenset(b.filter.layer_idx for b in model.blocks if isinstance(b, ParallelGatedConvBlock)
                             and (b.filter.fir_inner_filter_length is None or b.filter.fir_inner_filter_length >= 128))   # the HCL and HCM blocks: kfft's chains consume their featurizer input raw
    HyenaCascade.compute_filter = KF.compute_filter_k_guarded
    HyenaInferenceEngine.parallel_fir = _parallel_fir_featurizer
    HyenaInferenceEngine.parallel_iir = KF.parallel_iir_k_guarded
    RMSNorm.forward = M40.G["_rms_fused_forward"]
    K7.AttentionBlock.forward = M40.G["_attn_forward"]
    _vmm.StripedHyena.stateless_forward = K7._stateless_forward_pairs                         # E66: the blocks' closing adds carried into the next pre_norm
    # 2. the GEMM side (kit.gemm_apply: conditions on the pristine TE model, then the L2 fold / W1 / E50+E56 / E61), kit.fold's clone-free fold installed first
    M23._install_clone_free_fold()
    ginfo = LG.apply(model, GEMM_LEVERS)
    assert model.config.interleave is False and tuple(ginfo["levers"]) == tuple(GEMM_LEVERS), ginfo
    info["gemm"] = ginfo
    FP8E.CHANNELS_FIRST["on"] = True                                                             # E64: this route's featurizer (fir3rows) reads the transposed projection output
    _state_k["hcs_layers"] = frozenset(int(i) for i in (getattr(model.config, "hcs_layer_idxs", None) or []))   # E62 fold: the HCS blocks' featurizer runs in their gate call
    # 3. gate wrappers: each patched attribute becomes kit_fn <-> stock_fn by GATE.mode (kit.gate.gated)
    _install_k(HyenaInferenceEngine, "parallel_fir", K7._ORIGINALS["parallel_fir"], "HyenaInferenceEngine.parallel_fir")
    _install_k(HyenaInferenceEngine, "parallel_iir", K7._ORIGINALS["parallel_iir"], "HyenaInferenceEngine.parallel_iir")
    _install_k(HyenaCascade, "compute_filter", K7._ORIGINALS["compute_filter"], "HyenaCascade.compute_filter")
    _install_k(RMSNorm, "forward", K7._ORIGINALS["rms_fwd"], "RMSNorm.forward")
    _install_k(K7.AttentionBlock, "forward", K7._ORIGINALS["attn_fwd"], "AttentionBlock.forward")
    _install_k(ParallelGatedConvBlock, "forward", LG._ORIG["hyena_forward"], "ParallelGatedConvBlock.forward")
    _install_k(_vmm.StripedHyena, "stateless_forward", K7._ORIGINALS["stateless_forward"], "StripedHyena.stateless_forward")
    _install_k(te.Linear, "forward", LG._ORIG["te_linear_forward"], "transformer_engine.pytorch.Linear.forward")
    _install_k(te_linear, "general_gemm", LG._ORIG["general_gemm"], "transformer_engine.pytorch.module.linear.general_gemm")
    # 4. the shape manager (shape-bound: E61 out buffers; model-bound: the HCL filter cache + the W1 workspaces) and the model forward's gate
    shapes = GT.ShapeManager({**M2._stores(), **KF.stores()}, CTR, devices=M2._devices(model), model_stores={**M2._model_stores(model), **KF.model_stores()})
    _state_k["shapes"] = shapes
    _vmm.StripedHyena.forward = HK.forward_gate(M2._ORIG_MODEL_FORWARD, GATE, shapes)
    LG.reset_counters()
    info["gate"] = {"gated": sorted(_installed_k), "model_forward": "StripedHyena.forward (forward_gate: hooks rule, then every (batch, length) on the kit path)",
                    "shape_stores": sorted({**M2._stores(), **KF.stores()}), "model_stores": sorted({**M2._model_stores(model), **KF.model_stores()}), "devices": M2._devices(model)}
    info["levers"] = levers("vortex-kernels"); info["not_applicable"] = not_applicable("vortex-kernels")
    return info


# ---------------------------------------------------------------------------------------------------------------- torch-conv route
def _model_gate_parts():
    """(gate, manager) of the installed model-forward wrapper (the kit.gate / kit.hooks forward_gate closures name them ``gate`` / ``manager``)."""
    cur = _vmm.StripedHyena.forward
    cl = inspect.getclosurevars(cur).nonlocals
    return cl["gate"], cl["manager"]


def _apply_torch_conv_route(model, f: dict) -> dict:
    info = M23.apply(model, "v0")                                          # the installed chain: its launch conditions (kernels off, TE fp8 projections, the census configure() set) refuse by name on drift
    gate, manager = _model_gate_parts()
    assert gate is GATE, "the installed chain installed a different gate object"
    _vmm.StripedHyena.forward = HK.forward_gate(M2._ORIG_MODEL_FORWARD, GATE, manager)   # the hook rule first, then every (batch, length) on the kit path
    info["route"] = f["route"]
    info["levers"] = levers("torch-conv"); info["not_applicable"] = not_applicable("torch-conv")
    return info


# ---------------------------------------------------------------------------------------------------------------- apply / ensure_shape
def apply(model, version="v0") -> dict:
    """Configure from the model, then apply the route's chain; the model forward's gate carries this module's every-shape table either way."""
    assert _applied["version"] is None, f"kit already applied ({_applied['version']}, route {_applied['route']})"
    assert version in VERSIONS, version
    f = configure(model)
    info = _apply_kernels_route(model, f) if f["route"] == "vortex-kernels" else _apply_torch_conv_route(model, f)
    _install_e63()
    info["t9_modules"] = LT.install(model, wrap=lambda kit_fn, stock_fn: GT.gated(GATE, kit_fn, stock_fn, "T9"))   # T9: the bf16 nn.Linear GEMMs
    _install_g1(model)
    _applied["version"], _applied["route"] = version, f["route"]
    info["version"] = version; info["facts"] = configured_facts()
    return info


def _install_e63() -> None:
    """E63 on either route: the Hyena block's proj_norm gated kit <-> stock; its FP8 input buffers released with the shape-bound stores.
    R1 beside it: the rotary table update gated the same way."""
    gate, manager = _model_gate_parts()
    ParallelGatedConvBlock.proj_norm = GT.gated(gate, FP8E.proj_norm, FP8E.ORIG["proj_norm"], "ParallelGatedConvBlock.proj_norm")
    K7.FOLDC["pair_proj_norm"] = True                                                            # E66: this proj_norm takes the previous block's pair
    manager.stores["e63_fp8_input_rows"] = FP8E.release
    ROT.LinearlyScaledRotaryEmbedding._update_cos_sin_cache = GT.gated(gate, ROT._update_cos_sin_cache, ROT.ORIG["update"], "LinearlyScaledRotaryEmbedding._update_cos_sin_cache")


def _install_g1(model) -> None:
    """G1 on either route: the model forward's kit path runs through the graph-replay wrapper (inside the gate, so passthrough calls never
    reach it); its graphs are released with the shape-bound stores whose buffers they captured."""
    gate, manager = _model_gate_parts()
    devices = sorted({str(p.device) for p in model.parameters()})
    _vmm.StripedHyena.forward = HK.forward_gate(GR.graphed(M2._ORIG_MODEL_FORWARD, gate, devices), gate, manager)
    manager.stores["g1_forward_graphs"] = GR.release


def ensure_shape(B, L):
    """For callers that bypass the model forward (the pipelined 2-device forward calls the blocks directly)."""
    if _applied["route"] == "vortex-kernels":
        assert _state_k["shapes"] is not None, "ensure_shape before apply"
        return _state_k["shapes"].ensure(B, L)
    return M23.ensure_shape(B, L)


PROCESS_WIDE_PATCHES = M23.PROCESS_WIDE_PATCHES


def is_unpatched(model=None) -> bool:
    return M23.is_unpatched(model) and not _installed_k and _applied["version"] is None and _vmm.StripedHyena.forward is M2._ORIG_MODEL_FORWARD and KF.is_unpatched()


