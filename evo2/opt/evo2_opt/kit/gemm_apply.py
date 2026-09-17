"""Applies the GEMM-side levers (L2 fold into the TE master weight, W1, E50, E56, E61) to the constructed model after checking the TE route's conditions on the pristine model."""
from __future__ import annotations

import torch
import transformer_engine.pytorch as te
import transformer_engine.pytorch.module.base as te_base
import transformer_engine.pytorch.module.linear as te_linear
import vortex.model.layers as vl
from vortex.model.model import AttentionBlock, ParallelGatedConvBlock

from evo2_opt.kit import gemm as A
from evo2_opt.kit import base as V6

CTR = A.CTR
_ORIG = {"te_linear_forward": te.Linear.forward, "hyena_forward": ParallelGatedConvBlock.forward, "general_gemm": te_linear.general_gemm}
_applied: dict = {"levers": None, "model_id": None}


def hyena_blocks(model):
    return [b for b in model.blocks if isinstance(b, ParallelGatedConvBlock)]


def attn_blocks(model):
    return [b for b in model.blocks if isinstance(b, AttentionBlock)]


def check_conditions(model) -> dict:
    """The TE fp8 stock route's conditions at kit.gemm's census constants (every value read from the live model, refused by name on drift)."""
    import importlib.metadata as md
    cfg = model.config
    hb, ab = hyena_blocks(model), attn_blocks(model)
    if (len(hb), len(ab)) != (A.N_HYENA, A.N_ATTN):
        raise A.KitRefused(f"block census {len(hb)} hyena / {len(ab)} attention != {A.N_HYENA} / {A.N_ATTN}")
    if cfg.get("use_fp8_input_projections", False) is not True or not vl.HAS_TE:
        raise A.KitRefused("built for the TE route with fp8 input projections only")
    if cfg.interleave is not True:
        raise A.KitRefused("config.interleave must be True before the fold")
    for b in hb:
        p = b.projections
        if not (isinstance(p, vl.TELinear) and isinstance(p, te.Linear)):
            raise A.KitRefused(f"projections is {type(p).__module__}.{type(p).__name__}, not vortex TELinear(te.Linear)")
        if not p.use_fp8_input_projections or p.use_bias or not _empty_bias(p):
            raise A.KitRefused("projections: fp8 flag off or a bias is present")
        if p.weight.dtype != torch.bfloat16 or p.weight.dim() != 2 or not p.weight.is_contiguous():
            raise A.KitRefused(f"projections.weight {p.weight.dtype} {tuple(p.weight.shape)} contiguous={p.weight.is_contiguous()}")
        if p._fp8_workspaces:
            raise A.KitRefused("projections already hold an fp8 workspace (W1 applied, or TE cached outside this kit)")
        if type(b.out_filter_dense) is not torch.nn.Linear or b.out_filter_dense.bias is None:
            raise A.KitRefused("out_filter_dense is not torch.nn.Linear with a bias")
        if b.filter.short_filter_bias is not None or b.filter.fir_fn is not torch.nn.functional.conv1d:
            raise A.KitRefused("short_filter_bias present or fir_fn is not F.conv1d")
        if b.print_activations:
            raise A.KitRefused("print_activations is on")
    for b in ab:
        w = b.inner_mha_cls.Wqkv
        if type(w) is not torch.nn.Linear or w.bias is not None or tuple(w.weight.stride()) != A.WQKV_STRIDES:
            raise A.KitRefused(f"Wqkv layout {tuple(w.weight.stride())} / bias {w.bias is not None}")
    rec = next(iter(hb)).projections.fp8_recipe
    acc = A.accumulation_readback(model)
    if acc["effective_use_split_accumulator"] is not False or acc["base._2X_ACC_FPROP"] is not False or acc["linear._2X_ACC_FPROP"] is not False:
        raise A.KitRefused(f"fp8 forward accumulation mode drifted from the stock's fast-accumulation call: {acc}")
    info = {"accumulation_readback": acc, "vtx": md.version("vtx"), "transformer_engine": md.version("transformer_engine_cu12"), "torch": torch.__version__,
            "evo2": md.version("evo2"), "projections_class": f"{vl.TELinear.__module__}.{vl.TELinear.__name__}",
            "te_linear_class": f"{te.Linear.__module__}.{te.Linear.__name__}", "fp8_recipe": repr(rec)[:300],
            "n_hyena": len(hb), "n_attn": len(ab), "interleave": bool(cfg.interleave),
            "devices": sorted({str(next(b.parameters()).device) for b in model.blocks})}
    return info


def _empty_bias(p) -> bool:
    """TE registers an EMPTY tensor (not None) as ``bias`` when ``bias=False`` (linear.py L1101-1104)."""
    b = getattr(p, "bias", None)
    return b is None or (isinstance(b, torch.Tensor) and b.numel() == 0)


def fold_interleave_te(model) -> None:
    """kit.base's fold on the TE model: the empty ``bias`` tensors are taken out of the modules for the call (the fold
    permutes a bias only when it is not None) and put back; ``config.interleave`` is False afterwards."""
    hb = hyena_blocks(model)
    empties = {}
    for b in hb:
        bias = getattr(b.projections, "bias", None)
        if isinstance(bias, torch.Tensor) and bias.numel() == 0:
            empties[id(b)] = bias
            b.projections.bias = None
    try:
        V6.fold_interleave(model)
    finally:
        for b in hb:
            if id(b) in empties:
                b.projections.bias = empties[id(b)]
    for b in hb:                                     # the fold keeps the pre-fold tensors for unfold: held on the host
        st = V6._fold_state.get(id(b))
        if st:
            V6._fold_state[id(b)] = tuple(t.to("cpu") if t is not None else None for t in st)


def _devctx(fn):
    """Triton launches on the CURRENT device/stream: enter the input's device for the blocks placed on cuda:1."""
    def forward(self, u, *args, **kwargs):
        with torch.cuda.device((u[0] if isinstance(u, tuple) else u).device):                  # u: the block input or the carried pair (E66)
            return fn(self, u, *args, **kwargs)
    forward.__wrapped_stock__ = _ORIG["hyena_forward"]
    return forward


def apply(model, levers=A.RUNGS["r4"]) -> dict:
    """Apply ``levers`` (a subset of A.LEVERS, in A.LEVERS order) to the loaded vortex model. Incremental: levers
    already applied stay; the returned record names what is live. CTR is cleared."""
    lv = [l for l in A.LEVERS if l in set(levers)]
    if set(levers) - set(A.LEVERS):
        raise A.KitRefused(f"unknown levers {sorted(set(levers) - set(A.LEVERS))}")
    live = set(_applied["levers"] or ())
    if not live:
        info = check_conditions(model)
        _applied["model_id"] = id(model)
    else:
        info = {"incremental_over": sorted(live)}
        if _applied["model_id"] != id(model):
            raise A.KitRefused("apply on a second model object in one process")
    if "L2" in lv and "L2" not in live:
        fold_interleave_te(model)
        if model.config.interleave is not False:
            raise A.KitRefused("fold did not clear config.interleave")
    if "W1" in lv and "W1" not in live:
        for b in hyena_blocks(model):
            b.projections._v40_w1 = True
        te.Linear.forward = A.te_forward_with_weight_cache(_ORIG["te_linear_forward"], CTR)
    want_e56, want_e50 = "E56" in lv, "E50" in lv
    if want_e56 and "E56" not in live:
        ParallelGatedConvBlock.forward = _devctx(V6._e56_hyena_forward)
    elif want_e50 and not want_e56 and "E50" not in live:
        ParallelGatedConvBlock.forward = _devctx(V6._e50_hyena_forward)
    if "E61" in lv and "E61" not in live:
        te_linear.general_gemm = A.general_gemm_with_out_buffer(_ORIG["general_gemm"], CTR)
    _applied["levers"] = tuple(l for l in A.LEVERS if l in live | set(lv))
    CTR.clear(); V6.CTR.clear()
    return {"kit": A.KIT, "levers": list(_applied["levers"]), "expected_per_forward": expected_counts(_applied["levers"]), **info}


def expected_counts(levers=None) -> dict:
    return A.expected_counts_static(levers if levers is not None else (_applied["levers"] or ()))


def counters() -> dict:
    d = {k: int(v) for k, v in CTR.items()}
    for k, v in V6.CTR.items():                      # E50/E56 count in kit.base's own counter
        d[k] = d.get(k, 0) + int(v)
    return d


def reset_counters() -> None:
    CTR.clear(); V6.CTR.clear()


def is_unpatched(model=None) -> bool:
    ok = te.Linear.forward is _ORIG["te_linear_forward"] and ParallelGatedConvBlock.forward is _ORIG["hyena_forward"] and not V6._fold_state \
        and te_linear.general_gemm is _ORIG["general_gemm"]
    if model is not None:
        ok = ok and all(not b.projections._fp8_workspaces and not hasattr(b.projections, "_v40_w1") for b in hyena_blocks(model)) \
            and model.config.interleave is True
    return bool(ok)


PROCESS_WIDE_PATCHES = ("transformer_engine.pytorch.Linear.forward", "vortex.model.model.ParallelGatedConvBlock.forward",
                        "transformer_engine.pytorch.module.linear.general_gemm (E61)",
                        "in-place weight fold (L2) of projections.weight / short_filter_weight", "per-module fp8 weight workspace (W1)")
