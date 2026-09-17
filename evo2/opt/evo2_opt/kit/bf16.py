"""The Transformer-Engine-free route (a stack without transformer_engine, e.g. A100: upstream's bf16 input projections): the TE-free levers by name (L1b, E7, E11, L2 as a bf16 weight fold, E50, E56), the FP8 levers not applicable by name."""
from __future__ import annotations

import importlib.util
from evo2_opt.kit import ltsel as LT

from evo2_opt.kit.errors import KitRefused

ROUTE = "bf16-projections"
LEVERS = ("L1b_hcl_filter_cache", "E7_triton_fir3_featurizer", "E11_fused_rmsnorm_tail", "L2_interleave_fold_bf16_projection_weight",
          "E50_out_filter_dense_matmul_on_transposed_view", "E56_fused_bias_residual", "E65_attention_residual_norm", "E66_residual_into_next_pre_norm",
          "T9_cublaslt_config_selection")
NOT_APPLICABLE = {
    "L2_interleave_folded_into_te_master_weight": "Transformer Engine is absent on this stack (bf16 input projections): the fold is the bf16 projection-weight fold instead",
    "W1_fp8_weight_workspace_cache": "Transformer Engine is absent on this stack: no FP8 weight workspaces exist",
    "E61_general_gemm_out_buffer": "Transformer Engine is absent on this stack: transformer_engine general_gemm is not on the path",
    "GATE_forward_gate_shape_manager": "this route holds no shape-bound store (the HCL filter cache is model-bound): every (batch, length) runs the same patched classes",
    "G1_forward_graph_replay": "the graph replay rides on the forward gate, which this route does not install",
}
_state = {"installed": False, "levers": [], "facts": None}


def te_importable() -> bool:
    return importlib.util.find_spec("transformer_engine") is not None


# ------------------------------------------------------------------------------------------------------------- is this the route's stock
def served(evo2_model) -> tuple:
    """``(True, facts)`` when the constructed model is upstream's Transformer-Engine-free construction (TE not importable: vortex's TELinear
    falls back to bf16 input projections; either Hyena conv path — the levers here touch neither); ``(False, reason)`` otherwise."""
    if te_importable():
        return False, "transformer_engine is importable: the FP8 routes serve this stack, not the bf16 route"
    model = getattr(evo2_model, "model", evo2_model)
    cfg = model.config
    if bool(cfg.get("use_fp8_input_projections", False)):
        return False, "use_fp8_input_projections is on without transformer_engine importable: not upstream's bf16 fallback construction"
    from vortex.model.layers import HAS_TE
    from vortex.model.model import ParallelGatedConvBlock, AttentionBlock
    if HAS_TE:
        return False, "vortex reports HAS_TE: not the TE-free stack"
    hb = [b for b in model.blocks if isinstance(b, ParallelGatedConvBlock)]
    ab = [b for b in model.blocks if isinstance(b, AttentionBlock)]
    engines = [e for e in (getattr(b.filter, "engine", None) for b in hb) if e is not None]
    flags = {k: sorted({bool(getattr(e, k, False)) for e in engines}) for k in ("use_hcs_kernel", "use_hcm_kernel", "use_hcl_kernel")}
    if any(len(v) != 1 for v in flags.values()):
        return False, f"the Hyena kernel flags are not uniform across the blocks ({flags})"
    proj = type(hb[0].projections).__name__ if hb else None
    if proj != "TELinear":
        return False, f"projection module {proj}: the bf16 levers are built for vortex's TELinear fallback only"
    conv = "vortex-kernels" if flags["use_hcs_kernel"] == [True] else "torch-conv"
    return True, {"route": ROUTE, "conv_path": conv, "n_hyena": len(hb), "n_attn": len(ab), "n_rms": 2 * len(model.blocks) + 1, "hidden": int(cfg.get("hidden_size", 0)),
                  "n_blocks": len(model.blocks), "projections": proj, "kernel_flags": {k: v[0] for k, v in flags.items()}}


# ------------------------------------------------------------------------------------------------------------- levers
def _install(model):
    """Install the route's class patches + the bf16 fold; returns the ordered lever names installed. kit.base supplies the members, kit.ltsel the T9 install."""
    import torch
    from vortex.model.engine import HyenaInferenceEngine
    from vortex.model.model import HyenaCascade, ParallelGatedConvBlock, AttentionBlock, StripedHyena
    from vortex.model.layers import RMSNorm
    from evo2_opt.kit import base as K7
    K7.check_conditions(model)                                   # the TELinear-fallback / Wqkv-layout / featurizer facts the kit.base levers are built for (AssertionError on drift)

    def _parallel_fir_featurizer(self, fir_fn, u, weight, bias, L, dims, groups=None, gated_bias=False, column_split_hyena=False, dim_last=True,
                                 fir_length=3, gate=False, inference_params=None, prefill_mode=None, padding_mask=None):
        if (not gate and dim_last and fir_length == 3 and bias is None and inference_params is None and padding_mask is None
                and fir_fn is torch.nn.functional.conv1d and u.dim() == 3 and weight.shape[-1] == 3 and not column_split_hyena):
            K7.CTR["fir3_triton_fwd"] += 1
            return K7.fir3_triton(u, weight, 0), None                                     # (B, 3D, L) bf16, ascending tap order (kit.base fir3_triton)
        if inference_params is not None:
            K7.CTR["passthrough:generation_call"] += 1                                   # a prefill / decode call: vortex's own parallel_fir, counted by name
        return K7._ORIGINALS["parallel_fir"](self, fir_fn, u, weight, bias, L, dims, groups=groups, gated_bias=gated_bias, column_split_hyena=column_split_hyena,
                                             dim_last=dim_last, fir_length=fir_length, gate=gate, inference_params=inference_params, prefill_mode=prefill_mode,
                                             padding_mask=padding_mask)

    installed = []
    HyenaCascade.compute_filter = K7._cached_compute_filter; installed.append(LEVERS[0])
    HyenaInferenceEngine.parallel_fir = _parallel_fir_featurizer; installed.append(LEVERS[1])
    RMSNorm.forward = K7._rms_fused_forward; installed.append(LEVERS[2])
    K7.fold_interleave(model); installed.append(LEVERS[3])
    ParallelGatedConvBlock.forward = K7._e56_hyena_forward; installed.extend(LEVERS[4:6])
    AttentionBlock.forward = K7._attn_forward; installed.append(LEVERS[6])
    StripedHyena.stateless_forward = K7._stateless_forward_pairs; installed.append(LEVERS[7])
    LT.install(model); installed.append(LEVERS[8])
    K7.CTR.clear()
    return installed


def counters() -> dict:
    from evo2_opt.kit import base as K7
    d = {k: int(v) for k, v in K7.CTR.items()}
    d.update(LT.counters())
    return d


def apply(evo2_model) -> dict:
    """Install the route's levers on the constructed model (no forward runs here: the caches fill on the first call); refused by name when
    the model is not the TE-free construction or a launch condition of the levers does not hold."""
    ok, facts = served(evo2_model)
    if not ok:
        raise KitRefused(f"the bf16 route does not serve this model: {facts}")
    if _state["installed"]:
        raise KitRefused("the bf16 route is already installed in this process (one model per process)")
    levers = _install(evo2_model.model)
    _state.update(installed=True, levers=levers, facts=facts)
    return {"route": ROUTE, "conv_path": facts["conv_path"], "levers": list(levers), "not_applicable": dict(NOT_APPLICABLE), "facts": facts, "pipeline": (False, "one device")}
