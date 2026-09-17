"""fpf_msa.boltz2 — the Boltz-2 2.2.1 entry points of the fused MSA-module cells the shared core carries (opt_core.ops.msa_opm: the outer-product mean,
form 'mask_norm' = Boltz-2's norm / proj_a / proj_b / proj_o schema with the mask and num_mask division; opt_core.ops.msa_pwa: the pair-weighted
averaging, form 'masked' = norm_m / proj_m / proj_g / norm_z / proj_z / proj_o with the pair mask). Installed on the Boltz-2 classes by
boltz2_opt.msa_kernels (BOLTZ_FPF_MSA=opm,pwa); the FPF add-on's registry (fpf_engines, FPF_ENGINE=boltz2 FPF_OPS=outer_product_mean,msa_pair_weighted_avg
FPF_IMPL=outer_product_mean=fpf_msa.boltz2:outer_product_mean,...) takes the same functions.
Each fn(module, *stock_args, **stock_kwargs) -> stock return; raises FPFFallback (counted, stock runs) on unsupported cases. The Boltz-2
Transition class (MSA dim 64 / pair 128 / single 384) is not wired here: the pair-stack transition is boltz2_opt.transition's (the core's
kernels.transition rows) and the MSA dim-64 transition is boltz2_opt.msa2's trans2 cell.
"""
import torch
import os as _os
from ._compat import FPFFallback
from opt_core.ops import msa_opm as _O, msa_pwa as _P

def outer_product_mean(module, m, mask, chunk_size=None):
    if m.dim() != 4 or not m.is_cuda: raise FPFFallback("rank_or_device")
    if (module.c_hidden, module.proj_o.out_features) not in {(32, 128)}: raise FPFFallback("dims_not_pinned")
    return _O.forward_mask_norm(module, m, mask, chunk_size)

def msa_pair_weighted_avg(module, m, z, mask, chunk_heads=False):
    """DEFAULT = g_fo4p (fused LN_m->proj_m|proj_g->sigmoid prologue + i-fastest contraction kernel with fused gate and fused proj_o
    epilogue; Tier 2, same numerics class). FPF_PWA_CFG=<name> selects a CFG_VARIANTS entry
    (e.g. hp1 = stock torch prologue + kernel, for a bitwise-stock prologue)."""
    if m.dim() != 4 or not m.is_cuda: raise FPFFallback("rank_or_device")
    if (module.c_h, module.num_heads, m.shape[-1]) != (32, 8, 64): raise FPFFallback("dims_not_pinned")
    return _P.forward_masked(module, m, z, mask, chunk_heads)


# ---- L1 usage-layer lever (Boltz-2 only): run the engine's own UNCHUNKED stock code path at N > chunk_size_threshold (384).
# Same function, same class, same ops; numerics differ from the chunked path only by accumulation order / intermediate rounding
# (chunked OPM accumulates 8 bf16 projection partials and returns fp32; chunked PWA accumulates proj_o partials in bf16; chunked
# transition accumulates fc3 partials in bf16).  => TIER 2 by construction; memory: +~5 GB (OPM) / +~8 GB (PWA) at 705 tok, S=4724.
def _orig(op):
    import fpf_engines
    return fpf_engines.original(op)
def outer_product_mean_unchunked(module, m, mask, chunk_size=None):
    return _orig("outer_product_mean")(module, m, mask, None)
def msa_pair_weighted_avg_unchunked(module, m, z, mask, chunk_heads=False):
    return _orig("msa_pair_weighted_avg")(module, m, z, mask, False)

FPF_META = {**_O.FPF_META, **_P.FPF_META}
FPF_META["msa_pair_weighted_avg"]["variants"] = {"default(g_fo4p)": "fused LN_m/proj_m/proj_g/sigmoid prologue + contraction/gate/proj_o kernel",
                                                 "FPF_PWA_CFG=hp1": "stock torch prologue + head-pair contraction kernel (bitwise-stock prologue; 2.5x @705)"}
FPF_META["outer_product_mean_unchunked"] = {"ln_mode": "stock-call", "stages": "all:stock(unchunked path)", "z_passes_removed": "8-chunk loop -> 1 einsum + 1 GEMM", "tier": "TIER2 by rounding points (fp32 einsum, single bf16 rounding)"}
FPF_META["msa_pair_weighted_avg_unchunked"] = {"ln_mode": "stock-call", "stages": "all:stock(unchunked path)", "z_passes_removed": "8 head loops -> 1", "tier": "TIER2 by accumulation order (proj_o)"}
