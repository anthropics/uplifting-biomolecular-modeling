"""The top of the torch-conv chain: the clone-free interleave fold (the projection rows and featurizer taps permuted in place, no second copy of the weights) over the composed, gated chain."""
from __future__ import annotations

from evo2_opt.kit import c2r_apply as M8

M2, MF, LG, K7, M3, M4 = M8.M2, M8.MF, M8.LG, M8.K7, M8.M3, M8.M4
CTR = M8.CTR
VERSIONS = M8.VERSIONS                     # no new lever tag: the kernels are the chain's below (kit.c2r_apply)
GATE = M8.GATE
BASE_KIT = "v40_full_8"
_installed = M8._installed
N_HCL, N_HCM = M8.N_HCL, M8.N_HCM
PROCESS_WIDE_PATCHES = M8.PROCESS_WIDE_PATCHES
counters = M8.counters


def expected_counts(version="v0"):
    return M8.expected_counts(version)


_FOLD = {"installed": False}


def _install_clone_free_fold():
    """Patch kit.gemm_apply's fold (its apply calls fold_interleave_te by module-global name) and kit.base's unfold_interleave:
    the same permutation, no clone, no host copy; _fold_state keeps the permutation only (unfold = its inverse, a row permutation back)."""
    if _FOLD["installed"]:
        return
    import torch
    from evo2_opt.kit import gemm_apply as LV
    V6 = LV.V6
    LV_orig_fold, V6_orig_unfold = LV.fold_interleave_te, V6.unfold_interleave

    def fold_interleave_te_fast(model) -> None:
        cfg = model.config
        assert cfg.interleave is True, "fold requires config.interleave True (already folded?)"
        for b in LV.hyena_blocks(model):
            W = b.projections.weight
            assert W.dim() == 2 and W.dtype == torch.bfloat16 and W.is_contiguous(), (type(b.projections).__name__, W.shape, W.dtype, W.is_contiguous())
            perm = V6._interleave_perm(W.shape[0]).to(W.device)
            W.data = W.data[perm].contiguous()
            if not LV._empty_bias(b.projections):
                b.projections.bias.data = b.projections.bias.data[perm].contiguous()
            b.filter.short_filter_weight.data = b.filter.short_filter_weight.data[perm].contiguous()
            if b.filter.short_filter_bias is not None:
                b.filter.short_filter_bias.data = b.filter.short_filter_bias.data[perm].contiguous()
            assert W.is_contiguous() and b.filter.short_filter_weight.is_contiguous()
            V6._fold_state[id(b)] = (perm.to("cpu"),)                 # the permutation only (3·hidden int64 indices per block), never the weights
        cfg.interleave = False

    def unfold_interleave_fast(model) -> None:
        for b in LV.hyena_blocks(model):
            st = V6._fold_state.get(id(b))
            if st is None or len(st) == 4:                            # absent, or a state written by the original fold (not this kit's): left to the original unfold
                continue
            (perm,) = st
            V6._fold_state.pop(id(b))
            W = b.projections.weight
            inv = torch.empty_like(perm); inv[perm] = torch.arange(perm.numel(), dtype=perm.dtype, device=perm.device)   # on the perm's device (the CPU): a bare arange would land on the current CUDA device
            inv = inv.to(W.device)
            W.data = W.data[inv].contiguous()
            if not LV._empty_bias(b.projections):
                b.projections.bias.data = b.projections.bias.data[inv].contiguous()
            b.filter.short_filter_weight.data = b.filter.short_filter_weight.data[inv].contiguous()
            if b.filter.short_filter_bias is not None:
                b.filter.short_filter_bias.data = b.filter.short_filter_bias.data[inv].contiguous()
        if any(len(st) == 4 for st in V6._fold_state.values()):
            V6_orig_unfold(model)
        model.config.interleave = True

    LV.fold_interleave_te = fold_interleave_te_fast
    V6.unfold_interleave = unfold_interleave_fast
    _FOLD.update(installed=True, originals=(LV_orig_fold, V6_orig_unfold))


def apply(model, version="v0"):
    _install_clone_free_fold()
    info = M8.apply(model, version)
    info["base_kit"] = BASE_KIT
    info["fold"] = "clone-free (the permutation kept, unfold = its inverse); folded bytes = v8's"
    return info


def ensure_shape(B, L):
    return M8.ensure_shape(B, L)


def is_unpatched(model=None):
    return M8.is_unpatched(model)


