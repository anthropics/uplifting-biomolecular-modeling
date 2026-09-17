# rfd_se3fast/dense.py — dense (all-pairs) re-formulation of the SE3Transformer layer used by RFdiffusion v1's
# Str2Str blocks on the FULL graph (36 of the 40 SE(3) calls per denoising step: every node attends to all L-1 others).
#
# Stock path (NVIDIA SE3Transformer, MIT, as vendored in RFdiffusion @86507b65): per-edge tensors of shape (E=L*(L-1), ...),
# DGL message passing (e_dot_v, edge_softmax, copy_e_sum), gather of source-node features by edge, 2 bmm + 1 GEMM per
# VersatileConvSE3.  Dense path (this file): the same arithmetic expressed on (L, L-1, ...) tensors in destination-major
# edge order, so that softmax / neighbour sums are plain last-axis reductions and the per-edge contractions can be fused
# (torch here; Triton in kernels.py).  Precision policy equal to stock (fp32 everywhere, TF32 off).  NOT byte-equal to
# stock (re-associated reductions) -> Tier-2, tested at op level vs an fp64 reference and in-model distributionally.
#
# Edge-order facts this relies on (asserted at runtime by the patch layer, see patch.py):
#   * make_full_graph (and the kit's cached variant) enumerates edges with torch.where over (b, i, j), i.e. SOURCE-major:
#     eid = i*(L-1) + (j - (j>i)), src=i, dst=j.  Edge features `pair[b,i,j]` and rel_pos = xyz[j]-xyz[i] follow that order.
#   * The dense layout used here is DESTINATION-major: row d, column s' in [0, L-2] <-> source s = s' + (s' >= d).
#     PERM[d, s'] = eid of (s -> d) converts stock per-edge tensors to dense layout with one gather.
import math
from typing import Dict, Optional
import torch
from torch import Tensor

_PERM_CACHE = {}


def dst_major_perm(L: int, device) -> Tensor:
    """(L, L-1) int64: PERM[d, k] = source-major edge id of edge (s -> d), s = k + (k >= d)."""
    key = (L, str(device))
    p = _PERM_CACHE.get(key)
    if p is None:
        d = torch.arange(L, device=device).view(L, 1)
        k = torch.arange(L - 1, device=device).view(1, L - 1)
        s = k + (k >= d).long()
        # source-major eid of (s -> d): s*(L-1) + (d - (d > s))
        p = s * (L - 1) + (d - (d > s).long())
        _PERM_CACHE[key] = p
    return p


def src_index(L: int, device) -> Tensor:
    """(L, L-1) int64: source node of dense slot (d, k)."""
    d = torch.arange(L, device=device).view(L, 1)
    k = torch.arange(L - 1, device=device).view(1, L - 1)
    return k + (k >= d).long()


# ----------------------------------------------------------------------------------------------------------------
# Basis in dense layout.  Stock: basis['1,1'] etc. from spherical harmonics (e3nn) x Clebsch-Gordan; then
# update_basis_with_fused builds 'out0_fused' (E, 4, 2, 1), 'out1_fused' (E, 4, 3, 3), 'in0_fused' (E, 1, 2, 4),
# 'in1_fused' (E, 3, 3, 4) for max_degree=1 (zero-padded block layouts).  We keep the stock basis computation
# (it is <2 % of the step) and only permute rows to dst-major; the fused kernels consume the same fused tensors.
def permute_edge_tensor(x: Tensor, perm_flat: Tensor) -> Tensor:
    return x.index_select(0, perm_flat)


# ----------------------------------------------------------------------------------------------------------------
# Dense VersatileConvSE3:  out[e, co, f_out] = sum_{ci,f} R[e, co, ci*F+f] * (sum_i feat[e, ci, i] * basis[e, i, f*? ...])
# We reproduce the stock contraction order exactly in torch (bmm) for the reference/fallback path:
#   tmp = feat @ basis.view(E, in_dim, -1)  -> (E, ci, F*out_dim) -> view (E, ci*F, out_dim);  out = R @ tmp
def versatile_conv_dense(features: Tensor, radial_weights: Tensor, basis: Optional[Tensor], channels_in: int, freq_sum: int,
                         channels_out: int, fuse_full: bool = False) -> Tensor:
    E = features.shape[0]
    in_dim = features.shape[2]
    R = radial_weights.view(E, channels_out, channels_in * freq_sum)
    if basis is None:
        return R @ features
    out_dim = basis.shape[-1]
    if not fuse_full:
        out_dim += out_dim % 2 - 1
    basis_view = basis.view(E, in_dim, -1)
    tmp = (features @ basis_view).view(E, -1, basis.shape[-1])
    return (R @ tmp)[:, :, :out_dim]


# ----------------------------------------------------------------------------------------------------------------
# Dense attention (replaces AttentionSE3.forward on a full graph).  key/value: per-edge (dst-major dense) tensors,
# query: per-node.  Stock: logits = e_dot_v(key, query[dst]) / sqrt(F); softmax over incoming edges of each dst;
# out[dst] = sum_e w_e * value_e.
def attention_dense(key: Tensor, value: Tensor, query: Tensor, L: int, num_heads: int, scale: float) -> Tensor:
    """key: (L*(L-1), H, F) dst-major; query: (L, H, F); value: (L*(L-1), H, C/H, D) dst-major (fused degrees on last axis)
    returns (L, H, C/H, D)"""
    Lm = L - 1
    k = key.view(L, Lm, num_heads, -1)                       # (L, S, H, F)
    logits = torch.einsum('lshf,lhf->lhs', k, query) * scale  # (L, H, S)   [sum over F: re-associated vs DGL e_dot_v]
    w = torch.softmax(logits, dim=-1)                         # softmax over sources
    v = value.view(L, Lm, num_heads, value.shape[-2], value.shape[-1])   # (L, S, H, C, D)
    out = torch.einsum('lhs,lshcd->lhcd', w, v)
    return out
