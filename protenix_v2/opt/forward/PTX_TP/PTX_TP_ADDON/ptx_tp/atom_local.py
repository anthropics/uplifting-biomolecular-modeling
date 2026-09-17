"""ptx_tp/atom_local.py -- O(N_atom) construction of the local-attention padding masks (replicated-side lever).

Stock protenix 2.0.0 builds the [n_trunks, 32, 128] window padding bias of every atom local-attention call by
allocating a DENSE [N_atom + q_pad, N_atom + 176] tensor (primitives.py 493-506: q.new_zeros(...), three slice
fills with -inf, optimized_concat_split (a second dense copy), unfold, transpose) -- 6 calls per diffusion step
(3 encoder + 3 decoder blocks) and 3 in the trunk's input embedder; rearrange_qk_to_dense_trunk(compute_mask=True)
(primitives.py 380-396, called once per forward from protenix.update_input_feature_dict) does the same with a ones
tensor.  Bytes per call ~ 2 x (N_atom^2) x elt: 62k atoms fp32 = 2 x 15.4 GB; 250k atoms = 2 x 250 GB.

The window entries depend only on (n, n_queries, n_keys): entry [w, i, j] <-> query atom 32w+i, key atom
32w+j-48; value 0 (mask 1) iff 32w+i < n and 0 <= 32w+j-48 < n, else -inf (mask 0).  This module computes them
directly ([n_trunks, 32, 128], O(N_atom)); values are 0 / -inf(=-1e10 after the same dtype conversion) / bool, so
the replacement is exact by construction (torch.equal to the stock construction).

apply() monkeypatches primitives.rearrange_to_dense_trunk and {primitives,transformer}.rearrange_qk_to_dense_trunk;
apply_from_env() does so when PTX_TP_ATOM_LOCAL=1 (default 1).  remove() restores the stock functions.
Idea reference: window batching as in arXiv 2603.14806 (no code from it is used).
"""
import math
import os

import torch

from protenix.model.modules import primitives as _P
from protenix.model.modules import transformer as _T

_ORIG = {}
_STATS = {"dense_bias_calls_avoided": 0, "dense_mask_calls_avoided": 0, "bytes_avoided": 0}


def trunk_valid_mask(n: int, n_queries: int, n_keys: int, device) -> torch.Tensor:
    """[n_trunks, n_queries, n_keys] bool: window (w, i, j) is a real (query, key) atom pair."""
    n_trunks = int(math.ceil(n / n_queries))
    pad_left = (n_keys - n_queries) // 2
    w = torch.arange(n_trunks, device=device)
    qi = w[:, None] * n_queries + torch.arange(n_queries, device=device)[None, :]
    kj = w[:, None] * n_queries + torch.arange(n_keys, device=device)[None, :] - pad_left
    return (qi < n)[:, :, None] & ((kj >= 0) & (kj < n))[:, None, :]


def rearrange_to_dense_trunk(q, k, v, n_queries, n_keys, attn_bias=None, inf: float = 1e10):
    """Drop-in for primitives.rearrange_to_dense_trunk.  With a caller-supplied dense attn_bias the stock function is
    used (that path is inherently O(n^2) and is not used by Protenix-v2 inference)."""
    if attn_bias is not None:
        return _ORIG["rearrange_to_dense_trunk"](q, k, v, n_queries, n_keys, attn_bias=attn_bias, inf=inf)
    n, d = q.shape[-2:]
    q_trunked, kv_trunked, padding_info = _P.rearrange_qk_to_dense_trunk(
        q=q, k=[k, v], dim_q=-2, dim_k=[-2, -2], n_queries=n_queries, n_keys=n_keys, compute_mask=False,
    )
    valid = trunk_valid_mask(n, n_queries, n_keys, q.device)
    bias = torch.zeros(valid.shape, dtype=q.dtype, device=q.device).masked_fill_(~valid, -inf)
    bias = bias.reshape((1,) * len(q.shape[:-2]) + tuple(bias.shape))
    pad_left, pad_right = padding_info["k_pad_left"], padding_info["k_pad_right"]
    _STATS["dense_bias_calls_avoided"] += 1
    _STATS["bytes_avoided"] += 2 * (n + padding_info["q_pad"]) * (n + pad_left + pad_right) * q.element_size()
    return q_trunked, kv_trunked[0], kv_trunked[1], bias, padding_info["q_pad"]


def rearrange_qk_to_dense_trunk(q, k, dim_q, dim_k, n_queries: int = 32, n_keys: int = 128, compute_mask: bool = True):
    """Drop-in for primitives.rearrange_qk_to_dense_trunk: identical outputs; mask_trunked built in O(n)."""
    q_trunked, k_trunked, padding_info = _ORIG["rearrange_qk_to_dense_trunk"](
        q, k, dim_q, dim_k, n_queries=n_queries, n_keys=n_keys, compute_mask=False
    )
    if compute_mask:
        q0 = q[0] if isinstance(q, list) else q
        dq = dim_q[0] if isinstance(dim_q, list) else dim_q
        n = q0.shape[dq]
        valid = trunk_valid_mask(n, n_queries, n_keys, q0.device)
        padding_info["mask_trunked"] = valid.reshape((1,) * len(q0.shape[:-2]) + tuple(valid.shape))
        pad_left, pad_right = padding_info["k_pad_left"], padding_info["k_pad_right"]
        _STATS["dense_mask_calls_avoided"] += 1
        _STATS["bytes_avoided"] += 2 * (n + padding_info["q_pad"]) * (n + pad_left + pad_right) * q0.element_size()
    return q_trunked, k_trunked, padding_info


def apply() -> bool:
    if _ORIG:
        return False
    _ORIG["rearrange_to_dense_trunk"] = _P.rearrange_to_dense_trunk
    _ORIG["rearrange_qk_to_dense_trunk"] = _P.rearrange_qk_to_dense_trunk
    _P.rearrange_to_dense_trunk = rearrange_to_dense_trunk
    _P.rearrange_qk_to_dense_trunk = rearrange_qk_to_dense_trunk
    _T.rearrange_qk_to_dense_trunk = rearrange_qk_to_dense_trunk  # `from ... import` alias used by protenix.update_input_feature_dict
    return True


def remove() -> bool:
    if not _ORIG:
        return False
    _P.rearrange_to_dense_trunk = _ORIG.pop("rearrange_to_dense_trunk")
    _P.rearrange_qk_to_dense_trunk = _ORIG.pop("rearrange_qk_to_dense_trunk")
    _T.rearrange_qk_to_dense_trunk = _P.rearrange_qk_to_dense_trunk
    return True


def apply_from_env() -> bool:
    if os.environ.get("PTX_TP_ATOM_LOCAL", "1") == "1":
        return apply()
    return False


def report() -> dict:
    return dict(_STATS, applied=bool(_ORIG))


def dense_bytes_formula(n_atom: int, elt: int = 4, n_queries: int = 32, n_keys: int = 128) -> int:
    """Bytes of ONE stock dense construction (the zeros/ones tensor + its permuted copy)."""
    n_trunks = int(math.ceil(n_atom / n_queries))
    q_pad = n_trunks * n_queries - n_atom
    pad_left = (n_keys - n_queries) // 2
    pad_right = int((n_trunks - 1 / 2) * n_queries + n_keys / 2 - n_atom + 1 / 2)
    return 2 * (n_atom + q_pad) * (n_atom + pad_left + pad_right) * elt
