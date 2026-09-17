"""infopt_graphs.bucket — static-shape bucketing and pad-and-batch helpers (engine-agnostic).

Two modes, and they are NOT equivalent in fidelity:

* EXACT-SHAPE (default for Tier 1): one graph per realized (n_token, n_atom, ...) shape; no padding, so the captured
  kernels see exactly the stock tensors and the result is the stock result up to GPU nondeterminism.  Capture amortizes
  inside one prediction (a 200-step sampler replays the same graph 199 times), so exact shapes cost ~1 capture per new
  design length.  `Bucketer(mode="exact")` is then just a shape cache key.

* PADDED BUCKETS (Tier 2): shapes are rounded up to a bucket (e.g. multiples of 32 tokens / 256 atoms) and masks are
  supplied so that padded positions do not influence real positions.  This is a numerics-only change ONLY if every
  kernel in the captured region is mask-exact (attention with -inf bias on padded keys, masked LayerNorm statistics,
  scatter/segment ops restricted to real atoms); otherwise it changes the function.  It must be gated like any Tier-2
  change (same-function proof per kernel + gate deltas) and labelled.  `pad_to_bucket` / `batch_pad` produce the padded
  tensors + masks; they do not make the model mask-exact — that is an engine-level property, established separately.

Bucketer API
    b = Bucketer(mode="exact" | "bucket", token_buckets=[...], atom_buckets=[...])
    key = b.key(n_token=..., n_atom=...)           -> (n_token_pad, n_atom_pad)
    b.stats -> Counter of keys seen
Helpers
    pad_to_bucket(x, dim, target, value=0)          -> padded tensor (new allocation) or x if already target
    batch_pad([tree_1, ..., tree_B], pad_dims={name: (dim, value)}) -> (batched tree, masks) for same-bucket items
"""
from __future__ import annotations

import collections
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

DEFAULT_TOKEN_BUCKETS = [160, 192, 224, 256, 288, 320, 384, 448, 512, 576, 640, 704, 768]
DEFAULT_ATOM_BUCKETS = [b * 8 for b in DEFAULT_TOKEN_BUCKETS]  # ~7.7 atoms / token for protein; ligands raise it


def round_up(n: int, buckets: Sequence[int], multiple: Optional[int] = None) -> int:
    for b in buckets:
        if n <= b:
            return b
    if multiple:
        return int(math.ceil(n / multiple) * multiple)
    return n


class Bucketer:
    def __init__(self, mode: str = "exact", token_buckets: Sequence[int] = DEFAULT_TOKEN_BUCKETS,
                 atom_buckets: Sequence[int] = DEFAULT_ATOM_BUCKETS, token_multiple: int = 32, atom_multiple: int = 256):
        assert mode in ("exact", "bucket")
        self.mode = mode
        self.token_buckets = list(token_buckets)
        self.atom_buckets = list(atom_buckets)
        self.token_multiple = token_multiple
        self.atom_multiple = atom_multiple
        self.stats = collections.Counter()

    def key(self, n_token: int, n_atom: Optional[int] = None) -> Tuple[int, Optional[int]]:
        if self.mode == "exact":
            k = (n_token, n_atom)
        else:
            k = (round_up(n_token, self.token_buckets, self.token_multiple),
                 None if n_atom is None else round_up(n_atom, self.atom_buckets, self.atom_multiple))
        self.stats[k] += 1
        return k

    def waste(self, n_token: int, n_atom: Optional[int] = None) -> Dict[str, float]:
        kt, ka = self.key(n_token, n_atom)
        self.stats[(kt, ka)] -= 1
        out = {"token_pad_frac": (kt - n_token) / max(1, kt)}
        if n_atom is not None and ka is not None:
            out["atom_pad_frac"] = (ka - n_atom) / max(1, ka)
        # pair tensors scale quadratically in tokens
        out["pair_waste_frac"] = 1.0 - (n_token / kt) ** 2
        return out


def pad_to_bucket(x: torch.Tensor, dim: int, target: int, value: float = 0) -> torch.Tensor:
    n = x.shape[dim]
    if n == target:
        return x
    assert n < target, f"pad_to_bucket: size {n} > target {target} on dim {dim}"
    shape = list(x.shape)
    shape[dim] = target - n
    pad = torch.full(shape, value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


def batch_pad(items: List[Any], pad_dims: Dict[str, Tuple[int, float]], targets: Dict[str, int],
              mask_names: Optional[Dict[str, str]] = None) -> Tuple[Any, Dict[str, torch.Tensor]]:
    """Pad and stack same-bucket items (dicts of tensors).  pad_dims: {key: (dim, pad_value)}; targets: {key: target_size};
    mask_names: {key: mask_key} -> adds a bool mask [B, target] of real positions for those keys.  Keys absent from
    pad_dims are stacked as-is (they must already agree in shape)."""
    assert len(items) > 0
    out: Dict[str, Any] = {}
    masks: Dict[str, torch.Tensor] = {}
    keys = items[0].keys()
    for k in keys:
        vals = [it[k] for it in items]
        if k in pad_dims and isinstance(vals[0], torch.Tensor):
            dim, val = pad_dims[k]
            tgt = targets[k]
            padded = [pad_to_bucket(v, dim, tgt, val) for v in vals]
            out[k] = torch.stack(padded, dim=0)
            if mask_names and k in mask_names:
                m = torch.zeros(len(vals), tgt, dtype=torch.bool, device=vals[0].device)
                for i, v in enumerate(vals):
                    m[i, : v.shape[dim]] = True
                masks[mask_names[k]] = m
        elif isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out, masks
