"""msa2.exact_hoist — EXACT-tier (bitwise = stock) engineering of the Boltz-2 2.2.1 MSA-module layers: the stock statement sequences of
``PairWeightedAveraging.forward`` (pair_averaging.py:40-135), ``Transition.forward`` (transition.py:52-78) and ``OuterProductMean.forward``
(outer_product_mean.py:32-98) with THREE pure hoists and nothing else:

  H1 cast-once     Under ``torch.autocast('cuda', torch.bfloat16)`` (Boltz2.forward's trunk region) ``nn.LayerNorm`` returns fp32 and every ``x @ W.T`` /
                   ``nn.Linear`` re-casts that fp32 activation to bf16 (autocast caches casts of LEAF parameters only, never activations): PWA's chunk_heads
                   loop casts LN(m) [B,S,N,64] fp32 -> bf16 2x per head x 8 heads = 16x per call and LN(z) 8x; the chunked Transition casts LN(x) 2x per
                   hidden chunk x 8 = 16x; OPM casts LN(m) 2x. Here the cast is done ONCE (``.to(torch.bfloat16)``, the very conversion autocast's
                   cached_cast performs: aten::_to_copy, round-to-nearest-even) and the bf16 tensor is handed to the SAME matmul expressions — autocast
                   leaves an operand that is already bf16 untouched, so every cuBLAS call receives bit-identical operands (values, dtype, shape, strides)
                   and returns bit-identical results. The weight-side casts are left to autocast exactly as stock (same expressions).
  H2 num_mask      OPM's chunked path rebuilds ``num_mask`` (the [B,N,N,1] pair count from the [B,S,N] MSA mask, 64-row chunks, bf16 sums) on every
                   call — 16 calls per prediction on the SAME ``feats['msa_mask']`` tensor. Here it is computed by the stock statements once per mask
                   tensor (cache keyed on the tensor object, which is held alive so its identity cannot be recycled) and reused: identical values.
  H3 to(m) elided  OPM chunked: ``z.to(m) @ W_slice.T`` with z bf16 and m = LN(m) fp32: the fp32 copy is re-cast to bf16 by autocast before the GEMM;
                   fp32(bf16 x) -> bf16 is the identity for every value (incl. subnormals, inf, nan), so ``z @ W_slice.T`` feeds the GEMM the identical
                   bf16 operand (same contiguous layout) and skips a 4-byte write + read and a 2-byte write of [B,N,N,128] per chunk.

Everything else is the stock statement sequence verbatim (same chunking, same per-head / per-chunk GEMM shapes, same bf16 partial-sum chains, same
division, same einsum). Guard (served BY NAME, else the stock statements run, counted): CUDA autocast enabled with dtype bfloat16 and module in eval —
outside that regime an explicit cast would CHANGE numerics (``autocast_off`` / ``autocast_dtype`` / ``training``). No RNG is drawn in these layers
(the MSA-layer residual's dropout mask is drawn by MSALayer.forward, untouched here; boltz 2.2.1 dropout.py ``>=`` with dropout 0 in eval => mask == 1.0
provably).

Checks: unit bit-equality vs the stock modules on CUDA under autocast (boltz2_opt/tests/test_msa2_hoist.py) and sha256-identical
prediction files vs ``--mode off``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor

UNITS = ("pwa", "transition", "opm")
CENSUS: Dict[str, Dict[str, int]] = {u: {"calls": 0, "served": 0, "fallback": 0} for u in UNITS}
FALLBACK_BY: Dict[str, Dict[str, int]] = {u: {} for u in UNITS}
SHAPES: Dict[str, Dict[str, int]] = {u: {} for u in UNITS}
_NUM_MASK: Dict[str, Any] = {"mask": None, "value": None, "key": None, "hits": 0, "builds": 0}


def _regime(module) -> Optional[str]:
    """None when the hoists are bit-safe (eval, CUDA autocast bf16); else the fallback word."""
    if module.training:
        return "training"
    if not torch.is_autocast_enabled("cuda"):
        return "autocast_off"
    if torch.get_autocast_dtype("cuda") != torch.bfloat16:
        return "autocast_dtype"
    return None


def _count(unit: str, word: Optional[str], shape=None) -> None:
    c = CENSUS[unit]; c["calls"] += 1
    if word is None:
        c["served"] += 1
        if shape is not None:
            k = "x".join(str(int(s)) for s in shape); SHAPES[unit][k] = SHAPES[unit].get(k, 0) + 1
    else:
        c["fallback"] += 1; FALLBACK_BY[unit][word] = FALLBACK_BY[unit].get(word, 0) + 1


# ------------------------------------------------------------------------------------------------------------------------------------------
# PairWeightedAveraging.forward  (pair_averaging.py:40-135) — H1 on LN(m) (16 casts -> 1) and LN(z) (8 -> 1)
# ------------------------------------------------------------------------------------------------------------------------------------------
def pwa_forward(stock_forward):
    def forward(self, m: Tensor, z: Tensor, mask: Tensor, chunk_heads: bool = False) -> Tensor:
        word = _regime(self)
        _count("pwa", word, m.shape)
        if word is not None:
            return stock_forward(self, m, z, mask, chunk_heads)
        # Compute layer norms
        m = self.norm_m(m)
        z = self.norm_z(z)
        m = m.to(torch.bfloat16)          # H1: autocast's own cast of the fp32 LN output, once
        z = z.to(torch.bfloat16)          # H1

        if chunk_heads and not self.training:
            # Compute heads sequentially
            o_chunks = []
            for head_idx in range(self.num_heads):
                sliced_weight_proj_m = self.proj_m.weight[
                    head_idx * self.c_h : (head_idx + 1) * self.c_h, :
                ]
                sliced_weight_proj_g = self.proj_g.weight[
                    head_idx * self.c_h : (head_idx + 1) * self.c_h, :
                ]
                sliced_weight_proj_z = self.proj_z.weight[head_idx : (head_idx + 1), :]
                sliced_weight_proj_o = self.proj_o.weight[
                    :, head_idx * self.c_h : (head_idx + 1) * self.c_h
                ]

                # Project input tensors
                v: Tensor = m @ sliced_weight_proj_m.T
                v = v.reshape(*v.shape[:3], 1, self.c_h)
                v = v.permute(0, 3, 1, 2, 4)

                # Compute weights
                b: Tensor = z @ sliced_weight_proj_z.T
                b = b.permute(0, 3, 1, 2)
                b = b + (1 - mask[:, None]) * -self.inf
                w = torch.softmax(b, dim=-1)

                # Compute gating
                g: Tensor = m @ sliced_weight_proj_g.T
                g = g.sigmoid()

                # Compute output
                o = torch.einsum("bhij,bhsjd->bhsid", w, v)
                o = o.permute(0, 2, 3, 1, 4)
                o = o.reshape(*o.shape[:3], 1 * self.c_h)
                o_chunks = g * o
                if head_idx == 0:
                    o_out = o_chunks @ sliced_weight_proj_o.T
                else:
                    o_out += o_chunks @ sliced_weight_proj_o.T
            return o_out
        else:
            # Project input tensors
            v: Tensor = self.proj_m(m)
            v = v.reshape(*v.shape[:3], self.num_heads, self.c_h)
            v = v.permute(0, 3, 1, 2, 4)

            # Compute weights
            b: Tensor = self.proj_z(z)
            b = b.permute(0, 3, 1, 2)
            b = b + (1 - mask[:, None]) * -self.inf
            w = torch.softmax(b, dim=-1)

            # Compute gating
            g: Tensor = self.proj_g(m)
            g = g.sigmoid()

            # Compute output
            o = torch.einsum("bhij,bhsjd->bhsid", w, v)
            o = o.permute(0, 2, 3, 1, 4)
            o = o.reshape(*o.shape[:3], self.num_heads * self.c_h)
            o = self.proj_o(g * o)
            return o
    forward.__wrapped__ = stock_forward
    forward._msa2 = "hoist"
    return forward


# ------------------------------------------------------------------------------------------------------------------------------------------
# Transition.forward  (transition.py:52-78) — H1 on LN(x): chunked 16 casts -> 1, unchunked 2 -> 1.  `accept(module, x, chunk_size)` (the adapter's
# predicate) picks WHICH Transition calls this serves (the dim-64 MSA / template transitions the core's cell table does not serve); the others go
# to `stock_forward` (whatever served them before: the transition adapter's forward in the kit rows) uncounted.
# ------------------------------------------------------------------------------------------------------------------------------------------
def transition_forward(stock_forward, accept=None):
    def forward(self, x: Tensor, chunk_size: int = None) -> Tensor:
        if accept is not None and not accept(self, x, chunk_size):
            return stock_forward(self, x, chunk_size)
        word = _regime(self)
        _count("transition", word, (x.shape[-1], self.hidden, -1 if chunk_size is None else chunk_size))
        if word is not None:
            return stock_forward(self, x, chunk_size)
        x = self.norm(x)
        x = x.to(torch.bfloat16)          # H1

        if chunk_size is None or self.training:
            x = self.silu(self.fc1(x)) * self.fc2(x)
            x = self.fc3(x)
            return x
        else:
            # Compute in chunks
            for i in range(0, self.hidden, chunk_size):
                fc1_slice = self.fc1.weight[i : i + chunk_size, :]
                fc2_slice = self.fc2.weight[i : i + chunk_size, :]
                fc3_slice = self.fc3.weight[:, i : i + chunk_size]
                x_chunk = self.silu((x @ fc1_slice.T)) * (x @ fc2_slice.T)
                if i == 0:
                    x_out = x_chunk @ fc3_slice.T
                else:
                    x_out = x_out + x_chunk @ fc3_slice.T
            return x_out
    forward.__wrapped__ = stock_forward
    forward._msa2 = "hoist"
    return forward


# ------------------------------------------------------------------------------------------------------------------------------------------
# OuterProductMean.forward  (outer_product_mean.py:32-98) — H1 (2 casts -> 1), H2 (num_mask once per mask tensor), H3 (z.to(m) elided)
# ------------------------------------------------------------------------------------------------------------------------------------------
def _num_mask_chunked(mask: Tensor) -> Tensor:
    """The stock statements (outer_product_mean.py:58-67), verbatim."""
    for i in range(0, mask.shape[1], 64):
        if i == 0:
            num_mask = (
                mask[:, i : i + 64, None, :] * mask[:, i : i + 64, :, None]
            ).sum(1)
        else:
            num_mask += (
                mask[:, i : i + 64, None, :] * mask[:, i : i + 64, :, None]
            ).sum(1)
    num_mask = num_mask.clamp(min=1)
    return num_mask


def _num_mask_cached(raw_mask: Tensor, mask: Tensor) -> Tensor:
    """H2: num_mask for this ``feats['msa_mask']`` tensor (the caller's raw [B,S,N] mask, held alive as the key) at this expanded dtype."""
    c = _NUM_MASK
    key = (tuple(raw_mask.shape), mask.dtype, torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"))   # no ._version: inference-mode tensors (Lightning predict) do not track one
    if c["mask"] is raw_mask and c["value"] is not None and c["key"] == key:   # same tensor object (held alive below), unmodified, same expanded dtype and
        c["hits"] += 1                                                          # autocast state (the sums run in fp32 under autocast: the value's dtype follows)
        return c["value"]
    val = _num_mask_chunked(mask)
    c.update(mask=raw_mask, value=val, key=key)
    c["builds"] += 1
    return val


def num_mask_cache_clear() -> None:
    _NUM_MASK.update(mask=None, value=None, key=None)


def opm_forward(stock_forward):
    def forward(self, m: Tensor, mask: Tensor, chunk_size: int = None) -> Tensor:
        word = _regime(self)
        _count("opm", word, m.shape)
        if word is not None:
            return stock_forward(self, m, mask, chunk_size)
        raw_mask = mask
        # Expand mask
        mask = mask.unsqueeze(-1).to(m)

        # Compute projections
        m = self.norm(m)
        m16 = m.to(torch.bfloat16)        # H1
        a = self.proj_a(m16) * mask
        b = self.proj_b(m16) * mask

        # Compute outer product mean
        if chunk_size is not None and not self.training:
            # Compute pairwise mask
            num_mask = _num_mask_cached(raw_mask, mask)   # H2 (stock statements, once per mask tensor)

            # Compute squentially in chunks
            for i in range(0, self.c_hidden, chunk_size):
                a_chunk = a[:, :, :, i : i + chunk_size]
                sliced_weight_proj_o = self.proj_o.weight[
                    :, i * self.c_hidden : (i + chunk_size) * self.c_hidden
                ]

                z = torch.einsum("bsic,bsjd->bijcd", a_chunk, b)
                z = z.reshape(*z.shape[:3], -1)
                z = z / num_mask

                # Project to output
                if i == 0:
                    z_out = z @ sliced_weight_proj_o.T            # H3: z.to(m) elided (fp32 round trip of a bf16 value; autocast re-casts to the same bf16)
                else:
                    z_out = z_out + z @ sliced_weight_proj_o.T    # H3
            z_out = z_out + self.proj_o.bias  # add bias
            return z_out
        else:
            mask = mask[:, :, None, :] * mask[:, :, :, None]
            num_mask = mask.sum(1).clamp(min=1)
            z = torch.einsum("bsic,bsjd->bijcd", a.float(), b.float())
            z = z.reshape(*z.shape[:3], -1)
            z = z / num_mask

            # Project to output
            z = self.proj_o(z.to(m))
            return z
    forward.__wrapped__ = stock_forward
    forward._msa2 = "hoist"
    return forward


def census() -> Dict[str, Any]:
    return {"units": {u: dict(c) for u, c in CENSUS.items()}, "fallback_by": {u: dict(d) for u, d in FALLBACK_BY.items()},
            "shapes": {u: dict(d) for u, d in SHAPES.items()}, "num_mask": {"hits": _NUM_MASK["hits"], "builds": _NUM_MASK["builds"]}}
