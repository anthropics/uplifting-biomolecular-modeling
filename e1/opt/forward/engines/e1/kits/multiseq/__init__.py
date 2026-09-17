"""Kit ``multiseq`` for Profluent-E1 — the global (flex-attention) layers of forwards the attn adapter's dense route does not serve:
rows holding several sequences (retrieval-augmented prefill: context members + query in one row) and padded rows.

LEVERS (exact by construction / by the kernel's reduction order):
    docmask     `create_block_causal_mask_optimized(sequence_ids)` — upstream evaluates its `document_mask` closure on the dense
                (B, 1, L, L) grid through torch's `create_block_mask` (B·L² booleans and int64 gathers — a transient quadratic in
                L). The block-causal document mask of sorted sequence ids with a
                padding suffix is decided per (q-block, kv-block) pair by four block statistics (min / max valid id, all-valid,
                any-valid): a block is FULL iff both blocks are all-valid and min_q_id >= max_kv_id, EMPTY iff max_q_id < min_kv_id
                or either block has no valid token, PARTIAL otherwise. The (partial, full) block grids go through torch's own
                `_create_sparse_block_from_block_mask` with the stock's `mask_mod` closure — the BlockMask upstream builds, tensor for
                tensor (kv_num_blocks, kv_indices, full_kv_*, q_*, BLOCK_SIZE, seq_lengths, and the same mask_mod for partial blocks),
                from O(B·(L/128)²) work. Dense forwards keep the adapter's index-only mask; anything whose ids are not a sorted run with
                a padding suffix takes the stock call.
    flex_opts   the fallback route's flex call (`Attention._flex_attn` when the adapter does not route) runs the kit's pinned flex
                callable — the one the dense route runs (the adapter's compiled `flex_attention` with the A3 kernel options in force,
                anylen's hybrid form) — instead of upstream's compiled object at torch's default options. BLOCK_N (the kv tile a query
                row reduces over) is the stock's 128 either way and the block visit order is the BlockMask's, so the per-row online
                softmax sees the same sequence of tiles: bitwise equal on document masks as on dense ones. With A3 off (classes
                a100 / b200 / l40s) the options are torch's defaults, as the stock's.
Requires the attn adapter applied (kits/v1_2): docmask replaces the function the adapter's fallback calls
(E1.model.flex_attention.create_block_causal_mask_optimized), flex_opts sits in the adapter's _flex_attn fallback slot — dense forwards
never reach either, the adapter's own introspection stays valid.
Counters: docmask_built, docmask_stock:<reason>, flex_opts_calls, flex_stock:<reason>.
"""
from __future__ import annotations

import collections

import torch

KIT = "multiseq"
LEVERS = ("docmask", "flex_opts")
BLOCK = 128                      # torch's default BLOCK_SIZE for create_block_mask (the stock passes none)
CTR = collections.Counter()
_R: dict = {"installed": False, "orig": {}, "flex_key": None}


class MultiseqRefused(RuntimeError):
    """A precondition failed at install: nothing is left in force."""


# ------------------------------------------------------------------------------------------------------------------ docmask
def block_grids(sequence_ids: torch.Tensor, block: int = BLOCK):
    """(partial, full) boolean block grids (B, 1, nb, nb) of mask(q, kv) = ids[q] >= ids[kv] & ids[q] != -1 & ids[kv] != -1 for ids
    ascending per row with -1 only as a suffix. Pure tensor ops on the ids' device; no L x L intermediate."""
    B, L = sequence_ids.shape
    nb = (L + block - 1) // block
    pad = nb * block - L
    ids = sequence_ids
    if pad:
        ids = torch.cat([ids, ids.new_full((B, pad), -1)], dim=1)
    blk = ids.view(B, nb, block)
    valid = blk != -1
    big = torch.iinfo(ids.dtype).max
    mn = torch.where(valid, blk, torch.full_like(blk, big)).amin(-1)          # min valid id per block (big if none)
    mx = torch.where(valid, blk, torch.full_like(blk, -2)).amax(-1)           # max valid id per block (-2 if none)
    nvalid = valid.sum(-1)
    allvalid = nvalid == block
    anyvalid = nvalid > 0
    any_true = (mx[:, :, None] >= mn[:, None, :]) & anyvalid[:, :, None] & anyvalid[:, None, :]
    all_true = allvalid[:, :, None] & allvalid[:, None, :] & (mn[:, :, None] >= mx[:, None, :])
    return (any_true & ~all_true).unsqueeze(1), all_true.unsqueeze(1)


def ids_sorted_with_pad_suffix(sequence_ids: torch.Tensor) -> bool:
    """The precondition of the block statistics: per row, valid ids non-decreasing and every -1 after the last valid token.
    One device-to-host read (the forward's only one for the mask)."""
    ids = sequence_ids
    valid = ids != -1
    nxt_valid = valid[:, 1:]
    ok_pad = ~(nxt_valid & ~valid[:, :-1])                                   # no valid token right after a pad
    ok_sorted = ~(nxt_valid & valid[:, :-1] & (ids[:, 1:] < ids[:, :-1]))     # non-decreasing over valid neighbours
    return bool((ok_pad & ok_sorted).all())


def document_block_mask(sequence_ids: torch.Tensor):
    """The stock's BlockMask for `sequence_ids`, built from block statistics. Same mask_mod closure as upstream's."""
    from torch.nn.attention.flex_attention import _create_sparse_block_from_block_mask

    def document_mask(b, h, q_idx, kv_idx):  # upstream's closure (E1.model.flex_attention.create_block_causal_mask_optimized)
        return (
            (sequence_ids[b, q_idx] >= sequence_ids[b, kv_idx])
            & (sequence_ids[b, q_idx] != -1)
            & (sequence_ids[b, kv_idx] != -1)
        )

    B, L = sequence_ids.shape
    partial, full = block_grids(sequence_ids, BLOCK)
    return _create_sparse_block_from_block_mask((partial, full), document_mask, (L, L), BLOCK, BLOCK)


def _docmask(sequence_ids: torch.Tensor):
    """E1.model.flex_attention.create_block_causal_mask_optimized replacement — the function the attn adapter's fallback route (and
    anything else) calls for a non-dense forward: the block-statistics BlockMask, or the stock call when the ids are not a sorted run
    with a padding suffix."""
    if sequence_ids.dim() != 2 or sequence_ids.dtype not in (torch.int64, torch.int32):
        CTR["docmask_stock:ids_form"] += 1
        return _R["orig"]["stock_create"](sequence_ids)
    if not ids_sorted_with_pad_suffix(sequence_ids):
        CTR["docmask_stock:unsorted_ids"] += 1
        return _R["orig"]["stock_create"](sequence_ids)
    CTR["docmask_built"] += 1
    return document_block_mask(sequence_ids)


# ----------------------------------------------------------------------------------------------------------------- flex_opts
def _flex_attn_opts(self, query_states, key_states, val_states, sequence_ids, attention_args=None, output_attentions=False):
    """The attn adapter's _flex_attn FALLBACK (what it calls when its dense route does not apply): the stock's flex call computed by
    the kit's pinned flex callable (same block mask, same inputs); the stock function for what the callable does not cover."""
    from ..attn import adapter as A
    fn = A._S["flex_fns"].get(_R["flex_key"]) if _R["flex_key"] is not None else None
    fa = attention_args.get("flex_attention_args", None) if attention_args is not None else None
    block_mask = fa.get("block_mask", None) if fa is not None else None
    score_mod = fa.get("score_mod", None) if fa is not None else None
    if fn is None or score_mod is not None or output_attentions or query_states.shape[2] != key_states.shape[2]:
        CTR["flex_stock:" + ("no_fn" if fn is None else "score_mod" if score_mod is not None else "output_attentions" if output_attentions else "gqa")] += 1
        return _R["orig"]["_flex_attn"](self, query_states, key_states, val_states, sequence_ids, attention_args, output_attentions)
    bsz, q_len = query_states.shape[0], query_states.shape[1]
    q = query_states.transpose(1, 2).contiguous()            # (B, nh, L, hd) — as E1.model.flex_attention.flex_attention_func lays them out
    k = key_states.transpose(1, 2).contiguous()
    v = val_states.transpose(1, 2).contiguous()
    out = fn(q, k, v, block_mask=block_mask)
    CTR["flex_opts_calls"] += 1
    out = out.transpose(1, 2)
    return out.reshape(bsz, q_len, self.hidden_size).contiguous(), None


# ------------------------------------------------------------------------------------------------------------------- surface
def install() -> dict:
    """Put docmask in place of E1.model.flex_attention.create_block_causal_mask_optimized and flex_opts in the attn adapter's _flex_attn
    fallback slot. Requires kits/attn applied (the eager kit applies it first)."""
    import E1.modeling as M
    from E1.model import attention as A_
    from E1.model import flex_attention as FX
    from ..attn import adapter as A
    if _R["installed"]:
        raise MultiseqRefused(f"kit {KIT}: already installed in this process")
    cfg = A._S.get("cfg")
    if cfg is None or M.create_block_causal_mask_optimized is not A._block_mask or A_.Attention.__dict__.get("_flex_attn") is not A._flex_attn_routed \
            or A._S["orig"].get("_flex_attn") is None:
        raise MultiseqRefused(f"kit {KIT}: the attn adapter is not in force (apply kits/v1_2 first)")
    key = (cfg.flex_kernel_options, cfg.flex_dynamic)
    if A._S["flex_fns"].get(key) is None:
        A._flex_fn(cfg)                                        # the adapter compiles lazily; make the callable exist now
    _R["orig"] = {"stock_create": FX.create_block_causal_mask_optimized, "_flex_attn": A._S["orig"]["_flex_attn"]}
    _R["flex_key"] = key
    FX.create_block_causal_mask_optimized = _docmask
    A._S["orig"]["_flex_attn"] = _flex_attn_opts
    _R["installed"] = True
    CTR.clear()
    fn = A._S["flex_fns"].get(key)
    return {"kit": KIT, "levers": list(LEVERS), "block": BLOCK, "flex_key": [dict(key[0]), key[1]], "flex_callable": getattr(fn, "__name__", str(fn))}


def uninstall() -> None:
    if not _R["installed"]:
        return
    from E1.model import flex_attention as FX
    o = _R["orig"]
    if FX.create_block_causal_mask_optimized is _docmask:
        FX.create_block_causal_mask_optimized = o["stock_create"]
    try:
        from ..attn import adapter as A
        if A._S.get("orig", {}).get("_flex_attn") is _flex_attn_opts:
            A._S["orig"]["_flex_attn"] = o["_flex_attn"]
    except Exception:
        pass
    _R.update(installed=False, orig={}, flex_key=None)


def in_force() -> bool:
    from E1.model import flex_attention as FX
    try:
        from ..attn import adapter as A
        below = A._S.get("orig", {}).get("_flex_attn") is _flex_attn_opts
    except Exception:
        below = False
    return bool(_R["installed"]) and FX.create_block_causal_mask_optimized is _docmask and below


def counters() -> dict:
    return {k: int(v) for k, v in CTR.items()}


def is_pristine() -> dict:
    from E1.model import flex_attention as FX
    try:
        from ..attn import adapter as A
        below = A._S.get("orig", {}).get("_flex_attn") is _flex_attn_opts
    except Exception:
        below = False
    return {"create_block_causal_mask_optimized": FX.create_block_causal_mask_optimized is not _docmask, "_flex_attn_fallback": not below}
