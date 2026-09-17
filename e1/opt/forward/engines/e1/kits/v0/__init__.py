"""kits/v0 — EXACT data-movement levers of the E1 forward (the kit surface is in engines.e1.kits.__init__).

Levers (each reproduces the stock bytes by construction: no arithmetic is added, removed or reordered):
  P1 precast   the autocast weight casts done once: every ``nn.Linear`` weight/bias held in bf16 (autocast passes a bf16 weight through;
               the bf16 GEMM and its fp32 accumulate are the stock's). Norm weights and the MLM-head LayerNorm stay fp32.
  P2 unpad     ``_unpad_input`` / ``pad_input`` of the within-sequence layers computed ONCE per forward (a forward pre-hook on the model
               derives the unpad data from ``sequence_ids``; every layer reads it) with a pass-through fast path when the batch carries no
               padding (the stock's gathers / scatters are then pure copies -> reshapes). A layer whose tensors are not the forward's
               ``sequence_ids`` (KV-cache context mode) takes the stock path and is COUNTED (``expected_per_forward`` expects zero).
  P3 blockmask the global layers' block-causal document mask built once per distinct ``sequence_ids`` CONTENT (sha256 of its bytes;
               the stock builds it every forward, on the host), an LRU of BLOCKMASK_LRU (8).
"""
from __future__ import annotations

import hashlib
import os
from collections import OrderedDict

KIT = "v0"
LEVERS = ("P1_precast", "P2_unpad_once", "P3_blockmask_cache")
TESTED_SHAPES = ((1, 256), (16, 256), (252, 256), (127, 512), (63, 1024))
BLOCKMASK_LRU = 8

COUNTERS = {"P1_precast_linears": 0, "P2_unpad_ctx_compute": 0, "P2_unpad_served": 0, "P2_unpad_identity": 0, "P2_unpad_gather": 0,
            "P2_unpad_ctx_miss": 0, "P2_pad_identity": 0, "P2_pad_scatter": 0, "P3_blockmask_compute": 0, "P3_blockmask_hit": 0,
            "forwards": 0}
_state = {"applied": False, "size": None, "num_warps": None, "expected": None, "model_id": None, "hooks": [], "record": None}
_ctx = {}                       # the current forward's unpad data (set by the pre-hook, cleared by the post-hook)
_blockmasks = OrderedDict()


class KitRefused(RuntimeError):
    """The kit refuses to run: a pin or a counter expectation does not hold."""


def kit_files() -> dict:
    """{relative path: present} for the kit's source file — presence, not bytes."""
    return {os.path.basename(__file__): os.path.isfile(__file__)}


def counters_snapshot() -> dict:
    return dict(COUNTERS)


def counters_delta(prev: dict) -> dict:
    return {k: COUNTERS[k] - prev.get(k, 0) for k in COUNTERS}


def _layer_kinds(model) -> tuple[int, int]:
    """(n_within_seq_layers, n_global_layers) from the model's own attention modules."""
    from E1.model.attention import AttentionLayerType
    within = global_ = 0
    for layer in model.model.layers:                                        # modeling.py: DecoderLayer.norm_attn_norm.self_attn (Attention), attention.py: Attention.layer_type
        if layer.norm_attn_norm.self_attn.layer_type == AttentionLayerType.GLOBAL:
            global_ += 1
        else:
            within += 1
    return within, global_


def expected_per_forward(model, B: int, L: int) -> dict:
    """What ONE forward of an unpadded (B, L) batch must count: one unpad-context compute, every within-seq layer served from it on
    the pass-through fast path, zero stock gathers / scatters / context misses; one block-mask compute the first time a sequence_ids
    content is seen, a hit afterwards (the same batch run twice -> 1 compute + 1 hit over 2 forwards)."""
    n_within, n_global = _layer_kinds(model)
    return {"P2_unpad_ctx_compute": 1, "P2_unpad_served": n_within, "P2_unpad_identity": n_within, "P2_unpad_gather": 0,
            "P2_unpad_ctx_miss": 0, "P2_pad_identity": n_within, "P2_pad_scatter": 0, "P3_blockmask_per_distinct_content": 1,
            "n_within": n_within, "n_global": n_global}


# ------------------------------------------------------------------------------------------------------------------- levers
def _p1_precast(model) -> int:
    import torch
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            m.weight.data = m.weight.data.to(torch.bfloat16)
            if m.bias is not None:
                m.bias.data = m.bias.data.to(torch.bfloat16)
            n += 1
    COUNTERS["P1_precast_linears"] += n
    return n


def _pre_hook(module, args, kwargs):
    """Derive the forward's unpad data from its sequence_ids once (the stock derives it in every within-seq layer)."""
    import torch
    from E1.model import flash_attention_utils as U
    seq = kwargs.get("sequence_ids")
    if seq is None and len(args) > 3:
        seq = args[3]
    _ctx.clear()
    COUNTERS["forwards"] += 1
    if seq is None:
        return
    idx, cu, mx = U._get_unpad_data(seq)
    B, L = seq.shape
    ident = idx.numel() == B * L and bool(torch.equal(idx, torch.arange(B * L, device=idx.device)))
    _ctx.update(ptr=seq.data_ptr(), shape=tuple(seq.shape), idx=idx, cu=cu, mx=mx, ident=ident)
    COUNTERS["P2_unpad_ctx_compute"] += 1


def _post_hook(module, args, kwargs, output):
    _ctx.clear()


def _unpad_kit(q, k, v, q_ids, k_ids):
    from E1.model import flash_attention_utils as U
    same = q_ids is k_ids or (q_ids.data_ptr() == k_ids.data_ptr() and q_ids.shape == k_ids.shape)
    if not (_ctx and same and k_ids.data_ptr() == _ctx["ptr"] and tuple(k_ids.shape) == _ctx["shape"]):
        COUNTERS["P2_unpad_ctx_miss"] += 1                                   # context / cache mode: the stock path, counted
        return U._unpad_input(q, k, v, q_ids, k_ids)
    COUNTERS["P2_unpad_served"] += 1
    idx, cu, mx = _ctx["idx"], _ctx["cu"], _ctx["mx"]
    B, Lk, nh, hd = k.shape
    Lq, nq = q.shape[1], q.shape[2]
    if _ctx["ident"]:
        COUNTERS["P2_unpad_identity"] += 1
        return (q.reshape(B * Lq, nq, hd), k.reshape(B * Lk, nh, hd), v.reshape(B * Lk, nh, hd), idx, (cu, cu), (mx, mx))
    COUNTERS["P2_unpad_gather"] += 1
    return (U.index_first_axis(q.reshape(B * Lq, nq, hd), idx), U.index_first_axis(k.reshape(B * Lk, nh, hd), idx),
            U.index_first_axis(v.reshape(B * Lk, nh, hd), idx), idx, (cu, cu), (mx, mx))


def _pad_kit(h, indices, batch, seqlen):
    from E1.model import flash_attention_utils as U
    if h.shape[0] == batch * seqlen:
        COUNTERS["P2_pad_identity"] += 1
        return h.reshape(batch, seqlen, *h.shape[1:])
    COUNTERS["P2_pad_scatter"] += 1
    return U.pad_input(h, indices, batch, seqlen)


def _blockmask_kit(sequence_ids):
    from E1.model import flex_attention as FX
    key = (tuple(sequence_ids.shape), hashlib.sha256(sequence_ids.detach().cpu().numpy().tobytes()).hexdigest())
    if key in _blockmasks:
        COUNTERS["P3_blockmask_hit"] += 1
        _blockmasks.move_to_end(key)
        return _blockmasks[key]
    COUNTERS["P3_blockmask_compute"] += 1
    _blockmasks[key] = FX.create_block_causal_mask_optimized(sequence_ids)
    while len(_blockmasks) > BLOCKMASK_LRU:
        _blockmasks.popitem(last=False)
    return _blockmasks[key]


def _patches_in_force() -> bool:
    import E1.modeling as M
    import E1.model.flash_attention as FA
    return FA._unpad_input is _unpad_kit and FA.pad_input is _pad_kit and M.create_block_causal_mask_optimized is _blockmask_kit


# ------------------------------------------------------------------------------------------------------------------- apply
def apply(model, *, size: str, num_warps: int | None = None, require_gpu: bool = True, det: bool = False) -> dict:
    """Pins asserted -> autotune pin (W or an explicit censused W) -> P1 + P2 + P3 in force -> record.
    The model must be the stock model (fp32 weights) BEFORE any forward in this process (the pin must precede the
    first RMSNorm call)."""
    import torch
    import E1.modeling as M
    import E1.model.flash_attention as FA
    from .. import pins
    if _state["applied"]:
        raise KitRefused("kit v0 already applied in this process")
    if require_gpu and not torch.cuda.is_available():
        raise KitRefused("kit v0 needs a GPU")
    stack = pins.stack_report()
    kern = pins.kernel_revision_report()
    if size not in pins.WEIGHTS:
        raise KitRefused(f"unknown size {size!r}")
    dtypes = {str(p.dtype) for p in model.parameters()}
    if dtypes != {"torch.float32"}:
        raise KitRefused(f"the stock model holds fp32 weights before the kit (tools/score.py dtype=torch.float); got {sorted(dtypes)}")
    pin = pins.apply_autotune_pin(size, num_warps=num_warps)
    n_before = sum(p.numel() for p in model.parameters())
    n_lin = _p1_precast(model)
    FA._unpad_input, FA.pad_input = _unpad_kit, _pad_kit
    M.create_block_causal_mask_optimized = _blockmask_kit
    _state["hooks"] = [model.register_forward_pre_hook(_pre_hook, with_kwargs=True),
                       model.register_forward_hook(_post_hook, with_kwargs=True)]
    _state.update(applied=True, size=size, num_warps=pin["num_warps"], model_id=id(model))
    rec = {"kit": KIT, "levers": LEVERS, "kit_files": kit_files(), "size": size, "autotune_pin": pin, "stack": stack, "kernel": kern,
           "precast_linears": n_lin, "n_params": n_before, "tested_shapes": TESTED_SHAPES}
    if sum(p.numel() for p in model.parameters()) != n_before:
        raise KitRefused("the pre-cast changed the parameter count")
    _state["record"] = rec
    return rec


def stamp() -> dict:
    """The fail-closed stamp: refuses (raises) unless the kit is applied and its patches are in force."""
    import torch
    import E1.modeling as M
    from .. import pins
    if not _state["applied"] or not _patches_in_force():
        raise KitRefused("stamp: kit v0 is not in force in this process")
    return {"kit": KIT, "kit_files": kit_files(), "levers": LEVERS, "size": _state["size"], "num_warps": _state["num_warps"],
            "rmsnorm_path": "triton_layer_norm_kernel", "rmsnorm_snapshot": pins.KERNEL["rev"], "rmsnorm_file": getattr(M.layer_norm, "__file__", None),
            "autotune_readback": pins.autotune_readback(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "counters": counters_snapshot(), "tested_shapes": TESTED_SHAPES}
