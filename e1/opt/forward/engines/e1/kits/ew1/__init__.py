"""Kit ``ew1`` for Profluent-E1: fused elementwise Triton kernels that reproduce the stock's bf16 rounding sequence bitwise
(engines.e1.kits surface; pins and inputs from engines.e1.kits).

Levers (``LEVERS``; each a one-kernel replacement of a stock elementwise chain at the chain's own rounding points):
    rope     clamp(-8, 8) + RoPE on q/k (+ clamp on v) in ONE kernel, the flex layers' (B, H, L, hd) layout written directly
    glu      silu(w1 x) * (w3 x) in ONE kernel (IEEE division + expf in fp32, one bf16 rounding, one bf16 mul)
    addnorm  residual add (bf16) + RMSNorm (the hub kernel's reduction, pinned num_warps) in ONE kernel
    embed    token + sequence-id embedding gather, fp32 add, one bf16 rounding, in ONE kernel
The kit binds methods on the model's own Attention / GLUMLP / DecoderLayer / E1Model instances (no package globals), keeps one
bf16 cos/sin table pair per rotary module (built from that module's own cache, no cross-module check, no device-to-host copy on the
forward path) and counts every patched path (``counters()``).
"""
from __future__ import annotations

import os

from .patches import (CTR, EXACT_LEVERS, KitRefused, apply as _apply_patches, assert_counts, counters_snapshot, expected_counts,
                      stock_source_pins, unapply)

KIT = "ew1"
LEVERS = tuple(EXACT_LEVERS)
TESTED_SHAPES = ((1, 256), (16, 256), (252, 256), (127, 512), (63, 1024))
KERNELS = ("_clamp_rope_qkv_kernel", "_silu_mul_kernel", "_add_rmsnorm_kernel", "_embed_add_kernel")
_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULES = ("__init__.py", "patches.py", "kernels.py")
_state = {"applied": False, "size": None, "num_warps": None, "model_id": None, "record": None, "hook": None, "levers": None}


def kit_files() -> dict:
    """{relative path: present} for the kit's three modules (``_MODULES``) — presence, not bytes."""
    return {name: os.path.isfile(os.path.join(_HERE, name)) for name in _MODULES}


def counters_delta(prev: dict) -> dict:
    now = counters_snapshot()
    return {k: now.get(k, 0) - prev.get(k, 0) for k in set(now) | set(prev)}


def expected_per_forward(model, B: int, L: int) -> dict:
    """What ONE forward must count under the applied levers (shape-independent: one kernel call per patched site)."""
    return expected_counts(model, _state["levers"] if _state["levers"] is not None else LEVERS)


def _count_forward(module, args, kwargs):
    CTR["forwards"] += 1


def apply(model, *, size: str, num_warps: int | None = None, require_gpu: bool = True, det: bool = False,
          levers=LEVERS, rope_transposed: bool = True) -> dict:
    """Pins asserted (stack, stock commit, hub kernel revision, weights size known) -> the autotune pin (W or an
    explicit censused W; an equal pre-existing single-config pin is accepted) -> the levers bound on the model's modules ->
    record. The model must not have run a forward in this process before the pin."""
    import torch
    from .. import pins
    if _state["applied"]:
        raise KitRefused("kit ew1 already applied in this process")
    if require_gpu and not torch.cuda.is_available():
        raise KitRefused("kit ew1 needs a GPU")
    stack = pins.stack_report()
    commit = pins.assert_stock_commit()
    kern = pins.kernel_revision_report()
    if size not in pins.WEIGHTS:
        raise KitRefused(f"unknown size {size!r}")
    k = pins.rmsnorm_autotuner()
    if len(k.configs) == 1 and num_warps is not None and k.configs[0].num_warps != int(num_warps):
        raise KitRefused(f"the hub kernel is already pinned at num_warps {k.configs[0].num_warps} != {num_warps}")
    pin = pins.apply_autotune_pin(size, num_warps=num_warps)
    src = stock_source_pins()
    rec_p = _apply_patches(model, levers=tuple(levers), W=pin["num_warps"], rope_transposed=rope_transposed)
    _state["hook"] = model.register_forward_pre_hook(_count_forward, with_kwargs=True)
    CTR.clear()
    _state.update(applied=True, size=size, num_warps=pin["num_warps"], model_id=id(model), levers=tuple(levers))
    rec = {"kit": KIT, "levers": list(levers), "kit_files": kit_files(), "size": size, "autotune_pin": pin,
           "stack": stack, "stock_commit": commit, "kernel": kern, "stock_source_sha256": src, "tested_shapes": TESTED_SHAPES, **rec_p}
    _state["record"] = rec
    return rec


def stamp() -> dict:
    """The fail-closed stamp: refuses unless the kit is applied in this process."""
    if not _state["applied"]:
        raise KitRefused("stamp: kit ew1 is not in force in this process")
    import torch
    import E1.modeling as M
    from .. import pins
    return {"kit": KIT, "kit_files": kit_files(), "levers": list(_state["levers"]), "size": _state["size"],
            "num_warps": _state["num_warps"], "rmsnorm_path": "triton_layer_norm_kernel", "rmsnorm_snapshot": pins.KERNEL["rev"],
            "rmsnorm_file": getattr(M.layer_norm, "__file__", None), "autotune_readback": pins.autotune_readback(),
            "inductor": pins.inductor_readback(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "counters": counters_snapshot(), "tested_shapes": TESTED_SHAPES, "rope_transposed": _state["record"]["rope_transposed"] if _state["record"] else None,
            "triton_cache_keys": {k: pins.triton_cache_keys(kernel_substr=k) for k in KERNELS}}


def release(model) -> None:
    """Undo the patches and the forward hook (lever sets can be toggled in one process)."""
    if _state["hook"] is not None:
        _state["hook"].remove()
    unapply()
    _state.update(applied=False, size=None, num_warps=None, model_id=None, record=None, hook=None, levers=None)
