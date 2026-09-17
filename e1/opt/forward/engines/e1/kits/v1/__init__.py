"""kits/v1 — the EXACT composition of the component kits ew1 and attn plus P1, under the one-owner order:

    pins.apply_autotune_pin(size)                 the hub RMSNorm num_warps pin, first in the process
    -> ew1.apply   (owner of E1Model.forward + DecoderLayer.forward: fused clamp+RoPE, silu*gate, add+RMSNorm, embed)
    -> attn.apply  (owner of _unpad_input/pad_input, create_block_causal_mask_optimized, Attention._flash_attn/_flex_attn:
                    shape-derived cu_seqlens, the index-only cheap block mask, the pinned flex kernel options)
    -> v0._p1_precast(model)   every Linear's weight/bias cast to bf16 ONCE (the autocast casts become no-ops; the bits
                               the GEMMs read are the same bf16 values autocast would have produced per call)

``assert_components`` checks that ew1 and attn are present and self-identify correctly before every apply (refuses otherwise).
The composition adds no arithmetic of its own; exactness class EXACT. The model is CONSUMED by P1 (weights cast in place):
unapply() releases the patches (attn then ew1, LIFO) but cannot restore the fp32 weights — a fresh load is the stock model again.
"""
import os

from .. import pins
from . import _components

KIT = "v1"
#: the component kits this one composes (presence + self-identification checked at apply)
COMPONENTS = ("ew1", "attn")
ORDER = ("ew1", "attn", "P1_precast")                   # one-owner order (P1 last: attn asserts fp32 weights at its apply)
LEVERS = ("P1_precast",) + tuple(f"ew1:{l}" for l in _components.ew1.LEVERS) + tuple(f"attn:{l}" for l in _components.attn.LEVERS)
TESTED_SHAPES = pins.PROBE_SHAPES

_state = {"applied": False, "size": None, "num_warps": None, "model_id": None, "precast_linears": None, "record": None}


class KitRefused(RuntimeError):
    """Fail-closed: a pin, a component or a counter expectation does not hold."""


def kit_files() -> dict:
    """{relative path: present} for this kit's own source file — presence, not bytes."""
    return {os.path.basename(__file__): os.path.isfile(os.path.abspath(__file__))}


def assert_components() -> dict:
    """Each component module is present and self-identifies correctly (refuse otherwise)."""
    live = {"ew1": _components.ew1.KIT, "attn": _components.attn.KIT}
    bad = {k: v for k, v in live.items() if v != k}
    if bad:
        raise KitRefused(f"component kit misidentified (expected name -> live KIT): {bad}")
    return live


def counters_snapshot() -> dict:
    return {"ew1": _components.ew1.counters_snapshot(), "attn": _components.attn.counters_snapshot(),
            "P1_precast_linears": _state["precast_linears"] or 0}


def counters_delta(prev: dict) -> dict:
    return {"ew1": _components.ew1.counters_delta(prev["ew1"]), "attn": _components.attn.counters_delta(prev["attn"]),
            "P1_precast_linears": (_state["precast_linears"] or 0) - prev.get("P1_precast_linears", 0)}


def expected_per_forward(model, B: int, L: int) -> dict:
    """Both components' expectations for ONE forward (the composition adds no counted path of its own). attn's expectation
    is ALSO installed as attn's current one (``attn._state['expected']``)."""
    exp = {"ew1": _components.ew1.expected_per_forward(model, B, L), "attn": _components.attn.expected_per_forward(model, B, L)}
    _components.attn._state["expected"] = exp["attn"]
    return exp


def apply(model, *, size: str, num_warps: int | None = None, require_gpu: bool = True, det: bool = False) -> dict:
    """Pins (stack, hub kernel revision, stock commit, the components present) -> the autotune pin -> ew1 -> attn -> P1 ->
    record. The model must be the stock model (fp32 weights) BEFORE any forward in this process."""
    import torch
    if _state["applied"]:
        raise KitRefused("kit v1 already applied in this process")
    if require_gpu and not torch.cuda.is_available():
        raise KitRefused("kit v1 needs a GPU")
    stack = pins.stack_report()
    kern = pins.kernel_revision_report()
    stock = pins.assert_stock_commit()
    live = assert_components()
    if size not in pins.WEIGHTS:
        raise KitRefused(f"unknown size {size!r}")
    dtypes = {str(p.dtype) for p in model.parameters()}
    if dtypes != {"torch.float32"}:
        raise KitRefused(f"the stock model holds fp32 weights before the kit (tools/score.py dtype=torch.float); got {sorted(dtypes)}")
    W = pins.apply_autotune_pin(size, num_warps=num_warps)["num_warps"]
    # one-owner order: ew1 (model/layer forwards) -> attn (unpad/pad, mask, attention routes) -> P1 (weights, last: attn
    # asserts fp32 weights at its apply)
    rec_ew1 = _components.ew1.apply(model, size=size, num_warps=W, require_gpu=require_gpu, det=det)
    try:
        rec_attn = _components.attn.apply(model, size=size, num_warps=W, require_gpu=require_gpu, det=det)
    except Exception:
        _components.ew1.unapply()
        raise
    n_cast = _components.v0._p1_precast(model)
    _state.update({"applied": True, "size": size, "num_warps": W, "model_id": id(model), "precast_linears": n_cast})
    record = {"kit": KIT, "kit_files": kit_files(), "components": live, "order": ORDER, "levers": LEVERS, "size": size, "num_warps": W,
              "stack": stack, "kernel": kern, "stock": stock, "precast_linears": n_cast, "ew1": rec_ew1, "attn": rec_attn,
              "inductor": pins.inductor_readback()}
    _state["record"] = record
    return record


def unapply() -> None:
    """Release the patches in LIFO order (attn, then ew1). The P1 weight cast is not reversible: reload for a stock model."""
    if not _state["applied"]:
        return
    _components.attn.unapply()
    _components.ew1.unapply()
    _state.update({"applied": False, "model_id": None})


def stamp() -> dict:
    """The fail-closed stamp: refuses unless the composition is in force in this process."""
    if not _state["applied"]:
        raise KitRefused("stamp: kit v1 is not in force in this process")
    import torch
    return {"kit": KIT, "kit_files": kit_files(), "components": assert_components(), "order": ORDER, "levers": LEVERS,
            "size": _state["size"], "num_warps": _state["num_warps"], "precast_linears": _state["precast_linears"],
            "ew1": _components.ew1.stamp(), "attn": _components.attn.stamp(), "autotune_readback": pins.autotune_readback(),
            "inductor": pins.inductor_readback(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "tested_shapes": TESTED_SHAPES}
