"""kits/v1_2 — kits.v1's composition with the CLASS-KEYED A3 table.

A3 = the global layers' flex attention kernel options (attn.FLEX_KERNEL_OPTIONS: BLOCK_M 64 / BLOCK_N 128 / num_warps 4 /
num_stages 2). ``A3_BY_CLASS`` keys the lever by GPU class: on for h100 / h200, off by name for a100 / b200 / l40s (not bit-exact
with the stock there). A GPU of no listed class takes the lever set of the class the config names (``MODEL_OPT_TARGET_GPU``, h100
when unset) and its cite names the class as untested.

HOW: ``apply`` runs ``v1.apply`` with ``v1._components.attn.apply`` swapped, for the duration of that call only, for a wrapper that
passes the class-keyed ``attn.Config`` — ``attn.CONFIG`` (A1 + A2 + A3) where A3 is on, ``attn.CONFIG`` with ``flex_kernel_options=()``
(A1 + A2; A3 off = torch's own flex kernel config) elsewhere; ew1, P1 and the W pin are v1's untouched. The GPU class is read from
torch's device name through ``pins.gpu_class_key`` and stamped on every record with the A3 decision and its cite.
"""
from __future__ import annotations

import os
import time

from .. import load_kit
from .. import pins
from ..v1 import KitRefused

KIT = "v1.2"
LEVERS = ("class_keyed_A3", "ew1", "attn:A1", "attn:A2", "attn:A3_where_exact", "P1_precast")
TESTED_SHAPES = ((1, 256), (16, 256), (252, 256), (127, 512), (63, 1024), (324, 202))
#: the per-class A3 table: on where A3 is bit-exact with the stock (cite per class); off by name elsewhere
A3_BY_CLASS = {"h100": {"A3": True, "cite": "kit v1 EXACT on H100: the full panel (44,921/44,921 x 3 sizes) bitwise"},
               "h200": {"A3": True, "cite": "kit v1 EXACT on H200: the full panel (44,921/44,921 x 3 sizes) bitwise"},
               "a100": {"A3": False, "cite": "v1 NOT EXACT on a100 at the class W (1,911 of 44,921 differ; lever A3)"},
               "b200": {"A3": False, "cite": "v1 NOT EXACT on b200 at the class W (2,112-2,299 of 44,921 differ; lever A3)"},
               "l40s": {"A3": False, "cite": "v1 NOT EXACT on l40s at the class W (1,911 of 44,921 differ; lever A3)"}}
_BLANK = {"applied": False, "size": None, "num_warps": None, "gpu": None, "gpu_class": None, "a3": None, "a3_cite": None, "a3_untested": False, "config": None, "v1_record": None}
_state = dict(_BLANK)
_v1 = None


def _components():
    global _v1
    if _v1 is None:
        _v1 = load_kit("v1")
    return _v1


def kit_files() -> dict:
    """{relative path: present} for this kit's own source file — presence, not bytes."""
    return {os.path.basename(__file__): os.path.isfile(os.path.abspath(__file__))}


def counters_snapshot() -> dict:
    return _components().counters_snapshot()


def counters_delta(prev: dict) -> dict:
    return _components().counters_delta(prev)


def expected_per_forward(model, B: int, L: int) -> dict:
    return _components().expected_per_forward(model, B, L)


def a3_decision(gpu_name: str | None) -> dict:
    """The class-keyed A3 decision for a probed GPU name: {gpu_class, A3, cite, untested}. A GPU of no tested class takes the lever set of
    the class the kit's config names (``MODEL_OPT_TARGET_GPU``, `h100` when unset) — the lever ENGAGES as on that class and the cite NAMES
    the untested class: an untested GPU is never a reason to drop a lever."""
    import os
    key = pins.gpu_class_key(gpu_name)
    row = A3_BY_CLASS.get(key or "")
    if row is None:
        target = (os.environ.get("MODEL_OPT_TARGET_GPU") or "h100").strip().lower()
        trow = A3_BY_CLASS.get(target) or A3_BY_CLASS["h100"]
        return {"gpu_class": key, "A3": trow["A3"], "untested": True,
                "cite": f"untested class {key!r} ({gpu_name}): the {target} lever set of the kit's config applies, A3 {'on' if trow['A3'] else 'off'} as on {target} — not measured on this GPU ({trow['cite']})"}
    return {"gpu_class": key, "A3": row["A3"], "cite": row["cite"], "untested": False}


def attn_config_for(gpu_name: str | None):
    """attn.CONFIG on an A3-exact class; attn.CONFIG with flex_kernel_options=() (A3 off) elsewhere."""
    from .. import attn
    d = a3_decision(gpu_name)
    if d["A3"]:
        return attn.CONFIG, d
    return attn.Config(within=attn.CONFIG.within, global_=attn.CONFIG.global_, flex_kernel_options=(), flex_block_mask=attn.CONFIG.flex_block_mask,
                       flex_dynamic=attn.CONFIG.flex_dynamic, unpad_once=attn.CONFIG.unpad_once), d


def apply(model, *, size: str, num_warps: int | None = None, require_gpu: bool = True, det: bool = False, **_) -> dict:
    """v1.apply with the class-keyed attn config (v1._components.attn.apply swapped for the duration of the call)."""
    import torch
    if _state["applied"]:
        raise KitRefused("kit v1.2 already applied in this process")
    if require_gpu and not torch.cuda.is_available():
        raise KitRefused("kit v1.2 needs a GPU")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    cfg, dec = attn_config_for(gpu)
    v1 = _components(); attn = v1._components.attn
    real = attn.apply

    def _class_keyed_apply(model_, **kw):
        kw["config"] = cfg
        return real(model_, **kw)

    attn.apply = _class_keyed_apply
    try:
        rec = v1.apply(model, size=size, num_warps=num_warps, require_gpu=require_gpu, det=det)
    finally:
        attn.apply = real
    _state.update({"applied": True, "size": size, "num_warps": rec.get("num_warps"), "gpu": gpu, "gpu_class": dec["gpu_class"], "a3": dec["A3"], "a3_cite": dec["cite"], "a3_untested": dec.get("untested", False),
                   "config": str(cfg), "v1_record": rec})
    return {"kit": KIT, "kit_files": kit_files(), "levers": LEVERS, "size": size, "num_warps": rec.get("num_warps"), "gpu": gpu, "gpu_class": dec["gpu_class"],
            "A3": dec["A3"], "A3_cite": dec["cite"], "A3_untested": dec.get("untested", False), "attn_config": str(cfg), "v1": rec, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def unapply() -> None:
    if not _state["applied"]:
        return
    _components().unapply()
    _state.clear(); _state.update(_BLANK)


def stamp() -> dict:
    import torch
    return {"kit": KIT, "kit_files": kit_files(), "levers": LEVERS, "size": _state["size"], "num_warps": _state["num_warps"], "gpu": _state["gpu"], "gpu_class": _state["gpu_class"],
            "A3": _state["a3"], "A3_cite": _state["a3_cite"], "A3_untested": _state.get("a3_untested", False), "attn_config": _state["config"], "v1": (_components().stamp() if _state["applied"] else None),
            "gpu_probe": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
