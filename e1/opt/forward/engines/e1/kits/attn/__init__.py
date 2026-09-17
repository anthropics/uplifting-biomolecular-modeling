"""Kit ``attn`` for Profluent-E1 — the attention-kernel levers, on the E1 kit surface (``engines.e1.kits``).

LEVERS (dense mode = every row of the batch is one sequence with no padding — every masked-marginal batch the scorer
builds; decided once per forward from ``sequence_ids`` by a pre-hook on the inner ``E1Model``):
    A1_varlen_shape_cu_seqlens   WITHIN_SEQ layers: ``flash_attn_varlen_func`` on the flat (B*L) views with
                                 cu_seqlens = arange(0, (B+1)*L, L) cached per shape — the stock's kernel call with the
                                 stock's cu_seqlens values, without the unpad gathers, the pad scatter and the host syncs
    A2_cheap_blockmask           GLOBAL layers: ``flex_attention`` under the index-only all-true BlockMask of the stock
                                 document mask's block structure, built once per shape on the CPU — no per-forward
                                 ``create_block_mask`` and no per-score ``sequence_ids`` gathers in the partial blocks
    A3_flex_kernel_options       the GLOBAL layers' flex kernel options (FLEX_KERNEL_OPTIONS), bitwise with
                                 the stock's autotuned pick where that pick has the same BLOCK_N (the same per-row
                                 reduction order) — keyed by GPU class in kits/v1_2
Any other batch (padding, multi-sequence rows, a prefilled cache) takes the stock code path with the unpad index data
computed once per forward. The stock's own kernels (flash_attn 2.8.3.post1, torch 2.8.0 flex template) do the math;
the kit changes data movement and the mask object only.

Pins, the autotune pin and the stock forward come from ``engines.e1.kits``.
"""
from __future__ import annotations

import os

from .. import pins
from . import adapter
from .adapter import Config

KIT = "attn"
LEVERS = ("A1_varlen_shape_cu_seqlens", "A2_cheap_blockmask", "A3_flex_kernel_options")
TESTED_SHAPES = pins.PROBE_SHAPES
#: the padded mixed-length batch (``inputs.make_rows_mixed`` / ``parse_shape`` token ``8x256m``) that exercises the kit's FALLBACK
#: path (the stock code path with the unpad index data computed once per forward)
FALLBACK_SHAPES = ((8, "256m"),)
#: the GLOBAL layers' flex kernel options: the same BLOCK_N as the stock's autotuned pick (torch 2.8.0
#: template_heuristics.py h100_default_flex_config for (bf16, head_dim 64): BLOCK_M 128, BLOCK_N 128, num_warps 4, num_stages 3,
#: read back from the stock's generated kernel source) — the online-softmax tile order is unchanged, so the outputs are bitwise
#: with the stock's; BLOCK_M and num_stages change only the query tiling and the pipelining
FLEX_KERNEL_OPTIONS: dict = {"BLOCK_M": 64, "BLOCK_N": 128, "num_warps": 4, "num_stages": 2}
CONFIG = Config(within="fa2_varlen", global_="flex", flex_kernel_options=tuple(sorted(FLEX_KERNEL_OPTIONS.items())),
                flex_block_mask="cheap", flex_dynamic=True, unpad_once=True)
_HERE = os.path.dirname(os.path.abspath(__file__))
_state: dict = {"applied": False, "size": None, "num_warps": None, "model_id": None, "expected": None, "record": None, "sanitizer": None}


class KitRefused(RuntimeError):
    """The kit refuses to run: a pin or a counter expectation does not hold."""


def kit_files() -> dict:
    """{relative path: present} for the kit's source files (``__init__.py`` + ``adapter.py``) — presence, not bytes."""
    return {f: os.path.isfile(os.path.join(_HERE, f)) for f in ("__init__.py", "adapter.py")}


def counters_snapshot() -> dict:
    return adapter.counters()


def counters_delta(prev: dict) -> dict:
    return adapter.counters_delta(prev)


def _is_padded(L) -> bool:
    return isinstance(L, str) and L.endswith("m")


def expected_per_forward(model, B: int, L, padded: bool | None = None) -> dict:
    """What ONE forward must count. Dense (B, L): every within-seq layer on the varlen route, every global layer on the flex
    route, one cheap-mask lookup, nothing on a stock path. Padded (``L`` = '256m' or ``padded=True``): the fallback = the
    stock path with the unpad index data computed once and reused by the other within-seq layers, the stock mask cached per
    content (a miss the first time a content is seen, hits afterwards)."""
    lc = adapter.layer_counts(model)
    padded = _is_padded(L) if padded is None else padded
    if padded:
        return {"_mode": "fallback", "forwards": 1, "forwards_dense": 0, "within:stock": lc["within"], "global:stock": lc["global"],
                "unpad_computed": 1, "unpad_reused": lc["within"] - 1, "unpad_identity": 0, "pad_identity": 0, "mask_cheap": 0,
                f"within:{CONFIG.within}": 0, f"global:{CONFIG.global_}": 0, "n_within": lc["within"], "n_global": lc["global"]}
    return {"_mode": "dense", "forwards": 1, "forwards_dense": 1, f"within:{CONFIG.within}": lc["within"], f"global:{CONFIG.global_}": lc["global"],
            "mask_cheap": 1, "within:stock": 0, "global:stock": 0, "unpad_computed": 0, "unpad_reused": 0, "unpad_identity": 0,
            "pad_identity": 0, "mask_cached_miss": 0, "mask_cached_hit": 0, "mask_stock": 0, "mask_none": 0,
            "n_within": lc["within"], "n_global": lc["global"]}


def _patches_in_force() -> bool:
    return adapter._S["cfg"] is not None and not any(adapter.is_pristine().values())


def apply(model, *, size: str, num_warps: int | None = None, require_gpu: bool = True, det: bool = False,
          config: Config = CONFIG) -> dict:
    """Pins asserted -> autotune pin (W or an explicit censused W) -> the attention adapter in force -> record.
    The model must be the stock model (fp32 weights) BEFORE any forward in this process."""
    import torch
    if _state["applied"]:
        raise KitRefused("kit attn already applied in this process")
    if require_gpu and not torch.cuda.is_available():
        raise KitRefused("kit attn needs a GPU")
    stack = pins.stack_report()
    kern = pins.kernel_revision_report()
    if size not in pins.WEIGHTS:
        raise KitRefused(f"unknown size {size!r}")
    dtypes = {str(p.dtype) for p in model.parameters()}
    if dtypes != {"torch.float32"}:
        raise KitRefused(f"the stock model holds fp32 weights before the kit (tools/score.py dtype=torch.float); got {sorted(dtypes)}")
    if config.within == "stock" and config.global_ == "stock":
        raise KitRefused("a stock-only config is not a kit")
    pin = pins.apply_autotune_pin(size, num_warps=num_warps)
    arec = adapter.apply(model, config)
    adapter.reset_counters()
    _state.update(applied=True, size=size, num_warps=pin["num_warps"], model_id=id(model))
    rec = {"kit": KIT, "levers": LEVERS, "kit_files": kit_files(), "size": size, "autotune_pin": pin, "stack": stack,
           "kernel": kern, "config": config.as_dict(), "adapter": arec, "tested_shapes": TESTED_SHAPES, "fallback_shapes": FALLBACK_SHAPES,
           "cpu_model": pins.cpu_model(), "inductor": pins.inductor_readback()}
    _state["record"] = rec
    return rec


def unapply() -> None:
    """Restore every patched attribute and drop the hook (LIFO across kits)."""
    adapter.unapply()
    _state.update(applied=False, size=None, num_warps=None, model_id=None, expected=None)


def stamp() -> dict:
    """The fail-closed stamp: refuses (raises) unless the kit is applied and its patches are in force."""
    import torch
    import E1.modeling as M

    def framework_numerics_readback() -> dict:                      # the process's torch numerics switches, read back (report-only)
        return {"allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "float32_matmul_precision": torch.get_float32_matmul_precision()}
    if not _state["applied"] or not _patches_in_force():
        raise KitRefused("stamp: kit attn is not in force in this process")
    return {"kit": KIT, "kit_files": kit_files(), "levers": LEVERS, "config": CONFIG.as_dict(),
            "size": _state["size"], "num_warps": _state["num_warps"], "rmsnorm_path": "triton_layer_norm_kernel",
            "rmsnorm_snapshot": pins.KERNEL["rev"], "rmsnorm_file": getattr(M.layer_norm, "__file__", None),
            "autotune_readback": pins.autotune_readback(), "numerics": framework_numerics_readback(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "counters": counters_snapshot(), "tested_shapes": TESTED_SHAPES, "fallback_shapes": FALLBACK_SHAPES,
            "inductor": pins.inductor_readback(), "cpu_model": pins.cpu_model(), "sanitizer": _state["sanitizer"]}
