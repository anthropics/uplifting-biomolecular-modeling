"""Lever `pred_release` (EXACT, every mode): the previous item's prediction tensors leave the device before the
next item's forward.

The stock loop (`runner/inference.py` infer_predict in the pinned 2.0.0 tree) is `prediction = runner.predict(data)` ->
`runner.dumper.dump(..., pred_dict=prediction, ...)` (synchronous: the CIF / confidence JSON files are written before it returns) -> log ->
`torch.cuda.empty_cache()` -> next iteration. The local name `prediction` stays bound while the NEXT item's `runner.predict` runs, so item k's whole
prediction dict — coordinates plus the full-data confidence tensors, of which the PAE and PDE logits are two [N_sample, N_token, N_token, 64] fp32
tensors, about 2.4e-6·N² GiB — rides through item k+1's trunk, sampler and confidence head. Every multi-item process of every arm pays it.

The lever wraps `InferenceRunner.predict` (the core's fail-closed import-time patch, `opt_core.autoload.patch_attr_at_import`, like sampler_fuse):
at the entry of a call, the dict the PREVIOUS call returned (already dumped — the dump precedes the next predict in the one loop that calls predict;
a caller still holding the dict sees its device tensors replaced by None) has every CUDA tensor dropped and the allocator cache returned; then the
stock predict runs. Outputs are untouched (written before the release): EXACT, bitwise by construction. phase_timing wraps predict later (cli.py), outermost: the release is inside total_s, outside fwd_s.
Lines: the applied marker `PRED_RELEASE:armed|patched` (stack.MARKERS) and, per released item,
`[protenix-opt] PRED-RELEASE item=<previous item> released_gib=<x> allocated_gib=<after>`.
The TP line (--n_gpu P) runs one item per launch through its own entry: nothing to release there (the wrapper is inert, count 0).
Switch: PTX_PRED_RELEASE=1 (modes.PACKAGE_POST for exact and fast; big inherits its base's) — MODEL_OPT_LEVERS_OFF=pred_release removes it.
"""
from __future__ import annotations
import functools

TARGET, CLASS, METHOD = "runner.inference", "InferenceRunner", "predict"
TAG, MARK, ENV = "protenix-opt", "PRED_RELEASE:", "PTX_PRED_RELEASE"
_STATE = {"on": False, "patch": None, "last": None, "last_name": None, "released": [], "error": None}


def from_env(environ=None) -> bool:
    import os
    v = (environ if environ is not None else os.environ).get(ENV, "0")
    if v not in ("", "0", "1"):
        raise ValueError(f"{ENV}={v!r} is not 0|1")
    return v == "1"


def _purge(obj, depth: int = 0) -> int:
    """Replace every CUDA tensor inside the nested prediction container (dicts / lists) by None; returns the bytes of the storages dropped."""
    import torch
    if depth > 8:
        return 0
    n = 0
    items = list(obj.items()) if isinstance(obj, dict) else list(enumerate(obj)) if isinstance(obj, list) else []
    for k, v in items:
        if torch.is_tensor(v):
            if v.is_cuda:
                try:
                    n += v.untyped_storage().nbytes()
                except Exception:
                    pass
                obj[k] = None
        elif isinstance(v, (dict, list)):
            n += _purge(v, depth + 1)
    return n


def _item_name(data) -> str:
    try:
        return str(data.get("sample_name", "?")) if isinstance(data, dict) else "?"
    except Exception:
        return "?"


def make_wrapper(orig):
    """The wrapper factory the core's AttrPatch calls with the stock ``InferenceRunner.predict``."""

    @functools.wraps(orig)
    def predict(self, data, *a, **k):
        import torch
        last, name = _STATE["last"], _STATE["last_name"]
        _STATE["last"] = None
        if last is not None:
            nbytes = _purge(last)
            del last
            alloc = 0.0
            if torch.cuda.is_available():
                torch.cuda.empty_cache(); alloc = torch.cuda.memory_allocated() / 2**30
            _STATE["released"].append((name, nbytes))
            print(f"[protenix-opt] PRED-RELEASE item={name} released_gib={nbytes / 2**30:.2f} allocated_gib={alloc:.2f}", flush=True)
        out = orig(self, data, *a, **k)
        _STATE["last"] = out; _STATE["last_name"] = _item_name(data)
        return out
    predict._pred_release = True
    return predict


def install() -> str:
    """Arm the lever: ``InferenceRunner.predict`` patched now if ``runner.inference`` is imported, else at its import (the core's
    ``patch_attr_at_import``: fail-closed). Returns the applied marker ``PRED_RELEASE:patched|armed``."""
    from opt_core.autoload import patch_attr_at_import
    _STATE["on"] = True
    _STATE["patch"] = patch_attr_at_import(TARGET, f"{CLASS}.{METHOD}", make_wrapper, tag=TAG, name="pred_release")
    return f"{MARK}{'patched' if _STATE['patch'].state == 'installed' else 'armed'}"


def state() -> dict:
    """The lever's end-of-run record: switch on, patch state, items released, GiB released."""
    p = _STATE["patch"]
    rel = _STATE["released"]
    return {"on": _STATE["on"], "patch": (p.state if p is not None else None), "items_released": len(rel),
            "released_gib": round(sum(b for _, b in rel) / 2**30, 3), "error": _STATE["error"]}


def evidence() -> list:
    st = state()
    return [("patch", st["patch"]), ("items_released", st["items_released"]), ("released_gib", st["released_gib"])]
