"""Shared fixtures for the CPU tests: where the kit is, a temporary tree with a stock/PINS.json of the agreed SCHEMA (placeholder
values — the tests lock the schema and the gates, never a pin's value), a stubbed box (the pinned versions + a pinned GPU name, so the
gates pass without a GPU), a small stock Borzoi that needs no GPU and no flash-attn (config.flashed=False), and a stub kit package whose
KitRunner records its calls and installs the instance-level forward the way the real wrapper does (its tables COPIED at test time from
the kit's own file through modes.kit_table, never typed in).

The kit is found through the package's own lookup (stack.kit_root(): opt/forward/kits_v1_25 under the tree, or
opt/forward/kits_v1_25 under the tree); tests that need it call `require_kit()` and SKIP with a printed reason when it is absent (the kit files are carried
by the tree, not by this package). torch + borzoi-pytorch are needed by the tests that build a model (`require_upstream()`).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import textwrap
import unittest

from flashzoi_opt import modes, stack

PINNED = {"borzoi-pytorch": "0.5.1", "torch": "2.5.1+cu124", "triton": "3.1.0", "flash-attn": "2.7.0.post2", "transformers": "4.57.6", "python": "3.11.12"}
PINNED_GPU = None            # filled from the kit's PINS at test time (require_kit)


def kit_or_none():
    try:
        return stack.kit_root()
    except FileNotFoundError:
        return None


def require_kit():
    k = kit_or_none()
    if k is None:
        raise unittest.SkipTest("kit not found (no opt/forward/kits_v1_25 under the tree): kit-dependent test skipped")
    return k


def require_upstream():
    try:
        import torch  # noqa: F401
        import borzoi_pytorch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise unittest.SkipTest(f"torch / borzoi-pytorch not importable: {e!r}")


def pins_stub() -> dict:
    """A stock/PINS.json of the agreed schema with placeholder values (schema lock only)."""
    return {"package": {"name": "borzoi-pytorch", "version": PINNED["borzoi-pytorch"]},
            "stack": {"torch": PINNED["torch"].split("+")[0], "triton": PINNED["triton"], "flash-attn": PINNED["flash-attn"], "transformers": PINNED["transformers"], "python": "3.11"},
            "weights": {f"johahi/flashzoi-replicate-{k}": {"revision": f"{k:040x}", "model.safetensors_sha256": f"{k:064x}", "bytes": 4} for k in range(4)},
            "config_json_sha256": "f" * 64}


class Tree:
    """A temporary flashzoi/ tree: stock/PINS.json (stub) and, when asked, a stub kit under opt/forward/kits_v1_25."""

    def __init__(self, with_pins: bool = True, kit: str | None = None):
        self.root = tempfile.mkdtemp(prefix="flashzoi-tree-")
        os.makedirs(os.path.join(self.root, "stock"))
        if with_pins:
            json.dump(pins_stub(), open(os.path.join(self.root, "stock", "PINS.json"), "w"), indent=1)
        self.kit = kit
        self._saved = {}

    def enter(self):
        for k in (stack.ENV_TREE,):
            self._saved[k] = os.environ.pop(k, None)
        os.environ[stack.ENV_TREE] = self.root
        if self.kit:                                                   # the kit at the tree's one relpath (a symlink to the given kit dir)
            link = os.path.join(self.root, stack.KIT_RELPATH); os.makedirs(os.path.dirname(link), exist_ok=True); os.symlink(os.path.abspath(self.kit), link)
        return self

    def exit(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.root, ignore_errors=True)


STUB_CAPABILITY = {"NVIDIA L40S": ("8.9", 46068), "NVIDIA A100-SXM4-80GB": ("8.0", 81920), "NVIDIA A100 80GB PCIe": ("8.0", 81920)}   # (cc, total memory MiB) of the named stub devices


def stub_box(gpu_name: str | None = None, versions: dict | None = None):
    """Patch the gates' probes: the pinned versions installed, one pinned GPU visible. Returns the undo function."""
    v = dict(PINNED); v.update(versions or {})
    name = gpu_name or PINNED_GPU or "NVIDIA H100 80GB HBM3"
    saved = (stack.installed_versions, stack.torch_gpu_info)
    stack.installed_versions = lambda: dict(v)
    cap = STUB_CAPABILITY.get(name, ("9.0", 81559))                       # the stub device's compute capability + total memory follow its name (a name not listed = the H100 capability)
    stack.torch_gpu_info = lambda: {"probe": "stub", "name": name, "cc": cap[0], "sm": cap[0].replace(".", ""), "memory_mib": cap[1], "count": 1, "torch": v["torch"], "cuda": "12.4", "error": None}

    def undo():
        stack.installed_versions, stack.torch_gpu_info = saved
    return undo


def reset_activation():
    """Fresh activation state in this process (the tests share one interpreter)."""
    import gc
    from flashzoi_opt import report
    stack._uninstall_hook()
    gc.collect()
    stack._REPORT = None
    stack._ATTACHED.clear(); stack._RUNNERS.clear(); stack._MODELS.clear()
    stack._INSTANCES_AT_ENABLE["n"] = None
    report._ACTIVE_PRINTED["done"] = False
    report.ITEMS.update(items=0, ok=0, failed=0)
    os.environ.pop(modes.ENV_MODE, None)
    for k in list(os.environ):
        if k in stack.KIT_ENV_SWITCHES:
            del os.environ[k]


def small_borzoi(device: str = "cpu"):
    """A stock Borzoi with the smallest config that constructs (flashed=False: the FlashAttention branch imports flash_attn lazily)."""
    require_upstream()
    from borzoi_pytorch import Borzoi
    from borzoi_pytorch.config_borzoi import BorzoiConfig
    cfg = BorzoiConfig(dim=32, depth=1, heads=2, flashed=False)
    return Borzoi(cfg).to(device).eval()


STUB_KIT_WRAP = '''
"""A stub of the kit's wrapper for the CPU tests: KitRunner records its construction and installs the instance-level forward + the
attach mark the way the real wrapper does (its refusal of the stock forward on a kit-changed module); remove restores. The tables are the kit's, passed in."""
import types
CALLS = []
class KitRunner:
    def __init__(self, model, components=None, device="cuda", batch=1, pool_size=None, numerics="tf32"):
        if numerics not in ("tf32",):
            raise ValueError(f"numerics={numerics!r}")
        self.model = model; self.components = frozenset(LEVERS if components is None else components); self.numerics_knob = numerics
        self.route_class = "stub: exact route (the stock's TF32 class inside each call)"
        self.device_name = DEVICE_NAME; self.class_label = "exact"
        self.apply_s = {"patch_s": 0.0, "apply_s": 0.0, "runner_in_process": len(CALLS) + 1}; self.apply_line = "[stub kit] applied"
        self.refused_surfaces = []; self._helper_route = "stub: not routed"; self.forwards = 0
        runner = self
        def _forward(self_, x, is_human=True, data_parallel_training=False, return_embeddings=False):
            runner.forwards += 1
            return ("kit", x.shape if hasattr(x, "shape") else x)
        model._flashzoi_kit_runner = self
        model.forward = types.MethodType(_forward, model)
        CALLS.append({"model": id(model), "numerics": numerics})
        print(self.apply_line, flush=True)
    def effective_flags(self):
        return {lv: True for lv in self.components}
    def all_counts(self):
        return {"predict_calls": self.forwards}
    def jit_after_job(self):
        return {"compile_calls": 0}
    def close(self):
        self.model.__dict__.pop("forward", None); self.model.__dict__.pop("_flashzoi_kit_runner", None)
        return {"forward_restored": True}
def describe(components=None, device_names=None):
    return {"arm": ARM, "components": sorted(components or LEVERS), "per_component": {lv: LEVER_CLASS[lv] for lv in (components or LEVERS)}, "pins": dict(PINS)}
def remove(runners, **kw):
    return {f"runner{i}": r.close() for i, r in enumerate(runners)}
'''


def write_stub_kit(root: str, table: dict, device_name: str) -> str:
    """A stub kit package under `root` (opt/forward/kits_v1_25/engines/flashzoi/kits/v1_25) carrying the kit's tables verbatim."""
    kit_root = os.path.join(root, modes.KIT_RELPATH)
    pkg = os.path.join(kit_root, "engines", "flashzoi", "kits", "v1_25")
    os.makedirs(pkg)
    for d in ("engines", os.path.join("engines", "flashzoi"), os.path.join("engines", "flashzoi", "kits")):
        open(os.path.join(kit_root, d, "__init__.py"), "w").close()
    init = (f"LEVERS = {table['LEVERS']!r}\nALL = frozenset(LEVERS)\nPINS = {table['PINS']!r}\nLEVER_CLASS = {table['LEVER_CLASS']!r}\nARM = {table['ARM']!r}\n"
            f"DEVICE_NAME = {device_name!r}\n"
            "from ._wrap import KitRunner, describe, remove, CALLS\n")
    open(os.path.join(pkg, "__init__.py"), "w").write(init)
    wrap = "from . import LEVERS, LEVER_CLASS, PINS, ARM, DEVICE_NAME\n" + textwrap.dedent(STUB_KIT_WRAP)   # the names exist before __init__ imports _wrap
    open(os.path.join(pkg, "_wrap.py"), "w").write(wrap)
    return kit_root


def forget_kit_modules():
    for m in list(sys.modules):
        if m.split(".")[0] in stack.KIT_NAMESPACES:
            del sys.modules[m]
    for p in list(sys.path):
        if os.path.isdir(os.path.join(p or ".", "engines", "flashzoi", "kits")):
            sys.path.remove(p)
