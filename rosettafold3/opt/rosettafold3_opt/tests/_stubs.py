"""Stubs for the CPU tests: site-packages trees in a named state, a second "interpreter", a fake rc-foundry dist, a fake GPU.

A stub tree is a directory holding ``rf3/`` and ``foundry/`` with the real bytes of the state: the pristine files come from
``stock/src/`` (the one copy of the pristine bytes), the patched ones from the carried kit's ``patched/rf3/``. A stub interpreter
is a shell wrapper that sets PYTHONPATH to a stub tree and execs this interpreter, so ``tree.state_of(python)`` and the kit's own
``install.sh`` (which locates site-packages with ``python -c 'import rf3'``) see that tree. No torch, no GPU, no network.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import sys

import pytest

from .. import modes as _modes
from .. import stack as _stack
from .. import tree as _tree

CORE_GLOO_DENIED = pytest.mark.skip(
    reason="needs torch.distributed's gloo TCP rendezvous, which restricted test environments deny (RuntimeError at gloo "
           "tcp/device.cc, 'Operation not permitted'); the row-sharded line is exercised on GPU boxes, not in the CPU suite."
)

PRISTINE = {
    "rf3/diffusion_samplers/inference_sampler.py": "models/rf3/src/rf3/diffusion_samplers/inference_sampler.py",
    "rf3/model/RF3_structure.py": "models/rf3/src/rf3/model/RF3_structure.py",
    "rf3/model/layers/af3_diffusion_transformer.py": "models/rf3/src/rf3/model/layers/af3_diffusion_transformer.py",
    "rf3/loss/loss.py": "models/rf3/src/rf3/loss/loss.py",
    "foundry/utils/torch.py": "src/foundry/utils/torch.py",
}
PIN = "4010e3e2e7350edada3e25a45c908c6bf407df4d"


def tree_root() -> str:
    return _stack.tree_root()


def stock_src() -> str:
    return os.path.join(tree_root(), "stock", "src")


def _touch_packages(root: str) -> None:
    for d in ("rf3", "rf3/diffusion_samplers", "rf3/model", "rf3/model/layers", "rf3/loss", "rf3/inference_engines", "foundry", "foundry/utils"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
        init = os.path.join(root, d, "__init__.py")
        if not os.path.exists(init):
            open(init, "w").write("")


def make_tree(root: str, state: str = "stock") -> str:
    """A stub site-packages at ``root`` in state ``stock`` | ``patched`` | ``unknown``."""
    _touch_packages(root)
    for rel, src in PRISTINE.items():
        dst = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(stock_src(), src), dst)
    if state in ("patched", "unknown"):
        kit = os.path.join(_stack.kit_home(), "patched")
        for rel in _tree.RF3_FILES:
            shutil.copyfile(os.path.join(kit, rel), os.path.join(root, rel))
    if state == "unknown":
        with open(os.path.join(root, "rf3/loss/loss.py"), "a") as fh:
            fh.write("\n# modified\n")
    # a graph_flags stub is NOT written for stock (absent upstream); the patched state carries the kit's own file
    return root


def make_interpreter(bin_dir: str, tree: str, name: str = "python", hook: bool = False, isolated: bool = False) -> str:
    """A wrapper on this interpreter with PYTHONPATH=<tree> (several directories joined with os.pathsep are passed through); returns
    its path. ``hook=True`` gives the child what the installed package's ``.pth`` would: ``opt/`` on its path and a
    ``sitecustomize.py`` that imports ``rosettafold3_opt._autoload`` at start (the tests need no pip install). ``isolated=True``
    runs it with ``-S`` (no site-packages): for a test that asserts a package is ABSENT, which must hold on an interpreter that
    carries the pin (the proof box's) as on one that does not."""
    os.makedirs(bin_dir, exist_ok=True)
    p = os.path.join(bin_dir, name)
    path = tree
    if hook:
        site = os.path.join(os.path.abspath(bin_dir), "hook_site")
        os.makedirs(site, exist_ok=True)
        open(os.path.join(site, "sitecustomize.py"), "w").write("import rosettafold3_opt._autoload\n")
        path = os.pathsep.join([tree, _stack.opt_root(), site])
    with open(p, "w") as fh:
        fh.write(f'#!/bin/sh\nexport PYTHONPATH="{path}${{PYTHONPATH:+:$PYTHONPATH}}"\nexport PATH="{os.path.abspath(bin_dir)}:$PATH"\nexec "{sys.executable}"{" -S" if isolated else ""} "$@"\n')
    os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    for extra in ("python3",):
        q = os.path.join(bin_dir, extra)
        if not os.path.exists(q):
            os.symlink(p, q)
    return p


def make_dist(root: str, commit: str = PIN, version: str = "0.2.1.dev13+g4010e3e2e", route: str = "vcs", archive_sha: str | None = None) -> str:
    """A fake ``rc-foundry`` dist-info under ``root`` (for importlib.metadata) with a direct_url.json of the given route."""
    d = os.path.join(root, f"rc_foundry-{version}.dist-info")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "METADATA"), "w").write(f"Metadata-Version: 2.1\nName: rc-foundry\nVersion: {version}\n")
    if route == "vcs":
        du = {"url": "https://github.com/RosettaCommons/foundry.git", "vcs_info": {"commit_id": commit, "requested_revision": commit, "vcs": "git"}}
    elif route == "archive":
        du = {"url": "file:///x/stock/foundry-4010e3e2e.tar.gz", "archive_info": {"hash": f"sha256={archive_sha}"}}
    else:
        du = None
    if du is not None:
        json.dump(du, open(os.path.join(d, "direct_url.json"), "w"))
    return d


def fake_nvidia_smi(bin_dir: str) -> str:
    """An ``nvidia-smi`` on the stub interpreter's PATH that answers the package's query with an H100 row (stack.gpu_info)."""
    os.makedirs(bin_dir, exist_ok=True)
    p = os.path.join(bin_dir, "nvidia-smi")
    with open(p, "w") as fh:
        fh.write('#!/bin/sh\necho "NVIDIA H100 80GB HBM3, 9.0, 81559"\n')
    os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return p


FAKE_GPU = {"name": "NVIDIA H100 80GB HBM3", "cc": "9.0", "sm": "sm90", "memory_mib": 81559}


FPF_ADAPTER_STUB = '''"""stub of rf3fpf/fpf_rf3_adapter.py for the CPU tests: the add-on's API surface (apply_arm / describe / describe_v2), no torch."""
import os
CALLS = []
COUNTS = {"triattn": {}, "transition": {}, "apb": {}}
STATE = {"mode": "stock", "served": 0, "fallback": {}, "errors": 0,
         "cfg": {"triattn": "stock", "transition": "stock", "apb": "stock", "levers": None, "warm": False, "trunk_graph": False, "dattn": False, "res": False, "msa": False, "msa_units": [], "msa_aside": {}, "smsa": False},
         "trunk_graph": {"on": False, "n_graphs": 0, "captures": [], "replays": 0, "fallbacks": {}}, "dattn": {"on": False, "calls": 0, "fallback": 0},
         "res": {"on": False, "fused": 0, "unfused": 0}}
PAD = 16; TG_MAX = int(os.environ.get("FPF_RF3_TG_MAX", "6"))
FAIL = os.environ.get("FPF_STUB_FAIL")           # test knob: "raise" | "mismatch" | "silent" | "levers" (the @L step finds no kit module)
TG_FALLBACKS = {k: int(v) for k, v in (kv.split("=") for kv in os.environ.get("FPF_STUB_TG_FALLBACKS", "").split(",") if "=" in kv)}   # test knob: "capture_failed=1,evictions=2"
KERNEL_ERROR = os.environ.get("FPF_STUB_KERNEL_ERROR") == "1"                                                                       # test knob: one triattn kernel-error key
def apply_arm(arm):
    CALLS.append(arm)
    if FAIL == "raise":
        raise RuntimeError("stub adapter refused " + arm)
    body, _, lv = arm.partition("@"); parts = [p for p in body.split("+") if p] or ["stock"]
    cfg = STATE["cfg"]
    for p0 in parts:
        p = p0.partition(".")[0]
        if p in ("stock", "fast"): STATE["mode"] = p
        elif p == "gflash": cfg["triattn"] = p
        elif p == "xmul": cfg["xmul"] = p0.partition(".")[2] or "exact"; STATE.setdefault("xmul", {})["on"] = True
        elif p == "ttr": cfg["transition"] = "triton"
        elif p == "apb":
            cfg["apb"] = "triton"
            if p0.partition(".")[2] and p0.partition(".")[2] != "stmt": cfg["apb_word"] = p0.partition(".")[2]
        elif p == "sapb": cfg["apb"] = "safe"
        elif p == "xln": cfg["xln"] = p0.partition(".")[2] or "exact"; STATE.setdefault("xln", {})["on"] = True
        elif p == "xatt": cfg["xatt"] = p0.partition(".")[2] or "exact"
        elif p == "msa":                                  # msa[.<word>]: the word; the stub's card row = both cells for the bare word (a 9.0 box), the named cell otherwise
            w = p0.partition(".")[2] or "card"; cfg["msa"] = w; cfg["msa_units"] = ["opm", "pwa"] if w in ("card", "both") else [w]; cfg["msa_aside"] = {}; STATE.setdefault("msa", {})["on"] = True
        elif p == "smsa": cfg["smsa"] = True; STATE.setdefault("smsa", {})["on"] = True
        elif p == "tg": cfg["trunk_graph"] = True
        elif p == "dattn": cfg["dattn"] = True
        elif p == "res": cfg["res"] = True
    if FAIL == "mismatch":
        cfg["transition"] = "stock"
    step, *subs = (lv.split(".") if lv else ("",))
    cfg["warm"] = False
    if step == "L1":                                         # the adapter's set_kit_levers: module attributes of rf3.graph_flags, never the environment
        lv = "L1"
        try:
            if FAIL == "levers":
                raise ImportError("stub: the kit module is unavailable")
            import rf3.graph_flags as GF
        except Exception:
            cfg["levers"] = "unavailable"
        else:
            GF.CUDAGRAPH_MODE = "1"; GF.HOIST = True; GF.GRAPH_SAFE_OPS = True
            cfg["levers"] = True
            if "warm" in subs:
                if hasattr(GF, "set_levers"): GF.set_levers(warm=True)
                cfg["warm"] = True
    STATE["trunk_graph"]["on"] = cfg["trunk_graph"]; STATE["dattn"]["on"] = cfg["dattn"]; STATE["res"]["on"] = cfg["res"]
    return dict(cfg, trimul=STATE["mode"])
def fold():
    """what one fold would leave in the counters (the tests call it in place of a model run)"""
    if FAIL == "silent":
        return
    STATE["served"] += 96
    cfg = STATE["cfg"]
    if cfg["triattn"] != "stock": COUNTS["triattn"]["served:" + cfg["triattn"]] = 96
    if cfg["transition"] != "stock": COUNTS["transition"]["served:" + cfg["transition"]] = 48
    if cfg["apb"] != "stock": COUNTS["apb"]["served:" + cfg["apb"]] = 48
    if cfg["trunk_graph"]: STATE["trunk_graph"].update({"n_graphs": 1, "captures": [{"I": 199}], "replays": 10, "fallbacks": dict(TG_FALLBACKS)})
    if cfg["dattn"]: STATE["dattn"].update({"calls": 200})
    if cfg["res"]: STATE["res"].update({"fused": 144, "unfused": 4})
    if cfg.get("xatt"): STATE["xatt_served"] = 1040
    for comp in ("xmul", "xln", "msa", "smsa"):    # the provider-row / MSA-module components: served counts in their own describe_v2 sections (registry probes fpf_v2 <comp> served)
        if STATE.get(comp, {}).get("on"): STATE[comp]["served"] = 96
    if KERNEL_ERROR: COUNTS["triattn"]["kernel-error->contiguous:RuntimeError('stub')"] = 1
def describe():
    return {"mode": STATE["mode"], "served": STATE["served"], "fallback": dict(STATE["fallback"]), "errors": STATE["errors"], "pad": PAD}
def describe_v2():
    return {"cfg": dict(STATE["cfg"]), "counts": {k: dict(v) for k, v in COUNTS.items()}, "trunk_graph": dict(STATE["trunk_graph"]),
            "dattn": dict(STATE["dattn"]), "res": dict(STATE["res"]), "xatt": {"word": STATE["cfg"].get("xatt"), "served": STATE.get("xatt_served", 0)},
            **{comp: dict(STATE[comp]) for comp in ("xmul", "xln", "msa", "smsa") if comp in STATE}}
'''


def fpf_stub(root: str) -> str:
    """A stub FPF add-on directory at ``root``: every file stack.fpf_present() needs (empty) and a stub
    adapter with the add-on's API surface (``FPF_ADAPTER_STUB``: apply_arm / describe / describe_v2, the environment knobs, no torch)."""
    for rel in _modes.FPF_RUNTIME_MEMBERS:
        if rel == _modes.FPF_ADAPTER_RELPATH.replace(os.sep, "/"):
            continue                                          # written below: the stub adapter
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write("")
    os.makedirs(os.path.join(root, "rf3fpf"), exist_ok=True)
    open(os.path.join(root, "rf3fpf", "fpf_rf3_adapter.py"), "w").write(FPF_ADAPTER_STUB)
    return root


def rf3_structure_stub(tree: str) -> str:
    """Overwrite ``rf3/model/RF3_structure.py`` of a stub tree with an importable marker module (the FPF watch's trigger)."""
    p = os.path.join(tree, "rf3", "model", "RF3_structure.py")
    open(p, "w").write("STUB = True\n")
    return p


def graph_flags_stub(tree: str, body: str | None = None) -> str:
    """Overwrite ``rf3/graph_flags.py`` of a stub tree with a tiny module whose describe() reads the environment like the kit's
    (graph_flags.py:33,41,142-151) — for the apply-watch and exit-tally tests only (the tree is then 'unknown' by sha)."""
    p = os.path.join(tree, "rf3", "graph_flags.py")
    open(p, "w").write(body or (
        "import os\n"
        "def _norm(v):\n    v=(v or '').strip().lower()\n    return '1' if v in ('1','true','on','graph','cudagraph') else ('replay' if v in ('replay','eager','predraw') else '0')\n"
        "CUDAGRAPH_MODE=_norm(os.environ.get('RF3_CUDAGRAPH','1'))\n"
        "GRAPH_SAFE_OPS=(CUDAGRAPH_MODE=='1')\nGRAPH_WARMUP=3\n"
        "HOIST=os.environ.get('RF3_HOIST','0').strip().lower() in ('1','true','on')\n"
        "HOIST_STATS={'rollouts': 0, 'entries_last': 0}\nLAST_CAPTURE={}\n"
        "def hoist_begin(n_calls=None):\n    HOIST_STATS['rollouts'] += 1\ndef hoist_end():\n    pass\n"                    # the roll-out seams (graph_flags.py; mem.py wraps them)
        "def describe():\n    return dict(RF3_CUDAGRAPH=CUDAGRAPH_MODE, RF3_GRAPH_SAFE_OPS=GRAPH_SAFE_OPS, RF3_CUDAGRAPH_WARMUP=GRAPH_WARMUP, RF3_HOIST=HOIST)\n"))
    return p


DTK_DESCRIBE = {"on": True, "impl": "stub", "origin": "core", "min_tokens": 400, "bias": "hij", "census": {"calls": 0, "served": 0, "gated": 0}, "ok": True, "reason": None}


def dtk_stub(monkeypatch) -> None:
    """Stand the package's dtk lever in for the CPU tests: ``dtk.enable`` returns an installed describe() without touching rf3's diffusion
    transformer (the real lever recompiles AttentionPairBiasDiffusion.forward around the FPF add-on's source seam and imports the core's GPU
    kernel — GPU-only). The fast mode and its rows name dtk, so every CPU activation of them runs through this."""
    from .. import dtk as _dtk
    monkeypatch.setattr(_dtk, "enable", lambda **kw: dict(DTK_DESCRIBE))
    monkeypatch.setattr(_dtk, "describe", lambda: dict(DTK_DESCRIBE))
    monkeypatch.setattr(_dtk, "problems", lambda: [])


TGB_DESCRIBE = {"on": True, "max_i": 1000, "rule": "tg:budget (stub)", "census": {"graphed": 0, "skipped": 0, "by_key": {}}, "reason": None}


def tgb_stub(monkeypatch) -> None:
    """Stand the trunk-graph token budget in for the CPU tests: ``tgbudget.enable`` wraps rf3's Recycler.forward (GPU-box evidence,
    GPU-only); here it reports installed without importing rf3."""
    from .. import tgbudget as _tgb
    monkeypatch.setattr(_tgb, "enable", lambda adapter, environ=None: dict(TGB_DESCRIBE))


XTR_DESCRIBE = {"on": True, "impl": "stub", "origin": "core", "construction": "pair_fused.transition(impl=fpf,ln=stock)", "serve": ["128x512", "384x1536"],
                "routes": {"128x512": "serve:pf:fpf", "384x1536": "serve:pf:fpf"}, "cells_sha256": None, "prev": None,
                "census": {"calls": 0, "served": 0, "routed": 0, "fallback": 0, "by_key": {}}, "ok": True, "reason": None}


def xtr_stub(monkeypatch) -> None:
    """Stand the package's xtr lever in for the CPU tests: ``pf.enable`` needs torch + rf3's Transition class and the core's serve layer on a
    GPU (GPU-only); the exact mode names xtr, so every CPU activation of it runs through this."""
    from .. import pf as _pf
    monkeypatch.setattr(_pf, "enable", lambda environ=None: dict(XTR_DESCRIBE))
    monkeypatch.setattr(_pf, "describe", lambda: dict(XTR_DESCRIBE))
    monkeypatch.setattr(_pf, "problems", lambda: [])


MKDIT_DESCRIBE = {"on": True, "impl": "stub", "origin": "kit", "files_sha256": {}, "config": {}, "gpu": None, "min_tokens": 400,
                  "census": {"calls": 0, "served": 0, "gated": 0, "samples": 0, "hoists": 0, "atom_calls": 0, "objs": 0, "shapes": {}}, "ok": True, "reason": None}


def mkdit_stub(monkeypatch) -> None:
    """Stand the package's mkdit lever in for the CPU activation tests: ``mkdit.enable`` replaces rf3's DiffusionTransformer.forward and imports
    the carried Triton megakernel (GPU-only; the adapter's own CPU tests are tests/test_mkdit.py). The fast mode names
    mkdit, so every CPU activation of it runs through this."""
    from .. import mkdit as _mkdit
    monkeypatch.setattr(_mkdit, "enable", lambda **kw: dict(MKDIT_DESCRIBE))
    monkeypatch.setattr(_mkdit, "describe", lambda: dict(MKDIT_DESCRIBE))
    monkeypatch.setattr(_mkdit, "problems", lambda: [])

