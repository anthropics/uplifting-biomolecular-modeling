"""Activation: the tree and kit paths, the stock pins, the GPU and its key, the child environment, and the activation report.

Genie 3's optimizations live in the control flow of the kit's resident driver (opt/forward/fast_inference/driver/g3fast.py): the
persistent process, the sync-free DDIM loop of its own, the CUDA-graph capture of the instantiated model, the batching of designs. They
do not attach inside the stock command line's process (`genie3 generate` builds a lightning Trainer and runs its own loop,
src/genie3/generation/workflow.py:178-191; the env route hands that command to the design verb instead, _autoload.py), so `enable(mode)`
resolves and gates the mode on this box and ARMS it for this process's `design()` calls: the report says `attach=driver` and design.py runs the mode's line (stack.driver_command /
child_environment). Nothing is imported from the kit here; the kit's own driver applies the levers in its own process and leaves its
own evidence (registry.Lever.evidence), which design.py reads back.

`activate(mode, dry_run=True)` (= `check`) resolves, gates and reports without touching the environment. Gates (deployment facts only):
the shared core's pin (_core.core_gate: opt/pyproject.toml [tool.opt_core] version against the installed opt_core's version — a floor,
statement one of every entry); the tree and the kit directory present; the checkout found; the weight files the pass reads present (a
MISSING file only: weights_gate); a visible CUDA device for a kit line; the `genie3` console script for the stock line. Reported, never
gated: the stock pins (stock/check_pins.py: the checkout's bytes and HEAD, the torch / lightning / numpy freeze — `pinned=` on the ACTIVE
line, a NOTE line naming any difference; the pass runs on the checkout as installed, as upstream does) and the weight files' digests
against stock/PINS.json weights (`(pinned)` / `NOT PINNED`, memoised on disk: digest_memo). The package classes the card by compute capability and memory band (GPU_CLASSES: the
key the report and the manifest carry) and refuses on no hardware property: a card the kit never measured resolves with its class named,
and a compute capability above the pinned stack's kernel ceiling (MAX_CC_OF_STACK) is NOTED — `[genie3-opt] NOTE compute capability <cc>
above the pinned stack's ceiling … — kernels untested` (hardware_notes, report.note_line; recorded as ``notes``) — and the pass
proceeds: a kernel the stack lacks then fails as the runtime's own error. The late-activation rule: enable() may run any time after import, is
idempotent, and is refused by name once an upstream model instance exists in this process or the kit's patch module reports itself
applied here.
"""
from __future__ import annotations

import gc
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

from . import ActivationError, __version__
from . import _core
from . import report as _report
from .modes import DEFAULT_MODE, MODE_NAMES, ModeError, Resolution, batch_size, effective, precision_of, resolve, trimul_of
from .registry import KIT, KIT_DIRS, PACKAGE

ENV_MODE, ENV_HOME = "GENIE3_OPT", "GENIE3_OPT_HOME"
ENV_TREE = "MODEL_OPT"                                            # run.sh convention: this model's directory (genie3/)
ENV_ROOT, ENV_WEIGHTS = "GENIE3_ROOT", "GENIE3_WEIGHTS"           # the checkout (cwd of every line) and the directory holding checkpoints/step=600000.ckpt + config.yaml
ENV_TRITON_CACHE, ENV_TARGET_GPU = "TRITON_CACHE_DIR", "MODEL_OPT_TARGET_GPU"
UPSTREAM_DIST = "genie3"
MODEL_MODULE, MODEL_CLASS = "genie3.generation.model.implementation.base", "Denoiser"
KIT_PATCH_MODULE = "g3fast_patches"                               # the driver's patch module (driver/g3fast_patches.py); its apply() leaves the in-process markers below
PAIR_MODULE, GEO_MODULE = "genie3.generation.model.embedder.pair.v1", "genie3.generation.utils.geo_utils"   # the two upstream modules apply() marks / rebinds
# variables the package drops from every child process it starts (the driver line, the stock caller) so that the
# mode table stays the only source of a switch: the package's own switches by prefix (GENIE3_OPT would arm the .pth finder in a child
# that imports genie3), CUDA MPS client settings by prefix (no line of this tree runs under MPS; a stock child and a kit child see the
# same device the same way) and the CUDA libraries' TF32 override by name (read by cuBLAS / cuDNN, not by torch's flags; it would defeat
# the strict fp32 of the stock and exact lines on every arm)
DROP_ENV_PREFIXES = ("GENIE3_OPT", "GENIE3_", "CUDA_MPS_", "TORCH_ALLOW_TF32_")   # TORCH_ALLOW_TF32_*: torch's TF32 matmul override — a kit line's numerics are its mode's, never a library override's (the driver itself refuses one present, g3batch.py)
DROP_ENV_NAMES = ("NVIDIA_TF32_OVERRIDE",)
KEEP_ENV = (ENV_ROOT, ENV_WEIGHTS)                           # the two deployment parameters every child reads; GENIE3_OPT_HOME / MODEL_OPT are this process's (no child imports the package's stack side)
# GPU classes by compute capability and memory band (GiB): the key a box resolves to and the class the report names; the 80 GB H100
# (81,559 MiB) and the 94 GB H100 NVL (95,830 MiB) are two classes so the key names the card that ran
GPU_CLASSES: Tuple[Tuple[str, str, float, float], ...] = (("H200", "9.0", 130.0, 200.0), ("H100NVL", "9.0", 90.0, 100.0), ("H100", "9.0", 75.0, 85.0),
                                                           ("L40S", "8.9", 40.0, 50.0), ("A100", "8.0", 35.0, 90.0), ("A10", "8.6", 20.0, 30.0),
                                                           ("B200", "10.0", 150.0, 300.0))
MAX_CC_OF_STACK = 9.0                                              # torch 2.7.1+cu126 wheels carry kernels up to sm_90 (stock/PINS.json pinned_stack)
WEIGHT_FILES = ("checkpoints/step=600000.ckpt", "config.yaml")     # under GENIE3_WEIGHTS: the stock layout pretrained/v1/ (stock/PINS.json weights)
WEIGHT_KEYS = (("generation", "base", "checkpoint"), ("generation", "base", "config"))   # upstream's request keys for the two files (src/genie3/config/loader.py:201-202: checkout-relative defaults pretrained/v1/…)

_STATE: Dict[str, object] = {"report": None}


# ------------------------------------------------------------------------------------------------------------------------ paths
def tree_root() -> str:
    """The genie3/ directory: GENIE3_OPT_HOME, else MODEL_OPT (run.sh), else the package's own location (opt/genie3_opt/../..)."""
    for var in (ENV_HOME, ENV_TREE):
        v = os.environ.get(var)
        if v:
            return os.path.abspath(v)
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def kit_dir(kit: str = KIT) -> str:
    return os.path.join(tree_root(), *KIT_DIRS[kit].split("/"))


def kit_file(rel: str, kit: str = KIT) -> str:
    return os.path.join(kit_dir(kit), *rel.split("/"))


def chain_file(kit: str, rel: str) -> str:
    """One driver-chain entry: a kit file, or the package's own file when the entry names the package (registry.PACKAGE)."""
    if kit == PACKAGE:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), *rel.split("/"))
    return kit_file(rel, kit)


def pins_path() -> str:
    return os.path.join(tree_root(), "stock", "PINS.json")


def pins() -> dict:
    return json.load(open(pins_path(), encoding="utf-8"))


def _check_pins_module():
    path = os.path.join(tree_root(), "stock", "check_pins.py")
    spec = importlib.util.spec_from_file_location("genie3_check_pins", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def genie3_root() -> Optional[str]:
    """The upstream checkout: GENIE3_ROOT, else pip's recorded editable location."""
    cp = _check_pins_module()
    root, _ = cp.find_root(None)
    return root


def genie3_root_env() -> Optional[str]:
    """GENIE3_ROOT as set (absolute), or None — the fallback when the pins module cannot read the checkout."""
    v = os.environ.get(ENV_ROOT)
    return os.path.abspath(v) if v else None


def weights_dir() -> Optional[str]:
    """GENIE3_WEIGHTS, else the stock layout under the checkout (pretrained/v1) when it holds the checkpoint."""
    w = os.environ.get(ENV_WEIGHTS)
    if w:
        return w
    root = genie3_root()
    if root and os.path.isfile(os.path.join(root, "pretrained", "v1", *WEIGHT_FILES[0].split("/"))):
        return os.path.join(root, "pretrained", "v1")
    return None


def weights_gate(rep: dict) -> List[str]:
    """The weights gate of both routes, warn-and-run: reads the census (manifest.weights_record — the one reader of the weight files; digests
    memoised on disk, digest_memo), records it on the report (``weights_pinned``, ``weights_files`` {name: {sha256, pinned}}, ``weights_lines`` — one
    line per file, `weights=<name> sha256=<12> (pinned)` for the pinned digest, `weights=<name> sha256=<12> NOT PINNED — not the digest pinned in
    stock/PINS.json weights; the pass runs on these files, labelled` for any other, printed with the activation line) and returns the refusal
    reasons: a MISSING file only. A different checkpoint or model configuration runs, labelled not pinned. ``rep["weights_paths"]``, when the
    pass's request names its own generation.base.checkpoint / config, is the pair of files read instead of the directory's (design.weights_paths)."""
    from .manifest import weights_record
    rec = weights_record(rep.get("weights"), paths=rep.get("weights_paths"), refresh=bool(rep.get("dry_run")))
    rep["weights_pinned"] = rec["pinned"]
    rep["weights_files"] = {os.path.basename(rel): ({"sha256": f["sha256"], "pinned": f["pinned"]} if "sha256" in f else {"missing": True}) for rel, f in rec["files"].items()}
    rep["weights_lines"] = rec["lines"]
    if rec["missing"]:
        return [f"weights missing ({ENV_WEIGHTS}={rep.get('weights')!r}): {', '.join(rec['missing'])}"]
    return []


def check_pins(stack: bool = True, root: Optional[str] = None) -> Tuple[List[str], dict]:
    cp = _check_pins_module()
    p = pins()
    found, _ = cp.find_root(root)
    return cp.check(p, found, stack)


def dist_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def upstream_version() -> Optional[str]:
    return dist_version(UPSTREAM_DIST)


def stock_exe(python: Optional[str] = None) -> Optional[str]:
    """The upstream console script `genie3` (setup.py:14): beside the interpreter, else on PATH."""
    cand = os.path.join(os.path.dirname(os.path.abspath(python or sys.executable)), "genie3")
    if os.path.isfile(cand) and os.access(cand, os.X_OK):
        return cand
    return shutil.which("genie3")


# ------------------------------------------------------------------------------------------------------------------------- GPU
def gpu_probe() -> dict:
    """The first CUDA device without importing torch: nvidia-smi name / memory / compute capability; None fields when absent."""
    out = {"name": None, "mem_gib": None, "cc": None, "sm": None, "source": None}
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            r = subprocess.run([smi, "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
            line = (r.stdout or "").strip().splitlines()
            if r.returncode == 0 and line:
                name, mem, cc = [x.strip() for x in line[0].split(",")[:3]]
                out.update(name=name, mem_gib=round(float(mem) / 1024.0, 1), cc=cc, sm="sm" + cc.replace(".", ""), source="nvidia-smi")
        except Exception:  # noqa: BLE001
            pass
    return out


def gpu_class(gpu: dict) -> Optional[str]:
    cc, mem = gpu.get("cc"), gpu.get("mem_gib")
    if cc is None or mem is None:
        return None
    for cls, c, lo, hi in GPU_CLASSES:
        if cc == c and lo <= float(mem) < hi:
            return cls
    return None


def kernel_key(gpu: dict) -> Optional[str]:
    """The key grammar (cc | triton major.minor), e.g. 9.0|3.3 — Triton from the installed distribution, no import."""
    tv = dist_version("triton")
    if not gpu.get("cc") or not tv:
        return None
    return f"{gpu['cc']}|{'.'.join(tv.split('.')[:2])}"


def jit_cache_key(gpu: Optional[dict] = None) -> str:
    """The JIT cache key torch<version sans local tag>-cu<CUDA version sans dot>-sm<compute capability digits>, e.g. torch2.7.1-cu126-sm90:
    the grammar is the shared core's (opt_core.jit_cache.key); the three parts are this package's probes — the installed torch
    distribution's version string, its CUDA local tag (else the CUDA runtime distribution a PyPI-index wheel pins), nvidia-smi's compute
    capability — so no torch import happens here, and a part that cannot be read is named (``unknown:<fact>``), never blank."""
    _core.ensure_importable()
    from opt_core.jit_cache import key
    tv = dist_version("torch") or "unknown:PackageNotFoundError"    # named: never a bare 'unknown' baked into a cache key silently
    ver, _, local = tv.partition("+")
    cu = local[2:] if local.startswith("cu") else None
    if cu is None:                                                # a PyPI-index torch wheel carries no local tag: the CUDA runtime it pins names the toolkit
        rt = dist_version("nvidia-cuda-runtime-cu12") or dist_version("nvidia-cuda-runtime-cu11")
        cu = "".join(rt.split(".")[:2]) if rt else "unknown:no-cuda-runtime"
    g = gpu_probe() if gpu is None else gpu
    cc = g.get("cc") or ((g.get("sm") or "")[2:] if str(g.get("sm") or "").startswith("sm") else None) or "unknown:no-gpu"
    return key(version=ver, cuda=cu, cc=cc)


def tools(python: Optional[str] = None) -> dict:
    return {"genie3": stock_exe(python), "python": python or sys.executable, "nvidia-smi": shutil.which("nvidia-smi")}


# ------------------------------------------------------------------------------------------------------------- late activation
def instance_check() -> dict:
    """Upstream model instances alive in this process (gc scan of Denoiser subclasses when the module is imported)."""
    mod = sys.modules.get(MODEL_MODULE)
    if mod is None:
        return {"module_imported": False, "instances": 0}
    cls = getattr(mod, MODEL_CLASS, None)
    if cls is None:
        return {"module_imported": True, "instances": 0, "class_missing": True}
    n = sum(1 for o in gc.get_objects() if isinstance(o, cls))
    return {"module_imported": True, "instances": n}


def kit_levers_applied() -> List[str]:
    """The kit's patches applied in this process: the marker its apply() leaves (driver/g3fast_patches.py:110,118 —
    V1PairFeatureNet._g3fast_patched) or its module-level rebinding (geo_utils.batched_gather bound to the patch module's function)."""
    out = []
    pair = sys.modules.get(PAIR_MODULE)
    cls = getattr(pair, "V1PairFeatureNet", None) if pair is not None else None
    if cls is not None and getattr(cls, "_g3fast_patched", False):
        out.append(f"{PAIR_MODULE}.V1PairFeatureNet._g3fast_patched")
    geo = sys.modules.get(GEO_MODULE)
    fn = getattr(geo, "batched_gather", None) if geo is not None else None
    if fn is not None and getattr(fn, "__module__", "") == KIT_PATCH_MODULE:
        out.append(f"{GEO_MODULE}.batched_gather<-{KIT_PATCH_MODULE}")
    return out


# --------------------------------------------------------------------------------------------------------------------- activation
def mode_from_env(mode: Optional[str]) -> Optional[str]:
    if mode is not None and str(mode).strip():
        return str(mode).strip().lower()
    v = os.environ.get(ENV_MODE)
    return v.strip().lower() if v and v.strip() else None


def child_environment(environ=None) -> Tuple[Dict[str, str], List[str]]:
    """The environment every child process of this package starts from: the caller's, minus every GENIE3_OPT* / GENIE3_* / CUDA_MPS_* /
    TORCH_ALLOW_TF32_* name not in KEEP_ENV and minus DROP_ENV_NAMES (NVIDIA_TF32_OVERRIDE). Returns (env, dropped names, sorted)."""
    environ = os.environ if environ is None else environ
    env, dropped = {}, []
    for k, v in environ.items():
        if (k.startswith(DROP_ENV_PREFIXES) and k not in KEEP_ENV) or k in DROP_ENV_NAMES:
            dropped.append(k)
            continue
        env[k] = v
    return env, sorted(dropped)


def driver_environment(environ=None) -> Tuple[Dict[str, str], List[str]]:
    """The driver process's environment: child_environment (the table is the only source of a switch; the package's own switch never
    reaches a process that imports genie3). The recipe exports nothing: both arms set their determinism state themselves (det.py (2)).
    Returns (env, dropped names)."""
    return child_environment(environ)


def driver_command(res: Resolution, request_path: str, out_dir: str, timings_json: Optional[str] = None,
                   python: Optional[str] = None, verbose: bool = False, shard: Optional[Tuple[int, int]] = None) -> List[str]:
    """`python <genie3_opt>/g3batch.py --config <request.yaml> --outdir <out> <mode flags> [--timings <json>] [--verbose] [--shard-id K --num-shards M]`
    — the kit line, run with cwd = $GENIE3_ROOT (the deterministic recipe adds no driver flag: it travels in the request copy's seed and in each
    arm's own determinism settings, det.py); ``shard`` = (K, M) under upstream's --num-shards M --shard-id K: the driver slices the request as
    upstream's loader does."""
    chain = [chain_file(kit, rel) for kit, rel in res.driver]
    cmd = [python or sys.executable] + chain + ["--config", request_path, "--outdir", out_dir] + list(res.flags)
    if timings_json:
        cmd += ["--timings", timings_json]
    if verbose:
        cmd += ["--verbose"]
    if shard is not None and int(shard[1]) > 1:
        cmd += ["--shard-id", str(int(shard[0])), "--num-shards", str(int(shard[1]))]
    return cmd


def lever_notes(res: Resolution, gpu: dict) -> List[str]:
    """Deployment facts a planned lever depends on, noted before the pass (never refused): L7 (mode fast's fused TriangleMultiplication kernel)
    is served from a row for the card's compute capability in opt/genie3_opt/fpf_cells.json; without one the shared core serves its safe settings
    for that capability when it has them (engaged, named on the LEVER line); where the kernel cannot serve the card at all every call is a counted
    fallback and mode fast refuses by name after the pass (exit 3: a mode is all of its levers; mode exact does not plan the lever)."""
    from .modes import TRIMUL_FLAG, TRIMUL_LEVER, TRIMUL_ON
    if TRIMUL_LEVER not in res.levers or not gpu.get("cc"):
        return []
    from . import trimul
    if trimul.cell_row(gpu["cc"]):
        return []
    return [f"lever {TRIMUL_LEVER} ({TRIMUL_FLAG} {TRIMUL_ON}): opt/genie3_opt/fpf_cells.json has no row for compute capability {gpu['cc']} — the shared core serves "
            f"its safe settings for this capability when it has them (the lever engages; its LEVER line names them); where the kernel cannot serve the card at all "
            f"--mode fast refuses by name after the pass, exit 3 (--mode exact does not plan the lever)"]


def hardware_notes(gpu: dict) -> List[str]:
    """Hardware facts the activation NOTES and proceeds on — never a refusal (both routes; printed by report.note_line, recorded as the
    report's ``notes``): a compute capability above the pinned stack's kernel ceiling (MAX_CC_OF_STACK; stock/PINS.json arch_list)."""
    notes = []
    if gpu.get("cc") and float(gpu["cc"]) > MAX_CC_OF_STACK:
        torch_pinned = (pins().get("pinned_stack") or {}).get("torch")
        notes.append(f"compute capability {gpu['cc']} above the pinned stack's ceiling (sm_{int(MAX_CC_OF_STACK * 10)}, torch {torch_pinned}) — kernels untested; "
                     "the pass proceeds (a kernel the stack lacks fails as the runtime's own error)")
    return notes


def _gate(res: Resolution, rep: dict, gpu: dict) -> List[str]:
    """Refusal reasons for a driver mode on this box (empty = ok); each names a DEPLOYMENT fact (kit files, a CUDA device, the checkout, the
    weight files present) — never a property of the request or of the checkout's bytes."""
    why = []
    for kit, rel in res.driver:
        if kit == PACKAGE:
            if not os.path.isfile(chain_file(kit, rel)):
                why.append(f"package file missing: {PACKAGE}/{rel}")
        elif not os.path.isdir(kit_dir(kit)):
            why.append(f"kit directory missing: {KIT_DIRS[kit]} (tree {tree_root()})")
    if not gpu.get("name"):
        why.append("no CUDA device visible (nvidia-smi found none)")
    if not rep.get("genie3_root"):
        why.append(f"no Genie 3 checkout: set {ENV_ROOT}")
    why += weights_gate(rep)
    return why


def activate(mode: Optional[str], *, dry_run: bool = False, strict: bool = False, trigger: Optional[str] = None, python: Optional[str] = None,
             batch: Optional[int] = None, weights_paths: Optional[Tuple[str, str]] = None) -> dict:
    """Resolve, gate and (unless dry_run) arm `mode` for this process at the request's batch size (modes.effective: the report's levers, flags and
    line are the EFFECTIVE set the pass runs); `weights_paths` = the two weight files the request names itself, when it does (else the weights
    directory's). Returns the activation report; `strict` raises ActivationError instead of returning an inactive report. Idempotent: a second
    call returns the first report; a different mode is refused."""
    m = mode_from_env(mode)
    prev = _STATE["report"]
    if prev is not None and prev.get("active") and not dry_run:
        if prev.get("mode") == m or m is None:
            return prev
        rep = dict(prev, active=False, mode=m, reason=f"already activated as mode={prev.get('mode')} in this process; a different mode is refused")
        return _finish(rep, strict, dry_run)
    defaulted = m is None
    if defaulted:
        m = DEFAULT_MODE
    rep = {"active": False, "dry_run": bool(dry_run), "mode": m, "mode_defaulted": defaulted, "package_version": __version__, "trigger": trigger,
           "genie3_version": upstream_version(), "gpu": None, "stack": None, "pins": {}, "tools": tools(python), "tree": tree_root(),
           "genie3_root": None, "weights": weights_dir(), "weights_paths": weights_paths, "opt_core": None, "reason": None, "notes": []}
    try:
        res = effective(m, batch)                                # the mode's line at this pass's batch size (modes.with_batch refuses a batch below 1, by name)
    except ModeError as e:
        rep["reason"] = str(e)
        return _finish(rep, strict, dry_run)
    core = _core.core_gate()                                      # the core pin gate (statement one of every entry, off included; the same producer fills the report — a mismatch has refused with its line and exit 3 before this point)
    rep["opt_core"] = dict(core["installed"], ok=True, pinned=core["pinned"])
    rep.update(tier=res.tier, attach=res.attach, levers_planned=list(res.levers), flags=list(res.flags), batch_size=batch_size(res), precision=precision_of(res), trimul=trimul_of(res), driver=[f"{KIT_DIRS.get(k, 'opt/' + k)}/{r}" for k, r in res.driver], line=res.line,
               # the contract's applied / fallback / unavailable / partial keys: on this model the levers are applied by the kit's own
               # driver in its own process, so they are filled from the evidence read-back after a driver pass (record_pass)
               levers_applied=[], levers_fallback=[], levers_unavailable=[], partial=False)
    try:
        bad, detail = check_pins(stack=(res.attach == "driver"))
        rep["pins"] = {"bad": bad, "detail": detail}
        rep["genie3_root"] = (detail.get("checkout") or {}).get("root")
    except Exception as e:  # noqa: BLE001
        rep["pins"] = {"bad": [f"stock pins unreadable: {e!r}"], "detail": {}}
        rep["genie3_root"] = genie3_root_env()
    gpu = gpu_probe()
    gpu["class"] = gpu_class(gpu)
    gpu["key"] = kernel_key(gpu)
    rep["gpu"] = gpu
    rep["notes"] = hardware_notes(gpu) + [f"stock pins: {b} — reported, not gated: the pass runs on the checkout as installed" for b in (rep["pins"].get("bad") or [])] + lever_notes(res, gpu)   # noted, never refused (both routes): printed before the activation line, recorded in the run record
    sor = pins().get("pinned_stack", {})
    rep["stack"] = {"torch": dist_version("torch"), "lightning": dist_version("lightning"), "triton": dist_version("triton"), "numpy": dist_version("numpy"),
                    "pinned": all((dist_version(n) or "").split("+")[0] == str(sor.get(n, "")).split("+")[0] for n in ("torch", "lightning", "numpy")), "id": sor.get("id")}
    target = os.environ.get(ENV_TARGET_GPU)
    if target and gpu.get("class") and target.upper() != gpu["class"].upper():
        rep["target_gpu_mismatch"] = f"{ENV_TARGET_GPU}={target} but this box is {gpu['class']}"
    if res.attach == "stock-cli":
        why = []
        if not rep.get("genie3_root"):
            why.append(f"no Genie 3 checkout: set {ENV_ROOT}")
        if not rep["tools"].get("genie3"):
            why.append("no `genie3` console script beside the interpreter or on PATH (the upstream entry point, setup.py:14)")
        why += weights_gate(rep)
        rep["would_refuse"] = why
        if why:
            rep["reason"] = "; ".join(why)
        else:
            rep["active"] = not dry_run
        return _finish(rep, strict, dry_run)
    why = _gate(res, rep, gpu)
    late = []
    if not dry_run:
        ic = instance_check()
        if ic.get("instances"):
            late.append(f"{ic['instances']} {MODEL_CLASS} instance(s) already exist in this process: activation must precede the model")
        applied = kit_levers_applied()
        if applied:
            late.append("kit lever module(s) already applied in this process: " + ", ".join(applied))
    rep["would_refuse"] = why
    if why or late:
        rep["reason"] = "; ".join(late + why)
        return _finish(rep, strict, dry_run)
    if not dry_run:
        rep["active"] = True
        _env, dropped = driver_environment()
        rep["env_dropped"] = dropped
    return _finish(rep, strict, dry_run)


def _finish(rep: dict, strict: bool, dry_run: bool) -> dict:
    if not dry_run:
        _STATE["report"] = rep
    for line in rep.get("weights_lines") or []:                    # the weights census, one line per file (pinned | NOT PINNED), before the verdict line
        _report.emit(f"{_report.PREFIX} {line}")
    for note in rep.get("notes") or []:                             # hardware facts noted, never refused (hardware_notes), before the verdict line
        _report.emit(_report.note_line(note))
    _report.emit(_report.activation_line(rep))
    if strict and not rep.get("active") and not dry_run:
        raise ActivationError(rep.get("reason") or "not active")
    return rep


def status() -> dict:
    rep = _STATE["report"]
    return rep if rep is not None else {"active": False, "reason": "enable() has not run in this process"}


def record_pass(evidence: dict) -> Optional[dict]:
    """After a driver pass: fill the armed report's levers_applied / levers_unavailable / levers_declined / partial from the evidence read-back
    (design.read_evidence: {applied, missing, declined, forbidden}); a lever the mode names that left no evidence is `unavailable` (it could
    not run here: design.run refuses the mode by name, exit 3 — a mode is all of its levers; the report's `partial` flag records that one did, or
    that a forbidden line was printed); a lever that declined THIS request by name (registry.Lever.declined — L7 under the kernel's token
    floor) is `declined`: not applied, not unavailable. The kit's driver has no silent fallback path, so levers_fallback stays empty. Returns
    the report, or None before enable()."""
    rep = _STATE["report"]
    if rep is None or not rep.get("active"):
        return None
    applied = registry_order(set(rep.get("levers_applied", [])) | set(evidence.get("applied", [])))
    rep["levers_applied"] = applied
    rep["levers_unavailable"] = registry_order(set(rep.get("levers_unavailable", [])) | set(evidence.get("missing", [])))
    rep["levers_declined"] = registry_order(set(rep.get("levers_declined", [])) | set(evidence.get("declined") or ()))
    rep["partial"] = bool(rep.get("partial")) or bool(evidence.get("missing")) or bool(evidence.get("forbidden"))
    return rep


def registry_order(ids) -> List[str]:
    """Lever ids in the registry's own order (registry.LEVERS), unknown ids after them sorted by name."""
    from .registry import LEVERS
    rank = {lid: i for i, lid in enumerate(LEVERS)}
    return sorted(ids, key=lambda lid: (rank.get(lid, len(rank)), lid))


def reset_for_tests() -> None:
    _STATE["report"] = None
