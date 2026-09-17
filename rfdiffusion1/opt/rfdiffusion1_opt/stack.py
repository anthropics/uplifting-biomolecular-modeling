"""Activation: the tree and kit paths, the upstream checkout and weights, the GPU and its key, the mode's environment, and the activation report.

RFdiffusion-1's optimizations live in the control flow of the kit's resident driver (the base kit's `drivers/rfd_bench.py`) — the
resident process, the memoised constants and the CUDA-graph wrapping of the instantiated model. They
cannot attach to the stock command line's one-process-per-invocation shape, so `enable(mode)` resolves and gates the mode on this
box and ARMS it for this process's `design()` calls: the report says `attach=driver` and design.py runs the mode's line
(stack.driver_command / driver_environment). Nothing is imported from the kits here; the kit's own driver applies the levers in
its own process and prints its own evidence lines (registry.Lever.evidence), which design.py reads back.

`activate(mode, dry_run=True)` (= `check`) resolves, gates and reports without touching the environment. The shared core this
package runs on is the one opt/pyproject.toml pins: the core pin gate (`_core_gate.gate`, statement one of every entry, before this module's
opt_core import) has refused any other by name with exit 3, and the report's `core` block carries its facts (pinned, installed). Gates here:
the tree and the mode's kit directory present; an RFdiffusion checkout (RFD_ROOT) and a weights directory (WEIGHTS, or the typed
inference.model_directory_path); a visible CUDA device; for the packed line the packing launcher and nvidia-cuda-mps-control. The upstream pin
and the installed stack (stock/PINS.json, stock/check_pins.py) are REPORTED by `check` (pins_report), never gated: a checkout or stack other
than the pinned one runs, named. The probes,
the distribution versions, the instance count and the JIT-cache-key grammar are the core's (opt_core.gates,
opt_core.instances, opt_core.jit_cache), projected to this package's keys. The base driver's own card-name assertion (rfd_bench.py:71) is met on other cards by exporting its documented
override ALLOW_ANY_GPU=1; the package classes the card by compute capability and memory band (GPU_CLASSES: the key the report and
the manifest carry), NOTES a card above the stack's capability ceiling (MAX_CC_OF_STACK: kernels untested there — report.note_line,
recorded in the report's ``notes``; the run proceeds) and refuses only on the kits' own facts — a card the kits never measured resolves
with its class named, and the key decides. The late-activation rule:
enable() may run any time after import, is idempotent, and is refused by name once an upstream model instance exists in this
process or a kit lever module reports itself applied here.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from typing import Dict, List, Optional, Tuple

from opt_core import gates as core_gates, instances as core_instances, jit_cache as core_jit_cache

from . import ActivationError, __version__
from . import report as _report
from ._core_gate import gate as core_gate
from .modes import (DRIVER_FIXED, MODE_NAMES, ModeError, Resolution, default_mode, numerics, resolve)
from .registry import KIT_BASE, KIT_DIRS, LEVERS, PKG

ENV_MODE, ENV_HOME = "RFDIFFUSION1_OPT", "RFDIFFUSION1_OPT_HOME"
ENV_TREE = "MODEL_OPT"                                            # run.sh convention: this model's directory (rfdiffusion1/)
ENV_RFD_ROOT, ENV_WEIGHTS = "RFD_ROOT", "WEIGHTS"
ENV_TRITON_CACHE, ENV_TARGET_GPU = "TRITON_CACHE_DIR", "MODEL_OPT_TARGET_GPU"
DRIVER_GPU_OVERRIDE = ("ALLOW_ANY_GPU", "1")                      # the base driver's documented override of its H100 name assertion (rfd_bench.py:71)
UPSTREAM_DIST = "rfdiffusion"
DGL_BACKEND = ("DGLBACKEND", "pytorch")                           # DGL's backend word, set on both arms (the stock arm: design.stock_environment)
MODEL_MODULE, MODEL_CLASS = "rfdiffusion.RoseTTAFoldModel", "RoseTTAFoldModule"
KIT_LEVER_MODULES = ("rfd_fullgraph", "rfd_fastpath", "rfd_prep", "rfd_einsum", "rfd_layernorm")
# variables the package drops from every child process it starts (the driver line, the stock command line) so that the mode table and the
# recipe stay the only source of a switch: the kits' own knobs by prefix (the deployment parameters and the CUDA-graph modules' consistency-check
# cadence kept, KEEP_ENV), the package's own switches by prefix (RFDIFFUSION1_OPT would arm the .pth finder in a child that imports rfdiffusion),
# the CUDA libraries' and torch's TF32 overrides by name (NVIDIA_TF32_OVERRIDE is read by cuBLAS / cuDNN, TORCH_ALLOW_TF32_CUBLAS_OVERRIDE by torch at
# import — neither by any kit; either would defeat the recipe's strict fp32 on every arm)
DROP_ENV_PREFIXES = ("RFD_", "RFDIFFUSION1_")
DROP_ENV_NAMES = ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")
KEEP_ENV = (ENV_RFD_ROOT,)
# GPU classes by compute capability and memory band (GiB): the key a box resolves to and the class the report names; the 80 GB H100
# (81,559 MiB) and the 94 GB H100 NVL (95,830 MiB) are two classes so the key names the card that ran
GPU_CLASSES: Tuple[Tuple[str, str, float, float], ...] = (("H200", "9.0", 130.0, 200.0), ("H100NVL", "9.0", 90.0, 100.0), ("H100", "9.0", 75.0, 85.0),
                                                           ("L40S", "8.9", 40.0, 50.0), ("A100", "8.0", 35.0, 90.0), ("A10", "8.6", 20.0, 30.0),
                                                           ("B200", "10.0", 150.0, 300.0))
MAX_CC_OF_STACK = 9.0                                              # torch 2.4.0+cu121 wheels carry kernels up to sm_90 (stock/PINS.json pinned_stack); a card above it is NOTED (CC_NOTE_FMT), never refused
CC_NOTE_FMT = "compute capability {cc} above the pinned stack's ceiling (torch {torch} carries kernels up to sm_{sm}) — kernels untested; proceeding"
# the packing packing launcher, one copy for every model directory of the release tree (<release tree>/common/mps_packing/): K copies of one
# worker command on one GPU under uncapped CUDA MPS, the memory estimate (a NOTE, never a gate), fresh pipe / log directories, the OOM scan (mps_workers.sh:11-35)
PACK_LAUNCHER = ("common", "mps_packing", "mps_workers.sh")
MPS_CONTROL = "nvidia-cuda-mps-control"                            # the launcher refuses without it (mps_workers.sh:20, rc 4)

_STATE: Dict[str, object] = {"report": None}


# ------------------------------------------------------------------------------------------------------------------------ paths
def tree_root() -> str:
    """The rfdiffusion1/ directory: RFDIFFUSION1_OPT_HOME, else MODEL_OPT (run.sh), else the package's own location (opt/rfdiffusion1_opt/../..)."""
    for var in (ENV_HOME, ENV_TREE):
        v = os.environ.get(var)
        if v:
            return os.path.abspath(v)
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def kit_dir(kit: str) -> str:
    return os.path.join(tree_root(), *KIT_DIRS[kit].split("/"))


def pack_launcher() -> str:
    """The packing launcher's path: <release tree>/common/mps_packing/mps_workers.sh (the release tree = the parent of this model's directory)."""
    return os.path.join(os.path.dirname(tree_root()), *PACK_LAUNCHER)


def pins_path() -> str:
    return os.path.join(tree_root(), "stock", "PINS.json")


def pins() -> dict:
    return json.load(open(pins_path(), encoding="utf-8"))


def _check_pins_module():
    path = os.path.join(tree_root(), "stock", "check_pins.py")
    spec = importlib.util.spec_from_file_location("rfdiffusion1_check_pins", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rfd_root() -> Optional[str]:
    """The upstream checkout: RFD_ROOT, else pip's recorded editable location."""
    cp = _check_pins_module()
    root, _ = cp.find_root(None)
    return root


def weights_dir(typed: Optional[str] = None) -> Optional[str]:
    """The checkpoint directory: upstream's `inference.model_directory_path` as typed, else WEIGHTS."""
    return typed or os.environ.get(ENV_WEIGHTS)


def check_pins(stack: bool = True, root: Optional[str] = None) -> Tuple[List[str], dict]:
    cp = _check_pins_module()
    p = pins()
    found, _ = cp.find_root(root)
    return cp.check(p, found, stack)


def pins_report(root: Optional[str] = None) -> dict:
    """The upstream pin and the installed stack against stock/PINS.json (check_pins), as `check` reports them: `{"pinned": bool, "findings":
    [...], "detail": {...}, "lines": [the PINS lines to print]}`. A report, never a gate: a checkout or stack other than the pinned one is named."""
    try:
        bad, detail = check_pins(stack=True, root=root)
    except Exception as e:  # noqa: BLE001
        bad, detail = [f"stock pins unreadable: {e!r}"], {}
    p = {}
    try:
        p = pins()
    except Exception:  # noqa: BLE001
        pass
    up = (p.get("upstream") or {}).get(UPSTREAM_DIST) or {}                          # stock/PINS.json: upstream.<distribution>.{repo, commit}
    head = f"{_report.PREFIX} PINS upstream={up.get('repo')}@{str(up.get('commit') or '')[:8]} stack={(p.get('pinned_stack') or {}).get('id')} pinned={not bad}"
    return {"pinned": not bad, "findings": list(bad), "detail": detail, "lines": [head] + [f"{_report.PREFIX} PINS finding: {b}" for b in bad]}


def opt_home() -> str:
    """The package's project directory, rfdiffusion1/opt (pyproject.toml with the [tool.opt_core] pin, the carried kits under forward/)."""
    return os.path.join(tree_root(), "opt")




dist_version = core_gates.dist_version                             # an installed distribution's version from its metadata, no import (None when absent)


def upstream_version() -> Optional[str]:
    return dist_version(UPSTREAM_DIST)


# ------------------------------------------------------------------------------------------------------------------------- GPU
def gpu_probe() -> dict:
    """The first CUDA device without importing torch — the core's nvidia-smi probe (opt_core.gates.nvidia_smi_probe: name / compute
    capability / memory) projected to this package's keys {name, mem_gib, cc, sm, source}; None fields when absent, and then
    `error` names why (the probe's own word: a failed probe is not "no device")."""
    g = core_gates.nvidia_smi_probe()
    out = {"name": g.get("name"), "mem_gib": None, "cc": g.get("cc"), "sm": g.get("sm"), "source": None}
    if g.get("name"):
        out["source"] = g.get("probe")
        if g.get("memory_mib") is not None:
            out["mem_gib"] = round(float(g["memory_mib"]) / 1024.0, 1)
    else:
        out["error"] = str(g.get("probe") or "nvidia-smi found none")
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
    """The key grammar (cc | triton major.minor), e.g. 9.0|3.0 — Triton from the installed distribution, no import."""
    tv = dist_version("triton")
    if not gpu.get("cc") or not tv:
        return None
    return f"{gpu['cc']}|{'.'.join(tv.split('.')[:2])}"


def jit_cache_key(gpu: Optional[dict] = None) -> str:
    """The JIT cache key torch<version sans local tag>-cu<CUDA version sans dot>-sm<compute capability digits>, e.g. torch2.4.0-cu121-sm90 —
    the core's grammar (opt_core.jit_cache.key) over parts this package resolves without importing torch and refuses to guess: the installed
    torch distribution's version (none installed: refused by name rather than keying the cache on a placeholder), the CUDA toolkit from the
    version's +cu local tag or, for an index wheel without one, from the nvidia-cuda-runtime distribution it pins (neither: refused by name),
    and nvidia-smi's compute capability (`NA` without a device). `configs/h100.env` exports it as MODEL_OPT_STACK_KEY and keys TRITON_CACHE_DIR by it."""
    tv = dist_version("torch")
    if not tv:
        raise RuntimeError("jit_cache_key: no installed torch distribution (dist_version('torch') found nothing) — refuses rather than keying the JIT cache on a placeholder version")
    ver, _, local = tv.partition("+")
    cu = local[2:] if local.startswith("cu") else None
    if cu is None:                                                # a PyPI-index torch wheel carries no local tag: the CUDA runtime it pins names the toolkit
        rt = dist_version("nvidia-cuda-runtime-cu12") or dist_version("nvidia-cuda-runtime-cu11")
        if not rt:
            raise RuntimeError(f"jit_cache_key: torch {tv} names no CUDA toolkit (no +cu local tag, no nvidia-cuda-runtime distribution) — refuses rather than keying the JIT cache on a placeholder toolkit")
        cu = "".join(rt.split(".")[:2])
    g = gpu_probe() if gpu is None else gpu
    return core_jit_cache.key(version=ver, cuda=cu, cc=g.get("cc") or "NA")


def tools() -> dict:
    return {"python": sys.executable, "mps_control": shutil.which(MPS_CONTROL)}


# ------------------------------------------------------------------------------------------------------------- late activation
def instance_check() -> dict:
    """Upstream model instances alive in this process — the core's count of RoseTTAFoldModule (opt_core.instances.instance_check: a gc scan
    when its module is imported, nothing imported by the check) in this package's keys {module_imported, instances, method}."""
    mod = sys.modules.get(MODEL_MODULE)
    if mod is None:
        return {"module_imported": False, "instances": 0}
    if getattr(mod, MODEL_CLASS, None) is None:
        return {"module_imported": True, "instances": 0, "class_missing": True}
    c = core_instances.instance_check(MODEL_MODULE, MODEL_CLASS)
    return {"module_imported": True, "instances": int(c["n"]), "method": c["method"]}


def kit_levers_applied() -> List[str]:
    """Kit lever modules loaded in this process that report themselves applied (their own stats / state), by module name."""
    out = []
    for name in KIT_LEVER_MODULES:
        m = sys.modules.get(name)
        if m is None:
            continue
        try:
            st = m.stats() if callable(getattr(m, "stats", None)) else getattr(m, "STATS", None)
        except Exception:  # noqa: BLE001
            st = None
        applied = bool(getattr(m, "_APPLIED", None)) or (isinstance(st, dict) and any(bool(v) for v in st.values()))
        if applied:
            out.append(name)
    return out


# --------------------------------------------------------------------------------------------------------------------- activation
def _mode_from_env(mode: Optional[str]) -> Optional[str]:
    if mode is not None and str(mode).strip():
        return str(mode).strip().lower()
    v = os.environ.get(ENV_MODE)
    return v.strip().lower() if v and v.strip() else None


def child_environment(environ=None) -> Tuple[Dict[str, str], List[str]]:
    """The environment every child process of this package starts from: the caller's, minus every RFD_* / RFDIFFUSION1_* name not in
    KEEP_ENV and minus DROP_ENV_NAMES. Returns (env, dropped names, sorted)."""
    environ = os.environ if environ is None else environ
    env, dropped = {}, []
    for k, v in environ.items():
        if (k.startswith(DROP_ENV_PREFIXES) and k not in KEEP_ENV) or k in DROP_ENV_NAMES:
            dropped.append(k)
            continue
        env[k] = v
    return env, sorted(dropped)


def driver_environment(res: Resolution, gpu: dict, environ=None) -> Tuple[Dict[str, str], List[str]]:
    """The driver process's environment: child_environment (the table is the only source of a switch; the package's own switch
    never reaches a process that imports rfdiffusion), plus the mode's row, DGL's backend word (the same the stock arm gets), and the
    base driver's card override off-H100. Returns (env, dropped names)."""
    env, dropped = child_environment(environ)
    env.update(res.env)
    env.setdefault(*DGL_BACKEND)                                      # DGL's backend selection for the driver process (the driver setdefaults it too); not numerics
    if gpu.get("name") and "H100" not in str(gpu.get("name")):
        env[DRIVER_GPU_OVERRIDE[0]] = DRIVER_GPU_OVERRIDE[1]
    return env, sorted(dropped)


def driver_command(res: Resolution, cases_path: str, out_dir: str, tag: str, rfd: Optional[str] = None, weights: Optional[str] = None,
                   python: Optional[str] = None, compose: Optional[List[str]] = None) -> List[str]:
    """`python -m rfdiffusion1_opt.driver_run --fixed K=V --fixed K=V --cases C [--compose K=V ...] <driver> --rfd-root R --weights W
    <mode flags> --no-traj 0|1 --cases C --out O --tag T` — the kit line with the launch's values
    composed on the driver's hard-coded keys (`--fixed`, modes.DRIVER_FIXED) and every other typed key carried onto its configuration (`--compose`; driver_run.py); `compose` = further carried overrides."""
    chain = [os.path.join(kit_dir(kit), *rel.split("/")) for kit, rel in res.driver_chain]
    fixed = [o for o in res.settings.compose_overrides if o.split("=", 1)[0] in DRIVER_FIXED]
    carried = [o for o in res.settings.compose_overrides if o.split("=", 1)[0] not in DRIVER_FIXED] + list(compose or [])
    wrap = ["-m", "rfdiffusion1_opt.driver_run"] + [x for o in fixed for x in ("--fixed", o)] + ["--cases", cases_path] + [x for o in carried for x in ("--compose", o)]
    return ([python or sys.executable] + wrap + chain + ["--rfd-root", rfd or rfd_root() or "", "--weights", weights or weights_dir() or ""]
            + list(res.flags) + list(res.settings.driver_flags) + ["--cases", cases_path, "--out", out_dir, "--tag", tag])


def _gate(res: Resolution, rep: dict, gpu: dict) -> List[str]:
    """Refusal reasons for a driver mode on this box (empty = ok); each names the fact. A fact that gates nothing goes to rep["notes"]
    (printed as a NOTE line by _finish): a card above the stack's compute-capability ceiling."""
    why = []
    rep.setdefault("notes", [])
    for kit in list(dict.fromkeys([k for k, _rel in res.driver_chain] + [LEVERS[i].kit for i in res.levers if LEVERS[i].kit != PKG])):   # the driver's kit and every carried lever's kit (T2 lives in the SE(3) add-on's directory, not the driver's; IO1 is this package's own module)
        if not os.path.isdir(kit_dir(kit)):
            why.append(f"kit directory missing: {KIT_DIRS[kit]} (tree {tree_root()})")
    needs_triton = [l for l in ("T2", "K2") if l in res.levers]                                               # the fast line's Triton levers, gated here so `check` and `design` refuse alike (driver_run.arm_se3fast refuses T2 again in the driver process; the core's rfd_layernorm prints active = False without triton — no torch fallback is the mode's line)
    if needs_triton and not (rep.get("stack") or {}).get("triton"):
        why.append(f"triton is not installed in this stack: the fast line's lever(s) {','.join(needs_triton)} require it (T2: the SE(3) add-on's Triton kernels, opt/forward/se3fast_addon; "
                   "K2: the shared core's Triton row LayerNorm) — select --mode exact")
    if not gpu.get("name"):
        why.append("no CUDA device visible (" + (gpu.get("error") or "nvidia-smi found none") + ")")
    elif gpu.get("cc") and float(gpu["cc"]) > MAX_CC_OF_STACK:                                           # untested hardware: named, never refused — a kernel the stack lacks fails on its own terms
        rep["notes"].append(CC_NOTE_FMT.format(cc=gpu["cc"], torch=pins()["pinned_stack"]["torch"], sm=int(MAX_CC_OF_STACK * 10)))
    if not rep.get("rfd_root"):
        why.append("no RFdiffusion checkout: set RFD_ROOT")
    if not rep.get("weights"):
        why.append("no weights directory: set WEIGHTS")
    if res.served:
        if not os.path.isfile(pack_launcher()):
            why.append(f"the packing launcher is not in this tree: {'/'.join(PACK_LAUNCHER)} (expected at {pack_launcher()}; --pack runs "
                       "through it and nothing else — serve.py)")
        if not rep["tools"].get("mps_control"):
            why.append(f"{MPS_CONTROL} is not on PATH: the launcher refuses without it (mps_workers.sh:20) — the tested geometry is uncapped CUDA MPS")
    return why


def activate(mode: Optional[str], overrides=None, *, dry_run: bool = False, strict: bool = False,
             trigger: Optional[str] = None, served: bool = False, det: bool = False) -> dict:
    """Resolve, gate and (unless dry_run) arm `mode` for this process. Returns the activation report; `strict` raises ActivationError
    instead of returning an inactive report. Idempotent: a second call returns the first report; a different mode is refused.
    `served`: the packed line of `mode` (`design --pack K`; modes.resolve served=True), gated
    further on the packing launcher and the MPS control binary; the report carries `served` and `pack_launcher`. `det`: the deterministic
    recipe's level (`--det 1`), composed on the driver line (modes.settings_of)."""
    m = _mode_from_env(mode)
    if trigger is not None:                                           # the .pth route (opt_core.autoload fired on the stock command line's import): a kit mode is a driver line
        return refuse_at_trigger(m, trigger, strict)
    s = [str(o) for o in (overrides or [])]                           # upstream's hydra KEY=VALUE overrides as typed (modes.settings_of)
    typed_weights = next((o.split("=", 1)[1] for o in reversed(s) if o.split("=", 1)[0] == "inference.model_directory_path"), None)
    prev = _STATE["report"]
    if prev is not None and prev.get("active") and not dry_run:
        if prev.get("mode") == m and bool(prev.get("served")) == bool(served):
            return prev
        rep = dict(prev, active=False, reason=f"already activated as mode={prev.get('mode')} served={bool(prev.get('served'))} "
                                              "in this process; a different mode or route is refused")
        return _finish(rep, strict, dry_run)
    defaulted = m is None
    if defaulted:
        m = default_mode(served)
    rep = {"active": False, "dry_run": bool(dry_run), "mode": m, "mode_defaulted": defaulted, "overrides": list(s), "package_version": __version__, "trigger": trigger,
           "rfdiffusion_version": upstream_version(), "gpu": None, "stack": None, "tools": tools(), "tree": tree_root(),
           "rfd_root": rfd_root(), "weights": weights_dir(typed_weights), "det": int(bool(det)), "reason": None, "core": None,
           "served": bool(served), "pack_launcher": pack_launcher() if served else None}
    core = core_gate(__file__, tag=_report.TAG)                       # the core pin gate's facts: the same producer every entry ran as statement one (a core other than the pin never reaches this line)
    rep["core"] = dict(core["installed"], ok=True, pinned=core["pinned"])   # {package_dir, root, version, ok, pinned: {path, version, pyproject}}
    try:
        res = resolve(m, s, served=served, det=det)
    except ModeError as e:
        rep["reason"] = str(e)
        return _finish(rep, strict, dry_run)
    rep.update(numerics=numerics(res.mode, levers=res.levers))   # the numerics words: the numerics the line runs under, declared from the mode table (torch lives in the driver child; its record is compared after each pass, design.py)
    rep.update(mode=res.mode, tier=res.tier, attach=res.attach, levers_planned=list(res.levers), env=dict(res.env), flags=list(res.flags),
               driver_chain=[f"{KIT_DIRS[k]}/{r}" for k, r in res.driver_chain], line=res.line, overrides=list(res.settings.stock_overrides), served=bool(res.served),
               compose=list(res.settings.compose_overrides) if res.attach == "driver" else [],   # the launch's values composed in place of the driver's constants (driver_run.py)
               # the contract's applied / fallback / unavailable / partial keys: on this model the levers are applied by the kits' own
               # driver in its own process, so they are filled from the evidence read-back after a driver pass (record_pass)
               levers_applied=[], levers_fallback=[], levers_unavailable=[], partial=False)
    gpu = gpu_probe()
    gpu["class"] = gpu_class(gpu)
    gpu["key"] = kernel_key(gpu)
    rep["gpu"] = gpu
    sor = pins().get("pinned_stack", {})
    rep["stack"] = {"torch": dist_version("torch"), "dgl": dist_version("dgl"), "triton": dist_version("triton"),
                    "pinned": all((dist_version(n) or "").split("+")[0] == str(sor.get(n, "")).split("+")[0] for n in ("torch", "dgl")), "id": sor.get("id")}
    target = os.environ.get(ENV_TARGET_GPU)
    if target and gpu.get("class") and target.upper() != gpu["class"].upper():
        rep["target_gpu_mismatch"] = f"{ENV_TARGET_GPU}={target} but this box is {gpu['class']}"
    if res.attach == "stock-cli":
        why = []
        if not rep.get("rfd_root"):
            why.append("no RFdiffusion checkout: set RFD_ROOT")
        if not rep.get("weights"):
            why.append("no weights directory: set WEIGHTS")
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
    rep["driver_gate_override"] = dict([DRIVER_GPU_OVERRIDE]) if (gpu.get("name") and "H100" not in str(gpu["name"])) else {}
    if why or late:
        rep["reason"] = "; ".join(late + why)
        return _finish(rep, strict, dry_run)
    if not dry_run:
        rep["active"] = True
        env, dropped = driver_environment(res, gpu)
        rep["env_dropped"] = dropped
    return _finish(rep, strict, dry_run)


def refuse_at_trigger(mode: Optional[str], trigger: str, strict: bool = True) -> dict:
    """The .pth route's one outcome: a kit mode cannot serve upstream's one-process-per-invocation command line (the levers live in the
    resident driver's control flow), so the import of `trigger` under RFDIFFUSION1_OPT=<mode> is refused by name — ONE line naming the fact,
    the kit line's route and the stock route, `[rfdiffusion1-opt] NOT ACTIVE: <_autoload.FACT>` — and, `strict` (the .pth route's call),
    ActivationError: opt_core.autoload ends the process with exit 3 before anything of upstream's runs under the mode's name. `strict=False`
    (a library caller) gets the inactive report instead."""
    from ._autoload import ENV, FACT
    reason = FACT.format(mode=mode, trigger=trigger, env=ENV)
    rep = {"active": False, "dry_run": False, "mode": mode, "trigger": trigger, "attach": "stock-cli", "package_version": __version__, "reason": reason}
    _report.emit(_report.not_active_line(reason))
    if strict:
        raise ActivationError(reason)
    return rep


def _finish(rep: dict, strict: bool, dry_run: bool) -> dict:
    if not dry_run:
        _STATE["report"] = rep
    for note in rep.get("notes") or []:                                          # facts that gate nothing (the capability ceiling): one NOTE line each, before the verdict line
        _report.emit(_report.note_line(note))
    if rep.get("active") and not dry_run:
        rep["proven"] = False                                                        # armed, nothing has run: the fields go out under PLAN; ACTIVE follows the first application evidence (confirm_active), NOT ACTIVE a pass without any
        _report.emit(_report.activation_line(rep, word="PLAN"))
    else:
        _report.emit(_report.activation_line(rep))
    if strict and not rep.get("active") and not dry_run:
        raise ActivationError(rep.get("reason") or "not active")
    return rep


def confirm_active(evidence: str) -> bool:
    """Print the ACTIVE documented line ONCE, after the first application evidence of this process was read (design / serve: a driver pass whose
    log carries at least one lever's applied-line, or a stock process whose environment proof is ok); ``evidence`` names it on the report.
    Returns True when this call printed the line."""
    rep = _STATE.get("report")
    if not rep or not rep.get("active") or rep.get("proven"):
        return False
    rep["proven"] = True
    rep["proven_by"] = evidence
    _report.emit(_report.activation_line(rep))
    return True


def status() -> dict:
    rep = _STATE["report"]
    return rep if rep is not None else {"active": False, "reason": "enable() has not run in this process"}


def record_pass(evidence: dict) -> Optional[dict]:
    """After a driver pass: fill the armed report's levers_applied / levers_unavailable / partial from the evidence read-back
    (design.read_evidence: {applied, missing, forbidden}); a lever the mode names that left no evidence line is `unavailable`, and the
    pass is `partial` when one is missing or a forbidden line was printed. The kit's driver has no fallback path (a lever that cannot
    apply refuses), so levers_fallback stays empty. Returns the report, or None before enable()."""
    rep = _STATE["report"]
    if rep is None or not rep.get("active"):
        return None
    applied = sorted(set(rep.get("levers_applied", [])) | set(evidence.get("applied", [])))
    rep["levers_applied"] = applied
    rep["levers_unavailable"] = sorted(set(rep.get("levers_unavailable", [])) | set(evidence.get("missing", [])))
    rep["partial"] = bool(rep.get("partial")) or bool(evidence.get("missing")) or bool(evidence.get("forbidden"))
    return rep


def reset_for_tests() -> None:
    _STATE["report"] = None
