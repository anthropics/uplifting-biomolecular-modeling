"""Activation: lever directories, pins, GPU identity, the mode's environment, and the lever application through the lever modules themselves.

Two forms, one table (modes.py):

* the **process form** — what `boltzgen-opt
design` runs: the runner ``python forward/xattempt_addon/src/xa_run.py <run_dir> <seed>``
  in a child process whose environment ``mode_env()`` builds from the mode table (the lever directories' `src` on PYTHONPATH in the
  mode table's order, the runner's own switch defaults and the mode's overrides exported, every other lever switch dropped, and
  ``BOLTZGEN_OPT=<mode>`` with the one-shot ``BOLTZGEN_OPT_HANDOVER=1``, so the runner child activates the package at interpreter
  start through the hook-first route below and imports the mode's ``own_imports``, which neither the runner nor ``sitecustomize.py``
  imports). Levers are read off the lever modules' own lines in the child's log (``RUNNER_LINES``) and files (``inproc_times.json``:
  ``CALLER_PROVEN_LEVERS``); a planned lever no line shows is a named fallback, never dropped from the account.
* the **in-process form** — `enable(mode)` in a python process that uses the upstream API, and the env route
  (``BOLTZGEN_OPT=<mode>``, _autoload.py): after the gates, ``activate()`` exports the same environment, puts the lever directories'
  `src` at the front of ``sys.path`` in the same order and imports what the launch imports — ``fast_inference``'s ``sitecustomize``
  line (``bg_hook``: the seed hook, the graph sampler when ``BG_GRAPH`` != off), the runner's GPU-step lever imports (``xa_fastinit``,
  ``xa_hoist``) and the mode's ``own_imports``. The lever modules patch at import (their own design); nothing is transcribed. The
  in-process pipeline (`inproc`) is a process form and is reported ``unavailable`` here. An activated process keeps ``BOLTZGEN_OPT``
  (and sets it) beside the exported PYTHONPATH, so each of its children — upstream's per-step interpreters under `boltzgen run` —
  starts on the ``sitecustomize`` line with the finder armed; there ``activate()`` finds the hook line already present under the
  mode's own environment (``hook_line_mismatch``: every lever switch present equals the mode's export, the hook files are this
  tree's) and completes it with the remaining imports in the runner's order (``hook_first`` in the report). The same hook line under
  any other environment, or any other lever module already imported, is an activation by another route and is refused.

Gates (both forms), after the core pin gate every entry runs first (``boltzgen_opt.core_gate``: the pinned ``opt_core`` is the
installed one, checked before anything of the core is imported): the lever directories present with their launch files, ``boltzgen``
installed at the pinned version (stock/PINS.json), a visible GPU (a card other than the pinned H100 80GB by name or memory is admitted
with one ``NOTE`` line naming it and ``gpu_gate=noted`` in the report — also below the compute-capability floor of stock/PINS.json
gpu_of_record, where the NOTE says so: a lever that cannot run on the card names itself on its own line, the mode never refuses over
the card). Late activation follows one rule: allowed after ``boltzgen`` is imported, refused by name once a ``Boltz`` model instance
exists (counted by a ``__new__`` wrap on the class, armed at activation, plus a gc scan of instances that already exist —
``Boltz.__init__`` is never wrapped: Lightning's checkpoint loader inspects its signature) or once any lever module is already imported
(activated by another route; the package does not re-activate). ``activate(mode, dry_run=True)`` resolves and gates and applies nothing (`check`).
"""
from __future__ import annotations

import atexit
import importlib
import importlib.metadata
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import warnings
from typing import Dict, List, Optional, Tuple

from opt_core import home as core_home, instances as core_instances, jit_cache as core_jit_cache
from opt_core.oom import is_oom

from . import modes, registry, report
from .codes import EXIT_NOT_ACTIVE

ENV_HOME = "BOLTZGEN_OPT_HOME"           # the opt/ directory (default: this package's parent — the install must stay editable)
ENV_HANDOVER = "BOLTZGEN_OPT_HANDOVER"   # =1 in the process-form child (mode_env): the hand-over is one-shot — consumed by that child's activation, after which BOLTZGEN_OPT / this name leave its environment, so nothing it spawns (the runner's stock CPU-step subprocesses) inherits the mode
ENV_KERNELS = "BOLTZGEN_OPT_KERNELS"     # census.ENV_KERNELS: the accelerator census payload of a kit child (design.kit_env) / of an activated process's children (activate)
ENV_TEST_GPU = "BOLTZGEN_OPT_TEST_GPU"   # tests only: "<name>,<memory MiB>[,<compute capability>]" substitutes the GPU probe (documented test seam)
ENV_CACHE = "BOLTZGEN_CACHE"             # the HF cache root upstream's `--cache` reads (configs/<gpu>.env)
ENV_STEP = "BOLTZGEN_PIPELINE_STEP"      # upstream's `boltzgen run` sets it per pipeline step
GPU_STEPS = ("design", "inverse_folding", "folding", "design_folding", "affinity")   # upstream's GPU (model) steps — the ones the runner runs in-process
CPU_STEPS = ("analysis", "filtering")
KIT_MODULE_PREFIXES = ("xa_", "bg_")
HOOK_ROUTE_MODULES = ("sitecustomize", "bg_hook", "bg_graph_patch")     # what fast_inference's sitecustomize line imports (sitecustomize -> bg_hook -> bg_graph_patch under BG_GRAPH)
KIT_LAUNCH_FILES = {modes.KIT_XATTEMPT: (modes.RUNNER,),
                    modes.KIT_PARTNER: (modes.SITECUSTOMIZE, os.path.join("src", "bg_inproc.py"), os.path.join("src", "bg_hook.py"),
                                        os.path.join("src", "bg_graph_patch.py"))}
BOLTZ_MODULE, BOLTZ_CLASS = "boltzgen.model.models.boltz", "Boltz"     # the upstream model class whose instances the late-activation rule counts (opt_core.instances)
INSTANCE_COUNTER_WORDS = {"counted": "new-wrap", "gc": "gc-scan", "none": "armed"}   # opt_core.instances method -> the activation report's instance_counter value (opt_manifest.json)
PACKAGE_VERSION_FALLBACK = "0.0.0"

# the lever modules' own lines (in the child's log: its stdout and stderr) that show a lever engaged in a child process (process form)
RUNNER_LINES = {
    "fastinit": re.compile(r"^\[xa_fastinit\] coverage OK"),                      # xa_fastinit.py, once per checkpoint load
    "hoist": re.compile(r"^\[xa_hoist\] installed"),                              # xa_hoist.py install
    "graph_sampler": re.compile(r"^\[bg_graph_patch\] mode=graph"),                # bg_graph_patch.py apply
    "td_chunk": re.compile(r"^\[levers\] TokenDistanceModule row-chunked, rows/block=\d+"),   # sz_levers.py install(), printed the moment the class is patched (big)
    "cond_dedup": re.compile(r"^\[fl_levers\] cond_dedup installed"),                       # fl_levers.py install(), printed the moment the classes are patched (fast)
    "attn_bf16": re.compile(r"^\[fl_levers\] attn_bf16 installed"),                         # fl_levers.py install() (fast)
    "attn_cudnn": re.compile(r"^\[fl_levers\] attn_bf16 installed .*\bbackend=cudnn\b"),     # fl_levers.py install(): the pin is part of attn_bf16's install line (fast)
    "dit_fused": re.compile(r"^\[fl_levers\] dit_fused installed"),                         # fl_levers.py install(): the core kernels routed, the token layers patched (fast)
    "async_writer": re.compile(r"^\[hl_levers\] async_writer installed"),                   # hl_levers.py install(): the design writer patched, the core AsyncWriter constructed (every kit mode)
}
CALLER_PROVEN_LEVERS = ("inproc",)   # planned levers no child line shows: the caller reads them off files (design.evidence: inproc_times.json) — every other planned lever has a RUNNER_LINES entry or is a named fallback
RUNNER_DISABLED_LINES = {
    "fastinit": re.compile(r"^\[xa_fastinit\] (not installed|COVERAGE INCOMPLETE|coverage check failed|deferred inits replayed)"),   # xa_fastinit.py: the module could not install, or a load replayed the stock initialisers (the lever did not serve it: STATS fallback_reason)
    "hoist": re.compile(r"^\[xa_hoist\] DISABLED"),
    "graph_sampler": re.compile(r"^\[bg_graph_patch\] CAPTURE FAILED"),          # bg_graph_patch.py: the sampler runs eager predraw for the rest of the process — a fallback, partial by name
    "cond_dedup": re.compile(r"^\[fl_levers\] cond_dedup (DISABLED|GATE-FAIL)"),    # fl_levers.py report_lines(): a RowDedup site with counted fallbacks or no served call although rows > 1 were seen
    "attn_bf16": re.compile(r"^\[fl_levers\] attn_bf16 (DISABLED|GATE-FAIL)"),      # fl_levers.py: the bf16 kernel refused at first call (stock serves), or eligible calls never served
    "attn_cudnn": re.compile(r"^\[fl_levers\] attn_bf16 (DISABLED|GATE-FAIL)"),     # a refused pin disables attn_bf16 as a whole (the stock fp32 call serves): both levers fall together
    "dit_fused": re.compile(r"^\[fl_levers\] dit_fused (DISABLED|GATE-FAIL)"),      # fl_levers.py: the core's route refusal / a kernel error (the cond_dedup statements serve), or eligible layers never served
    "async_writer": re.compile(r"^\[hl_levers\] async_writer GATE-FAIL"),           # hl_levers.py report_lines(): submitted != written, or a failed write (the run fails: the drain re-raises, the core's exit guard)
}
RUNNER_STATS_LINE = re.compile(r"^\[(?:xa_run|fl_levers|hl_levers)\] (xa_fastinit|xa_hoist|cond_dedup|attn_bf16|dit_fused|async_writer) stats: (.*)$")   # xa_run.py's stats line; fl_levers.py / hl_levers.py report_lines()
RUNNER_OOM_LINE = re.compile(r'^\[sz\] \{"event": "oom"')   # sz_levers.py's own oom-trace print (every kit mode; dict key order is fixed by the literal, so a prefix match is
                                                            # exact, not a guess) — WHERE an out-of-memory happened; HOW MANY batches upstream skipped is the design census's (design.designs_census)


class ActivationError(RuntimeError):
    """A named refusal: the mode cannot be activated in this process (the reason is the message)."""


_REPORT: Optional[dict] = None
# ------------------------------------------------------------------------------------------------------------ paths and pins
def opt_home() -> str:
    """boltzgen/opt: ``BOLTZGEN_OPT_HOME`` (this directory) when set, else the core's tree rule (``$MODEL_OPT`` = boltzgen/, else two
    directories above this package file: the editable install)."""
    h = os.environ.get(ENV_HOME)
    if h:
        return os.path.abspath(h)
    return os.path.join(core_home.tree_home(__file__), "opt")


def tree_home() -> str:
    return os.path.dirname(opt_home())


def kit_dir(rel: str) -> str:
    return os.path.join(opt_home(), rel)


def kit_src(rel: str) -> str:
    return os.path.join(kit_dir(rel), "src")


def package_roots() -> List[str]:
    """The two sys.path entries this process imported the package and the shared core from (the directory above ``boltzgen_opt`` — this
    tree's ``opt/`` — and the one above ``opt_core``): ``mode_env`` appends them to a child's PYTHONPATH after the lever directories'
    entries, so every child the package starts imports the very copies its parent runs, whatever other install of them the interpreter's
    site directory carries (an environment with another editable install of this tree would otherwise hand the child that copy: PYTHONPATH
    entries precede every site ``.pth`` entry on ``sys.path``)."""
    import opt_core
    return [os.path.dirname(os.path.dirname(os.path.abspath(f))) for f in (__file__, opt_core.__file__)]


def kit_missing(rel: str) -> List[str]:
    """The launch files of a lever directory that are absent from the tree (empty: present)."""
    return [f for f in KIT_LAUNCH_FILES.get(rel, ()) if not os.path.isfile(os.path.join(kit_dir(rel), f))]


def pins_path() -> str:
    return os.path.join(tree_home(), "stock", "PINS.json")


_PINS: Optional[dict] = None


def pins() -> dict:
    global _PINS
    if _PINS is None:
        with open(pins_path(), "r", encoding="utf-8") as fh:
            _PINS = json.load(fh)
    return _PINS


def package_version() -> str:
    try:
        return importlib.metadata.version("boltzgen_opt")
    except Exception:
        return PACKAGE_VERSION_FALLBACK


def boltzgen_version() -> Optional[str]:
    try:
        return importlib.metadata.version("boltzgen")
    except importlib.metadata.PackageNotFoundError:
        return None


# ------------------------------------------------------------------------------------------------------------------- the GPU
def _nvidia_smi_gpu() -> Optional[dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    first = r.stdout.strip().splitlines()[0].split(",")
    name = first[0].strip()
    try:
        mem = int(float(first[1].strip()))
    except Exception:
        mem = None
    cap = first[2].strip() if len(first) > 2 else None
    return {"name": name, "memory_total_mib": mem, "compute_cap": cap, "count": len(r.stdout.strip().splitlines()), "source": "nvidia-smi"}


def _torch_gpu() -> Optional[dict]:
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return None
    p = torch.cuda.get_device_properties(0)
    return {"name": p.name, "memory_total_mib": int(round(p.total_memory / 2 ** 20)), "compute_cap": f"{p.major}.{p.minor}", "count": torch.cuda.device_count(), "source": "torch"}


def gpu_probe() -> Optional[dict]:
    """The visible GPU: the test seam, nvidia-smi (no torch import), or torch when it is already imported."""
    t = os.environ.get(ENV_TEST_GPU)
    if t:
        name, mem, cap = ([x.strip() for x in t.split(",")] + ["", ""])[:3]
        return {"name": name, "memory_total_mib": int(mem) if mem.isdigit() else None, "compute_cap": cap or None, "count": 1, "source": "test"}
    return _nvidia_smi_gpu() or _torch_gpu()


def compute_cap_tuple(cap) -> Optional[Tuple[int, int]]:
    """``'9.0'`` -> ``(9, 0)``; None for an absent or unreadable capability word."""
    m = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", str(cap)) if cap is not None else None
    return (int(m.group(1)), int(m.group(2))) if m else None


def gpu_gate(gpu: Optional[dict]) -> Tuple[bool, str, str]:
    """(ok, gate, reason) by compute capability (stock/PINS.json gpu_of_record): the pinned card by name and total memory is ``ok``;
    every other card is admitted as ``noted`` (reason = the NOTE's words: its name and memory differ from the pinned card's, and, for a
    card below the tested floor ``min_compute_capability`` or one whose capability cannot be read, that it is outside what the kit was
    run on — a lever that cannot run there says so on its own line; the card is never a refusal). No visible GPU is ``none``: the levers
    need one."""
    rec = pins()["gpu_of_record"]
    if gpu is None:
        return False, "none", "no GPU visible (nvidia-smi absent or no device); the levers need one GPU"
    mem, name = gpu.get("memory_total_mib"), gpu.get("name")
    if mem == rec["memory_total_mib"] and name == rec["name"]:
        return True, "ok", f"{name} ({mem} MiB), the pinned card (stock/PINS.json gpu_of_record)"
    floor, cc = compute_cap_tuple(rec["min_compute_capability"]), compute_cap_tuple(gpu.get("compute_cap"))
    sm = f"sm_{cc[0]}{cc[1]}" if cc else "unreadable"
    if cc is not None and cc >= floor:
        return True, "noted", (f"gpu={name} mem={mem}MiB differs from the pinned card {rec['name']} ({rec['memory_total_mib']}MiB, stock/PINS.json); "
                               f"proceeding (capability {sm})")
    if cc is not None:
        why = f"capability {sm} is below the tested floor {rec['min_compute_capability']}"
    else:
        why = f"capability unreadable (nvidia-smi compute_cap / torch report {gpu.get('compute_cap')!r})"
    return True, "noted", (f"gpu={name} mem={mem}MiB differs from the pinned card {rec['name']} ({rec['memory_total_mib']}MiB, stock/PINS.json) and its "
                           f"{why}; proceeding — a lever that cannot run on this card names itself on its own line")


def torch_dist_version() -> Optional[str]:
    """The installed torch's metadata version, e.g. ``2.13.0+cu130`` (no import); None when torch is not installed."""
    try:
        return importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None


def torch_dist_cuda() -> Optional[str]:
    """``torch.version.cuda`` of the installed torch read from its ``torch/version.py`` (no import): ``'13.0'``, ``'none'`` for a
    build without CUDA, None when unreadable."""
    try:
        path = importlib.metadata.distribution("torch").locate_file(os.path.join("torch", "version.py"))
        with open(path, "r", encoding="utf-8") as fh:
            m = re.search(r"^cuda\s*(?::\s*[^=]+)?=\s*(?:'([^']*)'|\"([^\"]*)\"|(None))", fh.read(), re.M)
    except Exception:
        return None
    if not m:
        return None
    return m.group(1) or m.group(2) or "none"


def stack_key(gpu: Optional[dict] = None) -> str:
    """``torch<ver>-cu<xyz>-sm<NN>`` (``opt_core.jit_cache.key`` formats it) — the JIT cache key the configs export: the installed torch's version and CUDA tag (torch when
    imported; else its metadata version, e.g. ``2.13.0+cu130`` -> ``torch2.13.0-cu130``, and for a build without a ``cu<digits>``
    tag its ``torch/version.py``) and the visible GPU's compute capability (``gpu`` or the probe). The pinned stack stands in
    only for what is absent: no torch installed, no GPU visible."""
    torch = sys.modules.get("torch")
    ver = torch.__version__ if torch is not None else torch_dist_version()
    rec = pins()["stack_of_record"]
    if ver is None:
        tv, cu = rec["torch"].split("+")[0], rec["cuda"].replace(".", "")
    else:
        tv, _, local = ver.partition("+")
        if local.startswith("cu") and local[2:].isdigit():
            cu = local[2:]
        elif torch is not None:
            cu = (getattr(getattr(torch, "version", None), "cuda", None) or "none").replace(".", "")
        else:
            cu = torch_dist_cuda()
            if cu is None:
                raise RuntimeError(f"stack_key: torch metadata version {ver!r} has no +cuXXX local tag and torch/version.py's cuda "
                                    f"string could not be read (import torch.md_backend or reinstall to determine the CUDA tag)")
            cu = cu.replace(".", "")
    if gpu is None:
        gpu = gpu_probe()
    cap = (gpu or {}).get("compute_cap") or pins()["gpu_of_record"]["compute_capability"]
    return core_jit_cache.key(version=tv, cuda=cu, cc=str(cap))              # the core's one key rule formats the parts this kit resolved (import-free, refusing by name)


# ------------------------------------------------------------------------------------------------- environment of a mode
def mode_env(res: modes.Resolution, base: Optional[dict] = None, keep_pythonpath: bool = False, keep_package_env: bool = False) -> Tuple[dict, List[str]]:
    """The child environment of the process form / the exports of the in-process form: ``(env, dropped_names)``.

    Every name under the lever switch prefixes is dropped except the observability names (a caller value equal to the mode's own
    export is not reported as dropped); the runner's own defaults and the mode's overrides are set; PYTHONPATH is the lever
    directories' `src` in the mode table's order, then the directories this process imported the package and the core from
    (``package_roots``) (an existing PYTHONPATH is dropped unless ``keep_pythonpath``, in which case its other entries follow).
    ``BOLTZGEN_OPT`` is kept in the in-process form (``keep_package_env``: the children of an activated process activate the same mode by the env route).

    In the process form a mode with modules of its own (the mode table's ``own_imports``, which neither the runner nor
    ``sitecustomize.py`` imports — every kit mode has them: `async_writer`'s ``hl_levers`` is in all three) hands its child the mode
    too — ``BOLTZGEN_OPT=<mode>`` exported with ``BOLTZGEN_OPT_HANDOVER=1``: the runner child then activates the package at
    interpreter start through the hook-first route every child of the in-process form takes (site imports ``sitecustomize`` →
    ``bg_hook`` imports ``boltzgen`` → the installed ``.pth``'s finder → ``activate()`` completes the line in ``res.imports`` order,
    the mode's own modules last), so its own levers are installed before the runner's body runs, and their install lines are what the
    caller reads (``RUNNER_LINES``). The hand-over is one-shot: ``activate()`` consumes ``BOLTZGEN_OPT_HANDOVER`` and removes the
    mode's names from that child's environment, so the runner's stock CPU-step subprocesses inherit no mode and stay stock. A mode
    row without ``own_imports`` would hand its child nothing of the package (``BOLTZGEN_OPT`` absent: the launch alone activates it)."""
    base = dict(os.environ if base is None else base)
    dropped = []
    for k in list(base):
        if k.startswith(modes.KIT_SWITCH_PREFIXES) and k not in modes.OBSERVABILITY_SWITCHES:
            if base[k] != res.env.get(k):
                dropped.append(k)
            del base[k]
    hands_mode = bool(res.active and modes.KIT_MODES[res.mode].own_imports)   # the child imports the mode's own levers by activating the package
    if ENV_KERNELS in base and not keep_package_env:                    # the accelerator census payload is set per run by the caller (design.kit_env), never inherited
        del base[ENV_KERNELS]
    if not keep_package_env:
        for k in (modes.ENV, ENV_HANDOVER):                         # the launch activates the child, not the package — unless the mode hands it over (hands_mode)
            if k in base and not (hands_mode and k == modes.ENV and base[k] == res.mode):
                dropped.append(k)
                del base[k]
        if hands_mode:
            base[modes.ENV] = res.mode
            base[ENV_HANDOVER] = "1"
    if not res.active:
        return base, sorted(dropped)
    base.update(res.env)
    entries = [kit_src(k) for k in res.kits]
    entries += [p for p in package_roots() if p not in entries]          # then the package and the core this process runs (package_roots): the child imports the same copies
    prev = base.pop("PYTHONPATH", None)
    if prev and keep_pythonpath:
        entries.extend(p for p in prev.split(os.pathsep) if p and p not in entries)
    elif prev:
        dropped.append("PYTHONPATH")
    base["PYTHONPATH"] = os.pathsep.join(entries)
    return base, sorted(set(dropped))


# ------------------------------------------------------------------------------------------------- kit state in this process
def kit_modules_loaded() -> List[str]:
    out = [n for n in sys.modules if n.startswith(KIT_MODULE_PREFIXES)]
    sc = sys.modules.get("sitecustomize")
    if sc is not None and _is_kit_sitecustomize(sc):
        out.append("sitecustomize")
    return sorted(out)


def _is_kit_sitecustomize(mod) -> bool:
    f = getattr(mod, "__file__", "") or ""
    return os.path.abspath(f).startswith(opt_home())


def boltz_class():
    mod = sys.modules.get(BOLTZ_MODULE)
    return getattr(mod, BOLTZ_CLASS, None) if mod is not None else None


def instance_count() -> int:
    """Instances of the upstream model class alive in this process (the core's count: the constructor wrap's live set, or a gc scan when
    the class is not wrapped)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                                   # isinstance on lazy proxies in gc can warn (torch.distributed)
        return core_instances.instance_check(BOLTZ_MODULE, BOLTZ_CLASS)["n"]


def models_built() -> int:
    """Model instances constructed in this process since the counter was armed (activation), or alive now when the class is counted by
    a gc scan — whether any model work happened here under the activation."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = core_instances.instance_check(BOLTZ_MODULE, BOLTZ_CLASS)
    return int(c.get("built") or 0) or (int(c.get("n") or 0) if c.get("method") != "counted" else 0)


def arm_instance_counter() -> str:
    """Count constructions of ``Boltz`` from now on: the core's ``__new__`` wrap on the class (never ``__init__``: Lightning inspects its
    signature) — installed now when its module is imported, the instances that already exist found once by a gc scan; otherwise at the
    module's import through the core's self-removing finder when the module cannot be imported yet. Returns how instances are counted, in
    this kit's report words (``INSTANCE_COUNTER_WORDS`` over the core's method): 'new-wrap' (the wrap's live set) | 'gc-scan' (a gc scan: the
    class could not be wrapped or takes no weak references) | 'armed' (the class is not importable here yet: the core's finder wraps it at import)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            importlib.import_module(BOLTZ_MODULE)                          # the class's module now (the kit lever modules import it next in any case), so the wrap sits on the class before any lever patches it
        except Exception:  # noqa: BLE001 — not importable here: the core arms its finder for a later import and the method says so ('none')
            pass
        method = core_instances.register_instance_counter(BOLTZ_MODULE, BOLTZ_CLASS)["method"]
    return INSTANCE_COUNTER_WORDS[method]


def levers_already_applied() -> List[str]:
    """What the kit modules' own records say is already applied in this process (before the package touched anything)."""
    out = []
    for name, lv in registry.LEVERS.items():
        if lv.probe is None:
            continue
        kind, module, key = lv.probe
        mod = sys.modules.get(module)
        if mod is None:
            continue
        if kind in ("stats", "mode"):
            st = getattr(mod, "STATS", {})
            v = st.get(key) if isinstance(st, dict) else None
            if (kind == "stats" and v) or (kind == "mode" and v not in (None, "off")):
                out.append(name)
        elif kind == "acc":
            acc = getattr(mod, "_acc", {})
            if isinstance(acc, dict) and acc.get(key):
                out.append(name)
    return out


# ------------------------------------------------------------------------------------------------------------- classify
def classify(res: modes.Resolution) -> dict:
    """After the imports: what the kit modules' own records show applied, fallen back (with the kit's own reason) or not
    applicable to the in-process form."""
    applied, fallback, unavailable, why = [], [], [], {}
    for name in res.levers:
        lv = registry.LEVERS[name]
        if lv.probe is None:
            unavailable.append(name)
            why[name] = f"{lv.form} form ({lv.file}): a process launched by `boltzgen-opt design`, not an in-process lever"
            continue
        kind, module, key = lv.probe
        mod = sys.modules.get(module)
        if mod is None:
            fallback.append(name); why[name] = f"kit module {module} not imported"
            continue
        if kind == "stats":
            st = getattr(mod, "STATS", {})
            if st.get(key):
                applied.append(name)
            else:
                fallback.append(name); why[name] = str(st.get("disabled_reason") or f"kit record STATS[{key!r}] is not set")
        elif kind == "mode":
            st = getattr(mod, "STATS", {})
            want = res.env.get(lv.switch)
            if st.get(key) == want:
                applied.append(name)
            else:
                fallback.append(name); why[name] = f"kit record STATS[{key!r}]={st.get(key)!r}, expected {want!r}"
        elif kind == "acc":
            acc = getattr(mod, "_acc", {})
            if acc.get(key):
                applied.append(name)
            else:
                fallback.append(name); why[name] = f"kit record _acc[{key!r}] is not set"
    return {"levers_applied": applied, "levers_fallback": fallback, "levers_unavailable": unavailable, "fallback_reasons": why,
            "partial": bool(fallback)}


# -------------------------------------------------------------------------------------------------------------- activate
def _base_report(res: modes.Resolution, gpu, dry_run: bool) -> dict:
    rep = {"active": False, "dry_run": dry_run, "mode": res.mode, "form": "inproc", "switches": modes.describe_line(res),
           "pythonpath": [kit_src(k) for k in res.kits], "imports": list(res.imports), "levers_planned": list(res.levers),
           "boltzgen_version": boltzgen_version(), "package_version": package_version(), "gpu": gpu,
           "stack_key": stack_key(gpu) if gpu else None, "python": sys.version.split()[0], "torch": getattr(sys.modules.get("torch"), "__version__", None),
           "notes": list(res.notes)}
    return rep


def gates(res: modes.Resolution, gpu) -> Tuple[Optional[str], str]:
    """The first failing gate as a reason (None: all pass) and the GPU gate label. The core pin gate is not here: it is statement one of
    the entry that led here (``boltzgen_opt.core_gate``), before this module and its core imports were loaded."""
    for k in res.kits:
        miss = kit_missing(k)
        if miss:
            return f"kit {k} is not in the tree ({kit_dir(k)}): missing {miss}", "n/a"
    v = boltzgen_version()
    pin = pins()["boltzgen_version"]
    if v is None:
        return f"boltzgen is not installed (pin {pin}: stock/PINS.json); install stock/boltzgen-{pin}-py3-none-any.whl", "n/a"
    if v != pin:
        return f"boltzgen {v} is installed, the pin is {pin} (stock/PINS.json)", "n/a"
    ok, gate, reason = gpu_gate(gpu)
    if not ok:
        return reason, gate
    if gate == "noted":                                                      # admitted by compute capability under another name / memory size: one NOTE line per process, never a refusal
        report.note(f"NOTE {reason}")
    return None, gate


def hook_line_mismatch(res: modes.Resolution) -> Optional[str]:
    """None when the ``sitecustomize`` hook line already in this process is the mode's own: every lever switch present in the
    environment is the mode's export with its value (`bg_hook`
    reads BG_GRAPH at import and `bg_graph_patch` its tuning switches at
    apply time), every name the mode exports is present with its value (a mode's own exports outside the switch prefixes too:
    `big`'s allocator setting), and `sitecustomize` / `bg_hook` are this tree's files; otherwise what differs."""
    present = {k: v for k, v in os.environ.items() if k.startswith(modes.KIT_SWITCH_PREFIXES) and k not in modes.OBSERVABILITY_SWITCHES}
    present.update({k: os.environ[k] for k in res.env if k in os.environ})          # the mode's own exports, whatever their prefix
    diffs = [f"{k}={present.get(k)!r}" for k in sorted(set(present) | set(res.env)) if present.get(k) != res.env.get(k)]
    if diffs:
        return f"kit switches {','.join(diffs)} where the mode's line is {modes.describe_line(res)}"
    partner = kit_dir(modes.KIT_PARTNER)
    for name in ("sitecustomize", "bg_hook"):
        mod = sys.modules.get(name)
        if mod is None:
            continue
        f = os.path.abspath(getattr(mod, "__file__", "") or "")
        if not f.startswith(partner):
            return f"{name} is not the tree's ({f or 'no file'})"
    return None


def _late_activation_refusal(res: modes.Resolution) -> Tuple[Optional[str], bool]:
    """``(reason, hook_first)``: the named refusal, or ``(None, True)`` when ``fast_inference``'s sitecustomize line (`bg_hook`, and
    `bg_graph_patch` under BG_GRAPH) has already run in this process under the mode's own line — a process started with the
    environment of an activated parent, such as one of upstream's per-step interpreters (_autoload.py): the package then completes
    the line with the runner's remaining imports instead of refusing."""
    loaded = kit_modules_loaded()
    hook_first = False
    if loaded:
        foreign = [m for m in loaded if m not in HOOK_ROUTE_MODULES]
        why = None if not foreign else f"{','.join(foreign)} is not the sitecustomize line"
        why = why or hook_line_mismatch(res)
        if why:
            return (f"kit module(s) already imported in this process ({','.join(loaded)}) — {why}: the kits were activated by their own route "
                    f"(PYTHONPATH/sitecustomize); the package does not re-activate", False)
        hook_first = True
    applied = levers_already_applied()
    if hook_first:
        applied = [a for a in applied if registry.LEVERS[a].probe[1] not in HOOK_ROUTE_MODULES]
    if applied:
        return f"lever(s) already applied in this process ({','.join(applied)})", False
    n = instance_count()
    if n:
        return f"{n} {BOLTZ_CLASS} model instance(s) already exist in this process: activate before the model is built", False
    return None, hook_first


def activate(mode: Optional[str] = None, dry_run: bool = False, strict: bool = False) -> dict:
    """Resolve, gate, and (unless dry_run) apply the mode in this process. Idempotent per process; a second call with a different
    mode raises ActivationError. ``strict``: a refusal raises instead of returning an inactive report."""
    global _REPORT
    mode = modes.check_mode(mode)
    if _REPORT is not None and _REPORT.get("active") and not dry_run:
        if _REPORT["mode"] == mode:
            return _REPORT
        raise ActivationError(f"already active in mode {_REPORT['mode']!r}; a process has one mode")
    res = modes.resolve(mode, opt_home())                     # a bad mode raises ValueError uncaught here
    gpu = gpu_probe()
    rep = _base_report(res, gpu, dry_run)
    if not res.active:
        rep["reason"] = "mode off: stock — nothing applied (`boltzgen-opt design --mode off` runs upstream alone, proven clean)"
        rep["gpu_gate"] = "n/a"
        if not dry_run:
            _REPORT = rep
        return rep
    reason, gate = gates(res, gpu)
    rep["gpu_gate"] = gate
    hook_first = False
    if reason is None and (not dry_run or kit_modules_loaded() or instance_count()):
        reason, hook_first = _late_activation_refusal(res)
    if reason is not None:
        rep["reason"] = reason
        if strict:
            raise ActivationError(reason)
        if not dry_run:
            report.emit(rep)
        return rep
    env, dropped = mode_env(res, keep_pythonpath=True, keep_package_env=True)   # in-process: the caller's PYTHONPATH follows the lever directories' entries
    rep["dropped_env"] = dropped
    rep["hook_first"] = hook_first
    if hook_first:
        rep["notes"].append("the partner's sitecustomize had imported bg_hook under the mode's own line before activation (a process started "
                            "with an activated parent's environment): the runner's remaining imports completed it (xa_run.py l.37-39)")
    if dry_run:
        rep["would_export"] = dict(res.env)
        return rep
    # ---- apply: the launch's imports, in this process
    handover = os.environ.get(ENV_HANDOVER) is not None                    # the process form's child: the launching verb judges its run from its lines (design.evidence); every other activated process judges itself at exit
    if not handover:
        arm_exit_verdict()                                                # registered before every other exit line of the kit: atexit runs it last
    report.register_exit_tally()
    for k in dropped:
        os.environ.pop(k, None)
    for k, v in res.env.items():
        os.environ[k] = v
    os.environ[modes.ENV] = mode                                         # every child of this process activates the same mode by the env route (upstream's per-step interpreters)
    os.environ["PYTHONPATH"] = env["PYTHONPATH"]                          # every child starts on fast_inference's sitecustomize line (its route), the finder completing it
    from . import census as _census                                      # the accelerator census: armed here (the line prints at exit) and, unless a caller already set a payload (design.kit_env), exported for the children
    os.environ.setdefault(ENV_KERNELS, _census.payload(mode, "on"))
    rep["kernels_census"] = dict(_census.arm(os.environ[ENV_KERNELS], print_if_unused=False))   # this process prints a line only if it imports the library itself; each model child it starts prints its own
    for p in reversed(rep["pythonpath"]):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    rep["instance_counter"] = arm_instance_counter()
    imported = []
    try:
        for m in res.imports:
            importlib.import_module(m)
            imported.append(m)
    except Exception as e:
        if is_oom(e):
            raise                                                         # an out-of-memory raised while a kit module installs is the caller's to see — never a NOT ACTIVE report, never a stock run in its place
        rep["reason"] = f"kit import {m!r} failed after {imported}: {e!r}"
        _REPORT = rep
        if strict:
            raise ActivationError(rep["reason"]) from e
        report.emit(rep)
        return rep
    rep.update(classify(res))
    rep["active"] = True
    rep["step_gate"] = arm_step_gate(res)                                 # a planned lever that cannot serve a model step's configuration refuses the step by name before its model loads
    rep["exit_verdict"] = not handover                                    # this process refuses by name at exit (3) when a lever fell back during the run
    rep["torch"] = getattr(sys.modules.get("torch"), "__version__", None)
    rep["gpu"] = gpu_probe() if gpu is None else gpu
    _REPORT = rep
    if os.environ.pop(ENV_HANDOVER, None) is not None:                   # the process form's one-shot hand-over (mode_env): this child is activated; nothing it spawns inherits the mode
        os.environ.pop(modes.ENV, None)
    report.emit(rep)                                                      # the activation line, in every activated process
    if rep["partial"]:                                                    # a lever of the mode could not run here — a mode is all of its levers: refused by name (the import hook, strict, ends the process with exit 3)
        reason = report.partial_reason(partial_levers(rep), rep["fallback_reasons"])
        if strict:
            raise ActivationError(reason)
        report.say(report.not_active_line(reason))                        # a library caller owns its own exit: the report's `partial` is its gate
    return rep


# ------------------------------------------------------------------------------------------ in-process judges (after activation)
IN_PROCESS_WHERE = "in the run: this process's outputs are not the mode's"
UPSTREAM_STEP_TASK = ("boltzgen.task.predict.predict", "Predict")      # upstream's model step: ``Predict.run`` loads the model and runs the predict loop (every GPU step of the pipeline)


def runtime_fallbacks(res: modes.Resolution) -> Tuple[List[str], Dict[str, str]]:
    """The planned levers whose kit modules' own records show, by now, that the lever STOPPED serving during the run (registry
    ``fell_back``: a module that disabled itself and says why; a module whose own gate reports its GATE-FAIL problems) — the in-process
    reading of the events the launching verb reads off a child's lines (``RUNNER_DISABLED_LINES``). ([levers], {lever: reason})."""
    fallen, why = [], {}
    for name in res.levers:
        lv = registry.LEVERS[name]
        for kind, module, key in lv.fell_back:
            mod = sys.modules.get(module)
            if mod is None:
                continue
            reason = None
            if kind == "set":
                v = (getattr(mod, "STATS", None) or {}).get(key)
                if v:
                    reason = f"[{module}] {key}: {str(v).strip().splitlines()[-1][:240]}"
            elif kind == "gate":
                try:
                    probs = [str(p) for p in (mod.gate() or [])]
                except Exception as e:                                    # a gate that cannot answer is itself the finding — OOM re-raised first
                    if is_oom(e):
                        raise
                    probs = [f"gate() raised {type(e).__name__}: {e}"]
                if key:
                    probs = [p for p in probs if p.startswith(key)]
                else:
                    claimed = tuple(k for other in registry.LEVERS.values() if other.name != name for kd, md, k in other.fell_back if kd == "gate" and md == module and k)
                    probs = [p for p in probs if not (claimed and p.startswith(claimed))]
                if probs:
                    reason = f"[{module}] GATE-FAIL {' | '.join(probs)[:400]}"
            if reason and name not in why:
                fallen.append(name); why[name] = reason
    return fallen, why


def step_refusals(res: modes.Resolution, task) -> Dict[str, str]:
    """{lever: reason} for every planned lever whose module says it cannot serve upstream's model step ``task`` (registry ``serves``),
    asked before the step's model loads."""
    out = {}
    for name in res.levers:
        sv = registry.LEVERS[name].serves
        mod = sys.modules.get(sv[0]) if sv else None
        fn = getattr(mod, sv[1], None) if mod is not None else None
        reason = fn(task) if fn is not None else None
        if reason:
            out[name] = str(reason)
    return out


def arm_step_gate(res: modes.Resolution) -> bool:
    """Wrap upstream's ``Predict.run`` once in this activated process: before a model step runs, a planned lever that cannot serve the
    step's configuration (``step_refusals``) is the mode's refusal by name — one NOT ACTIVE line, exit 3 (``end_process``: the census line first) before
    the model loads; nothing of the step runs under the mode's name with that lever off. True when armed now."""
    try:
        cls = getattr(importlib.import_module(UPSTREAM_STEP_TASK[0]), UPSTREAM_STEP_TASK[1])
    except Exception as e:                                                # no upstream task class importable here: no model step can run in this process either
        if is_oom(e):
            raise
        return False
    inner = cls.__dict__.get("run") or getattr(cls, "run")
    if getattr(inner, "_boltzgen_opt_step_gate", False):
        return False

    def run(self, *a, **kw):
        refused = step_refusals(res, self)
        if refused:
            end_process(EXIT_NOT_ACTIVE, report.not_active_line(report.step_refusal_reason(res.mode, refused, os.environ.get("BOLTZGEN_PIPELINE_STEP"))))
        return inner(self, *a, **kw)

    run._boltzgen_opt_step_gate = True
    run.__wrapped__ = inner
    cls.run = run
    return True


def end_process(code: int, line: str) -> None:
    """End this process with ``code`` now: its own exit lines first — the accelerator census's KERNELS / PEAK line (``census.print_line``,
    once; on the environment route it was armed at interpreter start, so its own exit hook would run after this one and be cut off) —,
    then ``line`` (the refusal, the last kit line of the process), logging flushed, then ``report.forced_exit`` (the kit's pending EXIT
    tallies, ``os._exit``)."""
    from . import census as _census
    _census.print_line()
    report.say(line)
    logging.shutdown()
    report.forced_exit(code)


def _exit_verdict() -> None:
    """At interpreter exit of an activated process that built a model (``models_built``: model work happened here): the levers that fell
    back — at activation (the report's) or during the run (``runtime_fallbacks``) — make the run not the mode's: one NOT ACTIVE line and
    exit 3 (``end_process``; registered before the kit modules' exit hooks, so it runs after their lines). A process that built no model did
    no model work under the mode's name: its exit is its own (a library caller that read the partial report and stopped). A process already
    ending on an uncaught exception keeps its own failure status. A verdict that cannot be reached is itself a refusal (exit 3), never a pass."""
    try:
        rep = _REPORT
        if not rep or not rep.get("active") or rep.get("dry_run"):
            return
        if getattr(sys, "last_exc", None) is not None or getattr(sys, "last_value", None) is not None:
            return
        if models_built() == 0:
            return
        res = modes.resolve(rep["mode"], opt_home())
        fallen, why = runtime_fallbacks(res)
        for name in partial_levers(rep):
            if name not in why:
                fallen.insert(0, name); why[name] = str((rep.get("fallback_reasons") or {}).get(name) or "fell back at activation")
        if not fallen:
            return
        line = report.partial_exit_line(fallen, why, where=IN_PROCESS_WHERE)
    except Exception as e:                                                # the fail-closed judge never passes silently: a verdict that raised is a refusal by name
        line = report.not_active_line(f"exit verdict failed: {e!r}; this process's outputs are not shown to be the mode's: exit {EXIT_NOT_ACTIVE}")
    end_process(EXIT_NOT_ACTIVE, line)


def arm_exit_verdict() -> bool:
    """Register ``_exit_verdict`` with atexit, once per process. True when registered now."""
    global _EXIT_VERDICT_ARMED
    if _EXIT_VERDICT_ARMED:
        return False
    _EXIT_VERDICT_ARMED = True
    atexit.register(_exit_verdict)
    return True


_EXIT_VERDICT_ARMED = False


def partial_levers(rep) -> List[str]:
    """The levers that fell back when the report says the activation is partial (its ``levers_fallback``); [] otherwise."""
    rep = rep or {}
    return list(rep.get("levers_fallback") or []) if rep.get("partial") else []


def status() -> dict:
    return _REPORT if _REPORT is not None else {"active": False, "reason": "not activated in this process"}


# ------------------------------------------------------------------------------------------------- process-form evidence
def levers_from_lines(lines, res: modes.Resolution) -> dict:
    """Levers shown by a child's output lines (process form): the lever modules' own lines (``RUNNER_LINES``). The planned levers of
    ``CALLER_PROVEN_LEVERS`` are the caller's to read off files (design.evidence). Every other planned lever without a line is a
    named fallback — a lever the mode planned and no line shows is never dropped from the account."""
    applied, fallback, why, stats = [], [], {}, {}
    for name in res.levers:
        if name in CALLER_PROVEN_LEVERS:
            continue
        rx = RUNNER_LINES.get(name)
        if rx is None:
            fallback.append(name); why[name] = "no kit line can prove it in the process form (stack.RUNNER_LINES has no entry for it)"
            continue
        hit = any(rx.match(ln) for ln in lines)
        dis = RUNNER_DISABLED_LINES.get(name)
        dis_line = next((ln for ln in lines if dis and dis.match(ln)), None)
        if hit and not dis_line:
            applied.append(name)
        else:
            fallback.append(name); why[name] = dis_line.strip() if dis_line else "no kit line proves it"
    oom = False
    for ln in lines:
        m = RUNNER_STATS_LINE.match(ln)
        if m:
            stats[m.group(1)] = m.group(2).strip()
        elif RUNNER_OOM_LINE.match(ln):
            oom = True
    return {"levers_applied": applied, "levers_fallback": fallback, "fallback_reasons": why, "kit_stats_lines": stats, "oom": oom}
