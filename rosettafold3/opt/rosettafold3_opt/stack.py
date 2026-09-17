"""Activation: tree paths, the two interpreters, pins, the GPU, and the mode's environment row.

`activate(mode)` resolves the mode by NAME (modes.resolve), proves the running interpreter's tree state by sha (tree.classify:
a kit mode needs ``patched``), gates (stock pins via ``stock/check_pins.py``, a visible CUDA GPU, no
contradicting ``RF3_*`` switch already in the environment), and then EXPORTS the row's switches into ``os.environ`` — the only
thing this package ever sets for the kit — before the kit's ``rf3.graph_flags`` reads them at its import. Nothing is patched
in-process: the kit's bytes are already in the interpreter's site-packages (install.py applied the add-on's ``install.sh`` once).

Late activation follows one rule: allowed any time before the patched bytes are imported, refused by name once
``rf3.graph_flags`` (or the sampler that imports it) is in ``sys.modules`` — the flags are read once, when ``graph_flags`` is imported.
A different mode in the same process is refused; repeated calls return the first report.

After activation an import watch prints the APPLIED line when ``rf3.graph_flags`` loads, from the kit's own ``describe()``
and refuses the process if the kit read a different row than the one exported; the exit tally
(report.py) reads the kit's counters at interpreter exit. `activate(mode, dry_run=True)` resolves, gates and reports without
exporting anything (``check``) and works without torch and without a GPU.

A row with an FPF arm (``fast``, ``exact``, ``big``) adds one in-process step: a second import watch fires
right after ``rf3.model.RF3_structure`` executes and applies the arm with the add-on's own adapter (``fpf_apply``: its ``rf3fpf/``
first on ``sys.path``, ``import fpf_rf3_adapter``, ``apply_arm(<arm>)`` — the add-on's own "Minimal use" recipe, its three kernels
served by name from ``opt_core.kernels``); the
adapter's ``describe_v2()`` must then name every component of the arm and the kit's ``describe()`` must still read the exported
row (``@L1`` re-sets the same levers), else the process stops. The FPF APPLIED line reports it; the exit tally reads the adapter's
served / fallback / error counters. Nothing of the add-on is imported before that point, and nothing at all under ``off``.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import threading
from typing import Dict, Optional, Tuple

from . import ActivationError, __version__
from . import big as _big
from . import dtk as _dtk
from . import pf as _pf
from . import mkdit as _mkdit
from . import confhoist as _confhoist
from . import hostlean as _hostlean
from . import upstream as _upstream
from . import awrite as _awrite
from . import prefetch as _prefetch
from . import leversoff as _leversoff
from . import tgbudget as _tgb
from . import mem as _mem
from . import report as _report
from .modes import (DEFAULT_MODE, describe_line, FPF_ENV_PREFIXES, FPF_RELDIR, FPF_RUNTIME_MEMBERS, FPF_SYS_PATH,
                    jit_cache_key, KIT_LEVERS, SUB_LEVERS, KIT_RELDIR, reach_env, Resolution, resolve)
from . import tree as _tree
from . import _core

_gates = _core.load("gates")
_home = _core.load("home")
_kernels = _core.load("kernels")

KERNELS = ("fpf_trimul_v4", "flash_triattn", "lnl_fused")   # the add-on's three kernel modules, served from opt_core.kernels and ROUTED by name at activation (kernels_route;
                                                             # each held to the core's SUMS/<name>.json before any import); the kit carries no copy of them.
                                                             # fpf_trimul_v4: the adapter's TriMul component; flash_triattn: a triangle-attention kernel of the gflash component;
                                                             # lnl_fused: the ln_linear prologue, gate_transpose epilogue and fused_transition Triton kernels of gflash / ttr / apb / res, its tile
                                                             # table the core's carried lnl_fused.tiles_by_arch.json (exported as PF_LNL_TILES by kernels_route)

ENV_STOCK_PYTHON = "ROSETTAFOLD3_OPT_STOCK_PYTHON"   # the pristine interpreter (default: <tree>/stock/venv/bin/python)
ENV_PYTHON = "ROSETTAFOLD3_OPT_PYTHON"            # the patched interpreter (default: this interpreter)
ENV_CKPT = "ROSETTAFOLD3_OPT_CKPT"                # the checkpoint file (stock/PINS.json "weights")
ENV_TALLY_FILE = "ROSETTAFOLD3_OPT_TALLY_FILE"    # where the exit tally also writes its JSON (set by pred, which reads it for its verdict)
ENV_DIGEST_DIR = "ROSETTAFOLD3_OPT_DIGEST_DIR"       # the weights digest memo directory (weights.py; optional)
ENV_N_GPU = "ROSETTAFOLD3_OPT_N_GPU"              # the memory mode's resource axis (--n_gpu P; pred exports it to the fold process, the hook reads it there)
N_GPU_SUPPORTED = (1, 2, 4, 8)                         # --n_gpu values the row-sharded pair stack (rowpair.py) supports; any other P is refused by name
BIG_MODE = "big"
ENV_NAMES = (ENV_STOCK_PYTHON, ENV_PYTHON, ENV_CKPT, ENV_TALLY_FILE, ENV_N_GPU, ENV_DIGEST_DIR)
                                                  # every ROSETTAFOLD3_OPT_* name this tree reads (plus the mode variable, modes.ENV); any other
                                                  # name under the prefix in the environment is a mistyped one and is refused (undeclared_env)
KIT_TRIGGER = "rf3.graph_flags"                   # the kit reads the exported row when this module executes: the APPLIED check runs right after
FPF_TRIGGER = "rf3.model.RF3_structure"           # the FPF arm is applied right after this upstream module executes (every class the adapter patches exists then)

STATE: Dict[str, object] = {"report": None}


class LateActivationError(ActivationError):
    """The kit's bytes are already imported: the flags were read at import and this package does not re-read them."""


# ------------------------------------------------------------------------------------------------------------------- paths
def opt_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tree_root() -> str:
    """The rosettafold3/ directory: two levels above this package (``opt_core.home.tree_home`` with no environment override: ``MODEL_OPT``
    names the tree run.sh runs, never this package's — two checkouts on one machine each resolve their own)."""
    return _home.tree_home(__file__, env_home=None, env_tree=None, levels=2)


def kit_home() -> str:
    return os.path.join(opt_root(), KIT_RELDIR)


def stock_src() -> str:
    """``stock/src``: the pinned upstream's files the kit touches, unpacked at their repository paths (tree.digests hashes them)."""
    return os.path.join(tree_root(), "stock", "src")


def tree_digests():
    """The reference digests every tree-state proof classifies against: the add-on's patched files and stock/src (tree.digests)."""
    return _tree.digests(kit_home(), stock_src())


def fpf_home() -> str:
    return os.path.join(opt_root(), FPF_RELDIR)


def fpf_present() -> Dict[str, object]:
    """The FPF add-on's files a row with an arm needs (``modes.FPF_RUNTIME_MEMBERS``): the adapter, its fused kernels and tile table, the
    fpf_trimul_v4 cell table. The three standalone kernels are not files of the add-on:
    ``kernels_route`` serves them from ``opt_core.kernels`` and holds them there."""
    fh = fpf_home()
    need = tuple(os.path.join(*p.split("/")) for p in FPF_RUNTIME_MEMBERS)
    missing = [f for f in need if not os.path.exists(os.path.join(fh, f))]
    return {"fpf_home": fh, "missing": missing, "ok": not missing, "sys_path": fpf_sys_path(fh)}


def fpf_sys_path(fh: str) -> list:
    """The directories a row with an arm imports from (``modes.FPF_SYS_PATH``: the adapter's ``rf3fpf/``); the kernels are routed names."""
    return [os.path.join(fh, d) for d in FPF_SYS_PATH]


CORE_TRIMUL_TABLE = os.path.join(_kernels.KERNELS_DIR, "fpf_trimul_v4", "table.json")   # opt_core/kernels/fpf_trimul_v4/table.json: the one launch-cell table (9.0 / 8.0 / 10.x rows)

KIT_LEVER_KERNELS = {"dtk": (_dtk.KERNEL,)}            # the core kernels a package lever of a row needs routed beside the add-on's three (modes.KIT_LEVERS)
ROUTED: list = []                                      # the kernel names routed in this process (KERNELS + the row's kit-lever kernels), in route order


def kernels_of(levers) -> Tuple[str, ...]:
    """The kernel names a row routes: the add-on's three plus those of the package levers the row names."""
    extra = [k for lv in (levers or []) for k in KIT_LEVER_KERNELS.get(lv, ()) if k not in KERNELS]
    return tuple(KERNELS) + tuple(dict.fromkeys(extra))


def kernels_route(fh: str, levers=None) -> Dict[str, dict]:
    """Route every kernel of ``kernels_of(levers)`` to the core copy (its launch cells are the core's own table), then hold each route
    (``opt_core.kernels.route_check``: the resolution, the carried bytes, the exports) — all before the first import. The record per name:
    ok, reason, resolved (the core copy), routed, already_imported, runtime_imports (where each run-time import of the kernel resolves)."""
    names = kernels_of(levers)
    for name in names:
        _kernels.route(name)
    ROUTED[:] = list(names)
    os.environ.update(_kernels.exports("fpf_trimul_v4", cells=CORE_TRIMUL_TABLE))   # the TriMul kernel's cells evidence = the shared core's ONE table (this kit carries no cell table of its own)
    if "lnl_fused" in names:
        os.environ.update(_kernels.exports("lnl_fused", tiles=os.path.join(_kernels.KERNELS_DIR, "lnl_fused.tiles_by_arch.json")))   # the carried tile table
    out = {}
    for name in names:
        g = _kernels.route_check(name, fpf_sys_path(fh))
        out[name] = {"ok": g.ok, "reason": g.reason, **{k: g.details.get(k) for k in ("resolved", "core_copy", "routed", "already_imported", "runtime_imports")}}
    return out


def kernels_imported() -> Dict[str, Optional[str]]:
    """Where each routed kernel actually executes from (its module's location once imported; None before)."""
    return {name: _kernels.resolve(name) if name in sys.modules else None for name in (ROUTED or KERNELS)}


def stock_python() -> str:
    return os.environ.get(ENV_STOCK_PYTHON) or os.path.join(tree_root(), "stock", "venv", "bin", "python")


def opt_python() -> str:
    return os.environ.get(ENV_PYTHON) or sys.executable


def checkpoint_path() -> Optional[str]:
    return os.environ.get(ENV_CKPT) or None


def kit_present() -> Dict[str, object]:
    kh = kit_home()
    need = ("install.sh",) + tuple(os.path.join(_tree.PATCHED_DIR, f) for f in _tree.RF3_FILES)
    missing = [f for f in need if not os.path.exists(os.path.join(kh, f))]
    return {"kit_home": kh, "missing": missing, "ok": not missing}


# -------------------------------------------------------------------------------------------------------------------- pins
def pins() -> dict:
    p = os.path.join(tree_root(), "stock", "PINS.json")
    return json.load(open(p, "r", encoding="utf-8"))


def _check_pins_module():
    p = os.path.join(tree_root(), "stock", "check_pins.py")
    spec = importlib.util.spec_from_file_location("rosettafold3_stock_check_pins", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pin_check(python: Optional[str] = None) -> dict:
    """``stock/check_pins.py`` (the one place the pin check lives) on this interpreter, or on ``python`` in a subprocess."""
    if python is None or os.path.realpath(python) == os.path.realpath(sys.executable):
        return _check_pins_module().check()
    p = os.path.join(tree_root(), "stock", "check_pins.py")
    r = subprocess.run([python, p, "--json"], capture_output=True, text=True, timeout=120)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {"ok": False, "route": "error", "detail": (r.stderr or r.stdout).strip()[-400:]}


_CUEQ_PROBE = ("import json\n"
               "try:\n"
               "    import cuequivariance_torch as _c\n"
               "    out = {'version': getattr(_c, '__version__', '?'), 'error': None}\n"
               "except Exception as e:\n"
               "    out = {'version': None, 'error': (type(e).__name__ + ': ' + str(e))[:200]}\n"
               "print(json.dumps(out))\n")


def cueq_probe(python: str, timeout: float = 180) -> dict:
    """Whether ``cuequivariance_torch`` imports on interpreter ``python`` — the import upstream's ``foundry`` makes before routing triangle
    attention / multiplicative update to the cuEquivariance kernels, falling back to its plain statements SILENTLY when it fails (a wheel
    built for another CUDA major, a missing library). ``{"ok", "version", "error"}``."""
    try:
        r = subprocess.run([python, "-c", _CUEQ_PROBE], capture_output=True, text=True, timeout=timeout)
        d = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"ok": False, "version": None, "error": f"probe failed: {e}"[:200]}
    return {"ok": d.get("version") is not None, "version": d.get("version"), "error": d.get("error")}


def cueq_word(probe: Optional[dict]) -> str:
    if not probe:
        return "cueq=?"
    return f"cueq={probe['version']}" if probe.get("ok") else f"cueq=ABSENT({(probe.get('error') or '').split(':')[0]})"


def upstream_versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {"foundry": None, "commit": None, "torch": None, "triton": None, "cuequivariance": None, "atomworks": None}
    try:
        out["foundry"] = importlib.metadata.version("rc-foundry")
        d = importlib.metadata.distribution("rc-foundry")
        du = d.read_text("direct_url.json")
        if du:
            out["commit"] = (json.loads(du).get("vcs_info") or {}).get("commit_id")
    except Exception:
        pass
    for k, dist in (("torch", "torch"), ("triton", "triton"), ("cuequivariance", "cuequivariance-torch"), ("atomworks", "atomworks")):
        try:
            out[k] = importlib.metadata.version(dist)
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------------------------------------------------- gpu
GPU_KEYS = ("name", "cc", "sm", "memory_mib")                # the probe's keys (the report carries this dict)


def gpu_info() -> Optional[Dict[str, object]]:
    """The first visible GPU by nvidia-smi (no torch; the core's probe): {name, cc, sm, memory_mib}; None when none is visible."""
    g = _gates.nvidia_smi_probe()
    return None if not g.get("name") else {k: g[k] for k in GPU_KEYS}


WARM_LIBRARIES = ("torch", "cuequivariance_ops_torch", "cuequivariance_torch")   # torch FIRST: libcue_ops.so links libnvrtc / libcudart, which the
                                                                                    # environment resolves through torch's own loading; importing the ops library
                                                                                    # before torch fails (libnvrtc.so.13 not found) and cuequivariance_torch
                                                                                    # then records its ops as unavailable for the rest of the process


def warm_imports() -> dict:
    """``opt_core.warm_imports(libraries=WARM_LIBRARIES, origin="kit")``: imports torch and then the stock kernel libraries the model will
    import anyway (cuEquivariance) once, at activation, so their import-time cost is paid before the first item rather than inside its forward; no output
    change in any mode. The order is load-bearing (torch before the ops library, see WARM_LIBRARIES). A core without the function steps aside BY NAME
    (``{"state": "absent:opt_core.warm_imports"}``); the call never raises."""
    try:
        import opt_core
        fn = getattr(opt_core, "warm_imports", None)
        if fn is None:
            return {"state": "absent:opt_core.warm_imports"}
        return {"state": "on", "report": dict(fn(libraries=WARM_LIBRARIES, origin="kit") or {})}
    except Exception as e:                                  # never a reason to refuse a mode
        return {"state": f"error:{type(e).__name__}"}


def core_gate() -> dict:
    """The pin facts of this process for the activation report: THE pin gate is ``_core_gate.gate`` — called here it returns the facts it compared (pinned path +
    minimum version, installed ``__version__``) or refuses by name (``CoreGateRefused``: its NOT ACTIVE line, exit 3). No second pin
    comparison lives in this tree."""
    from ._core_gate import gate
    facts = gate(_core.__file__)
    inst = facts["installed"]
    return {"ok": True, "reason": None, "origin": _core.origin(),
            "pinned": {k: facts["pinned"][k] for k in ("path", "version")},
            "imported": {"version": inst.get("version"), "root": inst.get("root")}}


# ----------------------------------------------------------------------------------------------------------- environment
def env_conflicts(res: Resolution, environ=None) -> Dict[str, str]:
    """Switches already in the environment whose value contradicts the row (a caller-set ``RF3_HOIST=0`` under ``exact``)."""
    environ = os.environ if environ is None else environ
    return {k: environ[k] for k, v in res.switches.items() if k in environ and environ[k].strip().lower() != v}


def undeclared_env(environ=None) -> Dict[str, str]:
    """``ROSETTAFOLD3_OPT_*`` / ``ROSETTAFOLD3_BIG_*`` names in the environment this tree does not read (the .pth hook's own filter, ``_autoload.undeclared``: the
    hook refuses them at interpreter start, the CLI before any command, activation before any gate)."""
    from . import _autoload
    return _autoload.undeclared(environ)


def env_extras(environ=None) -> Dict[str, Dict[str, str]]:
    environ = os.environ if environ is None else environ
    return {"fpf_knobs_set": {k: v for k, v in environ.items() if k.startswith(FPF_ENV_PREFIXES)}}


def bytes_imported() -> Optional[str]:
    for m in ("rf3.graph_flags", "rf3.diffusion_samplers.inference_sampler", "rf3.model.RF3_structure", "rf3.model.layers.af3_diffusion_transformer"):
        if m in sys.modules:
            return m
    return None


def fpf_imported() -> Optional[str]:
    """The FPF adapter (or a module it patches) already in this process: its class patches are applied at ``apply_arm`` time, so an
    adapter imported before activation may carry a different arm."""
    for m in ("fpf_rf3_adapter", "fpf_trimul_v4", "flash_triattn", "lnl_fused"):
        if m in sys.modules:
            return m
    return None


def export_row(res: Resolution) -> Dict[str, str]:
    for k, v in res.switches.items():
        os.environ[k] = v
    return dict(res.switches)


# ---------------------------------------------------------------------------------------------------------------- report
def status() -> dict:
    rep = STATE["report"]
    return dict(rep) if rep else {"active": False, "reason": "enable() has not run in this process"}


def _base_report(res: Resolution, dry_run: bool, trigger: Optional[str]) -> dict:
    g = gpu_info()
    up = upstream_versions()
    rep = {"active": False, "dry_run": dry_run, "mode": res.mode, "row": res.row, "is_house_mode": res.is_house_mode, "switches": dict(res.switches),
            "switch_line": describe_line(res), "levers": list(res.levers), "levers_off": list(res.levers_off), "levers_applied": [],
            **env_extras(), "tree_state": None, "tree_line": None, "site_packages": None, "gpu": g,
            "kernel_key": jit_cache_key(up.get("torch"), g.get("cc") if g else None), "upstream": up, "torch": up.get("torch"),
            "package_version": __version__, "python": sys.executable, "trigger": trigger, "applied": None, "reason": None,
            "target_gpu": os.environ.get("MODEL_OPT_TARGET_GPU") or None, "notes": list(res.notes), "n_gpu": 1, "sharding": "none", "visible_gpus": None,
            "absent": list(getattr(res, "absent", []) or []),                                                           # MODEL_OPT_LEVERS_OFF words with no lever class here (compile): ` compile=none`
            "withheld": list(getattr(res, "withheld", []) or []),                                                       # MODEL_OPT_LEVERS_OFF (leversoff.py): the ACTIVE / EXIT lines' withheld=<names>, the LEVER lines' reason
            "fpf": ({"arm": res.fpf_arm, "components": list(res.fpf_components), "lever_state": res.fpf_lever_state, "home": fpf_home(),
                     "sys_path": fpf_present()["sys_path"], "knobs": dict(res.fpf_knobs), "trigger": FPF_TRIGGER, "applied": None, "cfg": None,
                     "reason": None} if res.fpf_arm else None)}
    if g and rep["target_gpu"] and rep["target_gpu"].lower() not in str(g.get("name", "")).lower():
        rep["notes"].append(f"MODEL_OPT_TARGET_GPU={rep['target_gpu']} but the GPU is {g.get('name')}: not the GPU this configuration targets")
    return rep


def visible_gpus() -> int:
    """The number of CUDA devices this process may use: ``CUDA_VISIBLE_DEVICES`` when set (its entry count), else nvidia-smi's list (the
    core's probe; no torch import here — the hook runs before the engine imports it)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        return len([d for d in cvd.split(",") if d.strip() != ""])
    try:                                                    # nvidia-smi's device list (the core's probe reads one index; the count is this line count)
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
    except Exception:                                       # noqa: BLE001 — no nvidia-smi: none visible (the gate then refuses by name)
        return 0
    return len([ln for ln in out.stdout.splitlines() if ln.strip()]) if out.returncode == 0 else 0


def _rowpair_conflicts() -> Dict[str, str]:
    """The kit levers the row-sharding adapter turns OFF BY NAME under ``n_gpu > 1`` (``rowpair.CONFLICTS``: lever -> sentence)."""
    from . import rowpair as _rowpair
    return dict(_rowpair.CONFLICTS)


def in_rank_process() -> bool:
    """True inside a rank process the family launcher started (``opt_core.mem.rowpair.launch`` sets its ``ROWPAIR_*`` rank environment;
    absent in a P == 1 run and in the launching ``pred`` process)."""
    launch = _core.load("mem.rowpair.launch")
    return os.environ.get(launch.ENV_RANK) is not None


def n_gpu_requested(n_gpu=None) -> int:
    """``--n_gpu`` as given, else ``ROSETTAFOLD3_OPT_N_GPU``, else 1 — a positive integer (``opt_core.mem.ngpu.check_n_gpu``)."""
    ngpu = _core.load("mem.ngpu")
    raw = n_gpu if n_gpu is not None else (os.environ.get(ENV_N_GPU) or 1)
    try:
        return ngpu.check_n_gpu(raw)
    except ValueError as e:
        raise ActivationError(f"refused: {e}") from e


def n_gpu_gate(mode: str, n_gpu=None) -> dict:
    """The ``--n_gpu P`` rules, worded by ``opt_core.mem.ngpu`` (never here): P > 1 only under ``big`` (sharded reductions are not
    bitwise), only with >= P GPUs visible (never auto-sized), only when this tree carries the row-sharding adapter (``rowpair.py`` on
    ``opt_core.mem.rowpair``). Returns the report fields ``{n_gpu, sharding, visible_gpus}``; raises ActivationError with the sentence."""
    ngpu = _core.load("mem.ngpu")
    p = n_gpu_requested(n_gpu)
    vis = None
    try:
        ngpu.refuse_unless_big(p, mode)
        if p not in N_GPU_SUPPORTED:                            # the values the row-sharded pair stack (rowpair.py) supports; others refused by name
            raise ActivationError(f"refused: n_gpu={p} is not a supported value on rosettafold3 (supported: {','.join(str(x) for x in N_GPU_SUPPORTED)} — "
                                  "the values the row-sharded pair stack supports)")
        if p > 1 and not in_rank_process():                 # the visible-GPU rule and the plan are the LAUNCHING process's checks; a rank process
            vis = visible_gpus()                            # of the family launcher sees the devices the launcher bound it to (launch.rank_env)
            ngpu.refuse_unless_visible(p, vis)
    except ngpu.NGpuRefused as e:
        raise ActivationError(str(e.reason if hasattr(e, "reason") else e)) from e
    if p > 1 and not in_rank_process():
        try:
            from . import rowpair as _rowpair                 # the engine's row-sharding adapter (opt_core.mem.rowpair at the pair-stack sites)
        except ImportError as e:
            raise ActivationError(f"refused: n_gpu={p}: the row-sharding adapter is not importable in this tree ({e})") from e
        plan = _rowpair.plan(p, mode)
        if not plan or plan.get("sharding") != ngpu.sharding_value(p) or not plan.get("installs"):
            raise ActivationError(f"refused: n_gpu={p}: the row-sharding adapter declines ({(plan or {}).get('reason') or 'no installs planned'})")
    return {"n_gpu": p, "sharding": ngpu.sharding_value(p), "visible_gpus": vis if p > 1 else None}


def activate(mode: str, *, strict: bool = False, dry_run: bool = False, trigger: Optional[str] = None,
             n_gpu: Optional[int] = None) -> dict:
    """See module docstring. Returns the activation report; ``strict`` raises ActivationError instead of returning a report that is not active. ``n_gpu``: the
    memory mode's resource axis (``--n_gpu P``; None reads ``ROSETTAFOLD3_OPT_N_GPU``, absent = 1) — gated by ``opt_core.mem.ngpu`` (the one
    producer of the rule and its words): P > 1 only under ``big``, only with P GPUs visible, only with the row-sharding adapter present."""
    rep_prev = STATE["report"]
    if rep_prev and not dry_run:
        if rep_prev["row"] == (mode or DEFAULT_MODE):
            return dict(rep_prev)
        raise ActivationError(f"mode {rep_prev['row']!r} is already active in this process; {mode!r} refused (switches are read once at import)")
    try:
        res = resolve((mode or "").strip().lower() or DEFAULT_MODE, kit_home(), fpf_home=fpf_home())
    except _leversoff.WithholdError as e:                                        # MODEL_OPT_LEVERS_OFF refused BY NAME (leversoff.validate): the NOT ACTIVE line and rc 3 of every route, never a traceback
        res = _leversoff.without_request(resolve, (mode or "").strip().lower() or DEFAULT_MODE, kit_home(), fpf_home=fpf_home())
        rep = _base_report(res, dry_run, trigger)
        rep["reason"] = str(e)
        if not dry_run:
            _report.print_not_active(rep)
        if strict:
            raise ActivationError(str(e)) from e
        return rep
    except ValueError as e:
        raise ActivationError(str(e)) from e
    rep = _base_report(res, dry_run, trigger)

    def fail(reason: str, exc=ActivationError):
        rep["reason"] = reason
        if not dry_run:
            _report.print_not_active(rep)
        if strict:
            raise exc(reason)
        return rep

    if res.row == "off":
        rep["reason"] = "off: nothing to activate (stock is the upstream CLI on the pristine interpreter: pred --mode off)"
        return rep
    rep["core"] = core_gate()
    if not rep["core"]["ok"]:
        return fail(rep["core"]["reason"])
    if not dry_run:
        rep["warm_imports"] = warm_imports()                # the heavy stock-library imports (cuEquivariance) happen here, at activation, not inside the first item's forward
    try:
        rep.update(n_gpu_gate(res.mode, n_gpu))         # n_gpu, sharding, visible_gpus: the tokens of the ACTIVE / EXIT lines (opt_core.mem.ngpu)
    except ActivationError as e:
        return fail(str(e))
    kp = kit_present()
    if not kp["ok"]:
        return fail(f"kit files missing under {kp['kit_home']}: {kp['missing']}")
    if res.fpf_arm:
        fp = fpf_present()
        if not fp["ok"]:
            return fail(f"FPF add-on files missing under {fp['fpf_home']}: {fp['missing']}")
        early = fpf_imported()
        if early and not dry_run:
            return fail(f"{early} is already imported: the FPF adapter's patches are applied per arm at activation; activate before importing it",
                        LateActivationError)
        if not dry_run:
            rep["fpf"]["kernels"] = kernels_route(fp["fpf_home"], res.levers)
            bad_route = [f"{k}: {v['reason']}" for k, v in rep["fpf"]["kernels"].items() if not v["ok"]]
            if bad_route:
                return fail("kernel route: " + "; ".join(bad_route))
        else:
            rep["fpf"]["kernels"] = {name: {"core_copy": _kernels.carried_path(name), "carry": _kernels.verify_carry(name)} for name in kernels_of(res.levers)}
        if rep["fpf_knobs_set"]:
            rep["notes"].append("FPF knobs set in the environment (the add-on's defaults are the row's): " + ",".join(f"{k}={v}" for k, v in rep["fpf_knobs_set"].items()))
    late = bytes_imported()
    if late and not dry_run:
        return fail(f"{late} is already imported: its switches were read at import; activate before the first import of rf3.graph_flags", LateActivationError)
    try:
        lists = tree_digests()
        ts = _tree.classify(_tree.this_site_packages(), lists)
    except Exception as e:
        return fail(f"tree state unavailable: {e}")
    rep.update({"tree_state": ts.state, "tree_line": ts.line(), "site_packages": ts.site_packages, "tree_files": dict(ts.files)})
    if ts.state != res.tree_state:
        return fail(f"this interpreter's tree is {ts.rf3_count}, not {res.tree_state}: row {res.row} runs only on the {res.tree_state} "
                    f"interpreter (rosettafold3-opt install; opt/venv) — {ts.line()}")
    pc = pin_check()
    rep["pins"] = pc
    if not pc.get("ok"):
        return fail(f"stock pin check failed: {pc.get('detail')}")
    conflicts = env_conflicts(res)
    if conflicts:
        return fail(f"environment contradicts row {res.row}: " + ", ".join(f"{k}={v!r} (row says {res.switches[k]!r})" for k, v in conflicts.items()))
    undeclared = undeclared_env()
    if undeclared:
        return fail("undeclared ROSETTAFOLD3_OPT_* / ROSETTAFOLD3_BIG_* names in the environment (mistyped? this tree reads " + ", ".join(ENV_NAMES) + "): "
                    + ", ".join(f"{k}={v!r}" for k, v in undeclared.items()))
    try:
        _mem.apply(rep)                             # the memory policy (mem.py): resolved here as a gate — a value outside the table is refused by name
    except _mem.MemPolicyError as e:
        return fail(str(e))
    if rep["gpu"] is None:
        if dry_run:
            rep["notes"].append("no CUDA GPU visible here: the graph lever captures on a CUDA device at run time")
        else:
            return fail("no CUDA GPU visible (nvidia-smi): the sampler graph needs one")
    if res.row == BIG_MODE:
        from . import big as _big                   # the memory mode: composes the line and resolves the flags here (a gate), applies at its trigger
        try:
            _big.install(rep, arm=not dry_run)
        except _big.BigError as e:
            return fail(str(e))
        rep["levers"] = list(rep["levers"]) + [lv for lv in rep["big"]["expected"] if lv not in rep["levers"]]
        if rep["n_gpu"] > 1:
            rep["levers"] = list(rep["levers"]) + ["rowpair"]   # the row-sharded pair stack (F7.tensor_parallel): installed at the model trigger on every rank (rowpair.py)
    if dry_run:
        rep["applied"] = "dry-run"
        return rep
    export_row(res)
    for _k, _v in reach_env(rep.get("n_gpu", 1)).items():   # the row-sharded line's ROWPAIR_* levers at --n_gpu > 1 (modes.TP_ENV), exported and named on the ACTIVE line
        os.environ[_k] = _v; rep["switches"][_k] = _v
    rep["active"] = True
    rep["applied"] = "deferred"                     # the kit reads the row when rf3.graph_flags is imported; the memory policy's seams by the same watch
    if rep["fpf"]:
        rep["fpf"]["applied"] = "deferred"          # the adapter applies the arm right after rf3.model.RF3_structure executes
    STATE["report"] = rep
    _report.register_exit_tally(rep, tally_file=os.environ.get(ENV_TALLY_FILE))
    _install_apply_watch(rep)
    if rep["fpf"]:
        _install_fpf_watch(rep)
    elif "xtr" in rep["levers"]:
        _install_watch(FPF_TRIGGER, lambda module: xtr_apply(rep), rep)   # an arm-less row's package lever applies at the same trigger
    if rep.get("n_gpu", 1) > 1:
        _install_rowpair_watch(rep)
    _install_template_gate(rep)
    _confhoist.arm(rep, _install_watch)                       # the confidence-head prologue hoist (confhoist.py): armed here, patches ConfidenceHead when its module executes (rows naming it)
    _hostlean.arm(rep, _install_watch)                        # validation_step's dead symmetry resolutions (hostlean.py): armed here, patches the trainer classes when rf3.trainers.rf3 executes
    _upstream.arm(rep, _install_watch)                        # the upstream-outcome census (upstream.py): every row; counts items / early stops on the same trainer classes, byte-neutral
    _prefetch.arm(rep, _install_watch)                        # the persistent featurizer (prefetch.py): armed here when the row names it, installs on rf3.inference_engines.rf3 when it executes
    _awrite.arm(rep, _install_watch)                          # the asynchronous writer (awrite.py): same trigger; both fork their side process at _construct_pipeline, before CUDA
    _report.print_active(rep)
    if rep["fpf"]:
        _report.print_kernels(rep)
    _report.print_mem(rep)
    if rep.get("big"):
        _report.print_big(rep)
    return dict(rep)


# ------------------------------------------------------------------------------------------------------------ FPF apply
class _ImportWatch:
    """Meta-path finder that fires ``on_executed(module)`` ONCE, right after the module named ``trigger`` executes, and removes
    itself then. It disarms inside the exec_module wrapper, not in ``find_spec``: a spec lookup that is not followed by an import
    (``importlib.util.find_spec(trigger)``) leaves it armed for the real import (the same rule as ``_autoload.Finder``). Another
    finder that hooks the SAME module by resolving it through ``importlib.util.find_spec`` (a ``sitecustomize`` attached on
    ``PYTHONPATH`` with a post-import hook on ``rf3.model.RF3``, say) re-walks ``sys.meta_path`` while this finder is answering: that nested
    walk is not answered (a thread-local re-entrancy guard, so the loader is wrapped once), and a loader that still ends up wrapped more
    than once fires the callbacks once (the ``fired`` latch; the nesting is named under ``rep["watch_nested"]``)."""

    def __init__(self, trigger: str, on_executed, rep: dict):
        self.trigger = trigger
        self.callbacks = [on_executed]                       # every callback registered for this trigger, in installation order
        self.rep = rep
        self.fired = False                                  # the callbacks ran (once per watch, whatever the wrapping depth)
        self._busy = threading.local()                      # find_spec is answering on this thread: a nested walk is not answered again

    def chain(self, on_executed) -> None:
        """A further callback for the same trigger (``_install_watch`` with a trigger already watched): runs after the earlier ones."""
        self.callbacks.append(on_executed)

    def on_executed(self, module) -> None:
        for cb in list(self.callbacks):
            cb(module)

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.trigger or self.fired or getattr(self._busy, "on", False):
            return None
        self._busy.on = True
        try:
            spec = None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                try:
                    spec = finder.find_spec(fullname, path, target)
                except Exception as e:                      # a finder that refuses the name: recorded, the others are asked
                    self.rep.setdefault("finder_errors", []).append(f"{type(finder).__name__}: {e!r}")
                    spec = None
                if spec is not None:
                    break
        finally:
            self._busy.on = False
        if spec is None or spec.loader is None:
            return None
        orig = spec.loader.exec_module
        watch = self

        def exec_module(module, _orig=orig):
            _orig(module)
            if watch.fired:                                 # this loader was wrapped more than once (a nested spec resolution): the callbacks
                nested = watch.rep.setdefault("watch_nested", [])   # ran inside _orig already — named, never run again
                nested.append(watch.trigger)
                return
            watch.fired = True
            try:
                sys.meta_path.remove(watch)
            except ValueError:
                pass
            watch.on_executed(module)
        spec.loader.exec_module = exec_module
        return spec


def _install_watch(trigger: str, on_executed, rep: dict) -> None:
    """One watch per trigger module; a second callback for the same trigger is CHAINED after the first (in installation order: the
    big levers apply before the row-sharding adapter installs on ``rf3.model.RF3``), never a replacement."""
    for f in sys.meta_path:
        if isinstance(f, _ImportWatch) and f.trigger == trigger:
            f.chain(on_executed)
            return
    sys.meta_path.insert(0, _ImportWatch(trigger, on_executed, rep))


def _install_fpf_watch(rep: dict) -> None:
    _install_watch(FPF_TRIGGER, lambda module: fpf_apply(rep), rep)


def fpf_apply(rep: dict) -> dict:
    """Apply the row's FPF arm with the add-on's own adapter (once per process): ``sys.path[:0] = [<fpf>/rf3fpf]`` (its kernels are the
    routed names, served from the core copies since activation), ``import fpf_rf3_adapter``, ``apply_arm(<arm>)``; then check the
    kit's ``describe()`` still reads the exported row (``@L1`` re-sets the same levers), that the adapter's own config names every
    component of the arm, and that every kernel executes from the core copy (``kernels_imported``)."""
    f = rep["fpf"]
    if f.get("applied") == "configured":
        return rep
    for d in reversed(f["sys_path"]):                        # the add-on's directories recorded at activation (fpf_present)
        if d not in sys.path:
            sys.path.insert(0, d)
    try:
        import fpf_rf3_adapter as adp                        # the add-on's module, from its own directory
        cfg = adp.apply_arm(f["arm"])
        desc = adp.describe_v2()
        f["kernels_imported"] = kernels_imported()
    except Exception as e:
        f["applied"] = "error"
        f["reason"] = f"FPF apply_arm({f['arm']!r}) failed: {type(e).__name__}: {e}"
        rep["active"] = False
        rep["reason"] = f["reason"]
        _report.print_not_active(rep)
        raise ActivationError(f["reason"]) from e
    f["cfg"] = {k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v)) for k, v in dict(cfg).items()}
    f["adapter_file"] = getattr(adp, "__file__", None)
    bad = fpf_cfg_mismatch(f["components"], dict(desc.get("cfg") or {}), trimul_mode=dict(adp.describe()).get("mode"), lever_state=f["lever_state"])
    bad += [f"{k} executes from {v}, not the core copy {f['kernels'][k]['core_copy']}" for k, v in f["kernels_imported"].items()
            if v is not None and os.path.realpath(v) != os.path.realpath(f["kernels"][k]["core_copy"])]
    gf = sys.modules.get(KIT_TRIGGER)
    if gf is not None:
        bad += kit_describe_mismatch(gf, rep["switches"])
    elif f["lever_state"] is not None:
        bad.append(f"{KIT_TRIGGER} is not imported: the arm's @L step had no kit switches to set")
    if bad:
        f["applied"] = "mismatch"
        f["reason"] = "the adapter's configuration is not the arm's: " + "; ".join(bad)
        rep["active"] = False
        rep["reason"] = f["reason"]
        _report.print_not_active(rep)
        raise ActivationError(f["reason"])
    from .modes import fpf_lever_subs as _fpf_lever_subs
    want_subs = _fpf_lever_subs(f["arm"])
    if "warm" in want_subs and f["cfg"].get("warm") is not True:   # the arm's warm sub-step must engage (an older add-on reports 'unavailable'): all-or-refuse, by name
        f["applied"] = "mismatch"; f["reason"] = f"the arm's lever sub-step warm did not engage (adapter cfg warm={f['cfg'].get('warm')!r})"
        rep["active"] = False; rep["reason"] = f["reason"]; _report.print_not_active(rep)
        raise ActivationError(f["reason"])
    f["applied"] = "configured"
    if "tg" in (f.get("components") or []):                    # the trunk-graph token budget (tgbudget.py): above it the pre-graph forward, a named skip
        try:
            f["tg_budget"] = _tgb.enable(adp)
        except Exception as e:
            f["applied"] = "error"
            f["reason"] = f"trunk-graph budget not installed: {type(e).__name__}: {e}"
            rep["active"] = False
            rep["reason"] = f["reason"]
            _report.print_not_active(rep)
            raise ActivationError(f["reason"]) from e
    rep["levers_applied"] = list(dict.fromkeys(list(rep.get("levers_applied") or []) + [lv for lv in rep["levers"] if lv.startswith("fpf_")]))
    if "dtk" in (rep.get("levers") or []) and int(rep.get("n_gpu") or 1) > 1 and "dtk" in _rowpair_conflicts():
        rep["dtk"] = _dtk.decline("n_gpu", _rowpair_conflicts()["dtk"])   # off by name under n_gpu>1 (rowpair.CONFLICTS): the row-sharded statement owns the site; the tally carries the word
    elif "dtk" in (rep.get("levers") or []):                   # the package's own lever on the add-on's seam (dtk.py), after the arm
        try:
            rep["dtk"] = _dtk.enable(kernel_file=(f["kernels"].get(_dtk.KERNEL) or {}).get("core_copy"))
        except Exception as e:
            rep["dtk"] = {"on": False, "reason": f"dtk not installed: {type(e).__name__}: {e}"}
            rep["active"] = False
            rep["reason"] = rep["dtk"]["reason"]
            _report.print_not_active(rep)
            raise ActivationError(rep["reason"]) from e
        rep["levers_applied"] = list(dict.fromkeys(rep["levers_applied"] + ["dtk"]))
    if "mkdit" in (rep.get("levers") or []):                   # the package's megakernel lever (mkdit.py): DiffusionTransformer.forward class-wide, after the arm
        try:
            rep["mkdit"] = _mkdit.enable(opt_root=opt_root(), gpu=rep.get("gpu"))
        except _mkdit.MkditRefused as e:
            rep["mkdit"] = {"on": False, "reason": f"mkdit not installed: {e}"}
            rep["active"] = False
            rep["reason"] = rep["mkdit"]["reason"]
            _report.print_not_active(rep)
            raise ActivationError(rep["reason"]) from e
        rep["levers_applied"] = list(dict.fromkeys(rep["levers_applied"] + ["mkdit"]))
        _report.print_mkdit(rep)
    xtr_apply(rep)                                             # the package's exact-construction transition (pf.py), class-wide, after the arm
    _report.print_fpf_applied(rep)
    return rep


def xtr_apply(rep: dict) -> dict:
    """Apply the ``xtr`` lever (pf.py) when the row names it: right after the FPF arm (fpf_apply) on rows with an arm, or on its own at the
    same trigger (``rf3.model.RF3_structure`` executed: the Transition class exists) on rows without one. The core's serve layer routes and
    sum-checks the carried cells it launches (opt_core.attn.pair_fused); a refused precondition is the NOT ACTIVE line."""
    if "xtr" not in (rep.get("levers") or []) or (rep.get("xtr") or {}).get("on"):
        return rep
    try:
        rep["xtr"] = _pf.enable()
    except Exception as e:
        rep["xtr"] = {"on": False, "reason": f"xtr not installed: {type(e).__name__}: {e}"}
        rep["active"] = False
        rep["reason"] = rep["xtr"]["reason"]
        _report.print_not_active(rep)
        raise ActivationError(rep["reason"]) from e
    if rep["xtr"].get("on"):
        rep["levers_applied"] = list(dict.fromkeys(list(rep.get("levers_applied") or []) + ["xtr"]))
    _report.print_xtr(rep)                                       # on, or the named no-op of a CPU helper process (pf.CPU_PROCESS in the line's reason)
    return rep


def fpf_cfg_mismatch(components, cfg: Dict[str, object], trimul_mode: Optional[str] = None, lever_state: Optional[bool] = None) -> list:
    """Every component of the arm against the adapter's own record of what is installed: ``describe_v2()["cfg"]`` for the component
    levers and the ``@L1`` step (``cfg["levers"]``: True / "unavailable", ``set_kit_levers``), ``describe()["mode"]``
    (``trimul_mode``) for the trimul choice."""
    want: Dict[str, object] = {"trimul": "stock", "triattn": "stock", "transition": "stock", "apb": "stock", "trunk_graph": False, "dattn": False, "res": False, "msa": False, "smsa": False}
    if lever_state is not None:
        want["levers"] = lever_state
    for p0 in components:
        p = p0.partition(".")[0]                              # a sub-word (fast.tmk3_fast, gflash.k2b, xmul.tmk3_exact, apb.sdpa, xln.fast) names the row, not the component
        if p in ("stock", "fast"):
            want["trimul"] = p
        elif p == "gflash":
            want["triattn"] = p
        elif p == "ttr":
            want["transition"] = "triton"
        elif p == "apb":
            want["apb"] = "triton"
            if p0.partition(".")[2] and p0.partition(".")[2] != "stmt":
                want["apb_word"] = p0.partition(".")[2]             # apb.<word>: the attention core's provider row after the fused producer (fpf_rf3_apb_rows)
        elif p == "sapb":
            want["apb"] = "safe"
        elif p == "tg":
            want["trunk_graph"] = True
        elif p == "dattn":
            want["dattn"] = True
        elif p == "res":
            want["res"] = True
        elif p == "xmul":
            want["xmul"] = p0.partition(".")[2] or "exact"
        elif p == "xln":
            want["xln"] = p0.partition(".")[2] or "exact"
        elif p == "xatt":                                 # the exact tier's triangle attention: the provider word it asks (describe_v2 cfg["xatt"])
            want["xatt"] = p0.partition(".")[2] or "exact"
        elif p == "msa":
            want["msa"] = p0.partition(".")[2] or "card"          # msa[.<word>]: the word applied (the cells it binds on this card are the record's msa_units, a run-time value)
        elif p == "smsa":
            want["smsa"] = True
    out = []
    for k, v in want.items():
        if k == "trimul":
            if trimul_mode is not None and trimul_mode != v:
                out.append(f"trimul: adapter={trimul_mode!r} arm={v!r}")
            continue
        if k == "levers":
            if cfg.get("levers") is not v:
                out.append(f"levers: adapter={cfg.get('levers')!r} arm=@L{int(v)} ({v!r})")
            continue
        if k in cfg and cfg[k] != v:
            out.append(f"{k}: adapter={cfg[k]!r} arm={v!r}")
        elif k not in cfg and v not in ("stock", False):
            out.append(f"{k}: absent from the adapter's cfg, arm wants {v!r}")
    return out


# ---------------------------------------------------------------------------------------------------------- apply watch
def _install_template_gate(rep: dict) -> None:
    """The template census (templ.py) on every kit row: wraps ``rf3.utils.inference.apply_template_selection`` when that module executes
    (or at once when it is already imported); the report's ``template`` block is its census (report.tally reads it)."""
    from . import templ as _templ
    rep["template"] = _templ.STATE
    mod = sys.modules.get(_templ.MODULE)
    if mod is not None:
        _templ.wrap(mod)
        return
    _install_watch(_templ.MODULE, lambda module: _templ.wrap(module), rep)


ROWPAIR_TRIGGER = "rf3.model.RF3"                  # the model module: the row-sharding adapter patches the pair-stack classes once it has executed (rowpair.install)


def _install_rowpair_watch(rep: dict) -> None:
    """Under ``n_gpu > 1``: when the model module has executed, the row-sharding adapter installs its forwards on every rank
    (``rowpair.install``); a refusal there is the NOT ACTIVE line and exit 3 of this rank (the family launcher tears the group down)."""
    def on_executed(module):
        from . import rowpair as _rowpair
        try:
            rep["rowpair"] = _rowpair.install(rep, int(rep["n_gpu"]))
        except Exception as e:                              # noqa: BLE001 — named, then the kit's exit code; never a half-sharded run
            rep["reason"] = f"refused: n_gpu={rep['n_gpu']}: the row-sharding adapter failed to install ({type(e).__name__}: {e})"
            _report.print_not_active(rep)
            sys.stderr.flush()
            os._exit(_core.load("report").EXIT_NOT_ACTIVE)
        if not (rep["rowpair"] or {}).get("installed"):
            rep["reason"] = f"refused: n_gpu={rep['n_gpu']}: the row-sharding adapter installed nothing ({(rep['rowpair'] or {}).get('reason')})"
            _report.print_not_active(rep)
            sys.stderr.flush()
            os._exit(_core.load("report").EXIT_NOT_ACTIVE)
    _install_watch(ROWPAIR_TRIGGER, on_executed, rep)


def _install_apply_watch(rep: dict) -> None:
    """The kit module's watch: the APPLIED check, then the memory policy's seams (mem.py; the same module, one watch)."""
    def on_executed(module):
        applied_check(module, rep)
        if rep.get("mem", {}).get("policy") not in (None, "off"):
            _mem.wrap(module)
    _install_watch(KIT_TRIGGER, on_executed, rep)


def expected_describe(switches: Dict[str, str]) -> Dict[str, object]:
    """What the kit's describe() must show for a row: RF3_CUDAGRAPH as its normalised string, RF3_HOIST as a bool
    (``graph_flags.py``'s own normalisation of the two variables)."""
    want: Dict[str, object] = {}
    if "RF3_CUDAGRAPH" in switches:
        v = switches["RF3_CUDAGRAPH"]
        want["RF3_CUDAGRAPH"] = "1" if v in ("1", "true", "on", "graph", "cudagraph") else ("replay" if v in ("replay", "eager", "predraw") else "0")
    if "RF3_HOIST" in switches:
        want["RF3_HOIST"] = switches["RF3_HOIST"] in ("1", "true", "on")
    return want


def kit_describe_mismatch(gf_module, switches: Dict[str, str]) -> list:
    """The kit's own ``describe()`` against the exported row: one ``<switch>: kit=<read> row=<exported>`` per disagreement."""
    desc = dict(gf_module.describe())
    want = expected_describe(switches)
    return [f"{k}: kit={desc.get(k)!r} row={v!r}" for k, v in want.items() if desc.get(k) != v]


def applied_check(gf_module, rep: dict) -> dict:
    rep["describe"] = dict(gf_module.describe())
    bad = kit_describe_mismatch(gf_module, rep["switches"])
    if bad:
        rep["applied"] = "mismatch"
        rep["active"] = False
        rep["reason"] = "the kit read a different row than the one exported: " + ", ".join(bad)
        _report.print_not_active(rep)
        raise ActivationError(rep["reason"])
    rep["applied"] = "configured"
    rep["levers_applied"] = [lv for lv in rep["levers"] if not lv.startswith("fpf_") and lv not in KIT_LEVERS and lv not in SUB_LEVERS]   # the FPF levers and the package levers (dtk, xtr) are applied by fpf_apply / xtr_apply, later
    _report.print_applied(rep)
    return rep
