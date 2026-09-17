"""Activation: resolve a mode or line (modes.py), gate it on this box, apply it once per process by executing the kit's own shim.

What "apply" means here — and all it means: the line's switches are exported into ``os.environ`` (its must-be-absent switches
removed), the line's kit directories go to the front of ``sys.path`` in the line's order, and the hook's ``sitecustomize.py`` is
executed as a module — the same file the kit's README puts first on ``PYTHONPATH``. From there on the kit's code does everything the
kit's own route does: the served-levers hook wraps ``runner.batch_inference.get_default_runner`` and installs the kit levers, the
DITFAST levers and the ARM-T arm on the runner it returns. For the FPF TriMul line (``S1``) the package then calls the
kit's own ``fpf_engines.enable_from_env()`` at the line's exported contract (``FPF_ENGINE/FPF_OPS/FPF_IMPL``), the in-process enable
the kit spells for that line. Nothing is transcribed, nothing is patched by this package.

Late activation, one rule: ``enable()`` may run any time after the upstream packages are imported, and is refused by name once
(a) the served-levers hook reports a wrap or an install, an ARM-T arm, a DITFAST lever or an FPF op is installed, or (b) an
``InferenceRunner`` instance exists in the process. ``off`` is refused when a kit shim is already active.
"""
from __future__ import annotations

from opt_core.oom import is_oom                      # an out-of-memory error is re-raised before any reroute below (the core's one classifier)
import importlib.util
import json
import os
import sys
import threading
from typing import Optional

from opt_core import gates as _gates
from opt_core import home as _home
from opt_core import instances as _instances

from . import ENV_MODE, ActivationError, __version__, modes, registry
from . import settings as _settings
from . import alloc as _alloc
from . import bondmask as _bondmask
from . import chunklift as _chunklift
from . import lnstream as _lnstream
from . import lncore as _lncore
from . import tmpldedup as _tmpldedup
from . import keeppool as _keeppool
from . import stepgraph as _stepgraph
from . import precision as _precision
from . import schedhost as _schedhost
from . import zprephoist as _zprephoist
from . import structoksync as _structoksync
from . import writer_overlap as _writer_overlap
from . import prefetch as _itemcensus
from . import ablate as _ablate
from . import stockknob as _stockknob
from . import tp as _tp
from . import big as _big
from . import smalln as _smalln
from . import report as _report

ENV_TREE = "MODEL_OPT"                           # opendde/ directory (configs/<gpu>.env exports it; the package derives it otherwise)
ROUTE = "cli"                                    # the activation route word of the report and the ACTIVE line: the `pred` process (the package CLI, or the autoload finder in upstream's own CLI)
_LOCK = threading.RLock()
_REPORT: Optional[dict] = None
_PREDICTED = False                                                  # a prediction ran in this process (refresh(predicted=True) was called): later bare refresh() calls -- the exit
                                                                    # tally's LEVER rows -- replay the same ran-or-refuse bucketing the PRED line printed (inert rows read state=off)
_ACTIVATING = False
_APPLIED: dict = {}                              # what apply() did (exports, unsets, sys_path entries, shim) — for the manifest


# ---------------------------------------------------------------------------------------------------------------- locations
def tree_root() -> str:
    """The opendde/ directory: $MODEL_OPT, else the checkout this package lives in (opt/opendde_opt/../..)."""
    return _home.tree_home(__file__, env_tree=ENV_TREE, levels=2)


def pyproject() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pyproject.toml")


def pins(tree: str | None = None) -> dict:
    with open(os.path.join(tree or tree_root(), "stock", "PINS.json")) as fh:
        return json.load(fh)


def kit_present(res: modes.Resolution) -> list[str]:
    """Missing kit paths for a resolution (empty = all present)."""
    missing = []
    for d in res.sys_path:
        if not os.path.isdir(d):
            missing.append(d)
    if res.shim and not os.path.isfile(res.shim):
        missing.append(res.shim)
    return missing


# ---------------------------------------------------------------------------------------------------------------- facts of this box
def opendde_version() -> str | None:
    return _gates.dist_version("opendde")


def gpu_info() -> dict:
    """GPU facts without importing torch when it is not loaded: nvidia-smi name/memory/compute capability, torch's capability when
    torch is imported. The package gates nothing on them (the kits carry their own gates: ARM-T declines below sm80, odde_arm_t/__init__.py:720-723)."""
    info = {"name": None, "sm": None, "memory_mib": None, "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        import subprocess
        for fields in ("name,memory.total,compute_cap", "name,memory.total"):     # compute_cap needs a driver that knows the field
            out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            if out.returncode == 0 and out.stdout.strip():
                cols = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
                info["name"], info["memory_mib"] = cols[0], int(float(cols[1]))
                if len(cols) > 2 and cols[2].replace(".", "").isdigit():
                    info["sm"] = cols[2].replace(".", "")
                break
    except Exception:  # noqa: BLE001
        pass
    t = sys.modules.get("torch")
    if t is not None:
        try:
            if t.cuda.is_available():
                cc = t.cuda.get_device_capability(0)
                info["sm"] = f"{cc[0]}{cc[1]}"
                info["name"] = info["name"] or t.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            pass
    return info


def _core_pin_gate() -> str | None:
    """The first gate: the importable opt_core is the one the kit pins (opt/pyproject.toml [tool.opt_core]) — the package's copy of the
    core's template gate (``_core_gate.gate``, the one producer of the 'pinned X, installed Y' fact); its refusal line is the reason."""
    import io
    from ._core_gate import CoreGateRefused, gate
    try:
        gate(pyproject(), "opendde-opt", stream=io.StringIO())
    except CoreGateRefused as e:
        return e.line.split("NOT ACTIVE: ", 1)[-1].strip()
    return None


def _version_gate(tree: str) -> str | None:
    want = pins(tree)["upstream"]["version"]
    have = opendde_version()
    if have is None:
        return f"opendde is not installed (want {want}: stock/PINS.json)"
    if have != want:
        return f"opendde {have} is installed; the tree is pinned to {want} (stock/PINS.json) — version gate"
    return None


STACK_PACKAGES = ("torch", "cuequivariance", "cuequivariance-torch", "cuequivariance-ops-cu12", "cuequivariance-ops-torch-cu12")   # the stack assertion's set
STACK_NOTE = ("NOTE {sm} — the torch / cuEquivariance stack differs from stock/PINS.json (the STACK line says MISMATCH): an untested library version, "
              "named; every lever of the mode engages as on the pinned stack")


def stack_pins(tree: str) -> dict:
    """{package: exact version} the box must carry: torch from stock/PINS.json "pins", the cuEquivariance family from its "gpu_extra" —
    the one place the expected stack is written."""
    p = pins(tree)["pins"]
    want = {"torch": p["torch"]}
    want.update({k: v for k, v in (p.get("gpu_extra") or {}).items() if k in STACK_PACKAGES})
    return want


def stack_mismatch(tree: str) -> str | None:
    """The box-start STACK ASSERTION: every STACK_PACKAGES distribution is installed at exactly the pinned version (distribution metadata,
    nothing imported; a local tag such as ``2.7.1+cu126`` matches ``2.7.1``). The first mismatch reads ``stack_mismatch:<pkg>=<got>!=<want>``
    (``<got>`` = ``absent`` when not installed); None when the stack is the pinned one. Named, never a refusal (an untested library version
    is not a reason to refuse or to drop a lever): the activation report carries it (``stack_mismatch``, a note, the ``stack_mismatch=`` token of
    the ACTIVE / NOT ACTIVE line) and the STACK line says MISMATCH; every lever of the mode engages as on the pinned stack."""
    import importlib.metadata as md
    for pkg, want in stack_pins(tree).items():
        try:
            got = md.version(pkg)
        except md.PackageNotFoundError:
            got = None
        if got is None or got.split("+", 1)[0] != str(want):
            return f"stack_mismatch:{pkg}={got or 'absent'}!={want}"
    return None


def stack_versions(tree: str) -> dict:
    """{package: installed version | None} for STACK_PACKAGES + opendde (distribution metadata, nothing imported)."""
    out = {pkg: _gates.dist_version(pkg) for pkg in stack_pins(tree)}
    out["opendde"] = opendde_version()
    return out


def stack_line(tree: str) -> str:
    """The one greppable line of the box-start stack assertion: ``STACK OK torch=2.7.1+cu126 cuequivariance=0.10.0 … opendde=1.1.1 (pins:
    stock/PINS.json)`` when the stack is the pin's, else ``STACK MISMATCH stack_mismatch:<pkg>=<got>!=<want> …`` with the same fields."""
    have = stack_versions(tree)
    fields = " ".join(f"{k}={v or 'absent'}" for k, v in have.items())
    mm = stack_mismatch(tree)
    return (f"STACK OK {fields} (pins: stock/PINS.json)" if mm is None else f"STACK MISMATCH {mm} {fields} (pins: stock/PINS.json)")


CALLER_WINS = ("ODDE_TP_STRUCT_PAIR_DTYPE", "ROWPAIR_PARK_ZINIT", "ROWPAIR_PARK_ZRES", "ODDE_TP_STRUCT_TRIMUL_RB", "ROWPAIR_NCCL_TIMEOUT_S",
               "ODDE_TRIATTN", "ODDE_TRIATTN_CONF", "ODDE_TRIMUL", "ODDE_TRANSITION")   # the provider words: a caller may pin one provider row by name (ODDE_TRIATTN=<row>) over the line's tier word — named on the ACTIVE line          # line exports a caller may override by presetting them (the structural pair dtype =fp32, z_init on the card =0: comparison switches, named on the NOTE line)


def pending_rebase_reason(res: modes.Resolution, tree: str | None = None, version: str | None = None) -> str | None:
    """The PENDING-REBASE gate: a line whose levers are not all tested on the pin (registry.TESTED_ON, the one source of truth)
    refuses by name — ``pending_rebase mode=<m> line=<L> pin=opendde-<v> levers_untested=<a,b,…>`` (exit 3 on every route). ``version``
    defaults to the installed opendde's, else the tree's pin (an installed opendde other than the pin is the version gate's refusal, not
    this one's); None when the line may activate."""
    from . import registry
    if res.line is None:
        return None
    v = version if version is not None else (opendde_version() or pins(tree or tree_root())["upstream"]["version"])
    unc = registry.untested(res.levers, v)
    if unc:
        return f"pending_rebase mode={res.mode or '-'} line={res.line.name} pin=opendde-{v} levers_untested={','.join(unc)}"
    hold = registry.LINES_ON_HOLD.get(res.line.name) if v == registry.PIN else None
    if hold:                                                                       # a line held as a family on the pin (registry.LINES_ON_HOLD): refused by name, its levers tested or not
        return f"pending_rebase mode={res.mode or '-'} line={res.line.name} pin=opendde-{v} line_on_hold={hold}"
    return None


# ---------------------------------------------------------------------------------------------------------------- late activation
def shim_active() -> str | None:
    """Name the kit shim already active in this process, if any (the fact that refuses `off` and a second activation)."""
    m = sys.modules.get("odde_served_levers")
    if m is not None:
        st = getattr(m, "STATE", {})
        if st.get("wrapped") or st.get("active") or st.get("installs"):
            return f"odde_served_levers v{st.get('version')} (wrapped={st.get('wrapped')}, installs={len(st.get('installs') or [])})"
        if any(type(f).__name__ == "_Hook" and getattr(f, "__module__", "") == "odde_served_levers" for f in sys.meta_path):
            return "odde_served_levers (hook armed on sys.meta_path)"
    m = sys.modules.get("odde_arm_t")
    if m is not None and getattr(m, "COUNTS", {}).get("installed"):
        return f"odde_arm_t arm {m.COUNTS.get('arm')} installed"
    m = sys.modules.get("odde_addon")
    if m is not None and getattr(m, "_ACTIVE", {}):
        return f"odde_addon levers {sorted(m._ACTIVE)} installed"
    m = sys.modules.get("fpf_engines")
    if m is not None and getattr(m, "_ORIG", {}):
        return f"fpf_engines ops enabled {sorted(op for _e, op in m._ORIG)}"
    return None


def runner_instances() -> int:
    ri = sys.modules.get("runner.inference")
    if ri is None or getattr(ri, "InferenceRunner", None) is None:
        return 0
    return _instances.instance_check("runner.inference", "InferenceRunner")["n"]


def late_activation_reason(res: modes.Resolution) -> str | None:
    sa = shim_active()
    if sa:
        return f"late activation refused: a kit shim is already active in this process ({sa})"
    n = runner_instances()
    if n:
        return f"late activation refused: {n} runner.inference.InferenceRunner instance(s) already exist; the hook installs on the runner get_default_runner returns"
    return None


# ---------------------------------------------------------------------------------------------------------------- apply
def _base(mode_or_line: str) -> dict:
    is_mode = modes.is_mode_name(mode_or_line)                                   # an exact-case kit line name (`BIG`) wins over the case-insensitive mode name (`big`)
    return {"active": False, "mode": mode_or_line.lower() if is_mode else None, "line_name": None if is_mode else mode_or_line,
            "route": ROUTE, "opendde_version": opendde_version(), "package_version": __version__, "gpu": gpu_info(),
            "levers_applied": [], "levers_planned": [], "levers_fallback": [], "partial": False, "notes": [],
            "levers_refused_on_pin": dict(registry.REFUSED_ON_PIN)}                # levers the pin refuses for cause (no line carries them): named on every ACTIVE / PRED line




_CORE_KERNEL_GATES: dict = {}                                        # opt_core.kernels name -> Gate (route_check after route): the activation report's `core_kernels`


def _route_core_kernel(name: str) -> None:
    """Serve the top-level kernel name from the core's copy (``opt_core.kernels.route``). ``route_check``'s verdict is
    recorded for the activation report (``core_kernels``) but never gates activation: trust in the installed core is
    the version pin (``gates.imported_core()``, checked earlier at ``_core_gate``), not a kernel-level byte compare
    against the core's own sums file."""
    from opt_core import kernels as _kernels
    _kernels.route(name)
    _CORE_KERNEL_GATES[name] = _kernels.route_check(name)


def _apply(res: modes.Resolution, jobs=None) -> dict:
    """Export the line's switches, put its directories on sys.path, execute the hook's shim. Returns what was done."""
    done = {"exports": {}, "unset": [], "sys_path": [], "shim": res.shim}
    if res.line is not None and res.line.allocator:
        done["alloc"] = _big.apply(res, jobs=jobs)  # a big line's fixed allocator policy through the core: refused by name BEFORE any switch is touched
    if res.line is not None and res.line.fpf:
        _route_core_kernel("fpf_trimul")                           # the core's TM-K3 TriMul package under its top-level name (opt_core.kernels.route): the provider's tmk3 rows and any importer share ONE module/compiled-kernel set; the kit carries no copy
    if res.line is not None and "drop_bond_mask" in res.line.levers:
        _bondmask.install()                                        # the tree's own lever: wraps the stock featurizer at its import
    if res.line is not None and "chunk_lift" in res.line.levers:
        _chunklift.install()                                       # the tree's own lever: wraps the model class's chunk resolution at its import
    if res.line is not None and "lnstream" in res.line.levers:
        _lnstream.install()                                        # the tree's own lever: wraps upstream's fused-LayerNorm loader at its import (current-stream build)
    if res.line is not None and "ln_core" in res.line.levers:
        _lncore.install(res.exports)                               # the core LayerNorm provider by the line's tier word (ODDE_LN): wraps FusedLayerNorm.forward at the module's import
    if res.line is not None and "tmpl_dedup" in res.line.levers:
        _tmpldedup.install()                                       # the tree's own lever: wraps TemplateEmbedder.forward at the pairformer module's import
    if res.line is not None and "keep_pool" in res.line.levers:
        _keeppool.install()                                        # the tree's own lever: the keep-pool policy on torch.cuda.empty_cache
    if res.line is not None and "structok_sync" in res.line.levers:
        _structoksync.install()                                    # the tree's own lever: the structural-token expander's role-pair projections (one host read per call)
    if res.line is not None and "zprep_hoist" in res.line.levers:
        _zprephoist.install()                                      # the tree's own lever: the diffusion module's permute_final_dims name (per-step pair copy memo)
    if res.line is not None and "sched_host" in res.line.levers:
        _schedhost.install()                                       # the tree's own lever: the noise scheduler + the sampler's augmentation name (host round-trips out)
    if res.line is not None and "prefetch" in res.line.levers:
        _itemcensus.install_prefetch()                             # the tree's own lever: upstream's DataLoader with one worker — the next item featurised while this one runs
    if res.line is not None and "json_oneshot" in res.line.levers:
        _writer_overlap.install_json()                             # the tree's own lever: upstream's save_json encodes once with json.dumps and writes once (identical bytes)
    if res.line is not None and "writer_overlap" in res.line.levers:
        _writer_overlap.install()                                  # the tree's own lever: upstream's result dump on a background writer thread (bound at runner.inference's import)
    if res.line is not None:
        _itemcensus.install()                                      # the item-boundary census (ITEM lines: featurise / prepare / predict / dump / cleanup seconds per item; timing only)
    if res.line is not None and "stepgraph" in res.line.levers:
        _stepgraph.install()                                       # the tree's own lever: wraps the sampler name the model resolves at its import (one denoiser-step graph per call)
    if res.line is not None and _precision.LEVER in res.line.levers:
        _precision.install()                                       # the tree's own sampler-autocast word: wraps upstream's inference precision policy at the runner module's import
    if res.line is not None and _tp.LEVER in res.line.levers:
        _tp.install()                                              # the row-sharded pair stack (opt_core.mem.rowpair) in a rank process; nothing at n_gpu=1
    for k in res.unset:
        if k in os.environ:
            os.environ.pop(k)
            done["unset"].append(k)
    for k, v in res.exports.items():
        had = os.environ.get(k)
        if k in CALLER_WINS and had is not None and had != v:                       # the caller's value of this switch wins over the line's default: named, never silent
            done.setdefault("kept", {})[k] = had
            print(f"[opendde-opt] NOTE {k}={had!r} in the environment kept over the line's {v!r} (line {res.line.name})", file=sys.stderr, flush=True)
            done["exports"][k] = had
            continue
        if k == "ODDE_OFFLOAD" and had is not None and had != v:              # a caller's offload stages are the LINE's declaration (under --n_gpu P>1: struct only —
            done.setdefault("overridden", {})[k] = had                       # the trunk / conf pair stacks are row-sharded, not host-streamed): named, never silent
            print(f"[opendde-opt] NOTE {k}={had!r} in the environment overridden -> {v!r} (line {res.line.name}: the offload unit's stages are the line's "
                  f"declaration{'; under n_gpu>1 the trunk and confidence pair tracks are row-sharded (rowpair_tp), not host-streamed' if res.line.name == modes.BIG_TP_LINE else ''})",
                  file=sys.stderr, flush=True)
        os.environ[k] = v
        done["exports"][k] = v
    if res.line is not None and not res.line.allocator and "alloc_auto" in res.line.levers:
        done["alloc"] = _big.apply(res, jobs=jobs)  # alloc_auto's token-aware export AFTER the line's own unset list (which names the allocator variable: the default
        if (done["alloc"] or {}).get("decision") == "export":       # allocator is the line's declaration unless this lever decides otherwise for the call)
            done["exports"][modes.ALLOCATOR] = os.environ.get(modes.ALLOCATOR, "")
    for d in reversed(res.sys_path):
        if d in sys.path:
            sys.path.remove(d)
        sys.path.insert(0, d)
    done["sys_path"] = list(res.sys_path)
    os.environ["PYTHONPATH"] = os.pathsep.join(res.sys_path + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and p not in res.sys_path])
    spec = importlib.util.spec_from_file_location(f"sitecustomize_opendde_opt_{os.path.basename(res.line.hook or res.line.name)}", res.shim)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if res.line.fpf:                                                              # S1: the kit's own in-process enable at the exported contract
        import fpf_engines                                                        # the line's SRC directory is on sys.path now
        done["fpf"] = fpf_engines.enable_from_env()
    return done


def _armed(res: modes.Resolution) -> tuple[bool, str]:
    """Did the shim arm its hook? (the served-levers hook: STATE.wrapped or a finder on sys.meta_path; an FPF line: its ops patched.)"""
    if res.line.hook == modes.HOOK:
        m = sys.modules.get("odde_served_levers")
        if m is None:
            return False, "odde_served_levers was not imported by the shim"
        st = getattr(m, "STATE", {})
        if st.get("wrapped"):
            why = f"get_default_runner wrapped (v{st.get('version')})"
        elif any(getattr(f, "__module__", "") == "odde_served_levers" for f in sys.meta_path):
            why = f"hook armed on sys.meta_path (v{st.get('version')}); wraps runner.batch_inference at its import"
        else:
            return False, "the shim ran but neither wrapped runner.batch_inference nor armed a finder"
        if res.line.fpf:
            fe = sys.modules.get("fpf_engines")
            ops = sorted(op for _e, op in getattr(fe, "_ORIG", {})) if fe is not None else []
            want = sorted(x for x in os.environ.get("FPF_OPS", "").split(",") if x)
            if ops != want:
                return False, f"fpf_engines enabled {ops}, the line wants {want}"
            why += f"; fpf_engines {res.line.fpf} ops {ops}"
        return True, why
    return True, "no hook"


def _finish(rep: dict, strict: bool) -> dict:
    global _REPORT, _PREDICTED
    _REPORT = rep
    _PREDICTED = False                                              # a fresh activation: no prediction has run under this report yet
    _report.log_activation(rep)
    rep["logged"] = True
    if not rep.get("dry_run") and (rep.get("mode") == "off" or not rep.get("active")):
        _report.emit_lever_lines(rep)                                  # `off` and a refusal print the per-lever lines now (no exit tally runs); an applied line prints them at exit
    if strict and not rep.get("active") and not rep.get("dry_run"):
        raise ActivationError(rep.get("reason", "not active"))
    return dict(rep)


def activate(mode_or_line: str, strict: bool = False, trigger: Optional[str] = None, dry_run: bool = False,
             tree: str | None = None, overrides: dict | None = None,
             jobs=None, n_gpu: int = 1) -> dict:
    """Apply a mode (``off``/``exact``/``fast``/``big``) or a kit line by name, once per process (idempotent). Returns the activation report;
    ``strict`` raises ActivationError when not active; ``dry_run`` resolves, gates and reports without touching the environment,
    ``sys.path`` or any kit module.
    ``overrides`` = switches exported AFTER the line's (the deterministic recipe's, det.py): recorded in the report as
    ``applied.overrides`` and in ``exports`` as the effective values.
    ``n_gpu`` = the accepted ``--n_gpu`` of the call (tp.check ran first): recorded as ``n_gpu`` and printed on the ACTIVE / EXIT lines in
    the core's token text (``n_gpu=P sharding=rowpair`` / ``n_gpu=1 sharding=none``)."""
    global _ACTIVATING
    key = (mode_or_line or "").strip()
    if not key:
        raise ValueError("a mode or line name is required")
    tree = os.path.abspath(tree or tree_root())
    with _LOCK:
        base = _base(key)
        base["tree"] = tree
        base["n_gpu"] = int(n_gpu or 1)
        if trigger:
            base["trigger"] = trigger
        if dry_run:
            base["dry_run"] = True
        # resolve (a selection refused by name or an unknown name is a named refusal, never an exception on the env route)
        try:
            res = modes.resolve(key, tree, os.environ)
        except modes.OpenModeError as e:
            base["reason"] = str(e)
            return _finish(base, strict)
        except (ValueError, KeyError) as e:
            base["reason"] = str(e)
            return _finish(base, strict)
        base["line"] = modes.describe_line(res)
        base["small_input_floor"] = _smalln.planned()                              # the small-input floor's decision for this call (None: no query read / not a floor line)
        base["levers_gated_off"] = {**_smalln.gated_off(res.line), **_chunklift.gated_off(res.line), **_stepgraph.gated_off(res.line), **_ablate.gated_off(res.line),
                                    **_stockknob.gated_off(res.line)}       # + the kit levers a stated --triatt_kernel / --trimul_kernel put aside by name (aside:stock_knob:<flag>=<value>)
        base["stock_knobs"] = _stockknob.word()                                    # `<flag>=<value>[,…]` when upstream's triangle-kernel flags are stated other than auto (ACTIVE / PRED token), else None   # {lever: below_gate|above_gate:<tokens>/<gate>} — the levers the floor / the
        base["chunk_lift_ceiling"] = _chunklift.planned()                          # chunk lever's ceiling composed out (their LEVER rows' reason); the ceiling decision for this call
        base["stepgraph_gate"] = _stepgraph.planned()                                # the sampler step-graph's size-gate decision for this call (None: no query read / not on the line)
        base["ablated"] = list(_ablate.dropped())                                  # MODEL_OPT_LEVERS_OFF: the levers left out of this call's line by name (ACTIVE / PRED `ablated=`)
        base["notes"].extend(res.notes)
        base["notes"].extend(_settings.stock_env_notes())                          # a caller's own LAYERNORM_TYPE: runs as requested, named (never a refusal)
        lw = _settings.upstream_layernorm_word()
        if lw and not os.environ.get("LAYERNORM_TYPE"):                          # upstream imported its model before this activation (env route) on its default LayerNorm
            base["notes"].append(f"NOTE upstream's LayerNorm is already {lw!r} in this process (its model was imported before the kit activated): running as imported; "
                                 f"the stock base of this tree is LAYERNORM_TYPE=fast_layernorm — export it before starting the stock CLI under {ENV_MODE}")
        need = _settings.stock_env()                                               # the stock base's environment (settings.STOCK_ENV), every route and mode
        if need and not dry_run:
            os.environ.update(need)
        base["stock_env"] = {k: os.environ.get(k, v) if not dry_run else (os.environ.get(k) or v) for k, v in _settings.STOCK_ENV.items()}
        if res.line is None:                                                       # off
            sa = shim_active()
            if sa:
                base["reason"] = f"mode off refused: a kit shim is active in this process ({sa}); stock cannot run here"
                return _finish(base, strict)
            sm = stack_mismatch(tree)                                              # a torch / cuEquivariance stack other than the pin's: named (report + note + the line's token), never a refusal
            if sm:
                base["stack_mismatch"] = sm
                base["notes"].append(STACK_NOTE.format(sm=sm))
            base["active"] = False
            base["reason"] = "mode off: stock opendde (no switch exported, no kit directory on the path)"
            return _finish(base, strict)
        if _REPORT is not None and not dry_run and (_REPORT.get("active") or _REPORT.get("apply_failed")):
            if _REPORT.get("line") == base["line"]:
                return dict(_REPORT)
            r = dict(_REPORT)
            r["reason"] = f"already active as {_REPORT.get('mode') or _REPORT.get('line_name')}; a line is applied once per process (restart to change it)"
            if strict:
                raise ActivationError(r["reason"])
            return r
        if _ACTIVATING:
            return {"active": False, "mode": base["mode"], "reason": "activation in progress (re-entrant call ignored)"}
        # gates: the core pin first (opt/pyproject.toml [tool.opt_core] vs the imported opt_core), then the kit files, the upstream version
        cg = _core_pin_gate()
        if cg:
            base["reason"] = cg
            return _finish(base, strict)
        missing = kit_present(res)
        if missing:
            base["reason"] = f"kit files missing under {tree}/opt: {missing[:3]}{'…' if len(missing) > 3 else ''}"
            return _finish(base, strict)
        vg = _version_gate(tree)
        if vg:
            base["reason"] = vg
            return _finish(base, strict)
        sm = stack_mismatch(tree)                                                  # a torch / cuEquivariance stack other than the pin's: named on the line; the mode's levers all engage
        if sm:
            base["stack_mismatch"] = sm
            base["notes"].append(STACK_NOTE.format(sm=sm))
        pr = pending_rebase_reason(res, tree)                                      # the pin's testing: untested levers (or a held line) refuse the line by name
        if pr:
            base["reason"] = pr
            base["levers_untested"] = pr.rsplit("levers_untested=", 1)[-1].split(",") if "levers_untested=" in pr else []
            return _finish(base, strict)
        late = late_activation_reason(res)
        if late:
            base["reason"] = late
            return _finish(base, strict)
        base["levers_planned"] = list(res.levers)
        base["exports"], base["unset"], base["pythonpath"] = {**res.exports, **(overrides or {})}, list(res.unset), res.pythonpath()
        base["overrides"] = dict(overrides or {})
        base["tier"], base["hook"] = res.line.tier, res.line.hook
        if dry_run:
            base["reason"] = "dry run: resolved and gated; nothing applied"
            return _finish(base, strict)
        _ACTIVATING = True
        try:
            _disarm_autoload()
            done = _apply(res, jobs=jobs)
            for k, v in (overrides or {}).items():                                # the recipe's switches win over the line's
                os.environ[k] = v
                done["exports"][k] = v
            done["overrides"] = dict(overrides or {})
            _APPLIED.update(done)
            base["applied"] = done
            ok, why = _armed(res)
            base["notes"].append(why)
            if not ok:
                base["apply_failed"] = True
                base["reason"] = f"the kit shim {res.shim} did not arm its hook: {why}"
                return _finish(base, strict)
            base["active"] = True
            base["levers_applied"] = list(res.levers)                               # armed; the kit's own install records refine this (refresh())
            _report.register_exit_tally()
            return _finish(base, strict)
        except Exception as e:  # noqa: BLE001
            if is_oom(e): raise
            base["apply_failed"] = True
            base["reason"] = f"the kit shim raised: {e!r}"
            return _finish(base, strict)
        finally:
            _ACTIVATING = False


def _disarm_autoload() -> None:
    from . import _autoload
    _autoload.disarm()


def status() -> dict:
    if _REPORT is None:
        return {"active": False, "reason": "opendde_opt.enable() has not been called in this process"}
    return dict(_REPORT)


ARM_LEVERS = {"z": "arm_z", "u": "arm_u"}                          # the hook's arm letter -> registry lever
HOOK_RECORD_LEVERS = {                                              # other keys of the hook's install record -> registry lever
    "kit.cueq_cache": "cueq_tuned_cache", "accel_v2.dit_attn(TIER-2)": "dit_attn_bf16",
    "accel_v2.dit_attn": "dit_attn_bf16",                           # the key of the same lever in the hook's STATE["errors"]
}


def levers_from_hook_records(records, addon_levers, planned) -> tuple[set, set]:
    """(installed, fallback) registry levers read from the served-levers hook's install records (`rec["levers"]`,
    forward/fast_inference/levers/ACCEL/odde_served_levers.py:44-124) — the one reading of that record shape (refresh(), the in-process
    STATE). A record without an error is not an install: ARM-T declines below
    sm80 / on CPU without raising (forward/fast_inference/levers/ARMT/odde_arm_t/__init__.py:720-723, `NOT installing (stock path)`,
    COUNTS['arm'] stays None) and the hook records `{"arm": None, ...}` — that is the line's planned arm lever falling back to stock."""
    planned_arm = next((x for x in planned if x.startswith("arm_")), None)
    installed, fallback = set(), set()
    for lv in records:
        d = lv.get("ditfast")
        if isinstance(d, dict):
            (fallback if "error" in d else installed).update(addon_levers)
        a = lv.get("odde_arm_t")
        if isinstance(a, dict):
            arm = str(a.get("arm") or a.get("already_installed") or "").lower()
            if "error" in a or not arm:
                fallback.add(planned_arm or "arm_z")
            else:
                installed.add(ARM_LEVERS.get(arm, f"arm_{arm}"))
        for key, name in HOOK_RECORD_LEVERS.items():
            v = lv.get(key)
            if isinstance(v, dict) and "error" in v:
                fallback.add(name)
            elif v:
                installed.add(name)
    return installed, fallback


def refresh(predicted: bool = False, facts: dict | None = None) -> dict:
    """Fold the kits' own install records into the report after a run: which levers installed, which fell back (served-levers STATE).
    ``predicted`` = at least one prediction ran in this process: then every applied lever must prove it executed (ran.py — a counter at zero
    is ``lever_never_ran:<lever>`` in ``levers_fallback``, PARTIAL) unless the call's ``facts`` ({"det", "n_gpu", "token_floors"}, known
    before the run) put it outside its domain (registry.ENGAGEMENT): then it is ``lever_inert_by_design:<lever>(<reason>)`` — named in
    ``levers_inert`` / ``levers_inert_reasons``, the run complete."""
    global _REPORT, _PREDICTED
    if predicted:
        _PREDICTED = True
    predicted = predicted or _PREDICTED                              # the exit tally after a prediction: the same buckets as its PRED line (registry.ENGAGEMENT replayed)
    if _REPORT is None or not _REPORT.get("active"):
        return status()
    rep = dict(_REPORT)
    m = sys.modules.get("odde_served_levers")
    if m is not None:
        st = getattr(m, "STATE", {})
        recs = [rec.get("levers") or {} for rec in st.get("installs") or []]
        addon = [x for x in os.environ.get("ODDE_ADDON_LEVERS", "").split(",") if x]
        planned = rep.get("levers_planned") or []
        installed, fell = levers_from_hook_records(recs, addon, planned)
        for k in st.get("errors") or {}:                            # an install that raised never appended its record: STATE["errors"] names it
            if k == "ditfast":
                fell.update(addon)
            elif k == "odde_arm_t":
                fell.add(next((x for x in planned if x.startswith("arm_")), "arm_z"))
            else:
                fell.add(HOOK_RECORD_LEVERS.get(k, k))
        if st.get("installs"):
            rep["levers_applied"] = sorted(installed | {"served_levers_hook"})
        rep["levers_fallback"] = sorted(fell)
        rep["partial"] = bool(fell)
        rep["kit_state"] = {k: v for k, v in st.items() if k != "installs"}
        rep["kit_installs"] = st.get("installs")
    planned_now = rep.get("levers_planned") or []
    if "arm_u23" in planned_now:                                   # arm_u23 install record: ARM U's sub-arms live in the arm's own COUNTS (the hook records only the arm letter):
        a = sys.modules.get("odde_arm_t")                          # on when the arm installed as U with the U2 TriMul / U3 configuration live, else named with the arm's words
        c = getattr(a, "COUNTS", None) if a is not None else None
        applied = set(rep.get("levers_applied") or []) - {"arm_u23"}
        if isinstance(c, dict) and str(c.get("arm") or "").lower() == "u" and (c.get("u2_trimul") or c.get("u3")) and "arm_u" in applied:
            applied.add("arm_u23")
        elif "arm_u" in applied:                                   # the arm serves but its sub-arms did not engage: a fallback by name, never a silent row
            why = "no odde_arm_t COUNTS" if not isinstance(c, dict) else f"u2_trimul={c.get('u2_trimul')} u3={c.get('u3')}"
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | {f"arm_u23:{why}"}); rep["partial"] = True
        rep["levers_applied"] = sorted(applied)                    # (arm_u itself absent = the arm fell back: levers_from_hook_records named it already)
    for tri in ("triattn_core", "triattn_exact"):                 # the provider binding's install record: the arm installed (its lever applied) with the unit's word live and
        if tri not in planned_now:                                 # the arm's attention site bound through it (odde_arm_t COUNTS att_mode core_<word>); else named, never silent
            continue
        a = sys.modules.get("odde_arm_t"); b = sys.modules.get("odde_triattn_bind")
        c = getattr(a, "COUNTS", None) if a is not None else None
        applied = set(rep.get("levers_applied") or []) - {tri}
        arm_on = bool({"arm_u", "arm_z"} & applied)
        mode = str((c or {}).get("att_mode") or "") if isinstance(c, dict) else ""
        if arm_on and b is not None and getattr(b, "COUNTS", {}).get("active") and mode.startswith("core_"):
            applied.add(tri)
        elif arm_on:                                               # the arm serves but the binding did not take the site: a fallback by name
            why = ("odde_triattn_bind not imported" if b is None else "word unset (ODDE_TRIATTN)" if not getattr(b, "COUNTS", {}).get("active")
                   else f"att_mode={mode or 'none'}")
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | {f"{tri}:{why}"}); rep["partial"] = True
        rep["levers_applied"] = sorted(applied)                    # (the arm absent = the arm fell back: named already; the binding rides it)
    if "triattn_conf" in planned_now:                              # the confidence-stack sub-lever: on with its parent (the word live, ODDE_TRIATTN_CONF absent); a parent that
        b = sys.modules.get("odde_triattn_bind")                    # fell back takes it along (named there); the word live but CONF=stock under a line that planned it = named
        applied = set(rep.get("levers_applied") or [])
        parent_on = bool({"triattn_core", "triattn_exact"} & applied)
        if parent_on and b is not None and getattr(b, "CONF", None) is None:
            applied.add("triattn_conf")
        else:
            applied.discard("triattn_conf")
            if parent_on:
                rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | {"triattn_conf:ODDE_TRIATTN_CONF=stock in the environment"}); rep["partial"] = True
        rep["levers_applied"] = sorted(applied)
    if "fpf_trimul_exact" in planned_now:                          # fpf_trimul_exact install record: the exact line's FPF adapter binds the two stock TriMul forwards in-process
        e = sys.modules.get("fpf_engines")                         # (fpf_engines._ORIG keyed (engine, op)); bound = on, imported-but-unbound or absent = named
        ops = sorted(op for _eng, op in (getattr(e, "_ORIG", None) or {})) if e is not None else []
        applied = set(rep.get("levers_applied") or []) - {"fpf_trimul_exact"}
        if any(op.startswith("trimul") for op in ops):
            applied.add("fpf_trimul_exact")
        else:
            why = "fpf_engines not imported" if e is None else f"no trimul op bound (ops={','.join(ops) or 'none'})"
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | {f"fpf_trimul_exact:{why}"}); rep["partial"] = True
        rep["levers_applied"] = sorted(applied)
    for tri, host_lever in (("trimul_core", "arm_u23"), ("trimul_exact", "fpf_trimul_exact")):   # the TriMul provider binding's install record: its host route applied
        if tri not in planned_now:                                 # (ARM U's U2 route | the exact line's FPF adapter) with the unit's word live and the route pointing at it
            continue                                               # (arm COUNTS trimul_bind == word | FPF_IMPL names the unit's callables); else a fallback BY NAME
        b = sys.modules.get("odde_trimul_bind")
        applied = set(rep.get("levers_applied") or []) - {tri}
        host_on = host_lever in applied
        live = b is not None and bool(getattr(b, "COUNTS", {}).get("active"))
        if tri == "trimul_core":
            a = sys.modules.get("odde_arm_t"); c = getattr(a, "COUNTS", None) if a is not None else None
            routed = isinstance(c, dict) and c.get("trimul_bind") == getattr(b, "WORD", None) and live
        else:
            routed = live and "odde_trimul_bind:" in os.environ.get("FPF_IMPL", "")
        if host_on and routed:
            applied.add(tri)
        elif host_on:
            why = ("odde_trimul_bind not imported" if b is None else "word unset (ODDE_TRIMUL)" if not live
                   else ("arm route not bound (trimul_bind)" if tri == "trimul_core" else "FPF_IMPL does not name odde_trimul_bind"))
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | {f"{tri}:{why}"}); rep["partial"] = True
        rep["levers_applied"] = sorted(applied)
    for trn, host_lever in (("transition_core", "arm_u"), ("transition_exact", "arm_z")):   # the transition provider binding's install record: its host arm applied with the
        if trn not in planned_now:                                 # unit's word live and the arm's bind having applied the adapter to >= 1 pair_transition module
            continue                                               # (odde_arm_t COUNTS transition_bind == word, fpf_transition.applied_modules); else a fallback BY NAME
        b = sys.modules.get("odde_transition_bind")
        applied = set(rep.get("levers_applied") or []) - {trn}
        host_on = host_lever in applied
        live = b is not None and bool(getattr(b, "COUNTS", {}).get("active"))
        a = sys.modules.get("odde_arm_t"); c = getattr(a, "COUNTS", None) if a is not None else None
        ft = (c or {}).get("fpf_transition") if isinstance(c, dict) else None
        pending = isinstance(c, dict) and "transition_bind" not in c          # the arm binds a model's modules at its first design (bind()); before that the word alone is the route
        routed = live and isinstance(c, dict) and (pending or (c.get("transition_bind") == getattr(b, "WORD", None) and isinstance(ft, dict) and int(ft.get("applied_modules") or 0) > 0))
        if host_on and routed:
            applied.add(trn)
        elif host_on:
            why = ("odde_transition_bind not imported" if b is None else "word unset (ODDE_TRANSITION)" if not live
                   else "adapter not applied (fpf_transition_odde: " + str((c or {}).get("errors", {}).get("fpf_transition_import", "no pair_transition module in scope"))[:80] + ")")
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | {f"{trn}:{why}"}); rep["partial"] = True
        rep["levers_applied"] = sorted(applied)
    xl_planned = [x for x in (rep.get("levers_planned") or []) if x.startswith("xl_")]
    if xl_planned:                                                 # the XL unit installs at the model module's import (its shim's hook); a
        m = sys.modules.get("odde_xl")                             # planned lever it did not install is a fallback by name, never silent
        installed = set(getattr(m, "_INSTALLED", ()) or ()) if m is not None else set()
        xl_fell = {x for x in xl_planned if x[len("xl_"):] not in installed}
        rep["levers_applied"] = sorted((set(rep.get("levers_applied") or []) - set(xl_planned)) | (set(xl_planned) - xl_fell))
        if xl_fell:
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | xl_fell)
            rep["partial"] = True
    sp_planned = [x for x in (rep.get("levers_planned") or []) if x in modes.SAMPLER_LEVERS]
    hm = sys.modules.get("odde_served_levers")
    hook_ran = hm is not None and bool(getattr(hm, "STATE", {}).get("installs"))
    if sp_planned and hook_ran:                                   # the SAMPLER unit installs from the ACCEL lever's routing site inside the hook's install
        sm = sys.modules.get("odde_sampler")                       # (odde_accel_v2.install_dit_attn -> odde_sampler.install, raising by name on any failure): once the
        sp_inst = set((getattr(sm, "REPORT", {}) or {}).get("installed") or ()) if sm is not None else set()   # hook has run, a planned lever the unit did not install is a fallback by name
        sp_aside = dict((getattr(sm, "REPORT", {}) or {}).get("aside") or {}) if sm is not None else {}        # unless the unit STEPPED ASIDE by name (no kernel for the running stack — the stock
        sp_aside = {x: w for x, w in sp_aside.items() if x in sp_planned and x not in sp_inst}                    # statement serves, exact by construction): inert, LEVER state=off reason=aside:<why>, run complete
        sp_fell = {x for x in sp_planned if x not in sp_inst and x not in sp_aside}
        rep["levers_applied"] = sorted((set(rep.get("levers_applied") or []) - set(sp_planned)) | (set(sp_planned) - sp_fell - set(sp_aside)))
        if sp_aside:
            rep["levers_inert"] = sorted(set(rep.get("levers_inert") or []) | set(sp_aside))
            rep.setdefault("levers_inert_reasons", {}).update({x: f"aside:{w}" for x, w in sp_aside.items()})
        if sp_fell:
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | sp_fell)
            rep["partial"] = True
    off_planned = [x for x in (rep.get("levers_planned") or []) if x in modes._OFFLOAD_LEVERS]
    if off_planned:                                                # the offload unit installs at interpreter start (its shim); its own shim swallows
        off_rep = _big.report(off_planned)                       # the memory mode's adapter reads its unit
        off_fell = set(off_rep["uninstalled"])                        # every install exception — here a planned lever it did not install is named
        rep["levers_applied"] = sorted((set(rep.get("levers_applied") or []) - set(off_planned)) | (set(off_planned) - off_fell))
        off_fell |= set(off_rep["runtime_fallbacks"])                # every runtime fallback the unit counted is a named event
        if off_fell:
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | off_fell)
            rep["partial"] = True
    if "drop_bond_mask" in (rep.get("levers_planned") or []):          # the tree's own lever: applied by its own state (not a hook record)
        bm_fell = _bondmask.fallbacks(rep["levers_planned"])           # (syncs the patch state) imported-but-unwrapped or the key absent is named; a process that never
        applied = set(rep.get("levers_applied") or []) - {"drop_bond_mask"}    # a process that featurized nothing has no event
        if _bondmask.STATS["installed"] and not bm_fell:
            applied.add("drop_bond_mask")
        elif not bm_fell:                                               # never armed in this process (the featurizer was not imported): no call could reach the site
            rep["levers_inert"] = sorted(set(rep.get("levers_inert") or []) | {"drop_bond_mask"})
            rep.setdefault("levers_inert_reasons", {})["drop_bond_mask"] = "the featurizer was not imported in this process: no call could reach the site"
        rep["levers_applied"] = sorted(applied)
        if bm_fell:
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | set(bm_fell))
            rep["partial"] = True
    if "stepgraph" in (rep.get("levers_planned") or []):               # the tree's own lever on the sampler: applied by its own patch state; a failure (a capture that failed
        sg_fell = _stepgraph.fallbacks(rep["levers_planned"])             # or was not bit-exact, a failed hold / probe) is a fallback BY NAME (PARTIAL); sampler calls outside
        sg_aside = _stepgraph.aside_reason()                              # its domain (guidance, another rank, the memory admission, ...) ran the stock sampler BY DESIGN:
        applied = set(rep.get("levers_applied") or []) - {"stepgraph"}       # when none engaged, the lever is inert by design (complete; LEVER state=off reason=aside:<word>)
        if (_stepgraph.STATS["installed"] or _stepgraph.STATS["armed"]) and not sg_fell:
            if sg_aside is None:
                applied.add("stepgraph")
            else:
                rep["levers_inert"] = sorted(set(rep.get("levers_inert") or []) | {"stepgraph"})
                rep.setdefault("levers_inert_reasons", {})["stepgraph"] = sg_aside
        rep["levers_applied"] = sorted(applied)
        if sg_fell:
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | set(sg_fell))
            rep["partial"] = True
    if "ln_core" in (rep.get("levers_planned") or []):                 # the LayerNorm provider binding: applied by its own patch state with the word live; a bound-but-never-served
        lc_fell = _lncore.fallbacks(rep["levers_planned"])                # pass whose every call was the extension's by a named rule is inert by design (registry.ENGAGEMENT);
        lc_aside = _lncore.aside_word()                                   # an unbound wrapper under a live word / served-row errors are fallbacks BY NAME (PARTIAL)
        applied = set(rep.get("levers_applied") or []) - {"ln_core"}
        if (_lncore.STATS["installed"] or _lncore.STATS["armed"]) and _lncore.STATS["word"] is not None and not lc_fell:
            if lc_aside is None:
                applied.add("ln_core")
            else:
                rep["levers_inert"] = sorted(set(rep.get("levers_inert") or []) | {"ln_core"})
                rep.setdefault("levers_inert_reasons", {})["ln_core"] = lc_aside
        elif not lc_fell:                                                 # the word not exported / upstream's module never imported in this process: no LayerNorm could reach the wrapper (named, inert)
            rep["levers_inert"] = sorted(set(rep.get("levers_inert") or []) | {"ln_core"})
            rep.setdefault("levers_inert_reasons", {})["ln_core"] = ("ODDE_LN not exported in this process: the wrapper was not installed" if _lncore.STATS["word"] is None
                                                                   else "upstream's layer_norm module was not imported in this process: no LayerNorm could reach the wrapper")
        rep["levers_applied"] = sorted(applied)
        if lc_fell:
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | set(lc_fell)); rep["partial"] = True
    if "structok_sync" in (rep.get("levers_planned") or []):           # the tree's own lever on the structural-token expander: applied by its own patch state
        ss_fell = _structoksync.fallbacks(rep["levers_planned"])
        applied = set(rep.get("levers_applied") or []) - {"structok_sync"}
        if _structoksync.STATS["installed"] and not ss_fell:
            applied.add("structok_sync")
        elif not ss_fell:                                               # the expander module was not imported in this process (the patch waits at its import)
            rep["levers_inert"] = sorted(set(rep.get("levers_inert") or []) | {"structok_sync"})
            rep.setdefault("levers_inert_reasons", {})["structok_sync"] = "the structural_tokens module was not imported in this process: no expander call could reach the sites"
        rep["levers_applied"] = sorted(applied)
        if ss_fell:
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | set(ss_fell))
            rep["partial"] = True
    if "zprep_hoist" in (rep.get("levers_planned") or []):             # the tree's own lever on the denoiser's pair preparation: applied by its own patch
        zp_fell = _zprephoist.fallbacks(rep["levers_planned"])
        applied = set(rep.get("levers_applied") or []) - {"zprep_hoist"}
        if _zprephoist.STATS["installed"] and not zp_fell:             # bound on the imported diffusion module: ran-or-refuse reads its counter
            applied.add("zprep_hoist")
        elif not zp_fell:                                               # never bound on an imported module in this process: no denoiser call could reach the site
            rep["levers_inert"] = sorted(set(rep.get("levers_inert") or []) | {"zprep_hoist"})
            rep.setdefault("levers_inert_reasons", {})["zprep_hoist"] = "the diffusion module was not imported in this process: no denoiser call could reach the pair preparation"
        rep["levers_applied"] = sorted(applied)
        if zp_fell:
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | set(zp_fell))
            rep["partial"] = True
    if "sched_host" in (rep.get("levers_planned") or []):              # the tree's own lever on the sampler loop's host round-trips: applied by its own patch state
        sh_fell = _schedhost.fallbacks(rep["levers_planned"])
        applied = set(rep.get("levers_applied") or []) - {"sched_host"}
        if _schedhost.STATS["installed"] and not sh_fell:              # bound on the imported generator module: ran-or-refuse reads its counter
            applied.add("sched_host")
        elif not sh_fell:                                               # the generator module was not imported in this process (the patch waits at its import): no
            rep["levers_inert"] = sorted(set(rep.get("levers_inert") or []) | {"sched_host"})   # sampler call could reach the scheduler / augmentation sites
            rep.setdefault("levers_inert_reasons", {})["sched_host"] = "the generator module was not imported in this process: no sampler call could reach the sites"
        rep["levers_applied"] = sorted(applied)
        if sh_fell:
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | set(sh_fell))
            rep["partial"] = True
    if _precision.LEVER in (rep.get("levers_planned") or []):         # the tree's own sampler-autocast word: applied by its own patch state; the runner module imported but its
        pw_fell = _precision.fallbacks(rep["levers_planned"])             # policy function unwrapped is named (PARTIAL); armed on a runner module never imported in this process =
        applied = set(rep.get("levers_applied") or []) - {_precision.LEVER}   # inert (no prediction could reach the policy site), like the other house patches
        if (_precision.STATS["installed"] or _precision.STATS["armed"]) and not pw_fell:
            if _precision.STATS["installed"]:
                applied.add(_precision.LEVER)
            else:
                rep["levers_inert"] = sorted(set(rep.get("levers_inert") or []) | {_precision.LEVER})
                rep.setdefault("levers_inert_reasons", {})[_precision.LEVER] = "runner.inference was not imported in this process: no prediction could reach upstream's precision policy"
        rep["levers_applied"] = sorted(applied)
        if pw_fell:
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | set(pw_fell))
            rep["partial"] = True
    if "lnstream" in (rep.get("levers_planned") or []):                # the tree's own lever on upstream's LayerNorm loader: applied by its own patch state and build state
        ls_fell = _lnstream.fallbacks(rep["levers_planned"])              # imported-but-unwrapped, or the current-stream build refused / failed (the legacy build served): named
        applied = set(rep.get("levers_applied") or []) - {"lnstream"}
        if _lnstream.STATS["installed"] and not ls_fell:
            applied.add("lnstream")
        elif not ls_fell:                                               # never armed on an imported module in this process: no LayerNorm could reach the loader
            rep["levers_inert"] = sorted(set(rep.get("levers_inert") or []) | {"lnstream"})
            rep.setdefault("levers_inert_reasons", {})["lnstream"] = "upstream's layer_norm module was not imported in this process: no LayerNorm could reach the loader"
        rep["levers_applied"] = sorted(applied)
        if ls_fell:
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | set(ls_fell))
            rep["partial"] = True
    if "tmpl_dedup" in (rep.get("levers_planned") or []):              # the tree's own lever on TemplateEmbedder: applied by its own patch state
        td_fell = _tmpldedup.fallbacks(rep["levers_planned"])
        applied = set(rep.get("levers_applied") or []) - {"tmpl_dedup"}
        if _tmpldedup.kit_stats()["installed"] and not td_fell and "tmpl_dedup" not in (rep.get("levers_inert") or []):
            applied.add("tmpl_dedup")                                   # (a report that already holds it inert by design — a --n_gpu P rank, registry.ENGAGEMENT — keeps it inert)
        elif not td_fell and "tmpl_dedup" not in (rep.get("levers_inert") or []):                                               # never armed on an imported module in this process: no call could reach the site
            rep["levers_inert"] = sorted(set(rep.get("levers_inert") or []) | {"tmpl_dedup"})
            rep.setdefault("levers_inert_reasons", {})["tmpl_dedup"] = "upstream's pairformer module was not imported in this process: no call could reach TemplateEmbedder"
        rep["levers_applied"] = sorted(applied)
        if td_fell:
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | set(td_fell))
            rep["partial"] = True
    if "prefetch" in (rep.get("levers_planned") or []):                # the tree's own lever on upstream's DataLoader: applied by its own bind state
        pf_fell = _itemcensus.fallbacks_prefetch(rep["levers_planned"])
        applied = set(rep.get("levers_applied") or []) - {"prefetch"}
        pst = _itemcensus.PSTATS
        if pst["installed"] and not pf_fell and not (pst["calls"] and not pst["engaged"]):
            applied.add("prefetch")
        elif not pf_fell:
            why = ("runner.inference was not imported in this process: no dataloader could be built" if not pst["installed"] else
                   "aside:" + ",".join(k if v == 1 else f"{k}x{v}" for k, v in sorted(pst["aside"].items())))   # LEVER state=off reason=aside:<rule>
            rep["levers_inert"] = sorted(set(rep.get("levers_inert") or []) | {"prefetch"})
            rep.setdefault("levers_inert_reasons", {})["prefetch"] = why
        rep["levers_applied"] = sorted(applied)
        if pf_fell:
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | set(pf_fell))
            rep["partial"] = True
    if "json_oneshot" in (rep.get("levers_planned") or []):            # the tree's own lever on upstream's JSON writer: applied by its own bind state
        jo_fell = _writer_overlap.fallbacks_json(rep["levers_planned"])
        applied = set(rep.get("levers_applied") or []) - {"json_oneshot"}
        if _writer_overlap.JSTATS["installed"] and not jo_fell:
            applied.add("json_oneshot")
        elif not jo_fell:
            rep["levers_inert"] = sorted(set(rep.get("levers_inert") or []) | {"json_oneshot"})
            rep.setdefault("levers_inert_reasons", {})["json_oneshot"] = "runner.dumper was not imported in this process: no document could reach the writer"
        rep["levers_applied"] = sorted(applied)
        if jo_fell:
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | set(jo_fell))
            rep["partial"] = True
    if "writer_overlap" in (rep.get("levers_planned") or []):          # the tree's own lever on upstream's result dump: applied by its own bind state
        wo_fell = _writer_overlap.fallbacks(rep["levers_planned"])
        applied = set(rep.get("levers_applied") or []) - {"writer_overlap"}
        if _writer_overlap.STATS["installed"] and not wo_fell:          # DataDumper.dump bound on the imported runner: ran-or-refuse reads its counter
            applied.add("writer_overlap")
        elif not wo_fell:                                               # runner.dumper was not imported in this process (the bind waits at runner.inference's import): no item could reach the dump
            rep["levers_inert"] = sorted(set(rep.get("levers_inert") or []) | {"writer_overlap"})
            rep.setdefault("levers_inert_reasons", {})["writer_overlap"] = "runner.dumper was not imported in this process: no item could reach the dump"
        rep["levers_applied"] = sorted(applied)
        if wo_fell:
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | set(wo_fell))
            rep["partial"] = True
    if "keep_pool" in (rep.get("levers_planned") or []):               # the tree's own lever on the allocator: applied by its own install state
        kp_fell = _keeppool.fallbacks(rep["levers_planned"])
        applied = set(rep.get("levers_applied") or []) - {"keep_pool"}
        if _keeppool.STATS["installed"] and not kp_fell:
            applied.add("keep_pool")
        rep["levers_applied"] = sorted(applied)
        if kp_fell:
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | set(kp_fell))
            rep["partial"] = True
    if "chunk_lift" in (rep.get("levers_planned") or []):              # the tree's own lever on the model class: applied by its own patch state (not a hook record)
        cl_fell = _chunklift.fallbacks(rep["levers_planned"])             # the model module imported but the class unwrapped is named; a process that never imported
        applied = set(rep.get("levers_applied") or []) - {"chunk_lift"}       # the model has no event (inert: no call could reach the site)
        if _chunklift.STATS["installed"] and not cl_fell:
            applied.add("chunk_lift")
        elif not cl_fell:
            rep["levers_inert"] = sorted(set(rep.get("levers_inert") or []) | {"chunk_lift"})
            rep.setdefault("levers_inert_reasons", {})["chunk_lift"] = "the model module was not imported in this process: no call could reach the site"
        rep["levers_applied"] = sorted(applied)
        if cl_fell:
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | set(cl_fell))
            rep["partial"] = True
    rep = _big.refresh(rep)                                                     # the memory mode's record (opt_core.mem): the manifest's activation.big block, one census unit
                                                                                  # per prediction — a lever that did not run is a fallback by name (PARTIAL) — the size-gate decision
    if "torch" in sys.modules and "numerics" not in rep:                # the process numerics signature (opt_core.precision.policy): matmul precision, TF32, cuDNN, deterministic, autocast
        try:
            from opt_core.precision import policy as _policy
            rep["numerics"] = {k: (v if isinstance(v, (str, int, float, bool)) or v is None else str(v)) for k, v in _policy.numerics_signature()}
        except Exception as e:  # noqa: BLE001 — a signature the reader cannot take is named, never omitted silently
            rep["numerics"] = {"error": repr(e)}
    if _CORE_KERNEL_GATES:
        rep["core_kernels"] = {n: {"ok": g.ok, "resolved": (g.details or {}).get("resolved"), "routed": (g.details or {}).get("routed")} for n, g in _CORE_KERNEL_GATES.items()}
    if "alloc_auto" in (rep.get("levers_planned") or []):            # the allocator lever: a refused export or a contradicted read-back is named; `default` is its envelope
        al_fell = _alloc.fallbacks(rep["levers_planned"])
        applied = set(rep.get("levers_applied") or []) - {"alloc_auto"}
        d = _alloc.STATE["decision"] or {}
        if d.get("decision") == "export" and not al_fell:
            applied.add("alloc_auto")
        elif d.get("decision") == "default":
            rep["levers_declared_off"] = sorted(set(rep.get("levers_declared_off") or []) | {"alloc_auto"})
            rep.setdefault("declared_off_reasons", {})["alloc_auto"] = f"default allocator: {d.get('reason')}"
            note = f"alloc_auto: default allocator — {d.get('reason')}"
            if note not in (rep.get("notes") or []):
                rep.setdefault("notes", []).append(note)
        elif not d and not al_fell:                                   # no decision was taken in this process (the writer runs at `pred` activation): named inert
            rep["levers_inert"] = sorted(set(rep.get("levers_inert") or []) | {"alloc_auto"})
            rep.setdefault("levers_inert_reasons", {})["alloc_auto"] = "no allocator decision was taken in this process"
        rep["levers_applied"] = sorted(applied)
        if al_fell:
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | set(al_fell))
            rep["partial"] = True
    if _tp.LEVER in (rep.get("levers_planned") or []):                 # the row-sharded pair stack: applied when a rank process sharded a stack; inert at n_gpu=1
        tp_fell = _tp.fallbacks(rep["levers_planned"])                 # (installs nothing) or when no pair stack was reached; a stack that ran unsharded is PARTIAL
        applied = set(rep.get("levers_applied") or []) - {_tp.LEVER} - set(modes.TP_KERNEL_LEVERS)
        if _tp.STATS["n_gpu"] > 1 and _tp.STATS["calls"] and not tp_fell:
            applied.add(_tp.LEVER)
            from . import tp_kernels as _tpk                                # the sharded stack's row-block kernel lever: applied when its word is bound in this rank
            applied.update(x for x in modes.TP_KERNEL_LEVERS if x in rep["levers_planned"] and _tpk.TRIATT["kernel"])
        else:
            rep["levers_inert"] = sorted(set(rep.get("levers_inert") or []) | {_tp.LEVER}) if not tp_fell else rep.get("levers_inert")
        rep["levers_applied"] = sorted(applied)
        for n in _tp.notes(rep["levers_planned"]):
            if n not in rep["notes"]:
                rep["notes"].append(n)
        if tp_fell:
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | {_tp.LEVER})
            rep["partial"] = True
            rep["notes"].extend(x for x in tp_fell if x not in rep["notes"])
    from . import ran as _ran                                       # ran-or-refuse (kit-wide): every applied lever's own counter, after the first prediction
    facts = dict(facts or rep.get("facts") or {})
    facts.setdefault("n_gpu", _tp.STATS["n_gpu"] if _tp.STATS.get("n_gpu") else 1)
    facts["levers"] = list(rep.get("levers_planned") or [])         # the line's composition is a fact too (an installer lever engages only with the levers it installs)
    rep["facts"] = facts                                            # the call's pre-run facts the engagement predicates read (det, n_gpu, token floors, levers)
    live = [x for x in (rep.get("levers_applied") or [])                  # the levers claimed applied: each proves it executed (a lever its own fold declared
            if x not in (rep.get("levers_inert") or []) and x not in (rep.get("levers_declared_off") or [])   # inert / off keeps that fold's named reason)
            and not any(f == x or f.startswith(x + ":") for f in (rep.get("levers_fallback") or []))]
    rep["lever_counts"] = _ran.census(sorted(set((rep.get("levers_planned") or []) + live)))
    rep["levers_uncounted"] = _ran.uncounted(live)
    inert_bd = _ran.inert_by_design(sorted(set((rep.get("levers_planned") or []) + live)), facts)   # {lever: lever_inert_by_design:<lever>(<reason>)}
    if predicted:
        bucketed = set(rep.get("levers_applied") or []) | set(rep.get("levers_inert") or []) | set(rep.get("levers_declared_off") or []) \
                   | {lv for lv in (rep.get("levers_planned") or []) if any(f == lv or f.startswith(lv + ":") or f == f"lever_never_ran:{lv}" for f in (rep.get("levers_fallback") or []))}
        idle = {lv for lv in live if lv in inert_bd and (rep["lever_counts"].get(lv) or 0) == 0}       # outside its domain on this call and it did not run: inert, named
        idle |= {lv for lv in (rep.get("levers_planned") or []) if lv in inert_bd and lv not in bucketed}   # a planned lever the install records dropped because it had nothing to install
        idle |= {lv for lv in (rep.get("levers_applied") or []) if lv in inert_bd and lv in (rep.get("levers_inert") or [])}   # a replay (the exit tally): the install folds above re-claim a lever an
        if idle:
            rep["levers_applied"] = sorted(set(rep.get("levers_applied") or []) - idle)
            rep["levers_inert"] = sorted(set(rep.get("levers_inert") or []) | idle)
            rep.setdefault("levers_inert_reasons", {}).update({lv: inert_bd[lv] for lv in idle})
        nr = _ran.never_ran(live, facts)                            # inside its domain and zero: the defect class (PARTIAL)
        if nr:
            dead = {x.split(":", 1)[1] for x in nr}
            rep["levers_applied"] = sorted(set(rep.get("levers_applied") or []) - dead)
            rep["levers_inert"] = sorted(set(rep.get("levers_inert") or []) - dead)
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | set(nr))
            rep["partial"] = True
    rep["kit_stats"] = _report.kit_stats()
    fe = (rep["kit_stats"] or {}).get("fpf_engines") or {}
    fpf_fell = {op: st["fallbacks"] for op, st in (fe.get("stats") or {}).items() if st.get("calls") and not st.get("kernel")}
    if fpf_fell:                                                   # every call of a planned FPF op took the stock path: counted by the kit, named here
        lever = next((x for x in (rep.get("levers_planned") or []) if x.startswith("fpf_trimul_")), "fpf_trimul")
        rep["fpf_fallbacks"] = fpf_fell
        if lever in inert_bd:                                      # below the engine's own small-N regime on every item: the stock path IS the design (named, complete)
            rep["levers_applied"] = sorted(set(rep.get("levers_applied") or []) - {lever})
            rep["levers_inert"] = sorted(set(rep.get("levers_inert") or []) | {lever})
            rep.setdefault("levers_inert_reasons", {})[lever] = inert_bd[lever]
        else:
            rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | {lever})
            rep["partial"] = True
    if predicted:                                                  # total accounting: every planned lever in exactly ONE bucket (ran.partition) — the kits' install
        for lv, b in _ran.partition(rep).items():                  # records list only what the hook installed, so a planned lever no fold placed is settled by its
            if b:                                                  # own evidence: counter > 0 = it ran (applied); 0 = lever_never_ran (PARTIAL); no readable counter
                continue                                           # = a named gap in levers_uncounted (never silence, never a guess)
            n = rep["lever_counts"].get(lv)
            if n is not None and n > 0:
                rep["levers_applied"] = sorted(set(rep.get("levers_applied") or []) | {lv})
            elif n == 0:
                rep["levers_fallback"] = sorted(set(rep.get("levers_fallback") or []) | {f"lever_never_ran:{lv}"})
                rep["partial"] = True
            else:
                rep["levers_applied"] = sorted(set(rep.get("levers_applied") or []) | {lv})
                rep.setdefault("levers_uncounted", {})[lv] = (rep.get("levers_uncounted") or {}).get(lv) or \
                    f"no install record and no readable counter in this process ({'its module is not loaded' if lv in _ran.COUNTERS else 'uncounted lever'}): a named gap"
        rep["lever_buckets"] = _ran.partition(rep)
    arm_t = (_report.kit_stats().get("arm_t") if "odde_arm_t" in sys.modules else None)   # ARM T/U's TriMul: calls that ran the stock TriMul for a kernel EXCEPTION are fallbacks BY
    for lv, why in _report.trimul_aside_items(arm_t).items():                            # NAME — the run is PARTIAL; the U2 admission gate's refusals (planes exceed free memory / probe
        rep.setdefault("levers_asides", {})[lv] = why                                   # unavailable) are named STEP-ASIDES: stock by rule for that call, counted on the arm's
    for item in _report.trimul_fallback_items(arm_t):                                    # LEVER line (trimul_admission=...), the run complete — recorded here under levers_asides
        kept = [f for f in (rep.get("levers_fallback") or []) if ":trimul_stock(" not in f]   # the census as it stands now (one item per process, counts current)
        rep["levers_fallback"] = sorted(set(kept) | {item})
        rep["partial"] = True
    _REPORT = rep
    return dict(rep)


def applied() -> dict:
    return dict(_APPLIED)
