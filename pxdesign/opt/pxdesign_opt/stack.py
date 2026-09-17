"""Activation: tree paths, pins, the GPU description, the mode's switches, and the lever application through the kit's own install().

`activate(mode)` resolves the mode in the table (modes.resolve), gates (the carried lever kit present on disk — `registry.list_kit_files`;
a visible GPU — `opt_core.gates.torch_gpu_probe`; the installed upstream is the pinned stock — `stock/check_pins.py`), exports
the kit's own switches for the mode (dropping any caller-set switch so the table stays the source of truth), and ARMS the application on
upstream's `pxdesign.runner.inference.InferenceRunner`: the kit's `pxd_xattempt.hoist.install(runner.model)` runs when `load_checkpoint()`
returns — a wrapper on that method (the model is built and in eval mode there,
`runner/inference.py:113`) — or eagerly when `apply_to(runner)` is called on a runner built elsewhere (the hoist applies). install() is the only place the hoist is
installed; its class-level patches and the model tags are the kit's own.

Engagement rule. A lever engages wherever its mechanism applies; uncertainty about the environment — a GPU class outside
`MEASURED_CC`, a torch build other than the pin, a target-GPU word that does not match the box, TF32 library overrides in the environment —
is NAMED on the activation line (``notes=…``) and never a reason to disengage or refuse. A mode is ALL of its levers: it engages every lever of
its set or refuses by name — it never runs under its name with a subset. A lever that cannot run on the loaded model (the hoist's records show a
family not rebound, a package lever's patch point absent) is named on the APPLIED / PACKAGE lines (``fallbacks=``) and the mode refuses (NOT ACTIVE:
lever application incomplete, exit 3, on every route); a planned lever whose run-time evidence disagrees with its plan (TF32 not live at exit, sdedup
served nothing, designs written while no sampling call went through the installed levers) or a hook that never ran is a problem sentence of the
exit census (`exit_census`, the ONE producer for every route): the design verb prices it (NOT ACTIVE by name, exit 3, outputs kept and listed), and
on the PXDESIGN_OPT route — upstream's own program, no verb of the kit above it — the exit verdict registered at activation (`exit_verdict`) prints
the same NOT ACTIVE line at interpreter exit and ends the process with exit 3 through the core's forced exit (the EXIT tally printed first);
there is no opt-out on that route (PXDESIGN_OPT=off runs stock). The other refusals are the cannot-engage-at-all cases — no visible GPU, the
carried kit files absent, a model built before activation — and an installed upstream that is not the pinned stock
(`stock/check_pins.py`: another commit or a modified checkout changes what "stock" means; `_stock_gate`). The `check` verb reports the same pin
facts as its ``PINS`` line.

The core pin is not gated here: the core pin gate (`pxdesign_opt.core_gate`, `_core_gate.py`) is statement one of every entry and has
refused by name, exit 3, before this module — which imports the core — is reached when the importable `opt_core` is not the one
`opt/pyproject.toml` pins; the report's `core_pin` block is that gate's facts (`core_pin_facts`, the same producer called again), and no
switch overrides it. The report names what the kit's own records show patched (`classify`: the class attributes install() rebinds, the model's
`_pxd_hoist_ready` flag, the TF32 flags read back), never gated by the package. `activate(mode, dry_run=True)` resolves, gates and reports
without applying anything (`check`), and works without torch.

Late activation follows one rule: allowed after the upstream package is imported, refused with a named error once a
`ProtenixDesign` instance exists (`opt_core.instances`: a constructor wrap registered at activation) or once the kit's own module reports
the lever installed (`kit_levers_applied`).
"""
import atexit
import functools
import logging
import importlib
import importlib.util
import os
import sys
from typing import Optional

from opt_core import gates as _gates
from opt_core import home as _home
from opt_core import instances as _instances
from opt_core import report as _core_report

from . import TAG, core_gate
from . import precision as _precision
from . import report as _report
from .modes import kit_switch_defaults, resolve, check_mode
from .registry import KIT_RELPATH, KIT_VERSION_LINE, LEVERS, LEVER_FILE, LEVER_MODULE, list_kit_files

ENV_HOME = "PXDESIGN_OPT_HOME"                             # the pxdesign/ tree (default: MODEL_OPT, else derived from this package's location)
ENV_TARGET_GPU = "MODEL_OPT_TARGET_GPU"                    # deployment parameter (configs/<gpu>.env): the GPU class this configuration targets
MODEL_MODULE, MODEL_CLASS = "pxdesign.model.pxdesign", "ProtenixDesign"
RUNNER_MODULE, RUNNER_CLASS, RUNNER_METHOD = "pxdesign.runner.inference", "InferenceRunner", "load_checkpoint"
UPSTREAM = ("pxdesign", "protenix", "pxdbench")
MEASURED_CC = ("8.0", "9.0")                              # the compute capabilities the kit targets (sm_80 A100, sm_90 H100); any other class engages with a note on the activation line
_REPORT: Optional[dict] = None
_HOOK = {"armed": False, "wrapped": False, "finder": None, "applications": []}
_VERDICT = {"registered": False}                            # the PXDESIGN_OPT route's exit verdict: registered with atexit once per process (register_exit_verdict)
HOOK_NEVER_RAN = "the hook never ran (no InferenceRunner.load_checkpoint call): the lever was not applied"


class ActivationError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------------------------------ paths

def tree_home() -> str:
    return _home.tree_home(__file__, env_home=ENV_HOME, levels=2)


def opt_dir() -> str:
    return os.path.join(tree_home(), "opt")


def kit_home() -> str:
    return os.path.join(tree_home(), KIT_RELPATH)


def kit_dirs() -> list:
    """The directories a stock process must not have on its path or hold modules from: the carried lever kit (opt_core.stock_proof kit_dirs)."""
    return [os.path.join(opt_dir(), "forward"), os.path.join(opt_dir(), "serving")]


def pins_path() -> str:
    return os.path.join(tree_home(), "stock", "PINS.json")


def read_pins() -> dict:
    return _gates.load_pins(pins_path())


def check_pins_module():
    """stock/check_pins.py, loaded from the tree (one implementation of the pin check)."""
    p = os.path.join(tree_home(), "stock", "check_pins.py")
    spec = importlib.util.spec_from_file_location("pxdesign_stock_check_pins", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def package_version() -> str:
    from . import __version__
    return __version__


# ------------------------------------------------------------------------------------------------------------------------ gates

def upstream_versions() -> dict:
    return {name: _gates.dist_version(name) for name in UPSTREAM}


def core_pin_facts() -> dict:
    """The report's `core_pin` block: the core pin gate's facts (`pxdesign_opt.core_gate`, the gate every entry runs first, called again — cheap,
    idempotent, one producer): the installed core `{package_dir, root, version}` plus `ok` (always True here: a mismatch has exited 3
    at entry) and `pinned` `{path, version, pyproject}`."""
    facts = core_gate()
    return dict(facts["installed"], ok=True, pinned=facts["pinned"])


def kit_status(kh: Optional[str] = None) -> dict:
    """The lever kit is present on disk (its directory non-empty; registry.list_kit_files): a partial checkout or a packaging drop,
    not a byte check — git (or the sdist/wheel build) already guarantees the checked-out bytes."""
    kh = kh or kit_home()
    lever = os.path.join(kh, LEVER_FILE)
    st = {"kit_home": kh, "present": os.path.isfile(lever), "carry": None}
    if not st["present"]:
        return st
    counts = {"hoist": len(list_kit_files(kh))}
    empty = sorted(name for name, n in counts.items() if n == 0)
    st["carry"] = {"ok": not empty, "reason": None if not empty else f"carried directory empty or absent: {', '.join(empty)}", "counts": counts}
    return st


def pins_report(pins: Optional[dict] = None) -> dict:
    """stock/check_pins.py on the installed upstream packages — {"pinned", "bad", "detail", "stack"}: the `check` verb's PINS line, and the facts
    `_stock_gate` refuses a kit mode on (``off`` is not gated)."""
    pins = pins or read_pins()
    try:
        cp = check_pins_module()
        bad, detail = cp.check(pins["upstream"], tree=tree_home())
        stack = cp.stack_report(pins)
    except Exception as e:  # noqa: BLE001
        return {"pinned": False, "bad": [f"check_pins failed: {e!r}"], "detail": {}, "stack": None}
    return {"pinned": not bad, "bad": bad, "detail": detail, "stack": stack}


def gpu_probe() -> dict:
    """The GPU description: the core's torch probe (`name`, `cc`, `sm`, `memory_mib`, `probe`) plus `mem_gib`, the torch and CUDA versions and the
    device count (name None when no GPU or no torch)."""
    g = {"name": None, "cc": None, "sm": None, "memory_mib": None, "mem_gib": None, "torch": None, "cuda": None, "count": 0, "probe": "torch"}
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        g["probe"] = f"no torch: {type(e).__name__}"
        return g
    g["torch"] = torch.__version__; g["cuda"] = getattr(torch.version, "cuda", None)
    try:
        g.update(_gates.torch_gpu_probe())
        if g["name"] is not None:
            g["count"] = torch.cuda.device_count()
            g["mem_gib"] = round(g["memory_mib"] / 1024, 1) if g.get("memory_mib") is not None else None
    except Exception as e:  # noqa: BLE001
        g["probe"] = f"cuda probe failed: {type(e).__name__}: {e}"
    return g


def _stack_gates(base: dict) -> Optional[str]:
    """The reason activation would refuse on this box, or None: the lever kit absent on disk, no visible GPU (the core pin is not decided here:
    its gate refused at entry; the stock pin is `_stock_gate`). A GPU class outside `MEASURED_CC`, the torch build, the target-GPU word and the
    environment's TF32 words never refuse — those are notes (`_base`)."""
    ks = base["kit"]
    if not ks["present"]:
        return f"lever kit missing: {os.path.join(ks['kit_home'], LEVER_FILE)} (set {ENV_HOME} or MODEL_OPT to the pxdesign/ tree)"
    if ks["carry"] is not None and not ks["carry"]["ok"]:
        return f"carried kit files missing: {ks['carry']['reason']}"
    g = base["gpu"]
    if g["name"] is None:
        return "no visible GPU" + ("" if g["probe"] == "torch" else f" ({g['probe']})")
    return None


def _stock_gate(base: dict) -> Optional[str]:
    """The one environment refusal: the installed upstream packages are not the pinned stock (`stock/check_pins.py`: content against the tree's
    copy, pip's recorded commit as provenance) — another commit or a modified checkout changes what "stock" means, so a kit mode does not run on
    it (mode off is upstream as installed and is not gated). The facts ride the report (`stock_pins`); `check` prints them as its PINS line."""
    pr = pins_report()
    base["stock_pins"] = {"pinned": pr["pinned"], "bad": list(pr["bad"])}
    if pr["pinned"]:
        return None
    return "the installed upstream is not the pinned stock — " + "; ".join(str(b) for b in pr["bad"]) + " (STOCK.md; run.sh install)"


# ---------------------------------------------------------------------------------------------------------- instance counting

def register_instance_counter() -> dict:
    """Count ProtenixDesign instances from now on (opt_core.instances: a constructor wrap now, or at the model module's import)."""
    return _instances.register_instance_counter(MODEL_MODULE, MODEL_CLASS)


def instance_check() -> Optional[str]:
    ic = _instances.instance_check(MODEL_MODULE, MODEL_CLASS)
    if ic["n"] > 0:
        return f"a {MODEL_CLASS} instance already exists in this process (live={ic['n']}, built={ic['built']}, {ic['method']}): a model built before activation would sample without the lever"
    return None


def kit_levers_applied() -> Optional[str]:
    mod = sys.modules.get(LEVER_MODULE)
    if mod is not None and getattr(mod, "_STATE", {}).get("installed"):
        return f"the kit's own {LEVER_MODULE}.install() already ran in this process (_STATE['installed'] is True)"
    return None


# ------------------------------------------------------------------------------------------------------------------ the hook

class _ModuleFinder:
    """Meta-path finder that runs `on_loaded(module)` right after `module_name`'s body has executed. Fires once and removes itself."""

    def __init__(self, module_name, on_loaded):
        self.module_name, self.on_loaded, self.armed = module_name, on_loaded, True

    def find_spec(self, fullname, path=None, target=None):
        if not self.armed or fullname != self.module_name:
            return None
        spec = None
        for finder in sys.meta_path:
            if finder is self:
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:  # noqa: BLE001
                spec = None
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            return None
        self.armed = False
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig):
            _orig(module)
            self.remove()
            self.on_loaded(module)
        spec.loader.exec_module = exec_module
        return spec

    def remove(self):
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass


def _on_runner_module(module) -> None:
    cls = getattr(module, RUNNER_CLASS, None)
    if isinstance(cls, type):
        _wrap_runner(cls)


def _wrap_runner(cls) -> bool:
    if _HOOK["wrapped"]:
        return True
    orig_load = getattr(cls, RUNNER_METHOD)

    @functools.wraps(orig_load)
    def load_checkpoint(self, *args, **kwargs):
        out = orig_load(self, *args, **kwargs)
        try:
            apply_to(self)                                # the kit's activation point: after the checkpoint is loaded and eval() set
        except ActivationError as e:                      # a lever of the mode's set could not run: the mode refuses by name on every route — never a run under the mode's name with a subset
            _report.log(f"NOT ACTIVE: {e}")
            _REPORT["refusal_logged"] = True
            if (_REPORT or {}).get("trigger"):            # the PXDESIGN_OPT route (upstream's own console script, trigger = the importing package): end the process with the kit's code
                raise SystemExit(_report.EXIT_NOT_ACTIVE)
            raise                                         # the CLI verbs exit NOT ACTIVE (3) on it; a Python caller of enable() sees the error
        return out
    load_checkpoint._pxdesign_opt_hook = True
    setattr(cls, RUNNER_METHOD, load_checkpoint)
    _HOOK["wrapped"] = True
    return True


def arm_hook() -> str:
    """Wrap InferenceRunner.load_checkpoint now (module imported) or at its import."""
    _HOOK["armed"] = True
    mod = sys.modules.get(RUNNER_MODULE)
    if mod is not None:
        _wrap_runner(getattr(mod, RUNNER_CLASS))
        return "wrapped (module imported)"
    if _HOOK["finder"] is None:
        _HOOK["finder"] = _ModuleFinder(RUNNER_MODULE, _on_runner_module)
        sys.meta_path.insert(0, _HOOK["finder"])
    return "wrap at import"


def _kit_module():
    mod = sys.modules.get(LEVER_MODULE)
    if mod is not None:
        return mod
    _home.place_on_sys_path([kit_home()])
    return importlib.import_module(LEVER_MODULE)


def classify(model, hoist, runner=None, package_levers=()) -> dict:
    """What the records show applied after install(model): the rebinds hoist makes and the model's ready flag (h1..h5 → `levers_applied` /
    `levers_fallback`); a hoist family not shown applied is a fallback (a partial application). `package` is the package-lever hook's record
    (`sizeceil.apply` for the mode's registry.PACKAGE_LEVERS; skipped by name when the hoist base is not intact)."""
    from protenix.model.modules.diffusion import DiffusionModule
    from protenix.model.modules.transformer import AttentionPairBias, ConditionedTransitionBlock
    import protenix.model.modules.primitives as PR
    import pxdesign.model.pxdesign as P
    dm = model.diffusion_module if hasattr(model, "diffusion_module") else model
    recs = {"installed": bool(hoist._STATE.get("installed")), "model_ready": bool(getattr(dm, "_pxd_hoist_ready", False)),
            "f_forward": DiffusionModule.f_forward is hoist._f_forward_hoisted,
            "apb_forward": AttentionPairBias.forward is hoist._apb_forward_hoisted,
            "ctb_forward": ConditionedTransitionBlock.forward is hoist._ctb_forward_hoisted,
            "sample_diffusion_wrapped": bool(getattr(P.sample_diffusion, "_pxd_hoist_wrapper", False)),
            "local_attention": hoist._local_attention_hoisted in (getattr(PR, "_local_attention", None), getattr(getattr(PR, "_local_attention", None), "__wrapped__", None)),   # h5 itself, or h5 under the padmask seed wrapper (sizeceil.py)
            "mode": os.environ.get("PXD_HOIST_MODE"), "mask": os.environ.get("PXD_HOIST_MASK")}
    need = {"h1": ("f_forward", "sample_diffusion_wrapped", "model_ready"), "h2": ("apb_forward", "model_ready"), "h3": ("f_forward", "model_ready"),
            "h4": ("apb_forward", "ctb_forward", "model_ready"), "h5": ("local_attention",)}
    applied, fallback, why = [], [], {}
    for name in LEVERS:
        ok = recs["installed"] and all(recs[k] for k in need[name])
        if ok:
            applied.append(name)
        else:
            fallback.append(name); why[name] = "kit record not set after install(): " + ",".join(k for k in need[name] if not recs[k])
    pk = {"planned": list(package_levers), "applied": [], "fallback": [], "skipped": [], "fields": {}, "reasons": {}}   # the package-lever hook's record
    if package_levers and fallback:                      # the base is not intact: a package lever is never applied on top of a partial lever set
        pk["skipped"] = list(package_levers); pk["reasons"] = {name: "hoist install() incomplete: " + ",".join(fallback) for name in package_levers}
    elif package_levers:                                 # the mode's package levers (registry.PACKAGE_LEVERS, sizeceil.py), applied after install() on the intact base
        from . import sizeceil
        pk = sizeceil.apply(package_levers, hoist_mod=hoist)
    return {"levers_applied": applied, "levers_fallback": fallback, "fallback_reasons": why, "package": pk, "kit_records": recs}


def apply_to(runner_or_model) -> dict:

    """Apply the active mode's levers to a loaded model: the kit's own install(model), then the mode's package levers. Returns the application record."""
    rep = _REPORT
    model = getattr(runner_or_model, "model", runner_or_model)
    runner = runner_or_model if hasattr(runner_or_model, "model") else None
    if rep is None or not rep.get("active") or rep.get("mode") == "off":
        return {"applied": False, "reason": "no active kit mode"}
    package_levers = tuple(rep.get("package_levers") or ())
    hoist = _kit_module()
    stats = hoist.install(model)
    cl = classify(model, hoist, runner=runner, package_levers=package_levers)
    app = {"index": len(_HOOK["applications"]) + 1, "applied": stats is not None, **cl, "kit_version_line": KIT_VERSION_LINE}
    if "tf32" not in package_levers:
        app["precision"] = _precision.attest()                                               # a mode that holds the stock numerics policy: the live switches READ against it after checkpoint load (nothing set); `tf32` planned: the PACKAGE line carries the applied policy
        rep["precision"] = dict(app["precision"])
    _HOOK["applications"].append(app)
    rep["applications"] = list(_HOOK["applications"])
    rep["applied"] = "installed"
    pk = cl["package"]
    rep["levers_applied"], rep["levers_fallback"], rep["fallback_reasons"] = cl["levers_applied"], cl["levers_fallback"], cl["fallback_reasons"]
    rep["package_levers_applied"], rep["package_levers_fallback"], rep["levers_skipped"] = pk["applied"], pk["fallback"], pk["skipped"]
    rep["partial"] = bool(cl["levers_fallback"] or pk["fallback"])
    _report.log_applied(app)
    if rep["partial"]:                                                                      # a mode is all of its levers: a lever that could not run (named above, fallbacks=) makes the mode refuse by name — ActivationError, NOT ACTIVE on every route (the hook wrapper)
        rep["apply_failed"] = True
        reasons = {k: v for k, v in cl["fallback_reasons"].items() if k in cl["levers_fallback"]}
        reasons.update({k: v for k, v in pk["reasons"].items() if k in pk["fallback"]})
        rep["partial_reasons"] = dict(reasons)
        raise ActivationError(f"lever application incomplete: {reasons}")
    return app


# ----------------------------------------------------------------------------------------------------------------- reports

def _cc_string(g: dict) -> Optional[str]:
    cc = g.get("cc")
    if isinstance(cc, (tuple, list)):
        return ".".join(str(x) for x in cc)
    return None if cc is None else str(cc)


def environment_notes(base: dict) -> list:
    """The environment facts the activation line NAMES (``notes=``) and never refuses on: a GPU class outside `MEASURED_CC`, a torch build other
    than the tested stack's (`stock/PINS.json` pinned_stack), a target-GPU word that does not match the visible card, TF32 library overrides in the
    environment."""
    notes = []
    g = base["gpu"]
    cc = _cc_string(g)
    if g.get("name") and cc is not None and cc not in MEASURED_CC:
        notes.append(f"GPU {g['name']} (sm{cc.replace('.', '')}) is outside the measured classes (sm80, sm90): the levers engage, numerics and speed are unmeasured on it")
    want = ((base.get("pinned_stack") or {}).get("torch"))
    if want and g.get("torch") and g["torch"] != want:
        notes.append(f"torch {g['torch']} differs from the pinned {want} (stock/PINS.json)")
    if base["target_gpu"] and g.get("name") and base["target_gpu"].lower() not in g["name"].lower():
        notes.append(f"{ENV_TARGET_GPU}={base['target_gpu']} but the visible GPU is {g['name']}")
    tf32_env = _precision.override_words()
    if tf32_env:
        notes.append(f"TF32 library override in the environment ({tf32_env}): torch folds it into its matmul switch; the KERNELS and LEVER tf32 lines print the live value")
    return notes


def _base(mode: str, trigger: Optional[str]) -> dict:
    base = {"active": False, "mode": mode, "trigger": trigger, "package_version": package_version(), "upstream": upstream_versions(),
            "gpu": gpu_probe(), "core_pin": core_pin_facts(), "kit": kit_status(), "tree_home": tree_home(), "kit_home": kit_home(),
            "target_gpu": os.environ.get(ENV_TARGET_GPU), "pinned_stack": _pinned_stack(), "notes": [], "applications": []}
    base["notes"] = environment_notes(base)
    return base


def _pinned_stack() -> dict:
    """The tested stack of stock/PINS.json (`pinned_stack.pinned`: torch 2.3.1+cu121, cuda, triton, …; the block `check`'s PINS line compares the
    running stack with, stock/check_pins.stack_report) — {} when the file is unreadable."""
    try:
        return dict(((read_pins() or {}).get("pinned_stack") or {}).get("pinned") or {})
    except (OSError, ValueError, AttributeError, TypeError):                          # no PINS file / unreadable JSON: nothing to compare, no note (the kit-files gate names a broken tree)
        return {}


def pins_line(rep: dict) -> str:
    """The `check` verb's one PINS line from `pins_report`: ``PINS pinned=1 stack=<status>`` or ``PINS pinned=0 differs=<the check_pins sentences>``."""
    stack = rep.get("stack") or {}
    words = f"pinned={int(bool(rep.get('pinned')))} stack={stack.get('status', 'unknown')}"
    if stack and not stack.get("equal", True):
        words += f" stack_differs={stack.get('differs')}"
    if not rep.get("pinned"):
        words += " differs=" + "; ".join(str(b) for b in (rep.get("bad") or []))
    return f"PINS {words}"


def precision_check() -> dict:
    """The numerics policy planned by the applied lever set (TF32 iff `tf32` was applied, else STOCK), against the live switches re-read NOW."""
    rep = _REPORT or {}
    planned = _precision.policy_of(rep.get("mode") or "exact", rep.get("package_levers_applied") or [])
    return _precision.exit_check(planned)


def precision_gate() -> list:
    """Problem sentences (empty = pass): the live numerics switches disagree with the plan at exit — the design verb exits NOT ACTIVE by name."""
    return _precision.exit_problems(precision_check())


def runtime_census() -> dict:
    """The run-time counters of the applied package levers that keep some (sizeceil.RUNTIME_CENSUS): {lever: {...}} — the manifest's `package_census`."""
    from . import sizeceil
    rep = _REPORT or {}
    applied = list(rep.get("package_levers_applied") or [])
    return {n: sizeceil.RUNTIME_CENSUS[n]() for n in applied if n in sizeceil.RUNTIME_CENSUS}


def runtime_gate() -> list:
    """The problem sentences of the applied package levers whose activation evidence completes at run time (sizeceil.RUNTIME_GATES; empty =
    pass): a lever that served nothing, or fell back where no fallback is expected, makes the design verb exit NOT ACTIVE by name."""
    from . import sizeceil
    rep = _REPORT or {}
    out = []
    for n in rep.get("package_levers_applied") or []:
        if n in sizeceil.RUNTIME_GATES:
            out += [f"{n}: {p}" for p in sizeceil.RUNTIME_GATES[n]()]
    return out


def lever_lines() -> list:
    """The LEVER lines of the package's tolerance-class levers (tf32, sdedup) for the EXIT census: on / skipped / off per the active mode."""
    from . import sdedup as _sdedup
    rep = _REPORT or {}
    planned = list(rep.get("package_levers") or [])
    fields = {}
    for app in rep.get("applications") or []:
        for k, v in ((app.get("package") or {}).get("fields") or {}).items():
            if k.startswith("tf32_"):
                fields[k[len("tf32_"):]] = v
    return [_precision.lever_line(TAG, fields, "tf32" in planned), _sdedup.lever_line(TAG, "sdedup" in planned)]


def sampler_prepares() -> Optional[int]:
    """How many `sample_diffusion` calls went through the installed hoist in this process — the carried lever module's own `prepare_cache`
    count (`pxd_xattempt.hoist.stats()["prepares"]`); None when the module never loaded or its counters are unreadable."""
    mod = sys.modules.get(LEVER_MODULE)
    if mod is None:
        return None
    try:
        return int((mod.stats() or {}).get("prepares") or 0)
    except Exception:  # noqa: BLE001 — unreadable counters: not a number, never a guess
        return None


def exit_census(produced: Optional[int] = None) -> dict:
    """The active mode's end-of-run evidence — the ONE producer for every route (the design verb after its run; `exit_verdict` at interpreter
    exit on the PXDESIGN_OPT route): ``{"problems": [...], "precision": {...}, "package_census": {...}, "prepares": n}``. A problem sentence
    (empty list = the mode did its work) names a planned lever that did not do it: an applied package lever whose run-time evidence disagrees
    with its plan (`runtime_gate`: sdedup served nothing or fell back; `precision`: TF32 not live at exit), the hook that never ran (no
    `InferenceRunner.load_checkpoint` call: no lever applied), and — when the caller knows designs were written this run (``produced`` > 0,
    the design verb's file census) — a hoist that was installed and never prepared (prepares=0: no sampling call went through the levers, so
    those designs are not the mode's). The precision record and the package levers' run-time counters ride along for the manifest. Calling it
    marks the report ``exit_judged``: the caller owns the process's exit code and the PXDESIGN_OPT exit verdict stands down."""
    rep = _REPORT or {}
    problems, precision_rec, census, prepares = [], None, {}, sampler_prepares()
    if rep.get("active") and rep.get("mode") not in (None, "off"):
        problems += runtime_gate()
        precision_rec = precision_check()
        problems += _precision.exit_problems(precision_rec)
        census = runtime_census()
        if not rep.get("applications"):
            problems.append(HOOK_NEVER_RAN)
        elif produced and not prepares:
            problems.append(f"hoist: {produced} design(s) written this run but no sample_diffusion call went through the installed levers (prepares={prepares})")
    if _REPORT is not None:
        _REPORT["exit_judged"] = True
        _REPORT["exit_problems"] = list(problems)
    return {"problems": problems, "precision": precision_rec, "package_census": census, "prepares": prepares}


def _model_built() -> bool:
    """A ProtenixDesign was constructed in this process since activation (opt_core.instances: the constructor wrap's total ``built``, or the
    live count where the class could only be gc-scanned) — the program built the model, so a hook that never ran left it without the levers."""
    ic = _instances.instance_check(MODEL_MODULE, MODEL_CLASS)
    return int(ic.get("built") or 0) > 0 or int(ic.get("n") or 0) > 0


_ENV_ROUTE_EXIT = f"PXDESIGN_OPT route: exit {_report.EXIT_NOT_ACTIVE} (PXDESIGN_OPT=off runs stock)"


def exit_verdict() -> Optional[int]:
    """The PXDESIGN_OPT route's end-of-process rule (upstream's own program: no verb of the kit above it to price the exit census). Read at
    interpreter exit: nothing to judge (None) when no kit mode is active, when a verb already judged the exit (`exit_census` ran), or when nothing
    of the model ran in this process — no lever application and no ProtenixDesign built (a help text, an input check: the plan line stands
    unexercised and the EXIT tally names it). Otherwise the exit census is read; a problem sentence prints ``NOT ACTIVE: lever run-time census: …``
    and the code to force is EXIT_NOT_ACTIVE. No opt-out on this route (PXDESIGN_OPT=off runs stock)."""
    rep = _REPORT
    if not rep or not rep.get("active") or rep.get("mode") in (None, "off") or rep.get("exit_judged"):
        return None
    if not rep.get("applications") and not _model_built():
        return None
    problems = exit_census()["problems"]
    if not problems:
        return None
    _report.log("NOT ACTIVE: lever run-time census: " + "; ".join(problems) + f" — {_ENV_ROUTE_EXIT}")
    return _report.EXIT_NOT_ACTIVE


def _exit_verdict_hook() -> None:
    """atexit: force the verdict's code through the core's forced exit (upstream's logging handlers shut down, the EXIT tally and LEVER lines not
    yet printed print first, streams flushed, then ``os._exit`` — the interpreter's own status cannot be changed from an exit hook any other
    way). Fail-closed: a verdict that cannot be read is named and priced EXIT_NOT_ACTIVE, never a pass."""
    try:
        code = exit_verdict()
    except Exception as e:  # noqa: BLE001 — fail-closed at interpreter exit: named, exit 3
        _report.log(f"NOT ACTIVE: exit census unreadable at interpreter exit: {e!r} — {_ENV_ROUTE_EXIT}")
        code = _report.EXIT_NOT_ACTIVE
    if code is not None:
        logging.shutdown()                                   # upstream's log handlers flushed and closed: os._exit skips logging's own exit hook
        _core_report.forced_exit(code)


def register_exit_verdict() -> bool:
    """Register the PXDESIGN_OPT route's exit verdict with atexit, once per process; True when registered now. Registered AFTER the EXIT tally
    (`activate`), so it runs BEFORE it: the NOT ACTIVE line, then the tally, then the forced status."""
    if _VERDICT["registered"]:
        return False
    atexit.register(_exit_verdict_hook)
    _VERDICT["registered"] = True
    return True


def status() -> dict:
    if _REPORT is None:
        return {"active": False, "reason": "enable() has not run in this process"}
    return dict(_REPORT)


def _disarm_autoload() -> None:
    try:
        from . import _autoload
        _autoload.disarm()
    except Exception:  # noqa: BLE001
        return


def _dry_run(mode: str, trigger: Optional[str], strict: bool) -> dict:
    base = _base(mode, trigger)
    base["dry_run"] = True
    res = resolve(mode)
    if mode == "off":
        rep = dict(base, levers_planned=[], package_levers=[], env={}, tier="stock", reason="dry run: mode off is stock; nothing would be applied")
        _report.log_activation(rep); rep["logged"] = True
        return rep
    would_refuse = _stack_gates(base) or _stock_gate(base) or instance_check() or kit_levers_applied()
    rep = dict(base, levers_planned=res.levers_planned, package_levers=list(res.package_levers), tier=res.tier, env=dict(res.env), unset=res.dropped,
               notes=base["notes"] + res.notes, precision_policy=_precision.policy_of(mode, res.package_levers).name,
               would_refuse=would_refuse, reason=("dry run: resolved; activation would refuse: " + would_refuse) if would_refuse else "dry run: resolved and gated, nothing applied")
    _report.log_activation(rep); rep["logged"] = True
    if strict and would_refuse:
        raise ActivationError(would_refuse)
    return rep


def activate(mode: str, strict: bool = False, trigger: Optional[str] = None, dry_run: bool = False) -> dict:
    """Apply a mode once per process (idempotent). Returns the activation report; `strict` raises ActivationError when not active;
    `dry_run` resolves, gates and reports (report["dry_run"] = True) without exporting or applying anything."""
    global _REPORT
    mode = check_mode(mode)                                                                # exactly modes.MODES; any other name raises ValueError with the table's one sentence (modes.unknown_message)
    if dry_run:
        return _dry_run(mode, trigger, strict)
    _disarm_autoload()                                                                      # an explicit call wins over the PXDESIGN_OPT route
    if _REPORT is not None and (_REPORT.get("active") or _REPORT.get("apply_failed") or _REPORT.get("mode") == "off"):
        if _REPORT.get("mode") == mode:
            return dict(_REPORT)                                                           # idempotent
        reason = f"already active as mode={_REPORT.get('mode')}; the lever patches process-wide and is applied once per process (restart to change mode)"
        if strict:
            raise ActivationError(reason)
        return dict(_REPORT, active=False, refused=True, requested={"mode": mode}, reason=reason)
    base = _base(mode, trigger)
    if mode == "off":
        rep = dict(base, active=False, levers_planned=[], levers_applied=[], package_levers=[], env={}, tier="stock", reason="mode off: stock, nothing applied")
        _REPORT = rep
        _report.log_activation(rep); rep["logged"] = True
        return dict(rep)
    why = _stack_gates(base) or _stock_gate(base) or instance_check() or kit_levers_applied()
    if why:
        rep = dict(base, reason=why)
        _report.log_activation(rep); rep["logged"] = True
        if strict:
            raise ActivationError(why)
        return rep
    res = resolve(mode)
    for k in res.dropped:                                                                  # the table decides; caller switches are dropped
        os.environ.pop(k, None)
    for k, v in res.env.items():
        os.environ[k] = v
    _report.register_exit_tally()                                                          # before the kit module is imported
    if trigger:
        register_exit_verdict()                                                            # the PXDESIGN_OPT route: upstream's program owns the process, the verdict prices the exit census at its end (after the tally in atexit's LIFO: printed before it)
    counter = register_instance_counter()
    how = arm_hook()
    rep = dict(base, active=True, levers_planned=res.levers_planned, package_levers=list(res.package_levers), tier=res.tier,
               levers_applied=[], levers_fallback=[], levers_skipped=[], partial=False, env=dict(res.env), unset=res.dropped, notes=base["notes"] + res.notes, applied="deferred",
               hook=how, instance_counter=counter, kit_defaults=kit_switch_defaults(kit_home()), precision_policy=_precision.policy_of(mode, res.package_levers).name)
    _REPORT = rep
    _report.log_activation(rep); rep["logged"] = True
    return dict(rep)


def reset_for_tests() -> None:
    """Forget the activation state (tests only; a real process activates once)."""
    global _REPORT
    _REPORT = None
    _HOOK.update(armed=False, wrapped=False, finder=None, applications=[])
    _instances._COUNTERS.clear()
    from . import sdedup as _sdedup
    _sdedup.reset_for_tests()
    for f in list(sys.meta_path):
        if isinstance(f, (_ModuleFinder, _instances._ClassFinder)):
            sys.meta_path.remove(f)
