"""Activation: kit paths, pins, the GPU, and the component application through the kit's own apply line.

`activate(mode)` resolves the mode from the kit's own tables (modes.resolve), gates (kit present, stock/PINS.json, the stack pins
read via importlib.metadata, a visible CUDA device whose name is in the kit's PINS['device_names'], no kit OFF switch in the environment), imports
the kit module from its one PYTHONPATH root and checks that its live tables equal the ones read from the file, then ARMS the
application: Flashzoi has no import-time activation point — the kit attaches to a LOADED model by construction
(`kit.KitRunner(model)`) — so a class-level hook on `borzoi_pytorch.pytorch_borzoi_model.Borzoi.forward` runs the kit's documented
apply line on each model instance at its first forward, or eagerly when `apply_to(model)` is called (`pred` does). The kit's
KitRunner installs its own INSTANCE-level `forward` (routed to `runner.predict_tensor`) and the attach mark on the model, which
shadows the class-level hook for `model(x)` from then on; the hook itself dispatches to that instance forward. KitRunner is the
only place components are installed; the mode is its ONE constructor argument (modes.MODE_ARGS: exact = `KitRunner(model)` at the kit's
defaults, numerics='tf32'; the defaults and the accepted values are read from the kit's bytes and the defaults asserted on the live class). The report reads the numerics class back from the runner (`numerics_knob`, `route_class`,
`class_label`).

An instance on a CPU / non-CUDA device cannot take the kit (the kit attaches to CUDA models): under an active mode its first forward
is REFUSED by name (ActivationError; the CLI exits 3) — never a silent stock forward.

The report names what the kit's own records show (`runner.components` at attach, `runner.effective_flags()` after the job): a component the
kit's counters do not evidence after a run is `components_fallback` and the report says `partial`. `activate(mode, dry_run=True)`
resolves, gates and reports without applying anything (`check`), and works without the upstream package.

Late activation (adapted to a post-load kit): `enable()` is allowed before any Borzoi instance exists and after instances exist
(they attach at their first forward); it is refused by name once a kit runner exists in the process under other knobs (a user's
own `KitRunner(m, numerics='tf32')`); a runner under the mode's own knobs is adopted. Forwards run before `enable()` are not
witnessed by the package (there is no hook yet) — the report notes the instances found at activation; from `enable()` on, every
forward of the hooked class attaches or refuses, so no stock forward runs under the mode.
"""
from __future__ import annotations

import gc
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import sys
import time
from typing import Optional

from . import ActivationError, __version__
from . import report as _report
from .modes import (ENV_MODE, KIT_MODULE, KIT_RELPATH, KIT_KNOBS, MODES, Resolution, check_mode, kit_class_record, kit_class_rules, kit_table, kit_init_path, resolve,
                    stack_key)

ENV_TREE = "MODEL_OPT"                                            # run.sh convention: this model's directory (flashzoi/); the one override of the package-relative tree
PACKAGE_ENV = (ENV_MODE,)                                         # the package's own switch (stripped from the stock route's subprocess)
KIT_ENV_SWITCHES = ()                                             # the kit's OWN switch names, read from its bytes (opt/forward/kits_v1_25, the os.environ reads of the forward path): none — the kit reads no environment switch                                                                 # names, never a prefix: a variable the kit never reads is not a switch (an external caller may export its own KIT_*/FZ-less bookkeeping)
TF32_OVERRIDE_ENV = ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")   # library-level TF32 overrides: environment drift — named on the ACTIVE line (drift=…) and recorded, never a refusal (stock runs under them as it does)
STRIPPED_ENV = PACKAGE_ENV + KIT_ENV_SWITCHES                     # the ONE env list of the package — stripped from the stock route's subprocess (the package's and the kit's switches) …
FORBIDDEN_ENV = KIT_ENV_SWITCHES                                  # … and refused by the gates when set in this process: the kit's own switches only (none today); every report records env_checked / env_hits
KIT_NAMESPACES = ("engines",)                                     # the kit's closure root under the kit root
STOCK_DIST = "borzoi-pytorch"
STOCK_MODULE, STOCK_CLASS = "borzoi_pytorch.pytorch_borzoi_model", "Borzoi"
STOCK_HELPER_MODULE = "borzoi_pytorch.pytorch_borzoi_helpers"
TRIGGER_PACKAGE = "borzoi_pytorch"
PINS_RELPATH = os.path.join("stock", "PINS.json")
PIN_DISTS = {"torch": "torch", "triton": "triton", "flash-attn": "flash-attn", "transformers": "transformers"}   # PINS.json "stack" key -> distribution
PINS_SCHEMA = {"package": ("name", "version"), "stack": ("torch", "triton", "flash-attn", "transformers", "python"),
               "weights": (), "config_json_sha256": ()}

_REPORT: Optional[dict] = None
_ACTIVATING = False
_HOOK = {"installed": False, "original": None, "owner": None}
_ATTACHED: dict = {}                                              # id(model) -> attach record
_RUNNERS: list = []                                               # the kit runners the package attached, in order
_MODELS: list = []                                                # the attached models (strong refs: the runner holds the model anyway)
_INSTANCES_AT_ENABLE = {"n": None}


# ------------------------------------------------------------------------------------------------------------------- paths
def opt_home() -> str:
    """The package's install directory (flashzoi/opt) — the editable install keeps the kit beside it."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tree_home() -> str:
    """This model's directory (flashzoi/): $MODEL_OPT (run.sh's convention), else the package-relative tree. A named directory that
    does not exist is FileNotFoundError (never a silent fallback)."""
    for var in (ENV_TREE,):
        v = os.environ.get(var)
        if v:
            if not os.path.isdir(v):
                raise FileNotFoundError(f"{var}={v!r} is not a directory")
            return os.path.abspath(v)
    return os.path.dirname(opt_home())


def kit_root() -> str:
    """The kit's one PYTHONPATH root: opt/forward/kits_v1_25 under the tree; must carry the kit module."""
    root = os.path.join(tree_home(), KIT_RELPATH)
    if not os.path.isfile(kit_init_path(root)):
        raise FileNotFoundError(f"kit not found: {kit_init_path(root)} (the tree carries it under {KIT_RELPATH})")
    return root


def pins_path() -> str:
    return os.path.join(tree_home(), PINS_RELPATH)


def read_pins(path: Optional[str] = None) -> dict:
    """stock/PINS.json (the one place the stock + weights pins live; schema PINS_SCHEMA). Missing file / key -> a named error."""
    p = path or pins_path()
    if not os.path.isfile(p):
        raise FileNotFoundError(f"stock/PINS.json not found at {p} (the stock helper writes it; set {ENV_TREE} to the flashzoi/ directory)")
    pins = json.load(open(p, encoding="utf-8"))
    for key, fields in PINS_SCHEMA.items():
        if key not in pins:
            raise KeyError(f"{p}: missing key {key!r}")
        for f in fields:
            if f not in pins[key]:
                raise KeyError(f"{p}: missing key {key}.{f}")
    return pins


# ------------------------------------------------------------------------------------------------------------------- pins / gpu
def _dist_version(dist: str) -> Optional[str]:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def installed_versions() -> dict:
    """The installed versions of the pinned distributions (importlib.metadata; None when absent) and the python version."""
    out = {STOCK_DIST: _dist_version(STOCK_DIST)}
    out.update({k: _dist_version(d) for k, d in PIN_DISTS.items()})
    out["python"] = platform.python_version()
    return out


def _public(v: Optional[str]) -> Optional[str]:
    return None if v is None else v.split("+")[0]


def stock_pin_check(pins: dict, installed: Optional[dict] = None) -> list:
    """The ONE pin a mode refuses over: the installed stock package is the pinned upstream release (its version here; run.sh's check_pins.py
    compares the wheel's RECORD digests too) — anything else changes what "stock" means. Empty = met."""
    inst = installed if installed is not None else installed_versions()
    want = pins["package"]["version"]
    have = inst.get(STOCK_DIST)
    return [f"{STOCK_DIST} {have or 'not installed'} != pinned {want}"] if _public(have) != want else []


def pins_check(pins: dict, installed: Optional[dict] = None) -> list:
    """Where the installed STACK differs from stock/PINS.json (empty = as pinned): every "stack" pin compared on the public version (a wheel's
    local tag, e.g. +cu124, is not compared) and python major.minor. A difference is environment drift: NAMED on the ACTIVE line
    (drift=…) and recorded, never a reason to refuse or disengage — the stock pin is stock_pin_check's."""
    inst = installed if installed is not None else installed_versions()
    bad = []
    for key in PIN_DISTS:
        want = pins["stack"].get(key)
        have = inst.get(key)
        if _public(have) != _public(want):
            bad.append(f"{key} {have or 'not installed'} != pinned {want}")
    py_want = str(pins["stack"].get("python") or "")
    py_have = ".".join(str(inst.get("python") or "").split(".")[:2])
    if py_want and ".".join(py_want.split(".")[:2]) != py_have:
        bad.append(f"python {py_have} != pinned {py_want}")
    return bad


def torch_gpu_info() -> dict:
    """The visible CUDA device through torch (torch imported): name, compute capability, memory; `error` when none."""
    out = {"probe": "torch", "name": None, "cc": None, "sm": None, "memory_mib": None, "count": 0, "torch": None, "cuda": None, "error": None}
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        out["error"] = f"torch import failed: {e!r}"
        return out
    out["torch"] = getattr(torch, "__version__", None)
    out["cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        if not torch.cuda.is_available():
            out["error"] = "no CUDA device visible to torch"
            return out
        out["count"] = int(torch.cuda.device_count())
        out["name"] = torch.cuda.get_device_name(0)
        maj, mnr = torch.cuda.get_device_capability(0)
        out["cc"] = f"{maj}.{mnr}"; out["sm"] = f"{maj}{mnr}"
        out["memory_mib"] = int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))
    except Exception as e:  # noqa: BLE001
        out["error"] = f"torch GPU probe failed: {e!r}"
    return out


def env_hits(names, environ=None) -> list:
    """The names of `names` set in `environ` (this process's by default), sorted; exact names, never a prefix."""
    environ = os.environ if environ is None else environ
    return sorted(k for k in environ if k in names)


def upstream_version() -> Optional[str]:
    return _dist_version(STOCK_DIST)


# ------------------------------------------------------------------------------------------------------------------- the jobs form (pred --jobs)
def read_multi_jobs(path: str) -> list:
    """The jobs file: one job per line, `input<TAB>out` — `input` = what `pred --input` takes (a directory of `<item>.npy` one-hots
    or a JSON item list), `out` = that job's output directory; `#` lines and blank lines skipped; a relative path resolves against the
    file's directory. Refused by name (ActivationError): a malformed line, an input that does not exist, an output directory named twice,
    a file without jobs. Returns [(input, out), ...] in file order."""
    if not os.path.isfile(path):
        raise ActivationError(f"--jobs {path!r}: not a file")
    base = os.path.dirname(os.path.abspath(path))
    jobs, seen = [], set()
    with open(path, "r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            s = line.rstrip("\n")
            if not s.strip() or s.lstrip().startswith("#"):
                continue
            cols = s.split("\t")
            if len(cols) != 2 or not cols[0].strip() or not cols[1].strip():
                raise ActivationError(f"{path}:{ln}: a job line is `input<TAB>out` (two tab-separated paths)")
            inp, out = (c.strip() for c in cols)
            inp = inp if os.path.isabs(inp) else os.path.join(base, inp)
            out = out if os.path.isabs(out) else os.path.join(base, out)
            if not os.path.exists(inp):
                raise ActivationError(f"{path}:{ln}: input {inp!r} does not exist")
            key = os.path.normpath(out)
            if key in seen:
                raise ActivationError(f"{path}:{ln}: output directory {out!r} is named twice (one rows.jsonl per job)")
            seen.add(key)
            jobs.append((inp, out))
    if not jobs:
        raise ActivationError(f"{path}: no jobs")
    return jobs


# ------------------------------------------------------------------------------------------------------------------- kit import
def kit_sys_path(root: str) -> None:
    if root not in sys.path:
        sys.path.insert(0, root)


def import_kit(root: str, table: dict):
    """Import the kit module from its root and assert its live tables equal the ones read from the file (LEVERS, PINS'
    device_names, ARM) and KitRunner's knob defaults equal the read ones."""
    kit_sys_path(root)
    kit = importlib.import_module(KIT_MODULE)
    if tuple(kit.LEVERS) != tuple(table["LEVERS"]):
        raise ActivationError(f"kit table disagrees with its file: LEVERS {tuple(kit.LEVERS)!r} vs file {tuple(table['LEVERS'])!r}")
    if tuple(kit.PINS.get("device_names") or ()) != tuple(table["PINS"].get("device_names") or ()):
        raise ActivationError("kit table disagrees with its file: PINS['device_names']")
    if getattr(kit, "ARM", None) != table.get("ARM"):
        raise ActivationError(f"kit table disagrees with its file: ARM {getattr(kit, 'ARM', None)!r} vs file {table.get('ARM')!r}")
    sig = inspect.signature(kit.KitRunner.__init__).parameters
    live = {k: sig[k].default for k in KIT_KNOBS if k in sig}
    if live != dict(table["knob_defaults"]):
        raise ActivationError(f"KitRunner knob defaults disagree with the file: live {live!r} vs file {table['knob_defaults']!r}")
    return kit


# ------------------------------------------------------------------------------------------------------------------- instances
def _model_class() -> Optional[type]:
    mod = sys.modules.get(STOCK_MODULE)
    cls = getattr(mod, STOCK_CLASS, None) if mod is not None else None
    return cls if isinstance(cls, type) else None


def model_instances() -> list:
    """The live Borzoi instances in this process (a gc scan; [] when the upstream module is not imported)."""
    cls = _model_class()
    if cls is None:
        return []
    gc.collect()                                                             # live instances only (a released model is not an instance)
    return [o for o in gc.get_objects() if issubclass(type(o), cls)]        # type(o): no __class__ lookup on lazy proxies


def _runner_of(model):
    return model.__dict__.get("_flashzoi_kit_runner")


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


# ------------------------------------------------------------------------------------------------------------------- report
def status() -> dict:
    return dict(_REPORT) if _REPORT is not None else {"active": False, "reason": "flashzoi_opt.enable() has not run in this process"}


def _base(mode: str, trigger: Optional[str] = None) -> dict:
    return {"active": False, "mode": mode, "variant": None, "components_planned": [], "components_applied": [], "components_fallback": [], "components_unavailable": [],
            "partial": False, "gpu": {"name": None, "cc": None}, "borzoi_pytorch_version": upstream_version(), "package_version": __version__,
            "trigger": trigger, "applied": None, "models": [], "notes": [], "knobs": {}, "knobs_line": "none", "apply_line": None, "numerics": None,
            "route_class": None, "class_label": None, "stack_key": None, "kit_arm": None, "kit_root": None,
            "env_checked": list(FORBIDDEN_ENV), "env_hits": env_hits(FORBIDDEN_ENV), "env_noted": env_hits(TF32_OVERRIDE_ENV), "drift": [], "multi": None}


def _gates(base: dict, mode: str):
    """Gates in order; returns (reason or None, resolution or None, kit table or None, pins or None)."""
    try:
        root = kit_root()
    except FileNotFoundError as e:
        return str(e), None, None, None
    base["kit_root"] = root
    try:
        table = kit_table(root)
    except (FileNotFoundError, KeyError, SyntaxError) as e:
        return f"kit tables unreadable: {e}", None, None, None
    base["kit_arm"] = table.get("ARM")
    try:
        res = resolve(mode, root)
    except ValueError as e:
        return str(e), None, table, None
    base["components_planned"] = list(res.components); base["knobs"] = dict(res.knobs); base["knobs_line"] = res.knob_line; base["apply_line"] = res.apply_line
    base["numerics"] = res.knobs.get("numerics")
    try:
        pins = read_pins()
    except (FileNotFoundError, KeyError, ValueError) as e:
        return str(e), res, table, None
    base["pins_path"] = pins_path()
    bad = stock_pin_check(pins)
    if bad:                                                                  # the ONE environment refusal: another upstream release is another "stock"
        return "stock pin not met: " + "; ".join(bad) + " (stock/PINS.json 'package': the upstream release every mode, `off` included, is defined against)", res, table, pins
    unc = base["drift"]
    unc += [f"stack {b}" for b in pins_check(pins)]                          # torch / triton / flash-attn / transformers / python off their pins: named, served
    base["env_checked"] = list(FORBIDDEN_ENV); base["env_hits"] = env_hits(FORBIDDEN_ENV); base["env_noted"] = env_hits(TF32_OVERRIDE_ENV)
    sw = env_hits(KIT_ENV_SWITCHES)
    if sw:
        return (f"kit switches set in the environment {sw}: the mode table is the only switch of a run (the kit's own "
                f"switch names, read from its bytes, are OFF/override switches outside any mode) — unset them"), res, table, pins
    unc += [f"{k}={os.environ.get(k)}" for k in base["env_noted"]]          # a TF32 library override: both routes run under it; named, served
    g = torch_gpu_info()
    base["gpu"] = g
    if g.get("error"):
        return f"no usable GPU: {g['error']} (the stock needs a CUDA device as much as the kit)", res, table, pins
    base["stack_key"] = stack_key(g.get("cc"), installed_versions().get("triton"))
    device_names = tuple(table["PINS"].get("device_names") or ())
    rules = kit_class_rules(table["kit_root"])
    pinned = rules.pinned_serves(table["PINS"], g.get("name"), g.get("cc"), g.get("memory_mib")) if rules else g.get("name") in device_names   # the kit's pinned classes by capability (PINS['device_classes']) or name — the kit's own rule module
    base["device_class"] = "pinned" if pinned else None
    if not pinned:
        rec = kit_class_record(table["kit_root"], g.get("name"), g.get("cc"), g.get("memory_mib"))   # the kit dir's own class records (class_records/*.json) keyed by fz_exact.py's digest — the kit's assert_device_class reads the same files
        if rec is None:                                                      # no pinned class, no record: the levers engage and the class is NAMED unpinned (the kit's apply line words it too)
            base["device_class"] = "unpinned"
            unc.append(f"gpu {g.get('name')} cc {g.get('cc')} {g.get('memory_mib')} MiB: no pinned class or class record, bitwise vs stock unverified on this class")
        else:
            base["device_class"] = "record"; base["class_record"] = rec
    return None, res, table, pins


def _refuse(base: dict, reason: str, strict: bool, dry_run: bool = False) -> dict:
    global _REPORT
    rep = dict(base, reason=reason, dry_run=dry_run)
    if dry_run:
        rep["would_refuse"] = reason
        return rep
    _REPORT = rep
    _report.log_refusal(reason)
    if strict:
        raise ActivationError(reason)
    return dict(_REPORT)


def activate(mode: Optional[str] = None, strict: bool = False, trigger: Optional[str] = None, dry_run: bool = False, jobs: Optional[int] = None) -> dict:
    """Apply a mode once per process (idempotent). Returns the activation report; `strict` raises ActivationError when not active;
    `dry_run` resolves, gates and reports (report["dry_run"] = True) without importing the kit or arming anything."""
    global _REPORT
    m = check_mode(mode)
    if not dry_run:
        _disarm_autoload()                                                                  # an explicit call wins over the FLASHZOI_OPT route
    if not dry_run and _REPORT is not None and (_REPORT.get("active") or _REPORT.get("apply_failed") or _REPORT.get("mode") == "off"):
        if _REPORT.get("mode") == m:
            return dict(_REPORT)                                                            # idempotent
        reason = (f"already active as mode={_REPORT.get('mode')}; a mode is activated once per process and the kit attaches process-wide "
                  f"(one mode per process: restart to change mode)")
        if strict:
            raise ActivationError(reason)
        return dict(_REPORT, active=False, refused=True, requested={"mode": m}, reason=reason)
    if dry_run:
        return _dry_run(m, trigger)
    if _ACTIVATING:
        return {"active": False, "mode": m, "reason": "activation in progress (re-entrant call ignored)"}
    return _activate_locked(m, strict, trigger, jobs)


def _dry_run(mode: str, trigger: Optional[str]) -> dict:
    base = _base(mode, trigger)
    if mode == "off":
        return dict(base, dry_run=True, reason="mode off: stock borzoi-pytorch (the upstream API in a clean subprocess; no environment set, no component applied)")
    why, res, table, pins = _gates(base, mode)
    rep = dict(base, dry_run=True)
    if why:
        rep["would_refuse"] = why
    rep["instances"] = len(model_instances())
    return rep


def _disarm_autoload() -> None:
    from . import _autoload
    for f in list(sys.meta_path):
        if isinstance(f, _autoload.Finder):
            f.armed = False
            sys.meta_path.remove(f)


def _activate_locked(mode: str, strict: bool, trigger: Optional[str], jobs: Optional[int] = None) -> dict:
    global _ACTIVATING
    _ACTIVATING = True
    try:
        return _activate_body(mode, strict, trigger, jobs)
    finally:
        _ACTIVATING = False


def _activate_body(mode: str, strict: bool, trigger: Optional[str], jobs: Optional[int] = None) -> dict:
    global _REPORT
    base = _base(mode, trigger)
    base["multi"] = jobs                                              # pred --jobs: the job count on the ACTIVE line (report.py MULTI_SUFFIX); None = one job
    if mode == "off":
        _REPORT = dict(base, reason="mode off: stock borzoi-pytorch (the upstream API in a clean subprocess; no environment set, no component applied)")
        return dict(_REPORT)
    why, res, table, pins = _gates(base, mode)
    if why:
        return _refuse(base, why, strict)
    instances = model_instances()
    _INSTANCES_AT_ENABLE["n"] = len(instances)
    foreign = [(m, _runner_of(m)) for m in instances if _runner_of(m) is not None]
    for m, r in foreign:                                                                    # the late-activation rule: a runner under other knobs
        knobs = {"numerics": getattr(r, "numerics_knob", None)}
        if knobs != dict(res.knobs):
            return _refuse(base, f"a kit runner already exists in this process under knobs {knobs} (mode {mode} runs {dict(res.knobs)}): "
                                 f"one mode per process — restart to change mode", strict)
    notes = list(base["notes"]) + list(res.notes)
    if instances:
        notes.append(f"{len(instances)} {STOCK_CLASS} instance(s) existed at activation: they attach at their first forward; forwards they ran "
                     f"before activation were not witnessed by the package")
    if KIT_MODULE in sys.modules:
        notes.append(f"{KIT_MODULE} was imported before activation")
    try:
        kit = import_kit(base["kit_root"], table)
        _install_hook()
    except Exception as e:  # noqa: BLE001
        _REPORT = dict(base, apply_failed=True, reason=f"component arming failed: {e!r}", notes=notes)
        _report.log_refusal(_REPORT["reason"])
        if strict:
            raise ActivationError(_REPORT["reason"]) from e
        return dict(_REPORT)
    os.environ[ENV_MODE] = mode                                                             # children autoload the same mode
    _report.register_exit_tally(mode)
    _REPORT = dict(base, active=True, applied="deferred", components_applied=[], components_fallback=[], components_unavailable=[], partial=False,
                   notes=notes, resolution=res, pins={"package": pins["package"], "stack": pins["stack"]}, kit_table={k: table[k] for k in ("ARM", "LEVERS") if k in table},
                   device_names=list(table["PINS"].get("device_names") or []), instances_at_enable=len(instances))
    for m, r in foreign:                                                                    # runners under the mode's knobs: adopted
        _record_attach(m, r, trigger="adopted")
    return dict(_REPORT)


# ------------------------------------------------------------------------------------------------------------------- application
def _install_hook() -> None:
    """Wrap Borzoi.forward at class level so the kit's apply line runs for a model at its first forward (the lazy route)."""
    if _HOOK["installed"]:
        return
    mod = importlib.import_module(STOCK_MODULE)
    cls = getattr(mod, STOCK_CLASS)
    orig = cls.__dict__["forward"]

    def forward(self, x, *args, **kwargs):
        inst = self.__dict__.get("forward")
        if inst is not None:                                         # attached: the kit's instance-level forward (an explicit class-level call lands here)
            return inst(x, *args, **kwargs)
        if _REPORT is None or not _REPORT.get("active"):
            return orig(self, x, *args, **kwargs)                      # no active mode: upstream's forward, untouched
        apply_to(self, trigger="forward")
        return self.forward(x, *args, **kwargs)                       # the kit's instance-level forward (serve_surfaces), or upstream's own on a model served aside
    forward.__wrapped__ = orig
    forward.__name__ = orig.__name__; forward.__qualname__ = orig.__qualname__; forward.__doc__ = orig.__doc__
    cls.forward = forward
    _HOOK.update(installed=True, original=orig, owner=cls)


def _uninstall_hook() -> None:
    if _HOOK["installed"] and _HOOK["owner"] is not None:
        _HOOK["owner"].forward = _HOOK["original"]
        _HOOK.update(installed=False, original=None, owner=None)


def _record_attach(model, runner, trigger: str) -> dict:
    global _REPORT
    kit = sys.modules[KIT_MODULE]
    apply_s = dict(getattr(runner, "apply_s", {}) or {})
    rec = {"model_index": len(_ATTACHED), "trigger": trigger, "device_name": getattr(runner, "device_name", None),
           "components_applied": sorted(getattr(runner, "components", ())), "components_fallback": [], "knobs": {"numerics": getattr(runner, "numerics_knob", None)},
           "numerics": getattr(runner, "numerics_knob", None), "route_class": getattr(runner, "route_class", None),
           "class_label": getattr(runner, "class_label", None), "apply_s": apply_s, "apply_line": getattr(runner, "apply_line", None),
           "describe": kit.describe(getattr(runner, "components", kit.ALL)), "served_surfaces": list(getattr(runner, "served_surfaces", []) or []),
           "helper_route": getattr(runner, "_helper_route", None), "_runner": runner}
    _ATTACHED[id(model)] = rec; _RUNNERS.append(runner); _MODELS.append(model)   # the model stays attached for the process: the multi-job form serves every job on these runners
    models = list(_REPORT.get("models") or []) + [{k: rec[k] for k in ("model_index", "trigger", "device_name", "components_applied", "components_fallback", "knobs", "numerics", "route_class", "class_label")}]
    _REPORT = dict(_REPORT, applied="attached", models=models, components_applied=rec["components_applied"], components_fallback=[], partial=False,
                   numerics=rec["numerics"], route_class=rec["route_class"], class_label=rec["class_label"])
    _report.log_activation(_REPORT)                                  # ONCE per process, after the kit's own line
    return _public_record(rec)


def _public_record(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def apply_to(model, trigger: str = "explicit") -> dict:
    """Run the kit's apply line `kit.KitRunner(model, **MODE_ARGS[mode])` on `model` (idempotent per instance; a model the caller moved or re-typed
    since — the kit detached itself then — attaches afresh); returns the attach record. A model the kit's kernels are not built for — not on a
    CUDA device, or parameters not float32 as `from_pretrained` loads them — is not attached: upstream's own forward serves it, said on one line
    (step_aside), never refused. Refuses by name (ActivationError, the NOT ACTIVE line) only when no mode is active or when the kit CANNOT run
    here (a kernel that does not compile or launch): a mode is all of its levers. An environment the kit is merely not pinned to (GPU class,
    stack versions, TF32 switches) is named on the kit's line and engaged (drift)."""
    global _REPORT
    if _REPORT is None or not _REPORT.get("active"):
        raise ActivationError("apply_to: no active mode (call flashzoi_opt.enable(mode) first)")
    prev = _ATTACHED.get(id(model))
    if prev is not None and not _detached(model):
        return _public_record(prev)
    r = _runner_of(model)
    if r is not None and prev is None:
        return _record_attach(model, r, trigger="adopted")
    aside = _not_for_the_kit(model)
    if aside is not None:
        return _step_aside(model, aside)
    kit = sys.modules[KIT_MODULE]
    res = _REPORT.get("resolution")
    kwargs = dict(getattr(res, "constructor_kwargs", None) or {})
    try:
        runner = kit.KitRunner(model, **kwargs)                      # THE APPLY LINE: the mode's one argument (modes.MODE_ARGS); the kit prints its one line
    except Exception as e:  # noqa: BLE001 — the kit cannot run on this model / machine: the MODE refuses by name (all of its levers or none)
        reason = f"{getattr(res, 'apply_line', 'kit.KitRunner(model)')} cannot run here: {type(e).__name__}: {str(e)[:300]} — mode {_REPORT.get('mode')} refused (a mode engages all of its levers or refuses; no subset runs under its name)"
        _REPORT = dict(_REPORT, apply_failed=True, active=False, reason=reason)
        _report.log_refusal(reason)
        raise ActivationError(reason) from e
    if prev is not None:                                             # re-attached after a detach (the caller's .to() / .half() …): the record keeps its index, the new runner replaces the old
        old = prev.get("_runner")
        if old in _RUNNERS: _RUNNERS[_RUNNERS.index(old)] = runner
        else: _RUNNERS.append(runner)
        prev.update(_runner=runner, trigger=f"re-attached ({trigger})", apply_line=getattr(runner, "apply_line", None), served_surfaces=list(getattr(runner, "served_surfaces", []) or []))
        return _public_record(prev)
    return _record_attach(model, runner, trigger)


def _detached(model) -> bool:
    """True when the kit took itself off this model (KitRunner.detach: the caller moved or re-typed it) — the instance forward is gone then."""
    r = (_ATTACHED.get(id(model)) or {}).get("_runner")
    return r is not None and getattr(r, "detached", None) is not None and "forward" not in model.__dict__


def _not_for_the_kit(model):
    """None when the kit's kernels are built for this model as it stands; else the reason upstream's own forward serves it: not on a CUDA device,
    or parameters that are not float32 (the kit pre-casts from the fp32 checkpoint `from_pretrained` loads; a .half()/.double() model computes
    something else under upstream's own arithmetic, which the kit does not reproduce)."""
    dev = _model_device(model)
    if dev is None or getattr(dev, "type", None) != "cuda":
        return f"model on device {dev} (the kit's kernels run on a CUDA device)"
    dt = _model_dtype(model)
    if dt is not None and str(dt) != "torch.float32":
        return f"model parameters are {dt} (the kit serves the float32 model `from_pretrained` loads)"
    return None


def _model_dtype(model):
    try:
        return next(model.parameters()).dtype
    except (StopIteration, AttributeError, TypeError):
        return None


_ASIDE: dict = {}                                                    # id(model) -> reason: models upstream's own forward serves (named once each)


def _step_aside(model, reason: str) -> dict:
    """Serve upstream's own forward on a model the kit is not built for: bind the class's original forward on the instance (so the hook is not
    re-entered), say it once, record it. The model is looked at again if the caller later moves it to CUDA (serve_moves_unattached)."""
    import types
    orig = _HOOK.get("original")
    if orig is None:
        mod = importlib.import_module(STOCK_MODULE); orig = getattr(mod, STOCK_CLASS).__dict__["forward"]; orig = getattr(orig, "__wrapped__", orig)
    model.forward = types.MethodType(orig, model)
    _watch_moves(model)
    if _ASIDE.get(id(model)) != reason:
        _ASIDE[id(model)] = reason
        _report.log_aside(len(_ASIDE) - 1, reason)
    return {"model_index": None, "trigger": "aside", "aside": reason, "components_applied": [], "components_fallback": []}


def _watch_moves(model) -> None:
    """A model served aside is looked at again after the caller's .to() / .cuda() / .float(): the instance forward is dropped so the next forward
    re-enters the hook (and attaches the kit if the model is a CUDA float32 model by then)."""
    def _wrapped(name):
        def f(*a, **k):
            for n_ in ("forward",) + _MOVE_NAMES:
                model.__dict__.pop(n_, None)
            _ASIDE.pop(id(model), None)
            return getattr(type(model), name)(model, *a, **k)
        return f
    for n in _MOVE_NAMES:
        if callable(getattr(type(model), n, None)):
            object.__setattr__(model, n, _wrapped(n))


_MOVE_NAMES = ("to", "cuda", "cpu", "half", "float", "bfloat16", "double", "type")


def runners() -> list:
    return list(_RUNNERS)


PARTIAL_DETECTION = ("the kit's KitRunner.effective_flags() per attached runner after the run: a lever of the mode's composition (components_planned) "
                     "without a true flag — false, absent, or the flags unreadable — is components_fallback; the `pinned` lever on the package's route "
                     "(the documented helper) is evidenced by the helper's own per-call stamp instead (helper_pinned_evidence)")
HELPER_MODULE = "engines.flashzoi.predict_tracks_fast"                # the body the kit routes the documented helper to (_wrap.route_documented_helper): the one home of this route's device->host copy
PINNED_LANDINGS = ("direct", "pool", "staging")   # every landing form of predict_tracks_fast copies device->host into pinned host memory (LAST_CALL["landing_ran"]); "stock (cpu)" is the stock body
_HELPER_CALLS: dict = {}                                                 # landing_ran -> calls: one entry per documented-helper call the package's loop made (note_helper_call)
STAMP_READ = "_flashzoi_opt_read"                                        # the package's mark on a stamp it has counted: the helper clears LAST_CALL at every call it serves (predict_tracks_fast.py:113), so a marked stamp is STALE — this call was not the helper's


def note_helper_call() -> dict:
    """After each documented-helper call of the package's loop (loop.predict_item): the kit's per-call stamp (predict_tracks_fast.LAST_CALL —
    landing_requested / landing_ran / kind / overlap), counted by the landing form that RAN and marked read (STAMP_READ) so it counts once;
    `no stamp` when the helper module is not loaded (the stock body ran), `stale stamp` when the stamp is the one already counted (the helper
    did not serve this call — the stock body over models not all attached), `unstamped` when the module is loaded but stamped nothing.
    Returns the stamp."""
    m = sys.modules.get(HELPER_MODULE)
    lc = getattr(m, "LAST_CALL", None) if m is not None else None
    if lc is None:
        key = "no stamp"
    elif lc.get("landing_ran") == "stock (cpu)":                          # the helper's stock body updates the stamp without clearing it (predict_tracks_fast.py:104): its own label, read mark or not
        key = "stock (cpu)"
    elif lc.get(STAMP_READ):
        key = "stale stamp"
    else:
        key = lc.get("landing_ran") or "unstamped"
        lc[STAMP_READ] = True
    _HELPER_CALLS[key] = _HELPER_CALLS.get(key, 0) + 1
    return {k: v for k, v in (lc or {}).items() if k != STAMP_READ}


def helper_pinned_evidence(counts: dict, runner=None) -> tuple:
    """(effective, statement) of the `pinned` lever on the package's route. The runner's own flag reads its lease pool (`predict()`, the host
    route: leases > 0 — _wrap.effective_flags_from); the package's loop calls the documented helper, which the kit routes to
    predict_tracks_fast over the runner's device path (`predict_tensor(_sliced)`: predict_device_calls, no lease) and lands the device->host
    copy in ITS pinned host buffers on every landing form. Under the package's loop (per-call stamps counted): effective iff every predict
    call of the runner took the device route and every helper call landed pinned (the stamp counts equal the runner's predict_calls; no
    `stock (cpu)` / `no stamp` / `unstamped` call). Under the environment route (the user's own calls; no per-call count): effective iff the
    kit reports the helper routed (`runner._helper_route`), every predict call took the device route and the helper's last stamp landed
    pinned — named as the weaker form in the statement."""
    n = int(counts.get("predict_calls") or 0)
    n_dev = int(counts.get("predict_device_calls") or 0)
    if _HELPER_CALLS:
        pinned = sum(v for k, v in _HELPER_CALLS.items() if k in PINNED_LANDINGS)
        other = sum(v for k, v in _HELPER_CALLS.items() if k not in PINNED_LANDINGS)
        ok = n > 0 and n_dev == n and pinned == n and other == 0
        return ok, (f"documented helper ({HELPER_MODULE}) landings {dict(sorted(_HELPER_CALLS.items()))} vs the runner's predict_calls={n} "
                    f"(device route {n_dev}): pinned host landing on every call = {ok}")
    m = sys.modules.get(HELPER_MODULE)
    last = dict(getattr(m, "LAST_CALL", None) or {}) if m is not None else {}
    route = str(getattr(runner, "_helper_route", "") or "")
    routed = route.startswith(STOCK_HELPER_MODULE + ".predict_tracks -> helper")
    ok = n > 0 and n_dev == n and routed and last.get("landing_ran") in PINNED_LANDINGS
    return ok, (f"environment route (no per-call count): helper routed = {routed} ({route or 'no route recorded'}), predict_calls={n} "
                f"(device route {n_dev}), last stamp landing_ran = {last.get('landing_ran')!r}: pinned host landing = {ok}")


def settle() -> dict:
    """After a run: the kit's own evidence per runner (`effective_flags()`, `all_counts()`, `jit_after_job()`); a lever of the mode's
    composition without a true flag (false, absent from the flags, or the flags unreadable) moves to components_fallback and the report
    says partial (PARTIAL_DETECTION, recorded as `partial_detection`); the `pinned` lever's flag is the helper route's evidence
    (helper_pinned_evidence, recorded per runner as `pinned_evidence`). Returns the updated report."""
    global _REPORT
    if _REPORT is None or not _RUNNERS:
        return status()
    models = []
    union_fb = set()
    planned = list(_REPORT.get("components_planned") or [])
    for i, r in enumerate(_RUNNERS):
        try:
            eff = r.effective_flags()
        except Exception as e:  # noqa: BLE001
            eff = {"error": repr(e)}
        try:
            counts = r.all_counts()
        except Exception as e:  # noqa: BLE001
            counts = {"error": repr(e)}
        pinned_evidence = None
        if "pinned" in planned and "error" not in eff and not eff.get("pinned"):
            ok, pinned_evidence = helper_pinned_evidence(counts if "error" not in counts else {}, r)
            eff = dict(eff, pinned=ok)
        requested = list(planned) + [k for k in eff if k != "error" and k not in planned]
        fb = sorted(k for k in requested if not eff.get(k))
        union_fb |= set(fb)
        try:
            jit = r.jit_after_job()
        except Exception as e:  # noqa: BLE001
            jit = {"error": repr(e)}
        m = dict((_REPORT.get("models") or [{}])[i] if i < len(_REPORT.get("models") or []) else {"model_index": i})
        m.update(effective_flags=eff, components_fallback=fb, counts=counts, jit_after_job=jit, apply_s=dict(getattr(r, "apply_s", {}) or {}),
                 pinned_evidence=pinned_evidence)
        models.append(m)
    _REPORT = dict(_REPORT, models=models, components_fallback=sorted(union_fb), components_unavailable=sorted(union_fb), partial=bool(union_fb),
                   components_applied=[lv for lv in planned if lv not in union_fb], partial_detection=PARTIAL_DETECTION)
    return dict(_REPORT)


def remove() -> dict:
    """The kit's documented remove over the runners the package attached (`kit.remove(runners)`), the hook restored; the report
    becomes inactive with the remove record."""
    global _REPORT
    out = {}
    if _RUNNERS:
        kit = sys.modules[KIT_MODULE]
        out = kit.remove(list(_RUNNERS))
    _RUNNERS.clear(); _ATTACHED.clear(); _MODELS.clear(); _ASIDE.clear(); _HELPER_CALLS.clear()
    _uninstall_hook()
    if _REPORT is not None:
        _REPORT = dict(_REPORT, active=False, applied="removed", reason="removed: kit.remove(runners) ran; the hook is restored", remove=out)
    return out
