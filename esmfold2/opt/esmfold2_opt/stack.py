"""Activation: kit paths, pins, the GPU probe, the mode's environment, and the lever application through the kit's own configure().

`activate(mode, variant)` resolves the mode by NAME in the kit server's MODES table (modes.resolve), gates (kit present, stock pins,
upstream versions, a visible GPU), exports the kit's own override switches for the vendored levers (EF2_MK / EF2_MSA — the only
environment the package sets, and only for modes that need them; any caller-set EF2_W4 / EF2_MK / EF2_MSA is dropped so the table
stays the source of truth), puts the kit's driver directories on sys.path, imports the server and checks that its live table
agrees with the one read from the file, then ARMS the application: ESMFold2 has no import-time activation point — the levers are
instance patches on a loaded model — so the kit's `ef2_server.configure(model, <server mode>, builder)` runs at the first
`ESMFold2InputBuilder.fold()` of each model (a wrapper on that method), or eagerly when `apply_to(model, builder)` is called
(`pred` does). configure() is the only place levers are installed; its order (ef2_mk_sampler -> ef2_msa -> ef2_w4 -> ef2_opt.install)
and its no-graphs-yet assertion are the kit's own.

The report names what the kit's own records show applied or left off (`classify`: ef2_w4.describe(), ef2_msa._STATE, the MK
instance flag, ef2_opt.install()'s return), never gated by the package: the kit's per-device policy stands and the report says
`partial` with the levers it substituted or disabled. `activate(mode, variant, dry_run=True)` resolves, gates and reports
without applying anything (`check`), and works without torch.
"""
import ast
import functools
import glob
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
import weakref
from typing import Dict, List, Optional, Sequence, Tuple

from . import ActivationError, __version__
from . import report as _report
from .modes import (MODES, VARIANTS, SERVER_RELPATH, SWITCHES, VARIANT_USES_MSA, Resolution, check_variant, describe_line, jit_cache_key, resolve,
                    server_table)
from .ablation import env_text as ablation_text, foreign_note as ablation_foreign_note   # the declared ablation variables (ESMFOLD2_OPT_ABLATE + its release-tree alias MODEL_OPT_LEVERS_OFF): their text goes to modes.resolve; nothing else reads them
UPSTREAM_DISTS = ("esm", "transformers")          # the two upstream packages (stock/PINS.json upstream.<name>.version): the versions every report carries


def dist_version(name: str) -> Optional[str]:
    """The installed distribution's version from its metadata (no import), None when not installed — opt_core.gates."""
    from opt_core.gates import dist_version as _dv
    return _dv(name)


def upstream_versions() -> dict:
    return {n: dist_version(n) for n in UPSTREAM_DISTS}
from .registry import LEVERS

ENV_MODE, ENV_VARIANT, ENV_FORCE, ENV_HOME, ENV_KIT = "ESMFOLD2_OPT", "ESMFOLD2_VARIANT", "ESMFOLD2_OPT_FORCE", "ESMFOLD2_OPT_HOME", "ESMFOLD2_OPT_KIT"
ENV_TREE = "MODEL_OPT"                                            # run.sh convention: this model's directory (esmfold2/)
PACKAGE_ENV = (ENV_MODE, ENV_VARIANT, ENV_FORCE)                      # the package's own switches (one tuple: stripped from the stock arm, never exported)
CHECK_PINS_RELPATH = os.path.join("stock", "check_pins.py")       # the one place the commit-level pin check lives (run.sh calls the same file)
KIT_CLASS_DIRS = ("forward",)                                     # opt/<class>/<kit>/ hold the kit files (one copy each; this tree carries the forward class only)
DRIVER_DIRNAME = "driver"
PINS_RELPATH = os.path.join("stock", "PINS.json")
KEY_SEP = "|"
LOG = _report.PREFIX

MODEL_MODULE = "transformers.models.esmfold2.modeling_esmfold2"              # where upstream's ESMFold2Model class lives
MODEL_CLASS = "ESMFold2Model"
_INSTANCES: dict = {"state": None, "finder": None}            # the model-instance counter: the wrapped class's state, the pending finder
_REPORT: Optional[dict] = None
_ACTIVATING = False
_HOOK = {"installed": False, "original": None, "owner": None}
_CONFIGURED: Dict[int, dict] = {}                                 # id(model) -> application record


# ----------------------------------------------------------------------------------------------------------------- paths
def opt_home() -> str:
    """esmfold2/opt — the directory the package is installed from (editable), or $ESMFOLD2_OPT_HOME."""
    h = os.environ.get(ENV_HOME)
    if h:
        return os.path.abspath(h)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tree_home() -> str:
    """esmfold2/ — the model's directory ($MODEL_OPT when set, else the parent of opt_home())."""
    h = os.environ.get(ENV_TREE)
    return os.path.abspath(h) if h else os.path.dirname(opt_home())


def kit_home() -> str:
    """The kit directory: the one under opt/<class>/ that carries driver/ef2_server.py ($ESMFOLD2_OPT_KIT overrides)."""
    k = os.environ.get(ENV_KIT)
    if k:
        if not os.path.isfile(os.path.join(k, SERVER_RELPATH)):
            raise FileNotFoundError(f"{ENV_KIT}={k}: no {SERVER_RELPATH} there")
        return os.path.abspath(k)
    hits = []
    for cls in KIT_CLASS_DIRS:
        hits += sorted(glob.glob(os.path.join(opt_home(), cls, "*", SERVER_RELPATH)))
    if not hits:
        raise FileNotFoundError(f"kit not found: no opt/{{{','.join(KIT_CLASS_DIRS)}}}/*/{SERVER_RELPATH} under {opt_home()} "
                                f"(install the package editable from esmfold2/opt, or set {ENV_KIT})")
    return os.path.dirname(os.path.dirname(hits[0]))


def driver_dirs(kit: Optional[str] = None) -> List[str]:
    """sys.path entries for the kit modules: the kit's own driver/ first (the server inserts it itself at import), then every other
    opt/<class>/<kit>/driver in the tree (levers carried beside the kit), deduplicated."""
    kit = kit or kit_home()
    out = [os.path.join(kit, DRIVER_DIRNAME)]
    for cls in KIT_CLASS_DIRS:
        for d in sorted(glob.glob(os.path.join(opt_home(), cls, "*", DRIVER_DIRNAME))):
            if os.path.isdir(d) and os.path.abspath(d) not in [os.path.abspath(x) for x in out]:
                out.append(d)
    return out


def kit_sys_path(kit: Optional[str] = None) -> List[str]:
    return [os.path.abspath(d) for d in driver_dirs(kit)]




def _install_sys_path(entries: List[str]) -> None:
    from opt_core.home import place_on_sys_path
    place_on_sys_path(entries)


def pins_path() -> str:
    return os.path.join(tree_home(), PINS_RELPATH)


def pins() -> dict:
    with open(pins_path(), "r", encoding="utf-8") as fh:
        return json.load(fh)


def variant_repo(variant: str, p: Optional[dict] = None) -> str:
    """The variant's checkpoint repository id (stock/PINS.json variants.<variant>.hf_repo)."""
    p = p if p is not None else pins()
    v = (p.get("variants") or {}).get(variant) or {}
    repo = v.get("hf_repo")
    if not repo:
        raise ValueError(f"stock/PINS.json has no variants.{variant}.hf_repo")
    return repo


def stock_env_absent(p: Optional[dict] = None) -> Tuple[str, ...]:
    """Variables that must be absent from a stock process (stock/PINS.json stock_environment.must_be_absent_prefixes), plus the
    package's own switches."""
    p = p if p is not None else pins()
    spec = list((p.get("stock_environment") or {}).get("must_be_absent_prefixes") or ["EF2_"])
    for own in PACKAGE_ENV:
        if own not in spec:
            spec.append(own)
    return tuple(spec)


# ----------------------------------------------------------------------------------------------------------------- versions, GPU
def version_gate(p: dict, force: bool = False) -> Tuple[bool, Dict[str, Optional[str]], Optional[str]]:
    """Installed esm / transformers distribution versions against stock/PINS.json upstream.<name>.version (the commit is the pin;
    the version is what a process can check without network). Missing -> refuse; different -> refuse unless ESMFOLD2_OPT_FORCE=1."""
    have = upstream_versions()
    want = {n: ((p.get("upstream") or {}).get(n) or {}).get("version") for n in UPSTREAM_DISTS}
    missing = [n for n in UPSTREAM_DISTS if have[n] is None]
    if missing:
        return False, have, f"upstream package(s) not installed: {', '.join(missing)} (the levers patch the installed model; install the stock pins first)"
    diff = [f"{n} {have[n]} != pin {want[n]}" for n in UPSTREAM_DISTS if want[n] and have[n] != want[n]]
    if diff and not force:
        return False, have, "upstream version differs from stock/PINS.json: " + "; ".join(diff) + f" ({ENV_FORCE}=1 overrides)"
    if diff:
        sys.stderr.write(f"{LOG} WARNING: {'; '.join(diff)}; {ENV_FORCE}=1 set, continuing\n")
    return True, have, None


def _check_pins_module():
    import importlib.util
    path = os.path.join(tree_home(), CHECK_PINS_RELPATH)
    spec = importlib.util.spec_from_file_location("esmfold2_check_pins", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                                                           # type: ignore[union-attr]
    return mod


def pins_gate(p: dict, force: bool = False) -> Tuple[bool, Dict[str, dict], Optional[str]]:
    """The commit-level pin check (stock/check_pins.py: pip's direct_url.json commit, or the stock/ archive's own path) for both upstream
    forks. Not at the pin -> refuse unless ESMFOLD2_OPT_FORCE=1 (then a warning). Returns (ok, detail per fork, reason)."""
    try:
        cp = _check_pins_module()
        bad, detail = cp.check(p.get("upstream") or {})
    except Exception as e:  # noqa: BLE001
        return False, {}, f"the pin check could not run ({CHECK_PINS_RELPATH}: {e!r})"
    if bad and not force:
        return False, detail, "upstream not at the pinned commit: " + "; ".join(bad) + f" ({ENV_FORCE}=1 overrides)"
    if bad:
        sys.stderr.write(f"{LOG} WARNING: {'; '.join(bad)}; {ENV_FORCE}=1 set, continuing\n")
    return True, detail, None


GPU_KEYS = ("name", "cc", "sm", "probe")                          # the GPU identification (report, manifest): the core's probe projected to these


def nvidia_smi_probe() -> dict:
    """The GPU without torch: name and compute capability of GPU 0 from nvidia-smi (None when unavailable) — opt_core.gates."""
    from opt_core.gates import nvidia_smi_probe as probe
    return probe(keys=GPU_KEYS)


def gpu_probe_record() -> dict:
    return nvidia_smi_probe()


def class_cc(base: dict) -> Optional[str]:
    """The compute-capability class a set is resolved for: the visible GPU's (the probe record ``base["gpu"]["cc"]``), else the class the
    configuration targets (MODEL_OPT_TARGET_GPU through modes.target_cc), else None (nothing subtracted by class; the plan names every lever)."""
    from .modes import target_cc
    g = base.get("gpu") or {}
    cc = g.get("cc")
    if cc:
        return str(cc)
    return target_cc(base.get("target_gpu") or os.environ.get("MODEL_OPT_TARGET_GPU"))


def torch_gpu_info() -> dict:
    from opt_core.gates import torch_gpu_probe
    return torch_gpu_probe(keys=GPU_KEYS)


def torch_version() -> Optional[str]:
    t = sys.modules.get("torch")
    return getattr(t, "__version__", None) if t is not None else dist_version("torch")


def triton_version() -> Optional[str]:
    return dist_version("triton")


def kernel_key(cc: Optional[str]) -> str:
    """``<cc>|<triton major.minor>`` — the (capability, Triton) key of this process, reported on the DRY-RUN line; ``?`` when unknown."""
    tv = triton_version()
    tmm = ".".join(tv.split(".")[:2]) if tv else "?"
    return f"{cc or '?'}{KEY_SEP}{tmm}"


def stack_key(cc: Optional[str]) -> str:
    """``torch<ver>-sm<cc>-triton<ver>``: the running stack, for status() and cache keys."""
    return f"torch{torch_version() or '?'}-{('sm' + cc.replace('.', '')) if cc else 'sm?'}-triton{triton_version() or '?'}"


def weights_status(variant: Optional[str], p: Optional[dict] = None) -> dict:
    """Where upstream will look for the checkpoint: HF_HOME and whether the variant's snapshot directory exists there (reported only)."""
    hf_home = os.environ.get("HF_HOME")
    out = {"HF_HOME": hf_home, "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"), "ESMCFOLD_CCD_PATH": os.environ.get("ESMCFOLD_CCD_PATH")}
    if variant:
        try:
            repo = variant_repo(variant, p)
        except Exception as e:  # noqa: BLE001
            out["repo"] = None; out["repo_error"] = repr(e); return out
        out["repo"] = repo
        if hf_home:
            d = os.path.join(hf_home, "hub", "models--" + repo.replace("/", "--"))
            out["snapshot_dir"] = d; out["snapshot_present"] = os.path.isdir(d)
    if hf_home and os.path.isdir(hf_home):                                                   # the weights word on the ACTIVE line: pinned sha256 | unknown sha256 (proceeding) | absent
        try:
            seen = _WEIGHTS_SEEN.get((hf_home, variant))
            absent, unknown = (seen["absent"], seen["unknown"]) if seen is not None else weight_files_check(hf_home, p if p is not None else pins(), variant)
            seen = _WEIGHTS_SEEN.get((hf_home, variant)) or {}
            out["word"] = WEIGHTS_ABSENT if absent else (WEIGHTS_UNKNOWN if unknown else WEIGHTS_PINNED)
            out["sha256"] = {os.path.basename(r): d for r, d in (seen.get("digests") or {}).items()}
            out["digest_source"] = {os.path.basename(r): ("digested now" if u is None else f"cached digest {u}") for r, u in sorted((seen.get("cached") or {}).items())}
            out["via"] = seen.get("via", "this process"); out["memo"] = os.path.join(seen.get("memo_dir") or weights_memo_dir(), "weights_digests.json")
        except Exception as e:  # noqa: BLE001  (reported only)
            out["word"] = f"unread ({e!r})"
    return out


# ----------------------------------------------------------------------------------------------------------------- reports
# --------------------------------------------------------------------------------------------------------- model instances
def _model_class() -> Optional[type]:
    """Upstream's ``ESMFold2Model`` class when its module is imported (never imported by the package itself), else None."""
    mod = sys.modules.get(MODEL_MODULE)
    cls = getattr(mod, MODEL_CLASS, None) if mod is not None else None
    return cls if isinstance(cls, type) else None


def _count_instances_of(cls: type, seed_from_gc: bool = False) -> bool:
    """Wrap ``cls.__new__`` so every instance created from now on is counted (a total and a weak set of the live ones: ``__new__``
    sees every creation — ``from_pretrained``, a direct call, ``copy.deepcopy``, unpickling). One wrap per class object; the original
    ``__new__`` is called unchanged and the class signature is kept. With ``seed_from_gc`` the instances that already exist are
    added once by a gc scan (the class was imported before the counter). Returns False when the class cannot be wrapped (the
    counter then stays on the gc scan)."""
    st = _INSTANCES["state"]
    if st is not None and st["cls"] is cls:
        return st["wrapped"]
    live: "weakref.WeakSet" = weakref.WeakSet()
    state = {"cls": cls, "wrapped": False, "live": live, "built": 0, "weakref": True, "gc_seeded": None}
    _INSTANCES["state"] = state
    orig_new = cls.__new__

    @functools.wraps(orig_new)
    def __new__(klass, *args, **kwargs):
        obj = orig_new(klass) if orig_new is object.__new__ else orig_new(klass, *args, **kwargs)
        state["built"] += 1
        try:
            live.add(obj)
        except TypeError:                                                  # no weak references (__slots__ without __weakref__): total only
            state["weakref"] = False
        return obj

    if "__new__" not in cls.__dict__:
        try:
            __new__.__signature__ = inspect.signature(cls.__init__)        # inspect.signature(cls) keeps reporting __init__'s parameters
        except (TypeError, ValueError):
            pass
    try:
        cls.__new__ = staticmethod(__new__)
    except (TypeError, AttributeError):
        state["live"] = None
        return False
    if seed_from_gc:
        import gc
        n = 0
        for o in gc.get_objects():
            if isinstance(o, cls):
                n += 1
                try:
                    live.add(o)
                except TypeError:
                    state["weakref"] = False
        state["gc_seeded"] = n
    state["wrapped"] = True
    return True


class _ModelClassFinder:
    """Meta-path finder that wraps the model class's constructor right after its module body has run (the package imports nothing:
    the program's import order stands). Fires once and removes itself."""

    def __init__(self):
        self.armed = True

    def find_spec(self, fullname, path=None, target=None):
        if not self.armed or fullname != MODEL_MODULE:
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
            _orig(module)                                                  # the module body first, then the wrap on the class
            self.remove()
            cls = getattr(module, MODEL_CLASS, None)
            if isinstance(cls, type):
                _count_instances_of(cls)
        spec.loader.exec_module = exec_module
        return spec

    def remove(self) -> None:
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass


def register_instance_counter() -> dict:
    """Count model instances from now on by the constructor wrap: on the class now when its module is imported (the instances
    that already exist are found once by a gc scan), otherwise at the module's import through a meta-path finder. Idempotent.
    Returns :func:`instance_check`."""
    cls = _model_class()
    if cls is not None:
        _count_instances_of(cls, seed_from_gc=True)
    elif _INSTANCES["finder"] is None or _INSTANCES["finder"] not in sys.meta_path:
        f = _ModelClassFinder()
        sys.meta_path.insert(0, f)
        _INSTANCES["finder"] = f
    return instance_check()


def instance_check() -> dict:
    """How many upstream ``ESMFold2Model`` objects exist in this process and how that is known:
    ``{"n", "method": "counted" | "gc" | "none", "built"}`` — ``counted`` = the constructor wrap (deterministic), ``gc`` = a scan of
    the garbage collector's objects (the class could not be wrapped, or its instances take no weak references), ``none`` = the model
    module is not imported (0 instances). The late-activation rule: importing upstream before ``enable()`` is fine; constructing a
    model before it is not."""
    cls = _model_class()
    if cls is None:
        return {"n": 0, "method": "none", "built": 0}
    st = _INSTANCES["state"]
    same = st is not None and st["cls"] is cls
    if same and st["wrapped"] and st["weakref"]:
        return {"n": len(st["live"]), "method": "counted", "built": st["built"]}
    import gc
    return {"n": sum(1 for o in gc.get_objects() if isinstance(o, cls)), "method": "gc", "built": st["built"] if same else 0}


def model_instances() -> int:
    """The live ``ESMFold2Model`` instance count (:func:`instance_check`)."""
    return instance_check()["n"]


KIT_LEVER_MODULES = ("ef2_w4", "ef2_msa", "ef2_mk_sampler", "ef2_opt")                    # the kit's own lever modules (driver/)


def kit_levers_applied() -> List[str]:
    """Which of the kit's lever modules, if already imported in this process, report a lever ON: ``ef2_w4._STATE['enabled']``,
    ``ef2_msa._STATE['enabled']``, ``ef2_mk_sampler._ENABLED`` (non-empty), ``ef2_opt._MODE['models']`` (non-empty) — read, never set."""
    out = []
    for name in KIT_LEVER_MODULES:
        m = sys.modules.get(name)
        if m is None:
            continue
        st = getattr(m, "_STATE", None)
        if isinstance(st, dict) and st.get("enabled"):
            out.append(name)
        elif name == "ef2_mk_sampler" and getattr(m, "_ENABLED", None):
            out.append(name)
        elif name == "ef2_opt" and isinstance(getattr(m, "_MODE", None), dict) and m._MODE.get("models"):
            out.append(name)
    return out


MK_MODULE = "ef2_mk_sampler"                                            # the kit's sampler hoist; its own counters (STATS) say what ran
MK_GUARD_NOTE = "installed; inactive at num_diffusion_samples>1 (kit guard)"


def mk_plan_note(levers_planned, num_diffusion_samples) -> Optional[str]:
    """At plan time, from the settings: the MK hoist serves batch-1 folds only (the kit's own guard falls back to the stock forward
    for num_diffusion_samples > 1), so a plan that includes ``mk`` says so before anything runs."""
    if "mk" not in (levers_planned or ()):
        return None
    n = int(num_diffusion_samples or 1)
    return f"mk: {MK_GUARD_NOTE}" if n > 1 else "mk: installed; active at num_diffusion_samples=1"


def mk_after_run() -> Optional[dict]:
    """After a run: the sampler's own counters (``ef2_mk_sampler.STATS``). ``active`` is True only when a fold went through the hoisted
    path (``folds_seen``); ``fallback_batch_or_args`` counts the steps the kit guard sent back to the stock forward. None when the kit's
    sampler module was never imported."""
    mod = sys.modules.get(MK_MODULE)
    stats = getattr(mod, "STATS", None)
    if stats is None:
        return None
    fb, seen, side = int(stats.get("fallback_batch_or_args", 0)), int(stats.get("folds_seen", 0)), int(stats.get("side_steps", 0))
    active = seen > 0
    note = ("hoisted folds ran" if active else
            (f"every step fell back to the stock forward (batch_or_args x{fb}; the kit guard: num_diffusion_samples>1 or batch>1) — mk was not applied" if fb
             else "no fold reached the sampler"))
    return {"fallback_batch_or_args": fb, "folds_seen": seen, "side_steps": side, "active": active, "note": note}


GUARDED_LEVERS = ("ro", "kd", "dit")                                    # ef2_dit's levers: the same declared scope as mk (num_diffusion_samples == 1, batch 1)


def guards_plan_note(levers_planned, num_diffusion_samples) -> Optional[str]:
    """At plan time: ef2_dit's roll-out / device Kabsch / fused step serve sample() calls at num_diffusion_samples == 1 (batch 1); at another
    count every call runs the previously installed chain by name (counted) — said before anything runs, like mk's note."""
    on = [n for n in GUARDED_LEVERS if n in (levers_planned or ())]
    if not on:
        return None
    n = int(num_diffusion_samples or 1)
    scope = "active at num_diffusion_samples=1" if n == 1 else f"installed; inactive at num_diffusion_samples={n} (declared scope: the previous sampler chain runs every call, counted)"
    return f"{','.join(on)}: {scope}"


def guards_after_run() -> dict:
    """After a run, the data-dependent guards of the lever modules from their own counters: {lever: (kind, note)} with kind 'gated' (a declared,
    counted guard took the stock / previous path on some calls: named, exit unchanged), 'inactive' (installed, never engaged, every call gated:
    like mk's batch guard) or 'unreached' (no call reached the lever: no evidence of application). Modules not loaded contribute nothing."""
    out = {}
    dit = sys.modules.get("ef2_dit")
    if dit is not None and callable(getattr(dit, "applied_report", None)):
        r = dit.applied_report()
        folds, fb = int(r.get("rollout_folds", 0)), int(r.get("rollout_fallback", 0))
        if r.get("rollout_installed"):
            if folds == 0 and fb > 0:
                out["ro"] = ("inactive", f"every sample() call ran the previous chain ({fb} call(s): {','.join(r.get('fallback_reasons') or [])}) — outside the declared scope")
            elif folds == 0:
                out["ro"] = ("unreached", "no fold reached the sampler")
            elif fb > 0:
                out["ro"] = ("gated", f"{fb} sample() call(s) outside the declared scope ran the previous chain ({','.join(r.get('fallback_reasons') or [])}); {folds} fold(s) rolled out")
        if r.get("dit_installed"):
            steps, dfb = int(r.get("dit_steps", 0)), int(r.get("dit_fallback", 0))
            if steps == 0 and dfb > 0:
                out["dit"] = ("inactive", f"every diffusion step ran the previous forward ({dfb} step(s): {','.join(r.get('dit_fallback_reasons') or [])})")
            elif steps == 0 and folds == 0:
                out["dit"] = ("unreached", "no step reached the fused forward")
            elif dfb > 0:
                out["dit"] = ("gated", f"{dfb} step(s) outside the declared scope ran the previous forward ({','.join(r.get('dit_fallback_reasons') or [])}); {steps} fused step(s)")
    hoist = sys.modules.get("ef2_hoist")
    if hoist is not None:
        s = hoist.stats()
        for lever, key in (("trimul", "trimul_fallthrough"), ("glue", "glue_fallthrough")):
            if int(s.get(key, 0) or 0) > 0 and (s.get("levers") or {}).get(lever):
                out[lever] = ("gated", f"{int(s[key])} call(s) took the stock statements ({key}); {int(s.get(lever + '_calls', 0) or 0)} served")
    atom = sys.modules.get("ef2_atom")
    if atom is not None:
        s = dict(getattr(atom, "STATS", {}) or {})
        n = int(s.get("a5_layout_fallbacks", 0) or 0) + int(s.get("a5_nonprefix_folds", 0) or 0) + int(s.get("a6_unsorted_folds", 0) or 0)
        if n > 0:
            out["af"] = ("gated", f"atom layouts outside the fused kernels' scope on {n} fold(s)/call(s) ran the unfused path of the same tier "
                               f"(a5_layout_fallbacks={int(s.get('a5_layout_fallbacks', 0) or 0)} a5_nonprefix_folds={int(s.get('a5_nonprefix_folds', 0) or 0)} a6_unsorted_folds={int(s.get('a6_unsorted_folds', 0) or 0)})")
        if int(s.get("a2_nonprefix_folds", 0) or 0) > 0:
            out["ax"] = ("gated", f"a2 prefix var-len: {int(s['a2_nonprefix_folds'])} fold(s) with a non-prefix atom layout used the stock gather (same values)")
    msa2 = sys.modules.get("ef2_msa_v2")
    if msa2 is not None:
        s = dict(getattr(msa2, "STATS", {}) or {})
        for lever in ("m15", "m16", "m17"):
            if int(s.get(lever + "_fallthrough", 0) or 0) > 0:
                out[lever] = ("gated", f"{int(s[lever + '_fallthrough'])} call(s) took the stock statements; {int(s.get(lever + '_calls', 0) or 0)} served")
    cute = sys.modules.get("ef2_transition_cute")                      # t16: the kernel's own call counters are the evidence of application
    if cute is not None and (cute.levers_on() or {}).get("t16"):
        cs = dict(getattr(cute, "STATS", {}) or {})
        served = int(cs.get("transition_calls", 0) or 0) + int(cs.get("pair_transition_calls", 0) or 0)
        fell = int(cs.get("fallthrough_transition", 0) or 0) + int(cs.get("fallthrough_pair_transition", 0) or 0)
        if served == 0:
            out["t16"] = ("unreached", f"no Transition / PairTransition call reached the t16 kernel (transition_calls + pair_transition_calls = 0; fallthrough={fell})")
        elif fell > 0:
            out["t16"] = ("gated", f"{fell} call(s) outside the kernel's declared scope took the previous statements "
                                   f"(fallthrough_transition={int(cs.get('fallthrough_transition', 0) or 0)} fallthrough_pair_transition={int(cs.get('fallthrough_pair_transition', 0) or 0)}); {served} served")
    pair = sys.modules.get("ef2_pair_v2")
    if pair is not None:
        sc = (getattr(pair, "_STATE", {}) or {}).get("t6s_selfcheck") or {}
        if sc and not sc.get("identical", True):
            out["t6s"] = ("inactive", f"bitwise self-check failed ({sc.get('word')}, {sc.get('origin')}): disengaged by name, the exact transition kept the stock statistics kernel")
        if _report.N_GPU["P"] > 1 and (getattr(pair, "_STATE", {}) or {}).get("levers", {}).get("t15msa"):   # on the row-sharded route the MSA block forward is re-issued on
            ps_ = dict(getattr(pair, "STATS", {}) or {})                                                     # rows and calls the pair-transition module directly: the KERNEL
            if int(ps_.get("t15_msa_pair_transition_calls", 0) or 0) == 0:                                    # counter is the evidence of application there (the block wrapper's
                out["t15msa"] = ("unreached", "no MSA pair-transition call reached the t15 kernel on this route (t15_msa_pair_transition_calls=0)")   # own counter stays 0 on rows)
    rci = sys.modules.get("esmfold2_opt.rowchunk.install")                        # the multi-GPU route's row-chunking levers: a bound member whose engagement counter is 0
    if rci is not None and callable(getattr(rci, "engagement", None)):           # after a fold at or above the token floor never ran its statement: unreached (partial)
        for lever, kn in (rci.engagement() or {}).items():
            out[lever] = (str(kn[0]), str(kn[1]))
    return out


def settle_guards(rep: dict) -> dict:
    """After a run: apply guards_after_run() to the report — 'gated' levers stay applied with a ``gated`` note; 'inactive' levers leave
    ``levers_applied`` for ``gated`` + ``levers_gated`` (a declared guard, named; not ``partial``); 'unreached' levers are ``partial`` (no evidence of
    application), exactly as settle_mk treats mk."""
    global _REPORT
    g = guards_after_run()
    if not g or rep is None:
        return rep
    out = dict(rep, guards={k: {"kind": v[0], "note": v[1]} for k, v in g.items()})
    applied = list(out.get("levers_applied") or [])
    for lever, (kind, note) in g.items():
        if kind == "gated":
            out["gated"] = list(out.get("gated") or []) + [f"{lever}: {note}"]
        elif kind == "inactive" and lever in applied:
            applied = [n for n in applied if n != lever]
            out["levers_gated"] = sorted(set(out.get("levers_gated") or []) | {lever})
            out["gated"] = list(out.get("gated") or []) + [f"{lever}: {note}"]
        elif kind == "unreached" and lever in applied:
            applied = [n for n in applied if n != lever]
            out["levers_fallback"] = list(out.get("levers_fallback") or []) + [lever]
            reasons = dict(out.get("fallback_reasons") or {}); reasons[lever] = note
            out["fallback_reasons"] = reasons
            out["partial"] = sorted(set(out.get("partial") or []) | {lever})
    out["levers_applied"] = applied
    if _REPORT is not None and rep is _REPORT:
        _REPORT = out
    return out


def settle_mk(rep: dict, replaced_by: Optional[str] = None) -> dict:
    """Never report ``mk`` as applied when every step fell back: after a run, move it from ``levers_applied`` to ``levers_fallback``
    (with the reason) when the sampler's counters show no hoisted fold; the counters ride in the report under ``mk``. The kit guard's
    own fallback (``fallback_batch_or_args`` > 0: num_diffusion_samples > 1 or batch > 1, the declared precondition of the hoist) is a
    documented gate — recorded in ``gated``, not in ``partial``; a sampler no fold reached is ``partial`` (no evidence of application),
    unless ``replaced_by`` names the forward that took its place by design (the n_gpu > 1 row-sharded diffusion forward,
    ``rowpair.KIT_LEVERS_REPLACED``): then ``mk`` is recorded in ``levers_replaced`` + ``gated`` with that name."""
    global _REPORT
    mk = mk_after_run()
    if mk is None or rep is None:
        return rep
    out = dict(rep, mk=mk)
    if "mk" in (out.get("levers_applied") or []) and not mk["active"]:
        out["levers_applied"] = [n for n in out["levers_applied"] if n != "mk"]
        if replaced_by and not mk["fallback_batch_or_args"] and not mk["folds_seen"]:
            out["levers_replaced"] = sorted(set(out.get("levers_replaced") or []) | {"mk"})
            out["gated"] = list(out.get("gated") or []) + [f"mk: replaced by {replaced_by}"]
            if _REPORT is not None and rep is _REPORT:
                _REPORT = out
            return out
        out["levers_fallback"] = list(out.get("levers_fallback") or []) + ["mk"]
        reasons = dict(out.get("fallback_reasons") or {}); reasons["mk"] = mk["note"]
        out["fallback_reasons"] = reasons
        if mk["fallback_batch_or_args"]:
            out["gated"] = list(out.get("gated") or []) + [f"mk: {mk['note']}"]
        else:
            out["partial"] = sorted(set(out.get("partial") or []) | {"mk"})
    if _REPORT is not None and rep is _REPORT:
        _REPORT = out
    return out


def status() -> dict:
    if _REPORT is None:
        return {"active": False, "reason": "esmfold2_opt.enable() has not run in this process"}
    out = dict(_REPORT, seed_guard=dict(SEED_GUARD))
    out.update(_big.report())                                                # the memory line's state (xl) from the engine adapter
    return out


def _base(mode: str, variant: Optional[str], trigger: Optional[str] = None) -> dict:
    return {"active": False, "mode": mode, "variant": variant, "server_mode": None, "levers_applied": [], "levers_fallback": [],
            "gpu": {"name": None, "sm": None}, "package_version": __version__, "trigger": trigger,
            "upstream": upstream_versions(), "target_gpu": os.environ.get("MODEL_OPT_TARGET_GPU") or None, "models": []}


def _effective_variant(variant: Optional[str]) -> Optional[str]:
    """The variant: the argument, else $ESMFOLD2_VARIANT; both set and different -> ValueError (the flag never silently wins)."""
    env_v = (os.environ.get(ENV_VARIANT) or "").strip().lower() or None
    v = check_variant(variant)
    env_v = check_variant(env_v)
    if v and env_v and v != env_v:
        raise ValueError(f"--variant {v} disagrees with {ENV_VARIANT}={env_v}; unset one")
    return v or env_v


def _path_gates(base: dict, variant: Optional[str]) -> Tuple[Optional[str], Optional[dict]]:
    """Hard gates that also bind a dry run: kit present, pins present, a variant named. Returns (refusal reason or None, pins)."""
    try:
        kit = kit_home()
    except FileNotFoundError as e:
        return str(e), None
    base["kit_home"] = kit
    if not os.path.isfile(pins_path()):
        return f"stock pins not found at {pins_path()}", None
    p = pins()
    if variant is None:
        return f"a variant is required for mode {base['mode']!r}: --variant {'|'.join(VARIANTS)} or {ENV_VARIANT}", p
    return None, p


def pinned_weight_files(pins: dict, variant: Optional[str]) -> List[Tuple[str, str, int]]:
    """(repo, relative path under HF_HOME, size_bytes) of every pinned weights file the variant loads: the language model's repo
    (every variant) and the variant's own hf_repo (stock/PINS.json ``weights`` / ``variants``); every repo when the variant is unknown."""
    repos = set(pins["weights"])
    if variant and variant in pins["variants"]:
        repos = {pins["variants"][variant]["hf_repo"]} | {r for r in pins["weights"] if "ESMC" in r}
    out = []
    for repo in sorted(repos):
        w = pins["weights"][repo]
        for name, d in sorted(w["files"].items()):
            out.append((repo, os.path.join("hub", "models--" + repo.replace("/", "--"), "snapshots", w["snapshot_commit"], name), int(d["size_bytes"])))
    return out


def pinned_weight_digests(pins: dict, variant: Optional[str]) -> Dict[str, str]:
    """relative path under HF_HOME -> pinned sha256 (stock/PINS.json ``weights``) for every file `pinned_weight_files` names."""
    out = {}
    for repo, rel, _size in pinned_weight_files(pins, variant):
        out[rel] = pins["weights"][repo]["files"][os.path.basename(rel)]["sha256"]
    return out


def file_sha256(path: str, chunk: int = 8 << 20) -> str:
    """Streamed sha256 of a file (every byte read; the digest memo below records the result only after this returns)."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ---- the digest memo (``digest_memo.py``, the release tree's one helper text): <memo dir>/weights_digests.json, keyed by a file's (realpath, st_size,
# st_mtime_ns, st_ino) -> {sha256, utc}; an entry is written only after a full sha256 of that file completed (tmp + os.replace).
# Sameness is by digest: size, mtime and inode only select the memo entry; they never decide it on their own.
ENV_WEIGHTS_MEMO_DIR = "ESMFOLD2_OPT_WEIGHTS_MEMO_DIR"
_WEIGHTS_AFRESH = {"on": False}                                      # `check` digests afresh and rewrites the memo entries


def set_weights_afresh(on: bool = True) -> None:
    _WEIGHTS_AFRESH["on"] = bool(on)


def weights_memo_dir(env=None) -> str:
    """The digest memo directory: ``$ESMFOLD2_OPT_WEIGHTS_MEMO_DIR`` (configs/h100.env: ``$MODEL_OPT_JIT_ROOT/weights`` when that root is set), else the parent of
    ``$TRITON_CACHE_DIR``, else ``~/.cache/esmfold2_opt`` — a directory of this user's own (the same in every process of the user, so rank processes find
    the launching process's entries; never a fixed name in the shared temporary directory, where another account could plant a memo that vouches for a
    weights file)."""
    env = os.environ if env is None else env
    d = env.get(ENV_WEIGHTS_MEMO_DIR)
    if d:
        return d
    t = (env.get("TRITON_CACHE_DIR") or "").rstrip("/")
    return os.path.dirname(t) if t else os.path.join(os.path.expanduser("~"), ".cache", "esmfold2_opt")


def in_rank_process(env=None) -> bool:
    """True inside a rank process of the row-sharded launcher (opt_core.mem.rowpair.launch sets ROWPAIR_RANK in every worker, never at P = 1)."""
    env = os.environ if env is None else env
    return bool(str(env.get("ROWPAIR_RANK", "") or "").strip())


class WeightsMemoMiss(RuntimeError):
    """A rank process found no memo entry the launching process should have written (digest_memo.rank_digest): refused by name, never re-hashed."""


def weights_memo_writable(memo_dir: str) -> Tuple[bool, Optional[str]]:
    """(writable, reason): can this process create the memo dir and write a file in it? A read-only cache mount (the memo's usual home) is
    not an error — the digests are computed afresh and not memoised, with one named line (never a refusal)."""
    probe = os.path.join(memo_dir, f".weights_digests.probe.{os.getpid()}")
    try:
        os.makedirs(memo_dir, exist_ok=True)
        with open(probe, "w") as fh:
            fh.write("probe")
        os.remove(probe)
        return True, None
    except OSError as e:
        return False, f"{type(e).__name__}: {e}"


_WEIGHTS_SEEN: Dict[Tuple[str, Optional[str]], dict] = {}           # (HF_HOME, variant) -> the last weights record of this process (the report reuses it)


def weight_files_check(hf: str, pins: dict, variant: Optional[str], *, afresh: Optional[bool] = None, memo_dir: Optional[str] = None,
                       env=None) -> Tuple[List[str], List[str]]:
    """(absent, unknown) for the variant's pinned weights files under ``hf``: absent = not there; unknown = present with a sha256 that is
    not the pinned digest (stock/PINS.json ``weights``) — reported by name, never refused. Every present file's digest comes from a full
    sha256 of its bytes through ``digest_memo``: computed now, or the memo entry a previous full digest of THIS file wrote (selected by realpath,
    size, mtime_ns and inode — the selectors never decide sameness); ``afresh`` (``check``) digests every file now and rewrites its entries; a
    rank process of the row-sharded launcher reads the launching process's entries (``rank_digest``) and never hashes — a miss raises
    :class:`WeightsMemoMiss`. The byte count is printed beside a differing digest; it never decides sameness."""
    import time
    from . import digest_memo
    env = os.environ if env is None else env
    afresh = _WEIGHTS_AFRESH["on"] if afresh is None else bool(afresh)
    memo_dir = memo_dir or weights_memo_dir(env); rank = in_rank_process(env)
    files = pinned_weight_files(pins, variant); pinned = pinned_weight_digests(pins, variant)
    absent = [rel for _repo, rel, _size in files if not os.path.isfile(os.path.join(hf, rel))]
    present = [(rel, size) for _repo, rel, size in files if rel not in absent]
    digests: Dict[str, str] = {}; cached: Dict[str, Optional[str]] = {}; hashed_bytes = 0; t0 = time.perf_counter(); misses = []
    writable, ro_reason = (True, None) if rank else weights_memo_writable(memo_dir)      # a read-only memo dir (e.g. a cache directory mounted read-only): named, digested afresh, not memoised
    for rel, _size in present:
        path = os.path.join(hf, rel)
        if rank:
            try:
                digests[rel], cached[rel] = digest_memo.rank_digest(path, memo_dir)
            except RuntimeError as e:
                misses.append(str(e))
            continue
        if writable:
            digests[rel], cached[rel] = digest_memo.digest(path, memo_dir, refresh=afresh)
        else:                                                                                 # no write possible: a memo hit still serves (unless afresh); a miss is a full digest kept in this process only
            hit = None
            if not afresh:
                try:
                    hit = digest_memo.rank_digest(path, memo_dir)                             # the helper's read-only lookup (raises on a miss)
                except RuntimeError:
                    hit = None
            digests[rel], cached[rel] = hit if hit is not None else (digest_memo.sha256_file(path), None)
        if cached[rel] is None:
            hashed_bytes += os.path.getsize(path)
    if misses:
        raise WeightsMemoMiss(f"{len(misses)} weights file(s) have no digest memo entry in {memo_dir} — " + misses[0])
    dt = time.perf_counter() - t0
    unknown = []
    for rel, size in present:
        if digests[rel] != pinned[rel]:
            got = os.path.getsize(os.path.join(hf, rel))
            unknown.append(digest_memo.word(f"{rel} sha256={digests[rel][:12]} (pinned {pinned[rel][:12]})" + (f", {got} bytes (pinned {size})" if got != size else ""), cached[rel]))
    _WEIGHTS_SEEN[(hf, variant)] = {"absent": absent, "unknown": unknown, "digests": digests, "cached": cached, "dt": dt, "hashed_bytes": hashed_bytes,
                                    "memo_dir": memo_dir, "afresh": afresh, "via": "launcher memo" if rank else "this process",
                                    "memo_readonly": None if writable else ro_reason}
    return absent, unknown


def weights_memo_relay_for_ranks(hf: str, variant: Optional[str]) -> Optional[str]:
    """The launching process of a `--n_gpu P>1` run, after its own weights gate: when the configured memo dir was NOT writable, the digests it just
    computed are written to a process-private writable memo dir (``digest_memo.digest`` with the known digest as the hasher — nothing is re-read) and
    that dir is exported through :data:`ENV_WEIGHTS_MEMO_DIR`, so the rank processes' ``rank_digest`` finds the entries and still never hashes.
    Returns the relay dir (and prints one named line) or None when the configured dir held the entries already."""
    import tempfile
    from . import digest_memo
    rec = _WEIGHTS_SEEN.get((hf, variant)) or {}
    if not rec.get("memo_readonly"):
        return None
    relay = tempfile.mkdtemp(prefix="esmfold2_opt_weights_memo_")
    for rel, hexdigest in (rec.get("digests") or {}).items():
        digest_memo.digest(os.path.join(hf, rel), relay, refresh=True, hasher=lambda _p, _d=hexdigest: _d)
    os.environ[ENV_WEIGHTS_MEMO_DIR] = relay
    sys.stderr.write(f"{LOG} WEIGHTS digest memo relayed to the rank processes through {relay}/{digest_memo.MEMO_NAME} "
                     f"({len(rec.get('digests') or {})} entries; the configured {rec.get('memo_dir')} is not writable)\n")
    return relay


WEIGHTS_PINNED, WEIGHTS_UNKNOWN, WEIGHTS_ABSENT = "pinned sha256 (the pinned checkpoint)", "unknown sha256 (not a pinned checkpoint); proceeding", "absent"


def weights_unknown_words(unknown: List[str]) -> str:
    return (f"WEIGHTS unknown: {len(unknown)} file(s) under HF_HOME whose sha256 is not the pinned checkpoint's (stock/PINS.json \"weights\") — "
            f"not a pinned checkpoint; proceeding: " + "; ".join(unknown[:4]) + (f"; +{len(unknown) - 4} more" if len(unknown) > 4 else ""))


def weights_pinned_words(hf: str, variant: Optional[str]) -> str:
    from . import digest_memo
    rec = _WEIGHTS_SEEN.get((hf, variant)) or {"digests": {}, "cached": {}, "dt": 0.0, "hashed_bytes": 0, "afresh": False, "via": "this process"}
    dig, cached = rec["digests"], rec["cached"]
    hits = [u for u in cached.values() if u is not None]
    how = []
    if hits:
        how.append(digest_memo.word(f"{len(hits)}", max(hits)))              # "<k> (cached digest <utc>)"
    if len(dig) > len(hits) or not dig:
        how.append(f"{len(dig) - len(hits)} digested {rec['hashed_bytes'] / 2**30:.1f} GiB in {rec['dt']:.1f} s" + (" afresh (check)" if rec["afresh"] else ""))
    return (f"WEIGHTS pinned sha256: {len(dig)} file(s) for variant {variant or '<any>'} (" + ", ".join(f"{os.path.basename(r)}={d[:12]}" for r, d in sorted(dig.items())[:4])
            + (f", +{len(dig) - 4} more" if len(dig) > 4 else "") + "); " + "; ".join(how) + (" via=launcher" if rec.get("via") == "launcher memo" else ""))


def data_path_gate(env: Optional[dict] = None, variant: Optional[str] = None, pins: Optional[dict] = None) -> Tuple[Optional[str], List[str]]:
    """The weights root and the CCD pickle (``HF_HOME``: the HuggingFace cache holding the pinned snapshots; ``ESMCFOLD_CCD_PATH``: the CCD
    pickle — stock/PINS.json ``weights`` / ``ccd``). Frozen weights: the run never reads the library's default cache and never downloads
    (the offline pair is exported by run.sh and the command layer), so an UNSET or absent HF_HOME is a refusal by name, and every file
    the variant's snapshots load must be present under it (a missing file would otherwise die later in the hub's offline traceback); every
    present file is digested afresh (sha256, streamed) against stock/PINS.json ``weights``: all pinned → a ``WEIGHTS pinned sha256``
    line; any other digest is an UNKNOWN checkpoint: a named WARNING (``WEIGHTS unknown … sha256=… (pinned …); proceeding``) and the
    run proceeds in every mode — the pin is a note, never a refusal, and the byte count never decides sameness; a set ESMCFOLD_CCD_PATH must exist (unset: the library reads ccd.pkl from the pinned snapshot, one
    of the files checked). Returns (refusal or None, notes)."""
    env = os.environ if env is None else env
    notes: List[str] = []
    hf, ccd = env.get("HF_HOME"), env.get("ESMCFOLD_CCD_PATH")
    if not hf:
        return ("HF_HOME is not set: the weights root (the HuggingFace cache holding the pinned snapshots, stock/PINS.json \"weights\") is required — "
                "nothing is fetched and the library's default cache is never read (HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1); export HF_HOME=<that cache> "
                "or pass run.sh --config <gpu>", notes)
    if not os.path.isdir(hf):
        return (f"HF_HOME={hf} does not exist: set HF_HOME to the HuggingFace cache holding the pinned weights "
                "(stock/PINS.json \"weights\": the pinned snapshots and their sha256)", notes)
    if ccd and not os.path.isfile(ccd):
        return (f"ESMCFOLD_CCD_PATH={ccd} does not exist: set ESMCFOLD_CCD_PATH to the pinned CCD pickle (stock/PINS.json \"ccd\", sha256)", notes)
    p = pins if pins is not None else globals()["pins"]()
    try:
        absent, unknown = weight_files_check(hf, p, variant, env=env)
    except WeightsMemoMiss as e:                                                            # a rank process never hashes: the launching process digests before the ranks start
        return (f"{e} — refused", notes)
    if absent:                                                                                # a file the variant loads is not there: the fold cannot start (refused by name)
        return (f"HF_HOME={hf} lacks the weights files variant {variant or '<any>'} loads (stock/PINS.json \"weights\"): " + "; ".join(f"{rel} absent" for rel in absent[:6])
                + (f"; +{len(absent) - 6} more" if len(absent) > 6 else ""), notes)
    if unknown:                                                                               # present, sha256 not pinned: WARN by name and proceed (the pin is the pin note, never a refusal)
        word = weights_unknown_words(unknown) + (" via=launcher" if in_rank_process(env) else "")
        notes.append(word)
        sys.stderr.write(f"{LOG} WARNING: {word}\n")
    else:                                                                                     # every file's sha256 is the pinned digest
        word = weights_pinned_words(hf, variant)
        notes.append(word)
        sys.stderr.write(f"{LOG} {word}\n")
    rec = _WEIGHTS_SEEN.get((hf, variant)) or {}
    if rec.get("memo_readonly"):                                                              # the memo dir cannot be written (read-only mount): named, and the digests above were computed afresh
        # Worded to read unambiguously as informational, not a failure: no "not writable"/"not written" and no raw
        # OSError/Errno text embedded, so a log reader that greps stderr for generic error markers does not match a clean pass
        # over a read-only cache mount.
        ro = (f"WEIGHTS digest memo skipped: the cache directory is read-only ({os.path.join(rec.get('memo_dir') or '?', 'weights_digests.json')}); "
              f"every file above was digested in full this process, just not memoised; set {ENV_WEIGHTS_MEMO_DIR} to a writable cache directory to memoise")
        notes.append(ro); sys.stderr.write(f"{LOG} {ro}\n")
    if not ccd:
        notes.append("ESMCFOLD_CCD_PATH is not set: the library reads ccd.pkl from the pinned snapshot under HF_HOME (stock/PINS.json \"ccd\")")
    return None, notes


def _stack_gates(base: dict, p: dict) -> Optional[str]:
    """Gates on the running stack: the core's producers, upstream versions against the pins, a visible GPU, the data paths. Returns the
    refusal reason or None."""
    force = os.environ.get(ENV_FORCE, "") == "1"
    import io
    from ._core_gate import CoreGateRefused, gate as core_gate
    try:                                                                                  # first gate, THE pin gate: the importable opt_core is the one [tool.opt_core] pins
        facts = core_gate(os.path.join(opt_home(), "pyproject.toml"), stream=io.StringIO())   # (opt/pyproject.toml's pin vs the located core's __version__, read on disk; nothing of the core imported;
    except CoreGateRefused as e:                                                          # ESMFOLD2_OPT_FORCE never overrides it) — the refusal is this report's NOT ACTIVE reason
        return e.line.split(" NOT ACTIVE: ", 1)[-1]                                       # reason=core_missing:opt_core … | reason=core_mismatch: … | reason=core_pin_unreadable: …
    base["core"] = {"pinned": dict(facts["pinned"]), "installed": {k: facts["installed"].get(k) for k in ("version", "root")}}
    from ._producers import refusal as _producers_refusal
    why = _producers_refusal()                                                           # then: every core module this package imports is on disk under that core (producer_missing:<modules>)
    if why:
        return why
    ok, have, why = version_gate(p, force)
    base["upstream"] = have
    if not ok:
        return why
    ok, detail, why = pins_gate(p, force)                                                   # the commit, not the version string
    base["upstream_pins"] = detail
    if not ok:
        return why
    why = attn_gate(base, p, metadata_only=True)                                            # the fail-loud switch on the distribution metadata (dry run and activation alike: nothing of torch / upstream is imported before the allocator policy is exported); ESMFOLD2_OPT_FORCE never overrides it
    if why:
        return why
    g = gpu_probe_record()
    base["gpu"] = dict(g)
    if g["sm"] is None and not force:
        return f"no CUDA device visible ({g.get('probe')}; the levers need an NVIDIA GPU); {ENV_FORCE}=1 applies them anyway"
    if g["sm"] is None:
        sys.stderr.write(f"{LOG} WARNING: no CUDA device visible; {ENV_FORCE}=1 applies the levers anyway\n")
    if base["target_gpu"] and g["name"] and base["target_gpu"].lower() not in g["name"].lower():
        base.setdefault("notes", []).append(f"MODEL_OPT_TARGET_GPU={base['target_gpu']} but the GPU is {g['name']}: not the GPU this configuration targets")
    why, notes = data_path_gate(variant=base.get("variant"), pins=p)                       # the weights root unset/absent or a file missing: refuse, loudly; an unknown checkpoint: WARN and proceed
    base.setdefault("notes", []).extend(notes)
    return why


def attn_gate(base: dict, p: dict, environ: Optional[dict] = None, metadata_only: bool = False) -> Optional[str]:
    """Read the fast-environment words of THIS process into the report and apply the fail-loud switch (attn.py). Two forms: ``metadata_only``
    (the stack gates — a dry run, and an activation BEFORE the memory lines export their allocator policy, when nothing of torch or upstream may
    be imported yet): ``base["attn_metadata"]`` = the distribution words, and with ``ESMFOLD2_OPT_REQUIRE_FAST_ENV=1`` a required distribution
    absent is the refusal; the run-time form (an activation, after the allocator policy is exported and before the kit server is imported):
    upstream's two modeling modules are imported here (torch with them; the kit server imports the same modules next) and ``base["attn"]`` =
    attn.state() — the atom-attention flag and the ESMC switches as this interpreter has them, the words the ACTIVE line prints (the APPLIED line
    reads again on the configured model) — and a required word not at its accelerated value (attn.REQUIRED) is the refusal sentence (names each
    failing word, the import error behind it, and the pinned stack's accelerated layer). A switch value other than 1/0/unset is refused by
    name; ESMFOLD2_OPT_FORCE never overrides it. Returns the refusal or None."""
    from . import attn as _attn
    environ = os.environ if environ is None else environ
    image = (p or {}).get("image") or {}
    try:
        if metadata_only or base.get("dry_run"):
            base["attn_metadata"] = _attn.metadata_words()
            return _attn.require_refusal_metadata(environ=environ, image=image)
        try:
            st = _attn.state(load=True)
        except ImportError as e:                                                            # upstream's modules themselves do not import here (a stand-in upstream): the paths are unread, named
            st = _attn.state()
            base.setdefault("notes", []).append(f"fast-environment words unread: {type(e).__name__}: {e}")
        base["attn"] = st
        return _attn.require_refusal(st, environ=environ, image=image)
    except ValueError as e:                                                                 # ESMFOLD2_OPT_REQUIRE_FAST_ENV set to something other than 1 / 0
        return str(e)


def _gates(base: dict, variant: Optional[str]) -> Tuple[Optional[str], Optional[dict]]:
    """All gates (activation). Returns (refusal reason or None, pins)."""
    why, p = _path_gates(base, variant)
    if why:
        return why, p
    return _stack_gates(base, p), p


GRAPH_SWITCH_KEYS = (("EF2_GRAPH_BUDGET_TOKENS", "graph_budget_tokens"), ("EF2_GRAPH_LRU_SAMPLER", "graph_lru_sampler"),
                     ("EF2_GRAPH_BUDGET_TOKENS_TRUNK", "graph_budget_tokens_trunk"), ("EF2_GRAPH_BUDGET_TOKENS_ENCODER", "graph_budget_tokens_encoder"),
                     ("EF2_GRAPH_BUDGET_TOKENS_SAMPLER", "graph_budget_tokens_sampler"),
                     ("EF2_RECYCLE_GRAPH_MAX_TOKENS", "recycle_graph_max_tokens"),
                     ("EF2_GRAPH_CAPTURE", "graph_capture"))                      # (switch, report key); the names are modes.ENV_GRAPH_BUDGET / ENV_GRAPH_LRU_SAMPLER / ENV_GRAPH_BUDGET_<SITE> / ENV_RECYCLE_GRAPH_MAX /
                                                                                # ENV_GRAPH_CAPTURE (the capture policy: 0 = no CUDA graph at any site — the memory line; 1 = the budgets decide)
SITE_KEYS = ("graph_budget_tokens_trunk", "graph_budget_tokens_encoder", "graph_budget_tokens_sampler")


def effective_graph_settings(res, environ=None) -> dict:
    """The CUDA-graph switches as the levers will run with them in THIS process, each with its source: a mode override wins (activation exports
    it), else the caller's environment (the ``EF2_GRAPH_*`` switches are not dropped at activation), else the package's default
    (``package_graph_defaults(res, environ)``: 2 sampler step graphs per generation; no cap at the sampler site; the Full model's per-shape budget 800
    and the Fast model's trunk / encoder site budgets 800 under exact / fast — exported at activation, package_graph_exports), else the driver's
    import-time default (``modes.GRAPH_ENV_DEFAULTS``: budget 0 = every shape captured; site -1 = the site inherits the per-shape budget). The
    three site keys also carry ``<key>_effective``: the budget that governs the site (its own value when >= 0, else the per-shape budget).
    ``graph_capture`` is the capture POLICY over all of them (modes.ENV_GRAPH_CAPTURE: 0 = no CUDA graph at any site — the memory line's word,
    under which the budgets govern nothing; 1 = the budgets decide)."""
    from .modes import GRAPH_ENV_DEFAULTS as _D
    _P = package_graph_defaults(res, environ)
    environ = os.environ if environ is None else environ
    overrides = dict(getattr(res, "overrides", None) or {})
    out = {}
    for env_name, key in GRAPH_SWITCH_KEYS:
        if env_name in overrides:
            val, src = overrides[env_name], "mode"
        elif (environ.get(env_name) or "").strip():
            val, src = environ[env_name].strip(), "env"
        elif env_name in _P:
            val, src = _P[env_name], "package"
        else:
            val, src = _D[env_name], "default"
        try:
            val = int(val)
        except (TypeError, ValueError):
            val = str(val)
        out[key] = val; out[key + "_source"] = src
    for key in SITE_KEYS:                                                      # ef2_opt._site_budget: the site's own value when set (>= 0), else the per-shape budget
        v = out[key]
        out[key + "_effective"] = v if isinstance(v, int) and v >= 0 else out["graph_budget_tokens"]
    return out


def package_graph_defaults(res, environ=None) -> dict:
    """The package's own defaults for the graph switches for THIS resolution: ``modes.PACKAGE_GRAPH_DEFAULTS`` (the sampler step-graph budget, every
    mode), plus ``modes.PACKAGE_GRAPH_BUDGET_FULL`` (EF2_GRAPH_BUDGET_TOKENS=800) when the mode is exact / fast and the model is the Full one
    (``res.server_variant == "full"``), plus ``modes.PACKAGE_GRAPH_BUDGET_FASTMODEL`` (the trunk / encoder sites at 800) when the mode is exact / fast and
    the model is the Fast one, plus ``modes.PACKAGE_GRAPH_BUDGET_SAMPLER`` (the sampler site at 1536) under exact / fast on either model — the last two only
    when the caller sets no per-shape budget of its own (a caller's ``EF2_GRAPH_BUDGET_TOKENS`` governs every site it does not set itself).
    big gets none of these: its line captures no CUDA graph (modes.ENV_GRAPH_CAPTURE=0), so no site budget applies there. A mode override or the
    caller's own value always wins over these (effective_graph_settings)."""
    from .modes import (PACKAGE_GRAPH_DEFAULTS, PACKAGE_GRAPH_BUDGET_FULL, FULL_BUDGET_MODES, FULL_BUDGET_SERVER_VARIANT, PACKAGE_GRAPH_BUDGET_FASTMODEL,
                        FASTMODEL_BUDGET_SERVER_VARIANT, PACKAGE_GRAPH_BUDGET_SAMPLER, ENV_GRAPH_BUDGET)
    environ = os.environ if environ is None else environ
    out = dict(PACKAGE_GRAPH_DEFAULTS)
    mode, sv = getattr(res, "mode", None), getattr(res, "server_variant", None)
    if mode in FULL_BUDGET_MODES and sv == FULL_BUDGET_SERVER_VARIANT:
        out.update(PACKAGE_GRAPH_BUDGET_FULL)
    caller_global = (environ.get(ENV_GRAPH_BUDGET) or "").strip() or (getattr(res, "overrides", None) or {}).get(ENV_GRAPH_BUDGET)
    if mode in FULL_BUDGET_MODES and sv == FASTMODEL_BUDGET_SERVER_VARIANT and not caller_global:
        out.update(PACKAGE_GRAPH_BUDGET_FASTMODEL)
    if mode in FULL_BUDGET_MODES and getattr(res, "line", None) is None and not caller_global:   # exact / fast themselves (not a line another mode composes): the sampler site's cap
        out.update(PACKAGE_GRAPH_BUDGET_SAMPLER)
    return out


def package_graph_exports(res, environ=None) -> dict:
    """The kit graph switches the package exports at activation on top of the mode's own overrides: ``package_graph_defaults(res)`` for every switch
    neither the mode's overrides nor the caller's environment sets (effective_graph_settings source ``package``), as environment strings."""
    eff = effective_graph_settings(res, environ)
    return {env_name: str(eff[key]) for env_name, key in GRAPH_SWITCH_KEYS if eff[key + "_source"] == "package"}


def activation_exports(res, environ=None) -> dict:
    """Everything activation exports for this resolution: the mode's own override switches plus package_graph_exports (the activation report's ``env``)."""
    out = dict(getattr(res, "overrides", None) or {})
    out.update(package_graph_exports(res, environ))
    return out


def _resolution_fields(res: Resolution, cc: Optional[str], kit: str) -> dict:
    key = kernel_key(cc)
    out = {"server_mode": res.server_mode, "server_line": describe_line(res), "line": res.line, "server_entry": list(res.entry), "composition": res.composition,
           "effective": effective_graph_settings(res),                              # the graph switches as this process will run them (mode override > caller env > package default > driver default)
           "overrides": res.overrides, "levers_planned": res.levers_for_variant, "levers_not_for_variant": res.levers_not_for_variant,
           "levers_unknown": res.levers_unknown, "server_variant": res.server_variant,
           "levers_dropped": list(res.drop), "ablate": list(res.ablate), "ablate_tokens": list(res.ablate_tokens), "knobs": dict(res.knobs),   # the line's drops; the ablation variable's subtractions / sub-choices (report: ablate= word, LEVER state=off reason=ablated)
           "levers_off": list(res.levers_off), "levers_not_for_route": dict(res.not_for_route), "n_gpu": res.n_gpu,   # the row-sharded route's subtractions at n_gpu > 1 ({lever: reason}; LEVER state=skipped reason=not_for_route:…)
           "levers_not_for_class": dict(res.not_for_class), "class_cc": res.cc,                                   # the per-class subtractions ({lever: reason}; LEVER state=skipped reason=not_for_class:<class>:<reason>) and the class the set was resolved for
           "use_msa": VARIANT_USES_MSA.get(res.variant) if res.variant else None,
           "kernel_key": key, "stack_key": stack_key(cc), "jit_cache_key": jit_cache_key(cc=cc or "unknown", strict=False),           # display field (the cache is keyed by configs/h100.env, strict)
           "sys_path": kit_sys_path(kit)}
    return out


def _dry_run(mode: str, variant: Optional[str], strict: bool, line: Optional[str] = None) -> dict:
    """Resolve and report without applying: the path gates bind; the stack gates (upstream versions, GPU) are reported as the refusal
    activation would give, so `check` still shows the resolved line on a box without the stack."""
    base = _base(mode, variant)
    base["dry_run"] = True
    if mode == "off":
        rep = dict(base, reason="mode off: stock esmfold2 (the upstream API in a clean subprocess; no environment set, no lever applied)",
                   weights=weights_status(variant, pins() if os.path.isfile(pins_path()) else None))
        _report.log_activation(rep); rep["logged"] = True
        return rep
    why, p = _path_gates(base, variant)
    if why:
        rep = dict(base, reason=why)
        _report.log_activation(rep); rep["logged"] = True
        if strict:
            raise ActivationError(why)
        return rep
    would_refuse = _stack_gates(base, p)
    kit = base["kit_home"]
    try:
        res = resolve(mode, variant, kit, line=line, ablate=ablation_text(), n_gpu=_report.N_GPU["P"], cc=class_cc(base))
    except Exception as e:  # noqa: BLE001
        from opt_core.oom import is_oom
        if is_oom(e): raise                                                 # a GPU out-of-memory error propagates; it is never re-worded into a refusal
        rep = dict(base, reason=f"mode could not be resolved from the server table: {e!r}")
        _report.log_activation(rep); rep["logged"] = True
        if strict:
            raise ActivationError(rep["reason"])
        return rep
    fields = _resolution_fields(res, base["gpu"].get("cc"), kit)
    notes = list(base.get("notes", [])) + res.notes + [n for n in (ablation_foreign_note(),) if n]   # + MODEL_OPT_LEVERS_OFF words left to the core's packages, named once
    present = [k for k in SWITCHES if k in os.environ]
    if present:
        notes.append(f"caller environment sets {present}: activation drops them (the server table and the mode's own overrides decide)")
    rep = dict(base, **fields, levers_applied=list(fields["levers_planned"]), levers_fallback=[], partial=[], levers_unavailable=[],
               env=activation_exports(res), unset=present, weights=weights_status(variant, p), notes=notes, would_refuse=would_refuse,
               reason=("dry run: resolved; activation would refuse: " + would_refuse) if would_refuse else "dry run: resolved and gated, nothing applied")
    _report.log_activation(rep); rep["logged"] = True
    if strict and would_refuse:
        raise ActivationError(would_refuse)
    return rep


# ----------------------------------------------------------------------------------------------------------------- activation
def graph_lines_alloc_export(notes: list) -> dict:
    """The graph lines' allocator policy (``exact`` / ``fast``: ef2_opt captures CUDA graphs per shape — tg / sg / eg): expandable
    segments (the memory lines' policy, ``big.ALLOC_POLICY``, through the shared primitive ``opt_core.mem.torch_alloc.export`` with
    ``graphs_on=True, allow_with_graphs=True`` — composed with graph capture on purpose) exported BEFORE any CUDA work of this process.
    Without it the caching allocator fragments beside the graph private pools and the second fold of a large shape can fail to fit
    where the first one did; with it reserved memory tracks allocated. Soft by
    design: a process whose CUDA allocator is already initialised, or whose environment names another allocator configuration, keeps
    its allocator — the activation report records ``alloc_export='kept: <reason>'`` and a note; every fold below that headroom is
    unaffected (the setting is numerics-free)."""
    from opt_core.mem import MemLeverRefused, torch_alloc
    try:
        return dict(torch_alloc.export(_big.ALLOC_POLICY, None, lever="es", graphs_on=True, allow_with_graphs=True))
    except MemLeverRefused as e:
        notes.append(f"allocator policy not exported: {e}")
        return {"alloc": _big.ALLOC_POLICY, "alloc_conf": _big.ALLOC_CONF_XL, "alloc_export": f"kept: {e}"}


def activate(mode: str, variant: Optional[str] = None, strict: bool = False, trigger: Optional[str] = None, dry_run: bool = False,
             line: Optional[str] = None) -> dict:
    """Apply a mode once per process (idempotent). Returns the activation report; `strict` raises ActivationError when not active;
    `dry_run` resolves, gates and reports (report["dry_run"] = True) without exporting or applying anything; `line` selects a line of a
    internal composition key of big (modes.KIT_LINES; None = the mode itself)."""
    global _REPORT
    mode = (mode or "").strip().lower()
    line = (line or "").strip().lower() or None
    if mode not in MODES:
        if strict:                                                                          # the hook route: the kit's NOT ACTIVE line, then enable()'s own refusal — never a traceback under ESMFOLD2_OPT
            rep = dict(_base(mode, variant, trigger), reason=f"unknown mode {mode!r}; expected one of {MODES}")
            _report.log_activation(rep)
            raise ActivationError(rep["reason"])
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if not dry_run:
        _disarm_autoload()                                                                  # an explicit call wins over the ESMFOLD2_OPT route
    if not dry_run and _REPORT is not None and (_REPORT.get("active") or _REPORT.get("apply_failed")):
        req_variant = check_variant(variant) or _REPORT.get("variant")
        if _REPORT.get("mode") == mode and req_variant == _REPORT.get("variant") and (line is None or line == _REPORT.get("line")):
            return dict(_REPORT)                                                           # idempotent
        reason = (f"already active as mode={_REPORT.get('mode')} variant={_REPORT.get('variant')}; levers are applied once per process and "
                  f"patch process-wide (one mode and one variant per process: restart to change mode or variant)")
        if strict:
            raise ActivationError(reason)
        return dict(_REPORT, active=False, refused=True, requested={"mode": mode, "variant": variant}, reason=reason)
    try:
        variant = _effective_variant(variant)
    except ValueError as e:
        if dry_run or not strict:
            rep = dict(_base(mode, variant, trigger), reason=str(e), dry_run=dry_run)
            _report.log_activation(rep); rep["logged"] = True
            if not dry_run:
                _REPORT = dict(rep)
            return rep
        raise ActivationError(str(e)) from e
    if dry_run:
        return _dry_run(mode, variant, strict, line)
    if _ACTIVATING:
        return {"active": False, "mode": mode, "variant": variant, "reason": "activation in progress (re-entrant call ignored)"}
    try:
        return _activate_locked(mode, variant, strict, trigger, line)
    except ImportError as e:                                                            # the core (or a module the activation imports) is missing at activation: the kit's NOT ACTIVE
        missing = getattr(e, "name", None) or str(e)                                      # line with a named reason and exit 3 on every route — never a traceback, never a silent stock run
        kind = "core_missing" if str(missing).split(".")[0] == "opt_core" else "import_failed"
        reason = f"{kind}:{missing} ({e}) — the kit runs on the core: pip install -e common/opt_core -e esmfold2/opt"
        try:
            rep = dict(_base(mode, variant, trigger), reason=reason, apply_failed=True)
        except ImportError:                                                             # the base report reads versions through modules that may be the missing ones
            rep = {"active": False, "mode": mode, "variant": variant, "trigger": trigger, "package_version": __version__, "reason": reason, "apply_failed": True}
        if not (_REPORT or {}).get("logged"):
            _report.log_activation(rep)
        rep["logged"] = True; _REPORT = dict(rep)
        if strict:
            raise ActivationError(reason) from e
        return dict(rep)


def _disarm_autoload() -> None:
    from . import _autoload
    _autoload.disarm()


def _activate_locked(mode: str, variant: Optional[str], strict: bool, trigger: Optional[str], line: Optional[str] = None) -> dict:
    global _ACTIVATING
    _ACTIVATING = True
    try:
        return _activate_body(mode, variant, strict, trigger, line)
    finally:
        _ACTIVATING = False


def _activate_body(mode: str, variant: Optional[str], strict: bool, trigger: Optional[str], line: Optional[str] = None) -> dict:
    global _REPORT
    base = _base(mode, variant, trigger)
    if mode == "off":
        _REPORT = dict(base, reason="mode off: stock esmfold2 (the upstream API in a clean subprocess; no environment set, no lever applied)")
        return dict(_REPORT)

    def refuse(reason: str) -> dict:
        global _REPORT
        _REPORT = dict(base, reason=reason)
        _report.log_activation(_REPORT); _REPORT["logged"] = True
        if strict:
            raise ActivationError(reason)
        return dict(_REPORT)

    why, p = _gates(base, variant)
    if why:
        return refuse(why)
    chk = register_instance_counter()                                                       # the constructor wrap from here on
    if chk["n"]:                                                                            # the late-activation rule (1): a model exists
        return refuse(f"{chk['n']} ESMFold2Model instance(s) already exist in this process ({chk['method']}): enable() is allowed any time "
                      "after import but must run before a model is constructed (the levers are applied by the kit's configure() on each "
                      "model at its first fold(), and a model built before activation would fold unconfigured)")
    applied = kit_levers_applied()
    if applied:                                                                             # (2): the kit reports a lever already on
        return refuse(f"kit levers already applied in this process by {applied}: enable() cannot take over a process the kit's own "
                      "modules configured (the package never re-applies or disables a lever)")
    kit = base["kit_home"]
    try:
        res = resolve(mode, variant, kit, line=line, ablate=ablation_text(), n_gpu=_report.N_GPU["P"], cc=class_cc(base))
    except Exception as e:  # noqa: BLE001
        from opt_core.oom import is_oom
        if is_oom(e): raise                                                 # a GPU out-of-memory error propagates; it is never re-worded into a refusal
        return refuse(f"mode could not be resolved from the server table: {e!r}")
    notes = list(base.get("notes", [])) + res.notes + [n for n in (ablation_foreign_note(),) if n]   # + MODEL_OPT_LEVERS_OFF words left to the core's packages, named once
    if chk["method"] == "gc":
        notes.append(f"{MODEL_CLASS} instances are counted by a gc scan: the class could not be wrapped")
    for m in ("ef2_server", "ef2_opt", "ef2_w4", "ef2_msa", "ef2_mk_sampler"):
        if m in sys.modules:
            notes.append(f"{m} was imported before activation")
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None and getattr(getattr(torch_mod, "cuda", None), "is_initialized", lambda: False)():
        notes.append("CUDA was initialised before activation: allocator settings cannot take effect in this process")
    dropped = {k: os.environ.pop(k) for k in SWITCHES if k in os.environ}       # the table and the mode's overrides decide, never the caller's env
    if dropped:
        notes.append(f"dropped caller switches {sorted(dropped)} (the server table is the source of truth)")
    exports = activation_exports(res)                                         # the mode's own overrides + the package's graph-switch defaults the caller did not set
    os.environ.update(exports)
    if res.composition.get("conf_per_sample") and os.environ.get("EF2_CONF_PER_SAMPLE", "") != "0":   # the memory mode's confidence-head word (modes.KitMode.conf_per_sample;
        os.environ["EF2_CONF_PER_SAMPLE"] = "1"                                                    # driver/ef2_conf.py reads it in ef2_server.configure); a caller's "0" wins
    if res.composition.get("alloc_strict"):                                                    # the memory mode's allocator policy (modes.KitMode.alloc_strict): exported before any CUDA work or refused by name
        try:
            base["alloc"] = _big.alloc_export()
        except ActivationError as e:
            _REPORT = dict(base, apply_failed=True, reason=str(e), env=dict(exports), notes=notes)
            _report.log_activation(_REPORT); _REPORT["logged"] = True
            if strict:
                raise
            return dict(_REPORT)
    else:                                                                                       # the graph lines (exact / fast, fast's XL storage levers included): the same allocator policy beside the CUDA-graph pools, soft
        base["alloc"] = graph_lines_alloc_export(notes)
    why = attn_gate(base, p)                                                 # the run-time fast-environment words (ACTIVE line) and the fail-loud switch on them: upstream's modules are imported
    if why:                                                                   # here, AFTER the allocator policy is exported (an import that initialised CUDA first would void it) and before the kit
        for k in exports:                                                     # a refused process keeps the caller's environment: the exports leave with the refusal
            os.environ.pop(k, None)
        os.environ.update(dropped)
        base["notes"] = notes
        return refuse(why)
    os.environ[ENV_MODE] = mode                                               # children autoload the same mode / variant
    os.environ[ENV_VARIANT] = variant
    entries = kit_sys_path(kit)
    _install_sys_path(entries)
    _report.register_exit_tally()
    try:
        srv = importlib.import_module("ef2_server")
        live = srv.MODES.get(res.server_mode)
        if live is None or tuple(live) != tuple(res.entry):
            raise ActivationError(f"server table disagrees with its file: MODES[{res.server_mode!r}] = {live!r} vs file {res.entry!r}")
        _install_hook()
    except Exception as e:  # noqa: BLE001
        from opt_core.oom import is_oom
        if is_oom(e): raise                                                 # a GPU out-of-memory error propagates; it is never re-worded into a refusal
        _REPORT = dict(base, apply_failed=True, reason=f"lever arming failed: {e!r}", env=dict(exports), notes=notes)
        _report.log_activation(_REPORT); _REPORT["logged"] = True
        if strict:
            raise ActivationError(_REPORT["reason"]) from e
        return dict(_REPORT)
    try:
        g = torch_gpu_info()
        cur = base.get("gpu") or {}
        base["gpu"] = {"name": cur.get("name") or g["name"], "sm": cur.get("sm") or g["sm"], "cc": cur.get("cc") or g["cc"],
                       "probe": cur.get("probe") if cur.get("name") else g["probe"], "torch": g}
        if cur.get("cc") and g["cc"] and cur["cc"] != g["cc"]:
            notes.append(f"GPU probes disagree: {cur.get('probe')} says {cur.get('name')} (cc {cur['cc']}), torch says {g['name']} (cc {g['cc']})")
    except Exception as e:  # noqa: BLE001
        notes.append(f"torch GPU probe failed: {e!r}")
    fields = _resolution_fields(res, base["gpu"].get("cc"), kit)
    _REPORT = dict(base, active=True, **fields, levers_applied=[], levers_fallback=[], levers_substituted={}, partial=[], levers_unavailable=[],
                   fallback_reasons={}, applied="deferred", env=dict(exports), unset=sorted(dropped),
                   weights=weights_status(variant, p), notes=notes, resolution=res)
    _report.log_activation(_REPORT); _REPORT["logged"] = True
    return dict(_REPORT)


# ----------------------------------------------------------------------------------------------------------- the feature-cache seed guard
SEED_GUARD = {"installed": 0, "smiles_inputs": 0, "smiles_calls": 0}     # the guard's own record (status()["seed_guard"])


def smiles_chain_ids(input) -> list:
    """The ids of the SMILES-ligand chains of a StructurePredictionInput (a chain with a `smiles` value); [] for every other input."""
    return [getattr(ch, "id", None) for ch in (getattr(input, "sequences", None) or []) if getattr(ch, "smiles", None)]


def guard_feature_cache(builder) -> bool:
    """The kit's `fc` lever (opt/forward/fast_inference/driver/ef2_opt.py install_feature_cache) caches ``builder.prepare_input`` per
    (content, device) — seed-blind — while upstream seeds the SMILES conformer generation (stock/src/esm/models/esmfold2/prepare_input.py
    tokenize_ligand_smiles(…, seed)): for an input with a SMILES-ligand chain the features are seed-dependent, and a cached entry would serve
    one seed's conformer to every later seed. This guard, installed on the builder right after the kit's configure(), routes every such input
    to the builder class's own prepare_input (uncached, per seed) and prints one named line per input; every other input keeps the kit's cache
    (CCD ligands take dictionary conformers, no RNG). Idempotent per builder; False when there is no builder or no kit cache on it."""
    if builder is None or not getattr(builder, "_ef2opt_fc", False) or getattr(builder, "_esmfold2_opt_seed_guard", False):
        return False
    cached, uncached, seen = builder.prepare_input, type(builder).prepare_input, set()

    @functools.wraps(cached)
    def prepare_input(input, seed=None, device=None):
        ids = smiles_chain_ids(input)
        if not ids:
            return cached(input, seed=seed, device=device)
        SEED_GUARD["smiles_calls"] += 1
        if id(input) not in seen:
            seen.add(id(input)); SEED_GUARD["smiles_inputs"] += 1
            sys.stderr.write(f"{_report.PREFIX} feature_cache: bypassed for SMILES ligand chain(s) {','.join(map(str, ids))} — upstream seeds the conformer "
                             f"generation, so the features are seed-dependent: computed per seed, never served across seeds\n"); sys.stderr.flush()
        return uncached(builder, input, seed=seed, device=device)

    builder.prepare_input = prepare_input
    builder._esmfold2_opt_seed_guard = True
    SEED_GUARD["installed"] += 1
    return True


# ----------------------------------------------------------------------------------------------------------------- application
def _install_hook() -> None:
    """Wrap ESMFold2InputBuilder.fold so the kit's configure() runs for a model at its first fold (the lazy route)."""
    if _HOOK["installed"]:
        return
    from esm.models.esmfold2.processor import ESMFold2InputBuilder
    orig = ESMFold2InputBuilder.fold

    @functools.wraps(orig)
    def fold(self, model, input, *args, **kwargs):
        if id(model) not in _CONFIGURED and _REPORT is not None and _REPORT.get("active"):
            apply_to(model, self, trigger="fold")
        return orig(self, model, input, *args, **kwargs)

    ESMFold2InputBuilder.fold = fold
    _HOOK.update(installed=True, original=orig, owner=ESMFold2InputBuilder)


def apply_to(model, builder=None, trigger: str = "explicit", samples: int = 1, out_dir: Optional[str] = None) -> dict:
    """Install the active mode's levers on `model` through the kit server's own configure() (idempotent per model instance);
    returns the application record and updates the activation report (levers applied / fallback from the kit's records). An xl mode
    installs the XL add-on's levers first (xl_install; `samples` = the fold call's num_diffusion_samples, which decides x2b)."""
    global _REPORT
    if _REPORT is None or not _REPORT.get("active"):
        raise ActivationError("apply_to: no active mode (call esmfold2_opt.enable(mode, variant) first)")
    if id(model) in _CONFIGURED:
        return dict(_CONFIGURED[id(model)])
    res: Resolution = _REPORT["resolution"]
    srv = sys.modules["ef2_server"]
    from .modes import KIT_MODES as _KIT_MODES, kit_mode as _kit_mode
    km = _kit_mode(res.mode, res.line) if res.mode in _KIT_MODES else None           # the RESOLVED line's entry (a mode with lines: the line, never the mode's default)
    calls = []
    if km is not None and km.backend:                                        # a memory mode: the named backend's two model calls (stock_fold.model_calls, the one table) before its add-on installs
        from .stock_fold import model_calls as _model_calls
        calls = _model_calls(None, km.backend)
        for name, value in calls:
            getattr(model, name)(value)
        _REPORT = dict(_REPORT, model_calls=[list(c) for c in calls])
    try:
        mem = _big.apply(model, res, samples, out_dir)                        # the engine adapter: the memory line's install (the XL add-on) before configure()
    except ActivationError as e:
        _REPORT = dict(_REPORT, apply_failed=True, active=False, reason=str(e))
        _report.log_activation(_REPORT); _REPORT["logged"] = True
        raise
    if mem:
        _REPORT = dict(_REPORT, **mem)
    try:
        desc = srv.configure(model, res.server_mode, builder, off=res.levers_off, knobs=res.knobs)   # the set minus the line's drops, the ablated levers and the levers not for this variant; the ablation knobs
    except Exception as e:  # noqa: BLE001
        from opt_core.oom import is_oom
        if is_oom(e): raise                                                 # a GPU out-of-memory error propagates; it is never re-worded into a refusal
        _REPORT = dict(_REPORT, apply_failed=True, active=False, reason=f"ef2_server.configure({res.server_mode!r}) raised {e!r}")
        _report.log_activation(_REPORT); _REPORT["logged"] = True
        raise ActivationError(_REPORT["reason"]) from e
    guard_feature_cache(builder)                                        # the seed guard sits on the kit's feature cache (SMILES ligands per seed)
    rec = classify(res, model, desc)
    rec.update(trigger=trigger, model_index=len(_CONFIGURED), has_msa_encoder=getattr(model, "msa_encoder", None) is not None, seed_guard=dict(SEED_GUARD))
    _CONFIGURED[id(model)] = rec
    models = list(_REPORT.get("models") or []) + [{k: rec[k] for k in ("model_index", "trigger", "has_msa_encoder", "levers_applied", "levers_fallback")}]
    union_fb = sorted({n for m in models for n in m["levers_fallback"]})
    _REPORT = dict(_REPORT, applied="configured", models=models, levers_applied=rec["levers_applied"], levers_fallback=rec["levers_fallback"],
                   levers_substituted=rec["levers_substituted"], fallback_reasons=rec["fallback_reasons"], levers_not_in_variant=rec["levers_not_for_variant"],
                   levers_deferred=list(rec["levers_deferred"]),
                   partial=list(union_fb), levers_unavailable=union_fb, configure_desc=desc, kit_records=rec["kit_records"])
    from . import attn as _attn
    rec["attn"] = _attn.state(model)                                                # the paths as THIS model runs them: upstream's flag × the forward every SWA3DRoPEAttention resolves after every lever
    _CONFIGURED[id(model)] = rec                                                     # is installed, the ESMC classes bound in the LM (the APPLIED line's words; status()'s activation report)
    _REPORT = dict(_REPORT, attn=rec["attn"])
    sys.stderr.write(_report.applied_line(_REPORT, rec) + "\n")
    why = _attn.require_refusal(rec["attn"], image=(pins() or {}).get("image"))     # ESMFOLD2_OPT_REQUIRE_FAST_ENV=1 and the configured model is not on the accelerated paths (the activation gate read
    if why:                                                                          # the switches; this reads the bound classes/forwards): refused by name, exit 3 — never a slow pass under the switch
        sys.stderr.write(_report.not_active_line(why) + "\n")
        raise SystemExit(_report.EXIT_NOT_ACTIVE)
    sys.stderr.write("".join(l + "\n" for l in _report.lever_lines(_REPORT, rec, include_deferred=False))); sys.stderr.flush()   # one evidence line per lever per arm per process (the release tree's per-lever
    return dict(rec)                                                                 # convention); the route's deferred levers print theirs from settle_route, once rowpair.install_rank has installed them


def settle_route(route: dict, model=None) -> dict:
    """After ``rowpair.install_rank`` (n_gpu > 1): judge the row-chunking levers classify() deferred from the installer's record
    (``route`` = ``esmfold2_opt.rowchunk.install.install(...)``: ``{'levers': {name: bool}, 'why': {name: reason}}``). A member the record
    shows bound joins ``levers_applied``; a member of the set it does not show bound joins ``levers_fallback`` + ``partial`` with its reason —
    the exit rule then refuses the run by name (report.verdict: EXIT_NOT_ACTIVE), exactly as for a kernel lever that could not launch. Prints the
    members' LEVER lines (the definitive ones: apply_to left them out). Returns the updated report ({} when no mode is active)."""
    global _REPORT
    if _REPORT is None or not _REPORT.get("active"):
        return {}
    on = dict((route or {}).get("levers") or {}); why = dict((route or {}).get("why") or {})
    rep = dict(_REPORT)
    deferred = [n for n in (rep.get("levers_deferred") or []) if n not in (rep.get("levers_applied") or [])]
    applied = list(rep.get("levers_applied") or []); fallback = list(rep.get("levers_fallback") or [])
    reasons = dict(rep.get("fallback_reasons") or {}); partial = set(rep.get("partial") or [])
    for n in deferred:
        if on.get(n):
            applied.append(n)
        else:
            fallback.append(n); partial.add(n)
            reasons[n] = str(why.get(n) or "rowpair.install_rank did not bind it")
    rep.update(levers_applied=applied, levers_fallback=fallback, fallback_reasons=reasons, partial=sorted(partial),
               levers_unavailable=sorted(set(rep.get("levers_unavailable") or []) | (partial & set(deferred))), levers_deferred=[], route_levers=on)
    rec = dict(_CONFIGURED.get(id(model)) or {}) if model is not None else {}
    rec.update(levers_applied=applied, levers_fallback=fallback, fallback_reasons=reasons, levers_deferred=[])
    if model is not None and id(model) in _CONFIGURED:
        _CONFIGURED[id(model)] = dict(_CONFIGURED[id(model)], levers_applied=applied, levers_fallback=fallback, fallback_reasons=reasons, levers_deferred=[])
    _REPORT = rep
    names = set(deferred)
    lines = [l for l in _report.lever_lines(rep, rec) if any((" name=%s " % n) in l + " " for n in names)]
    if lines:
        sys.stderr.write("".join(l + "\n" for l in lines)); sys.stderr.flush()
    return dict(rep)


def _install_info(desc: str) -> Optional[dict]:
    """The dict ef2_opt.install() returned, as configure()'s desc string records it (``ef2_opt.install([...]) -> {...}``)."""
    m = re.search(r"ef2_opt\.install\(\[.*?\]\) -> (\{.*\})", desc or "")
    if not m:
        return None
    try:
        v = ast.literal_eval(m.group(1))
        return v if isinstance(v, dict) else None
    except Exception:
        return None


def kit_records(model) -> dict:
    """The kit modules' own records after configure(): ef2_w4.describe(), ef2_msa._STATE, the MK instance flag, the group modules' lever records."""
    out: dict = {}
    w4 = sys.modules.get("ef2_w4")
    if w4 is not None:
        try:
            out["w4"] = w4.describe()
        except Exception as e:  # noqa: BLE001
            out["w4"] = {"error": repr(e)}
    msa = sys.modules.get("ef2_msa")
    if msa is not None:
        st = getattr(msa, "_STATE", None)
        out["msa"] = {k: v for k, v in st.items() if k != "models"} if isinstance(st, dict) else None
        out["msa_models"] = len(st.get("models") or []) if isinstance(st, dict) else None
    try:
        dm = model.structure_head.diffusion_module
        out["mk"] = bool(getattr(dm, "_mk_enabled", False))
    except Exception:
        out["mk"] = None
    for field_name, mod_name, reader in (("atom", "ef2_atom", lambda m: m.groups_on()), ("feats", "ef2_feats", lambda m: bool(m.active())),
                                         ("msa2", "ef2_msa_v2", lambda m: list(m.levers_on())), ("pair", "ef2_pair_v2", lambda m: m.levers_on()),
                                         ("cute", "ef2_transition_cute", lambda m: m.levers_on()),
                                         ("hoist", "ef2_hoist", lambda m: dict(m.stats().get("levers") or {})), ("dit", "ef2_dit", lambda m: m.levers_on()),
                                         ("ln", "ef2_xln", lambda m: m.levers_on()), ("xte", "ef2_xte", lambda m: m.levers_on())):
        mod = sys.modules.get(mod_name)                                    # only the modules configure() imported (a group not in the set is never imported here)
        if mod is None:
            continue
        try:
            out[field_name] = reader(mod)
        except Exception as e:  # noqa: BLE001
            out[field_name] = {"error": repr(e)}
    return out


def classify(res: Resolution, model, desc: str) -> dict:
    """Applied / fallback / substituted / not-for-this-variant, from the kit's own records (registry.Lever.probe)."""
    recs = kit_records(model)
    info = _install_info(desc) or {}
    w4 = recs.get("w4") if isinstance(recs.get("w4"), dict) else {}
    disabled = {str(k).lower(): v for k, v in ((w4.get("device") or {}).get("disabled") or {}).items()}
    msa_state = recs.get("msa") or {}
    has_msa_enc = getattr(model, "msa_encoder", None) is not None
    applied, fallback, why, substituted, out_of_scope = [], [], {}, {}, {}
    deferred = []                                                        # the row-chunking levers of the n_gpu > 1 route: rowpair.install_rank installs them after configure(); settle_route judges them
    for name in res.levers_for_variant:
        lv = LEVERS[name]
        kind, key = lv.probe
        ok = False
        if kind == "rc":
            deferred.append(name)
            continue
        if kind == "base":
            ok = "set_kernel_backend('fused')" in (desc or "")
        elif kind == "opt":
            ok = bool(info.get(key)) if info else (f"'{key}'" in (desc or ""))
            if name == "msa" and not has_msa_enc:
                ok = False; why[name] = "model has no msa_encoder"
        elif kind == "w4":
            ok = bool(w4.get(key))
            if name in disabled:
                ok = False; why[name] = disabled[name]
        elif kind == "msa":
            if not has_msa_enc:
                ok = False; why[name] = "model has no msa_encoder (Fast) -> skipped by the server"
            else:
                ok = bool(msa_state.get(key))
        elif kind == "mk":
            ok = bool(recs.get("mk"))
        elif kind in ("atom", "pair", "cute", "hoist", "dit", "ln", "xte"):  # {name: bool} records of ef2_atom / ef2_pair_v2 / ef2_transition_cute / ef2_hoist / ef2_dit / ef2_xln / ef2_xte
            rec = recs.get(kind)
            ok = isinstance(rec, dict) and bool(rec.get(key))
            if isinstance(rec, dict) and "error" in rec:
                why[name] = f"{kind} record unreadable: {rec['error']}"
            elif name == "xte" and not ok and sys.modules.get("ef2_xte") is not None:
                word = sys.modules["ef2_xte"].refusal()                          # the shared core's transition row cannot serve this box (class / core / probe / self-check): named, the exact line keeps its W4 transition kernels (a declared guard)
                if word:
                    out_of_scope[name] = f"off:{word}"
                    continue
            elif name == "xln" and not ok and sys.modules.get("ef2_xln") is not None:
                word = sys.modules["ef2_xln"].refusal()                          # the shared core's LayerNorm row cannot serve this box (cc / bindings / core version): named, the mode keeps ATen's LayerNorm (a declared guard)
                if word:
                    out_of_scope[name] = f"off:{word}"
                    continue
            elif name == "t6s" and not ok and sys.modules.get("ef2_pair_v2") is not None:
                sc = (getattr(sys.modules["ef2_pair_v2"], "_STATE", {}) or {}).get("t6s_selfcheck") or {}
                if sc and not sc.get("identical", True):                       # the bitwise guard said no: named, the exact tier keeps the stock statistics kernel (a declared guard, not a subset)
                    out_of_scope[name] = f"off:{sc.get('word')}"
                    continue
            elif name == "xtr" and not ok and sys.modules.get("ef2_pair_v2") is not None:
                xw = (getattr(sys.modules["ef2_pair_v2"], "_STATE", {}) or {}).get("xtr_words") or {}
                if xw and not any((e or {}).get("word") for e in xw.values()):          # no provider word serves on this class / stack (refused by name at install): the statements run — a declared scope, not a fallback
                    kinds = sorted({str(k) for e in xw.values() for k in ((e or {}).get("refused") or {}).values()})
                    out_of_scope[name] = "off:" + ("+".join(kinds) if kinds else "no_word")
                    continue
        elif kind == "feats":
            ok = bool(recs.get("feats"))
        elif kind == "msa2":
            if not has_msa_enc:
                ok = False; why[name] = "model has no msa_encoder (Fast)"
            else:
                ok = key in (recs.get("msa2") or [])
        elif kind == "xl":
            ok = name in (_big.XL_STATE.get("levers") or [])
            if not ok:
                scope = _big.XL_STATE.get({"x2b": "x2b", "x4": "x4"}.get(name, ""))
                if scope and str(scope).startswith("off"):
                    out_of_scope[name] = scope                                   # off by the add-on's own scope rule, not a fallback
                    continue
                why[name] = "not installed"
        if ok:
            applied.append(name)
        else:
            fallback.append(name)
            why.setdefault(name, f"kit record {kind}:{key} is not set after configure()")
    if "t6" in fallback and w4.get("transition_nolin") and "t6" in disabled and "t1" not in res.levers_for_variant:
        substituted["t6"] = "t1"                                              # the kit's own small-shared-memory policy
    if "t10" in fallback and w4.get("transition_nolin") and "t10" in disabled and "t1" not in res.levers_for_variant:
        substituted["t10"] = "t1"
    return {"levers_applied": applied, "levers_fallback": fallback, "fallback_reasons": why, "levers_substituted": substituted, "levers_deferred": deferred,
            "levers_not_for_variant": list(res.levers_not_for_variant), "install_info": info,
            "levers_out_of_scope": out_of_scope, "kit_records": recs, "configure_desc": desc}


# ------------------------------------------------------------------------------------------------------- the memory adapter
from . import big as _big                                                                                              # noqa: E402
_BIG_NAMES = ("XL_RELPATH", "ALLOC_CONF", "ALLOC_CONF_XL", "xl_home", "xl_knobs", "xl_levers_on", "XL_STATE",
                "xl_install", "xl_stats", "XL_STOCK_CALLS", "XL_PROBES",
                "xl_stock_probes", "xl_gate", "XL_LM_TOKENS", "xl_lm_probe", "x4_census")


def __getattr__(name):
    """big.py is the engine adapter: the memory lines' install bytes live there (apply() / report()); its names are readable through
    this module too (status(), classify(), the tests keep one spelling). Imported last: the adapter reads kit_home / driver_dirs from here."""
    if name in _BIG_NAMES:
        return getattr(_big, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
