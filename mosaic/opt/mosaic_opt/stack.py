
"""Activation: kit paths, pins, GPU identity, the P1 directory, and the lever application through the shared JAX persistent-cache
primitive (`opt_core.jax_design.pcc`, `p1_lever`); the shared core's pin is the entries' gate (`mosaic_opt.core_gate`), whose facts fill the report's `core` block.

`activate(mode)` resolves the mode to its row (modes.resolve), gates (kit present, upstream at the stock pins, the installed `mosaic/fast/`
— when present — carrying the kit's full set of lever files (`installed_fast_check`: the driver route executes that copy), a visible GPU, a P1
directory; the stack's own versions against the kit's pins are a note on the activation line, never a refusal), and — for the in-process route (`enable()` / the .pth hook) — applies P1 through
`opt_core.jax_design.pcc.enable(DIR)` (`p1_lever`: the persistent compilation cache + the XLA autotune pin, dump when the autotune file is absent,
load otherwise — the variables the row exports at the process boundary). Nothing else is applied in-process: P2 and P3 are
call-site replacements (`registry.CALL_SITE`) that only the driver route composes (`cli.design`: the row's environment and flags at
the process boundary).

The late-activation rule, in the kit's terms: activation is allowed any time after `mosaic` / `jax` are imported and refused by name
(1) once the JAX backend is initialised (`XLA_FLAGS` — the autotune dump/load — are read at backend initialisation; `backend_initialised`),
(2) once a `Boltz2` instance exists (`instance_check`: its executables were compiled without the cache), (3) once the kit's own state
shows P1 already on (`kit_levers_applied`: `XLA_FLAGS` carrying autotune flags, a set `JAX_COMPILATION_CACHE_DIR`, or
`jax.config.jax_compilation_cache_dir`) — the package never re-applies. `activate(mode, dry_run=True)` resolves, gates and reports
without applying anything (`check`), and works without jax.
"""
import functools
import importlib
import importlib.machinery
import importlib.util
import json
import os
import sys
import weakref
from typing import Dict, List, Optional, Tuple

from opt_core.oom import is_oom                                             # the core's one out-of-memory classifier: a served handler that reroutes re-raises OOM first

from . import ActivationError, __version__
from . import report as _report
from .inputs import features_state
from .modes import AUTOTUNE_FILENAME, DRIVER_ONLY, KIT_MODES, MODES, describe_line, install_levers_of, p1_form, resolve, unknown_mode_message
from .registry import CALL_SITE, IN_PROCESS, KIT_FAST_DIR, LEVERS

ENV_MODE, ENV_FORCE, ENV_HOME, ENV_KIT = "MOSAIC_OPT", "MOSAIC_OPT_FORCE", "MOSAIC_OPT_HOME", "MOSAIC_OPT_KIT"
ENV_CACHE_ROOT, ENV_CACHE_DIR = "MOSAIC_OPT_CACHE_ROOT", "MOSAIC_OPT_CACHE_DIR"
ENV_TREE = "MODEL_OPT"                                            # run.sh convention: this model's directory (mosaic/)
ENV_TARGET_GPU = "MODEL_OPT_TARGET_GPU"
LIBRARY_CACHE_ENV = "MOSAIC_CACHE_DIR"                            # the library's own weights cache (src/mosaic/cache.py:11)
PACKAGE_ENV = (ENV_MODE, ENV_FORCE, ENV_CACHE_DIR)                # the package's own switches (one tuple: stripped from the stock arm, never exported)
PATH_ENV = (ENV_HOME, ENV_KIT, ENV_CACHE_ROOT)                    # path names the package reads in its own process; stripped from every arm with the MOSAIC_OPT prefix
CHECK_PINS_RELPATH = os.path.join("stock", "check_pins.py")       # the one place the commit-level pin check lives (run.sh calls the same file)
KIT_MARKER = os.path.join("tools", "public_design_run.py")        # the kit directory is found by its driver
PINS_RELPATH = os.path.join("stock", "PINS.json")
UPSTREAM_DISTS = ("mosaic", "joltz", "boltz")
STACK_DISTS = ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "equinox", "torch", "numpy")
CAMPAIGN_DIRNAME = "campaign"                                     # the in-process route's P1 directory under the cache root when no shape is named (a transparent row's: `campaign_<mode>`)
P1_ASIDE_REASON = "MOSAIC_OPT_CACHE_ROOT unset"                   # why a transparent P1 steps aside (the ACTIVE line's `p1=none(…)`, the resolver's aside["P1"])
WEIGHTS_RELPATH = os.path.join("boltz", "boltz2_conf.ckpt")       # under MOSAIC_CACHE_DIR (src/mosaic/losses/boltz2.py:50)
LOG = _report.PREFIX

MODEL_MODULE = "mosaic.models.boltz2"                             # where upstream's Boltz2 class lives
MODEL_CLASS = "Boltz2"
_INSTANCES: dict = {"state": None, "finder": None}
_REPORT: Optional[dict] = None
_ACTIVATING = False


# ----------------------------------------------------------------------------------------------------------------- paths
def tree_home() -> str:
    """This model's directory (mosaic/): MOSAIC_OPT_HOME, else MODEL_OPT, else derived from the package location (opt/mosaic_opt/ -> mosaic/)."""
    for var in (ENV_HOME, ENV_TREE):
        v = os.environ.get(var)
        if v:
            if not os.path.isfile(os.path.join(v, PINS_RELPATH)):
                raise FileNotFoundError(f"{var}={v} is not the mosaic/ tree ({PINS_RELPATH} missing)")
            return os.path.abspath(v)
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if os.path.isfile(os.path.join(here, PINS_RELPATH)):
        return here
    raise FileNotFoundError(f"the mosaic/ tree was not found around {__file__}: install with `pip install -e mosaic/opt` or set {ENV_HOME} / {ENV_TREE}")


def kit_home() -> str:
    """The kit directory: MOSAIC_OPT_KIT, else this package's own directory (opt/mosaic_opt/: tools/, mosaic_fast/ beside the modules),
    recognised by tools/public_design_run.py."""
    v = os.environ.get(ENV_KIT)
    here = os.path.abspath(v) if v else os.path.dirname(os.path.abspath(__file__))
    if not os.path.isfile(os.path.join(here, KIT_MARKER)):
        raise FileNotFoundError(f"{ENV_KIT + '=' + v if v else here} holds no {KIT_MARKER}")
    return here


def kit_file(rel: str) -> str:
    return os.path.join(kit_home(), rel)


def opt_home() -> str:
    """mosaic/opt/ — the directory of `pyproject.toml` (the package's project)."""
    return os.path.join(tree_home(), "opt")


def core_pin_path() -> str:
    """The pyproject whose `[tool.opt_core]` table pins the shared core (path, version, package tree sha): the file `mosaic_opt.core_gate` reads."""
    return os.path.join(opt_home(), "pyproject.toml")


def pins_path() -> str:
    return os.path.join(tree_home(), PINS_RELPATH)


@functools.lru_cache(maxsize=1)
def pins() -> dict:
    with open(pins_path(), "r", encoding="utf-8") as fh:
        return json.load(fh)


def stock_env_absent(p: Optional[dict] = None) -> List[str]:
    p = p or pins()
    return list((p.get("stock_environment") or {}).get("must_be_absent_prefixes") or [])


def stock_env_exceptions(p: Optional[dict] = None) -> List[str]:
    p = p or pins()
    return list((p.get("stock_environment") or {}).get("allowed_exceptions") or [])


def _check_pins_module():
    path = os.path.join(tree_home(), CHECK_PINS_RELPATH)
    spec = importlib.util.spec_from_file_location("mosaic_opt._check_pins", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------------------------- gates
def _dist_version(name: str) -> Optional[str]:
    """The installed distribution's version (no import of the package), or None: the core's reader (`opt_core.gates.dist_version`)."""
    from opt_core.gates import dist_version
    return dist_version(name)


def upstream_versions() -> Dict[str, Optional[str]]:
    out = {name: _dist_version(name) for name in UPSTREAM_DISTS}
    out.update({name: _dist_version(name) for name in STACK_DISTS})
    return out


def pins_gate(p: dict, force: bool = False) -> Tuple[bool, Dict[str, dict], Optional[str]]:
    """The commit-level pin check (stock/check_pins.py: pip's direct_url.json commit, or the archive it names) for the three
    upstream packages. Not at the pin -> refuse unless MOSAIC_OPT_FORCE=1 (then a warning). Returns (ok, detail per package, reason)."""
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



def forced_word(detail) -> Optional[str]:
    """`upstream_pins(<package>,…)` — the upstream packages a pins detail (`stock/check_pins.py` check(): {name: {pinned, commit, …}}) shows OFF
    their pinned commit, i.e. what `MOSAIC_OPT_FORCE=1` let through; None when every package is at its pin (or the detail has no per-package rows)."""
    off = sorted(n for n, d in (detail or {}).items() if isinstance(d, dict) and d.get("pinned") is False)
    return f"upstream_pins({','.join(off)})" if off else None


def forced_note(word: str) -> str:
    """The activation-line note of a forced run: the override is recorded on every line that carries the mode word (ACTIVE `notes=`, DONE `forced=`)."""
    return f"forced={word}: {ENV_FORCE}=1 overrides the upstream-commit pins — the libraries this run imports are not the pinned stock"

def version_gate(p: dict, force: bool = False) -> Tuple[bool, Dict[str, dict], Optional[str]]:
    """The stack pins the kit names (`pins_asserted_by_kit`: jax, jaxlib, jax-cuda12-plugin, jax-cuda12-pjrt, equinox at their exact
    versions) against the installed distributions: a NOTE when they differ, never a refusal — the levers engage wherever their mechanism
    applies; a lever that cannot run on this stack makes its mode refuse by name (levers.install), and the P1 files are keyed per stack. Returns
    (True, detail per distribution, the note or None); `force` is accepted for the callers' uniform signature and changes nothing here."""
    want = {k: (p.get("pins") or {}).get(k) for k in (p.get("pins_asserted_by_kit") or [])}
    detail, bad = {}, []
    for name, v in want.items():
        got = _dist_version(name)
        detail[name] = {"want": v, "got": got, "ok": got == v}
        if got != v:
            bad.append(f"{name} {got or 'not installed'} (kit pins {v})")
    if bad:
        return True, detail, "stack differs from the kit's pins: " + "; ".join(bad) + " — named, not refused: every lever of the mode engages, or the mode refuses by name"
    return True, detail, None


def installed_mosaic_dir(modules=None) -> Optional[str]:
    """The installed upstream `mosaic` package directory, found without importing it: `sys.modules` when it is already imported, else the
    path-based finder alone (`importlib.machinery.PathFinder`, never the meta path — the .pth hook's finder is not consulted and stays
    armed, and no module body runs). None when `mosaic` is not installed."""
    modules = sys.modules if modules is None else modules
    m = modules.get("mosaic")
    if m is not None and getattr(m, "__path__", None):
        return os.path.abspath(list(m.__path__)[0])
    spec = importlib.machinery.PathFinder.find_spec("mosaic")
    if spec is None or not spec.submodule_search_locations:
        return None
    return os.path.abspath(list(spec.submodule_search_locations)[0])


def installed_fast_check(kit: Optional[str] = None, mosaic_dir: Optional[str] = None) -> dict:
    """The kit's lever files as `run.sh install` places them (`site-packages/mosaic/fast/`, `mosaic_opt.leverfiles`) against the
    carried kit copies (`registry.KIT_FAST_DIR`): per file `present` | `absent`, plus any `.py` the kit does not ship (`extra`).
    `installed` is False when no `mosaic/fast/` directory exists (the driver's observers then import the kit's own files instead: the
    driver's `recipe.kit_modules()` call); `clean` is False when a kit file is absent from the installed copy (an extra `.py` there,
    which nothing of the kit imports, is recorded in `extra` and named as a note, not a refusal) — a present kit file is trusted as-is, with
    no byte-content check against the carried copy (the git commit names the bytes, not a runtime hash). What the driver route executes for P2 and the observers is this installed copy, so the activation gates, `check`
    and the stock entry (stock_design.py) all report it."""
    kit = kit or kit_home()
    src = os.path.join(kit, KIT_FAST_DIR)
    md = installed_mosaic_dir() if mosaic_dir is None else mosaic_dir
    fast = os.path.join(md, "fast") if md else None
    installed = bool(fast and os.path.isdir(fast))
    out = {"mosaic_dir": md, "path": fast if installed else None, "installed": installed, "kit_files": os.path.join(kit, KIT_FAST_DIR),
           "files": {}, "extra": [], "clean": True}
    if not installed:
        return out
    for fn in sorted(f for f in os.listdir(src) if f.endswith(".py")):
        got = os.path.join(fast, fn)
        if not os.path.isfile(got):
            out["files"][fn] = "absent"
            out["clean"] = False
        else:
            out["files"][fn] = "present"
    out["extra"] = sorted(f for f in os.listdir(fast) if f.endswith(".py") and f not in out["files"])   # named as a note by the gates; not a refusal
    return out


def installed_fast_label(fc: Optional[dict]) -> str:
    """One token for the activation lines: `not-installed` | `kit-bytes@<path>` | `kit-bytes+<n>extra@<path>` | `ABSENT(<files>)@<path>`."""
    if not fc:
        return "unknown"
    if not fc.get("installed"):
        return "not-installed"
    if fc.get("clean"):
        return f"kit-bytes{('+' + str(len(fc['extra'])) + 'extra') if fc.get('extra') else ''}@{fc.get('path')}"
    bad = [k for k, v in (fc.get("files") or {}).items() if v == "absent"]
    return f"ABSENT({','.join(bad)})@{fc.get('path')}"


def installed_fast_refusal(fc: dict) -> str:
    return (f"installed mosaic/fast at {fc.get('path')} is not the kit's lever files ({installed_fast_label(fc)}): the driver route would execute "
            f"that copy for P2 and the observers; place the kit's files again (bash run.sh install, or python -m mosaic_opt.leverfiles: from {fc.get('kit_files')}) or remove the directory")


GPU_KEYS = ("name", "cc", "sm", "memory_mib", "probe")          # the GPU's identifying fields (activation line, manifest, jit_cache_key)


def nvidia_smi_probe() -> dict:
    """GPU identity without jax: name, compute capability, `sm<digits>` and memory (MiB) of GPU 0 from nvidia-smi — the core's probe
    (`opt_core.gates.nvidia_smi_probe`) projected to GPU_KEYS; `probe` names the source or why there is none (the other fields then None)."""
    from opt_core.gates import nvidia_smi_probe as core_probe
    return core_probe(keys=GPU_KEYS)


def gpu_identity() -> dict:
    return nvidia_smi_probe()


def target_gpu_note(gpu: dict) -> Optional[str]:
    """`MODEL_OPT_TARGET_GPU` (configs/<gpu>.env) against the product name of the GPU seen: a note when they disagree (the P1 files are per
    GPU type; the cache key carries the full product name, so a wrong card compiles its own set rather than loading the wrong one). The
    card's memory size is recorded on every activation line (`report.gpu_label`), never compared here."""
    tg = os.environ.get(ENV_TARGET_GPU)
    name = gpu.get("name") or ""
    if not tg or not name:
        return None
    if tg.lower() not in name.lower():
        return f"{ENV_TARGET_GPU}={tg} but the GPU seen is {name}: the P1 files of this configuration belong to another GPU type"
    return None


WEIGHTS_MISSING_MARKER = "boltz2_conf.ckpt absent"                     # the one substring cmd_warm's dry-run filter matches to let
                                                                        # warm's own Step 0 (the sanctioned staging path) proceed


def data_path_gate(env: Optional[dict] = None) -> Tuple[Optional[str], List[str]]:
    """`MOSAIC_CACHE_DIR` (the weights cache: `boltz/boltz2_conf.ckpt` + `boltz/mols/`, stock/PINS.json "weights"). boltz has no offline
    switch, so every case short of "checkpoint present" is a refusal here: an unset variable resolves to the library's own default
    cache (`~/.cache/mosaic`, `src/mosaic/cache.py:11`) -- exactly where boltz's own unguarded download
    (`src/mosaic/losses/boltz2.py:51`) would land with no gate of ours in between; a value naming a directory that does not exist, or
    that exists but lacks the checkpoint, is the same gap. Stage it first through `tools/fetch_public_inputs.py --weights` (sha256
    against stock/PINS.json "weights"; `mosaic-opt warm`'s Step 0 runs this before this gate would matter to it -- see
    `WEIGHTS_MISSING_MARKER`, which `cmd_warm` filters out of its own dry-run gate for exactly this reason)."""
    env = os.environ if env is None else env
    notes: List[str] = []
    root = env.get(LIBRARY_CACHE_ENV)
    if root and not os.path.isdir(root):
        return (f"{LIBRARY_CACHE_ENV}={root} does not exist: set {LIBRARY_CACHE_ENV} to the weights cache holding {WEIGHTS_RELPATH} "
                f"(stock/PINS.json \"weights\", sha256)"), notes
    if not root:
        return (f"{LIBRARY_CACHE_ENV} unset: the library's default cache ~/.cache/mosaic (src/mosaic/cache.py:11) is exactly where "
                f"boltz's own unguarded download would land -- set {LIBRARY_CACHE_ENV} to the weights cache holding {WEIGHTS_RELPATH}"), notes
    elif not os.path.isfile(os.path.join(root, WEIGHTS_RELPATH)):
        return (f"{os.path.join(root, WEIGHTS_RELPATH)} {WEIGHTS_MISSING_MARKER}: refusing before design would reach boltz's own "
                f"download (boltz has no offline switch) -- stage it first through `mosaic-opt warm` (tools/fetch_public_inputs.py "
                f"--weights, sha256 against stock/PINS.json \"weights\")"), notes
    return None, notes


def cache_root(env: Optional[dict] = None) -> Optional[str]:
    env = os.environ if env is None else env
    return env.get(ENV_CACHE_ROOT) or None


def p1_state(cache_dir: Optional[str], form: str = "pinned") -> dict:
    """The P1 directory as it stands: {"cache_dir", "exists", "autotune_file", "autotune_present", "n_cache_entries", "autotune", "form"}
    (`autotune` = the phase for this directory: under the pinned form "load" when the autotune file exists, "dump" otherwise — the ONE
    populating process of the directory: `warm` on the driver route, the first process on the shared directory of the in-process route;
    passed to pcc explicitly, whose own `auto` is load-or-refuse; under the transparent form always "off" — the compilation cache alone,
    no autotune flag, whichever process compiles first fills it)."""
    if not cache_dir:
        return {"cache_dir": None, "exists": False, "autotune_file": None, "autotune_present": False, "n_cache_entries": 0, "autotune": None, "form": form}
    pcc = p1_lever()
    af = pcc.autotune_file_of(cache_dir)
    present = os.path.isfile(af)
    ident = pcc.identity_key(cache_dir, af)
    return {"cache_dir": cache_dir, "exists": os.path.isdir(cache_dir), "autotune_file": af, "autotune_present": present,
            "n_cache_entries": ident["n_cache_entries"], "autotune": ("load" if present else "dump") if form == "pinned" else "off", "form": form}


# ----------------------------------------------------------------------------------------------------------------- the P1 lever (shared)
def p1_lever():
    """P1's mechanism — the shared JAX persistent compilation cache + XLA autotune pin (`opt_core.jax_design.pcc`): the in-process
    route's `enable`, the exit tally's evidence fields, the autotune phase and the never-re-apply rule. The driver route exports the
    row instead (modes.ROWS: the same variables, `tests/test_core_adoption.py`); the kit's own `mosaic/fast/repro_cache.py` is the
    same mechanism for programs that import the kit directly."""
    from opt_core.jax_design import pcc
    return pcc


def backend_initialised() -> Tuple[bool, str]:
    """Whether a JAX backend exists in this process (XLA_FLAGS are read at backend initialisation): `pcc.backend_initialized` on the live jax."""
    jax = sys.modules.get("jax")
    if jax is None:
        return False, "jax not imported"
    if p1_lever().backend_initialized(jax) is True:
        return True, "jax backend initialised"
    return False, "jax imported, no backend yet"


def kit_levers_applied() -> List[str]:
    """What the process's own state shows already applied: P1 when XLA_FLAGS carries an autotune dump/load flag, when
    JAX_COMPILATION_CACHE_DIR is set, or when jax's `jax_compilation_cache_dir` config is set (the environment row, `pcc.enable()` or the
    kit's `repro_cache.enable()` did it) — `pcc.already_enabled`, the never-re-apply rule."""
    why = p1_lever().already_enabled()
    return [f"P1 ({why})"] if why else []


# ----------------------------------------------------------------------------------------------------------------- the instance counter
def _model_class():
    mod = sys.modules.get(MODEL_MODULE)
    cls = getattr(mod, MODEL_CLASS, None) if mod is not None else None
    return cls if isinstance(cls, type) else None


def _count_instances_of(cls, seed_from_gc: bool = False) -> bool:
    """Wrap `cls.__init__` so every construction from now on is counted: a weak reference to the instance is appended to a list after
    the original constructor has returned. The instance is never hashed, compared or inspected by the wrap — upstream's `Boltz2` is an
    `equinox.Module`, whose `__hash__` reads every field, and its fields do not exist before `__init__` has run (a `WeakSet` would
    hash it). Idempotent per class."""
    state = _INSTANCES["state"]
    if state is not None and state["cls"] is cls and state["wrapped"]:
        return True
    refs: list = []
    state = {"cls": cls, "wrapped": False, "built": 0, "refs": refs, "weakref": True}

    def record(obj) -> None:
        try:
            refs.append(weakref.ref(obj))
        except TypeError:                                  # the type takes no weak references: instance_check falls back to a gc scan
            state["weakref"] = False
    if seed_from_gc:
        import gc
        for o in gc.get_objects():
            if isinstance(o, cls):
                record(o)
    orig_init = cls.__init__

    @functools.wraps(orig_init)
    def __init__(self, *a, **kw):
        state["built"] += 1
        result = orig_init(self, *a, **kw)                 # the instance is complete only after this returns
        record(self)
        return result
    try:
        cls.__init__ = __init__
    except (TypeError, AttributeError):
        _INSTANCES["state"] = state
        return False
    state["wrapped"] = True
    _INSTANCES["state"] = state
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
            except Exception:
                spec = None
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            return None
        self.armed = False
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig):
            _orig(module)                              # the module body first, then the wrap on the class
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
    """Count Boltz2 instances from now on by the constructor wrap: on the class now when its module is imported (the instances that
    already exist are found once by a gc scan), otherwise at the module's import through a meta-path finder. Idempotent. Returns
    :func:`instance_check`."""
    cls = _model_class()
    if cls is not None:
        _count_instances_of(cls, seed_from_gc=True)
    elif _INSTANCES["finder"] is None or _INSTANCES["finder"] not in sys.meta_path:
        f = _ModelClassFinder()
        sys.meta_path.insert(0, f)
        _INSTANCES["finder"] = f
    return instance_check()


def instance_check() -> dict:
    """How many upstream `Boltz2` objects exist in this process and how that is known: {"n", "method": "counted"|"gc"|"none", "built"}."""
    cls = _model_class()
    if cls is None:
        return {"n": 0, "method": "none", "built": 0}
    st = _INSTANCES["state"]
    same = st is not None and st["cls"] is cls
    if same and st["wrapped"] and st["weakref"]:
        st["refs"][:] = [r for r in st["refs"] if r() is not None]      # the dead references dropped; the live ones counted
        return {"n": len(st["refs"]), "method": "counted", "built": st["built"]}
    import gc
    return {"n": sum(1 for o in gc.get_objects() if isinstance(o, cls)), "method": "gc", "built": st["built"] if same else 0}


def model_instances() -> int:
    return instance_check()["n"]


# ----------------------------------------------------------------------------------------------------------------- reports
def status() -> dict:
    if _REPORT is None:
        return {"active": False, "reason": "mosaic_opt.enable() has not run in this process"}
    return dict(_REPORT)

CALL_SITE_REASON = ("Boltz2() -> mosaic.fast.fastload.load_stock_fast_init() (--weights fastinit); model.binder_features(...) -> the driver's "
                    "--features-in/--features-sha inline load (the driver's `--features-in` branch through tools/recipe.py load_frozen_features, the in-line equivalent of frozen.load_features)")


def _base(mode: str, trigger: Optional[str]) -> dict:
    return {"active": False, "mode": mode, "route": None, "levers_applied": [], "levers_unavailable": [], "partial": False, "p1": {},
            "gpu": None, "upstream": upstream_versions(), "package_version": __version__, "trigger": trigger, "pid": os.getpid()}


def _gates(base: dict, force: bool) -> Tuple[Optional[str], Optional[dict]]:
    """Hard gates: kit present, pins present and upstream at its pinned commits (the one refusal environment drift can cause: a different
    stock is a different "stock"), the kit's lever files all installed, a visible GPU; `force` (MOSAIC_OPT_FORCE=1) is the recorded override of
    the upstream pins. The stack's own versions and extra files beside the installed lever files are NOTES (`base["gate_notes"]`), never refusals. The shared core's pin is not gated here:
    THE core pin gate already ran as statement one of the entry (`mosaic_opt.core_gate`, no override), and the report's `core`
    block is filled from that same producer. Returns (refusal, pins)."""
    from . import core_gate
    core = core_gate()                                                                      # the entry's gate passed: the same facts, not a second gate
    base["core"] = dict(core["installed"], ok=True, pinned=core["pinned"])                  # the opt_core this process imports: package_dir, root, version + the pin
    try:
        kit = kit_home()
    except FileNotFoundError as e:
        return str(e), None
    base["kit_home"] = kit
    if not os.path.isfile(pins_path()):
        return f"stock pins not found at {pins_path()}", None
    p = pins()
    ok, detail, why = pins_gate(p, force=force)
    base["pins"] = detail
    if not ok:
        return why, p
    _ok, vdetail, vnote = version_gate(p)
    base["stack_pins"] = vdetail
    base["gate_notes"] = [vnote] if vnote else []                                            # named on the activation line, never a refusal
    base["forced"] = forced_word(detail) if force else None                                 # MOSAIC_OPT_FORCE=1 let upstream off its pinned commit through: a recorded override, worded on ACTIVE notes= and DONE forced=
    if base["forced"]:
        base["gate_notes"].append(forced_note(base["forced"]))
    fc = installed_fast_check(kit)                                                          # the installed lever files are the kit's own (presence checked, not content)
    base["kit_install"] = fc
    if not fc["clean"]:
        return installed_fast_refusal(fc), p
    if fc.get("extra"):
        base["gate_notes"].append(f"the installed lever directory {fc.get('path')} carries {len(fc['extra'])} file(s) the kit does not ship ({','.join(fc['extra'])}): nothing of the kit imports them")
    gpu = gpu_identity()
    base["gpu"] = gpu
    if not gpu.get("name"):
        return f"no GPU visible ({gpu.get('probe')}): the kit's levers need one NVIDIA GPU", p
    return None, p



def levers_off() -> Tuple[str, ...]:
    """The ablation switch as this process reads it (`levers.levers_off`: MODEL_OPT_LEVERS_OFF) — a refusal by name (ActivationError) for an
    unknown id, so every route names it the same way."""
    from . import levers as _levers
    try:
        return _levers.levers_off()
    except _levers.LeverError as e:
        raise ActivationError(str(e)) from None


def mode_needs(mode: str, off: Tuple[str, ...] = ()) -> dict:
    """What composing `mode` needs on this box, the levers switched off by name (`off`: levers_off()) left out: {"p1": the row exports the
    P1 variables, "p1_form": ``pinned`` (a populate row + a warm shape: exact) | ``transparent`` (the compilation cache alone, filled by the
    first process, stepping aside without a cache root: fast, big) | None, "populate": the populate row a pinned P1 needs, "features": the
    row reads frozen features (P3), "per_step": the per-step levers `levers.install(mode)` applies, "flag_levers": the row's call-site levers
    (driver flags: they step aside by name in-process, DRIVER_ONLY), "levers_off": the row's levers `off` names} — read from the mode table, never
    from the mode's name."""
    lv = [l for l in KIT_MODES[mode]["levers"] if l not in off]
    form = p1_form(mode) if any(LEVERS[l].route == "env" for l in lv) else None            # a lever switched by the environment before the process starts = the P1 files
    return {"p1": form is not None, "p1_form": form, "populate": KIT_MODES[mode].get("populate") if form == "pinned" else None,
            "features": any("$F1" in (LEVERS[l].flag or "") for l in lv),                    # a lever whose flag reads the frozen-features placeholder
            "per_step": [l for l in install_levers_of(mode) if l not in off], "flag_levers": [l for l in lv if LEVERS[l].route == "flag"],
            "levers_off": [l for l in KIT_MODES[mode]["levers"] if l in off]}


def _creatable(d: str) -> bool:
    """Whether `d` could be created: its nearest existing ancestor is writable by this process (nothing is created)."""
    parent = os.path.dirname(os.path.abspath(d))
    while parent and not os.path.isdir(parent):
        nxt = os.path.dirname(parent)
        if nxt == parent:
            break
        parent = nxt
    return os.access(parent or ".", os.W_OK)


def transparent_dir_unusable(d: str) -> Optional[str]:
    """None when the transparent P1 directory exists or can be created; else the reason P1 steps aside by name (`MOSAIC_OPT_CACHE_ROOT not
    writable: …` — a read-only or foreign root). An existing directory this process cannot write is usable: its entries are read, and a
    failed entry write is jax's own warning, never an error."""
    if os.path.isdir(d):
        return None
    try:
        os.makedirs(d, exist_ok=True)
        return None
    except OSError as e:
        return f"{ENV_CACHE_ROOT} not writable ({e.strerror or type(e).__name__}: {d})"


def _p1_dir_for(cache_dir: Optional[str], form: str = "pinned", mode: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """The in-process route's P1 directory: the argument, else MOSAIC_OPT_CACHE_DIR, else <MOSAIC_OPT_CACHE_ROOT>/campaign (a transparent
    row's own: campaign_<mode>). (refusal, dir, aside): with no directory a pinned P1 is refused by name, a transparent one steps aside."""
    d = cache_dir or os.environ.get(ENV_CACHE_DIR)
    if not d:
        root = cache_root()
        if not root:
            if form == "transparent":
                return None, None, P1_ASIDE_REASON
            return (f"no P1 directory: set {ENV_CACHE_DIR} (this process's cache + autotune directory) or {ENV_CACHE_ROOT} "
                    f"(the P1 files' root, one per MODEL_OPT_STACK_KEY; README \"Variables\")"), None, None
        d = os.path.join(root, CAMPAIGN_DIRNAME if form == "pinned" else f"{CAMPAIGN_DIRNAME}_{mode}")
    d = os.path.abspath(d)
    if form == "transparent":
        why_aside = transparent_dir_unusable(d)
        if why_aside:
            return None, None, why_aside
    return None, d, None


def activate(mode: str, cache_dir: Optional[str] = None, strict: bool = False, trigger: Optional[str] = None, dry_run: bool = False,
             shape=None) -> dict:
    """Apply a mode once per process (idempotent). Returns the activation report; `strict` raises ActivationError when not active;
    `dry_run` resolves, gates and reports (report["dry_run"] = True) without exporting or applying anything."""
    mode = (mode or "").strip().lower()
    if mode not in MODES:
        raise ValueError(unknown_mode_message(mode))                  # a tier word with no lever wired is named as such
    if not dry_run:
        _disarm_autoload()
    if _REPORT is not None and not dry_run:
        if _REPORT.get("mode") == mode:
            if strict and not _REPORT.get("active") and mode != "off":
                raise ActivationError(_REPORT.get("reason") or "not active")
            return dict(_REPORT)
        reason = (f"already active as mode={_REPORT.get('mode')}; P1 is applied once per process (the compilation cache and the autotune "
                  f"flags are read at backend initialisation): restart to change mode")
        if strict:
            raise ActivationError(reason)
        return dict(_REPORT, active=False, refused=True, requested={"mode": mode}, reason=reason)
    if dry_run:
        return _dry_run(mode, strict, cache_dir=cache_dir, shape=shape)
    if _ACTIVATING:
        return {"active": False, "mode": mode, "reason": "activation already under way (re-entrant call ignored)"}
    return _activate_locked(mode, cache_dir, strict, trigger)


def _disarm_autoload() -> None:
    from . import _autoload
    _autoload.disarm()


def _activate_locked(mode: str, cache_dir: Optional[str], strict: bool, trigger: Optional[str]) -> dict:
    global _ACTIVATING
    _ACTIVATING = True
    try:
        return _activate_body(mode, cache_dir, strict, trigger)
    finally:
        _ACTIVATING = False


def _activate_body(mode: str, cache_dir: Optional[str], strict: bool, trigger: Optional[str]) -> dict:
    global _REPORT
    base = _base(mode, trigger)
    base["route"] = "in-process"
    if mode == "off":
        _REPORT = dict(base, reason="mode off: stock mosaic (the upstream API; no environment set, no lever applied)")
        return dict(_REPORT)

    def refuse(reason: str) -> dict:
        global _REPORT
        _REPORT = dict(base, reason=reason)
        _report.log_activation(_REPORT); _REPORT["logged"] = True
        if strict:
            raise ActivationError(reason)
        return dict(_REPORT)

    force = bool(os.environ.get(ENV_FORCE))
    why, p = _gates(base, force)
    if why:
        return refuse(why)
    from .det import precision_refusal
    why = precision_refusal()
    if why:
        return refuse(why)
    try:
        off = levers_off()                                                                  # the ablation switch (MODEL_OPT_LEVERS_OFF): an unknown id is a refusal by name
    except ActivationError as e:
        return refuse(str(e))
    needs = mode_needs(mode, off)
    applied = kit_levers_applied() if needs["p1"] else []                                   # the late-activation rule (3): the kit's own state
    if applied:
        return refuse("P1 already on in this process by " + "; ".join(applied) + ": the package never re-applies (unset the variables or drop MOSAIC_OPT)")
    init, how = backend_initialised()                                                       # (1): XLA_FLAGS and the levers' declared variables are read at backend initialisation; per-step levers precede any trace
    if init:
        return refuse(f"{how}: the mode's levers must be installed before the first jax computation (pcc.enable and the per-step levers refuse past it); activate earlier or restart")
    why, d, aside = _p1_dir_for(cache_dir, needs["p1_form"], mode) if needs["p1"] else (None, None, None)
    if why:
        return refuse(why)
    # data_path_gate is a driver-route-only concern (see _dry_run's matching comment): _activate_body is ALWAYS the in-process route
    # (it takes no shape at all) and only ever applies P1 -- it never loads the model, so it never reaches boltz's download either.
    notes: List[str] = list(base.get("gate_notes") or [])                                   # stack-version drift / extra installed files: named here, never refused
    chk = register_instance_counter()                                                       # (2), after every gate: the constructor wrap from here on
    if chk["n"]:                                                                            # a model exists
        return refuse(f"{chk['n']} Boltz2 instance(s) already exist in this process ({chk['method']}): their executables compile without the "
                      f"persistent cache; enable() before the model is built, or run through `mosaic-opt design`")
    st = p1_state(d, needs["p1_form"])
    driver_only = {l: DRIVER_ONLY for l in needs["flag_levers"]}                           # the row's call-site levers (P2, P3: driver flags): replacements only the program's own calls could make —
                                                                                            # they step aside BY NAME on this route (row=…-aside[P2:driver_only]); never a partial activation, never an exit
    res = resolve(mode, cache_dir=d or "$C1", features="$F1", features_sha="$FSHA", phase=st["autotune"] or "warm", levers_off=off, p1_aside=aside, aside=driver_only)   # the row the in-process route stands in for
    applied_now: List[str] = []
    p1_rep: dict = {"aside": aside} if aside else {}
    info: dict = {}
    if needs["p1"] and d:
        try:
            info = p1_lever().enable(d, autotune=st["autotune"])                           # P1 in this process: the cache + the autotune pin (pinned: dump if absent, load otherwise; transparent: the cache alone, autotune "off")
        except Exception as e:  # noqa: BLE001
            if is_oom(e): raise                                                           # an out-of-memory is the caller's to see, never a NOT ACTIVE report the program continues past on stock
            return refuse(f"pcc.enable({d!r}) failed: {type(e).__name__}: {e}")
        applied_now.append("P1")
        p1_rep = {"cache_dir": d, "autotune_file": info.get("autotune_file"), "autotune": st["autotune"], "xla_flags": info.get("xla_flags"),
                  "form": needs["p1_form"], "activation": "opt_core.jax_design.pcc.enable()"}
    lever_state: dict = {}
    if needs["per_step"]:
        from . import levers as _levers
        try:
            lever_state = _levers.install(mode)                                              # the per-step levers, once, before any trace (the ONE installer; LEVER lines printed there)
        except _levers.LeverError as e:
            return refuse(str(e))
        applied_now += list(lever_state["levers"])
    unavailable: List[str] = []                                                             # a lever this route SHOULD apply and did not — what `partial=` reports (exit 3 at the hook unless
                                                                                            # MOSAIC_OPT_ALLOW_PARTIAL=1): none is left silent here, pcc.enable() and levers.install() refuse by name above
    stood_in = " + ".join((["pcc.enable()"] if (needs["p1"] and d) else []) + (["levers.install()"] if needs["per_step"] else [])) or "nothing"
    rep = dict(base, active=True, levers_applied=applied_now, levers_unavailable=unavailable, partial=bool(unavailable), allow_partial=_report.allow_partial_env(),
               p1=p1_rep, row=res.row, row_line=f"{describe_line(res)} (stood in for in-process by {stood_in})", notes=notes + list(n for n in res.notes if "steps aside" in n or "switched off" in n),
               levers_off=list(res.levers_off), levers_aside=dict(res.aside), levers_driver_only=list(driver_only),
               instances=chk, levers={k: LEVERS[k].title for k in res.levers},
               lever_lines=list(lever_state.get("lines", [])), levers_effective=dict(lever_state.get("effective", {})),
               env_required=dict(lever_state.get("env_required", {})))
    if driver_only:
        rep["driver_only_reason"] = CALL_SITE_REASON                                        # what the kit driver does for them (`mosaic-opt design`): recorded on the report, named on the line, never gated
    tg = target_gpu_note(base["gpu"] or {})
    if tg:
        rep["notes"] = list(notes) + [tg]
    _REPORT = rep
    _report.log_activation(_REPORT); _REPORT["logged"] = True
    _report.register_exit_tally(lambda: _report.tally_line(mode, "in-process", d, autotune=st["autotune"], via=info.get("applied_via"), why_none=aside))
    return dict(_REPORT)


def _dry_run(mode: str, strict: bool, cache_dir: Optional[str] = None, shape=None) -> dict:
    """`check`: resolve and gate without applying. For the driver route (a shape given) the report carries the row's environment and
    flags, the shape's features and P1 state under the cache root; otherwise the in-process route's P1 directory."""
    base = _base(mode, None)
    base["dry_run"] = True
    if mode == "off":
        base["route"] = "driver"
        base["reason"] = "mode off: stock mosaic (the kit driver with every lever off in a clean subprocess; no environment set, no lever applied)"
        try:
            base["kit_home"] = kit_home()
            base["kit_install"] = installed_fast_check(base["kit_home"])
        except FileNotFoundError as e:
            base["reason"] = str(e)
        res = resolve("off")
        base.update(row=res.row, row_line=describe_line(res), levers_planned=[], env={}, flags=res.flags)
        base["stock_env_absent"] = stock_env_absent(pins()) if os.path.isfile(pins_path()) else None
        base["gpu"] = gpu_identity()
        if shape is not None:
            base["shape"] = {**shape.record(), "tokens": None}
            try:
                from .inputs import public_target
                base["shape"]["tokens"] = shape.tokens(public_target(kit_home())["length"])
            except Exception:  # noqa: BLE001
                pass
        why, notes = data_path_gate()
        base["notes"] = list(notes) + ([why] if why else [])
        return base
    would: List[str] = []
    force = bool(os.environ.get(ENV_FORCE))
    why, p = _gates(base, force)
    if why:
        would.append(why)
    why = None
    from .det import precision_refusal
    why = precision_refusal()
    if why:
        would.append(why)
    from .inputs import features_path, p1_dir
    root = cache_root()
    notes: List[str] = list(base.get("gate_notes") or [])                                   # stack-version drift / extra installed files: named, never refused
    try:
        off = levers_off()
    except ActivationError as e:
        off = (); would.append(str(e))
    needs = mode_needs(mode, off)
    base["levers_off"] = list(needs["levers_off"])
    if needs["per_step"]:                                                                   # the per-step levers' plan and declared environment, without installing anything
        from . import levers as _levers
        try:
            base["levers_plan"] = _levers.plan(mode)
            base["env_required"] = _levers.env_required(mode)
            wrong = [f"{k}={v!r} (found {os.environ.get(k)!r})" for k, v in base["env_required"].items() if shape is None and os.environ.get(k) != v]
            if wrong:                                                                       # the in-process route needs them exported already; the driver route exports the row's own
                would.append("lever_env_required: export before the interpreter starts: " + "; ".join(wrong))
        except _levers.LeverError as e:
            would.append(str(e))
    if shape is not None:
        base["route"] = "driver"
        # data_path_gate is a driver-route-only concern: only the driver route (a real design/warm through the CLI) can ever reach
        # boltz's model load and its unguarded download; the in-process route (shape is None, enable()/the .pth hook) never applies
        # P2/P3 and never loads the model at all, so the weights check does not belong in its would_refuse_all.
        why, notes = data_path_gate()
        if why:
            would.append(why)
        feats, d, aside = None, None, None
        pinned = needs["p1_form"] == "pinned"
        if (needs["p1"] and pinned) or needs["features"]:
            if not root:
                would.append(f"{ENV_CACHE_ROOT} unset (README \"Variables\"): the {'P1 files' if needs['p1'] else 'frozen features'} of this shape live under it")
            else:
                feats = features_path(root, shape) if needs["features"] else None
                d = p1_dir(root, shape) if needs["p1"] else None
        elif needs["p1"]:                                                                   # transparent: the shape's own cache directory under the root, or aside by name (no root; a root this process cannot create it under)
            d, aside = (p1_dir(root, shape, mode), None) if root else (None, P1_ASIDE_REASON)
            if d and not os.path.isdir(d) and not _creatable(d):
                d, aside = None, f"{ENV_CACHE_ROOT} not writable ({d})"                    # check never creates: it reports what design would do
        fs = features_state(feats)
        st = p1_state(d, needs["p1_form"] or "pinned")
        res = resolve(mode, cache_dir=d or "$C1", features=feats or "$F1", features_sha=fs["sha256"] or "$FSHA",
                      phase="warm" if (st["autotune_present"] or not pinned) else "populate", levers_off=off, p1_aside=aside)
        base.update(shape={**shape.record(), "tokens": None},
                    features=fs, p1=dict(st, aside=aside) if aside else st, env=res.env, flags=res.flags, levers_aside=dict(res.aside))
        try:
            from .inputs import public_target
            base["shape"]["tokens"] = shape.tokens(public_target(kit_home())["length"])
        except Exception:  # noqa: BLE001
            pass
        if (pinned and not st["autotune_present"]) or (needs["features"] and not fs["present"]):
            notes = list(notes) + [f"shape {shape.key} not warm under {root or ENV_CACHE_ROOT}: a cache miss, named — `design` runs it as stock compiles and featurizes (P1, P3 step aside by name); `mosaic-opt warm` fills it ahead of time"]
    else:
        base["route"] = "in-process"
        why, d, aside = _p1_dir_for(cache_dir, needs["p1_form"], mode) if needs["p1"] else (None, None, None)
        if why:
            would.append(why)
        st = p1_state(d, needs["p1_form"] or "pinned")
        driver_only = {l: DRIVER_ONLY for l in needs["flag_levers"]}                       # the row's call-site levers step aside by name on this route (as `enable()` reports them): not planned, not partial
        res = resolve(mode, cache_dir=d or "$C1", features="$F1", features_sha="$FSHA", phase=st["autotune"] or "warm", levers_off=off, p1_aside=aside, aside=driver_only)
        base.update(p1=dict(st, aside=aside) if aside else st, env=res.env, flags=res.flags, levers_unavailable=[], partial=False, allow_partial=False,
                    levers_aside=dict(res.aside), levers_driver_only=list(driver_only))
        if driver_only:
            base["driver_only_reason"] = CALL_SITE_REASON
        applied = kit_levers_applied() if needs["p1"] else []
        if applied:
            would.append("P1 already on in this process by " + "; ".join(applied))
    tg = target_gpu_note(base.get("gpu") or {})
    if tg:
        notes = list(notes) + [tg]
    base.update(row=res.row, row_line=describe_line(res), levers_planned=list(res.levers), notes=list(notes) + list(res.notes),
                would_refuse=would[0] if would else None, would_refuse_all=would, reason=("would refuse: " + would[0]) if would else "dry run: nothing applied")
    if strict and would:
        raise ActivationError(would[0])
    return base

