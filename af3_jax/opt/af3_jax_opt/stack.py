"""Paths, the box's stack key, the gates, and the activation report. This model is a command line, run by a separate interpreter; the
package never imports ``alphafold3`` or ``jax`` — activation instead composes a command: ``off`` = the stock script; ``exact`` = the kit's
script plus its flag row and the Pallas add-on's levers; ``fast`` = the same script through the FlashPairformer add-on's launcher. The
outputs repeat bit for bit only within one autotune class: one cache shared by ``exact`` and ``off`` under ``detrecipe``, its own for ``fast``.
There is no import-time hook on the JAX path: setting ``AF3_JAX_OPT`` for a caller that runs ``run_alphafold.py`` directly changes nothing.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Sequence

try:
    from opt_core import gates as _gates
except ImportError as _e:  # the core is installed beside the kit: pip install -e common/opt_core -e af3_jax/opt
    raise ImportError(f"opt_core not importable ({_e}): install the core beside this kit (pip install -e common/opt_core)", name=getattr(_e, "name", None) or "opt_core") from _e

ENV_PREFIX = "AF3_JAX_"                                                # the package's prefix (stock/PINS.json stock_proof.prefix_owners)
ENV_MODE, ENV_VARIANT = "AF3_JAX_OPT", "AF3_JAX_VARIANT"
ENV_HOME, ENV_TREE = "AF3_JAX_OPT_HOME", "MODEL_OPT"
ENV_REPO, ENV_PY, ENV_PARAMS_ROOT, ENV_CACHE_ROOT = "AF3_JAX_REPO", "AF3_JAX_PY", "AF3_JAX_PARAMS_ROOT", "AF3_JAX_CACHE_ROOT"
ENV_N_GPU = f"{ENV_PREFIX}N_GPU"                                          # the memory mode's resource axis (--n_gpu P; the command line wins, this is its environment route and the model process's reading)
LEVER_SWITCH_ENV = (f"{ENV_PREFIX}DATTN", f"{ENV_PREFIX}TTR", f"{ENV_PREFIX}TRIATT_XLA", f"{ENV_PREFIX}SAMPLER_BF16", f"{ENV_PREFIX}ATOM_ATTN", f"{ENV_PREFIX}TRIMUL_CD", f"{ENV_PREFIX}LNP", f"{ENV_PREFIX}HOIST_LOGITS", f"{ENV_PREFIX}COND_SHARE", f"{ENV_PREFIX}ATOM_COND_HOIST")   # the tree levers' model-process switches (modes.TREE_LEVER_ENV; inprocess/{dattn,ttr}.py ENV_SWITCH): written by the wrapper for a composition that names the lever, read inside the model process — declared, so an interpreter that carries the kit's start-up hook accepts them
DECLARED_ENV = (ENV_MODE, ENV_VARIANT, ENV_HOME, ENV_REPO, ENV_PY, ENV_PARAMS_ROOT, ENV_CACHE_ROOT, ENV_N_GPU) + LEVER_SWITCH_ENV   # every AF3_JAX_* name the kit reads (_autoload.py spells the same set); any other is refused, never silently ignored


def undeclared_env(environ=None) -> List[str]:
    """AF3_JAX_* names set in the caller's environment that the package does not read (a mistyped switch, a memory-lever switch name: the
    memory mode's levers have no environment switch at all)."""
    environ = os.environ if environ is None else environ
    return sorted(k for k in environ if k.startswith(ENV_PREFIX) and k not in DECLARED_ENV)
ENV_JIT_ROOT = "MODEL_OPT_JIT_ROOT"                                    # the tree-wide compile-cache root run.sh exports (its jit-cache block): stands in for an unset AF3_JAX_CACHE_ROOT
DEFAULT_CACHE_ROOT = os.path.abspath(os.path.join(os.environ.get("TMPDIR") or "/tmp", f"model_opt_jit-uid{os.geteuid()}"))   # neither variable set: the per-user root run.sh's block also makes — this user's alone (mode 0700), checked before use (default_root_refusal); upstream's own default is /tmp/alphafold_cache (run_alphafold.py:350)
PRIVATE_CACHE_PREFIX = "af3_jax_opt_cache-"                            # the name prefix of the private temporary root a process compiles into when the per-user root is refused
# the pinned image's checkout and interpreter are read from stock/PINS.json image.repo_dir / image.python (repo_dir(), venv_python())
KITS = {                                                               # the kit directories carried under opt/, each named by the file that identifies it
    "fast_inference": (os.path.join("forward", "fast_inference"), os.path.join("patches", "patched_files", "run_alphafold.py")),
    "pallas": (os.path.join("forward", "pallas_addon"), os.path.join("patches", "af3_pallas_levers.py")),
    "fpf": (os.path.join("forward", "flashpairformer"), os.path.join("af3_flashpairformer", "__init__.py")),
}
STOCK_SCRIPT, FAST_SCRIPT = "run_alphafold.py", "run_alphafold_fast.py"
PINS_RELPATH = os.path.join("stock", "PINS.json")
MIN_MEMORY_GB, MIN_COMPUTE_CAP = 24.0, 8.0                             # the kit's reference hardware line (bf16 flash attention needs Ampere or newer; the kit's tested GPUs carry 40-80 GB): a device below it is NAMED on a GPU NOTE line, never refused (gpu_notes)
SRC_PREFIX = "src/alphafold3/"                                         # stock files under it live in the install's site-packages/alphafold3/

_REPORT: Optional[dict] = None
_LAUNCHED: List[dict] = []                                            # (mode, variant) of every model process launched by this interpreter
_CACHE: Dict[str, object] = {}


class ActivationError(RuntimeError):
    """A second, different activation after a model process was launched in this interpreter (the late-activation rule)."""


# ----------------------------------------------------------------------------------------------------------------- paths
def opt_home() -> str:
    """af3_jax/opt — the directory the package is installed from (editable), or $AF3_JAX_OPT_HOME."""
    h = os.environ.get(ENV_HOME)
    return os.path.abspath(h) if h else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def core_dir() -> str:
    """common/opt_core — the directory holding the pinned `opt_core` package this interpreter imports (the gate held it to the kit's
    [tool.opt_core] pin at start-up). The model process reaches the core's CARRIED KERNELS by this path (model_process_env puts it on
    PYTHONPATH for the lever modes; the fork's interpreter never has the core installed): the FlashPairformer add-on imports opt_core.kernels.fpf_pallas,
    the Pallas add-on's levers module imports opt_core.kernels.pallas_glut, mode big imports opt_core.mem."""
    import opt_core
    return os.path.dirname(os.path.dirname(os.path.abspath(opt_core.__file__)))


def tree_home() -> str:
    """af3_jax/ — the model's directory ($MODEL_OPT when set, else the parent of opt_home())."""
    h = os.environ.get(ENV_TREE)
    return os.path.abspath(h) if h else os.path.dirname(opt_home())


def kit_home(name: str = "fast_inference") -> str:
    """The add-on directory `name` (a key of the add-on table above) under opt/, checked by the file that identifies it."""
    if name not in KITS:
        raise KeyError(f"unknown kit {name!r}; kits: {' | '.join(KITS)}")
    rel, marker = KITS[name]
    d = os.path.join(opt_home(), rel)
    if not os.path.isfile(os.path.join(d, marker)):
        raise FileNotFoundError(f"kit {name!r} not found: no {marker} under {d} (install the package editable from af3_jax/opt, or set {ENV_HOME})")
    return d


def pins_path() -> str:
    return os.path.join(tree_home(), PINS_RELPATH)


def pins() -> dict:
    if "pins" not in _CACHE:
        with open(pins_path(), encoding="utf-8") as f:
            _CACHE["pins"] = json.load(f)
    return _CACHE["pins"]  # type: ignore[return-value]


def repo_dir() -> str:
    """The fork checkout that carries run_alphafold.py (and, after `warm`/`pred exact|fast`, run_alphafold_fast.py beside it):
    $AF3_JAX_REPO, else the pinned image's (stock/PINS.json image.repo_dir)."""
    return os.path.abspath(os.environ.get(ENV_REPO) or pins()["image"]["repo_dir"])


def venv_python() -> str:
    """The interpreter of the fork's environment: $AF3_JAX_PY, else the pinned image's (stock/PINS.json image.python); never this one."""
    return os.environ.get(ENV_PY) or pins()["image"]["python"]


def config_exports() -> str:
    """Shell lines for configs/*.env: the pinned image's checkout, interpreter and XLA environment (stock/PINS.json image), each
    exported only where the caller has not set it — `export K="${K:-<pinned value>}"`. The one derivation of those values in shell."""
    im = pins()["image"]
    kv = [(ENV_REPO, im["repo_dir"]), (ENV_PY, im["python"])] + list(im.get("env", {}).items())
    return "\n".join(f'export {k}="${{{k}:-{v}}}"' for k, v in kv)


def params_root() -> Optional[str]:
    r = os.environ.get(ENV_PARAMS_ROOT)
    return os.path.abspath(r) if r else None


def default_root_refusal(path: str) -> Optional[str]:
    """Why the per-user default root ``path`` cannot hold this user's caches — None when it is a real directory (not a symbolic link) owned by
    this process's uid with no group / other write bit; made here (mode 0700, its parent as needed) when absent. Serialized XLA executables and
    the weights digest memo are read back from the root as found, so a directory another account made, owns or can write is nothing to read
    from or write through: the owner and mode are the check (mkdir first, then lstat what is there — whoever made it first is who owns it)."""
    uid = os.geteuid()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    except OSError as e:
        return f"{path} cannot be made ({e.strerror or type(e).__name__}); fix: set {ENV_CACHE_ROOT} to a writable root of your own"
    try:
        st = os.lstat(path)
    except OSError as e:
        return f"{path} cannot be read ({e.strerror or type(e).__name__}); fix: set {ENV_CACHE_ROOT} to a writable root of your own"
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        return f"{path} is a symbolic link or not a directory; fix: remove it, or set {ENV_CACHE_ROOT} to a root of your own"
    if st.st_uid != uid:
        return f"{path} belongs to uid {st.st_uid}, not to this process (uid {uid}); fix: remove it, or set {ENV_CACHE_ROOT} to a root of your own"
    if st.st_mode & 0o022:
        return f"{path} is writable by group or other (mode {st.st_mode & 0o7777:04o}); fix: chmod go-w {path}"
    return None


def cache_root() -> str:
    """The compile-cache root: $AF3_JAX_CACHE_ROOT, else $MODEL_OPT_JIT_ROOT (run.sh exports one: the image's cache in place, a root of your
    own, or the per-user root it made), else the per-user root DEFAULT_CACHE_ROOT (${TMPDIR:-/tmp}/model_opt_jit-uid<uid>) — used only when
    it is this user's own private directory (default_root_refusal). Refused, it is named once on a `CACHE REFUSED` line and this process
    compiles into a fresh private temporary root instead (tempfile.mkdtemp, mode 0700; this run only): nothing under the refused root is read."""
    named = os.environ.get(ENV_CACHE_ROOT) or os.environ.get(ENV_JIT_ROOT)
    if named:
        return os.path.abspath(named)
    if "cache_root" not in _CACHE:
        why = default_root_refusal(DEFAULT_CACHE_ROOT)
        root = DEFAULT_CACHE_ROOT
        if why is not None:
            root = os.path.abspath(tempfile.mkdtemp(prefix=PRIVATE_CACHE_PREFIX))
            from .report import PREFIX                                        # late: report imports this module
            print(f"{PREFIX} CACHE REFUSED {DEFAULT_CACHE_ROOT}: {why} — nothing under it is read or written; this process compiles into {root} "
                  f"(private, this run only; {ENV_CACHE_ROOT} names a persistent root of your own)", file=sys.stderr, flush=True)
        _CACHE["cache_root"] = root
    return _CACHE["cache_root"]  # type: ignore[return-value]


sha256_file = _gates.sha256_file                                      # the one file digest (opt_core.gates)


# ------------------------------------------------------------------------------------------------------------ the box's key
def gpu_info(index: int = 0) -> Optional[dict]:
    """The first GPU's name, memory (MiB) and compute capability from nvidia-smi; None when there is no GPU or no nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    if out.returncode != 0 or len(lines) <= index:
        return None
    name, mem, cc = [x.strip() for x in lines[index].split(",")[:3]]
    try:
        return {"name": name, "memory_mib": int(float(mem)), "compute_cap": float(cc), "count": len(lines)}
    except ValueError:
        return {"name": name, "memory_mib": None, "compute_cap": None, "count": len(lines)}


def jax_build(py: Optional[str] = None) -> Optional[dict]:
    """jax / jaxlib / python versions of the fork's interpreter (a subprocess that imports jax, not jax.devices(): no GPU is touched)."""
    py = py or venv_python()
    key = ("jax_build", py)
    if key in _CACHE:
        return _CACHE[key]  # type: ignore[return-value]
    if not os.path.isfile(py):
        _CACHE[key] = None
        return None
    code = ("import json, sys, importlib.metadata as m\n"
            "def v(n):\n"
            "    try: return m.version(n)\n"
            "    except m.PackageNotFoundError: return None\n"
            "print(json.dumps({'python': sys.version.split()[0], 'jax': m.version('jax'), 'jaxlib': m.version('jaxlib'), "
            "'alphafold3': v('alphafold3'), 'executable': sys.executable}))")
    try:
        out = subprocess.run([py, "-I", "-c", code], capture_output=True, text=True, timeout=120)
        res = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError, ValueError):
        res = None
    _CACHE[key] = res
    return res


def slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in ".+-" else "_" for c in s.strip())


def stack_reasons(repo: str, python: str, build: Optional[dict]) -> List[str]:
    """Why the configured fork checkout ($AF3_JAX_REPO) and interpreter ($AF3_JAX_PY) cannot run the model, each named by its variable —
    [] when the checkout carries run_alphafold.py and the interpreter exists and imports jax (``build`` = jax_build() of that interpreter).
    The one capability check of the stack; activation (enable) reads it."""
    reasons: List[str] = []
    if not os.path.isfile(os.path.join(repo, STOCK_SCRIPT)):
        reasons.append(f"no {STOCK_SCRIPT} under {ENV_REPO}={repo}")
    if not os.path.isfile(python):
        reasons.append(f"no interpreter at {ENV_PY}={python}")
    elif not build:
        reasons.append(f"{python} cannot import jax (not the fork's environment?)")
    return reasons


def cache_key(gpu: Optional[dict] = None, build: Optional[dict] = None) -> Optional[str]:
    """The autotune-class key of this box: (GPU model, jax build) — the two the cache is valid for (caches and executables
    are per (jax/jaxlib build, GPU model, model config)). None without a GPU or a build."""
    gpu = gpu if gpu is not None else gpu_info()
    build = build if build is not None else jax_build()
    if not gpu or not build:
        return None
    return f"{slug(gpu['name'])}__jax{build['jax']}_jaxlib{build['jaxlib']}"


def cache_dir(key: Optional[str] = None, mode: str = "off", region: Optional[str] = None) -> Optional[str]:
    """The cache class a run reads: <cache root>/<key><cache_class of the mode> — exact has <key>/, fast <key>__fast/, big
    <key>__big-big-<sha8 of its lever set> (modes.cache_class_of); ``off`` runs with the EMPTY class (``--cache_dir=``: the JAX cache at
    ./jax under the pass's output directory, settings.CWD_JAX_CACHE) unless the caller names a ``--cache_dir`` (cli.pred takes that one)."""
    from .modes import cache_class_of
    from . import settings as _s
    if mode == "off":
        return _s.CWD_JAX_CACHE                                            # ./jax under the pass's output dir (cli.pred resolves it) — or the caller's own --cache_dir
    key = key or cache_key()
    return os.path.join(cache_root(), key + cache_class_of(mode, region=region)) if key else None


def executables_dir(cdir: str) -> str:
    return os.path.join(cdir, "executables")                           # serialized-executable files a class may hold (counted on the ACTIVE / CACHE / WARM lines)


def executables(cdir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(executables_dir(cdir), "exec_b*.bin")))


def gate_gpu(gpu: Optional[dict]) -> Optional[str]:
    """None when a GPU is visible (the one thing no mode can run without); else the reason. A device below the kit's reference line is not
    a reason — gpu_notes names it."""
    if not gpu:
        return "no NVIDIA GPU visible (nvidia-smi)"
    return None


def gpu_notes(gpu: Optional[dict]) -> List[str]:
    """The device facts below the kit's reference hardware line (MIN_MEMORY_GB, MIN_COMPUTE_CAP), each in the words of ONE `GPU NOTE` line —
    named, never a refusal: the modes run, a lever that cannot run on the part says so on its own line."""
    notes: List[str] = []
    if not gpu:
        return notes
    mem_gb = (gpu["memory_mib"] or 0) * 1048576 / 1e9
    if gpu["memory_mib"] is not None and mem_gb < MIN_MEMORY_GB:
        notes.append(f"device memory {mem_gb:.1f} GB < {MIN_MEMORY_GB:g} GB ({gpu['name']}): below the kit's reference line — the modes run, the size that fits is the device's")
    if gpu["compute_cap"] is not None and gpu["compute_cap"] < MIN_COMPUTE_CAP:
        notes.append(f"compute capability {gpu['compute_cap']} < {MIN_COMPUTE_CAP} ({gpu['name']}): below the kit's reference line (Ampere or newer for the fork's Triton flash attention and the kit's Pallas kernels) — the modes run; a kernel that cannot run on this part names itself on its line")
    return notes


# --------------------------------------------------------------------------------------------------------------- the install
def _site_dir(py: Optional[str] = None) -> Optional[str]:
    py = py or venv_python()
    key = ("site", py)
    if key in _CACHE:
        return _CACHE[key]  # type: ignore[return-value]
    res = None
    if os.path.isfile(py):
        try:
            out = subprocess.run([py, "-I", "-c", "import alphafold3, os; print(os.path.dirname(alphafold3.__file__))"],
                                 capture_output=True, text=True, timeout=120)
            res = out.stdout.strip().splitlines()[-1] if out.returncode == 0 and out.stdout.strip() else None
        except (OSError, subprocess.SubprocessError):
            res = None
    _CACHE[key] = res
    return res


def stock_file_path(rel: str, repo: Optional[str] = None, site: Optional[str] = None) -> Optional[str]:
    """Where a stock file of stock/PINS.json stock_files lives in the install: top-level scripts in the checkout, src/alphafold3/** in the
    installed package."""
    if rel.startswith(SRC_PREFIX):
        return os.path.join(site, rel[len(SRC_PREFIX):]) if site else None
    return os.path.join(repo or repo_dir(), rel)


def install_state(repo: Optional[str] = None, py: Optional[str] = None) -> dict:
    """Whether the install carries the stock files the add-ons patch or pin (stock/PINS.json stock_files): per file `present` | `missing`;
    overall `present` when every file is, `absent` when none is, else `mixed`. Presence only (the files are the checked-out git tree, not
    a stored digest): the tree overlays nothing — the pinned stock is stock/src (the fork at the pin with stock/patches applied at
    image build), the kit's script is a second file beside it."""
    repo = repo or repo_dir()
    site = _site_dir(py)
    per_file, counts = {}, {"present": 0, "missing": 0}
    for rel in pins().get("stock_files", ()):
        path = stock_file_path(rel, repo, site)
        state = "present" if path and os.path.isfile(path) else "missing"
        per_file[rel] = state; counts[state] += 1
    n = len(per_file)
    overall = "present" if counts["present"] == n else ("absent" if counts["missing"] == n else "mixed")
    return {"install": overall, "repo": repo, "site": site, "files": per_file, "counts": counts,
            "run_alphafold_fast": os.path.isfile(os.path.join(repo, FAST_SCRIPT))}


# ------------------------------------------------------------------------------------------------------------ activation
def status() -> dict:
    return dict(_REPORT) if _REPORT else {"active": False, "reason": "no activation in this process"}


def launched(mode: str, variant: str, n_gpu: int = 1, region: Optional[str] = None) -> None:
    _LAUNCHED.append({"mode": mode, "variant": variant, "n_gpu": int(n_gpu), "region": region})


PARTIAL_CONDITIONS = {                                                 # a kit mode's partial activations: NAMED on the ACTIVE line (partial=<condition>), never a refusal
    "cold_cache": "no autotune class for this key: this process builds the class it then belongs to (the autotune results apply from the next process on)",
}


def activate(mode: str, variant: Optional[str] = None,
             strict: bool = True, gpu: Optional[dict] = None, build: Optional[dict] = None,
             n_gpu: int = 1, model_dir: Optional[str] = None, refresh_digest: bool = False,
             region: Optional[dict] = None, stated: Sequence[str] = (), caller_cache_dir: Optional[str] = None) -> dict:
    """Resolve and gate (mode, variant) on this box; compose nothing else. ``refresh_digest`` (the `check` verb): the weights are hashed afresh.
    ``region`` (big only; the record modes.effective_line resolved for the verb — absent, the size-unknown rule names region reach): region fast resolves the fast line's program lever for lever,
    region reach the memory levers'; the class gated is that line's (modes.effective_line: the class warm builds is the class pred reads); ``stated``: the caller's model-shape flags (settings.stated), which key the serialized executables. Idempotent per process for the same (mode, variant);
    ``caller_cache_dir``: a ``--cache_dir`` the caller states — it replaces the mode's class directory and every cache gate below reads it;
    a different activation after a model process was launched here raises ActivationError (strict) or is reported inactive. A kit mode
    whose activation is partial (PARTIAL_CONDITIONS — a cold cache class) ACTIVATES: the condition is named on the ACTIVE line (``partial``,
    ``partial_conditions``, ``note``), never a refusal; a device below the kit's reference hardware line is named (``gpu_notes``), never refused."""
    from ._autoload import TAG as _TAG
    from ._core_gate import gate as _core_gate
    core_facts = _core_gate(__file__, tag=_TAG)                           # THE pin gate (_core_gate.py): a core absent or older than the pin is NOT ACTIVE by name, SystemExit 3 — the first statement of enable()
    global _REPORT
    from . import ablation as _ablation, carry as _carry, modes as _modes, settings as _settings, variants as _variants
    up = pins().get("upstream", {})
    rep: dict = {"active": False, "mode": mode, "variant": variant, "package_version": _pkg_version(),
                 "alphafold3_version": f"{up.get('tag')}@{str(up.get('commit', ''))[:8]}",   # the pinned stock (stock/PINS.json upstream)
                 "levers_applied": [], "partial": False, "partial_conditions": [], "gpu_notes": [], "reason": None,
                 "n_gpu": None, "sharding": None, "superseded": [], "xla_pool": None, "levers_ablated": [],
                 "region": (dict(region) if (region and mode == _modes.BIG) else None), "mode_effective": mode}
    _missing = _modes.missing_producers()                               # an older core beside the kit (one without opt_core.mem.ngpu): NOT ACTIVE by name before anything resolves
    if _missing:
        rep["reason"] = _modes.producer_missing_reason(_missing)
        _REPORT = rep
        return dict(rep)
    try:
        from opt_core.mem import ngpu as _ngpu                           # the resource axis' one producer; the find_spec gate above names it first, this guard is the same words
    except ImportError as _e:
        rep["reason"] = _modes.producer_missing_reason([getattr(_e, "name", None) or "opt_core.mem.ngpu"])
        _REPORT = rep
        return dict(rep)
    if mode == _modes.BIG:                                             # the memory mode's own gate (big.py): the core's lever registry present and every lever of the line registered —
        from . import big as _big                                    # else NOT ACTIVE by name
        why = _big.gate()
        sel = None if why else _big.selection()
        if why or sel["refusals"]:
            rep["reason"] = why or "; ".join(sel["refusals"])
            _REPORT = rep
            return dict(rep)
        rep["big"] = sel                                               # the mode's lever set the launcher will apply: line, base, levers in order
    try:
        line = _modes.effective_line(mode, n_gpu=n_gpu, record=region)              # THE resolver (modes.effective_line): the region record the verb resolved (pred / check / warm) or, with none (the
        rep["region"], rep["mode_effective"] = line["record"], line["mode_effective"]      # Python API), the size-unknown rule named on the ACTIVE line; the mode whose program runs, the class gated below
        _region = line["region"]                                         # and the add-ons carried come from it alone — the memory mode's region word (modes.BIG_REGIONS), None for every other mode
    except (_modes.UnsupportedMode, ValueError) as e:
        raise ValueError(str(e)) from None
    early_kits = _modes.KIT_MODES[line["mode_effective"]]["kits"]        # the carry gate ahead of modes.resolve(): resolve() reads switch values out of the kit files themselves
    if early_kits:                                                       # (modes.py's own words) and would otherwise raise FileNotFoundError on a missing one with no named reason
        carry = _carry.carry_check(early_kits)
        if not carry["ok"]:
            rep["reason"] = f"kit file {carry['missing'][0]} is missing under opt/forward/: the kit this mode runs is not carried"
            _REPORT = rep
            return dict(rep)
    try:
        res = _modes.resolve(mode, region=_region, size=line["record"])  # raises on an unknown mode; big reach: the composition for THIS input's size (fused pair kernels kept at or below their serve edge)
        res = _modes.with_n_gpu(res, n_gpu)                              # --n_gpu P: P > 1 only under the memory mode (NGpuRefused by name otherwise)
        res = _modes.with_levers_off(res, _ablation.requested(), n_gpu)   # MODEL_OPT_LEVERS_OFF: the named levers leave the composition (ablation.py) — or the request is refused by name
    except _ngpu.NGpuRefused as e:                                       # `refused: n_gpu>1 requires --mode big (...)`: NOT ACTIVE by name, exit 3
        rep.update(n_gpu=int(n_gpu), reason=str(e.reason))
        _REPORT = rep
        return dict(rep)
    except _ablation.AblationError as e:                                 # `MODEL_OPT_LEVERS_OFF refused — <name>: <why>`: NOT ACTIVE by name, exit 3
        rep.update(n_gpu=int(n_gpu), reason=f"{_ablation.ENV} refused — {e}")
        _REPORT = rep
        return dict(rep)
    except (_modes.UnsupportedMode, ValueError) as e:
        raise ValueError(str(e)) from None
    rep.update(n_gpu=res["n_gpu"], sharding=res["sharding"], superseded=res["superseded"], levers_ablated=list(res.get("ablated") or []))
    rep.update(script=res["script"], row=res["row"], levers=res["levers"], launcher=[os.path.basename(x) for x in res["launcher"]][:1] or None,
               lever_env=res["env"], cache_class=res["cache_class"])
    if _LAUNCHED and any(x["mode"] != mode or x["variant"] != variant
                         or x.get("n_gpu", 1) != res["n_gpu"] or x.get("region") != _region for x in _LAUNCHED):
        msg = (f"late activation refused: a model process already ran in this interpreter as {_LAUNCHED[-1]}; "
               f"one (mode, variant) per process")
        if strict:
            raise ActivationError(msg)
        rep["reason"] = msg
        _REPORT = rep
        return dict(rep)
    if _REPORT and _REPORT.get("active") and (_REPORT["mode"], _REPORT["variant"], (_REPORT.get("params") or {}).get("dir"), (_REPORT.get("region") or {}).get("region")) == \
            (mode, variant, _variants.params_dir(variant, model_dir=model_dir) if variant in _variants.VARIANTS else None, _region):
        return dict(_REPORT)                                             # idempotent per (mode, variant, weights dir, region)
    if variant not in _variants.VARIANTS:
        rep["reason"] = f"a variant is required: {'|'.join(_variants.VARIANTS)} (got {variant!r})"
        _REPORT = rep
        return dict(rep)
    rep["stated"] = _settings.stated(list(stated))                       # the caller's model-shape flags (none: the fork's own defaults)
    rep["repo"], rep["python"] = repo_dir(), venv_python()
    gpu = gpu if gpu is not None else gpu_info()
    if res["n_gpu"] > 1:                                                 # the resource rule: P devices must be visible (never auto-sized, never shrunk)
        try:
            _ngpu.refuse_unless_visible(res["n_gpu"], int((gpu or {}).get("count") or 0))
        except _ngpu.NGpuRefused as e:                                   # `refused: n_gpu=P visible=K`
            rep["reason"] = str(e.reason)
            _REPORT = rep
            return dict(rep)
    rep["xla_pool"] = xla_pool_fraction(res["n_gpu"])                   # the XLA pool fraction at P devices: kept | written by the kit | a contradiction refused by name (decided here, before anything launches)
    if rep["xla_pool"]["refused"]:
        rep["reason"] = rep["xla_pool"]["refused"]
        _REPORT = rep
        return dict(rep)
    build = build if build is not None else jax_build()
    rep["gpu"] = gpu
    rep["jax_build"] = build
    rep["alphafold3_dist"] = (build or {}).get("alphafold3")          # the fork's installed distribution version, as the interpreter reports it
    rep["key"] = cache_key(gpu, build)
    rep["cache_dir"] = caller_cache_dir or (cache_dir(rep["key"], mode, region=_region) if rep["key"] else None)   # the caller's own --cache_dir replaces the class directory (the gates below read it)
    reasons = []
    inst = core_facts["installed"]                                          # the pin gate passed above: the importable core is at least the pinned version
    rep["core"] = {"ok": True, "version": inst.get("version"), "package_dir": inst.get("package_dir"),
                   "pinned": {k: core_facts["pinned"][k] for k in ("path", "version")}}
    g = gate_gpu(gpu)
    if g:
        reasons.append(g)
    rep["gpu_notes"] = gpu_notes(gpu)                                          # a device below the kit's reference line: named (the verbs print one GPU NOTE line each), never a reason
    reasons.extend(stack_reasons(rep["repo"], rep["python"], build))
    rep["install"] = install_state(rep["repo"], rep["python"]) if not reasons else None
    if rep["install"] and rep["install"]["install"] != "present":
        bad = sorted(f"{k}={v}" for k, v in rep["install"]["files"].items() if v != "present")
        reasons.append(f"the install does not carry the pinned stock ({rep['install']['install']}: {', '.join(bad)}); the pinned stock is stock/src = the archive + stock/patches (an image missing one of these files refuses here)")
    pc = _variants.check_params(variant, digest=True, model_dir=model_dir, refresh=refresh_digest)       # the weights (the caller's --model_dir or the variant's directory): digested, pinned or not — only a missing file refuses
    rep["params"] = pc
    if not pc["present"]:
        reasons.append(pc["reason"])
    kits = _modes.KIT_MODES[line["mode_effective"]]["kits"]                          # big in region fast gates the fast line's add-ons
    if kits:
        carry = _carry.carry_check(kits)
        rep["carry"] = {k: carry[k] for k in ("kits", "files", "ok", "missing")}
        if not carry["ok"]:
            reasons.append(f"kit file {carry['missing'][0]} is missing under opt/forward/: the kit this mode runs is not carried")
    if mode != "off" and not reasons:
        rep["executables"] = len(executables(rep["cache_dir"]))            # the ACTIVE line's executables= word: serialized-executable files present in the class
        cold = not (os.path.isdir(rep["cache_dir"]) and os.listdir(rep["cache_dir"]))
        rep["cold_cache"] = cold
        rep["partial_conditions"] = [c for c in PARTIAL_CONDITIONS if rep[c]]
        rep["levers_applied"] = list(res["levers"])
        if rep["partial_conditions"]:                                          # named on the ACTIVE line (partial=cold_cache), never a refusal: this process builds the class it then belongs to;
            rep["partial"] = True                                              # `warm` first is the deterministic recipe (README), not a precondition
            rep["note"] = "; ".join(f"{c}: {PARTIAL_CONDITIONS[c]}" for c in rep["partial_conditions"])
    if reasons:
        rep["reason"] = "; ".join(reasons)
    else:
        rep["active"] = True
    _REPORT = rep
    return dict(rep)


def _pkg_version() -> str:
    from . import __version__
    return __version__


def check(mode: str, variant: Optional[str] = None,
          n_gpu: int = 1, model_dir: Optional[str] = None, refresh_digest: bool = False,
          region: Optional[dict] = None, stated: Sequence[str] = ()) -> dict:
    """The dry run: the same resolution and gates as pred, nothing launched, nothing written. Never raises on the late-activation rule."""
    return activate(mode, variant, strict=False, n_gpu=n_gpu, model_dir=model_dir, refresh_digest=refresh_digest,
                    region=region, stated=stated)


def xla_pool_fraction(n_gpu: int, environ: Optional[dict] = None) -> dict:
    """The XLA client pool fraction of the model process at ``n_gpu`` devices — decided up front, named, never silent. The row-sharded stack's
    precondition (inprocess/rowpair.py MEM_FRACTION_CEILINGS, read by modes.rowpair_mem_fraction_ceiling: P >= 8 → the pool <= 0.9, NCCL's
    communicators allocate outside it) against the fraction the process would resolve: the two variable names jaxlib reads
    (opt_core.mem.rowpair_jax.mesh.MEM_FRACTION_VARS) in the precedence it reads them (opt_core.mem.peak.MEM_FRACTION_PRECEDENCE), the caller's
    environment over the image's pinned variables (stock/PINS.json image.env). A value the caller did not set — absent, or the image's own
    (`XLA_CLIENT_MEM_FRACTION=0.95`) — is the IMAGE's: over the ceiling the kit writes the ceiling under the name jaxlib reads first
    (``source=kit``); a value the caller set: within the ceiling it is kept (``source=user``), over it the request contradicts the precondition and
    is refused by name (``refused``); both names set is refused by name too (jaxlib raises on both at CUDA plugin initialisation). Below the table's
    first device count nothing applies (``applies: False``: P = 1, 2, 4 run the image's pool untouched).
    Returns {applies, n_gpu, ceiling, name, value, source: kit|user|image, env: {name: value} to lay over the model process's environment, refused: words|None}."""
    from opt_core.mem.peak import MEM_FRACTION_PRECEDENCE
    from opt_core.mem.rowpair_jax.mesh import MEM_FRACTION_VARS
    P = int(n_gpu or 1)
    from . import modes as _m
    ceiling = _m.rowpair_mem_fraction_ceiling(P)
    rec = {"applies": ceiling is not None, "n_gpu": P, "ceiling": ceiling, "name": None, "value": None, "source": None, "env": {}, "refused": None}
    if ceiling is None:
        return rec
    environ = dict(os.environ if environ is None else environ)
    image = dict(pins().get("image", {}).get("env", {}))
    names = [n for n in MEM_FRACTION_PRECEDENCE if n in MEM_FRACTION_VARS] + [n for n in MEM_FRACTION_VARS if n not in MEM_FRACTION_PRECEDENCE]
    composed = {n: environ.get(n, image.get(n)) for n in names if environ.get(n, image.get(n)) is not None}   # what model_process_env composes: the caller's value, else the image's
    user = {n: v for n, v in composed.items() if n in environ and environ[n] != image.get(n)}                # set by the caller: present and not the image's own value
    if len(composed) > 1:
        rec.update(name="+".join(composed), value="+".join(composed.values()), source="user" if user else "image",
                   refused=f"n_gpu={P} needs ONE XLA pool fraction variable: {' and '.join(f'{n}={v}' for n, v in composed.items())} are both set "
                           f"(jaxlib reads {names[0]} first and raises on both at CUDA plugin initialisation) — set {names[0]} alone, <= {ceiling}")
        return rec
    name, value = (next(iter(composed.items())) if composed else (names[0], None))
    try:
        frac = float(value) if value is not None else None
    except ValueError:
        rec.update(name=name, value=value, source="user" if user else "image", refused=f"n_gpu={P}: {name}={value!r} is not a fraction")
        return rec
    if frac is not None and frac <= ceiling:
        rec.update(name=name, value=value, source="user" if user else "image", env={})            # within the precondition: kept as composed
        return rec
    if user:                                                                                        # the caller's own value over the ceiling: a contradiction, refused by name
        rec.update(name=name, value=value, source="user",
                   refused=f"n_gpu={P} needs the XLA pool fraction <= {ceiling} (NCCL communicators allocate outside XLA's pool); the caller's {name}={value} "
                           f"contradicts it — unset it (the kit writes {ceiling}) or set it <= {ceiling}")
        return rec
    rec.update(name=name, value=f"{ceiling}", source="kit", env={name: f"{ceiling}"})              # the image's value (or none) over the ceiling: the kit writes the ceiling
    return rec


def model_process_env(environ: Optional[dict] = None, mode_env: Optional[dict] = None, core_path: bool = False, n_gpu: int = 1) -> dict:
    """The environment of every model process: the caller's, minus every variable with a must-be-absent prefix (stock/PINS.json
    stock_proof.must_be_absent_prefixes — the package's own switches and the add-ons' lever switches), plus the image's XLA variables of
    record where the caller has not set them (the recipe, stock/PINS.json image.env), plus — when ``core_path`` (the lever modes and the add-on tests; never mode off) — PYTHONPATH led by the pinned core's directory
    (core_dir: the add-ons import the core's carried kernels by path), plus the mode's own lever variables (modes.mode_env)."""
    environ = dict(os.environ if environ is None else environ)
    prefixes = tuple(pins().get("stock_proof", {}).get("must_be_absent_prefixes", ["AF3_JAX_"]))
    env = {k: v for k, v in environ.items() if not k.startswith(prefixes)}
    for k, v in pins().get("image", {}).get("env", {}).items():
        env.setdefault(k, v)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")             # no .pyc in the mounted tree (the launchers import kit modules from it)
    if core_path:                                              # lever modes only (exact / fast / big, the add-on tests): the pinned core's carried kernels, by path;
        env["PYTHONPATH"] = core_dir() + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")   # mode off stays stock-pure (no tree path in its environment)
    env.update(mode_env or {})
    env.update(xla_pool_fraction(n_gpu, environ)["env"])            # --n_gpu P: the pool fraction the row-sharded stack's precondition needs, when the caller has not set one (stack.xla_pool_fraction)
    return env
