"""Paths, the stack key, the gates, the activation and the run hook — the one activation implementation.

The model is a command line (``colabfold_batch`` = ``colabfold.batch:main``) whose ``main()`` parses the flags and calls
``colabfold.batch.run(queries=..., result_dir=..., data_dir=..., ...)`` (colabfold/batch.py:1743, :2202-2252); the model runners are built
inside ``run`` at the first job (:1537 ``load_models_and_params``), which is why an activation before ``run`` — in the same process — reaches
the model: ``activate(mode, queries=...)`` applies the mode's levers in the table's order (modes.TABLE) — ``DEVICE_RESIDENT``
(device_resident.enable(): ``alphafold.model.model.RunModel`` rebound to the resident subclass; `exact` and `fast`) and ``AF_PALLAS_ATTN``
(`fast`: the carried kit's ``af2_pallas_flash/`` on ``sys.path``, the kit's own switch ``AF_PALLAS_ATTN=1`` exported, its
``af2_pallas_attn.enable()`` (:69-103) called, which rebinds ``alphafold.model.modules.Attention`` to a subclass whose ``__call__``
runs the Pallas kernel (:58-66, :95-101)). ``hook_run`` wraps ``colabfold.batch.run`` so that activation
sees the run's queries (their token count: the ACTIVE line's tokens= and the sub-batch lever's input size, inputs.tokens) and its ``data_dir`` (the
parameters gate), and writes its manifest into the launch's work
directory when `pred` named one (manifest.work_dir); the env route (``_autoload.py``, ``COLABFOLD_OPT=fast``) and the explicit route (``colabfold_opt.enable``) both end here.
Before the stock ``run`` the hook reads, from ``run``'s own arguments, whether colabfold_batch will build a model at all (manifest.no_model_run:
``num_models == 0`` — ``--num-models 0`` / ``--msa-only`` —, or every job of the queries already complete under ``result_dir`` and kept, which
``run`` skips): on such a run the applied levers have no call to serve — ``idle`` by name on the IDLE line and on their LEVER lines
(``no_model_run=<case>``), never a partial activation, and the process exits as stock does.

Gates for a kit mode (``check`` and the activation apply them; ``off`` needs nothing): the kit directory; the stock pin (stock/PINS.json:
the two distributions' versions and the three pinned files present, through stock/check_pins.py — what `stock` means); jax importable and an
NVIDIA GPU visible at all; a refused kit switch (registry.REFUSED_SWITCHES); the parameters PRESENT (the marker file and the five files,
stock/PINS.json "weights") under the run's ``data_dir`` — ``$COLABFOLD_OPT_DATA_DIR`` on a dry run — their bytes a printed verdict (pinned |
NOT PINNED, weights_check), never a refusal; the late-activation rule (a run in progress, or the kit's ``_STATE["enabled"]`` already true from
outside this package). The environment is never a refusal: a jax outside the tested range (af2_pallas_flash/README.md:20: 0.5.3 - 0.7.x), a GPU below compute
capability 8.0 (af2_pallas_flash/README.md:19) or another model than the configuration's (``MODEL_OPT_TARGET_GPU``) is a note on the line (``notes=…``) and
every lever of the mode engages. A mode is all of its levers: one that cannot run on the stack (no tile table for the part, its requirement
raised, its patch did not land) refuses the mode by name (exit 3) — never a run under the mode's name with a subset of its levers.

Environment read by the package: ``COLABFOLD_OPT`` (mode), ``COLABFOLD_OPT_DATA_DIR``, ``COLABFOLD_OPT_HOME``, ``COLABFOLD_OPT_KIT``,
``COLABFOLD_OPT_JIT_ROOT``, ``COLABFOLD_OPT_N_GPU`` / ``COLABFOLD_OPT_LAUNCH_ID`` /
``COLABFOLD_OPT_WORK_DIR`` (set by `pred` for its model process),
``MODEL_OPT``, ``MODEL_OPT_TARGET_GPU``, ``MODEL_OPT_LEVERS_OFF`` (the ablation switch: the mode minus the named levers, ablation.py).
Exported to the process by the activation: ``AF_PALLAS_ATTN=1`` only (and not when ``AF_PALLAS_ATTN`` is ablated).
"""
from __future__ import annotations

import functools
import glob
import importlib
import importlib.util
import inspect
import json
import os
import shutil
import site
import subprocess
import sys
import sysconfig
from typing import Dict, List, Optional

from opt_core import gates as _core_gates
from opt_core.oom import is_oom                                          # out-of-memory is never rerouted: every broad handler on the served path re-raises it first

from . import _autoload, ablation as _ablation, digest_memo, inputs as _inputs, modes as _modes, registry as _registry, report as _report

ENV_HOME, ENV_KIT, ENV_TREE, ENV_TARGET_GPU = "COLABFOLD_OPT_HOME", "COLABFOLD_OPT_KIT", "MODEL_OPT", "MODEL_OPT_TARGET_GPU"
ENV_DATA = "COLABFOLD_OPT_DATA_DIR"
PACKAGE_ENV_PREFIXES = (_modes.ENV, _modes.KIT_SWITCH)                 # stock/PINS.json stock_proof.must_be_absent_prefixes
KIT_CLASS_DIRS = ("forward", "datapath", "serving")
KIT_PY_DIR = "af2_pallas_flash"                                         # the kit's importable directory (af2_pallas_flash/README.md:23)
KIT_MODULE_FILE = os.path.join(KIT_PY_DIR, _modes.KIT_MODULE + ".py")   # the file that identifies the kit directory
TRIGGER_MODULE = "colabfold.batch"
RUN_FUNCTION = "run"                                                    # colabfold/batch.py:1171
PINS_RELPATH = os.path.join("stock", "PINS.json")
CHECK_PINS_RELPATH = os.path.join("stock", "check_pins.py")
PYPROJECT = "pyproject.toml"
MIN_COMPUTE_CAP = 8.0                                                   # af2_pallas_flash/README.md:19 — the tested parts are compute capability >= 8.0; below it is a note on the line (gpu_note)
KIT_JAX_MIN, KIT_JAX_BELOW = (0, 5, 3), (0, 8)                          # af2_pallas_flash/README.md:20 — the tested range jax[cuda12] 0.5.3 - 0.7.x; outside it is a note on the line (jax_note)
DISTS = {"colabfold": "colabfold", "alphafold_colabfold": "alphafold-colabfold", "jax": "jax", "jaxlib": "jaxlib", "dm_haiku": "dm-haiku"}
CUDA_RUNTIME_DIST = "nvidia-cuda-runtime-cu12"                          # the CUDA runtime the jax CUDA plugin loads (environment/requirements.lock)

_STATE: Dict[str, object] = {"report": None, "running": 0, "manifest_dir": None, "hooked": None}
_CACHE: Dict[str, object] = {}


class ActivationError(RuntimeError):
    """A refused activation under strict=True: the gates, or the late-activation rule."""


# ----------------------------------------------------------------------------------------------------------------- paths
def opt_home() -> str:
    """colabfold/opt — the directory the package is installed from (editable), or $COLABFOLD_OPT_HOME."""
    h = os.environ.get(ENV_HOME)
    return os.path.abspath(h) if h else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tree_home() -> str:
    """colabfold/ — the model's directory ($MODEL_OPT when set, else the parent of opt_home())."""
    h = os.environ.get(ENV_TREE)
    return os.path.abspath(h) if h else os.path.dirname(opt_home())


def kit_home() -> str:
    """The kit directory: the one under opt/<class>/ that carries af2_pallas_flash/af2_pallas_attn.py ($COLABFOLD_OPT_KIT overrides)."""
    k = os.environ.get(ENV_KIT)
    if k:
        if not os.path.isfile(os.path.join(k, KIT_MODULE_FILE)):
            raise FileNotFoundError(f"{ENV_KIT}={k}: no {KIT_MODULE_FILE} there")
        return os.path.abspath(k)
    hits: List[str] = []
    for cls in KIT_CLASS_DIRS:
        hits += sorted(glob.glob(os.path.join(opt_home(), cls, "*", KIT_MODULE_FILE)))
    if len(hits) != 1:
        raise FileNotFoundError(f"kit not found: expected exactly one opt/{{{','.join(KIT_CLASS_DIRS)}}}/*/{KIT_MODULE_FILE} under "
                                f"{opt_home()}, found {hits} (install the package editable from colabfold/opt, or set {ENV_KIT})")
    return os.path.dirname(os.path.dirname(hits[0]))


def kit_pythonpath() -> str:
    """The directory the kit puts on PYTHONPATH (af2_pallas_flash/README.md:23): <kit>/af2_pallas_flash."""
    return os.path.join(kit_home(), KIT_PY_DIR)


def pins_path() -> str:
    return os.path.join(tree_home(), PINS_RELPATH)


def pyproject_path() -> str:
    """opt/pyproject.toml — the kit's own project file, the core pin's home (`[tool.opt_core]`)."""
    return os.path.join(opt_home(), PYPROJECT)


def pins() -> dict:
    if "pins" not in _CACHE:
        import json
        with open(pins_path(), encoding="utf-8") as f:
            _CACHE["pins"] = json.load(f)
    return _CACHE["pins"]  # type: ignore[return-value]


def check_pins_module():
    """stock/check_pins.py loaded as a module: the one implementation of the pin, file and parameter checks (the script run.sh runs)."""
    if "check_pins" not in _CACHE:
        path = os.path.join(tree_home(), CHECK_PINS_RELPATH)
        spec = importlib.util.spec_from_file_location("colabfold_opt_check_pins", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        _CACHE["check_pins"] = mod
    return _CACHE["check_pins"]


def data_dir_env() -> Optional[str]:
    d = os.environ.get(ENV_DATA)
    return os.path.abspath(d) if d else None


sha256_file = _core_gates.sha256_file


# ---------------------------------------------------------------------------------------------------- the autoload .pth in site (the env route's hook)
PTH_NAME = "colabfold_opt_autoload.pth"          # the wheel-root .pth (opt/colabfold_opt_autoload.pth, placed by opt/_build_backend.py) that installs the finder at interpreter start


def site_dirs(executable: Optional[str] = None) -> List[str]:
    """The site-packages directories of this interpreter (the process that runs `colabfold_batch`: stock_pred.launch_form — the console
    script's shebang names this interpreter): site.getsitepackages(), the user site when enabled, sysconfig's purelib; deduplicated."""
    dirs: List[str] = []
    for d in (list(getattr(site, "getsitepackages", lambda: [])()) + ([site.getusersitepackages()] if getattr(site, "ENABLE_USER_SITE", False) else [])
              + [sysconfig.get_paths().get("purelib") or ""]):
        d = os.path.abspath(d) if d else ""
        if d and d not in dirs:
            dirs.append(d)
    return dirs


HOOK_MODULE = "colabfold_opt._autoload"          # the module the .pth imports at interpreter start (the hook)
_HOOK_PROBE = ("import sys, json; m = sys.modules.get(%r); print(json.dumps({'live': m is not None, 'file': getattr(m, '__file__', None)}))" % HOOK_MODULE)


def autoload_pth_diagnostic(dirs_site: List[str]) -> str:
    """Where the .pth is, for the refusal line: `present in <site dir> but not processed` (site.py did not run it: -S, a broken line;
    the user site is among the searched dirs only when enabled, and then site.py processes it), `present beside <dir> but not processed
    (not a site directory)` (a copy on PYTHONPATH / sys.path — site.py never reads it there), or `absent from the searched sites (<dirs>)`."""
    for d in dirs_site:
        if os.path.isfile(os.path.join(d, PTH_NAME)):
            return f"{PTH_NAME} present in {d} but not processed"
    others = [os.path.abspath(p) for p in (os.environ.get("PYTHONPATH", "").split(os.pathsep) + sys.path) if p and os.path.isdir(p)]
    for d in others:
        if d not in dirs_site and os.path.isfile(os.path.join(d, PTH_NAME)):
            return f"{PTH_NAME} present beside {d} but not processed (not a site directory: site.py never reads it there)"
    return f"{PTH_NAME} absent from the searched sites ({', '.join(dirs_site) or 'no site directory'})"


def autoload_pth_check(executable: Optional[str] = None) -> tuple:
    """(reasons, details): the .pth's EFFECT — a FRESH interpreter (`python -c`, started as the route's own is: PYTHONPATH and the user
    site as the route sees them, only the kit's own variables stripped) carries the hook module in sys.modules at start, and that module
    is this package's own `_autoload.py`. The environment route (`COLABFOLD_OPT=<mode> colabfold_batch …`) fires through that hook
    alone: a package merely importable (PYTHONPATH, a checkout on sys.path) installs no finder — site.py processes no .pth outside its
    site directories (the venv / system site, the user site when enabled) — and the stock CLI runs STOCK silently; the environment route
    refuses by name before anything runs when the hook is not live. The line names the diagnostic (autoload_pth_diagnostic:
    present-but-unprocessed / absent / a stale copy)."""
    py = executable or sys.executable
    env = {k: v for k, v in os.environ.items() if not k.startswith(PACKAGE_ENV_PREFIXES)}   # the probe imports the hook only; no mode set, no finder
    try:
        r = subprocess.run([py, "-c", _HOOK_PROBE], capture_output=True, text=True, env=env, timeout=120)   # no -I: the route's `python` processes the user site and PYTHONPATH — so does the probe
        probe = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else {"live": False, "file": None, "rc": r.returncode, "stderr": r.stderr[-300:]}
    except Exception as e:  # noqa: BLE001
        probe = {"live": False, "file": None, "probe_failed": repr(e)}
    dirs = site_dirs()
    det = {"hook": HOOK_MODULE, "python": py, "probe": probe, "site": dirs, "pth": PTH_NAME}
    own = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_autoload.py"))
    if not probe.get("live"):
        return [f"{HOOK_MODULE} is not live at interpreter start ({py}): {autoload_pth_diagnostic(dirs)} — the environment route "
                f"`COLABFOLD_OPT=<mode> colabfold_batch` would run stock silently; install the kit into this interpreter "
                f"(`pip install -e <kit>/opt`), not PYTHONPATH"], det
    if probe.get("file") and os.path.realpath(probe["file"]) != own:
        return [f"{HOOK_MODULE} is live at interpreter start but from {probe['file']} — a stale copy: this package is at {own}; "
                f"reinstall the kit into this interpreter (`pip install -e <kit>/opt`)"], det
    return [], det


# ------------------------------------------------------------------------------------------------------------ the stack key
dist_version = _core_gates.dist_version


def versions() -> Dict[str, Optional[str]]:
    """The installed distributions' versions (importlib.metadata; nothing imported)."""
    return {k: dist_version(d) for k, d in DISTS.items()}


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


def cuda_runtime() -> Optional[str]:
    """CUDA runtime major.minor from the nvidia-cuda-runtime-cu12 distribution (12.9.79 -> 12.9); None when absent."""
    v = dist_version(CUDA_RUNTIME_DIST)
    return ".".join(v.split(".")[:2]) if v else None


def stack_key(cc: Optional[float] = None, jax_version: Optional[str] = None) -> str:
    """jax<ver>-cu<x.y>-sm<cc>, e.g. jax0.5.3-cu12.9-sm90 (informational: MODEL_OPT_STACK_KEY in configs/h100.env)."""
    if cc is None:
        g = gpu_info()
        cc = g["compute_cap"] if g else None
    sm = f"sm{str(cc).replace('.', '')}" if cc is not None else "smunknown"
    return f"jax{jax_version or versions()['jax'] or 'unknown'}-cu{cuda_runtime() or 'unknown'}-{sm}"


def kernel_key(cc: Optional[float], jax_version: Optional[str]) -> Optional[str]:
    """<cc>|<jax> (e.g. 9.0|0.5.3), the key form of the activation tables; None without a GPU."""
    return f"{cc}|{jax_version}" if cc is not None and jax_version else None


def _vtuple(v: str) -> tuple:
    out = []
    for part in v.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


# ----------------------------------------------------------------------------------------------------------------- gates
def gate_gpu(gpu: Optional[dict]) -> Optional[str]:
    """None when an NVIDIA GPU is visible; else the reason (no device: nothing of a kit mode can run). The part's compute capability is
    never a refusal (gpu_note)."""
    if not gpu:
        return "no NVIDIA GPU visible (nvidia-smi)"
    return None


def gpu_note(gpu: Optional[dict]) -> Optional[str]:
    """The ACTIVE line's note for a part below the tested compute capability (af2_pallas_flash/README.md:19), else None: untested, every lever engages; a
    lever that then cannot run there refuses the mode by name."""
    if gpu and gpu.get("compute_cap") is not None and gpu["compute_cap"] < MIN_COMPUTE_CAP:
        return f"compute capability {gpu['compute_cap']} < {MIN_COMPUTE_CAP} ({gpu['name']}): untested part, the levers engage"
    return None


def gate_jax(v: Optional[str]) -> Optional[str]:
    """None when jax is importable; else the reason. Its version is never a refusal (jax_note)."""
    if not v:
        return "jax is not installed"
    return None


def jax_note(v: Optional[str]) -> Optional[str]:
    """The ACTIVE line's note for a jax outside the tested range (af2_pallas_flash/README.md:20), else None."""
    if v and not (KIT_JAX_MIN <= _vtuple(v) < KIT_JAX_BELOW):
        return f"jax {v} outside the tested range 0.5.3 - 0.7.x: untested, the levers engage"
    return None


def gate_switches(environ=None) -> Optional[str]:
    """A kit switch the activation refuses by name when the caller's environment turns it on (registry.REFUSED_SWITCHES), else None."""
    env = os.environ if environ is None else environ
    bad = [f"{n}={env[n]} refused: {why}" for n, why in _registry.REFUSED_SWITCHES.items() if env.get(n, "0") not in ("", "0")]
    return "; ".join(bad) or None


def pins_check() -> tuple:
    """(reasons, details): the two distributions at the pinned versions and the three pinned files byte-identical to the wheels
    (stock/check_pins.py packages_report / files_report on stock/PINS.json)."""
    cp = check_pins_module()
    p = pins()
    reasons, det = [], {}
    for key, up in p["upstream"].items():
        got = dist_version(up["package"])
        det[key] = {"version": got, "pinned": up["version"], "ok": got == up["version"]}
        if got is None:
            reasons.append(f"stock pin: {up['package']} is not installed")
        elif got != up["version"]:
            reasons.append(f"stock pin: {up['package']} {got} != {up['version']} (stock/PINS.json)")
    files = cp.files_report(p)
    det["files"] = files["files"]
    if not files["ok"]:
        bad = [rel for rel, d in files["files"].items() if d["state"] != "ok"]
        reasons.append(f"stock pin: installed files differ from stock/src/: {', '.join(bad)}")
    return reasons, det


def digest_memo_dir(environ=None) -> str:
    """The directory of the on-disk weights digest memo (``digest_memo.MEMO_NAME`` = weights_digests.json inside it): the kit's cache
    root ``COLABFOLD_OPT_JIT_ROOT`` (xla_cache.ROOT_ENV — configs/<gpu>.env exports it, default ``~/.cache/colabfold_opt/jit``); the same
    default when the variable is unset."""
    from . import xla_cache as _xla_cache  # noqa: PLC0415
    env = os.environ if environ is None else environ
    return env.get(_xla_cache.ROOT_ENV) or os.path.join(os.path.expanduser("~"), ".cache", "colabfold_opt", "jit")


def weights_digest(path: str, refresh: bool = False) -> str:
    """sha256 of a parameter file (opt_core.gates.sha256_file over the full contents) through the on-disk digest memo (digest_memo.py:
    ``<digest_memo_dir()>/weights_digests.json``; an entry is selected by (realpath, size, mtime_ns, inode) and written only after a full
    sha256 — the stat fields select the entry, they never stand in for the digest), behind this process's own cache keyed the same way: the five
    pinned files (≈1.9 GB) are hashed once, a later process reads the memo. ``refresh`` (the `check` verb) hashes afresh and rewrites the
    entry. The memo's answer per file — ``cached_utc`` (None when hashed by this call) and a note when the memo file could not be written
    (the digest is then computed all the same) — is kept for ``weights_check`` / ``weights_lines``."""
    real = os.path.realpath(path)
    st = os.stat(real)
    key = (real, st.st_size, st.st_mtime_ns, st.st_ino)
    cache = _CACHE.setdefault("weights_sha256", {})
    if refresh or key not in cache:
        note = None
        try:
            got, cached_utc = digest_memo.digest(path, digest_memo_dir(), refresh=refresh, hasher=_core_gates.sha256_file)
        except OSError as e:                                              # the memo file could not be written (read-only / foreign cache root): the digest is computed, the line names it — never a refusal
            got, cached_utc, note = _core_gates.sha256_file(real), None, f"digest memo unwritable: {type(e).__name__}"
        cache[key] = {"sha256": got, "cached_utc": cached_utc, "note": note}
    return cache[key]["sha256"]


def _weights_memo_entry(path: str) -> dict:
    real = os.path.realpath(path)
    st = os.stat(real)
    return _CACHE.get("weights_sha256", {}).get((real, st.st_size, st.st_mtime_ns, st.st_ino), {})


def weights_check(data_dir: Optional[str], refresh: bool = False) -> tuple:
    """(reasons, details) for the parameters under ``data_dir`` (stock/check_pins.py weights_report, digests always computed here —
    through the digest memo, ``refresh`` = the `check` verb: afresh):
    PRESENCE is the gate — no data directory, the marker missing (colabfold would fetch from the network) or a parameter file missing is
    a refusal by name; the BYTES are a reported state — the pinned digests are `pinned`, any other checkpoint under the stock file names is
    `not_pinned` and RUNS (the kit's measurements apply to the pinned weights only). ``details["pinned"]`` and each
    file's sha256 land in the activation report / the manifest; ``weights_lines(details)`` are the printed lines."""
    cp = check_pins_module()
    rep = cp.weights_report(pins(), data_dir, digest=True, digest_fn=functools.partial(weights_digest, refresh=refresh))
    for d in rep.get("files", {}).values():                                # the memo's answer per present file, for the lines and the manifest
        if isinstance(d, dict) and d.get("path"):
            m = _weights_memo_entry(d["path"])
            d["digest_cached_utc"], d["digest_memo_note"] = m.get("cached_utc"), m.get("note")
    reasons = []
    if rep["ok"] is None:
        reasons.append(rep["reason"])
    elif not rep["ok"]:
        missing = [rel for rel, d in rep["files"].items() if d["state"] == "missing"]
        parts = []
        if not rep["marker"]["present"]:
            parts.append(f"marker {rep['marker']['path']} missing (colabfold would fetch from the network)")
        if missing:
            parts.append("files missing: " + ", ".join(missing))
        reasons.append("parameters under " + str(data_dir) + ": " + "; ".join(parts))
    return reasons, rep


def weights_lines(details: Optional[dict]) -> List[str]:
    """The parameter state lines (stock/check_pins.py weights_lines with this package's prefix): `weights=<name> sha256=<12> (pinned)`
    per pinned file, `weights=<name> sha256=<12> NOT PINNED — the kit's measurements apply to the pinned weights only` per other file;
    either word followed by ` (cached digest <utc>)` when the digest came from the on-disk memo (digest_memo.word) and by
    ` (digest memo unwritable: <error>)` when the memo file could not be written."""
    if not details:
        return []
    by_name = {os.path.basename(rel): d for rel, d in details.get("files", {}).items() if isinstance(d, dict)}
    out = []
    for line in check_pins_module().weights_lines(details, prefix=_report.PREFIX):
        name = line.split("weights=", 1)[1].split()[0] if "weights=" in line else None
        d = by_name.get(name, {})
        line = digest_memo.word(line, d.get("digest_cached_utc"))
        if d.get("digest_memo_note"):
            line += f" ({d['digest_memo_note']})"
        out.append(line)
    return out


def kit_state() -> Optional[dict]:
    """The kit's ``_STATE`` dict (af2_pallas_attn.py:16) when its module is imported, else None."""
    m = sys.modules.get(_modes.KIT_MODULE)
    return getattr(m, "_STATE", None) if m is not None else None


def host_lever_state() -> Optional[dict]:
    """The package transfer lever's ``_STATE`` dict (device_resident.py) when its module is imported, else None."""
    m = sys.modules.get(_modes.HOST_LEVER_MODULE)
    return getattr(m, "_STATE", None) if m is not None else None


def tp_lever_state() -> Optional[dict]:
    """The n_gpu axis' state (big.py ``state()``: enabled, n_gpu, sharding, pad events, sites) when its module is imported, else None."""
    m = sys.modules.get(_modes.TP_LEVER_MODULE)
    return m.state() if m is not None else None


KERNEL_SERVE = "opt_core.kernels.pallas_attn_serve"      # the core's serve layer of the carried Pallas attention kernel (its KERNEL_FILE = the module name the adapter imports)


def route_kernel_to_core():
    """Bind the top-level module name the carried adapter imports (``af2_flash_pallas``, ``af2_pallas_attn.py:11``) to the shared
    core's carried copy of that kernel (``opt_core.kernels.pallas_attn.af2_flash_pallas``, held to the core's SUMS) BEFORE the adapter is
    imported, so one kernel module lives in the process — the core's — and the core's serve layer (``PALLAS_MSA``) and the adapter
    (``AF_PALLAS_ATTN``) drive the same module object. A name already bound is left as it is (the core's own check then refuses a copy
    from another directory by name, ``twin_loaded``); a core that cannot provide the kernel raises its ``Refusal`` by name."""
    PAS = importlib.import_module(KERNEL_SERVE)
    name = PAS.KERNEL_FILE
    if name in sys.modules:
        return sys.modules[name]
    mod = PAS.kernel_module(require_gpu=False)
    sys.modules[name] = mod
    return mod


def msa_lever_state() -> Optional[dict]:
    """The MSA-attention lever's ``_STATE`` dict (msa_attn.py) when its module is imported, else None."""
    m = sys.modules.get(_modes.MSA_LEVER_MODULE)
    return getattr(m, "_STATE", None) if m is not None else None



def col_lever_state() -> Optional[dict]:
    """The cuDNN MSA-column lever's ``_STATE`` dict (msa_col_cudnn.py) when its module is imported, else None."""
    m = sys.modules.get(_modes.COL_LEVER_MODULE)
    return getattr(m, "_STATE", None) if m is not None else None


def apply_pair_word(kit, mode: str) -> str:
    """AF_PALLAS_ATTN's binding (triattn_xla.bind): the carried adapter's ``FLASH_OP`` = the shared core provider's attention face called with
    the mode's tier word (modes.tier_word) and the site-naming subclass over the adapter's class; returns the word (the activation report's
    ``attn_word``). The carried adapter's own ``_STATE`` is not touched (its census shape is the kit's)."""
    from . import triattn_xla as _pair
    return _pair.bind(kit, _modes.tier_word(mode))


def compute_capability() -> Optional[str]:
    """The first jax GPU device's compute capability ("9.0", "8.0", …); None without jax / a GPU device."""
    try:
        import jax
        for d in jax.devices():
            cc = getattr(d, "compute_capability", None)
            if cc:
                return str(cc)
    except Exception:  # noqa: BLE001
        return None
    return None


def trimul_lever_state() -> Optional[dict]:
    """The triangle-multiplication lever's ``_STATE`` dict (trimul_pallas.py) when its module is imported, else None."""
    m = sys.modules.get(_modes.TRIMUL_LEVER_MODULE)
    return getattr(m, "_STATE", None) if m is not None else None


def triattn_lever_state() -> Optional[dict]:
    """The pre-compiled triangle-attention lever's ``_STATE`` dict (triattn_xla.py) when its module is imported, else None."""
    m = sys.modules.get(_modes.TRIATTN_LEVER_MODULE)
    return getattr(m, "_STATE", None) if m is not None else None


def transition_lever_state() -> Optional[dict]:
    """The transition lever's ``_STATE`` dict (transition.py) when its module is imported, else None."""
    m = sys.modules.get(_modes.TRANSITION_LEVER_MODULE)
    return getattr(m, "_STATE", None) if m is not None else None


def templ_lever_state() -> Optional[dict]:
    """The template-dedup lever's ``_STATE`` dict (templ_dedup.py) when its module is imported, else None."""
    m = sys.modules.get(_modes.TEMPL_LEVER_MODULE)
    return getattr(m, "_STATE", None) if m is not None else None


def subbatch_lever_state() -> Optional[dict]:
    """The sub-batch lever's ``_STATE`` dict (subbatch.py) when its module is imported, else None."""
    m = sys.modules.get(_modes.SUBBATCH_LEVER_MODULE)
    return getattr(m, "_STATE", None) if m is not None else None


def kit_lever_state() -> Optional[dict]:
    """AF_PALLAS_ATTN's evidence: the carried adapter's ``_STATE`` (enabled, calls, fallbacks — the EXIT line's dict, untouched) followed by the
    binding's census (triattn_xla.bind_state: word=, provider_calls=, served=<row>:<n>, sites=, cells=, provider=) when the binding is installed."""
    st = kit_state()
    if not isinstance(st, dict):
        return st
    m = sys.modules.get(_modes.TRIATTN_LEVER_MODULE)
    if m is None or not m._BIND.get("bound"):
        return st
    out = dict(st)
    out.update(m.bind_state())
    return out


def lever_states() -> dict:
    """lever -> its live ``_STATE`` dict (None when its module is not imported): the EXIT/LEVER lines' and the manifest's one source."""
    return {_modes.HOST_LEVER: host_lever_state(), _modes.SUBBATCH_LEVER: subbatch_lever_state(), _modes.TRIMUL_LEVER: trimul_lever_state(),
            _modes.LEVER: kit_lever_state(), _modes.MSA_LEVER: msa_lever_state(),
            _modes.TRIATTN_LEVER: triattn_lever_state(), _modes.COL_LEVER: col_lever_state(), _modes.TEMPL_LEVER: templ_lever_state(), _modes.TRANSITION_LEVER: transition_lever_state(),
            _modes.TP_LEVER: tp_lever_state()}


def gate_n_gpu(mode: str, n_gpu: int, visible: Optional[int]) -> Optional[str]:
    """The ``--n_gpu`` gate (big.gate): None, or the refusal sentence. ``visible``: the GPU count of this host (gpu_info()["count"]; None =
    not counted here — the model process counts again when it builds the mesh). A kit mode at n_gpu=1 other than big imports nothing
    here; big (any P) and every P>1 need the shared core's n_gpu modules — absent, the refusal names the module."""
    if int(n_gpu) == 1 and mode not in _modes.TP_MODES:
        return None
    try:
        big = importlib.import_module(_modes.TP_LEVER_MODULE)
    except ImportError as e:                                             # opt_core without mem.ngpu / mem.rowpair_jax: named, exit 3 — never a P=1 run
        import re as _re
        m = _re.search(r"cannot import name '(\w+)' from '([\w.]+)'", str(e))
        missing = f"{m.group(2)}.{m.group(1)}" if m else (getattr(e, "name", None) or str(e))
        return f"core_missing:{missing} (--mode big and --n_gpu need the shared core's opt_core.mem.ngpu and opt_core.mem.rowpair_jax)"
    return big.gate(mode, int(n_gpu), visible)


def late_activation_reason(mode: str) -> Optional[str]:
    for name, st in lever_states().items():
        if _STATE["report"] is None and st is not None and st.get("enabled"):
            return f"{name} reports enabled=True already (its _STATE): it was switched on outside this package"
    if _STATE["running"]:
        return f"a prediction run is in progress ({TRIGGER_MODULE}.{RUN_FUNCTION} entered): one activation per process, before the run"
    rep = _STATE["report"]
    if rep is not None and rep.get("active") and rep.get("mode") != mode:
        return f"mode {rep['mode']} is active in this process already; one mode per process"
    return None


# ------------------------------------------------------------------------------------------------------------ activation
def _pkg_version() -> str:
    from . import __version__
    return __version__


def _base_report(mode: str) -> dict:
    v = versions()
    return {"active": False, "mode": mode, "levers": [], "levers_applied": [], "levers_unavailable": [], "partial": [], "reason": None,
            "tokens_min": None, "tokens_max": None, "queries": None,
            "package_version": _pkg_version(), "colabfold_version": v["colabfold"], "alphafold_colabfold_version": v["alphafold_colabfold"],
            "jax_version": v["jax"], "jaxlib_version": v["jaxlib"], "dm_haiku_version": v["dm_haiku"], "gpu": None, "key": None,
            "stack_key": None, "kit": None, "kit_env": {}, "notes": [], "route": None, "trigger": None, "data_dir": None, "pins": None,
            "weights": None, "kit_state": None, "core": None}


def activate(mode: str, queries=None, strict: bool = False, dry_run: bool = False,
             trigger: Optional[str] = None, manifest_dir: Optional[str] = None, data_dir: Optional[str] = None,
             route: Optional[str] = None, print_line: bool = True, n_gpu: Optional[int] = None, refresh_digests: bool = False) -> dict:
    """Resolve, gate and — unless dry_run — apply ``mode`` in this process; the activation report is returned and printed as one line.

    ``queries``: the run's queries (colabfold/batch.py:1172): their token count is the line's tokens= and the sub-batch lever's input size;
    None on a dry run or an activation before the run (tokens=none, the sub-batch value by name without a size). ``strict``: a refusal raises
    ActivationError (after the NOT ACTIVE line) instead of returning an inactive report — a lever of the mode that cannot run on this GPU is
    one (a mode is all of its levers). Idempotent: a second call for the same
    mode returns the first report. ``manifest_dir``: where opt_manifest.json is
    written (the launch's work directory; None: no manifest); ``data_dir``: the run's parameters root (None = $COLABFOLD_OPT_DATA_DIR). ``print_line`` False: a dry run
    whose report is returned without its DRY-RUN line (`pred`'s gate call: the run's lines are the model process's). ``n_gpu``: the
    resource axis of big (None = modes.n_gpu_from_env(): COLABFOLD_OPT_N_GPU, absent = 1); gated by gate_n_gpu, installed by big.apply."""
    from . import manifest as _manifest
    res = _modes.resolve(mode)                                          # raises UnsupportedMode by name (unknown)
    n_gpu_reason = None
    if n_gpu is None:
        try:
            n_gpu = _modes.n_gpu_from_env()
        except ValueError as e:
            n_gpu, n_gpu_reason = 1, f"refused: {e}"
    if int(n_gpu) == 1 and _modes.TP_LEVER in res["levers"]:            # P = 1: the axis installs no sharding — the process's lever set is fast's (its LEVER line reads state=off reason=n_gpu=1); the axis' tokens are still recorded at activation below
        res = dict(res, levers=tuple(l for l in res["levers"] if l != _modes.TP_LEVER))
    dropped = {}
    for pm, pm_module in _modes.ONE_DEVICE_LEVERS:
        if int(n_gpu) > 1 and pm in res["levers"]:                      # P > 1: the row-sharded pair stack owns the pair sites (opt_core.mem.rowpair_jax) — a one-device pair lever steps aside; its LEVER line reads state=off reason=n_gpu>1
            res = dict(res, levers=tuple(l for l in res["levers"] if l != pm))
            dropped[pm] = importlib.import_module(pm_module).N_GPU_REASON
    if _STATE["report"] is not None and _STATE["report"].get("mode") == mode and not dry_run and _STATE["report"].get("active"):
        return dict(_STATE["report"])                                    # idempotent per process
    rep = _base_report(mode)
    rep.update(levers=list(res["levers"]), kit_env=dict(res["kit_env"]), levers_dropped=dict(dropped), route=route or ("dry-run" if dry_run else "in-process"),
               trigger=trigger, data_dir=data_dir or data_dir_env(), n_gpu=int(n_gpu))
    rep["gpu"] = gpu_info()
    if mode in _modes.TP_MODES or int(n_gpu) != 1:                      # the axis' tokens on the line (evidence.active_fields) and its gate, before any other
        g = n_gpu_reason or gate_n_gpu(mode, int(n_gpu), (rep["gpu"] or {}).get("count"))
        if g is None:
            rep["n_gpu_fields"] = importlib.import_module(_modes.TP_LEVER_MODULE).active_fields(int(n_gpu))
            rep["sharding"] = dict(rep["n_gpu_fields"]).get("sharding")
        else:
            rep.update(reason=g, levers_unavailable=list(res["levers"]))
            if dry_run:
                rep["would_refuse"] = g
            if not dry_run:
                _STATE["report"] = rep
            if print_line or not dry_run:
                _report.print_activation(rep, dry_run=dry_run)
            if manifest_dir and not dry_run:
                _manifest.write(manifest_dir, rep)
            if strict and not dry_run:
                raise ActivationError(g)
            return dict(rep)
    abl = _ablation.requested()                                          # MODEL_OPT_LEVERS_OFF: the mode minus the named levers (ablation.py) — validated against the set THIS process applies
    if abl:                                                              # at this --n_gpu, refused by name otherwise (unknown / outside the mode / the axis / empties the mode / under off)
        try:
            abl = _ablation.validate(mode, abl, res["levers"], int(n_gpu))   # the validated names (a lever whose binding another ablated lever provides is ablated with it, by name: ablation.validate)
        except _ablation.AblationError as e:
            rep.update(reason=str(e), levers_unavailable=list(res["levers"]), levers_ablation_refused=list(abl))
            if dry_run:
                rep["would_refuse"] = str(e)
            else:
                _STATE["report"] = rep
            if dry_run and mode != "off":
                if print_line:
                    _report.print_activation(rep, dry_run=True)          # DRY-RUN … would_refuse='MODEL_OPT_LEVERS_OFF refused — …' (the caller exits 3)
            else:
                _report.emit(_report.NOT_ACTIVE_FMT.format(prefix=_report.PREFIX, reason=str(e)))   # NOT ACTIVE: MODEL_OPT_LEVERS_OFF refused — … (under off too: never the OFF line)
            if manifest_dir and not dry_run:
                _manifest.write(manifest_dir, rep)
            if strict and not dry_run:
                raise ActivationError(str(e))
            return dict(rep)
        res = _ablation.apply(res, abl)                                  # the levers tuple without the ablated names; AF_PALLAS_ATTN's switch out of kit_env when it is one of them
        rep.update(levers=list(res["levers"]), kit_env=dict(res["kit_env"]), levers_ablated=list(abl))
        if not dry_run:
            for k in _ablation.switches(abl):                            # a caller's export of an ablated lever's switch cannot re-engage it
                os.environ.pop(k, None)
    cc = (rep["gpu"] or {}).get("compute_cap")
    rep["key"] = kernel_key(cc, rep["jax_version"])
    rep["stack_key"] = stack_key(cc, rep["jax_version"])
    target = os.environ.get(ENV_TARGET_GPU)
    if target and rep["gpu"] and target.lower() not in rep["gpu"]["name"].lower():
        rep["notes"].append(f"{ENV_TARGET_GPU}={target} but the visible GPU is {rep['gpu']['name']}")
    if mode == "off":
        if not dry_run:
            _STATE["report"] = rep
        if print_line or not dry_run:
            _report.print_activation(rep, dry_run=dry_run)
        return dict(rep)
    reasons: List[str] = []
    facts = _autoload.gate_core()                                        # the core pin gate (every entry ran it as statement one; here it yields the facts for the report — or exits 3 by name)
    rep["core"] = {"ok": True, "pinned": facts["pinned"], "installed": facts["installed"]}
    try:
        rep["kit"] = kit_home()
    except FileNotFoundError as e:
        reasons.append(str(e))
    try:                                                                 # a gate that cannot run is a refusal, never a traceback out of the hook
        pr, pdet = pins_check()
        rep["pins"] = pdet
        reasons += pr
        g = gate_jax(rep["jax_version"])
        if g:
            reasons.append(g)
        g = gate_gpu(rep["gpu"])
        if g:
            reasons.append(g)
        rep["notes"] += [n for n in (jax_note(rep["jax_version"]), gpu_note(rep["gpu"])) if n]   # the environment, named on the line — never a refusal
        g = gate_switches()
        if g:
            reasons.append(g)
        wr, wdet = weights_check(rep["data_dir"], refresh=refresh_digests)   # the `check` verb hashes afresh and rewrites the memo; pred / the hook read it
        rep["weights"] = wdet
        reasons += wr
        if print_line and not launched_by_pred():                        # the parameter state lines (pinned | NOT PINNED); `pred` prints them once in the launcher
            for line in weights_lines(wdet):
                _report.emit(line)
    except Exception as e:  # noqa: BLE001
        if is_oom(e):
            raise
        reasons.append(f"gate failed: {type(e).__name__}: {e}")
    if not dry_run:
        late = late_activation_reason(mode)
        if late:
            reasons.append("late activation refused: " + late)
    if reasons:
        rep["reason"] = "; ".join(reasons)
        rep["levers_unavailable"] = list(res["levers"])
        if dry_run:
            rep["would_refuse"] = rep["reason"]
            if print_line:
                _report.print_activation(rep, dry_run=True)
            return dict(rep)
        _STATE["report"] = rep
        _report.print_activation(rep)
        if manifest_dir:
            _manifest.write(manifest_dir, rep)
        if strict:
            raise ActivationError(rep["reason"])
        return dict(rep)
    counts = [_inputs.tokens(q) for q in queries] if queries is not None else []      # the run's size: every query's residues, every copy of a chain counted
    rep.update(tokens_min=min(counts) if counts else None, tokens_max=max(counts) if counts else None, queries=len(counts) if queries is not None else None)
    if dry_run:
        if print_line:
            _report.print_activation(rep, dry_run=True)
        return dict(rep)
    _place_deployment(rep)                                               # the deployment levers (XLA_CACHE): placed in each of the kit's modes before any lever compiles — unless ablated
    # --- the mode's levers, in the table's order: the package's transfer lever (device_resident.enable), then the carried kit's own row
    # (af2_pallas_flash/README.md:23-25: its directory on the path, its switch exported, its enable() called)
    kit = None
    refused: Dict[str, str] = {}                                         # lever -> the refusal that kept it off: EVERY lever is tried, one line names them all
    held: Dict[str, str] = {}                                            # lever -> the named fallback word of a lever held on this part; no lever of this kit reports one (the dict stays empty; the refusal route below reads it)
    try:
        if mode in _modes.TP_MODES and _modes.TP_LEVER not in res["levers"]:      # big at P = 1: the axis is not in the process's lever set (fast's bytes); apply(1) records
            importlib.import_module(_modes.TP_LEVER_MODULE).apply(int(n_gpu))    # the axis' tokens and installs no sharding
        for name in res["levers"]:
            try:
                if name == _modes.HOST_LEVER:
                    importlib.import_module(_modes.HOST_LEVER_MODULE).enable()
                elif name == _modes.SUBBATCH_LEVER:                       # the sub-batch value for this run: the largest input's tokens against the visible device (subbatch.decide)
                    mem = (rep.get("gpu") or {}).get("memory_mib")
                    rep["subbatch"] = importlib.import_module(_modes.SUBBATCH_LEVER_MODULE).enable(
                        tokens=rep.get("tokens_max"), device_bytes=int(mem) * 2 ** 20 if mem else None, n_gpu=int(n_gpu))   # the axis: P > 1 takes the chunk (subbatch.decide)
                elif name == _modes.TRIMUL_LEVER:                         # the fused triangle multiplication through the shared core's provider face by the mode's tier word: the class rebound before any model is traced (trimul_pallas.enable; its floor by name)
                    importlib.import_module(_modes.TRIMUL_LEVER_MODULE).enable(mode=mode)
                elif name == _modes.MSA_LEVER:                            # MSA-column + extra-MSA row attention through the flash kernel: Attention rebound over the carried lever's class (msa_attn.enable)
                    importlib.import_module(_modes.MSA_LEVER_MODULE).enable()
                elif name == _modes.TRIATTN_LEVER:                        # the provider's pre-compiled triangle-attention row allowed inside AF_PALLAS_ATTN's binding (triattn_xla.enable; the bridge's load floor by name)
                    importlib.import_module(_modes.TRIATTN_LEVER_MODULE).enable()
                elif name == _modes.COL_LEVER:                            # MSA column attention through cuDNN with key lengths: Attention rebound over PALLAS_MSA's class (msa_col_cudnn.enable; its floor + probe by name)
                    importlib.import_module(_modes.COL_LEVER_MODULE).enable()
                elif name == _modes.TEMPL_LEVER:                          # identical template rows embedded once: modules_multimer.TemplateEmbedding rebound + the host census on model.RunModel (templ_dedup.enable)
                    importlib.import_module(_modes.TEMPL_LEVER_MODULE).enable()
                elif name == _modes.TRANSITION_LEVER:                     # AlphaFold's Transition on the shared core's provider face by this mode's tier word: modules.Transition rebound before any model is traced (transition.enable)
                    importlib.import_module(_modes.TRANSITION_LEVER_MODULE).enable(mode=mode)
                elif name == _modes.TP_LEVER:                             # big's axis at P > 1: the mesh + the recipe
                    importlib.import_module(_modes.TP_LEVER_MODULE).apply(int(n_gpu))
                elif name == _modes.LEVER:
                    kp = kit_pythonpath()
                    if kp not in sys.path:
                        sys.path.insert(0, kp)
                    os.environ.update(res["kit_env"])
                    route_kernel_to_core()                               # one kernel per process: `import af2_flash_pallas` in the carried adapter answers with the core's carried copy
                    kit = importlib.import_module(_modes.KIT_MODULE)     # with the switch set, the kit enables itself at import (:124-125)
                    st = kit_state()
                    if not (st and st.get("enabled")):
                        kit.enable()                                     # the module was imported before the switch: the kit's own call (:69)
                    rep["attn_word"] = apply_pair_word(kit, mode)        # the adapter's FLASH_OP = the shared core provider's attention face by the mode's tier word (triattn_xla.bind); word= / served= on its LEVER line
                else:
                    raise RuntimeError(f"mode {mode} names a lever this package cannot apply: {name}")
            except Exception as e:  # noqa: BLE001
                if is_oom(e):                                             # a lever's installation ran out of device memory: the run stops there — no NOT ACTIVE route, no stock run
                    raise
                refused[name] = repr(e)                                   # keep the words, try the next lever: one pass names every refusal
    except Exception as e:  # noqa: BLE001 — the axis' own refusal at P = 1 (big.apply before the lever loop)
        if is_oom(e):
            raise
        refused[_modes.TP_LEVER] = repr(e)
    if refused:
        rep["reason"] = "enable() failed: " + "; ".join(f"{n}: {why}" for n, why in refused.items())
        rep["levers_unavailable"] = list(res["levers"])
        if held:
            rep["lever_fallbacks"] = dict(held)
        _STATE["report"] = rep
        _report.print_activation(rep)
        if manifest_dir:
            _manifest.write(manifest_dir, rep)
        if strict:
            raise ActivationError(rep["reason"]) from None
        return dict(rep)
    st = kit_state()
    rep["kit_state"] = dict(st) if st else None
    hs = host_lever_state()
    rep["lever_states"] = {n: (dict(s) if s else None) for n, s in lever_states().items() if n in res["levers"]}
    probes = _registry.applied(kit)                                      # the registry's probes against each lever's own state (registry.py)
    rep["levers_applied"] = [n for n in res["levers"] if probes.get(n)]
    rep["levers_unavailable"] = [n for n in res["levers"] if not probes.get(n)]
    rep["patch_marker"] = _registry.patch_marker_present() if _modes.LEVER in res["levers"] else None   # Attention._pallas_patched (af2_pallas_attn.py:96)
    rep["markers"] = {n: _registry.marker_present(n) for n in res["levers"]}                               # lever -> the marker its patch left on the class it rebinds
    rep["switches_not_wired"] = {n: probes.get(n) for n in _registry.NOT_WIRED}
    unmarked = [n for n in rep["levers_applied"] if not rep["markers"].get(n)]
    if not rep["levers_unavailable"] and unmarked:                       # enabled=True with no patched class: what haiku / colabfold would run is stock
        rep["levers_applied"], rep["levers_unavailable"] = [], list(res["levers"])
        m = _registry.MARKERS[unmarked[0]]
        rep["reason"] = (f"{unmarked[0]}: enable() returned enabled=True but {m[1]}.{m[2]} carries no {m[3]} marker: "
                         f"the patch did not land (the stock class would run)")
        _STATE["report"] = rep
        _report.print_activation(rep)
        if manifest_dir:
            _manifest.write(manifest_dir, rep)
        if strict:
            raise ActivationError(rep["reason"])
        return dict(rep)
    if rep["levers_unavailable"] and held and all(n in held for n in rep["levers_unavailable"]):   # a lever of the mode cannot run on this part (no tile table): a mode is all of its levers — the run is refused by name, ONE line, exit 3
        rep["lever_fallbacks"] = {n: held[n] for n in rep["levers_unavailable"]}
        rep.update(active=False, reason="lever_fallback: " + _report.lever_fallback_detail(rep))
        _STATE["report"] = rep
        _report.emit(_report.lever_refused_line(rep))                    # NOT ACTIVE: <LEVER>=fallback:no_tiles_cc<NN>(<kind>): …; exit 3 (--mode off runs stock)
        if manifest_dir:
            _manifest.write(manifest_dir, rep)
        if strict:
            raise ActivationError(rep["reason"])
        return dict(rep)
    if rep["levers_unavailable"]:
        rep["reason"] = f"enable() returned but the probe is not true for {rep['levers_unavailable']}: states={rep['lever_states']!r}"
        _STATE["report"] = rep
        _report.print_activation(rep)
        if strict:
            raise ActivationError(rep["reason"])
        return dict(rep)
    rep.update(active=True, reason=None)
    _STATE["report"] = rep
    _report.print_activation(rep)
    _register_tally(manifest_dir)
    if manifest_dir:
        _manifest.write(manifest_dir, rep)
    return dict(rep)


def _place_deployment(rep: dict) -> None:
    """The deployment levers, in registry order, each unless MODEL_OPT_LEVERS_OFF names it (its LEVER line then reads state=off reason=ablated)."""
    abl = set(rep.get("levers_ablated") or [])
    from . import xla_cache as _xla_cache
    if _xla_cache.NAME in abl:
        _xla_cache._STATE.update(enabled=False, state="off", reason=_ablation.REASON, detail=_ablation.ENV, source=None, dir=None)
        rep["xla_cache"] = dict(_xla_cache._STATE)
    else:
        _place_xla_cache(rep)


def _place_xla_cache(rep: dict) -> None:
    """The deployment lever: JAX's persistent compilation cache keyed by the stack (xla_cache.apply); its state into the report."""
    from . import xla_cache as _xla_cache
    try:
        rep["xla_cache"] = _xla_cache.apply(stack_key())
    except Exception as e:  # noqa: BLE001 — a placement failure is named on the lever's line, never fatal to the run; out-of-memory propagates
        if is_oom(e):
            raise
        _xla_cache._STATE.update(enabled=False, state="skipped", reason="placement_error", detail=repr(e), source=None, dir=None)
        rep["xla_cache"] = dict(_xla_cache._STATE)


def _register_tally(manifest_dir: Optional[str]) -> None:
    """The exit lines of this process: the carried kit's `EXIT _STATE` line when the mode applies the kernel, one LEVER line per lever of
    the table (on with its live counters / off not in this mode / skipped with the reason), and the manifest's exit record."""
    from . import manifest as _manifest
    if manifest_dir:
        _STATE["manifest_dir"] = manifest_dir
    rep = _STATE["report"] or {}
    in_mode, applied = list(rep.get("levers") or []), list(rep.get("levers_applied") or [])
    skipped_reason = "activation_refused"   # a blank-free token for the LEVER line; the text is the NOT ACTIVE line's and the manifest's
    states = lever_states()
    abl = set(rep.get("levers_ablated") or [])                           # MODEL_OPT_LEVERS_OFF removed them from the mode for this run: off / ablated (ablation.REASON)
    levers = {}
    for name in _registry.IN_MODE:
        if name in abl:
            levers[name] = ("off", _ablation.REASON, None)
        elif name not in in_mode:
            levers[name] = ("off", (rep.get("levers_dropped") or {}).get(name, "not_in_mode"), None)
        elif name in applied and rep.get("active"):
            levers[name] = ("on", None, functools.partial(_exit_evidence, name))   # the live counters read at exit, plus the by-design words (superseded_by= / no_model_run=) when they apply
        elif name in (rep.get("lever_fallbacks") or {}):                # a table-backed lever held on this part: its named fallback word
            levers[name] = ("skipped", rep["lever_fallbacks"][name], None)
        else:
            levers[name] = ("skipped", skipped_reason, None)
    if rep.get("mode") in _modes.TP_MODES:                               # big: the axis' line is the family's (opt_core.mem.rowpair_jax.evidence.line: on at P>1, off reason=n_gpu=1 at P=1)
        levers[_modes.TP_LEVER] = (levers[_modes.TP_LEVER][0], levers[_modes.TP_LEVER][1], importlib.import_module(_modes.TP_LEVER_MODULE).exit_line)
    from . import xla_cache as _xla_cache                                # the deployment lever's line: its placement now, its entries read at exit
    xs = _xla_cache._STATE
    if _xla_cache.NAME in abl:
        levers[_xla_cache.NAME] = ("off", _ablation.REASON, None)
    elif xs["state"] == "on":
        levers[_xla_cache.NAME] = ("on", None, _xla_cache.exit_evidence)
    else:                                                                # skipped at placement (its own reason) or never placed (the activation's hold)
        levers[_xla_cache.NAME] = ("skipped", xs["reason"] if xs["state"] == "skipped" else skipped_reason, None)
    _report.register_exit_tally(kit_state() if _modes.LEVER in in_mode else None, _modes.KIT_MODULE if _modes.LEVER in in_mode else None,
                                on_exit=lambda st: _manifest.record_exit(_STATE["manifest_dir"], st, lever_states()) if _STATE["manifest_dir"] else None,
                                levers=levers)



def _exit_evidence(name: str):
    """An applied lever's exit counters, read NOW (its live ``_STATE``), plus the word that names why `calls=0` is by design on this run when
    one applies — `superseded_by=<lever>:<calls>`: nothing reached it and the lever bound over it served the calls (modes.SUPERSEDES;
    manifest.superseded_by); `no_model_run=<case>`: colabfold_batch built no model at all (manifest.no_model_case) — the same readings the
    manifest records and `pred`'s verdict makes. The counters alone otherwise."""
    from . import manifest as _manifest
    st = lever_states().get(name)
    if not isinstance(st, dict):
        return st
    man = {"activation_report": _STATE["report"] or {}, "lever_states_exit": {n: (dict(s) if isinstance(s, dict) else None) for n, s in lever_states().items()}}
    out = dict(st)
    word = _manifest.superseded_by(man, name)
    if word is not None:
        out["superseded_by"] = word
    case = _manifest.no_model_case(man)
    calls = st.get("calls")
    if case is not None and not (isinstance(calls, int) and calls >= 1):
        out["no_model_run"] = case
    return out if len(out) != len(st) else st

def status() -> dict:
    rep = _STATE["report"]
    return dict(rep) if rep is not None else {"active": False, "reason": "no activation in this process"}


def check(mode: str, data_dir: Optional[str] = None, print_line: bool = True, n_gpu: Optional[int] = None, refresh_digests: bool = False) -> dict:
    """The dry run: the same resolution and gates as the activation, nothing applied, nothing written (``refresh_digests``: the `check`
    verb — the parameter digests are computed afresh and their memo entries rewritten; `pred`'s gate and the hook read the memo)."""
    return activate(mode, dry_run=True, data_dir=data_dir, print_line=print_line, n_gpu=n_gpu, refresh_digests=refresh_digests)


# --------------------------------------------------------------------------------------------------------------- the hook
def hook_run(mode: str, strict: bool = True, trigger: Optional[str] = None, module=None):
    """Wrap ``colabfold.batch.run`` so the activation happens at its call, with the run's queries and data_dir; then the stock
    ``run``. Under ``strict`` a refusal exits the process with 3 (stock never runs silently); otherwise (the explicit route) the stock run
    proceeds after the NOT ACTIVE line. Installed once per process;
    returns the wrapper."""
    cb = module if module is not None else importlib.import_module(TRIGGER_MODULE)
    orig = getattr(cb, RUN_FUNCTION)
    if getattr(orig, "_colabfold_opt_hook", False):
        return orig

    @functools.wraps(orig)
    def run(*args, **kwargs):
        queries = kwargs["queries"] if "queries" in kwargs else (args[0] if args else None)
        data_dir = kwargs.get("data_dir")
        from . import manifest as _manifest
        census = no_model_run_census(orig, args, kwargs, queries, safe_filename=getattr(cb, "safe_filename", None))   # will colabfold_batch build a model at all on this run — from run()'s own arguments, before it runs (manifest.no_model_run)
        try:
            activate(mode, queries=queries, strict=strict, trigger=trigger, route="env" if trigger else "hook",
                     manifest_dir=_manifest.work_dir(), data_dir=str(data_dir) if data_dir is not None else None)   # the manifest goes to the launch's work directory `pred` named (absent: none is written)
        except _modes.UnsupportedMode as e:                              # `exact` or an unknown COLABFOLD_OPT value: refused here, by name
            _report.emit(_report.NOT_ACTIVE_FMT.format(prefix=_report.PREFIX, reason=str(e)))
            if strict:
                sys.exit(_report.EXIT_NOT_ACTIVE)
        except ActivationError:
            sys.exit(_report.EXIT_NOT_ACTIVE)                            # the NOT ACTIVE line is printed; the process stops here
        record_no_model_run(census)                                      # into the live report and its manifest: the exit readings (post_run_partial, the LEVER lines, record_exit) and `pred`'s verdict follow it
        _STATE["running"] += 1
        try:
            out = orig(*args, **kwargs)
        finally:
            _STATE["running"] -= 1
        post_run_partial(strict)
        return out

    run._colabfold_opt_hook = True          # type: ignore[attr-defined]
    run._colabfold_opt_stock = orig         # type: ignore[attr-defined]
    setattr(cb, RUN_FUNCTION, run)
    _STATE["hooked"] = {"mode": mode, "strict": strict, "trigger": trigger}
    return run


def post_run_partial(strict: bool) -> None:
    """After the stock ``run`` returns under the hook: the activation active and the kit's counters showing no kernel call
    (manifest.partial_levers — the one reading, the same the manifest records at exit) is a PARTIAL activation — the mode ran with one of
    its levers doing nothing: the package owns the process exit on this route: the partial-activation line and exit 3 under ``strict``. A run on which
    colabfold_batch built no model at all (manifest.no_model_case: ``--num-models 0`` / ``--msa-only``, every job already complete — read from
    ``run``'s arguments before it ran) left the levers with no call to serve: the IDLE line names it (under ``strict`` when this process is the
    entry point; `pred` prints it from its verdict) and the run keeps stock's own exit — never exit 3. A fallback share
    beyond the documented class (manifest.excess_fallbacks, `fallback_excess`) is named on ONE line and never an exit: those calls ran the
    stock operation. Nothing when the activation was not active (a refusal exited already)."""
    from . import manifest as _manifest
    rep = _STATE["report"]
    if not rep or not rep.get("active"):
        return
    st = kit_state()
    man = {"activation_report": rep, "kit_state_exit": dict(st) if st else None,
           "lever_states_exit": {n: (dict(s) if s else None) for n, s in lever_states().items()}}
    case = _manifest.no_model_case(man)
    if case is not None:                                                  # no model was built (by the tool's own facts): the levers had nothing to serve — named once, the run's own exit
        if strict and not launched_by_pred():
            idle = _manifest.idle_levers(man)
            _report.emit(_report.idle_line(case, idle, rep.get("queries"),
                                           _manifest.no_model_detail(case, how=rep.get("no_model_run_how"), done=rep.get("no_model_run_done"), idle=idle)))
        return
    partial = _manifest.partial_levers(man)
    if partial:                                                           # no kernel call: the lever did not run — the mode ran with a subset in effect: PARTIAL by name, exit 3
        if strict:
            _report.emit(_report.partial_exit_line(_manifest.engaged_detail(partial, man)))
            sys.exit(_report.EXIT_NOT_ACTIVE)
        return
    excess = _manifest.excess_fallbacks(man)                              # the kernel engaged but fell back beyond the documented class: ONE named line, the run's own exit
    if excess and strict and not launched_by_pred():                     #  (those calls ran the stock operation; `pred` prints the line from its verdict, the manifest records it)
        _report.emit(_report.fallback_excess_line(_manifest.fallback_detail(excess, _manifest.fallback_census(dict(st), _manifest.superseded_calls(man, _modes.LEVER)))))


def launched_by_pred() -> bool:
    """True in a model process `pred` launched (it exports the launch id, manifest.ENV_LAUNCH_ID): `pred` prints the fallback_excess line of
    its verdict once, the model process prints it only when it is the entry point (the env route on its own)."""
    from . import manifest as _manifest
    return _manifest.ENV_LAUNCH_ID in os.environ


def run_argument(orig, args: tuple, kwargs: dict, name: str, default=None):
    """The value ``colabfold.batch.run`` receives for its parameter `name` on this call: the keyword when passed, else the positional at the
    parameter's index in ``run``'s own signature, else that parameter's default there, else `default`."""
    if name in kwargs:
        return kwargs[name]
    try:
        params = inspect.signature(orig).parameters
    except (TypeError, ValueError):
        return default
    names = list(params)
    if name in names:
        i = names.index(name)
        if params[name].kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) and i < len(args):
            return args[i]
        if params[name].default is not inspect.Parameter.empty:
            return params[name].default
    return default


def no_model_run_census(orig, args: tuple, kwargs: dict, queries, safe_filename=None) -> dict:
    """Whether colabfold_batch will build a model at all on this ``run`` call, read from its own arguments before it runs (manifest.no_model_run):
    ``num_models`` (0 under ``--num-models 0`` / ``--msa-only``), and — under ``keep_existing_results`` (no ``--overwrite-existing-results``) —
    the completion artefact of every job the queries name (inputs.job_names: colabfold's own naming; manifest.completion: its own finished-job
    test) under ``result_dir``. ``{"case": <word|None>, "how": …, "done": {job: artefact}}``; a census that cannot be taken (no queries, an
    argument colabfold itself will reject) is ``case=None``: a model is expected and the partial rule stands."""
    from . import manifest as _manifest
    out = {"case": None, "how": None, "done": {}}
    try:
        num_models = run_argument(orig, args, kwargs, "num_models")
        result_dir = run_argument(orig, args, kwargs, "result_dir")
        keep = run_argument(orig, args, kwargs, "keep_existing_results", True)
        prefix = run_argument(orig, args, kwargs, "jobname_prefix", None)
        names = _inputs.job_names(queries, prefix, safe_filename=safe_filename) if queries is not None else []
        rd = str(result_dir) if result_dir is not None else None
        done = {n: _manifest.completion(rd, n) for n in names}
        case = _manifest.no_model_run(num_models, names, rd, keep_existing=bool(keep))
        out.update(case=case, how=(f"num_models={num_models}" if case == _manifest.NUM_MODELS_0 else None),
                   done={n: a for n, a in done.items() if a} if case == _manifest.ALL_JOBS_DONE else {})
    except Exception as e:  # noqa: BLE001 — the census is read-only bookkeeping; a failure to take it leaves the run and the partial rule as they are
        if is_oom(e):
            raise
        out["error"] = repr(e)
    return out


def record_no_model_run(census: Optional[dict]) -> None:
    """The census into this process's live activation report (``no_model_run``, ``no_model_run_how``, ``no_model_run_done`` — the exit
    readings and the LEVER lines follow it) and, when `pred` named a work directory, into the manifest there (its verdict reads the model
    process's own decision). A report of an earlier run in this process is overwritten: the case is this run's."""
    from . import manifest as _manifest
    rep = _STATE["report"]
    if rep is None:
        return
    census = census or {}
    had = rep.get("no_model_run")
    rep["no_model_run"] = census.get("case")
    rep["no_model_run_how"] = census.get("how")
    rep["no_model_run_done"] = dict(census.get("done") or {})
    mdir = _manifest.work_dir()
    if (rep["no_model_run"] is not None or had is not None) and mdir and os.path.isfile(_manifest.path(mdir)):
        _manifest.write(mdir, rep)


def reset_for_tests() -> None:
    _STATE.update(report=None, running=0, manifest_dir=None, hooked=None)
    _CACHE.clear()
    _report.reset_tally_for_tests()
    m = sys.modules.get(_modes.HOST_LEVER_MODULE)
    if m is not None:
        m.reset_for_tests()
    x = sys.modules.get("colabfold_opt.xla_cache")
    if x is not None:
        x.reset_for_tests()
    b = sys.modules.get(_modes.TP_LEVER_MODULE)
    if b is not None:
        b.reset_for_tests()
    for name in (_modes.SUBBATCH_LEVER_MODULE, _modes.TRIMUL_LEVER_MODULE, _modes.MSA_LEVER_MODULE, _modes.TRIATTN_LEVER_MODULE, _modes.COL_LEVER_MODULE,
                 _modes.TEMPL_LEVER_MODULE, _modes.TRANSITION_LEVER_MODULE):
        lm = sys.modules.get(name)
        if lm is not None:
            lm.reset_for_tests()
