"""Paths, pins, the process environment and the installed-tree state — the facts every other module of caliby_opt reads.

The levers of both kits are whole-file replacements of upstream modules of ``caliby`` / ``chroma`` / ``protpardelle`` (``TOUCHED``);
``exact`` loads the kits' files by import hook (overlay.py) over an installed tree that stays the pinned upstream, and the kits' own
by-hand install scripts (outside this tree) put the same files into site-packages for a user patching a tree by hand;
nothing here applies a lever. This module answers: where is the tree (``opt_home`` / ``tree_home`` / ``kit_dir``), what are the pins
(``pins``), which upstream files do the kits touch and what state is the installed tree in (``installed_files`` / ``tree_state``:
``stock`` | ``exact`` | ``mixed`` | ``unknown`` | ``absent``), what does the environment carry (``kit_switches_in`` /
``strip_env`` / ``stock_env_proof``), and can the box run a mode (``gpu_info`` / ``compiler`` / ``weights_gate``). Reference digests
(``expected_digests``, for classifying the INSTALLED tree only) are hashed on the spot from the kit's own ``stock/`` and ``fast/``
files — no digest of repo content is ever stored (repo files are tracked in this tree). ``TAG`` is the kit's one tag on every line it prints
(``[caliby-opt]``: report.PREFIX, the autoload ``.pth`` guard, the core pin
gate); ``core_gate()`` is statement one of every entry of the package (``python -m caliby_opt`` / the console script, ``enable()`` /
``status()`` / ``check()``, the design children, the ``.pth`` finder's trigger, ``run.sh``, ``configs/<gpu>.env``): the core pin gate
(``_core_gate.gate`` — ``common/opt_core/kit_template/_core_gate.py`` byte for byte) compares ``opt/pyproject.toml``
``[tool.opt_core]`` (path + MINIMUM version) with the installed ``opt_core``'s own ``__version__``, located without importing it, and
refuses by name — one ``[caliby-opt] NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: … | core_pin_unreadable: …`` line,
exit 3 — on an absent, older or unreadable-pin core; a newer core passes (the pin is a floor). Nothing upstream is imported by this
module (torch only in gpu_info()); the shared core
(``opt_core``) only inside the functions that use it, and nothing of it at module level (the standard library and ``_core_gate`` only).
"""
from __future__ import annotations

import hashlib
import importlib.metadata as _md
import importlib.util
import json
import os
import platform
import shutil
import sys
from typing import Dict, List, Optional, Tuple

from ._core_gate import gate as _gate                                  # the core's kit template, inside this package: standard library only, imports nothing of the core

TAG = "caliby-opt"                                                     # the kit's tag on every line it prints ([caliby-opt] ...): report.PREFIX, the .pth guard line, the core pin gate's line — one spelling
ENV_MODE, ENV_VARIANT = "CALIBY_OPT", "CALIBY_VARIANT"                 # the package's own two switches
ENV_TREE = "MODEL_OPT"                                                 # run.sh convention: this model's directory (caliby/)
ENV_STATE = "MODEL_OPT_STATE"                                          # where tree snapshots and the kits' install logs live (out of tree)
ENV_LABEL = "MODEL_OPT_ENV_LABEL"                                      # a free-text label of the container / host, echoed as the STACK line's image= (report.stack_line); never compared
ENV_TARGET_GPU = "MODEL_OPT_TARGET_GPU"
PACKAGE_ENV = (ENV_MODE, ENV_VARIANT)
KIT_SWITCH_PREFIXES = ("CALIBY_FAST_", "CALIBY_X_")                    # the two kits' own switches (PINS.json stock_environment)
STOCK_ABSENT_PREFIXES = KIT_SWITCH_PREFIXES + ("CALIBY_OPT", "CALIBY_VARIANT")
DATA_ENV = ("MODEL_PARAMS_DIR", "HF_HUB_OFFLINE", "PDB_MIRROR_PATH", "CCD_MIRROR_PATH")   # what upstream reads (PINS.json stock_environment.reads)
KIT_PARTNER, KIT_ADDON = "fast_inference", "xattempt_addon"           # opt/forward/<kit>
KIT_CLASS_DIRS = ("forward", "datapath")
PINS_RELPATH = os.path.join("stock", "PINS.json")
PYPROJECT_RELPATH = os.path.join("opt", "pyproject.toml")                # carries [tool.opt_core]: the shared core's pin (core_gate reads it beside the package; this path is the tree's copy for tools and tests)
CHECK_PINS_RELPATH = os.path.join("stock", "check_pins.py")
UPSTREAM_DISTS = ("caliby", "atomworks-caliby", "protpardelle")
# modules whose file the kits replace: once one is imported, a file replacement no longer reaches this process
LEVER_MODULES = ("caliby.api", "caliby.model.seq_denoiser.denoisers.seq_design.potts", "caliby.model.seq_denoiser.denoisers.atom_mpnn_denoiser",
                 "caliby.eval.eval_utils.seq_des_utils", "caliby.eval.eval_utils.inference_dataloader", "chroma.layers.complexity",
                 "protpardelle.core.models", "protpardelle.data.pdb_io")
# the installed path of every touched file, relative to its package root (the paths a by-hand install of the kits writes to)
TOUCHED = {
    "potts.py": ("caliby", "model/seq_denoiser/denoisers/seq_design/potts.py"),
    "atom_mpnn_denoiser.py": ("caliby", "model/seq_denoiser/denoisers/atom_mpnn_denoiser.py"),
    "api.py": ("caliby", "api.py"),
    "seq_des_utils.py": ("caliby", "eval/eval_utils/seq_des_utils.py"),
    "inference_dataloader.py": ("caliby", "eval/eval_utils/inference_dataloader.py"),
    "complexity.py": ("chroma", "layers/complexity.py"),
    "clean_pdbs.py": ("caliby", "data/preprocessing/atomworks/clean_pdbs.py"),
    "protpardelle_core_models.py": ("protpardelle", "core/models.py"),
    "protpardelle_data_pdb_io.py": ("protpardelle", "data/pdb_io.py"),
}
PP_FILES = ("protpardelle_core_models.py", "protpardelle_data_pdb_io.py")
# replaced by no kit file and named by no kit manifest: expected at its
# upstream digest (stock/src) in every tree state
NEVER_REPLACED = ("clean_pdbs.py",)
STOCK_SRC = {"clean_pdbs.py": os.path.join("stock", "src", "caliby", "caliby", "data", "preprocessing", "atomworks", "clean_pdbs.py")}
CORE_FILES = tuple(n for n in TOUCHED if n not in PP_FILES and n not in NEVER_REPLACED)
PIP_FREEZE_RULE = "sha256 over the sorted non-empty lines of `python -m pip freeze`, newline-joined with a trailing newline"


# ----------------------------------------------------------------------------------------------------------------- paths
def opt_home() -> str:
    """caliby/opt — the directory the package is installed from (editable)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tree_home() -> str:
    """caliby/ — the model's directory ($MODEL_OPT when set, else the parent of opt_home())."""
    h = os.environ.get(ENV_TREE)
    return os.path.abspath(h) if h else os.path.dirname(opt_home())


def kit_dir(name: str) -> str:
    """opt/<class>/<name> for the two kits (the class directories are searched in order)."""
    for cls in KIT_CLASS_DIRS:
        d = os.path.join(tree_home(), "opt", cls, name)
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(f"kit {name!r} not found under {os.path.join(tree_home(), 'opt')}/{{{','.join(KIT_CLASS_DIRS)}}}/ "
                            f"(the package stays installed editable from caliby/opt; {ENV_TREE}, when set, must name a caliby/ tree that holds the kits)")


def state_dir() -> str:
    """Out-of-tree state: tree snapshots and the kits' install logs ($MODEL_OPT_STATE, else ~/.cache/caliby_opt)."""
    d = os.environ.get(ENV_STATE) or os.path.join(os.path.expanduser("~"), ".cache", "caliby_opt")
    os.makedirs(d, exist_ok=True)
    return d


def pins_path() -> str:
    return os.path.join(tree_home(), PINS_RELPATH)


def pyproject_path() -> str:
    return os.path.join(tree_home(), PYPROJECT_RELPATH)


def core_gate() -> dict:
    """Statement one of every entry: the core pin gate under this package's tag (``_core_gate.gate`` anchored on this package,
    so the pin it reads is the ``[tool.opt_core]`` of the ``opt/pyproject.toml`` this package is installed from). Returns the gate's facts
    (``{"pinned": {path, version, pyproject}, "installed": {package_dir, root, version}, "tag"}``); an absent / older core or an
    unreadable pin is its one ``[caliby-opt] NOT ACTIVE: reason=...`` line on
    stderr and ``SystemExit(3)`` (``_core_gate.CoreGateRefused``) — never a traceback, never a silent stock run. Idempotent and cheap:
    activation calls it again for the report's core block (the same producer, not a second gate)."""
    return _gate(__file__, tag=TAG)


def pins() -> dict:
    with open(pins_path(), "r", encoding="utf-8") as fh:
        return json.load(fh)


# ----------------------------------------------------------------------------------------------------------------- installed tree
def package_root(name: str) -> Optional[str]:
    """Directory of an installed top-level package without executing it (find_spec on a top-level name imports nothing)."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return list(spec.submodule_search_locations)[0]


def sha256_file(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_digests() -> Dict[str, Dict[str, Optional[str]]]:
    """Per touched file: {"upstream": sha, "addon": sha|None} — each hashed directly from the corresponding repo file (the kit's own
    stock/ and fast/ copies; stock/src for the one NEVER_REPLACED file), never read from a stored digest table: a repo file is tracked in
    this tree, so there is nothing to cache. These are the values that classify the INSTALLED tree below (installed_files()) — the
    comparison's subject is always the installed, out-of-git file, never a repo one."""
    kx = kit_dir(KIT_ADDON)
    out: Dict[str, Dict[str, Optional[str]]] = {}
    for name in TOUCHED:
        if name in STOCK_SRC:                                          # clean_pdbs.py: NEVER_REPLACED, no lever touches it
            up, addon = sha256_file(os.path.join(tree_home(), STOCK_SRC[name])), None
        else:                                                          # the eight replaced modules: the kit's stock/ and fast/ copies
            up, addon = sha256_file(os.path.join(kx, "stock", name)), sha256_file(os.path.join(kx, "fast", name))
        out[name] = {"upstream": up, "addon": addon}
    return out


def installed_files() -> Dict[str, dict]:
    """Every touched file on this interpreter's path: {name: {path, sha256, state}}, state upstream|addon|absent|unknown|package-absent."""
    exp = expected_digests()
    roots = {p: package_root(p) for p in ("caliby", "chroma", "protpardelle")}
    out = {}
    for name, (pkg, rel) in TOUCHED.items():
        root = roots.get(pkg)
        path = os.path.join(root, rel) if root else None
        sha = sha256_file(path)
        if root is None:
            state = "package-absent"
        elif sha is None:
            state = "absent"
        elif sha == exp[name]["addon"]:
            state = "addon"
        elif sha == exp[name]["upstream"]:
            state = "upstream"
        else:
            state = "unknown"
        out[name] = {"path": path, "sha256": sha, "state": state}
    return out


def tree_state(files: Optional[Dict[str, dict]] = None) -> str:
    """stock: every touched file upstream; exact: the kit's files installed (the protpardelle pair
    addon, or upstream when the ensemble levers were skipped, or that package absent);
    unknown: a digest none of the manifests names; mixed otherwise; absent: caliby not installed. A NEVER_REPLACED file (tracked by
    the kit's tree tool, written by no install) must be upstream in every state."""
    f = files if files is not None else installed_files()
    if f["api.py"]["state"] == "package-absent":
        return "absent"
    core = {n: f[n]["state"] for n in CORE_FILES}
    pp = {n: f[n]["state"] for n in PP_FILES}
    fixed = {n: f[n]["state"] for n in NEVER_REPLACED}
    if any(s != "upstream" for s in fixed.values()):                   # a file no install may change is not the pin's: never stock, never exact
        return "unknown" if any(s == "unknown" for s in fixed.values()) else "mixed"
    if all(s == "upstream" for s in core.values()) and all(s in ("upstream", "package-absent") for s in pp.values()):
        return "stock"
    if all(s == "addon" for s in core.values()) and all(s in ("addon", "upstream", "package-absent") for s in pp.values()):
        return "exact"
    if any(s == "unknown" for s in list(core.values()) + list(pp.values())):
        return "unknown"
    return "mixed"


DIGEST_TREES = ("caliby", "chroma", "protpardelle")                    # the installed trees the tree digest covers (stock/PINS.json "tree_digest_upstream")


def tree_digest() -> Tuple[str, dict]:
    """The digest over the installed caliby/, chroma/ and protpardelle/ trees that stock/PINS.json "tree_digest_upstream" pins: per tree,
    sha256 over ``<relative path>\\0<sha256 of the file>\\n`` of every file in sorted walk order (``__pycache__`` directories and
    ``.pyc`` / ``.pyo`` files excluded), then sha256 over ``<tree>\\0<tree digest>\\n`` of the three trees in name order; a tree not
    installed hashes as empty. The packages are located with ``find_spec`` (``package_root``): nothing upstream is imported.
    Returns (sha256 hex, {tree: {root, n_files, sha256}})."""
    per = {}
    hall = hashlib.sha256()
    for name in sorted(DIGEST_TREES):
        root = package_root(name)
        h = hashlib.sha256()
        n = 0
        if root and os.path.isdir(root):
            for dp, dns, fns in os.walk(root):
                dns[:] = sorted(d for d in dns if d != "__pycache__")
                for fn in sorted(fns):
                    if fn.endswith((".pyc", ".pyo")):
                        continue
                    fp = os.path.join(dp, fn)
                    with open(fp, "rb") as fh:
                        fh_hex = hashlib.sha256(fh.read()).hexdigest()
                    h.update(f"{os.path.relpath(fp, root)}\0{fh_hex}\n".encode())
                    n += 1
        per[name] = {"root": root, "n_files": n, "sha256": h.hexdigest()}
        hall.update(f"{name}\0{per[name]['sha256']}\n".encode())
    return hall.hexdigest(), per


def lever_modules_loaded() -> List[str]:
    return [m for m in LEVER_MODULES if m in sys.modules]


def pip_freeze_sha256() -> Tuple[Optional[str], int]:
    """(digest, line count) of this interpreter's ``pip freeze`` by ``PIP_FREEZE_RULE`` — the STACK line's ``pip_freeze_sha256``; (None, 0)
    when pip cannot be run."""
    import subprocess
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):                     # a report value, never a gate
        return None, 0
    if r.returncode != 0:
        return None, 0
    lines = sorted(ln for ln in r.stdout.splitlines() if ln.strip())
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest(), len(lines)


def upstream_versions() -> dict:
    out = {}
    for name in UPSTREAM_DISTS:
        try:
            dist = _md.distribution(name)
            raw = dist.read_text("direct_url.json")
            info = json.loads(raw) if raw else {}
            out[name] = {"version": dist.version, "commit": (info.get("vcs_info") or {}).get("commit_id")}
        except _md.PackageNotFoundError:
            out[name] = None
    return out


def check_pins_detail() -> Tuple[List[str], dict]:
    """stock/check_pins.py's own check(), imported from the tree (the one place the pin check lives)."""
    path = os.path.join(tree_home(), CHECK_PINS_RELPATH)
    spec = importlib.util.spec_from_file_location("caliby_check_pins", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.check(pins()["upstream"])


# ----------------------------------------------------------------------------------------------------------------- environment
def kit_switches_in(env: Optional[dict] = None) -> Dict[str, str]:
    env = os.environ if env is None else env
    return {k: v for k, v in env.items() if k.startswith(KIT_SWITCH_PREFIXES)}


def dist_version(name: str) -> Optional[str]:
    """The installed version of distribution ``name`` from its metadata (no import), or None when it is not installed."""
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:                                                  # noqa: BLE001 — absent or unreadable metadata: no version to name
        return None


def strip_env(env: Optional[dict] = None, prefixes=STOCK_ABSENT_PREFIXES) -> Tuple[dict, List[str]]:
    """A copy of ``env`` without every variable starting with one of ``prefixes``; returns (env, removed names)."""
    env = dict(os.environ if env is None else env)
    removed = sorted(k for k in env if k.startswith(prefixes))
    for k in removed:
        env.pop(k)
    return env, removed


def stock_env_proof(env: Optional[dict] = None) -> dict:
    """What the stock arm asserts about its environment: no kit or package switch present, and the data-path names it read."""
    env = os.environ if env is None else env
    present = sorted(k for k in env if k.startswith(STOCK_ABSENT_PREFIXES))
    return {"must_be_absent_prefixes": list(STOCK_ABSENT_PREFIXES), "present": present, "clean": not present,
            "data_env": {k: env.get(k) for k in DATA_ENV if k in env}}


# ----------------------------------------------------------------------------------------------------------------- box gates
def gpu_info() -> dict:
    """The GPU as torch sees it (torch is imported here — called when activating or checking, never at interpreter start)."""
    try:
        import torch
    except Exception as e:                                             # noqa: BLE001
        return {"available": False, "name": None, "sm": None, "reason": f"torch import failed: {e}", "torch": None, "cuda": None}
    if not torch.cuda.is_available():
        return {"available": False, "name": None, "sm": None, "reason": "no CUDA device", "torch": torch.__version__, "cuda": torch.version.cuda}
    cc = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    return {"available": True, "name": torch.cuda.get_device_name(0), "sm": f"{cc[0]}.{cc[1]}", "memory_gb": round(props.total_memory / 2**30, 1),
            "torch": torch.__version__, "cuda": torch.version.cuda, "reason": None}


def stack_key(gpu: Optional[dict] = None) -> str:
    """torch<version sans local tag>-cu<CUDA sans dot>-sm<cc digits>, e.g. torch2.6.0-cu124-sm90 — the shared core's cache-key rule
    (``opt_core.jit_cache.key``) over the values torch reports in ``gpu_info()`` (the JIT cache key of configs/*.env);
    ``unknown:<reason>`` (never a bare ``unknown``) when torch itself is not introspectable (gpu_info()'s own reason, sanitised) — a
    caller must refuse on an ``unknown:`` prefix rather than use it as a cache key (configs/*.env's guard does)."""
    g = gpu or gpu_info()
    torch_v = g.get("torch")
    if not torch_v:
        reason = str(g.get("reason") or "no-torch")
        tag = "".join(c if (c.isalnum() or c in "_.-") else "_" for c in reason)[:40].strip("_") or "no-torch"
        return f"unknown:{tag}"
    from opt_core import jit_cache
    return jit_cache.key(version=torch_v, cuda=(g.get("cuda") or "cpu"), cc=(g.get("sm") or "none"))


def compiler() -> Optional[str]:
    """A C compiler on PATH (Triton JIT for CALIBY_X_LCP=1) or None."""
    return shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def weights_gate(variant: str, p: Optional[dict] = None, env: Optional[dict] = None, ckpt: Optional[str] = None) -> Tuple[Optional[str], dict]:
    """The weight files the variant reads under $MODEL_PARAMS_DIR (PINS.json weights.files: ``used_by`` names the variants; a sequence-design
    checkpoint carries ``ckpt`` and is required only when it is the run's checkpoint — ``ckpt``, default upstream's default checkpoint
    (PINS weights.weight_set = load_model's default, unless --model_name names another). Returns (refusal or None, detail)."""
    p = p or pins()
    env = os.environ if env is None else env
    ckpt = ckpt or p["weights"]["weight_set"]
    root = env.get("MODEL_PARAMS_DIR")
    detail = {"MODEL_PARAMS_DIR": root, "ckpt": ckpt, "files": {}}
    if not root:
        return "MODEL_PARAMS_DIR is not set (the weights root holding caliby/<ckpt>.ckpt and protpardelle-1c/; README.md \"Variables\")", detail
    missing = []
    for rel, info in p["weights"]["files"].items():
        if variant not in info.get("used_by", []):
            continue
        if info.get("ckpt") is not None and info["ckpt"] != ckpt:
            continue
        path = os.path.join(root, rel)
        ok = os.path.isfile(path)
        detail["files"][rel] = {"path": path, "present": ok, "size_bytes": os.path.getsize(path) if ok else None, "sha256_pinned": info.get("sha256")}
        if not ok:
            missing.append(path)
    if missing:
        return "weights missing: " + ", ".join(missing), detail
    return None, detail


def python_version() -> str:
    return platform.python_version()
