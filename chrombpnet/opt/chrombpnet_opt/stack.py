"""Activation: kit paths, the kit's own route resolution (fastdefault.py loaded by path), the GPU probe (the kit's detect_gpu), the
environment of each arm, and the report.

`activate(mode)` resolves the mode by NAME (modes.check_mode), gates (the kit present at opt/kit_ho with opt/kit beside it, its
route-critical files present) and reports. Nothing is applied inside this process on any route: the kit's form is a replacement entry
script (its documented line), so the report says where the mode engages (`applied`). `activate(mode, dry_run=True)` is `check`.

The route word. At activation (the CLI verb, the env route) it is the kit's route BY ITS OWN TABLES for this class and mode,
from the kit's own table functions only: arch_default_route(kit, arch_key(cap, cls), route_mode()) (fastdefault.py, torch/chrombpnet_k1/
arch_tiles.json), else class_route(cls, cap, mode) (the class table; a class in neither table with a device engages K1 with a cold JIT, said);
kit-internal names found in the caller's environment are removed from the kit's environment and named, never refused; a K1 default reads
`k1`, a stock-route default reads `tf_function` under the DET recipe and `keras_predict_fileorder` in prod (the kit's mapping in
fastdefault.resolve). No torch/triton probe runs in the wrapper (the kit's resolve() imports the K1 stack on a K1-default class — the
exec'd kit process pays that import itself); the kit's own status line states the forward it actually ran, and a HOLD/OFF there is the
kit's word. The dry run (`check`) is the exception: it runs the FULL resolve() with the probe — that is where a missing torch stack is
reported before any job. `exact` on a class whose tables give no K1 route is refused by name here (no bitwise Triton route on that arch).
Under --det the recipe's variables are overlaid on this process's environment for the decision so the kit reads the DET mode, then restored.

Kit integrity: `check` (the dry run) confirms every route-critical kit file is present (byte identity is the checked-out git commit's
job, not this package's); an activation counts only the files the route executes (ROUTE_MEMBERS) and says so (kit_integrity.scope).
"""
import importlib.util
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

from . import ActivationError, __version__
from . import det, modes, registry
from . import report as _report

ENV = modes.ENV
ENV_HOME = "CHROMBPNET_OPT_HOME"                                 # the tree root when the package is imported outside its tree (run.sh sets it from its own location)
ENV_CACHE_TAR = "CHROMBPNET_OPT_CACHE_TAR"                       # the config's driver-cache tarball (the one documented deviation, see cache_tar_env)
KIT_RECORD_ENV = registry.KIT_RECORD_NAME                        # the run-record handshake: the package names a file (cli.run_fast), the kit's entry script writes its run record there — one entry per item — and the package folds it into opt_manifest.json (none on a stock_cli exec)
ITEMS_REFUSED_STOCK_FLAGS = ("-r", "--regions", "-op", "--output-prefix", "-os", "--output-prefix-stats")   # `pred_bw --items`: every item names its own regions / prefix / stats
KIT_HO_RELPATH = os.path.join("opt", "kit_ho")
ENV_TARGET_GPU = "MODEL_OPT_TARGET_GPU"                          # the config's target; a note in the report, never a gate (run.sh gates)
KIT_RELPATH = os.path.join("opt", "kit")
KIT_VERSION_FILE = os.path.join("tf", "chrombpnet_fastkit", "__init__.py")   # __version__ = "..." (the kit's own stamp)
FASTDEFAULT_RELPATH = os.path.join("tf", "chrombpnet_fastkit", "fastdefault.py")
CACHE_RELDIR = "cache"
ROUTE_MEMBERS = ("tf/pred_bw_fast.py", "tf/chrombpnet_fastkit/*.py", "tf/det_subprocess/sitecustomize.py", "torch/chrombpnet_k1/*.py", "torch/chrombpnet_k1/*.json")
TORCH_MEMBERS = tuple(p for p in ROUTE_MEMBERS if p.startswith("torch/"))   # the torch-side members opt/kit_ho reads from opt/kit (kit_tables_root)
CACHE_TAR_FMT = "nv_compute_cache_{}.tar"                        # pred_bw_fast.py _JIT_DEFAULT: cache/nv_compute_cache_<class>.tar
KIT_JIT_ENV = "CHROMBPNET_JIT_CACHE_TAR"                         # pred_bw_fast.py _JIT_TAR
PACKAGE_ENV_PREFIX = "CHROMBPNET_OPT"                            # every package switch (the mode, det, home, cache tar, the config's data variables)
STOCK_FORBIDDEN_PREFIXES = (PACKAGE_ENV_PREFIX,)                 # the package's own namespace: the one legitimate prefix rule
STOCK_FORBIDDEN_NAMES = registry.KIT_SWITCH_NAMES                # the kit's own switch names, by name (registry.py: the grep of the kit's bytes)
STOCK_DET_EXCEPTIONS = registry.KIT_SWITCH_DET_NAMES             # the recipe's own names, allowed on the stock arm under --det only
KIT_SWITCH_NAMES = registry.KIT_SWITCH_NAMES                     # the one-mode table keys on these NAMES, never on a prefix
KIT_SWITCH_DET_NAMES = registry.KIT_SWITCH_DET_NAMES             # composed by the package under --det 1; found in the caller's environment otherwise they are removed and named like the rest
_STATE = {"report": None}                                         # type: Dict[str, Optional[dict]]


# ----------------------------------------------------------------------------------------------------------------------- paths
def opt_home() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tree_home() -> str:
    return os.environ.get(ENV_HOME) or os.path.dirname(opt_home())


def carried_kit(kit: str) -> str:
    """The carried kit beside `kit`: `opt/kit` — the torch-side package and the driver caches opt/kit_ho's entry script reads (pred_bw_fast.py
    _BASE_KIT) — when `kit` is another directory under opt/ and opt/kit carries torch/chrombpnet_k1, else `kit` itself."""
    base = os.path.join(os.path.dirname(os.path.abspath(kit)), "kit")
    return base if os.path.abspath(kit) != base and os.path.isdir(os.path.join(base, "torch", "chrombpnet_k1")) else kit


def kit_tables_root(kit: str) -> str:
    """The directory whose torch/ side carries the route tables and the K1 package for `kit`: `kit` itself when it has torch/chrombpnet_k1,
        else opt/kit beside it (opt/kit_ho carries no torch/ — its entry script reads opt/kit's, pred_bw_fast.py _BASE_KIT)."""
    if os.path.isdir(os.path.join(kit, "torch", "chrombpnet_k1")):
        return kit
    base = os.path.join(os.path.dirname(os.path.abspath(kit)), "kit")
    return base if os.path.isdir(os.path.join(base, "torch", "chrombpnet_k1")) else kit


def multi_items(path: str) -> int:
    """The item count of an items file (`pred_bw --items`) (`regions<TAB>output_prefix[<TAB>stats]` per line; # = comment); raises on a malformed line."""
    n = 0
    with open(path, "r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            s = line.rstrip("\n")
            if not s.strip() or s.lstrip().startswith("#"):
                continue
            cols = s.split("\t")
            if len(cols) < 2 or not cols[0] or not cols[1]:
                raise ActivationError("{}:{} needs `regions<TAB>output_prefix[<TAB>stats]`".format(path, ln))
            n += 1
    if n == 0:
        raise ActivationError("{}: no items".format(path))
    return n


def multi_prefixes(path: str) -> List[str]:
    """The output prefixes of an items file, in file order (the grammar of multi_items)."""
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.rstrip("\n")
            if not s.strip() or s.lstrip().startswith("#"):
                continue
            cols = s.split("\t")
            if len(cols) >= 2 and cols[1]:
                out.append(cols[1])
    return out


def read_record(path: Optional[str]) -> Optional[dict]:
    """The kit's run record as its entry script wrote it to `path` ({"kit", "items": [one entry per finished item]}), or None when absent / unreadable."""
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else None
    except (OSError, ValueError):
        return None


def has_cm_model(args) -> bool:
    """True when the stock arguments name a -cm / --chrombpnet-model (the model K1 composes); -cmb / -bm alone run stock's own graph."""
    return any(a in ("-cm", "--chrombpnet-model") or str(a).startswith("--chrombpnet-model=") for a in (args or []))


def records_by_prefix(doc: Optional[dict], prefixes) -> Dict[str, Optional[dict]]:
    """{prefix: the run record's entry for that output prefix, None when the kit recorded none} over the expected prefixes."""
    by = {str(it.get("prefix")): it for it in ((doc or {}).get("items") or []) if isinstance(it, dict)}
    return {p: by.get(str(p)) for p in prefixes}


def applied(rep: dict, stamps: Dict[str, Optional[dict]]) -> dict:
    """The applied set of a fast run, judged after it from the kit's run record (`stamps`: output prefix -> that item's record entry,
    None when the kit recorded none; records_by_prefix): the forward the kit ran (`import_witness.forward` / `fastdefault.forward`) against
    the route the kit's tables give for this machine (`rep["route"]`); the kit's `fastdefault.skipped` entries — its declared rules (a class whose lever set
    excludes K1, native dilation off under the recipe) when the route stands, the fallback's reasons when it does not; the shipped driver cache's untar fallback
    (`jit_cache.fallback`, the kit's own tarball or the config's, cache_tar_env). A lever of the class's set that cannot start makes
    the kit refuse the mode by name before any of this (the record's `refused`, handled by cli.run_fast); a forward other than the route
    here is the backstop for the same rule (partial -> NOT ACTIVE, exit 3). No entry at all = the kit's record of the run is absent
    (partial); an item without an entry beside items that have one did not complete (incomplete).
    Returns `levers_applied`, `partial` (registry names), `partial_reasons`, `gated`, `evidence` (per prefix), `partial_detection`, and
    `incomplete` ("<ok>/<n>": items whose entry lists outputs all on disk, against the prefixes) — outputs short of the request are the
    run's failure, never a partial activation."""
    route = rep.get("route")
    evidence: Dict[str, str] = {}
    partial: List[str] = []
    reasons: List[str] = []
    gated: List[str] = []
    n_ok = 0
    any_entry = any(st is not None for st in stamps.values())

    def fell(name: str, why: str) -> None:
        if name not in partial:
            partial.append(name)
        reasons.append("{}: {}".format(name, why))

    for prefix, st in stamps.items():
        tag = os.path.basename(prefix) or prefix
        if st is None:
            if any_entry:
                reasons.append("outputs: {}: no run-record entry — the item did not complete".format(tag))
            else:
                fell("forward_route", "{}: no run record: the kit's record of the run is absent".format(tag))
            continue
        else:
            fd = st.get("fastdefault") if isinstance(st.get("fastdefault"), dict) else {}
            fwd = (st.get("import_witness") or {}).get("forward") or fd.get("forward")
            skipped = [str(s) for s in (fd.get("skipped") or [])]
            if fwd != route:
                fell("forward_route", "{}: the kit ran forward={}, the tables' route is {}{}".format(tag, fwd, route, (": " + "; ".join(skipped)) if skipped else (" ({})".format(fd.get("forward_reason")) if fd.get("forward_reason") else "")))
            else:
                evidence[tag] = "forward={}".format(fwd)
                for s in skipped:
                    g = "forward_route: {} (the kit's declared rule; the tables' route stands)".format(s)
                    if g not in gated:
                        gated.append(g)
        jc = st.get("jit_cache") if isinstance(st.get("jit_cache"), dict) else {}
        if jc.get("fallback"):
            fell("jit_cache", "{}: the shipped driver cache fell back — {}".format(tag, jc["fallback"]))
        elif jc:
            evidence[tag] = evidence.get(tag, "") + "; jit_cache untar recorded ({})".format(jc.get("tarball") or jc.get("untar_s"))
        outputs = [str(o) for o in (st.get("outputs") or [])]
        missing = [o for o in outputs if not os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(prefix)), o))]
        if missing:
            reasons.append("outputs: {}: {} of the {} outputs the kit listed are absent ({})".format(tag, len(missing), len(outputs), ", ".join(missing[:4])))
        else:
            n_ok += 1
    n = len(stamps)
    incomplete = "{}/{}".format(n_ok, n) if n and n_ok < n and any_entry else None
    levers = []
    if "forward_route" not in partial and evidence:
        levers.append("forward_route")
    if "jit_cache" not in partial and any(isinstance(st, dict) and st.get("jit_cache") for st in stamps.values()):
        levers.append("jit_cache")
    return {"levers_applied": levers, "partial": partial, "partial_reasons": reasons, "gated": gated, "evidence": evidence,
            "incomplete": incomplete, "partial_detection": "the kit's run record ({} of {} item entries present)".format(sum(1 for st in stamps.values() if st is not None), n)}


def multi_line(kit: str, args, items_path: str, python: str = None) -> list:
    """The `--items` form's line: the kit's entry script with its --items loop —
    `python opt/kit_ho/tf/pred_bw_fast.py --items <file> <the stock arguments minus -r/-op/-os>` (one source: the same script as the one-item line)."""
    return [python or "python", modes.kit_line_script(kit), "--items", items_path] + list(args or [])


def kit_home() -> str:
    """The kit directory of the fast mode: `<tree>/opt/kit_ho` (the tf/ side — the entry script, the chrombpnet_fastkit package, det_subprocess,
    the kit's CPU tests), with the carried kit `<tree>/opt/kit` beside it (the torch-side package torch/chrombpnet_k1
    and the driver caches); the tree from CHROMBPNET_OPT_HOME or the package's own location. Raises when the kit's entry script is not there,
    or when opt/kit's torch side is missing beside it."""
    k = os.path.join(tree_home(), KIT_HO_RELPATH)
    carried_torch = os.path.join(tree_home(), KIT_RELPATH, "torch", "chrombpnet_k1")
    if not os.path.isdir(carried_torch):
        raise ActivationError("kit not found: the carried kit's torch side {} is missing beside {} (set {})".format(carried_torch, KIT_HO_RELPATH, ENV_HOME))
    if not os.path.isfile(modes.kit_line_script(k)):
        raise ActivationError("kit not found: {} is missing (set {})".format(modes.kit_line_script(k), ENV_HOME))
    return k


def kit_version(kit: Optional[str] = None) -> Optional[str]:
    """The kit's __version__ read from its own file (no kit import: the kit package imports numpy at its import)."""
    import re
    p = os.path.join(kit or kit_home(), KIT_VERSION_FILE)
    try:
        with open(p, "r", encoding="utf-8") as fh:
            m = re.search(r"^__version__\s*=\s*[\"\']([^\"\']+)[\"\']", fh.read(), re.M)
        return m.group(1) if m else None
    except OSError:
        return None


def fastdefault(kit: Optional[str] = None):
    """The kit's fastdefault module loaded by path: no TensorFlow / torch at its import, and no kit package import (chrombpnet_fastkit's
        __init__ imports numpy)."""
    p = os.path.join(kit or kit_home(), FASTDEFAULT_RELPATH)
    spec = importlib.util.spec_from_file_location("_chrombpnet_opt_fastdefault", p)
    mod = importlib.util.module_from_spec(spec)
    saved = sys.dont_write_bytecode
    sys.dont_write_bytecode = True                                     # no .pyc inside the kit directory
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = saved
    return mod


def kit_switches_set(environ: Optional[dict] = None, det_on: bool = False) -> list:
    """The kit's own switch names present in the environment (sorted), by NAME from registry.KIT_SWITCH_NAMES — never by prefix (a caller's
    bookkeeping variables may share the kit's prefixes). A mode is the whole composition: these names are removed from the kit's environment
    (kit_env) and named on one line (report.ignored_line), never a refusal. The recipe's names count too unless the package composes them itself (--det 1)."""
    environ = os.environ if environ is None else environ
    return sorted(k for k in environ if k in KIT_SWITCH_NAMES and not (det_on and k in KIT_SWITCH_DET_NAMES))


def kit_switch_note(names) -> str:
    return "kit-internal name{} {} set in the environment: removed for the run (the mode is the whole composition)".format("s" if len(names) != 1 else "", ",".join(names))


# ----------------------------------------------------------------------------------------------------------------------- probes
def gpu_probe(fd=None) -> dict:
    """The kit's own class detection (fastdefault.detect_gpu: nvidia-smi name -> class in H100|H200|A100|L40S|B200|unknown)."""
    try:
        fd = fd or fastdefault()
        cls, name, cap = fd.detect_gpu()
        q = fd.nvsmi_query() if hasattr(fd, "nvsmi_query") else {}
        available = not (isinstance(q, dict) and "error" in q) and cap is not None
        return {"class": cls, "name": name, "compute_cap": cap, "available": bool(available), "driver": (q or {}).get("driver")}
    except Exception as e:  # noqa: BLE001
        return {"class": "unknown", "name": "probe failed ({}: {})".format(type(e).__name__, e), "compute_cap": None, "available": False, "driver": None}


def target_gpu_note(gpu: dict) -> Optional[str]:
    target = os.environ.get(ENV_TARGET_GPU) or None
    name = (gpu or {}).get("name") or ""
    if target and (gpu or {}).get("available") and target.lower() not in name.lower():
        return "{}={} but the GPU is {}: not the GPU this configuration targets".format(ENV_TARGET_GPU, target, name)
    return None


def kit_integrity(kit: Optional[str] = None, scope: str = "full") -> dict:
    """The carried kit files' presence — {"scope", "n_present", "missing"}. Byte identity is the checked-out git commit's job, not
    this function's: it only confirms the files the port depends on are actually on disk. Every LITERAL (non-wildcard) member of
    ROUTE_MEMBERS, and of TORCH_MEMBERS when kit_tables_root(kit) is a different directory, is required and named in `missing` if
    absent — in BOTH scopes (a required file is required whether the check is narrow or thorough). A wildcard member (e.g.
    `tf/chrombpnet_fastkit/*.py`) names a directory's role, not a fixed file count, so it never appears in `missing`. `n_present`
    differs by scope: "route" counts glob matches over ROUTE_MEMBERS/TORCH_MEMBERS only (what an activation touches); "full" is a
    plain recursive count of every file under `kit` (bytecode caches excluded) — broader, for `check`."""
    import glob
    if scope not in ("full", "route"):
        raise ValueError("scope: full | route")
    kit = kit or kit_home()
    tables = kit_tables_root(kit)
    missing = []

    def literal_missing(root, pats):
        base = os.path.basename(os.path.abspath(root))
        for pat in pats:
            if any(c in pat for c in "*?[") or glob.glob(os.path.join(root, pat)):
                continue
            missing.append("{}/{}".format(base, pat))

    literal_missing(kit, ROUTE_MEMBERS)
    if tables != kit:
        literal_missing(tables, TORCH_MEMBERS)

    if scope == "route":
        present = sum(len(glob.glob(os.path.join(kit, pat))) for pat in ROUTE_MEMBERS)
        if tables != kit:
            present += sum(len(glob.glob(os.path.join(tables, pat))) for pat in TORCH_MEMBERS)
        return {"scope": scope, "n_present": present, "missing": missing}

    n = 0
    for dp, dns, fns in os.walk(kit):
        dns[:] = [d for d in dns if d != "__pycache__"]
        n += sum(1 for fn in fns if not fn.endswith(".pyc"))
    return {"scope": scope, "n_present": n, "missing": missing}


# ------------------------------------------------------------------------------------------------------------------- environments
def strip_env(environ: dict, names=(), prefixes=(), keep=()) -> Tuple[dict, list]:
    out, stripped = {}, []
    for k, v in environ.items():
        if k in keep:
            out[k] = v
        elif k in names or k.startswith(tuple(prefixes)):
            stripped.append(k)
        else:
            out[k] = v
    return out, sorted(stripped)


def _under(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([os.path.realpath(path), os.path.realpath(root)]) == os.path.realpath(root)
    except ValueError:
        return False


def stock_env(environ: dict, det_on: bool, kit: Optional[str] = None, tree: Optional[str] = None) -> Tuple[dict, list, dict]:
    """The stock arm's environment: all kit and package variables removed (STOCK_FORBIDDEN_*), the PYTHONPATH entries under opt/kit
    and under the tree removed, then the recipe composed on top under --det (its variable and the seed-hook directory are the allowed
    exceptions). Returns (env, names stripped, the launching process's proof from names: forbidden names still present — asserted empty — and
    the PYTHONPATH entries dropped); the child proves its own state again (stock_pred_bw.py)."""
    env, stripped = strip_env(environ, names=STOCK_FORBIDDEN_NAMES, prefixes=STOCK_FORBIDDEN_PREFIXES)
    kit = kit or (kit_home() if det_on else None)
    tree = tree or tree_home()
    roots = [r for r in (kit, tree) if r]
    dropped = [p for p in (env.get("PYTHONPATH") or "").split(os.pathsep) if p and any(_under(p, r) for r in roots)]
    kept = [p for p in (env.get("PYTHONPATH") or "").split(os.pathsep) if p and p not in dropped]
    if kept:
        env["PYTHONPATH"] = os.pathsep.join(kept)
    else:
        env.pop("PYTHONPATH", None)
    if det_on:
        env = det.env(kit, env, route=None)
    proof = env_proof(env, det_on)
    proof["pythonpath_dropped"] = dropped
    return env, stripped, proof


def env_proof(env: dict, det_on: bool) -> dict:
    forbidden = sorted(k for k in env if (k in STOCK_FORBIDDEN_NAMES or k.startswith(STOCK_FORBIDDEN_PREFIXES)) and not (det_on and k in STOCK_DET_EXCEPTIONS))
    return {"forbidden_prefixes": list(STOCK_FORBIDDEN_PREFIXES), "forbidden_names": list(STOCK_FORBIDDEN_NAMES), "forbidden_present": forbidden,
            "det_present": sorted(k for k in det.NAMES if k in env) if det_on else [], "ok": not forbidden}


def kit_env(environ: dict, det_on: bool, kit: str, route: Optional[str], gpu_class: Optional[str]) -> Tuple[dict, list, dict]:
    """The kit line's environment: the caller's minus the package's own switches (CHROMBPNET_OPT*: the mode is expressed once, by the
    line itself, and the kit process imports `chrombpnet` — the finder must not fire there) and minus the kit's own internal names
    (KIT_SWITCH_NAMES: a mode is the whole composition; a caller-set one is removed here and named by activate(), never refused); the
    recipe composed under --det 1 (its own names re-set by det.env); the driver-cache deviation (cache_tar_env). `stripped` lists both."""
    env, stripped = strip_env(environ, names=KIT_SWITCH_NAMES, prefixes=(PACKAGE_ENV_PREFIX,))
    if det_on:
        env = det.env(kit, env, route=route)
    cache = cache_tar_env(environ, kit, gpu_class)
    if cache.get("set"):
        env[KIT_JIT_ENV] = cache["path"]
    return env, stripped, cache


def cache_tar_env(environ: dict, kit: str, gpu_class: Optional[str]) -> dict:
    """THE ONE DOCUMENTED DEVIATION. The kit reads CHROMBPNET_JIT_CACHE_TAR first (pred_bw_fast.py _JIT_TAR) and falls back
    to its own cache/nv_compute_cache_<class>.tar. The tree does not carry the tarball (its size), so the config names it
    (CHROMBPNET_OPT_CACHE_TAR) and the package hands it to the kit by the kit's own variable ONLY when the kit's own tarball for this
    class is absent — the kit's own copy keeps precedence wherever it is present (a caller-set CHROMBPNET_JIT_CACHE_TAR is a kit-internal
    name: kit_env removed it before this composition; this is the one exception where the package sets a kit variable)."""
    out = {"set": False, "path": None, "source": None, "kit_tar": None, "kit_tar_present": None, "config": environ.get(ENV_CACHE_TAR) or None}
    if gpu_class and gpu_class != "unknown":
        out["kit_tar"] = os.path.join(carried_kit(kit), CACHE_RELDIR, CACHE_TAR_FMT.format(gpu_class))   # the carried kit's cache/ (opt/kit_ho reads it there, _BASE_KIT)
        out["kit_tar_present"] = os.path.isfile(out["kit_tar"])
    if out["kit_tar_present"]:
        out.update(path=out["kit_tar"], source="kit (HERE)"); return out
    if out["kit_tar"] is None:
        out["source"] = "none (GPU class unknown: the config's tarball is per class)"; return out
    cfg = out["config"]
    if cfg and os.path.isfile(cfg):
        out.update(set=True, path=cfg, source="config ({} -> {})".format(ENV_CACHE_TAR, KIT_JIT_ENV)); return out
    out["source"] = "none" + (" (config path absent: {})".format(cfg) if cfg else "")
    return out


# ---------------------------------------------------------------------------------------------------------------------- resolve
class _Environ(object):
    """Temporarily overlay variables on os.environ (the kit's route_mode reads os.environ)."""

    def __init__(self, overlay: dict):
        self.overlay, self.saved = overlay, {}

    def __enter__(self):
        for k, v in self.overlay.items():
            self.saved[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)                       # a kit-internal name the caller set: masked while the tables are read
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *a):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _overlay(kit: str, det_on: bool) -> dict:
    masked = {k: None for k in kit_switches_set(det_on=det_on)}          # the caller's kit-internal names never reach the tables (kit_env removes them for the run)
    return dict(masked, **{k: v for k, v in det.env(kit, {}, route=None).items() if k != "PYTHONPATH"}) if det_on else masked


def table_route(kit: str, det_on: bool, fd=None) -> dict:
    """The kit's route BY ITS OWN TABLES for this machine (no probe, no torch import): {"route", "reason", "mode_word", "gpu",
    "source"}: arch_default_route() for a listed arch, else class_route() (the class table; a class in neither table with a device engages
    K1), mapped to the forward word as the kit maps it (fastdefault.resolve). The caller's kit-internal names are masked (kit_env removes them)."""
    fd = fd or fastdefault(kit)
    with _Environ(_overlay(kit, det_on)):
        gpu = gpu_probe(fd)
        mode = fd.route_mode()
        cls, cap = gpu.get("class"), gpu.get("compute_cap")
        ar = fd.arch_default_route(kit_tables_root(kit), fd.arch_key(cap, cls), mode)
        if ar is not None:
            k1_default, basis, source = (ar[0] == "k1"), ar[1], ar[2]
        else:
            k1_default, basis = fd.class_route(cls, cap, mode); k1_default, source = bool(k1_default), "class_route"
    route = "k1" if k1_default else ("tf_function" if mode == "det" else "keras_predict_fileorder")
    return {"route": route, "reason": "[{}/{}] route by the kit's table: {}".format(cls, mode, basis), "mode_word": mode, "gpu": gpu, "source": source}


def resolve_route(kit: str, det_on: bool, fd=None) -> dict:
    """The kit's FULL decision for the documented line on this machine, probe included (`check`): {"route", "reason", "mode_word",
    "kit_resolution", "gpu", "source"} from resolve()["forward"] (fastdefault.py); the caller's kit-internal names masked."""
    fd = fd or fastdefault(kit)
    with _Environ(_overlay(kit, det_on)):
        gpu = gpu_probe(fd)
        res = fd.resolve(kit_root=kit_tables_root(kit))
    slim = {k: v for k, v in res.items() if k not in ("torch_stack",)}
    slim["torch_stack"] = {k: v for k, v in (res.get("torch_stack") or {}).items() if k in ("present", "source", "missing", "torch", "triton", "import_error", "probe", "skipped")}
    return {"route": res.get("forward"), "reason": res.get("forward_reason"), "mode_word": res.get("mode"), "kit_resolution": slim, "gpu": gpu, "source": "resolve (probe included)"}


# --------------------------------------------------------------------------------------------------------------------- activate
def status() -> dict:
    return _STATE["report"] or {"active": False, "reason": "enable() has not run in this process"}


def _refuse(rep: dict, reason: str, strict: bool, quiet: bool = False) -> dict:
    rep.update(active=False, reason=reason)
    if not quiet:
        _report.print_mode_line(rep)
    if strict:
        raise ActivationError(reason)
    return rep


def activate(mode: str, det: bool = False, *, dry_run: bool = False, strict: bool = False, trigger: Optional[str] = None,
             route_label: Optional[str] = None, quiet: bool = False, args=None, items: Optional[str] = None) -> dict:
    """The report for `mode` (the mode line printed unless `quiet`, first on stdout). `dry_run` = check. `args` (the stock pred_bw
    arguments) fills the line the arm would run; `items` (`pred_bw --items <file>`) makes it the many-items line."""
    from . import det as _det
    m0 = (mode or "").strip().lower()
    det_on = (m0 in modes.DET_MODES) if (m0 in modes.MODES and m0 != "off") else (bool(det) or _det.requested())   # a kit mode carries its numerics: exact = the recipe, fast = shipped; off: --det / CHROMBPNET_OPT_DET
    rep = {"active": False, "mode": (mode or "").strip().lower(), "dry_run": bool(dry_run), "route_label": route_label or ("env" if trigger else "enable"),
           "trigger": trigger, "package_version": __version__, "det": det_on, "route": None, "kit_version": None, "gpu": None, "notes": [],
           "levers": [], "applied": None}   # type: Dict[str, object]
    try:
        m = modes.check_mode(mode)
    except ValueError as e:
        return _refuse(rep, str(e), strict, quiet)
    rep["mode"] = m
    prev = _STATE["report"]
    if not dry_run and prev is not None and rep["route_label"] == "enable":
        if prev.get("mode") == m:
            return prev
        return _refuse(rep, "already activated in mode {!r} in this process; a different mode is refused".format(prev.get("mode")), strict, quiet)
    if m == "off":
        if items:
            return _refuse(rep, "--items with mode off: the stock CLI predicts one regions file per process — refused", strict, quiet)
        rep.update(route="stock_cli", reason=None, applied="none: the stock console script in a clean subprocess (chrombpnet-opt pred_bw --mode off)",
                   line=[modes.STOCK_CONSOLE_SCRIPT, modes.SUBCOMMAND] + list(args or []))
        try:
            rep["gpu"] = gpu_probe(fastdefault())
        except Exception:  # noqa: BLE001
            rep["gpu"] = None
        if not dry_run and rep["route_label"] == "enable":
            _STATE["report"] = rep
        if not quiet:
            _report.print_mode_line(rep)
        return rep
    # ---- exact | fast: the kit's documented line (the mode carries the numerics)
    if m not in modes.DET_MODES and _det.requested():
        return _refuse(rep, "{}=1 with mode {}: `{} --det 1` does not exist — `--mode exact` is the kit under upstream's determinism settings, `--mode off --det 1` the stock reference it equals".format(modes.ENV_DET, m, m), strict, quiet)
    try:
        kit = kit_home()
    except ActivationError as e:
        return _refuse(rep, str(e), strict, quiet)
    rep.update(kit=kit, kit_version=kit_version(kit), multi=None)
    if items:
        items_path = items
        if not os.path.isfile(items_path):
            return _refuse(rep, "--items {}: not a file — refused".format(items_path), strict, quiet)
        bad = [a for a in (args or []) if a in ITEMS_REFUSED_STOCK_FLAGS or a.split("=", 1)[0] in ITEMS_REFUSED_STOCK_FLAGS]
        if bad:
            return _refuse(rep, "--items with {} on the command line: every item names its own regions / prefix / stats in the items file — refused".format(" ".join(bad)), strict, quiet)
        try:
            rep["multi"] = multi_items(items_path)
        except ActivationError as e:
            return _refuse(rep, "--items: {} — refused".format(e), strict, quiet)
        rep["multi_items_file"] = os.path.abspath(items_path)
    switches = kit_switches_set(os.environ, det_on)
    if switches:                                                   # the caller's kit-internal names: masked for the tables (_overlay), removed for the run (kit_env), named — never a refusal
        rep["ignored"] = switches
        rep.setdefault("notes", []).append(kit_switch_note(switches))
    rep["documented_form"] = " ".join(modes.documented_line(os.path.relpath(kit, tree_home()), ["<the stock pred_bw arguments>"]))
    integ = kit_integrity(kit, scope="full" if dry_run else "route")
    rep["kit_integrity"] = integ
    if integ["missing"]:
        return _refuse(rep, "kit files are missing: {}".format(", ".join(integ["missing"][:5])), strict, quiet)
    try:
        fd = fastdefault(kit)
        r = resolve_route(kit, det_on, fd) if dry_run else table_route(kit, det_on, fd)
    except Exception as e:  # noqa: BLE001
        return _refuse(rep, "the kit's route resolution failed: {}: {}".format(type(e).__name__, e), strict, quiet)
    rep.update(route=r["route"], route_reason=r["reason"], route_source=r["source"], kit_resolution=r.get("kit_resolution"), gpu=r["gpu"], mode_word=r["mode_word"])
    if det_on:                                     # exact = the fp32 Triton kernels in stock's summation order, bit for bit: a class whose arch has no bitwise Triton route under the
        t = r if not dry_run else table_route(kit, det_on, fd)   # recipe (sm_80: its counts-head order varies with the rows per call; B200 / no CUDA device: the Triton forward cannot run) has no
        if t["route"] != "k1":                     # exact mode — refused by name before anything runs (the tables' verdict for the class, not the probe's)
            g = t.get("gpu") or {}
            return _refuse(rep, "the kit refused by name: no bitwise Triton route on {} (gpu={}) — exact is the fp32 Triton kernels in stock's summation order; "
                                "--mode fast is the kit's mode on this card, --mode off --det 1 the deterministic stock run".format(fd.arch_key(g.get("compute_cap"), g.get("class")) or "this box", _report.gpu_word(g)), strict, quiet)
    if rep["route"] == "k1" and args is not None and not has_cm_model(args):      # K1 composes the -cm model; a line with only -cmb / -bm runs stock's own graph (pred_bw_fast.py _TF_FORWARD)
        rep.update(route="tf_function", route_reason="no -cm model on the line: K1 composes the -cm model only; -cmb / -bm run stock's own graph (the TensorFlow route) | " + str(r["reason"]))
    note = target_gpu_note(rep["gpu"])
    if note:
        rep["notes"].append(note)
    rep["levers"] = list(registry.LEVERS)
    rep["line"] = multi_line(kit, args or [], rep["multi_items_file"], python=sys.executable) if rep.get("multi") else modes.documented_line(kit, args or [], python=sys.executable)
    rep["cache_tar"] = cache_tar_env(os.environ, kit, (rep["gpu"] or {}).get("class"))
    rep["applied"] = ("none in this process: the kit's form is a replacement entry script; the mode engages on the process boundary "
                      "(chrombpnet-opt pred_bw --mode fast, or CHROMBPNET_OPT=fast chrombpnet pred_bw)")
    rep.update(active=True, reason=None)
    if not dry_run and rep["route_label"] == "enable":
        _STATE["report"] = rep
    if not quiet:
        _report.print_mode_line(rep)
    return rep
