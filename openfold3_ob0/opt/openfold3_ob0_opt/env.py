"""The stock environment: what a stock process must not carry, how it is stripped, and how a process proves it.

* ``stock_environment()`` — stock/PINS.json "stock_environment", the ONE list: ``must_be_absent()`` = its name prefixes a stock process carries
  none of (the add-ons' switch families, this package's own variables, the deterministic-recipe workspace
  variable — exported by the recipe itself, under the det carve-out —, CUDA MPS and the DS4Sci CUTLASS override); ``upstream_reads()`` = the
  names UNDER those prefixes that upstream OpenFold3 itself reads (its Triton kernel switches): passed to the stock child unchanged and recorded,
  never stripped, never a violation; ``reads()`` = the variables the stock caller reads, recorded in the proof (``reads_present``). Nothing in
  this module restates the list; tests/test_stock_environment.py locks it against the mode table and the lever registry.
* ``strip()`` removes every must-be-absent variable and every add-on directory on PYTHONPATH from an environment copy (opt_core.stock_proof) —
  the stock subprocess starts from it; ``forbidden_present()`` lists what is present.
* ``kit_dirs()`` — every hook-carrying directory of the tree; ``kit_modules_loaded()`` / ``kit_dirs_on_path()`` / ``kit_hooks_installed()`` —
  the scans (kit finders on ``sys.meta_path`` by the file that defines them: under a hook directory of this tree).
* ``env_proof()`` — the dict the stock caller hands to its parent (``stock_pred --proof-json``) before it imports openfold3 (opt_core.stock_proof.env_proof
  plus this tree's words: kit hooks, openfold3 already loaded, core modules beyond the proof machinery — ``DET_CORE_MODULES`` allowed under the
  det carve-out only —, the reads present, the OpenFold cache directory and whether it holds a user ``runner.yml``).
* ``check_pins()`` / ``pins()`` — stock/check_pins.py as a module and stock/PINS.json (the upstream pin, the wheel, the pinned weights).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

from opt_core import stock_proof

from . import modes

STOCK_ENVIRONMENT_KEYS = ("must_be_absent_prefixes", "reads", "upstream_reads")   # stock/PINS.json "stock_environment": the ONE list (must_be_absent, reads, upstream_reads read it; nothing here restates it)


def stock_environment(home: Optional[str] = None) -> dict:
    """stock/PINS.json "stock_environment", the one list of the stock route: `must_be_absent_prefixes` (the name prefixes a stock process carries
    none of), `reads` (the names the stock caller reads, recorded in the proof), `upstream_reads` (`{NAME: "upstream file:line"}`: the names UNDER
    those prefixes that upstream OpenFold3 itself reads — carved out of the refusal, passed to the stock child unchanged and recorded). Refused by
    name when the block or one of its keys is absent."""
    block = pins(home).get("stock_environment")
    if not isinstance(block, dict) or any(k not in block for k in STOCK_ENVIRONMENT_KEYS):
        raise ValueError(f"stock/PINS.json \"stock_environment\" must carry {STOCK_ENVIRONMENT_KEYS} (got {sorted(block) if isinstance(block, dict) else block!r})")
    return block


def must_be_absent(home: Optional[str] = None) -> List[str]:
    """The must-be-absent name prefixes (stock/PINS.json stock_environment.must_be_absent_prefixes), in file order."""
    return [str(p) for p in stock_environment(home)["must_be_absent_prefixes"]]


def upstream_reads(home: Optional[str] = None) -> Dict[str, str]:
    """`{NAME: "upstream file:line"}` — upstream's own switches under the add-ons' prefixes (stock_environment.upstream_reads)."""
    return {str(k): str(v) for k, v in dict(stock_environment(home)["upstream_reads"]).items()}


def reads(home: Optional[str] = None) -> List[str]:
    """The names the stock caller reads (stock_environment.reads), recorded in the proof when present."""
    return [str(r) for r in stock_environment(home)["reads"]]


DET_CORE_MODULES: tuple = ("opt_core.det", "opt_core.gates", "opt_core.precision", "opt_core.precision.recipe", "opt_core.precision.policy")   # the core modules the det site loads to apply the recipe's statements (opt_core.precision.recipe.apply_torch reads the numerics signature through precision.policy, which imports gates): allowed in the stock child under the det carve-out, a violation otherwise
USER_RUNNER_YML = "runner.yml"                                          # upstream merges <OPENFOLD_CACHE>/runner.yml under any --runner-yaml (run_openfold.py:209-228): recorded, and named on stderr when present


def tree_home(environ: Optional[dict] = None) -> str:
    """The openfold3_ob0/ directory: OPENFOLD3_OB0_OPT_HOME, else MODEL_OPT, else the package's parent tree (opt/openfold3_ob0_opt -> openfold3_ob0/) — opt_core.home.tree_home."""
    from opt_core.home import tree_home as _tree_home
    return _tree_home(__file__, env_tree="MODEL_OPT", env_home="OPENFOLD3_OB0_OPT_HOME", levels=2, environ=environ)


_CHECK_PINS: dict = {}


def check_pins(home: Optional[str] = None):
    """stock/check_pins.py as a module (the pin checker run.sh runs as a script): its readers `read_pins` and `sha256_file` are the tree's
    one copy of each; loaded once per home, from the file, without a package import."""
    home = home or tree_home()
    if home not in _CHECK_PINS:
        import importlib.util
        path = os.path.join(home, "stock", "check_pins.py")
        spec = importlib.util.spec_from_file_location("openfold3_ob0_stock_check_pins", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CHECK_PINS[home] = mod
    return _CHECK_PINS[home]


def pins(home: Optional[str] = None) -> dict:
    home = home or tree_home()
    return check_pins(home).read_pins(os.path.join(home, "stock", "PINS.json"))


def _without_upstream(environ, home: Optional[str] = None) -> Dict[str, str]:
    ups = upstream_reads(home)
    return {k: v for k, v in environ.items() if k not in ups}


def forbidden_present(prefixes: List[str], environ: Optional[dict] = None, home: Optional[str] = None) -> List[str]:
    """The names present under `prefixes` — upstream's own reads (stock_environment.upstream_reads) excepted."""
    return stock_proof.forbidden(_without_upstream(os.environ if environ is None else environ, home), prefixes)


def strip(prefixes: List[str], environ: Optional[dict] = None, home: Optional[str] = None) -> Dict[str, str]:
    """A copy of `environ` without the must-be-absent names (upstream's own reads kept) and without any kit directory on PYTHONPATH (opt_core.stock_proof)."""
    environ = os.environ if environ is None else environ
    out, _ = stock_proof.strip_env(_without_upstream(environ, home), prefixes)
    out.update({k: environ[k] for k in upstream_reads(home) if k in environ})   # upstream's own switches pass through unchanged
    pp, _ = stock_proof.strip_pythonpath(out.get("PYTHONPATH"), [os.path.abspath(d) for d in kit_dirs(home)])
    if pp:
        out["PYTHONPATH"] = pp
    else:
        out.pop("PYTHONPATH", None)
    return out


def kit_dirs(home: Optional[str] = None, det: int = 0) -> List[str]:
    """Every hook-carrying directory of the tree: the carried add-ons and ports and the package's own hooks (the two tables in modes); under
    the det recipe (det >= 1) also the det site (det.DET_SITE) — the one such directory the recipe's carve-out then allows on the child's path."""
    home = home or tree_home()
    dirs = [os.path.join(home, rel) for rel in list(modes.KITS.values()) + list(modes.PACKAGE_HOOKS.values())]
    if int(det or 0) >= 1:
        from . import det as _det
        dirs.append(_det.det_site(home))
    return dirs


def _under(path: str, dirs: List[str]) -> bool:
    """The core's ONE path-containment predicate (opt_core.stock_proof._under: realpath on both sides)."""
    return stock_proof._under(path, [os.path.realpath(d) for d in dirs if d])


def kit_modules_loaded(dirs: Optional[List[str]] = None) -> List[str]:
    """The loaded modules whose file lives under a kit directory (opt_core.stock_proof.kit_modules_loaded, by file)."""
    return sorted(stock_proof.kit_modules_loaded([os.path.abspath(d) for d in (dirs or kit_dirs())], ()))


def kit_dirs_on_path(dirs: Optional[List[str]] = None) -> List[str]:
    dirs = [os.path.abspath(d) for d in (dirs or kit_dirs())]
    return [p for p in sys.path if p and _under(p, dirs)]


def finder_file(f) -> Optional[str]:
    """The file that DEFINES a meta-path finder's class: its find_spec's code object first (the add-ons' hook files run through runpy / exec, so
    their class's module is not in sys.modules), else the defining module's file, else what inspect resolves; None for a builtin/frozen
    importer or a class made at a prompt. The ONE such resolution of the package (hooks.installed reads it too)."""
    cls = type(f)
    for attr in ("find_spec", "find_module", "__init__"):
        code = getattr(getattr(cls, attr, None), "__code__", None)
        if code is not None and code.co_filename and os.path.isfile(code.co_filename):
            return os.path.abspath(code.co_filename)                            # as found; the containment predicate (_under) applies realpath once, on both sides
    mod = sys.modules.get(getattr(cls, "__module__", None) or "")
    file = getattr(mod, "__file__", None) if mod is not None else None
    if not file:
        try:
            import inspect
            file = inspect.getfile(cls)
        except (TypeError, OSError):
            return None
    return os.path.abspath(file) if os.path.exists(file) else None


def finder_words(f) -> str:
    """`<Class>@<defining file>` — how a log names a meta-path finder (the ENV-CLEAN line, the kit-hook refusals)."""
    return f"{type(f).__name__}@{finder_file(f) or 'builtin'}"


def kit_hooks_installed(dirs: Optional[List[str]] = None, home: Optional[str] = None, meta_path=None) -> List[str]:
    """The KIT hooks on sys.meta_path, as `<Class>@<file>` words: a finder is a kit hook iff the module defining its class lives under one of this
    tree's hook directories (kit_dirs: the carried add-ons and ports, the package's own hooks) — judged by file, never by class name, so an
    instrument attached from outside the tree (a measuring layer's probe whose finder class happens to share a kit's class name) never reads as one."""
    dirs = list(dirs) if dirs else kit_dirs(home)                                 # the stock proof passes the directories it already holds (no tree lookup inside the proven child)
    meta_path = sys.meta_path if meta_path is None else meta_path
    out = []
    for f in meta_path:
        file = finder_file(f)
        if file and _under(file, dirs):
            out.append(finder_words(f))
    return out


def interpreter_facts(environ: Optional[dict] = None) -> dict:
    """The stock child's interpreter as found: its sys.flags (isolated, ignore_environment, no_user_site, safe_path), every PYTHONPATH entry
    (the stock route: none under --det 0, exactly the det site under --det 1), every sys.meta_path finder as `<Class>@<defining file>`, and a census of the
    package's / core's modules loaded at proof time (the proof's own machinery — nothing of a kit hook)."""
    environ = os.environ if environ is None else environ
    flags = {k: int(getattr(sys.flags, k)) for k in ("isolated", "ignore_environment", "no_user_site", "no_site") if hasattr(sys.flags, k)}
    if hasattr(sys.flags, "safe_path"):
        flags["safe_path"] = int(sys.flags.safe_path)
    pp = [p for p in (environ.get("PYTHONPATH") or "").split(os.pathsep) if p]
    census = sorted(m for m in sys.modules if m.split(".")[0] in ("openfold3_ob0_opt", "opt_core", "openfold3", "torch"))
    return {"sys_flags": flags, "pythonpath_entries": pp, "meta_path": [finder_words(f) for f in sys.meta_path], "module_census": census}


def openfold_cache(environ: Optional[dict] = None) -> dict:
    """The OpenFold cache directory the stock call resolves (``$OPENFOLD_CACHE``, else upstream's ``~/.openfold3``: entry_points/parameters.py:26) and
    whether it holds a user default ``runner.yml``, which upstream deep-merges under any ``--runner-yaml`` (run_openfold.py:209-228)."""
    environ = os.environ if environ is None else environ
    d = environ.get("OPENFOLD_CACHE") or os.path.join(os.path.expanduser("~"), ".openfold3")
    yml = os.path.join(d, USER_RUNNER_YML)
    return {"dir": d, "from_env": bool(environ.get("OPENFOLD_CACHE")), "user_runner_yml": yml if os.path.isfile(yml) else None}


def env_proof(prefixes: List[str], dirs: Optional[List[str]] = None, environ: Optional[dict] = None, home: Optional[str] = None,
              det_exception: Optional[dict] = None) -> dict:
    """The proof dict (opt_core.stock_proof.env_proof: forbidden names present, kit modules loaded, kit dirs on sys.path, the autoload finder armed,
    the kit's sitecustomize, torch already loaded; under `det_exception` exactly the recipe's carve-out, `deviations` naming every difference) plus
    this tree's words: kit hooks on sys.meta_path, openfold3 already loaded, core modules beyond the proof machinery (DET_CORE_MODULES allowed under
    the carve-out), upstream's own reads and the declared reads present, the OpenFold cache directory. `ok` is True only when every list is empty."""
    environ = os.environ if environ is None else environ
    dirs = list(dirs or kit_dirs(home))
    proof = stock_proof.env_proof(env_absent=list(prefixes), kit_dirs=dirs, environ=_without_upstream(environ, home), det_exception=det_exception)
    allowed_core = stock_proof.CORE_ALLOWED_IN_STOCK + (DET_CORE_MODULES if det_exception else ())
    proof.update({
        "must_be_absent_prefixes": list(prefixes),
        "kit_hooks_installed": kit_hooks_installed(dirs),
        "openfold3_loaded_before_proof": "openfold3" in sys.modules,
        "core_modules_loaded": stock_proof.core_modules_loaded(allowed=allowed_core),
        "core_modules_allowed": list(allowed_core),
        "upstream_reads_present": {k: environ[k] for k in upstream_reads(home) if k in environ},
        "reads_present": {k: environ.get(k) for k in reads(home) if environ.get(k) is not None},
        "openfold_cache": openfold_cache(environ),
        "interpreter": interpreter_facts(environ),
    })
    if det_exception is None and proof["torch_loaded_before_proof"]:
        pass                                                            # already in the core's conjunction
    proof["ok"] = bool(proof["ok"]) and not (proof["kit_hooks_installed"] or proof["openfold3_loaded_before_proof"] or proof["core_modules_loaded"])
    return proof


def after_call(dirs: List[str], det_exception: Optional[dict] = None, upstream_fix: bool = False) -> dict:
    """The scan once the stock call has returned (opt_core.stock_proof.after_call_check plus the kit-hook and core-module scans): kit or core code
    loaded during the call means the arm was not stock. ``upstream_fix``: the call installed requested ``--upstream-fix`` entries, so the
    process holds the flag's core module (opt_core.stock_proof.UPSTREAM_FIX_MODULE) beside the proof machinery — named, nothing else."""
    rec = stock_proof.after_call_check(dirs, (), det_exception=det_exception)
    allowed_core = stock_proof.CORE_ALLOWED_IN_STOCK + (DET_CORE_MODULES if det_exception else ()) + ((stock_proof.UPSTREAM_FIX_MODULE,) if upstream_fix else ())
    rec["kit_hooks_installed_after"] = kit_hooks_installed(dirs)
    rec["core_modules_loaded_after"] = stock_proof.core_modules_loaded(allowed=allowed_core)
    rec["after_ok"] = bool(rec["after_ok"]) and not (rec["kit_hooks_installed_after"] or rec["core_modules_loaded_after"])
    return rec
