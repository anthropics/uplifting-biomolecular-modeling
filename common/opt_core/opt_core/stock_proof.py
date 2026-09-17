"""The stock proof: a stock run (mode ``off``) is proven clean before it runs, and the proof is a file.

Contract. The kit's ``pred --mode off`` launches its stock caller as ``python -s -m <kit.stock_module> --proof-json <path>
--env-absent <prefixes> --kit-dirs <dirs> [--module-prefixes <p>] [--det-env … --det-path …] -- <stock arguments>`` with every environment
name under the kit's must-be-absent prefixes stripped (:func:`strip_env`) and every PYTHONPATH entry inside a kit directory removed
(:func:`strip_pythonpath`). Before importing anything of the stock package the child proves what it is (:func:`env_proof`): no forbidden
name, no kit module loaded (by name prefix or by file under a kit directory), no kit directory on ``sys.path``, no autoload finder armed,
the kit's sitecustomize not this process's, torch not yet imported. ``ok`` is the conjunction; every list names the violations.
Under the deterministic recipe (``det_exception``: ``{"env": {name: value}, "pythonpath": [dir]}``) the proof allows exactly the recipe's
carve-out and nothing beyond it — ``deviations`` names every difference. A failed proof exits ``EXIT_NOT_STOCK`` without running anything.
:func:`after_call_check` is the same scan once the stock call returned (a lever module imported by the stock call is a defect).
The stock caller module stays in the kit — its entry point and its ``proven stock`` sentence are the kit's; it composes these.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Iterable, List, Mapping, Optional, Sequence

EXIT_NOT_STOCK = 3                                                      # the kit's EXIT_NOT_ACTIVE: nothing ran


def forbidden(environ: Mapping[str, str], prefixes: Iterable[str]) -> List[str]:
    """The names of ``environ`` starting with one of ``prefixes``, sorted."""
    prefixes = tuple(prefixes)
    return sorted(k for k in environ if any(k.startswith(p) for p in prefixes))


def strip_env(environ: Mapping[str, str], prefixes: Iterable[str], names: Iterable[str] = ()) -> tuple:
    """A copy of ``environ`` without the names under ``prefixes`` and without ``names``; returns ``(env, stripped_names)``."""
    prefixes = tuple(prefixes)
    names = set(names)
    out, stripped = {}, []
    for k, v in environ.items():
        if k in names or any(k.startswith(p) for p in prefixes):
            stripped.append(k)
        else:
            out[k] = v
    return out, sorted(stripped)


def _under(path: str, roots: Sequence[str]) -> bool:
    p = os.path.realpath(path)
    return any(p == r or p.startswith(r + os.sep) for r in roots)


def _roots(kit_dirs: Iterable[str]) -> List[str]:
    return [os.path.realpath(d) for d in kit_dirs if d]


def strip_pythonpath(value: Optional[str], kit_dirs: Iterable[str]) -> tuple:
    """PYTHONPATH without the entries inside a kit directory; returns ``(value_or_None, stripped_entries)``."""
    roots = _roots(kit_dirs)
    keep, stripped = [], []
    for e in (value or "").split(os.pathsep):
        if not e:
            continue
        (stripped if roots and _under(e, roots) else keep).append(e)
    return (os.pathsep.join(keep) or None), stripped


def _under_prefix(name: str, prefixes: Sequence[str]) -> bool:
    """``name`` is ``p`` itself or a submodule of it (``p.<anything>``) for some declared ``p`` -- dotted-package boundaries, never a
    bare substring: a kit prefix ``pkg`` names ``pkg`` and ``pkg.anything``, never an unrelated ``pkg_extra``."""
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def kit_modules_loaded(kit_dirs: Iterable[str], module_prefixes: Iterable[str], modules: Optional[Mapping] = None) -> List[str]:
    """The loaded modules that are the kit's: by name (``module_prefixes``, dotted-package boundaries) or by file (under a kit
    directory)."""
    modules = sys.modules if modules is None else modules
    roots = _roots(kit_dirs)
    prefixes = tuple(module_prefixes)
    def _file(mod):
        f = getattr(mod, "__file__", None)
        return f if isinstance(f, str) and f else None             # builtins, namespace packages and frozen modules have no file: never the kit's
    return sorted(m for m, mod in list(modules.items())
                  if (prefixes and _under_prefix(m, prefixes)) or (roots and _file(mod) is not None and _under(_file(mod), roots)))


HARNESS_MODULES = ("bench_peak_meter",)                                 # the launcher's own measurement instrumentation: present identically in
# every arm, including stock -- never a kit's, so a stock proof's census (kit modules, armed finders) must not name it a violation.


def _is_harness(obj) -> bool:
    """True for the launcher's own instrumentation: an explicit ``__harness__ = True`` marker on the object's class, or a class whose
    ``__module__`` is one of :data:`HARNESS_MODULES` (present identically in every arm, incl. stock -- named, not guessed)."""
    cls = type(obj)
    return bool(getattr(cls, "__harness__", False)) or cls.__module__ in HARNESS_MODULES


CORE_ALLOWED_IN_STOCK = ("opt_core", "opt_core.stock_proof")            # the proof machinery itself: the only core modules a stock process may hold
UPSTREAM_FIX_MODULE = "opt_core.upstream_fix"                            # the --upstream-fix plumbing (standard library only): held IN ADDITION by a stock process that installs
CORE_ALLOWED_WITH_UPSTREAM_FIX = CORE_ALLOWED_IN_STOCK + (UPSTREAM_FIX_MODULE,)   # a requested fix after its proof (the arm is then stock plus the named fixes, recorded by the caller)
# A stock caller therefore never imports opt_core.det (or any other core module) in the child: the --det carve-out is computed in the
# parent (det.stock_exception) and passed on the command line (--det-env / --det-path), which is what stock_command does.


def core_modules_loaded(modules: Optional[Mapping] = None, allowed: Sequence[str] = CORE_ALLOWED_IN_STOCK) -> List[str]:
    """The core modules loaded beyond the proof machinery (``opt_core.autoload``, ``opt_core.gates`` … in a stock process are a violation)."""
    modules = sys.modules if modules is None else modules
    return sorted(m for m in modules if (m == "opt_core" or m.startswith("opt_core.")) and m not in set(allowed))


def armed_finders(meta_path: Optional[Sequence] = None, finder_module_suffix: str = "._autoload") -> List[str]:
    """The armed autoload finders on the meta path: any finder whose class comes from a ``*._autoload`` module or from
    ``opt_core.autoload`` and whose ``armed`` is true. The launcher's own instrumentation (:func:`_is_harness`) is never a
    finder this proof reports -- it is present identically in every arm, including stock, and is not a kit's."""
    meta_path = sys.meta_path if meta_path is None else meta_path
    out = []
    for f in meta_path:
        if _is_harness(f):
            continue
        mod = type(f).__module__ or ""
        if (mod.endswith(finder_module_suffix) or mod == "opt_core.autoload") and getattr(f, "armed", True):
            out.append(type(f).__name__)
    return out


def env_proof(*, env_absent: Iterable[str], kit_dirs: Iterable[str], module_prefixes: Iterable[str] = (), environ: Optional[Mapping] = None,
              modules: Optional[Mapping] = None, path: Optional[Sequence[str]] = None, meta_path: Optional[Sequence] = None,
              det_exception: Optional[dict] = None) -> dict:
    """The clean-process proof (module contract). ``kit_modules_loaded_after`` is added by :func:`after_call_check`."""
    environ = os.environ if environ is None else environ
    modules = sys.modules if modules is None else modules
    path = sys.path if path is None else path
    meta_path = sys.meta_path if meta_path is None else meta_path
    env_absent = list(env_absent)
    kit_dirs = list(kit_dirs)
    roots = _roots(kit_dirs)
    hits = forbidden(environ, env_absent)
    kit_mods = kit_modules_loaded(kit_dirs, module_prefixes, modules)
    kit_dirs_on_path = sorted(p for p in path if p and roots and os.path.exists(p) and _under(p, roots))   # import-hook keys and zip entries are not directories
    armed = armed_finders(meta_path)
    sc = modules.get("sitecustomize")
    sc_file = getattr(sc, "__file__", None) if sc is not None else None
    kit_sitecustomize = sc_file if (sc_file and roots and _under(sc_file, roots)) else None
    torch_loaded = "torch" in modules
    core_extra = core_modules_loaded(modules)
    proof = {"env_absent": env_absent, "forbidden_present": hits, "kit_modules_loaded": kit_mods, "kit_dirs_on_path": kit_dirs_on_path,
             "autoload_armed": armed, "kit_sitecustomize": kit_sitecustomize, "torch_loaded_before_proof": torch_loaded,
             "core_modules_loaded": core_extra, "no_user_site": bool(sys.flags.no_user_site), "python": sys.executable, "det_exception": None}
    if det_exception is None:
        proof["ok"] = not (hits or kit_mods or kit_dirs_on_path or armed or kit_sitecustomize or torch_loaded or core_extra)
        return proof
    dev = []
    exp_env = dict(det_exception.get("env") or {})
    exp_dirs = [os.path.realpath(d) for d in det_exception.get("pythonpath") or []]
    exp_forbidden = forbidden(exp_env, env_absent)                       # the recipe's names under the must-be-absent prefixes
    if hits != exp_forbidden:
        dev.append(f"forbidden names present {hits} != the recipe's {exp_forbidden}")
    for k, v in exp_env.items():
        if environ.get(k) != v:
            dev.append(f"{k}={environ.get(k)!r} != the recipe's {v!r}")
    if sorted(os.path.realpath(p) for p in kit_dirs_on_path) != sorted(exp_dirs):
        dev.append(f"kit directories on sys.path {kit_dirs_on_path} != the recipe's {list(det_exception.get('pythonpath') or [])}")
    exp_sc = [os.path.join(d, "sitecustomize.py") for d in exp_dirs]
    if exp_sc and (not kit_sitecustomize or os.path.realpath(kit_sitecustomize) not in exp_sc):
        dev.append(f"the process's sitecustomize is {kit_sitecustomize or sc_file or 'none'}, not the recipe's {exp_sc[0]}")
    allowed_mods = ["sitecustomize"] if exp_sc else []
    if [m for m in kit_mods if m not in allowed_mods]:
        dev.append(f"kit modules loaded {kit_mods} != {allowed_mods}")
    if armed:
        dev.append(f"autoload finder armed: {armed}")
    proof["det_exception"] = {"env": exp_env, "pythonpath": list(det_exception.get("pythonpath") or []), "sitecustomize": kit_sitecustomize,
                              "modules": kit_mods, "torch_loaded_by_sitecustomize": torch_loaded, "deviations": dev}
    proof["ok"] = not dev
    return proof


def after_call_check(kit_dirs: Iterable[str], module_prefixes: Iterable[str] = (), *, lever_modules: Iterable[str] = (),
                     det_exception: Optional[dict] = None, modules: Optional[Mapping] = None) -> dict:
    """The scan once the stock call has returned: the kit modules loaded now and whether any of ``lever_modules`` was imported."""
    modules = sys.modules if modules is None else modules
    after = kit_modules_loaded(kit_dirs, module_prefixes, modules)
    allowed = ["sitecustomize"] if det_exception and det_exception.get("pythonpath") else []
    extra = [m for m in after if m not in allowed]
    levers = sorted(m for m in lever_modules if m in modules)
    return {"kit_modules_loaded_after": after, "lever_modules_imported": levers, "after_ok": not extra and not levers}


def violations_sentence(proof: Mapping) -> str:
    """The NOT STOCK detail: every violation class with its value (the kit prefixes it with its own ``[tag stock] NOT STOCK:``)."""
    if proof.get("det_exception"):
        return "det exception not met: " + "; ".join(proof["det_exception"]["deviations"])
    return (f"forbidden env {proof['forbidden_present']}, kit modules {proof['kit_modules_loaded']}, kit dirs {proof['kit_dirs_on_path']}, "
            f"autoload {proof['autoload_armed']}, kit sitecustomize {proof['kit_sitecustomize']}, torch loaded {proof['torch_loaded_before_proof']}")


def clean_sentence(proof: Mapping) -> str:
    """The ENV-CLEAN detail: ``absent=<prefixes> kit_modules=none kit_dirs=none no_user_site=<bool>``."""
    return (f"absent={','.join(proof['env_absent']) or 'none'} kit_modules={','.join(proof['kit_modules_loaded']) or 'none'} "
            f"kit_dirs={','.join(proof['kit_dirs_on_path']) or 'none'} no_user_site={proof['no_user_site']}")


# ------------------------------------------------------------------------------------------------------------------ the command contract


def stock_command(python: str, module: str, *, proof_json: str, env_absent: Iterable[str], kit_dirs: Iterable[str],
                  args: Sequence[str], module_prefixes: Iterable[str] = (), det: Optional[dict] = None) -> List[str]:
    """The child's argv: ``<python> -s -m <module> --proof-json … --env-absent … --kit-dirs … [--module-prefixes …] [--det-env … --det-path …] -- <args>``."""
    cmd = [python, "-s", "-m", module, "--proof-json", proof_json, "--env-absent", ",".join(env_absent), "--kit-dirs", os.pathsep.join(kit_dirs)]
    prefixes = list(module_prefixes)
    if prefixes:
        cmd += ["--module-prefixes", ",".join(prefixes)]
    if det:
        cmd += ["--det-env", ",".join(f"{k}={v}" for k, v in (det.get("env") or {}).items()),
                "--det-path", os.pathsep.join(det.get("pythonpath") or [])]
    return cmd + ["--", *args]


def split_argv(argv: Sequence[str]) -> tuple:
    """``[own options] -- [stock arguments]``."""
    argv = list(argv)
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def parse_stock_argv(argv: Sequence[str], prog: str = "python -s -m <stock module>") -> tuple:
    """The child's side of :func:`stock_command`: returns ``(options, stock_args)`` where options has ``proof_json``, ``env_absent``
    (list), ``kit_dirs`` (list), ``module_prefixes`` (list) and ``det_exception`` (dict or None)."""
    import argparse
    own, stock = split_argv(argv)
    ap = argparse.ArgumentParser(prog=prog, description="the stock call, proven clean first")
    ap.add_argument("--proof-json", required=True)
    ap.add_argument("--env-absent", required=True)
    ap.add_argument("--kit-dirs", default="")
    ap.add_argument("--module-prefixes", default="")
    ap.add_argument("--det-env", default="")
    ap.add_argument("--det-path", default="")
    a = ap.parse_args(own)
    a.env_absent = [s for s in a.env_absent.split(",") if s]
    a.kit_dirs = [d for d in a.kit_dirs.split(os.pathsep) if d]
    a.module_prefixes = [s for s in a.module_prefixes.split(",") if s]
    a.det_exception = None
    if a.det_env or a.det_path:
        a.det_exception = {"env": dict(kv.split("=", 1) for kv in a.det_env.split(",") if kv),
                           "pythonpath": [d for d in a.det_path.split(os.pathsep) if d]}
    return a, stock


def write_proof(path: str, proof: Mapping) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(proof, fh, indent=1, default=str)
        fh.write("\n")
    return path
