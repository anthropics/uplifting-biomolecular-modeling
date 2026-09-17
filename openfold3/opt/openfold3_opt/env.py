"""The stock environment: what a stock process must not carry, how it is stripped, and how a process proves it (stock/PINS.json
"stock_environment" is the one list; this module reads it).

* ``must_be_absent()`` — the name prefixes of stock/PINS.json; ``strip()`` removes every matching variable from an environment copy
  (the stock subprocess starts from it); ``forbidden_present()`` lists what is present.
* ``kit_dirs()`` — every add-on directory; ``kit_modules_loaded()`` — modules of ``sys.modules`` whose file lives under one of them;
  ``kit_dirs_on_path()`` — ``sys.path`` entries under one of them; ``kit_hooks_installed()`` — kit finders on ``sys.meta_path``
  (the classes the hooks define: ``_LeverFinder`` in the kit's and the atom-hoist's ``sitecustomize.py``, ``_Finder`` in the trunk-kernels'
  and the FlashPairformer's).
* ``env_proof()`` — the dict the stock caller hands to its parent (``stock_pred --proof-json``) before it imports openfold3.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

from opt_core import stock_proof

from . import modes

HOOK_FINDER_CLASSES = ("_LeverFinder", "_Finder", "_PostImportFinder", "_TPFinder")     # the add-ons' finders, the offload port's and the package confhead hook's (offload/of3o/sitecustomize.py, hooks/confhead/sitecustomize.py) and the tp line's (tp_rowpair/hook/sitecustomize.py)


def tree_home(environ: Optional[dict] = None) -> str:
    """The openfold3/ directory: OPENFOLD3_OPT_HOME, else MODEL_OPT, else the package's parent tree (opt/openfold3_opt -> openfold3/) — opt_core.home.tree_home."""
    from opt_core.home import tree_home as _tree_home
    return _tree_home(__file__, env_tree="MODEL_OPT", env_home="OPENFOLD3_OPT_HOME", levels=2, environ=environ)


_CHECK_PINS: dict = {}


def check_pins(home: Optional[str] = None):
    """stock/check_pins.py as a module (the pin checker run.sh runs as a script): its readers `read_pins`, `read_sums` and `sha256_file` are
    the tree's one copy of each; loaded once per home, from the file, without a package import."""
    home = home or tree_home()
    if home not in _CHECK_PINS:
        import importlib.util
        path = os.path.join(home, "stock", "check_pins.py")
        spec = importlib.util.spec_from_file_location("openfold3_stock_check_pins", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CHECK_PINS[home] = mod
    return _CHECK_PINS[home]


def pins(home: Optional[str] = None) -> dict:
    home = home or tree_home()
    return check_pins(home).read_pins(os.path.join(home, "stock", "PINS.json"))


def must_be_absent(home: Optional[str] = None) -> List[str]:
    return list(pins(home)["stock_environment"]["must_be_absent_prefixes"])


def reads(home: Optional[str] = None) -> List[str]:
    return list(pins(home)["stock_environment"]["reads"])


def forbidden_present(prefixes: List[str], environ: Optional[dict] = None) -> List[str]:
    return stock_proof.forbidden(os.environ if environ is None else environ, prefixes)


def strip(prefixes: List[str], environ: Optional[dict] = None) -> Dict[str, str]:
    """A copy of `environ` without the must-be-absent names and without any kit directory on PYTHONPATH (opt_core.stock_proof)."""
    out, _ = stock_proof.strip_env(os.environ if environ is None else environ, prefixes)
    pp, _ = stock_proof.strip_pythonpath(out.get("PYTHONPATH"), [os.path.abspath(d) for d in kit_dirs()])
    if pp:
        out["PYTHONPATH"] = pp
    else:
        out.pop("PYTHONPATH", None)
    return out


def kit_dirs(home: Optional[str] = None) -> List[str]:
    """Every hook-carrying directory of the tree: the carried add-ons and ports (modes' add-on table) and the package's own hooks (modes.PACKAGE_HOOKS)."""
    home = home or tree_home()
    return [os.path.join(home, rel) for rel in list(modes.KITS.values()) + list(modes.PACKAGE_HOOKS.values())]


def _under(path: str, dirs: List[str]) -> bool:
    """The core's ONE path-containment predicate (opt_core.stock_proof._under: realpath on both sides)."""
    return stock_proof._under(path, [os.path.realpath(d) for d in dirs if d])


def kit_modules_loaded(dirs: Optional[List[str]] = None) -> List[str]:
    """The loaded modules whose file lives under a kit directory (opt_core.stock_proof.kit_modules_loaded, by file)."""
    return sorted(stock_proof.kit_modules_loaded([os.path.abspath(d) for d in (dirs or kit_dirs())], ()))


def kit_dirs_on_path(dirs: Optional[List[str]] = None) -> List[str]:
    dirs = [os.path.abspath(d) for d in (dirs or kit_dirs())]
    return [p for p in sys.path if p and _under(p, dirs)]


def kit_hooks_installed() -> List[str]:
    """Kit finders on sys.meta_path, by class name (modes.HOOK_FINDER_CLASSES: the add-ons' and the package's own finder classes; any
    other finder on the path is not kit code and does not read as one)."""
    return [type(f).__name__ for f in sys.meta_path if type(f).__name__ in HOOK_FINDER_CLASSES]


def env_proof(prefixes: List[str], dirs: Optional[List[str]] = None, environ: Optional[dict] = None, home: Optional[str] = None) -> dict:
    """The proof dict: forbidden names present, kit modules loaded, kit dirs on sys.path, kit hooks on sys.meta_path, the autoload
    finder armed, core modules beyond the proof machinery (opt_core.stock_proof.core_modules_loaded: the stock process may hold
    `opt_core` and `opt_core.stock_proof`, nothing else of the core), torch already loaded, no user site. `ok` is True only when every
    list is empty. `home` is passed through to the pins read (the stock child never derives the tree itself)."""
    environ = os.environ if environ is None else environ
    autoload = stock_proof.armed_finders()
    proof = {
        "forbidden_present": forbidden_present(prefixes, environ),
        "kit_modules_loaded": kit_modules_loaded(dirs),
        "kit_dirs_on_path": kit_dirs_on_path(dirs),
        "kit_hooks_installed": kit_hooks_installed(),
        "autoload_armed": autoload,
        "torch_loaded_before_proof": "torch" in sys.modules,
        "openfold3_loaded_before_proof": "openfold3" in sys.modules,
        "no_user_site": bool(sys.flags.no_user_site),
        "must_be_absent_prefixes": list(prefixes),
        "core_modules_loaded": stock_proof.core_modules_loaded(),
        "reads_present": {k: environ.get(k) for k in reads(home) if environ.get(k) is not None},
    }
    proof["ok"] = not (proof["forbidden_present"] or proof["kit_modules_loaded"] or proof["kit_dirs_on_path"] or proof["kit_hooks_installed"]
                       or proof["autoload_armed"] or proof["core_modules_loaded"] or proof["torch_loaded_before_proof"] or proof["openfold3_loaded_before_proof"])
    return proof
