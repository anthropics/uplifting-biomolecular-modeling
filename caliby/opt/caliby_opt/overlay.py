"""The exact tree as an import overlay: the kits' files loaded under the upstream module names; site-packages never touched.

The levers of both kits are whole-file replacements of eight upstream modules (``stack.TOUCHED``; the
exact row's bytes are the kit's ``opt/forward/xattempt_addon/fast/<file>``, whose ``potts.py`` and ``atom_mpnn_denoiser.py``
already contain the fast_inference patches' patches). This module makes those files the modules the process imports, without writing into
the installed tree:

    plan(files)      module name -> {file, path (the kit file), installed (the module's path in site-packages), sha256 (the served
                     file's own digest, for the record — every overlay file is tracked in this tree, so nothing is compared
                     against it)} for every lever file of the kit rows (the two protpardelle modules only when that package is
                     installed; ``clean_pdbs.py`` never — no kit replaces it);
    missing(plan)    every overlay file proven present at its path — returns the findings;
    install(plan)    a meta-path finder, first on ``sys.meta_path``, that answers exactly those module names with the kit file
                     (a source loader that writes no bytecode); refused once any of them is already imported;
    loaded()         after the run: every overlaid name in ``sys.modules`` with the file it was actually loaded from;
    exit_state(mode) ``exact`` when every loaded planned module is its planned file, ``stock`` when (mode off) no finder is installed
                     and every loaded lever module lives under its installed package, ``mixed`` otherwise — the EXIT line's ``tree=``
                     (report.py).

``off`` installs no finder: the stock child imports the installed upstream files and nothing else (``plan(files, "off")`` is empty).

The installed ``caliby`` / ``chroma`` / ``protpardelle`` trees stay the pinned upstream bytes in every mode (``stack.tree_state()``
== ``stock`` and the kit's tree digest == ``stock/PINS.json`` "tree_digest_upstream" are activation gates of BOTH modes,
activate.py), so ``off`` and ``exact`` processes run side by side on one interpreter. The modules execute the same bytes under the
same names and the same ``__file__`` as an installed-file form of the kits would have (their files copied into
site-packages by hand): ``__file__`` is the installed module's path because the
modules resolve package resources from it (``caliby/api.py`` reads ``configs/`` beside itself), while the code is read from the kit
file and tracebacks name the kit file (the loader's path; ``loaded()`` reports both). One consequence to know when debugging:
``inspect.getsource`` of a CLASS defined in an overlaid module reads the file ``__file__`` names — the installed upstream text — while
functions and tracebacks (``co_filename``) read the kit file; ``loaded()`` / ``source_of()`` are the authority on what executed. Forked workers (the loader and background-CIF
levers, the DataLoader) inherit the modules.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
from typing import Dict, List, Optional

from . import modes, stack


class OverlayError(RuntimeError):
    """The overlay cannot be installed: a planned file is missing, or a lever module is already imported."""


def module_name(package: str, relpath: str) -> str:
    """``("caliby", "model/seq_denoiser/denoisers/seq_design/potts.py")`` -> ``caliby.model.seq_denoiser.denoisers.seq_design.potts``."""
    return package + "." + relpath[:-3].replace("/", ".")


def overlay_dir() -> str:
    """The kit's ``fast/`` directory: the kit rows' file for every entry of ``stack.TOUCHED`` it replaces."""
    return os.path.join(stack.kit_dir(stack.KIT_ADDON), "fast")


def plan(files: Optional[Dict[str, dict]] = None, mode: Optional[str] = "exact") -> Dict[str, dict]:
    """module name -> {"file": <TOUCHED key>, "path": <the kit file served>, "installed": <site-packages path of the module>,
    "sha256": <the served file's own digest, hashed on the spot — the record of what shipped, not a value checked against anything;
    every overlay file is tracked in this tree>} — what the finder serves in ``mode``: a kit mode (fast, exact) = the kit rows' files
    (``opt/forward/xattempt_addon/fast/``); ``off`` = nothing (empty: the stock child imports the installed upstream files). ``files`` is
    ``stack.installed_files()`` (a protpardelle module is planned only when that package is installed)."""
    files = files if files is not None else stack.installed_files()
    out: Dict[str, dict] = {}
    if modes.is_kit_mode(mode):
        for key, (pkg, rel) in stack.TOUCHED.items():
            if key in stack.NEVER_REPLACED:
                continue
            if files[key]["state"] == "package-absent":
                continue
            installed = files[key].get("path") or os.path.join(stack.package_root(pkg) or pkg, rel)
            path = os.path.join(overlay_dir(), key)
            out[module_name(pkg, rel)] = {"file": key, "path": path, "installed": installed, "sha256": stack.sha256_file(path)}
    return out


def missing(modules: Dict[str, dict]) -> List[str]:
    """Every overlay file proven present at its path: every overlay file is tracked in this repo, so
    presence is the whole check, never a pinned digest. One ``<file>: missing at <path>`` per finding (empty = proven)."""
    bad = []
    for name, ent in modules.items():
        if stack.sha256_file(ent["path"]) is None:
            bad.append(f"{ent['file']}: missing at {ent['path']}")
    return bad


class _Loader(importlib.machinery.SourceFileLoader):
    """Source loader for a kit file that never writes bytecode (nothing lands inside the release tree)."""

    def set_data(self, path, data, *args, **kwargs):
        return None


class OverlayFinder:
    """Meta-path finder answering the planned module names with the kit files; every other name passes through. The spec's origin
    (the module's ``__file__``) is the installed module's path; the loader reads the kit file."""

    def __init__(self, modules: Dict[str, dict]):
        self.modules = dict(modules)
        self.served: Dict[str, str] = {}

    def find_spec(self, fullname, path=None, target=None):
        ent = self.modules.get(fullname)
        if ent is None:
            return None
        loader = _Loader(fullname, ent["path"])
        spec = importlib.util.spec_from_file_location(fullname, ent.get("installed") or ent["path"], loader=loader)
        self.served[fullname] = ent["path"]
        return spec

    def invalidate_caches(self):
        return None


_FINDER: Optional[OverlayFinder] = None


def installed() -> Optional[OverlayFinder]:
    return _FINDER


def install(modules: Dict[str, dict]) -> OverlayFinder:
    """Prove the files and put the finder first on ``sys.meta_path`` (idempotent for the same plan). Raises OverlayError when a
    planned file is absent, or when one of the modules is already imported (an import hook cannot reach a
    loaded module)."""
    global _FINDER
    if _FINDER is not None:
        if set(_FINDER.modules) != set(modules):
            raise OverlayError(f"an overlay is already installed for {sorted(_FINDER.modules)}; refused for {sorted(modules)}")
        return _FINDER
    loaded = [m for m in modules if m in sys.modules]
    if loaded:
        raise OverlayError(f"late activation: {loaded[0]} is already imported in this process; the kit files are loaded by import hook, "
                           f"so enable() must run before the model package is imported (import caliby alone imports caliby.api)")
    bad = missing(modules)
    if bad:
        raise OverlayError("overlay files not proven (missing at their planned path): " + "; ".join(bad))
    _FINDER = OverlayFinder(modules)
    sys.meta_path.insert(0, _FINDER)
    importlib.invalidate_caches()
    return _FINDER


def uninstall() -> bool:
    """Remove the finder (tests). Modules already imported stay what they are."""
    global _FINDER
    if _FINDER is None:
        return False
    try:
        sys.meta_path.remove(_FINDER)
    except ValueError:
        pass
    _FINDER = None
    return True


def source_of(mod) -> Optional[str]:
    """The file a module's code was read from: the overlay loader's path for an overlaid module, else its ``__file__``."""
    spec = getattr(mod, "__spec__", None)
    loader = getattr(spec, "loader", None) if spec is not None else None
    if isinstance(loader, _Loader):
        return loader.path
    return getattr(mod, "__file__", None)


def loaded(modules: Optional[Dict[str, dict]] = None) -> Dict[str, dict]:
    """Every planned module present in ``sys.modules``: {name: {"expected": <kit file>, "source": <the file its code was read from>,
    "file": <its __file__>, "ok": source is the kit file}}."""
    modules = modules if modules is not None else (_FINDER.modules if _FINDER is not None else {})
    out = {}
    for name, ent in modules.items():
        mod = sys.modules.get(name)
        if mod is None:
            continue
        src = source_of(mod)
        ok = bool(src) and os.path.realpath(src) == os.path.realpath(ent["path"])
        out[name] = {"expected": ent["path"], "source": src, "file": getattr(mod, "__file__", None), "ok": ok}
    return out


def foreign_lever_modules(exclude=()) -> List[str]:
    """The lever modules (``stack.TOUCHED``) loaded in this process from outside their installed package root, ``exclude`` apart — on
    ``off`` each one is a kit file in a stock process."""
    roots = {p: stack.package_root(p) for p in ("caliby", "chroma", "protpardelle")}
    out = []
    for key, (pkg, rel) in stack.TOUCHED.items():
        name = module_name(pkg, rel)
        if name in exclude:
            continue
        mod = sys.modules.get(name)
        if mod is None:
            continue
        f, root = source_of(mod), roots.get(pkg)
        if not f or not root or not os.path.realpath(f).startswith(os.path.realpath(root) + os.sep):
            out.append(name)
    return out


def exit_state(mode: Optional[str]) -> str:
    """The EXIT line's ``tree=``: what the lever modules loaded in this process actually are.
    a kit mode (fast, exact): ``exact`` iff the overlay is installed and every loaded planned module is its planned file, else ``mixed``;
    off: ``stock`` iff no finder is installed and every loaded lever module lives under its installed package root, else ``mixed``."""
    if modes.is_kit_mode(mode):
        if _FINDER is None:
            return "mixed"
        got = loaded()
        return "exact" if all(v["ok"] for v in got.values()) else "mixed"
    if _FINDER is not None:
        return "mixed"
    return "mixed" if foreign_lever_modules() else "stock"
