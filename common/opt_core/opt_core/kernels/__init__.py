"""Carried kernels: one copy per kernel, matching the kit byte-for-byte, routed BY NAME and held to the core's own copy
before import.

Contract. A carried kernel ``<name>`` lives here as ``<name>/`` (a package) or ``<name>.py`` (a module) beside its metadata file
``META/<name>.json``: the kernel's kind/version/licence, what is deliberately not carried (kit-side data such as a tuned cell table),
the environment variables a kit exports before the first import (``exports``), and the other kernel names the carried code imports at
run time (``runtime_imports``). The core changes no byte of a carried kernel; the carried bytes are the one place in the core that may
name the model (their docstrings are the kit's). Equality is the git commit: ``META/<name>.json`` carries no digest of the
carried bytes, and no list of them either — the file set a carried kernel consists of is read live off disk (``sums(name)["files"]``,
always current, never stale). Where a copy came from is not recorded here. A carried package that ships compiled binaries keeps
their integrity digests in its OWN manifest files, which its own loader reads and checks at load time (a binary-integrity check of
that package, carried byte-for-byte like the rest of it) — never in ``META``.

Routing is by NAME, never by directory. ``route(name)`` installs one meta-path finder (first on ``sys.meta_path``) that serves exactly
the routed top-level names from the core copies; a name a kit did not route stays invisible, so a kit that keeps its own copy of another
kernel keeps importing it, and a kernel that another carried kernel imports at run time (``runtime_imports``) is reachable only when the
kit routes it too — a ROUTE never changes what executes beyond the routed name. ``route_check(name)`` resolves the name without
importing it (an already-imported module answers with its own location), compares every file the core currently carries against the
resolved location BY BYTES (a live read-and-compare, not a stored digest), and holds every required export in the environment (the
variable set and, for a path, the file present) — the carried code reads them at import, so the check runs BEFORE the first import;
a missing export, a neighbouring version, a missing or differing file: a named refusal. ``verify_carry(name)`` is the same byte
comparison without the Gate/exports wrapping, for a quick "is what would import here clean" check.

A kit that keeps its OWN adapted fork of a carried kernel (a different calling convention, an engine-default policy specific to that
kit, a feature not yet upstreamed) never calls ``route``/``route_check`` for that name at all — carrying and adapting are for the kit's
own maintainers to weigh; this module only serves a kit that uses the core's copy unmodified.
"""
import filecmp
import glob
import importlib.machinery
import importlib.util
import json
import os
import sys
from typing import Dict, List, Mapping, Optional

from ..gates import Gate

KERNELS_DIR = os.path.dirname(os.path.abspath(__file__))
META_DIR = os.path.join(KERNELS_DIR, "META")


def names() -> List[str]:
    """The carried kernel names (one metadata file each)."""
    return sorted(f[:-5] for f in os.listdir(META_DIR) if f.endswith(".json"))


def _live_files(name: str, kind: str) -> List[str]:
    """Every file the core currently carries for ``name``, relative to its root (the package directory, or ``KERNELS_DIR`` for a
    module, whose carried files sit there as ``name.*``) -- read off disk now, never stored: the file set IS whatever is there."""
    if kind == "package":
        root = os.path.join(KERNELS_DIR, name)
        out = []
        for r, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            out.extend(os.path.relpath(os.path.join(r, f), root) for f in files if not f.endswith(".pyc"))
        return sorted(out)
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(KERNELS_DIR, name + ".*")))


def sums(name: str) -> dict:
    """The metadata of ``name``: kind, version, license, not_carried, exports, runtime_imports, and ``files`` -- the live file
    list (relative paths), computed now, not stored."""
    path = os.path.join(META_DIR, name + ".json")
    if not os.path.isfile(path):
        raise KeyError(f"no carried kernel {name!r} (carried: {', '.join(names())})")
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    doc["files"] = _live_files(name, doc["kind"])
    return doc


def carried_path(name: str) -> str:
    """The core copy: the package directory or the module file."""
    return os.path.join(KERNELS_DIR, name if sums(name)["kind"] == "package" else name + ".py")


def carried_files() -> List[str]:
    """Every carried file, relative to this directory (the bytes the hygiene rules exempt)."""
    out = []
    for name in names():
        doc = sums(name)
        out.extend(f"{name}/{f}" if doc["kind"] == "package" else f for f in doc["files"])
    return sorted(out)


def _hold(target_root: str, name: str, kind: str) -> Dict[str, List[str]]:
    """Compare ``target_root`` (a package dir, or the directory holding a module's files) against the core's own carried copy of
    ``name``, file by file, live -- every file the core currently carries, byte-for-byte (no stored digest either side)."""
    core_root = os.path.join(KERNELS_DIR, name) if kind == "package" else KERNELS_DIR
    differing, missing = [], []
    for rel in _live_files(name, kind):
        p = os.path.join(target_root, rel)
        if not os.path.isfile(p):
            missing.append(rel)
        elif not filecmp.cmp(p, os.path.join(core_root, rel), shallow=False):
            differing.append(rel)
    return {"differing": differing, "missing": missing}


def verify_carry(name: str, path: Optional[List[str]] = None) -> List[str]:
    """Wherever ``import <name>`` would resolve (see ``resolve``) held to the core's own copy, live, file by file -- or, when
    nothing resolves it (no route, not imported, not on ``path``), the core's own copy held to itself (always clean; there is
    nothing else to compare). A list of problems (empty = clean). The simpler sibling of ``route_check`` -- the same byte
    comparison, no Gate, no exports check."""
    kind = sums(name)["kind"]
    where = resolve(name, path)
    if where is None:
        where = carried_path(name)
    root = where if kind == "package" else os.path.dirname(where)
    h = _hold(root, name, kind)
    return [f"{name}/{f}: differs from the core copy" for f in h["differing"]] + [f"{name}/{f}: missing" for f in h["missing"]]


# ------------------------------------------------------------------------------------------------------------ routing by name

class RouteFinder:
    """The one meta-path finder of the routed names: serves exactly ``self.routed`` (top-level names) from the core copies. Submodules
    of a routed package resolve through the package's own ``__path__`` (the core directory), as for any package."""

    def __init__(self):
        self.routed: Dict[str, str] = {}                      # name -> the core copy (dir or file)

    def find_spec(self, fullname, path=None, target=None):
        where = self.routed.get(fullname)
        if where is None:
            return None
        if os.path.isdir(where):
            return importlib.util.spec_from_file_location(fullname, os.path.join(where, "__init__.py"), submodule_search_locations=[where])
        return importlib.util.spec_from_file_location(fullname, where)

    def invalidate_caches(self):
        pass


def _finder() -> Optional[RouteFinder]:
    for f in sys.meta_path:
        if isinstance(f, RouteFinder):
            return f
    return None


def routed() -> List[str]:
    """The names routed to the core copies in this process."""
    f = _finder()
    return sorted(f.routed) if f else []


def route(name: str) -> str:
    """Route ``import <name>`` to the core copy (this name only). Returns the core copy's path. Idempotent."""
    where = carried_path(name)
    f = _finder()
    if f is None:
        f = RouteFinder()
        sys.meta_path.insert(0, f)
    f.routed[name] = where
    importlib.invalidate_caches()
    return where


def unroute(name: str) -> bool:
    """Forget a route (the finder stays; an already-imported module is untouched). True when it was routed."""
    f = _finder()
    if f is None or name not in f.routed:
        return False
    del f.routed[name]
    return True


def resolve(name: str, path: Optional[List[str]] = None) -> Optional[str]:
    """Where ``import <name>`` lands (no import): an already-imported module's own location; else the core copy when the name is
    routed; else the first match on ``path`` (``sys.path``); None when not importable."""
    mod = sys.modules.get(name)
    if mod is not None:
        origin = getattr(mod, "__file__", None)
        if not origin:
            return None
        origin = os.path.abspath(origin)
        return os.path.dirname(origin) if os.path.basename(origin) == "__init__.py" else origin
    f = _finder()
    if f is not None and name in f.routed:
        return f.routed[name]
    spec = importlib.machinery.PathFinder.find_spec(name, sys.path if path is None else list(path))
    if spec is None or not spec.origin or spec.origin in ("built-in", "frozen"):
        return None
    origin = os.path.abspath(spec.origin)
    return os.path.dirname(origin) if spec.submodule_search_locations is not None else origin


def exports(name: str, **values) -> Dict[str, str]:
    """The environment a kit exports BEFORE importing ``name``: each export names its parameter; a required one missing raises by name."""
    doc = sums(name)
    env = {}
    for var, spec in sorted(doc.get("exports", {}).items()):
        val = values.get(spec["param"])
        if val is None:
            if spec.get("required"):
                raise ValueError(f"{name}: {var} needs {spec['param']}= ({spec['what']})")
            continue
        env[var] = os.path.abspath(str(val)) if spec.get("path") else str(val)
    return env


def exports_check(name: str, environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """The required exports of ``name`` held in ``environ``: a list of problems (variable unset; a path export naming a missing file)."""
    environ = os.environ if environ is None else environ
    problems = []
    for var, spec in sorted(sums(name).get("exports", {}).items()):
        val = environ.get(var)
        if not val:
            if spec.get("required"):
                problems.append(f"{var} not exported ({spec['what']})")
            continue
        if spec.get("path") and not os.path.exists(val):
            problems.append(f"{var}={val}: no such file")
    return problems


def route_check(name: str, path: Optional[List[str]] = None, environ: Optional[Mapping[str, str]] = None) -> Gate:
    """Gate: ``import <name>`` resolves to bytes identical to the core's own copy (a live comparison, not a stored digest), and
    every required export is in the environment and names an existing file. Run it BEFORE the first import of the kernel (the
    carried code reads its exports at import). ``details`` carries the resolution of every ``runtime_imports`` name (None = not
    importable: the kernel's own fallback applies) — the kit records it."""
    doc = sums(name)
    core = carried_path(name)
    where = resolve(name, path)
    details = {"kernel": name, "core_copy": core, "resolved": where, "routed": name in routed(), "already_imported": name in sys.modules,
               "runtime_imports": {n: resolve(n, path) for n in doc.get("runtime_imports", [])}}
    gate = f"kernel_route:{name}"
    if where is None:
        return Gate(name=gate, ok=False, details=details, reason=f"{name} not importable: kernels.route({name!r}) routes it to the core copy")
    root = where if doc["kind"] == "package" else os.path.dirname(where)
    h = _hold(root, name, doc["kind"])
    details.update(h)
    if h["differing"] or h["missing"]:
        return Gate(name=gate, ok=False, details=details,
                    reason=(f"{name} resolves to {where}, not the carried bytes ({doc['version']}): differing [{', '.join(h['differing'][:6]) or '-'}] "
                            f"missing [{', '.join(h['missing'][:6]) or '-'}] — a neighbouring version is never served silently"))
    problems = exports_check(name, environ)
    details["exports"] = problems
    if problems:
        when = "already imported without them" if details["already_imported"] else "export them before the first import"
        return Gate(name=gate, ok=False, details=details, reason=f"{name} exports: {'; '.join(problems)} ({when})")
    return Gate(name=gate, ok=True, details=details)
