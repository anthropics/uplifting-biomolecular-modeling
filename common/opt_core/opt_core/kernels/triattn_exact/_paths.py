"""triattn_exact._paths — the one place that knows the on-disk layout of the package (see LAYOUT.md).

Every shipped module locates files through these helpers; nothing else builds paths by hand.

    package directory        this directory (read-only at run time; nothing is ever written into it)
    kernel sources           csrc_dir(<route>, <file>)   default: the `csrc/` directory that is a SIBLING of the package
                                                         directory (i.e. <parent>/csrc next to <parent>/triattn_exact);
                                                         override: TRIATTN_EXACT_CSRC=<dir>
    proven-cell table        cells_path()                default: <parent>/CELLS.json; override: TRIATTN_EXACT_CELLS=<file>
    prebuilt kernels         prebuilt_dir()              <package>/_prebuilt (manifest.json + blobs/), always inside the package
    cache / build output     cache_dir(<sub>)            $TRIATTN_EXACT_CACHE/<sub> (created on demand).  The prebuilt path never
                                                         writes; only the nvcc JIT (development path) does.  When the variable is
                                                         unset a per-user temporary directory is used (development default).
    route-source fingerprint route_source_files(...)     logical names 'csrc/<...>' and 'triattn_exact/<...>' independent of where
                                                         the two directories actually live, so a source fingerprint certified in one
                                                         layout verifies unchanged in another.
"""
from __future__ import annotations

import os
import sys
import tempfile

PKG_NAME = "triattn_exact"
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(PKG_DIR)

ENV_CSRC = "TRIATTN_EXACT_CSRC"
ENV_CELLS = "TRIATTN_EXACT_CELLS"
ENV_CACHE = "TRIATTN_EXACT_CACHE"


def _refused(reason, cell=None):
    from . import Refused          # late import: __init__ imports nothing from here, but keep the edge one-way anyway
    return Refused(reason, cell or {})


def pkg_dir(*sub) -> str:
    return os.path.join(PKG_DIR, *sub)


def csrc_root() -> str:
    return os.environ.get("TRIATTN_EXACT_CSRC") or os.path.join(PARENT_DIR, "csrc")


def csrc_dir(*sub) -> str:
    """Path under the kernel-source root (existence is not checked here; see kernel_source())."""
    return os.path.join(csrc_root(), *sub)


def kernel_source(route: str, filename: str) -> str:
    """Absolute path of one kernel translation unit; Refused by name when the layout is broken."""
    p = csrc_dir(route, filename)
    if not os.path.isfile(p):
        raise _refused(f"kernel source {route}/{filename} not found under {csrc_root()} (layout: csrc/ next to the package directory, "
                       f"or set {ENV_CSRC})", {"route": route})
    return p


def cells_path() -> str:
    return os.environ.get("TRIATTN_EXACT_CELLS") or os.path.join(PARENT_DIR, "CELLS.json")


def prebuilt_dir(*sub) -> str:
    return os.path.join(PKG_DIR, "_prebuilt", *sub)


_CACHE_DIRS: dict = {}


def _private_cache_dir(path: str, tag: str, own_only: bool = False) -> str:
    """``path`` (created 0700 when absent) when nothing another account could have written would be loaded from it; else — one stderr line:
    what, why, the fix — a fresh directory private to this process, where its build products are compiled again. Refused: a directory (or,
    when group or other can enter it, a file in it) that is writable by group or other, or whose owner is neither this uid nor root — uid 0
    accepts any owner (a container's root reading a bind-mounted host directory). ``own_only`` (the per-user default under the shared
    temporary directory): this uid alone, and never a symbolic link. No digest kept beside a file would add to this: whoever can write the
    directory can rewrite the digest, so the owner / mode rule is the check."""
    import stat
    if path in _CACHE_DIRS:
        return _CACHE_DIRS[path]
    os.makedirs(path, mode=0o700, exist_ok=True)
    uid, why = os.geteuid(), None
    names = [""] + (sorted(os.listdir(path)) if os.stat(path).st_mode & 0o011 else [])
    for name in names:
        p = os.path.join(path, name) if name else path
        st = os.lstat(p) if own_only else os.stat(p)
        if own_only and not name and (os.path.islink(p) or st.st_uid != uid):
            why = f"{p} is a symbolic link or belongs to uid {st.st_uid}, not to this process (uid {uid}); fix: remove it, or name a directory of your own"
        elif st.st_mode & 0o022 and not stat.S_ISLNK(st.st_mode):     # a link's own mode says nothing (os.stat above already followed it unless own_only)
            why = f"{p} is writable by group or other (mode {st.st_mode & 0o7777:04o}); fix: chmod go-w {p}"
        elif uid != 0 and st.st_uid not in (uid, 0):
            why = f"{p} belongs to uid {st.st_uid}, not to this process (uid {uid}) or root; fix: name a cache directory of your own, or chown {p}"
        if why is not None:
            break
    out = path
    if why is not None:
        out = tempfile.mkdtemp(prefix=tag + "-")
        print(f"[opt_core] {tag}: REFUSED cache directory {path}: {why} — nothing in it is loaded; this process compiles again, privately, in {out}",
              file=sys.stderr, flush=True)
    _CACHE_DIRS[path] = out
    return out


def cache_root() -> str:
    """$TRIATTN_EXACT_CACHE, else a per-user temporary directory (development default; production integrators set the variable —
    the JIT development path is the only writer, the prebuilt path never writes)."""
    root = os.environ.get("TRIATTN_EXACT_CACHE")
    if root:
        return root
    return os.path.join(tempfile.gettempdir(), f"triattn_exact_cache-uid{os.getuid()}")   # per user: the temporary directory is shared


def cache_dir(*sub) -> str:
    """Writable directory for build products: <cache_root>/<sub...>, created on demand (0700).  The package directory is never used; a
    directory another account could have written is refused by name and a private one is returned instead (_private_cache_dir)."""
    path = os.path.join(cache_root(), *sub)
    real_pkg = os.path.realpath(PKG_DIR)
    if os.path.commonpath([os.path.realpath(path), real_pkg]) == real_pkg:
        raise _refused(f"{ENV_CACHE}={cache_root()} points inside the package directory; the package is read-only at run time", {})
    try:                                                       # shared libraries are loaded from here as found: never from a directory another account could write
        own = not os.environ.get(ENV_CACHE)
        root = _private_cache_dir(cache_root(), "triattn_exact_cache", own_only=own)
        path = _private_cache_dir(os.path.join(root, *sub), "triattn_exact_cache", own_only=own) if sub else root
    except OSError as e:
        raise _refused(f"cache directory {path} is not writable ({type(e).__name__}: {e}); set {ENV_CACHE} to a writable root", {})
    return path


# ---- logical layout for route-source fingerprints -----------------------------------------------------------------------------
def resolve_logical(logical_dir: str) -> str:
    """'csrc/<x>' -> <csrc root>/<x>; 'triattn_exact/<x>' -> <package dir>/<x>.  Other prefixes are refused."""
    head, _, rest = logical_dir.partition("/")
    if head == "csrc":
        return csrc_dir(*rest.split("/")) if rest else csrc_root()
    if head == PKG_NAME:
        return pkg_dir(*rest.split("/")) if rest else PKG_DIR
    raise _refused(f"unknown logical directory {logical_dir!r} (expected csrc/... or {PKG_NAME}/...)", {})


def route_source_files(logical_dirs, exclude_subdirs=(), extensions=()):
    """-> sorted [(logical relative path, absolute path)] of the files under the given logical directories.
    The logical path is what enters a fingerprint, so the digest does not depend on where csrc/ or the package live."""
    out = []
    excl = set(exclude_subdirs)
    for ld in logical_dirs:
        root = resolve_logical(ld)
        if not os.path.isdir(root):
            continue
        for r, ds, fns in os.walk(root):
            ds[:] = sorted(x for x in ds if x not in excl)
            for fn in sorted(fns):
                if extensions and not fn.endswith(tuple(extensions)):
                    continue
                rel_inside = os.path.relpath(os.path.join(r, fn), root)
                logical = ld.rstrip("/") + "/" + rel_inside.replace(os.sep, "/")
                out.append((logical, os.path.join(r, fn)))
    out.sort(key=lambda t: t[0])
    return out


__all__ = ["PKG_DIR", "PARENT_DIR", "pkg_dir", "csrc_root", "csrc_dir", "kernel_source", "cells_path", "prebuilt_dir", "cache_root",
           "cache_dir", "resolve_logical", "route_source_files"]
