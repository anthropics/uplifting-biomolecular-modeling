"""Prebuilt, fingerprinted route kernels (cubins) launched through the CUDA driver API: no nvcc, no torch C++ ABI.

Layout (read-only at run time; nothing is ever written next to these files):
    manifest.json           one entry per (route, source fingerprint, arch, toolkit): cubin sha256, kernel entry names and
                            template parameters, launch-config table, register counts, compiler identity and flags
    blobs/*.cubin.xz        the cubins, xz-compressed; identity = sha256 of the decompressed cubin, re-verified at load
    driver.py               libcuda.so.1 through ctypes (or cuda-python): module load, function attributes, occupancy,
                            tensor-map encoding, launches on torch's current stream
    <route>.py              per-route launcher with the route's attention()/supports() contract, reusing the route's
                            Python-side input checks, staging and configuration policy by import

Selecting a build for (route, device), see select_build():
    arch     the device's compute capability must equal the cubin's SM (sm_80 <-> 8.0; sm_90 and sm_90a <-> 9.0); no family fallback.
    driver   the CUDA driver must be at least the cubin's toolkit version; among loadable builds the toolkit whose major version
             matches the caller's preference (the CUDA major torch was built with) wins, then the newest;
             TRIATTN_EXACT_PREBUILT_TOOLKIT=12.6|13.0 pins one.
    source   without a certified sha set the build's source fingerprint must equal the fingerprint of the route source in this
             tree (same algorithm as the face's route fingerprint), i.e. the cubin is exactly what a JIT build here would
             produce and the imported Python marshalling matches the kernel ABI.  A caller holding certified cubin sha256
             values (CELLS binary fingerprints) passes them instead.  TRIATTN_EXACT_PREBUILT_SOURCE=<prefix>|any overrides.
Every problem (missing or corrupt manifest/blob, fingerprint mismatch, no build for the device) is a typed triattn_exact.Refused.
Scratch space, when a tool needs one, lives under cache_dir() = $TRIATTN_EXACT_CACHE or a per-user temporary directory.
"""
from __future__ import annotations

import hashlib
import json
import lzma
import os
import tempfile
import threading

try:
    from .. import Refused
except ImportError:  # pragma: no cover  # hygiene: no-cuda (standalone use of the subpackage)
    class Refused(RuntimeError):
        def __init__(self, reason, cell=None):
            self.reason = reason; self.cell = cell or {}
            RuntimeError.__init__(self, f"triattn_exact refused: {reason} cell={self.cell}")
try:
    from .. import _paths                                     # package layout helpers (kernel-source location, cache root)
except ImportError:  # pragma: no cover  # hygiene: no-cuda (older trees without the layout module)
    _paths = None

_HERE = os.path.dirname(os.path.abspath(__file__))
_TREE = os.path.dirname(os.path.dirname(_HERE))               # directory holding triattn_exact/ and csrc/ side by side
MANIFEST_PATH = os.environ.get("TRIATTN_EXACT_PREBUILT_MANIFEST", os.path.join(_HERE, "manifest.json"))
_lock = threading.Lock()
_manifest_cache = {"mtime": None, "doc": None}
_blob_cache = {}
_fp_cache = {}

# Route-source fingerprint: the same algorithm as the face's route_fingerprint (kept in sync deliberately), computed here as
# well so that build selection works in trees whose face lacks it.
FP_ROUTE_DIRS = {
    "cuda_mma": ["csrc/cuda_mma", "triattn_exact/cuda_mma"], "fallback": ["triattn_exact/fallback"],
    "hopper": ["csrc/hopper", "triattn_exact/hopper"], "tk": ["csrc/tk", "triattn_exact/tk"], "smalls": ["csrc/smalls", "triattn_exact/smalls"],
    "breadth": ["csrc/cuda_mma/variants/breadth", "triattn_exact/breadth"],
}
FP_EXCLUDE = {"variants", "tests", "microtests", "__pycache__"}
FP_EXT = (".cu", ".cuh", ".h", ".hpp", ".py", ".json", ".ptx")


def cache_dir(sub=None):
    """Writable scratch root: $TRIATTN_EXACT_CACHE, else a per-user temporary directory.  Created on demand (0700); never inside the package;
    a directory another account could have written is refused by name and a private one is returned in its place (the rule of _paths.cache_dir)."""
    if _paths is not None:
        return _paths.cache_dir(sub) if sub else _paths.cache_dir()
    root = os.environ.get("TRIATTN_EXACT_CACHE")
    named = bool(root)
    if not root:
        try:
            uid = str(os.getuid())
        except AttributeError:  # pragma: no cover
            uid = "user"
        root = os.path.join(tempfile.gettempdir(), "triattn_exact-" + uid)
    root = _private_dir(root, named)
    return _private_dir(os.path.join(root, sub), named) if sub else root


def _private_dir(path, named=False):
    """``path`` made on demand with mode 0700 and returned when this process may trust what is in it: not writable by group or other and owned
    by this account (or by root, for a directory named through $TRIATTN_EXACT_CACHE); the per-user default must also not be a symbolic link.
    Anything else is refused by name on stderr -- nothing in it is read -- and a fresh private directory (tempfile.mkdtemp) is returned."""
    os.makedirs(path, mode=0o700, exist_ok=True)
    try:
        uid = os.getuid()
    except AttributeError:  # pragma: no cover
        return path
    st = os.lstat(path)
    why = None
    if not named and (os.path.islink(path) or st.st_uid != uid):
        why = "is a symbolic link or belongs to uid %d, not to this process (uid %d)" % (st.st_uid, uid)
    elif os.stat(path).st_mode & 0o022:
        why = "is writable by group or other (mode %04o)" % (os.stat(path).st_mode & 0o7777)
    elif os.stat(path).st_uid not in (uid, 0):
        why = "belongs to uid %d, not to this process (uid %d) or root" % (os.stat(path).st_uid, uid)
    if why is None:
        return path
    out = tempfile.mkdtemp(prefix="triattn_exact-")
    import sys  # noqa: PLC0415
    print("[opt_core] triattn_exact: REFUSED cache directory %s: %s; nothing in it is loaded, this process uses %s instead" % (path, why, out),
          file=sys.stderr, flush=True)
    return out


def source_fingerprint(route):
    """sha256 over the route's source files in this tree (the face's route_fingerprint value when available, else computed here)."""
    if route in _fp_cache:
        return _fp_cache[route]
    fp = None
    try:
        from ..face import route_fingerprint as _face_fp
        fp = _face_fp(route)
    except Exception:  # noqa: BLE001  # hygiene: no-cuda (face absent or older: local implementation below, pure file hashing)
        fp = None
    if fp is None and route in FP_ROUTE_DIRS:
        if _paths is not None:                                 # logical names, independent of where csrc/ and the package live
            files = _paths.route_source_files(FP_ROUTE_DIRS[route], FP_EXCLUDE, FP_EXT)
        else:
            files = []
            for d in FP_ROUTE_DIRS[route]:
                root = os.path.join(_TREE, d)
                if not os.path.isdir(root):
                    continue
                for r, ds, fns in os.walk(root):
                    ds[:] = sorted(x for x in ds if x not in FP_EXCLUDE)
                    for fn in sorted(fns):
                        if fn.endswith(FP_EXT):
                            files.append((os.path.relpath(os.path.join(r, fn), _TREE).replace(os.sep, "/"), os.path.join(r, fn)))
            files.sort(key=lambda t: t[0])
        if files:
            h = hashlib.sha256()
            for logical, path in files:
                with open(path, "rb") as f:
                    b = f.read()
                h.update(logical.encode() + b"\0" + hashlib.sha256(b).hexdigest().encode() + b"\n")
            fp = h.hexdigest()
    _fp_cache[route] = fp
    return fp


def manifest():
    """The parsed manifest (re-read when the file changes).  Missing file -> empty manifest; unreadable file -> Refused."""
    try:
        mt = os.stat(MANIFEST_PATH).st_mtime_ns
    except OSError:
        return {"schema": 1, "builds": []}
    if _manifest_cache["doc"] is None or _manifest_cache["mtime"] != mt:
        try:
            with open(MANIFEST_PATH) as f:
                doc = json.load(f)
        except (OSError, ValueError) as e:
            raise Refused(f"prebuilt manifest unreadable: {type(e).__name__}: {e}", {"manifest": os.path.basename(MANIFEST_PATH)})
        if not isinstance(doc, dict) or not isinstance(doc.get("builds", []), list):
            raise Refused("prebuilt manifest malformed (no 'builds' list)", {"manifest": os.path.basename(MANIFEST_PATH)})
        _manifest_cache["doc"] = doc; _manifest_cache["mtime"] = mt
    return _manifest_cache["doc"]


def builds(route=None):
    return [b for b in manifest().get("builds", []) if route is None or b.get("route") == route]


def cubin_bytes(build):
    """Decompress and verify (sha256, length) the build's cubin; cached in memory per process.  Any problem -> Refused."""
    key = build["cubin_sha256"]
    data = _blob_cache.get(key)
    if data is None:
        with _lock:
            data = _blob_cache.get(key)
            if data is None:
                path = os.path.join(os.path.dirname(MANIFEST_PATH), build["blob"])
                try:
                    with open(path, "rb") as f:
                        raw = f.read()
                    data = lzma.decompress(raw) if path.endswith(".xz") else raw
                except OSError as e:
                    raise Refused(f"prebuilt blob {build['blob']} not readable: {type(e).__name__}", {"route": build.get("route")})
                except lzma.LZMAError as e:
                    raise Refused(f"prebuilt blob {build['blob']} is corrupt: {e}", {"route": build.get("route")})
                have = hashlib.sha256(data).hexdigest()
                if have != key or len(data) != build.get("cubin_bytes", len(data)):
                    raise Refused(f"prebuilt blob {build['blob']} fails its fingerprint ({have[:16]} != {key[:16]})", {"route": build.get("route")})
                _blob_cache[key] = data
    return data


def arch_of_cc(cc):
    return cc[0] * 10 + cc[1]


def select_build(route, cc, driver_version, tree_fingerprint=None, allowed_shas=None, prefer_major=None, abi=None, toolkit=None, variant=None):
    """-> (build | None, reason).  `prefer_major`: CUDA major version whose toolkit builds are preferred (e.g. torch's);
    `abi` / `variant`: restrict to manifest entries with this abi string / compile-time variant name; `toolkit`: pin one
    toolkit ("12.6" | "13.0"), like the env knob."""
    cands = builds(route)
    if abi is not None:
        cands = [b for b in cands if b.get("abi") == abi]
    if variant is not None:
        cands = [b for b in cands if b.get("variant", "") == variant]
    if not cands:
        return None, f"no prebuilt builds for route '{route}'" + (f" abi {abi}" if abi else "") + (f" variant {variant}" if variant else "")
    sm = arch_of_cc(cc)
    c_arch = [b for b in cands if int(b["sm"]) == sm]
    if not c_arch:
        return None, f"no prebuilt cubin for sm_{sm}; built archs: {sorted({b['arch'] for b in cands})}"
    c_drv = [b for b in c_arch if b["toolkit"]["version_int"] <= driver_version]
    if not c_drv:
        need = min(b["toolkit"]["version_int"] for b in c_arch)
        return None, (f"CUDA driver {driver_version // 1000}.{(driver_version % 1000) // 10} is older than every prebuilt toolkit for sm_{sm} "
                      f"(oldest needs >= {need // 1000}.{(need % 1000) // 10})")
    pin_tk = str(toolkit) if toolkit is not None else os.environ.get("TRIATTN_EXACT_PREBUILT_TOOLKIT")
    if pin_tk:
        c_drv = [b for b in c_drv if b["toolkit"]["cuda"] == pin_tk]
        if not c_drv:
            return None, f"TRIATTN_EXACT_PREBUILT_TOOLKIT={pin_tk}: no such toolkit build for sm_{sm}"
    src_env = os.environ.get("TRIATTN_EXACT_PREBUILT_SOURCE")
    if allowed_shas is not None:
        c_src = [b for b in c_drv if b["cubin_sha256"] in allowed_shas]
        why = "none of the loadable cubins is in the cell's binary_fingerprint set"
    elif src_env == "any":
        c_src = list(c_drv); why = ""
    elif src_env:
        c_src = [b for b in c_drv if any(f.startswith(src_env) for f in [b["source_fingerprint"]] + list(b.get("source_fingerprints", ())))]
        why = f"TRIATTN_EXACT_PREBUILT_SOURCE={src_env} matches no build"
    else:
        c_src = [b for b in c_drv if tree_fingerprint is None or b["source_fingerprint"] == tree_fingerprint
                 or tree_fingerprint in b.get("source_fingerprints", ())]
        why = (f"route source in this tree ({(tree_fingerprint or '?')[:12]}) differs from every prebuilt build's source "
               f"({sorted({b['source_fingerprint'][:12] for b in c_drv})}); rebuild the cubins or use the matching release")
    if not c_src:
        return None, why

    def rank(b):
        major = b["toolkit"]["version_int"] // 1000
        plain_arch = 0 if b.get("arch_specific") else 1          # sm_90 (family-portable, the JIT's gencode) before sm_90a builds of the same source
        return (plain_arch, 1 if (prefer_major is not None and major == int(prefer_major)) else 0, b["toolkit"]["version_int"], b.get("blob", ""))
    c_src.sort(key=rank, reverse=True)
    return c_src[0], f"prebuilt {c_src[0]['blob']}"


def describe_builds():
    rows = []
    for b in builds():
        rows.append(f"{b['route']} {b['abi']} {b.get('variant', '')} src={b['source_fingerprint'][:12]} {b['arch']} cuda{b['toolkit']['cuda']} cubin={b['cubin_sha256'][:12]}")
    return "\n".join(rows)


__all__ = ["manifest", "builds", "cubin_bytes", "select_build", "describe_builds", "source_fingerprint", "cache_dir", "Refused"]
