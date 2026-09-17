"""kernels.trimul.native -- the unified triangle-multiplication kernel package ``trimul_native`` (arch-keyed CUDA cubins: a fused
LN_in / four-projection / gate / mask prologue writing channel-major bf16 planes | cuBLAS bmm | a fused LN_out / output-projection / gate
(+ residual) epilogue, loaded with ``cuModuleLoadData`` and launched through the CUDA driver on the framework's stream) carried SEALED under
``pkg/<ACTIVE_PKG>/`` and served through ONE face, whatever the package version.

    from opt_core.kernels.trimul import native as T
    T.ACTIVE_PKG                       # "v1": the payload directory this face serves (a version swap edits ACTIVE_PKG + DIGESTS and drops pkg/<new>/ in)
    T.admits((9, 0), "bf16", 256, 256, 800, "outgoing")                     # None = served | the refusal word (pure: json only, no torch)
    rep = T.install()                  # DIGESTS + SHA256SUMS -> the package's own check(): driver binding -> arch -> manifest-checked cubin loads
                                       #   -> load check (registers / local bytes == the ptxas record) -> byte gate (its sealed vectors, bitwise); dict
    wp = T.pack(w10, cache)            # the package's weight pack from the ten canonical tensors (cached in the caller's dict by tensor identity)
    out = T.forward(z, mask, direction="outgoing", weights=w10, residual=False, cache=cache, variant="fast")   # update | z + update

Contract (the provider's sealed-package interface, as kernels.trimul.esm_v61 and kernels.triattn.triattn_native):

* payload = ``pkg/<ver>/`` = the producer's sealed tree byte for byte: ``VERSION``, ``python/trimul_native/`` (face, launch, driver binding,
  op assembly, tile tables, vectors), ``csrc/`` (the sources the cubins were built from), ``build/manifest.json`` + ``build/<arch>/<unit>.cubin``
  (+ ptxas logs), ``testvectors/`` (closed-form inputs, expected bytes per device class, witnesses), ``CELLS.json`` (its measured rows),
  ``SHA256SUMS``, ``README.md``, ``INTEGRATION.md``.  Nothing in it is edited, restated or rebuilt here; a byte that differs from the record is
  a refusal BY NAME (``digest:<relpath>``), never a rebuild.
* ``DIGESTS`` (this file) pins the package's own digest list (``SHA256SUMS``) and its three records (``VERSION``, ``build/manifest.json``,
  ``testvectors/manifest.json``); ``verify_digests()`` checks those four, then every line of ``SHA256SUMS`` against the payload, then that no
  file outside the list is present under the payload's recorded directories.
* the package's python is imported BY PATH under a PRIVATE module name (``_MODNAME``) -- never as ``trimul_native`` from ``sys.path``: an
  engine image or a development tree may carry its own copy under that name and the two never meet (separate module objects, separate cubin
  handles, separate caches).  While the private import runs, the package's location variables (``TRIMUL_NATIVE_BUILD_DIR``,
  ``TRIMUL_NATIVE_CELLS``) are masked so the carried face binds to the carried build / cells and nothing else.
* keyed by ARCHITECTURE, not by the framework ABI: the cubins hold SASS for ``sm_90a`` (and ``sm_80`` when the payload's vectors carry that
  device class) and load through the driver, so one payload serves every torch / CPython it is loaded under; "built per stack" for this row means
  ``install()`` checked per stack (load + load check + byte gate), recorded in ``TRIMUL_CELLS.json`` ``rows.native.verified_on``.
* imports at module level: standard library only.  ``install`` / ``pack`` / ``forward`` import torch (and the driver binding: the
  ``cuda.bindings`` wheel when the image has it, else ctypes over ``libcuda.so.1``) when called.

Refusals are ``Unavailable`` (``.kind`` = the word, ``.detail`` free text): ``digest:<relpath>`` | ``payload:<what>`` | ``import:<what>`` and
the package's own words passed through unchanged (``driver_unavailable:<why>`` | ``cc_unsupported:<cc>`` | ``no_cubin:<arch>`` |
``manifest:<what>`` | ``load:<driver error>`` | ``loadcheck_failed:<unit.kernel | case>`` | ``vectors:<what>`` | ``shape:`` / ``dtype:`` /
``direction:`` / ``weights:`` / ``variant:<what>`` | ``no_cell:<key>``).
"""
import collections
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import threading

__all__ = ["ACTIVE_PKG", "VERSION", "DIGESTS", "ARCHS", "WIDTHS", "Unavailable", "pkg_dir", "verify_digests", "manifest", "vectors_manifest",
           "device_classes", "face", "module", "admits", "install", "report", "pack", "forward", "notices", "payload_cache", "shared_geometries", "workspace_bytes", "SHARED_GEOMETRIES", "served", "STAMP_ENV", "stamp_dir", "stamp_dirs", "stamp_facts", "stamp_read", "stamp_write", "payload_version"]

ACTIVE_PKG = "v5"                                   # pkg/<ACTIVE_PKG>/ is served; earlier payload directories may stay on disk unreferenced
VERSION = "v5 1.2.2"                             # the payload's VERSION line as sealed (the record; install() re-reads the payload's own)
_MODNAME = "opt_core_trimul_native_" + ACTIVE_PKG.replace(".", "_")          # the private top-level module name the payload's python binds under
ARCHS = ("sm_90a", "sm_80")                         # architectures the package knows; served = those whose device class the payload's vectors carry
WIDTHS = (64, 128, 256, 384)                        # the package's width vocabulary (c_z, c_hidden); served pairs = its build manifest's `serves` records
CC_OF_ARCH = {"sm_90a": ("9.0",), "sm_80": ("8.0", "8.6", "8.7", "8.9")}
DIGESTS = {                                         # sha256 of the payload's digest list and records, as sealed (a version swap rewrites these four lines)
    "SHA256SUMS": "d3e9ed452dcfb8df8f10b5eb4c396a95e1a86e9566c48a7addc1fd4b60143f54",
    "VERSION": "1876bbaa5910f058ef6e383ddf2f0226a7323190f1da1c440a9defabad3024bb",
    "build/manifest.json": "4d554269bf60d0f09832abbcdb0b32661a3ef30d3ecf4fa85cfcfe8f7c7b6a84",
    "testvectors/manifest.json": "02b55a4c3aec1393eb1fa71aefc00013123048455eb00598afadc541e5c399a2",
}
LISTED_DIRS = ("python", "csrc", "build", "testvectors", "tests")            # payload directories whose every file must be a SHA256SUMS line
_LOCK = threading.RLock()
_STATE = {"face": None, "digests": None, "report": {}, "classes": None}
STAMP_ENV = "OPT_CORE_VERDICT_DIR"                  # stamp directory override; "0" / "off" = no stamp (the byte gate runs in every process)
STAMP_SCHEMA = "opt_core.trimul_native.gate_stamp/v1"


class Unavailable(RuntimeError):
    """Raised BY NAME when the payload cannot be served here (``.kind`` is the word; ``.detail`` free text)."""

    def __init__(self, kind, detail=""):
        RuntimeError.__init__(self, "kernels.trimul.native[%s]: %s%s" % (ACTIVE_PKG, kind, (" (" + str(detail)[:300] + ")") if detail else ""))
        self.kind = kind
        self.detail = detail


# ----------------------------------------------------------------------------------------------------------------------------- payload (pure)
def pkg_dir(ver=None):
    """Absolute path of pkg/<ver>/ (default ACTIVE_PKG)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "pkg", ver or ACTIVE_PKG)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sums(root):
    """[(relpath, sha256)] from the payload's SHA256SUMS ('<hex>  <relpath>' lines)."""
    out = []
    p = os.path.join(root, "SHA256SUMS")
    if not os.path.isfile(p):
        raise Unavailable("payload:no_SHA256SUMS", p)
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            hexd, _, rel = line.partition("  ")
            if len(hexd) != 64 or not rel:
                raise Unavailable("payload:SHA256SUMS_line", line[:80])
            out.append((rel.strip(), hexd))
    return out


def verify_digests(ver=None):
    """The four DIGESTS records, then every SHA256SUMS line, then 'no unlisted file under the listed directories'.  Returns {relpath: sha256}
    of everything checked; raises Unavailable('digest:<rel>' | 'payload:<what>') by name.  Cached per process for ACTIVE_PKG."""
    root = pkg_dir(ver)
    with _LOCK:
        if ver in (None, ACTIVE_PKG) and _STATE["digests"] is not None:
            return _STATE["digests"]
    if not os.path.isdir(root):
        raise Unavailable("payload:absent", root)
    for rel, want in DIGESTS.items():
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            raise Unavailable("payload:missing:%s" % rel)
        if want is not None and _sha256(p) != want:
            raise Unavailable("digest:%s" % rel)
    listed = _sums(root)
    seen = {}
    for rel, want in listed:
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            raise Unavailable("payload:missing:%s" % rel)
        got = _sha256(p)
        if got != want:
            raise Unavailable("digest:%s" % rel)
        seen[rel] = got
    for d in LISTED_DIRS:
        base = os.path.join(root, d)
        if not os.path.isdir(base):
            continue
        for dp, _dn, fns in os.walk(base):
            if "__pycache__" in dp.split(os.sep):
                continue
            for fn in fns:
                if fn.endswith((".pyc", ".pyo")):
                    continue
                rel = os.path.relpath(os.path.join(dp, fn), root).replace(os.sep, "/")
                if rel not in seen:
                    raise Unavailable("payload:unlisted:%s" % rel)
    seen["SHA256SUMS"] = _sha256(os.path.join(root, "SHA256SUMS"))
    with _LOCK:
        if ver in (None, ACTIVE_PKG):
            _STATE["digests"] = seen
    return seen


def manifest(ver=None):
    """The payload's build record (build/manifest.json) parsed."""
    with open(os.path.join(pkg_dir(ver), "build", "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)


def vectors_manifest(ver=None):
    """The payload's test-vector record (testvectors/manifest.json) parsed ({} when absent)."""
    try:
        with open(os.path.join(pkg_dir(ver), "testvectors", "manifest.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def device_classes(ver=None):
    """The device classes ('sm90', 'sm80') the payload's vectors carry expected bytes for = the architectures this row can byte-gate, hence serve."""
    with _LOCK:
        if ver in (None, ACTIVE_PKG) and _STATE["classes"] is not None:
            return _STATE["classes"]
    t = vectors_manifest(ver)
    cl = set((t.get("classes") or {}).keys())
    for case in t.get("cases") or []:
        cl.update((case.get("expected") or {}).keys())
    out = tuple(sorted(cl))
    with _LOCK:
        if ver in (None, ACTIVE_PKG):
            _STATE["classes"] = out
    return out


def payload_version(ver=None):
    try:
        with open(os.path.join(pkg_dir(ver), "VERSION"), encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return None


def served(ver=None):
    """['<arch> z<c_z> h<c_hidden> <bf16|f32z> <fast(+exact)>', ...] from the build manifest's `serves` records, restricted to the architectures
    whose device class the vectors carry (pure)."""
    F = face()
    cls = device_classes(ver)
    out = []
    for (arch, cz, ch, form), v in sorted(F.served_table(os.path.join(pkg_dir(ver), "build")).items()):
        if F.DEVICE_CLASS.get(arch) in cls:
            out.append("%s z%d h%d %s %s" % (arch, cz, ch, "f32z" if form == "f" else "bf16", "+".join(v["variants"])))
    return out


# ----------------------------------------------------------------------------------------------------------------------------- the carried face
def face():
    """The payload's ``trimul_native.face`` module, imported by path under the private name ``_MODNAME`` (pure at import: json / os / threading).
    Raises Unavailable('import:<what>' | 'payload:absent')."""
    with _LOCK:
        if _STATE["face"] is not None:
            return _STATE["face"]
        root = pkg_dir()
        pkg_py = os.path.join(root, "python", "trimul_native")
        init = os.path.join(pkg_py, "__init__.py")
        if not os.path.isfile(init):
            raise Unavailable("payload:absent", init)
        masked = {k: os.environ.pop(k) for k in ("TRIMUL_NATIVE_BUILD_DIR", "TRIMUL_NATIVE_CELLS") if k in os.environ}
        try:
            top = sys.modules.get(_MODNAME)
            if top is None or os.path.dirname(os.path.abspath(getattr(top, "__file__", "") or "")) != os.path.abspath(pkg_py):
                spec = importlib.util.spec_from_file_location(_MODNAME, init, submodule_search_locations=[pkg_py])
                top = importlib.util.module_from_spec(spec)
                sys.modules[_MODNAME] = top
                try:
                    spec.loader.exec_module(top)
                except Exception as e:                                       # noqa: BLE001 -- a payload whose python does not import here refuses by name
                    sys.modules.pop(_MODNAME, None)
                    raise Unavailable("import:%s" % type(e).__name__, str(e))
            try:
                F = importlib.import_module(_MODNAME + ".face")
            except Exception as e:                                           # noqa: BLE001
                raise Unavailable("import:face:%s" % type(e).__name__, str(e))
            want_build = os.path.join(root, "build")
            if os.path.abspath(F.BUILD_DIR) != os.path.abspath(want_build) or os.path.abspath(F.CELLS_PATH) != os.path.abspath(os.path.join(root, "CELLS.json")):
                raise Unavailable("import:face_bound_elsewhere", "%s | %s" % (F.BUILD_DIR, F.CELLS_PATH))
            _memoise_check(F)                                                # erratum-2 shim (payload 1.0.1): see _memoise_check; removed when a payload memoises its own check()
            _memoise_manifest(F)                                             # erratum-3 shim (payload 1.0.1): serve() -> admits() -> release_archs() re-parsed build/manifest.json per CALL
            _STATE["face"] = F
            return F
        finally:
            os.environ.update(masked)


SHARED_GEOMETRIES = 2                                                        # payload caches kept per device: the most recent (N, c_z, c_hidden) geometries (workspaces + plans + packs)
_SHARED = collections.OrderedDict()                                          # (device index, N, c_z, c_hidden) -> the payload's serve cache for that geometry (LRU of SHARED_GEOMETRIES per device)


def workspace_bytes(n_tokens, c_hidden):
    """The payload's persistent per-geometry workspace: a|b planes [2 c_hidden, Np, Np] + contraction planes [c_hidden, Np, Np], bf16, Np = ceil16(N)."""
    np_ = -(-int(n_tokens) // 16) * 16
    return 3 * int(c_hidden) * np_ * np_ * 2


def payload_cache(device_index, n_tokens, c_z, c_hidden):
    """WORKSPACE RESIDENCY POLICY (this adapter's; the payload keeps its a|b and contraction planes, launch plans, tensor maps and weight packs in
    whatever dict the caller hands it).  Kit faces hand the provider ONE dict PER PATCHED MODULE (opt_core.trimul.by_word), so passing that
    through would park 3*c_hidden*Np^2 bf16 bytes of planes in EVERY layer (0.75 GiB per module at N=1024, c 128).  Instead every served call of
    one geometry (device, N, c_z, c_hidden) shares ONE payload cache held here: one workspace set per live geometry (layers run stream-ordered on
    the caller's current stream, so sharing is race-free for single-stream callers; CUDA-graph capture sees stable addresses), weight packs of
    all layers coexist in it (keyed by tensor address), and at most SHARED_GEOMETRIES geometries per device are kept (LRU; an evicted geometry's
    dict is dropped whole and its buffers return to the framework's caching allocator).  ``config={'cache': 'caller'}`` opts one caller out."""
    key = (device_index, int(n_tokens), int(c_z), int(c_hidden))
    with _LOCK:
        d = _SHARED.get(key)
        if d is not None:
            _SHARED.move_to_end(key)
            return d
        d = _SHARED[key] = {}
        mine = [k for k in _SHARED if k[0] == device_index]
        for k in mine[:-SHARED_GEOMETRIES]:
            _SHARED.pop(k, None)
        return d


PACK_POLICY_ENV = "OPT_CORE_TRIMUL_PACK_CACHE"                                # per_layer | lru1 : forces the weight-pack residency policy for every tier word (kits' A/B knob)
PACK_KEY_TAG = "trimul_native.pack"                                          # the payload's pack-cache key tag: (tag, arch, device, (ptr, shape, dtype) x 10)
CALLS_MEMO_KEY = "trimul_native.calls"                                       # the payload's per-call plan memo (entries hold a reference to their layer's pack)
_PACK_STATS = {"evicted": 0}


def pack_policy_for(word=None, policy=None):
    """The weight-pack residency policy of a served call: 'lru1' under the tier word big (the shared per-geometry cache keeps at most ONE
    layer's packed weights: a trunk's resident bytes no longer grow with its layer count), 'per_layer' under fast / exact / a row word (every
    layer's pack stays cached: no re-pack per call); OPT_CORE_TRIMUL_PACK_CACHE=per_layer|lru1 forces either for every word; an explicit
    ``policy`` argument wins over the word (not over the env knob)."""
    env = (os.environ.get(PACK_POLICY_ENV) or "").strip().lower()
    if env in ("per_layer", "lru1"):
        return env
    if policy in ("per_layer", "lru1"):
        return policy
    return "lru1" if str(word or "") == "big" else "per_layer"


def _nbytes(x, _depth=0):
    """Bytes held by a pack entry (tensors: numel * element_size; objects with .nbytes; containers recursively)."""
    if x is None or _depth > 4:
        return 0
    if hasattr(x, "numel") and hasattr(x, "element_size"):
        try:
            return int(x.numel()) * int(x.element_size())
        except (TypeError, ValueError, AttributeError, RuntimeError):        # a tensor-like without sizes: not counted
            return 0
    if hasattr(x, "nbytes") and not isinstance(x, (bytes, bytearray, str)):
        try:
            return int(x.nbytes)
        except (TypeError, ValueError, AttributeError):
            return 0
    if isinstance(x, dict):
        return sum(_nbytes(v, _depth + 1) for v in x.values())
    if isinstance(x, (list, tuple, set)):
        return sum(_nbytes(v, _depth + 1) for v in x)
    return 0


def _is_pack_key(k):
    return isinstance(k, tuple) and len(k) >= 1 and k[0] == PACK_KEY_TAG


def _weights_sig(weights):
    """((ptr, shape, dtype) x 10) of the ten canonical tensors -- the identity part of the payload's pack key."""
    ts = [_wget(weights, k) for k in ("ln_in_w", "ln_in_b", "w_ag", "w_ap", "w_bg", "w_bp", "ln_out_w", "ln_out_b", "w_o", "w_og")]
    return tuple((int(t.data_ptr()), tuple(t.shape), str(t.dtype)) for t in ts if t is not None)


PACK_CACHE_CAP_ENV = "OPT_CORE_TRIMUL_PACK_CACHE_CAP"                          # hard cap on pack-class entries per shared cache under every word (default PACK_CACHE_CAP)
PACK_CACHE_CAP = 512
PLAIN_KEY_TAG = "trimul_native.plain_weights"                                # the payload's cc-8.0 member: plain (cast) weight copies, address-keyed like the packs
_PACK_STATS.setdefault("gc_evicted", 0); _PACK_STATS.setdefault("cap_evicted", 0)


def _is_packclass_key(k):
    return isinstance(k, tuple) and len(k) >= 1 and k[0] in (PACK_KEY_TAG, PLAIN_KEY_TAG)


def pack_cache_cap():
    try:
        v = int(os.environ.get(PACK_CACHE_CAP_ENV) or PACK_CACHE_CAP)
    except (TypeError, ValueError):
        v = PACK_CACHE_CAP
    return max(1, v)


def _packclass_keys(cache):
    return [k for k in list((cache or {}).keys()) if _is_packclass_key(k)]


def _evict_key(cache, key, w_ptr=None, why="gc"):
    """Drop one pack-class entry (and the per-call memo entries of that weight address) from ``cache``; counts in _PACK_STATS."""
    if cache.pop(key, None) is not None:
        _PACK_STATS["%s_evicted" % why] = _PACK_STATS.get("%s_evicted" % why, 0) + 1
    calls = cache.get(CALLS_MEMO_KEY)
    if isinstance(calls, dict) and w_ptr is not None:
        for fk in [fk for fk, ent in list(calls.items()) if isinstance(ent, dict) and ent.get("w_ptr") == w_ptr]:
            calls.pop(fk, None)


def after_call(cache, weights, keys_before=()):
    """After a served call: (1) every pack-class entry the payload just created for ``weights`` is tied to the OWNER tensor's lifetime -- a
    weakref finalizer on the caller's w_ap (and ln_in_w) pops the entry when that tensor is garbage-collected, so per-call FRESH weight copies
    (a caller casting its parameters anew each call) cannot pile up; (2) a bounded LRU under every word: pack-class entries beyond
    pack_cache_cap() leave oldest-first (dict order).  Stable weights: one pack per layer stays (no re-pack per call).  Never touches the
    module's own parameters -- only the dict entries of packed / cast copies."""
    if not cache:
        return 0
    import weakref
    before = set(keys_before or ())
    new_keys = [k for k in _packclass_keys(cache) if k not in before]
    if new_keys:
        wp = _wget(weights, "w_ap"); w_ptr = None
        try:
            w_ptr = int(wp.data_ptr()) if wp is not None else None
        except (TypeError, ValueError, AttributeError, RuntimeError):
            w_ptr = None
        for k in new_keys:
            for owner in (wp, _wget(weights, "ln_in_w")):
                if owner is None:
                    continue
                try:
                    weakref.finalize(owner, _evict_key, cache, k, w_ptr, "gc")
                except TypeError:                                            # an owner that cannot be weak-referenced: the LRU cap still bounds the dict
                    pass
                break
    n = 0
    keys = _packclass_keys(cache); cap = pack_cache_cap()
    while len(keys) > cap:
        _evict_key(cache, keys.pop(0), None, "cap"); n += 1
    return n


def pack_cache_stats(cache):
    """{'packs': n, 'bytes': total} of the weight packs resident in ``cache`` (the cache's own bookkeeping: pack entries by key tag)."""
    packs = [(k, v) for k, v in (cache or {}).items() if _is_packclass_key(k)]
    return {"packs": len(packs), "bytes": sum(_nbytes(v) for _k, v in packs), "evicted_total": _PACK_STATS["evicted"],
            "gc_evicted": _PACK_STATS.get("gc_evicted", 0), "cap_evicted": _PACK_STATS.get("cap_evicted", 0), "cap": pack_cache_cap()}


def apply_pack_policy(cache, weights, policy):
    """Enforce the residency policy on ``cache`` BEFORE a call with ``weights``: under 'lru1' every OTHER layer's pack entry leaves the dict
    (and the payload's per-call memo entries that reference another layer's pack), so after the call at most this layer's pack is resident;
    'per_layer' touches nothing.  Never frees or mutates the module's own parameters (only the packed copies' dict entries).  Returns the
    number of pack entries evicted."""
    if policy != "lru1" or not cache:
        return 0
    sig = _weights_sig(weights)
    n = 0
    for k in [k for k in list(cache.keys()) if _is_packclass_key(k)]:
        trip = tuple(x for x in k if isinstance(x, tuple) and len(x) == 3)
        quad = tuple((x[1], x[3], x[2]) for x in k if isinstance(x, tuple) and len(x) == 4)   # plain-weight keys: (name, ptr, dtype, shape) -> (ptr, shape, dtype)
        if (trip or quad) and not (set(trip or quad) <= set(sig)):          # another layer's pack / plain copy (its (ptr, shape, dtype) triples differ)
            cache.pop(k, None); n += 1
    calls = cache.get(CALLS_MEMO_KEY)
    if isinstance(calls, dict) and n:
        wp = _wget(weights, "w_ap"); cur = int(wp.data_ptr()) if wp is not None else None
        for fk in [fk for fk, ent in list(calls.items()) if isinstance(ent, dict) and ent.get("w_ptr") != cur]:
            calls.pop(fk, None)
    _PACK_STATS["evicted"] += n
    return n


def shared_geometries():
    """[(device, N, c_z, c_hidden, workspace_bytes)] currently held (newest last)."""
    with _LOCK:
        return [(k[0], k[1], k[2], k[3], workspace_bytes(k[1], k[3])) for k in _SHARED]


def _device_key(device):
    """One key for 'the current device' however a caller names it (None | int | 'cuda:N' | torch.device)."""
    if device is None:
        try:
            import torch
            device = torch.cuda.current_device() if torch.cuda.is_available() else None
        except Exception:                                                    # noqa: BLE001 -- no torch / no device: the key 'None' (check() itself refuses by name there)
            device = None
    d = str(device)
    return d[5:] if d.startswith("cuda:") else d


def _memoise_manifest(F):
    """ERRATUM-3 SHIM for payload 1.0.1: `face.serve()` calls the pure `admits()` per call, which calls `release_archs()`, which re-reads and
    re-parses build/manifest.json (`_read_manifest`) and stats every unit's cubin on EVERY call -- ~0.5-1 ms of host time per served call (a
    floor larger than the kernel below N ~ 512).  The manifest is a SEALED record whose digest install() checked: memoise `_read_manifest` and
    `release_archs` per build directory (runtime attribute assignment in this adapter; no sealed file edited).  Removed when a payload caches
    its own manifest reads."""
    if getattr(F, "_read_manifest_unmemoised", None) is not None:
        return F
    orig_read, orig_archs = F._read_manifest, F.release_archs
    reads, archs = {}, {}
    lock = threading.Lock()

    def _read_manifest(build_dir=None):
        key = os.path.abspath(build_dir or F.BUILD_DIR)
        with lock:
            if key in reads:
                return reads[key]
        man = orig_read(build_dir)
        if man is not None:                                                  # an unreadable manifest is not cached (the original returns None; a later call retries)
            with lock:
                reads[key] = man
        return man

    def release_archs(build_dir=None):
        key = os.path.abspath(build_dir or F.BUILD_DIR)
        with lock:
            if key in archs:
                return set(archs[key])
        out = orig_archs(build_dir)
        if out:
            with lock:
                archs[key] = frozenset(out)
        return set(out)

    _read_manifest.__wrapped__ = orig_read
    release_archs.__wrapped__ = orig_archs
    release_archs.__doc__ = (orig_archs.__doc__ or "") + "  [opt_core erratum-3 shim: memoised per build directory.]"
    F._read_manifest_unmemoised, F._release_archs_unmemoised = orig_read, orig_archs
    F._read_manifest, F.release_archs = _read_manifest, release_archs
    return F


def _memoise_check(F):
    """ERRATUM-2 SHIM for payload 1.0.1: its face.check() means to memoise its report per (device, probe, gate, cases) but a shadowed loop variable
    stores it under another key, so face.serve() -- which calls check(idx, gate=...) for every NEW call key (weights object / N / mask geometry) --
    re-ran the ~10 s byte-gate replay per key.  This wraps the module attribute ``check`` (runtime assignment in THIS adapter; no sealed file is
    edited) with a memo keyed by (device, probe, gate, cases, binding, verbose): the gate stays MANDATORY and runs exactly ONCE per (process,
    device) -- at install() / native_install() / the kit's warm call, or at the first served call -- and a gate=True report satisfies a later
    gate=False request of the same key.  A failing gate raises the payload's Refusal (kind prefixed gate_failed: by install()); nothing is cached
    for a failure, so a later call re-runs it.  Removed when the carried payload memoises its own check (1.1.0)."""
    if getattr(F, "_check_unmemoised", None) is not None:
        return F
    orig = F.check
    memo = {}
    lock = threading.Lock()

    def check(device=None, probe=False, gate=True, binding=None, cases="gate", verbose=False):
        dev = _device_key(device)
        key = (dev, bool(probe), bool(gate), str(cases), binding, bool(verbose))
        with lock:
            rep = memo.get(key)
            if rep is None and not gate:
                rep = memo.get((dev, bool(probe), True, str(cases), binding, bool(verbose)))
            if rep is None and binding is None:                              # a report made under an explicit binding satisfies the default-binding request
                for k, v in memo.items():
                    if k[0] == dev and k[1] == bool(probe) and (k[2] or not gate) and k[3] == str(cases) and k[5] == bool(verbose):
                        rep = v
                        break
        if rep is not None:
            return rep
        rep = orig(device=device, probe=probe, gate=gate, binding=binding, cases=cases, verbose=verbose)
        with lock:
            memo[key] = rep
        return rep

    check.__doc__ = (orig.__doc__ or "") + "\n\n[opt_core erratum-2 shim: memoised per (device, probe, gate, cases, binding, verbose); the original is _check_unmemoised.]"
    check.__wrapped__ = orig
    F._check_unmemoised = orig
    F._check_memo = memo
    F.check = check
    return F


def module(name):
    """A carried submodule by short name ('launch', 'ops', 'kernel', 'vectors', ...) under the private package name."""
    face()
    try:
        return importlib.import_module(_MODNAME + "." + name)
    except Exception as e:                                                   # noqa: BLE001
        raise Unavailable("import:%s:%s" % (name, type(e).__name__), str(e))


def _cc_tuple(cc):
    if isinstance(cc, (tuple, list)) and len(cc) == 2:
        return (int(cc[0]), int(cc[1]))
    s = str(cc).strip()
    if "." in s:
        a, b = s.split(".", 1)
        return (int(a), int(b[:1] or 0))
    if s.isdigit() and len(s) >= 2:
        return (int(s[:-1]), int(s[-1]))
    raise ValueError(cc)


def admits(cc, dtype, c_z, c_hidden, n_tokens, direction="outgoing", residency=None, residual=False, batch=1, variant="fast"):
    """None when the ACTIVE payload serves this call on a device of capability ``cc``, else the refusal word.  Pure (json + the carried face's
    pure admits; no torch, no device).  ``dtype`` = the compute dtype ('bf16'); an fp32-resident z under bf16 autocast is ``residency='fp32'``
    (the f32z form); ``variant`` fast | exact (exact = the package's bit-identical-to-the-reference-library variant, bf16 z)."""
    try:
        cct = _cc_tuple(cc)
    except (TypeError, ValueError):
        return "cc_unsupported:%r" % (cc,)
    try:
        F = face()
    except Unavailable as e:
        return e.kind
    word = F.admits(cct, dtype, c_z, c_hidden, n_tokens, direction, residency=residency, residual=residual, batch=batch, variant=variant,
                    build_dir=os.path.join(pkg_dir(), "build"))
    if word is not None:
        return word
    arch = F.ARCH_OF_CC.get(cct)
    if F.DEVICE_CLASS.get(arch) not in device_classes():
        return "vectors:no_%s_class(%s carries expected bytes for %s)" % (F.DEVICE_CLASS.get(arch), ACTIVE_PKG, ",".join(device_classes()) or "-")
    return None


# ----------------------------------------------------------------------------------------------------------------------------- install / serve
def stamp_dirs(leaf="trimul_native"):
    """The stamp directory candidates in resolution order, as [(kind, path)] of READABLE existing-or-created directories:
    ``env`` = ``$OPT_CORE_VERDICT_DIR`` (when set it is the only candidate; "0" / "off" = none: the byte gate runs in every process) |
    ``triattn_root`` = the verdict-cache root kernels.triattn's triattn_native uses (<MODEL_OPT_JIT_ROOT>[/<MODEL_OPT_STACK_KEY>] | beside TRITON_CACHE_DIR)
    + ``/trimul_native`` | ``xdg`` = ``$XDG_CACHE_HOME/opt_core/trimul_native`` | ``home`` = ``~/.cache/opt_core/trimul_native`` | ``tmp`` =
    ``<tempfile.gettempdir()>/opt_core-uid<uid>/trimul_native`` (created 0700; honoured only when owned by this uid and not group / other writable).
    The same list serves a pip-installed tree and a repo checkout (nothing is keyed by or written under the package path).  Never raises.
    ``leaf`` names the provider sub-directory under each root (the fpf_trimul_v4 warm probe stamps its verdicts beside this gate's with leaf "fpf_trimul_v4")."""
    v = os.environ.get(STAMP_ENV)
    cands = []
    if v is not None:
        if v.strip().lower() in ("", "0", "off", "no", "false", "none"):
            return []
        cands.append(("env", os.path.join(v, leaf)))
    else:
        try:
            _cc_verdict_dir = importlib.import_module("opt_core.kernels.triattn.triattn_native")._verdict_dir   # kernels.triattn's verdict-cache root (stdlib at import; absolute: this package is routable top-level); sibling dir here
            d = _cc_verdict_dir()
            if d:
                cands.append(("triattn_root", os.path.join(os.path.dirname(os.path.abspath(d)), leaf)))
        except Exception:                                                        # noqa: BLE001 -- that provider absent / its env unset: the generic cache roots below
            pass
        if os.environ.get("XDG_CACHE_HOME"):
            cands.append(("xdg", os.path.join(os.environ["XDG_CACHE_HOME"], "opt_core", leaf)))
        try:
            cands.append(("home", os.path.join(os.path.expanduser("~"), ".cache", "opt_core", leaf)))
        except Exception:                                                        # noqa: BLE001 -- no home for this uid
            pass
        try:
            import tempfile
            uid = os.getuid() if hasattr(os, "getuid") else 0
            cands.append(("tmp", os.path.join(tempfile.gettempdir(), "opt_core-uid%d" % uid, leaf)))
        except Exception:                                                        # noqa: BLE001 -- no temp dir
            pass
    out = []
    for kind, d in cands:
        try:
            if kind == "tmp":
                root = os.path.dirname(d)
                os.makedirs(root, mode=0o700, exist_ok=True)
                st = os.stat(root)
                if (hasattr(os, "getuid") and st.st_uid != os.getuid()) or (st.st_mode & 0o022):
                    continue                                                     # another user's / a group-writable temp root is never trusted
            os.makedirs(d, exist_ok=True)
        except OSError:
            if not os.path.isdir(d):
                continue
        if os.path.isdir(d) and os.access(d, os.R_OK):
            out.append((kind, d))
    return out


def stamp_dir():
    """The first WRITABLE stamp directory (where a passed gate is recorded), or None (= no stamp can be written: the gate runs once per process
    and install() says ``verdict_stamp = disabled:cache_dir_unwritable`` by name)."""
    for kind, d in stamp_dirs():
        if os.access(d, os.W_OK | os.X_OK):
            return d
    return None


def _stamp_dir_kind(path):
    for kind, d in stamp_dirs():
        if d == path:
            return kind
    return "-"


def stamp_facts(device=None, binding=None, cases="gate", digests=None):
    """The identity a stamp is keyed by: {payload SHA256SUMS digest (as checked on disk this process), package path, ACTIVE_PKG,
    payload version, device cc + name, driver version, CUDA (torch build) version, torch version, binding, cases}.  None when the framework or the
    device facts are unavailable (then no stamp is read or written and the gate simply runs)."""
    try:
        import torch                                                             # served calls arrive with tensors: the framework is loaded by then
        if not torch.cuda.is_available():
            return None
        idx = torch.cuda.current_device() if device is None else int(device)
        cc = "%d.%d" % tuple(torch.cuda.get_device_capability(idx))
        name = torch.cuda.get_device_name(idx)
        drv = None
        try:
            drv = int(torch.cuda.driver_version()) if hasattr(torch.cuda, "driver_version") else None
        except Exception:                                                        # noqa: BLE001 -- an older torch without the query: recorded as unknown, still a key part
            drv = None
        facts = collections.OrderedDict([
            ("schema", STAMP_SCHEMA), ("sums_sha256", (digests or {}).get("SHA256SUMS") or _sha256(os.path.join(pkg_dir(), "SHA256SUMS"))),
            ("pkg", os.path.abspath(pkg_dir())), ("active_pkg", ACTIVE_PKG), ("payload_version", payload_version()), ("cc", cc), ("device_name", name),
            ("driver_version", drv), ("cuda_version", str(getattr(torch.version, "cuda", None))), ("torch", str(torch.__version__)),
            ("binding", binding or "auto"), ("cases", str(cases)), ("python", sys.version.split()[0])])
        return facts
    except Exception:                                                            # noqa: BLE001 -- no framework / no device here: no stamp (the gate decides, by name)
        return None


def _stamp_key(facts):
    return hashlib.sha256(json.dumps(facts, sort_keys=True).encode("utf-8")).hexdigest()


def _stamp_path(sdir, facts):
    return os.path.join(sdir, "gate-%s.json" % _stamp_key(facts)[:24])


def stamp_read(sdir, facts):
    """The stamp dict if a stamp for exactly these facts exists and is intact (its recorded facts == facts, schema current), else None."""
    if not sdir or not facts:
        return None
    p = _stamp_path(sdir, facts)
    try:
        with open(p, encoding="utf-8") as fh:
            st = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(st, dict) or st.get("schema") != STAMP_SCHEMA or st.get("facts") != json.loads(json.dumps(facts)):
        return None
    st["_path"] = p
    return st


def stamp_write(sdir, facts, rep):
    """Record a PASSED gate for these facts (atomic write); returns the path, or None when the directory is not writable (never raises)."""
    if not sdir or not facts:
        return None
    p = _stamp_path(sdir, facts)
    body = collections.OrderedDict([("schema", STAMP_SCHEMA), ("facts", facts), ("written_utc", __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime())),
                                    ("gate", {k: rep.get(k) for k in ("device_class", "arch", "gate", "cases", "units_loaded", "version") if k in rep})]
                                   + ([("record", rep["record"])] if isinstance(rep, dict) and "record" in rep else []))   # another provider's verdict record (the v4 warm probe)
    tmp = "%s.%d.tmp" % (p, os.getpid())
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(body, fh, indent=1, default=str)
        os.replace(tmp, p)
        return p
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return None




# ----------------------------------------------------------------------------------------------------------------- byte-gate memory scope
GATE_CHUNK_BYTES = 128 << 20                                                 # the install-time byte gate regenerates its closed-form inputs in row slabs whose int64
_GATE_MEM = {"scopes": 0, "max_slab_bytes": 0, "cases_served": 0, "cache_evictions": 0, "released": 0}   # temporaries stay under this many bytes


def gate_memory_stats():
    """Bookkeeping of the byte gate's memory scope: {'scopes', 'max_slab_bytes', 'cases_served', 'cache_evictions', 'released'}."""
    return dict(_GATE_MEM)


def vectors_module(F=None):
    """The payload's ``trimul_native.vectors`` module (the byte gate's case recipes and replay), imported beside the face."""
    F = F if F is not None else face()
    return importlib.import_module(F.__name__.rsplit(".", 1)[0] + ".vectors")


def chunked_closed_form_inputs(V, chunk_bytes=GATE_CHUNK_BYTES):
    """``V.closed_form_inputs`` re-derived in row slabs: the SAME tensors byte for byte (the replay re-checks every case's input digests), with the
    int64 index / hash temporaries of one slab (about four alive at once) bounded by ``chunk_bytes`` instead of three full [n, n, c_z] int64 tensors."""
    def closed_form_inputs(case, device, weight_salt=0):
        import torch
        cz, ch, n, B = case["c_z"], case["c_hidden"], case["n"], case.get("batch", 1)
        dev = torch.device(device)
        f32z = case["form"] == "f32z"
        if case["form"] not in ("f32z", "bf16"):
            return V_orig(case, device, weight_salt=weight_salt)
        rows = max(1, int(chunk_bytes) // max(1, n * cz * 8 * 4))
        slab = min(rows, n) * n * cz * 8
        _GATE_MEM["max_slab_bytes"] = max(_GATE_MEM["max_slab_bytes"], slab)
        z = torch.empty(((B, n, n, cz) if B > 1 else (n, n, cz)), dtype=(torch.float32 if f32z else torch.bfloat16), device=dev)
        j = torch.arange(n, dtype=torch.int64, device=dev).view(1, n, 1)
        c = torch.arange(cz, dtype=torch.int64, device=dev).view(1, 1, cz)
        for b in range(B):
            zb_out = z[b] if B > 1 else z
            for i0 in range(0, n, rows):
                i = torch.arange(i0, min(n, i0 + rows), dtype=torch.int64, device=dev).view(-1, 1, 1)
                r = 100 + (7 * i + 3 * j) % 155
                k = V._h(torch, i, j, c, 1 + 16 * b, dev) % (2 * r + 1) - r
                zb = k.to(torch.float32) / 128.0
                if f32z:
                    k2 = V._h(torch, i, j, c, 2 + 16 * b, dev) % 129 - 64
                    zb = zb + k2.to(torch.float32) * (2.0 ** -16)
                else:
                    zb = zb.to(torch.bfloat16)
                zb_out[i0:i0 + rows] = zb
                del i, r, k, zb
        mask = None
        if case["mask"] == "tail3":
            m = torch.ones(n, n, dtype=torch.float32, device=dev)
            m[n - 3:, :] = 0
            m[:, n - 3:] = 0
            mask = m if B == 1 else m.unsqueeze(0).expand(B, n, n).contiguous()
        elif case["mask"] != "none":
            raise V.VectorsError("corrupt:mask=%s" % case["mask"])

        def mat(rows_, cols, salt):
            o = torch.arange(rows_, dtype=torch.int64, device=dev).view(rows_, 1)
            q = torch.arange(cols, dtype=torch.int64, device=dev).view(1, cols)
            return ((V._h(torch, o, q, 0, salt, dev) % 255) - 127).to(torch.float32) / 2048.0

        def vec(length, salt, base, scale):
            o = torch.arange(length, dtype=torch.int64, device=dev)
            return base + ((V._h(torch, o, 0, 0, salt, dev) % 65) - 32).to(torch.float32) / scale
        ws = 1000 * int(weight_salt)
        w = {"ln_in_w": vec(cz, 21 + ws, 1.0, 256.0), "ln_in_b": vec(cz, 22 + ws, 0.0, 512.0),
             "w_ag": mat(ch, cz, 11 + ws), "w_ap": mat(ch, cz, 12 + ws), "w_bg": mat(ch, cz, 13 + ws), "w_bp": mat(ch, cz, 14 + ws),
             "ln_out_w": vec(ch, 23 + ws, 1.0, 256.0), "ln_out_b": vec(ch, 24 + ws, 0.0, 512.0),
             "w_o": mat(cz, ch, 15 + ws), "w_og": mat(cz, cz, 16 + ws)}
        if case["form"] == "bf16":
            w = {k_: v_.to(torch.bfloat16) for k_, v_ in w.items()}
        return z, mask, w
    V_orig = V.closed_form_inputs
    closed_form_inputs._chunked = True
    return closed_form_inputs


class gate_memory_scope(object):
    """Bound the byte gate's transient device memory for the duration of ``face.check(gate=True)``: (1) closed-form inputs regenerated in row slabs
    (chunked_closed_form_inputs); (2) each case served through a per-case cache -- the replay's shared cache is emptied of the previous case's
    workspaces / packs when the case changes (the multiweight interleave of ONE case still runs through one shared cache, which is what it checks);
    (3) on exit the shims are removed and the caching allocator's freed blocks are returned to the driver, so nothing the gate allocated is held
    afterwards.  Verdict semantics unchanged: the same cases, inputs (digest-checked), kernels and expected bytes."""

    def __init__(self, F=None, chunk_bytes=GATE_CHUNK_BYTES):
        self.F = F
        self.chunk_bytes = chunk_bytes
        self.V = None
        self._saved = None
        self._last_case = None

    def __enter__(self):
        try:
            self.V = vectors_module(self.F)
        except (Unavailable, ImportError, AttributeError):                  # a payload without the vectors module: nothing to bound (the gate itself refuses by name)
            self.V = None
            return self
        V = self.V
        self._saved = (V.closed_form_inputs, V._serve_case)
        orig_serve = V._serve_case
        scope = self

        def serve_case(F_, case, z, mask, w, cache):
            cid = case.get("id") if isinstance(case, dict) else None
            if cache and scope._last_case is not None and cid != scope._last_case:
                n_ = len(cache)
                cache.clear()                                                # the previous case's workspaces / packs / plans: a per-case cache
                _GATE_MEM["cache_evictions"] += n_
            scope._last_case = cid
            _GATE_MEM["cases_served"] += 1
            return orig_serve(F_, case, z, mask, w, cache)
        V.closed_form_inputs = chunked_closed_form_inputs(V, self.chunk_bytes)
        V._serve_case = serve_case
        _GATE_MEM["scopes"] += 1
        return self

    def __exit__(self, *exc):
        if self.V is not None and self._saved is not None:
            self.V.closed_form_inputs, self.V._serve_case = self._saved
        self._saved = None
        release_device_blocks()
        _GATE_MEM["released"] += 1
        return False


def release_device_blocks():
    """gc + hand the caching allocator's unused blocks back to the driver (after the gate's transients are dropped)."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError, AttributeError):                      # no framework / no device: nothing cached
        pass


def install(device=None, gate=True, binding=None, cases="gate", force_gate=False):
    """Once per (process, device, gate) (``force_gate``: replay the byte gate in THIS call even over an intact stamp / memo -- the pre-stamp verb's
    --verify, meant for a fresh process): verify_digests() -> the carried face's check(device): driver binding (``binding`` None = auto |
    'ctypes' | 'cuda_bindings') -> capability -> every compute unit's cubin loaded after its manifest digests -> load check -> byte gate
    (``gate``; ``cases`` 'gate' | 'all').  Returns the report dict (package facts + 'pkg', 'payload_version', 'digests'); raises Unavailable by name."""
    key = (str(device), bool(gate), cases, binding)
    with _LOCK:
        rep = _STATE["report"].get(key)
        if rep is not None and not force_gate:
            return rep
    import time as _time
    t_inst = _time.perf_counter()
    dig = verify_digests()                                                   # every process: the four DIGESTS records + every SHA256SUMS line re-hashed from disk (~0.03 s)
    F = face()
    facts = stamp = None; wdir = None; rdirs = []; env_off = False
    if gate:                                                                 # the byte gate is MANDATORY once per INSTALLED TREE (x device class x stack): a stamp written after a
        facts = stamp_facts(device, binding, cases, dig)                     # passed gate lets later processes on the same tree / device / driver / torch skip the vector replay
        v = os.environ.get(STAMP_ENV)                                        # (cubin digests, loads and the load check still run here; a changed payload byte changes the
        env_off = v is not None and v.strip().lower() in ("", "0", "off", "no", "false", "none")   # SHA256SUMS digest verify_digests() just proved, hence the key -> the gate runs again)
        if facts is not None and not env_off:
            rdirs = stamp_dirs()
            if not force_gate:                                               # force_gate (the pre-stamp verb's --verify): replay the gate even over an intact stamp, then re-write it
                for _kind, d in rdirs:                                       # read: the first intact stamp along the chain (a pre-stamped read-only cache counts)
                    stamp = stamp_read(d, facts)
                    if stamp is not None:
                        stamp["_kind"] = _kind
                        break
            wdir = next((d for _kind, d in rdirs if os.access(d, os.W_OK | os.X_OK)), None)
    run_gate = bool(gate) and stamp is None
    t_chk = _time.perf_counter()
    try:
        if run_gate:                                                             # the byte gate's transients bounded and released (gate_memory_scope)
            with gate_memory_scope(F):
                arch_rep = F.check(device=device, gate=run_gate, binding=binding, cases=cases)
        else:
            arch_rep = F.check(device=device, gate=run_gate, binding=binding, cases=cases)
    except F.Refusal as e:
        kind = e.kind
        if kind.startswith(("gate", "vectors", "bytes", "mismatch")):        # the byte gate ran and did not pass: by name, never a silent serve
            kind = "gate_failed:" + kind
        raise Unavailable(kind, getattr(e, "detail", "") or str(e))
    except (ImportError, OSError) as e:                                      # the driver binding / torch not importable here
        raise Unavailable("import:%s" % type(e).__name__, str(e))
    except Exception as e:                                                   # noqa: BLE001 -- the payload's driver / loader errors outside its Refusal vocabulary
        if type(e).__name__ in ("DriverError", "LoadError", "Unresolvable"):   # (cuInit on a host without a device, a driver that rejects the binding): by name
            raise Unavailable("driver_unavailable:%s" % (str(e).split("->")[-1].strip().replace(" ", "_")[:80] or type(e).__name__), str(e))
        raise
    cls = arch_rep.get("device_class")
    if cls not in device_classes():
        raise Unavailable("vectors:no_%s_class" % cls, "the payload's vectors carry %s" % (",".join(device_classes()) or "-"))
    t_end = _time.perf_counter()
    rep = dict(arch_rep)
    rep.update({"pkg": ACTIVE_PKG, "payload_version": payload_version(), "digests": len(dig), "module": _MODNAME, "gate_ran": run_gate, "gate_mem": gate_memory_stats(),
                "check_s": round(t_end - t_chk, 3), "gate_s": round(t_end - t_chk, 3) if run_gate else 0.0})
    if not gate:
        rep["verdict_stamp"] = "not_consulted"                               # the caller asked for no gate (development); nothing read or written
    elif stamp is not None:
        rep["verdict_stamp"] = "hit"; rep["verdict_stamp_path"] = stamp.get("_path"); rep["verdict_stamp_dir_kind"] = stamp.get("_kind"); rep["verdict_stamp_written_utc"] = stamp.get("written_utc")
        rep["gate_note"] = "byte gate PASSED earlier on this installed tree / device / stack (verification stamp); this process: digests + cubin loads + load check"
    elif env_off:
        rep["verdict_stamp"] = "disabled:env_off"                            # $OPT_CORE_VERDICT_DIR=0: the gate ran and will run in every process
    elif facts is None:
        rep["verdict_stamp"] = "disabled:no_device_facts"
    elif wdir is None:
        rep["verdict_stamp"] = "disabled:cache_dir_unwritable"               # no candidate directory writable (read-only HOME / XDG / tmp): the gate ran; BY NAME here and in the face's token
    else:
        pth = stamp_write(wdir, facts, rep)
        rep["verdict_stamp"] = "written" if pth else "unwritable"
        rep["verdict_stamp_path"] = pth; rep["verdict_stamp_dir_kind"] = _stamp_dir_kind(wdir)
    rep["install_s"] = round(_time.perf_counter() - t_inst, 3)
    with _LOCK:
        _STATE["report"][key] = rep
    return rep


def report(device=None):
    """The install report if install() ran for ``device`` in this process (any gate mode), else None."""
    with _LOCK:
        for (dev, _g, _c, _b), rep in _STATE["report"].items():
            if dev == str(device):
                return rep
    return None


def pack(weights, cache=None, device=None):
    """The package's weight pack from the ten canonical tensors (cached in ``cache`` by tensor identity when given)."""
    F = face()
    try:
        return F.pack_weights(weights, cache, device=device)
    except F.Refusal as e:
        raise Unavailable(e.kind, getattr(e, "detail", ""))


def forward(z, mask=None, *, direction, weights, residual=False, cache=None, eps=1e-5, variant="fast", config=None, gate=True, binding=None, pack_policy=None):
    """Serve the op through the payload: install() once per (process, device), then the carried face's serve().  ``z`` bf16 (bf16 form) or fp32
    inside a bf16 autocast region (the f32z form: bf16 compute, update / residual sum returned in fp32); ``variant`` fast | exact; ``config``
    optional launch-plan overrides passed through (the payload's CELLS.json decides otherwise).  Raises Unavailable by name; kernel errors propagate."""
    F = face()
    idx = z.device.index
    irep = install(device=idx, gate=gate, binding=binding)
    cfg = dict(config) if isinstance(config, dict) else {}
    use_caller = cfg.get("cache") == "caller"
    cfg = {k: v for k, v in cfg.items() if k in ("incoming_mode", "k1_cfg", "k3_cfg", "sm80_tiles", "cells", "f32_resident_ok")}
    if not irep.get("gate_ran", True):                                       # the stamp honoured the gate for this tree: the payload's per-call check() is the load-check key
        cfg["gate"] = False
    if not use_caller:                                                       # the workspace residency policy: one payload cache per (device, N, c_z, c_hidden), shared by every layer
        zs = z if z.dim() == 4 else z[None]
        cache = payload_cache(idx if idx is not None else 0, int(zs.shape[1]), int(zs.shape[-1]), int(weights["w_ap"].shape[0]))
    apply_pack_policy(cache, weights, pack_policy_for(policy=pack_policy))   # weight-pack residency: under big (lru1) the shared cache keeps at most this layer's pack
    _keys_before = _packclass_keys(cache)                                    # under EVERY word: new pack-class entries are tied to their owner tensor's lifetime + a bounded LRU (after_call)
    if variant != "fast":
        cfg["variant"] = variant                                             # 1.0.1's measured-rows table keys fast-class cells only: the exact variant refuses BY NAME
    try:                                                                     # (no_cell:<key>) until a payload keys exact-class cells -- never hand-keyed here (A4: no cross-class inheritance)
        return F.serve(z, mask, direction=direction, weights=weights, residual=bool(residual), cache=cache, eps=eps, config=(cfg or None))
    except F.Refusal as e:
        raise Unavailable(e.kind, getattr(e, "detail", ""))
    finally:
        after_call(cache, weights, _keys_before)


def _wget(weights, k):
    return weights.get(k) if isinstance(weights, dict) else getattr(weights, k, None)


def notices():
    """The package's information tokens emitted so far in this process ('UNCOVERED_CELL:<key> ...', 'NO_CELLS_TABLE'); [] before the face loads."""
    with _LOCK:
        F = _STATE["face"]
    return list(F.notices()) if F is not None else []
