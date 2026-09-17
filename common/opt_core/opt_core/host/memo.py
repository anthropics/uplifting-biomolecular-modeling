"""Content-addressed memo of an expensive host-side build (features, MSA parses, conformers, language-model embeddings).

Contract. ``Memo(name).get_or_build(parts, build)`` returns ``build()``'s value for the key :func:`digest` ``(parts)`` — computing it
once per key per process (and once per directory when ``disk_dir`` is set). ``parts`` is a mapping naming EVERY input that
determines the value: the input bytes (or their :func:`file_digest`), the seed, the builder's name and version, the settings; a part
the digest cannot canonicalise (an arbitrary object) is a :class:`~opt_core.host.events.HostRefused` ``reason=undigestible:<type>``
— the adapter either names that input properly or calls ``get_or_build(parts, build, bypass="<reason>")``, which builds without
caching and COUNTS the bypass under its reason (the feature-cache-bypass-by-name form: nothing uncached is silent).

The memo is bounded (``max_entries``, least-recently-used eviction; ``None`` = unbounded) and thread-safe (one build per key at a
time; concurrent callers of the same key wait for the first and share its value). ``verify_every=N`` recomputes on every N-th hit of the memo
(counted across keys) and compares with ``equal`` (default: equal digests of the two values): a disagreement raises :class:`MemoMismatch` (event ``MISMATCH``)
— a memo that disagrees with a fresh build is a wrong key, and the run must not continue on it. ``disk_dir`` mirrors entries as
``<key>.pkl`` + ``<key>.json`` (value sha256, parts digest, name) written atomically (temporary file + ``os.replace``); a file whose
bytes do not match its recorded sha256 is counted ``disk_corrupt`` and rebuilt (``strict_disk=True`` raises instead). Pickle is the
format: the directory is the operator's own cache, never a channel for untrusted files — so nothing is unpickled before the sha256
matches, nor from a directory or file that :func:`_refusal` rejects (writable by group or other, or — unless this process is uid 0 —
owned by neither this uid nor root): such an entry is REFUSED by name on stderr (what, why, the fix), counted ``disk_corrupt`` and
treated as absent (rebuilt; never raised). The sha256 catches truncation and corruption, not a writer of the directory — hence the
owner / mode rule.

Evidence: ``evidence()`` / ``active_line(tag, name)`` (``[<tag>] LEVER name=F6.feature_cache state=on impl=host.memo origin=core memo=..
max_entries=.. disk=.. verify_every=..``) once at enable; ``tally()`` / ``tally_line(tag)`` (``... TALLY name=.. hits=.. misses=.. builds=.. bypass=<reason:n,..|none> evictions=.. verified=..
mismatches=.. disk_hits=.. disk_writes=.. disk_corrupt=.. disk_unpicklable=.. disk_write_failed=.. entries=.. build_s=..``) at exit; a value
pickle cannot carry, or a write the filesystem refuses, leaves that key memory-only — counted (``disk_unpicklable`` / ``disk_write_failed``,
event ``FALLBACK``; ``events`` holds the first occurrence per reason, the tally the counts). Numerics: none moved — the value IS a build's value; the class of
a kit line using it rests on the key's completeness and on the build taking its randomness from the parts (seed in the key).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import struct
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Mapping, Optional

from .. import report
from .events import HostEvent, HostRefused

LEVER = "host.memo"
DIGEST_VERSION = "d1"           # part of every key: a change to the canonical form below changes this tag, never silently re-keys


class MemoMismatch(RuntimeError):
    """verify_every recomputation disagreed with the cached value. ``.event`` is the ``MISMATCH`` :class:`HostEvent`."""

    def __init__(self, name: str, key: str, cached_digest: str, fresh_digest: str) -> None:
        self.event = HostEvent(LEVER, "MISMATCH", "verify", {"name": name, "key": key, "cached": cached_digest[:16], "fresh": fresh_digest[:16]})
        super().__init__(f"HOST {LEVER} MISMATCH " + report.kv(*self.event.pairs()))


# ------------------------------------------------------------------------------------------------------------------ digest


def _feed(h: "hashlib._Hash", tag: bytes, payload: bytes) -> None:
    h.update(tag)
    h.update(struct.pack("<Q", len(payload)))
    h.update(payload)


def _walk(h: "hashlib._Hash", obj: Any) -> None:
    t = type(obj)
    mod = t.__module__
    if obj is None:
        _feed(h, b"N", b"")
    elif t is bool:
        _feed(h, b"B", b"1" if obj else b"0")
    elif t is int:
        _feed(h, b"I", str(obj).encode())
    elif t is float:
        _feed(h, b"F", struct.pack("<d", obj))
    elif t is str:
        _feed(h, b"S", obj.encode("utf-8", "surrogatepass"))
    elif t in (bytes, bytearray, memoryview):
        _feed(h, b"Y", bytes(obj))
    elif t in (tuple, list):
        _feed(h, b"L" if t is list else b"T", str(len(obj)).encode())
        for x in obj:
            _walk(h, x)
    elif t in (dict, OrderedDict) or isinstance(obj, Mapping):
        items = list(obj.items())
        for k, _ in items:
            if type(k) not in (str, int):
                raise HostRefused(LEVER, "undigestible:mapping_key", key_type=type(k).__name__)
        items.sort(key=lambda kv_: (type(kv_[0]).__name__, str(kv_[0])))
        _feed(h, b"D", str(len(items)).encode())
        for k, v in items:
            _walk(h, k)
            _walk(h, v)
    elif t in (set, frozenset):
        raise HostRefused(LEVER, "undigestible:set", hint="pass a sorted list")
    elif hasattr(obj, "__opt_core_digest__"):
        _feed(h, b"O", t.__name__.encode())
        _walk(h, obj.__opt_core_digest__())
    elif mod == "numpy" or mod.startswith("numpy."):
        np = sys.modules.get("numpy")
        if np is None or not isinstance(obj, (np.ndarray, np.generic)):
            raise HostRefused(LEVER, "undigestible:" + t.__name__)
        if getattr(getattr(obj, "dtype", None), "kind", "") == "O":
            raise HostRefused(LEVER, "undigestible:ndarray[object]", hint="an object array holds references, not content")
        arr = np.ascontiguousarray(obj)
        _feed(h, b"A", (arr.dtype.str + "|" + ",".join(str(s) for s in arr.shape)).encode())
        _feed(h, b"a", arr.tobytes())
    elif mod == "torch" or mod.startswith("torch."):
        torch = sys.modules.get("torch")
        if torch is None or not isinstance(obj, torch.Tensor):
            raise HostRefused(LEVER, "undigestible:" + t.__name__)
        ten = obj.detach()
        if ten.device.type != "cpu":
            ten = ten.cpu()
        ten = ten.contiguous()
        _feed(h, b"P", (str(ten.dtype) + "|" + ",".join(str(s) for s in ten.shape)).encode())
        raw = ten.reshape(-1).view(torch.uint8) if ten.numel() else ten.new_empty((0,), dtype=torch.uint8)
        _feed(h, b"p", raw.numpy().tobytes())
    else:
        raise HostRefused(LEVER, "undigestible:" + t.__name__, module=mod)


def digest(obj: Any) -> str:
    """sha256 (hex) of a canonical encoding of ``obj``: None, bool, int, float, str, bytes, tuples/lists (ordered), mappings with
    str/int keys (order-free), numpy arrays and torch tensors (dtype, shape, bytes — a tensor on a device is copied to host), and
    objects offering ``__opt_core_digest__()``. Anything else is :class:`HostRefused` ``reason=undigestible:<type>`` — a key must be
    made of content, never of object equality."""
    h = hashlib.sha256()
    h.update(DIGEST_VERSION.encode())
    _walk(h, obj)
    return h.hexdigest()


def file_digest(path: str, chunk: int = 1 << 20) -> str:
    """sha256 (hex) of a file's bytes — the part to key on for an input file (never its path or mtime)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def default_equal(a: Any, b: Any) -> bool:
    return digest(a) == digest(b)


def _refusal(*paths: str) -> Optional[str]:
    """Why the mirror files ``paths`` (one directory) must not be read (reason + fix), or None: neither they nor their directory may be
    writable by group or other, and all must belong to this uid or to root — uid 0 accepts any owner (a container's root reading a
    bind-mounted host directory)."""
    uid = os.geteuid()
    for p in (os.path.dirname(os.path.abspath(paths[0])),) + paths:
        st = os.stat(p)
        if st.st_mode & 0o022:
            return f"{p} is writable by group or other (mode {st.st_mode & 0o7777:04o}); fix: chmod go-w {p}"
        if uid != 0 and st.st_uid not in (uid, 0):
            return f"{p} belongs to uid {st.st_uid}, not to this process (uid {uid}) or root; fix: name a disk_dir of your own, or chown {p}"
    return None


# ------------------------------------------------------------------------------------------------------------------ memo


class Memo:
    def __init__(self, name: str, *, max_entries: Optional[int] = 8, disk_dir: Optional[str] = None, verify_every: int = 0,
                 equal: Callable[[Any, Any], bool] = default_equal, strict_disk: bool = False) -> None:
        if max_entries is not None and max_entries < 1:
            raise ValueError("max_entries must be >= 1 or None (unbounded)")
        if verify_every < 0:
            raise ValueError("verify_every must be >= 0")
        self.name = str(name)
        self.max_entries = max_entries
        self.disk_dir = disk_dir
        self.verify_every = int(verify_every)
        self.equal = equal
        self.strict_disk = bool(strict_disk)
        self._entries: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()
        self._key_locks: Dict[str, threading.Lock] = {}
        self._t = {"hits": 0, "misses": 0, "builds": 0, "evictions": 0, "verified": 0, "mismatches": 0,
                   "disk_hits": 0, "disk_writes": 0, "disk_corrupt": 0, "disk_unpicklable": 0, "disk_write_failed": 0, "build_s": 0.0}
        self._bypass: Dict[str, int] = {}
        self.events = []                                  # HostEvent records (BYPASS reasons once, disk corruption) for the manifest
        if disk_dir is not None:
            os.makedirs(disk_dir, mode=0o755, exist_ok=True)   # never group/other-writable, whatever the umask (_refusal rejects such a directory)

    # -- keys
    def key(self, parts: Mapping[str, Any]) -> str:
        if not isinstance(parts, Mapping) or not parts:
            raise HostRefused(LEVER, "undigestible:parts", hint="parts is a non-empty mapping naming every input of the build")
        return digest({"memo": self.name, "parts": parts})

    # -- the one entry point
    def get_or_build(self, parts: Mapping[str, Any], build: Callable[[], Any], *, bypass: Optional[str] = None) -> Any:
        """The value for ``parts``: cached (memory, then disk) or built now. ``bypass="<reason>"`` builds without caching, counted."""
        if bypass:
            return self._bypassed(bypass, build)
        k = self.key(parts)
        with self._lock:
            if k in self._entries:
                self._entries.move_to_end(k)
                value = self._entries[k]
                self._t["hits"] += 1
                nth = self._t["hits"]
                hit = True
            else:
                hit = False
                lock = self._key_locks.setdefault(k, threading.Lock())
        if hit:
            self._maybe_verify(k, value, build, nth)
            return value
        try:
            with lock:                                    # one builder per key; later callers of the key find it in memory
                with self._lock:
                    if k in self._entries:
                        self._entries.move_to_end(k)
                        self._t["hits"] += 1
                        return self._entries[k]
                    self._t["misses"] += 1
                value, from_disk = self._disk_read(k)
                if not from_disk:
                    t0 = time.perf_counter()
                    value = build()
                    dt = time.perf_counter() - t0
                    with self._lock:
                        self._t["builds"] += 1
                        self._t["build_s"] += dt
                    self._disk_write(k, parts, value)
                with self._lock:
                    self._entries[k] = value
                    self._entries.move_to_end(k)
                    if self.max_entries is not None:
                        while len(self._entries) > self.max_entries:
                            self._entries.popitem(last=False)
                            self._t["evictions"] += 1
                return value
        finally:
            with self._lock:                              # a failed build leaves no lock behind for its key
                self._key_locks.pop(k, None)

    def _bypassed(self, reason: str, build: Callable[[], Any]) -> Any:
        with self._lock:
            first = reason not in self._bypass
            self._bypass[reason] = self._bypass.get(reason, 0) + 1
            if first:
                self.events.append(HostEvent(LEVER, "BYPASS", str(reason), {"name": self.name}))
        return build()

    def _maybe_verify(self, k: str, cached: Any, build: Callable[[], Any], nth_hit: int) -> None:
        if not self.verify_every or nth_hit % self.verify_every:
            return
        fresh = build()
        with self._lock:
            self._t["verified"] += 1
        if not self.equal(cached, fresh):
            with self._lock:
                self._t["mismatches"] += 1
            try:
                cd, fd = digest(cached), digest(fresh)
            except HostRefused:
                cd, fd = "n/a", "n/a"
            exc = MemoMismatch(self.name, k[:16], cd, fd)
            self.events.append(exc.event)
            raise exc

    # -- disk mirror
    def _paths(self, k: str):
        return os.path.join(self.disk_dir, k + ".pkl"), os.path.join(self.disk_dir, k + ".json")

    def _disk_read(self, k: str):
        if self.disk_dir is None:
            return None, False
        pkl, meta = self._paths(k)
        if not (os.path.isfile(pkl) and os.path.isfile(meta)):
            return None, False
        try:
            why = _refusal(pkl, meta)
        except OSError as exc:
            why = f"{pkl} cannot be examined ({exc}); fix: none needed, it is rebuilt"
        if why is not None:                               # nothing was read: named, counted, then exactly the absent entry (never raised)
            with self._lock:
                self._t["disk_corrupt"] += 1
            print(f"[opt_core.host.memo:{self.name}] REFUSED cache entry {pkl}: {why} — not unpickled, treated as absent (rebuilt)", file=sys.stderr, flush=True)
            return None, False
        try:
            with open(meta, encoding="utf-8") as fh:
                doc = json.load(fh)
            with open(pkl, "rb") as fh:
                raw = fh.read()
            if hashlib.sha256(raw).hexdigest() != doc.get("value_sha256") or doc.get("key") != k:
                print(f"[opt_core.host.memo:{self.name}] REFUSED cache entry {pkl}: its bytes do not match the sha256 recorded in {meta} (truncated or "
                      f"corrupted) — not unpickled, {'raised (strict_disk)' if self.strict_disk else 'treated as absent (rebuilt and rewritten)'}; "
                      f"fix: none needed", file=sys.stderr, flush=True)
                raise ValueError("sha256 mismatch")
            value = pickle.loads(raw)
        except Exception as exc:                          # a corrupt or foreign file: counted, named, rebuilt (or raised when strict)
            with self._lock:
                self._t["disk_corrupt"] += 1
                if self._t["disk_corrupt"] == 1:          # the first occurrence is the event record; the tally carries the count
                    self.events.append(HostEvent(LEVER, "FALLBACK", "disk_corrupt", {"name": self.name, "key": k[:16], "error": type(exc).__name__}))
            if self.strict_disk:
                raise HostRefused(LEVER, "disk_corrupt", name=self.name, key=k[:16], error=type(exc).__name__)
            return None, False
        with self._lock:
            self._t["disk_hits"] += 1
        return value, True

    def _disk_write(self, k: str, parts: Mapping[str, Any], value: Any) -> None:
        if self.disk_dir is None:
            return
        try:
            raw = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:                          # noqa: BLE001 — a value pickle cannot carry: memory-only for this key, counted and named
            with self._lock:
                self._t["disk_unpicklable"] += 1
                if self._t["disk_unpicklable"] == 1:
                    self.events.append(HostEvent(LEVER, "FALLBACK", "disk_unpicklable", {"name": self.name, "key": k[:16], "error": type(exc).__name__}))
            return
        doc = {"memo": self.name, "key": k, "digest_version": DIGEST_VERSION, "parts_digest": digest(dict(parts)),
               "value_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "written_unix": time.time()}
        pkl, meta = self._paths(k)
        try:
            for path, payload in ((pkl, raw), (meta, json.dumps(doc, sort_keys=True).encode())):
                fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=self.disk_dir)
                try:
                    with os.fdopen(fd, "wb") as fh:
                        fh.write(payload)
                    os.replace(tmp, path)
                except BaseException:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
                    raise
        except OSError as exc:                            # disk full, permissions: the mirror is lost for this key, the memory entry is not
            with self._lock:
                self._t["disk_write_failed"] += 1
                if self._t["disk_write_failed"] == 1:
                    self.events.append(HostEvent(LEVER, "FALLBACK", "disk_write_failed", {"name": self.name, "key": k[:16], "error": type(exc).__name__}))
            return
        with self._lock:
            self._t["disk_writes"] += 1

    # -- census and evidence
    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def evidence(self) -> dict:
        return {"lever": LEVER, "memo": self.name, "max_entries": "unbounded" if self.max_entries is None else self.max_entries,
                "disk": self.disk_dir or "none", "verify_every": self.verify_every}

    def active_line(self, tag: str, name: str = "F6.feature_cache", origin: str = "core") -> str:
        """The kit's ONE activation line for this lever: ``[<tag>] LEVER name=<name> state=on impl=host.memo origin=core ...`` (``name`` = strategy id)."""
        ev = self.evidence()
        return HostEvent(LEVER, "ACTIVE", "", {k: v for k, v in ev.items() if k != "lever"}).lever_line(tag, name, origin)

    def tally(self) -> dict:
        with self._lock:
            out = dict(self._t)
            out["bypass"] = dict(self._bypass)
            out["entries"] = len(self._entries)
        out["name"] = self.name
        out["build_s"] = round(out["build_s"], 3)
        return out

    def tally_line(self, tag: str) -> str:
        t = self.tally()
        fields = OrderedDict()
        for k in ("name", "hits", "misses", "builds", "bypass", "evictions", "verified", "mismatches", "disk_hits", "disk_writes", "disk_corrupt", "disk_unpicklable", "disk_write_failed", "entries", "build_s"):
            fields[k] = t[k]                              # an empty bypass census prints bypass=none (report.kv's dict form)
        return HostEvent(LEVER, "TALLY", "", fields).line(tag)

    def register_exit_tally(self, tag: str) -> bool:
        """Print :meth:`tally_line` once at interpreter exit through :func:`opt_core.report.register_exit_tally`."""
        return report.register_exit_tally(tag + ":" + LEVER + ":" + self.name, lambda: self.tally_line(tag))
