"""Persistent XLA executables for a JAX engine: compile a jitted function once per (shape bucket, configuration, stack) and load the SAME
binary in every later process — plus the environment rule for JAX's own persistent compilation cache.

Contract. :class:`ExecutableCache` lives in a directory (the kit's keyed cache root: :func:`opt_core.jit_cache.cache_dirs` gives
``JAX_EXEC_CACHE_DIR``) and is keyed by the adapter's ``config`` mapping — everything the executable depends on that is not an argument shape:
model config, parameter equality, kernel-implementation flags. The STACK part of the key (jax + jaxlib versions, device kind, platform
version) the cache adds itself, so a wheel or driver change is a miss BY NAME, never a stale load. ``load_or_compile(tag, lower)`` (``tag``
names the shape bucket, e.g. ``b512``; ``lower()`` returns the ``jax.stages.Lowered`` of the jitted function at that bucket's arguments):
a hit deserializes and returns the stored executable; a miss compiles, serializes atomically (tmp + ``os.replace`` — concurrent boxes may
race on one file harmlessly), then RELOADS the file it wrote so that writer and readers execute the identical artefact. A stored file is
one line ``sha256:<hex>`` over the rest of the file, then the pickle; :meth:`ExecutableCache.load` unpickles nothing before that line
matches and :func:`cache_refusal` passes the file and its directory — a file that fails either is REFUSED by name on stderr (what, why,
the fix), counted ``refused`` and treated as absent: compiled and stored again. The line catches truncation and corruption, not a writer
of the directory — hence the owner / mode rule. Every outcome is
a named event on the returned record and in :meth:`ExecutableCache.stats`: ``loaded`` · ``compiled+stored`` · ``compiled(unstored:<why>)``
(serialization failed in this jax — the in-memory executable is used, the reason printed) · ``reloaded-failed(<why>)`` · ``stale(<why>)``
(a file whose recorded stack differs — ignored, recompiled). Exactness: loading a serialized executable replays the compiled program the
writer ran; whether two COMPILATIONS of one program agree bit-exact is XLA's autotuning question (the engine's deterministic recipe pins
``--xla_gpu_autotune_level`` where it matters) — the cache removes that question for every process after the first. Hence the tier
switch ``exact``: under ``exact=True`` (the default: an exact line's cache) an executable that ran but NOT from the stored artefact — an
unstored compile, a stored file that failed to load — is a cannot-run event of the lever, raised as :class:`XlaCacheUnavailable`
(``cannot_run = True``) when it happens, and the kit refuses the mode by name; under ``exact=False`` (a tolerance-class line) the same
events are a cache MISS: the run proceeds on the in-memory executable and the record carries the word ``cache=miss(<n>)``
(:meth:`ExecutableCache.words`), exit 0. Evidence: :meth:`ExecutableCache.evidence_fields` / :meth:`ExecutableCache.evidence_line` —
``LEVER name=<name> state=on impl=xla_exec_cache cache=<loaded|stored|compiled|idle> loads= stores= compiles= unstored= stale= load_s=
compile_s= [cache=miss(n)]``.

:func:`persistent_cache_env` is the environment of JAX's built-in persistent compilation cache under a keyed root (``JAX_COMPILATION_CACHE_DIR``
+ the two thresholds that make every compile eligible); :func:`enable_persistent_cache` applies the same in-process through ``jax.config``.
jax is imported inside the functions that need it (``import opt_core.capture.xla_cache`` is standard library only; Python 3.8 floor).
"""
from __future__ import annotations

import hashlib
import inspect
import os
import pickle
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .. import report
from ..oom import is_oom

def _canonical(config) -> str:
    """A canonical text of the adapter's config mapping for hashing and recording (sorted keys; JSON where possible, ``repr`` per leaf otherwise)."""
    import json  # noqa: PLC0415
    def enc(v):
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if isinstance(v, (list, tuple)):
            return [enc(x) for x in v]
        if isinstance(v, dict):
            return {str(k): enc(v[k]) for k in sorted(v, key=str)}
        return {"__repr__": repr(v), "__type__": type(v).__name__}
    return json.dumps(enc(dict(config)), sort_keys=True, separators=(",", ":"))


PERSISTENT_CACHE_ENV = {"JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0", "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1"}
DIGEST_HEAD = b"sha256:"                                        # line 1 of a stored file: DIGEST_HEAD + <64 hex> + b"\n", over every byte after it


def cache_refusal(path: str, st=None) -> Optional[str]:
    """Why the cache file ``path`` must not be unpickled (reason + fix), or None. Neither the file nor its directory may be writable by
    group or other, and both must belong to this uid or to root — uid 0 accepts any owner (a container's root reading a bind-mounted
    host directory). ``st``: the file's ``os.fstat`` when it is already open."""
    uid, d = os.geteuid(), os.path.dirname(os.path.abspath(path))
    for what, p, s in (("directory", d, os.stat(d)), ("file", path, st if st is not None else os.stat(path))):
        if s.st_mode & 0o022:
            return f"{what} {p} is writable by group or other (mode {s.st_mode & 0o7777:04o}); fix: chmod go-w {p}"
        if uid != 0 and s.st_uid not in (uid, 0):
            return f"{what} {p} belongs to uid {s.st_uid}, not to this process (uid {uid}) or root; fix: name a cache directory of your own, or chown {p}"
    return None


class XlaCacheUnavailable(RuntimeError):
    """jax (or ``jax.experimental.serialize_executable``) is missing, or — under ``exact=True`` — an executable did not come from the stored
    artefact: the named refusal of the mechanism, a cannot-run event (``opt_core.gates.is_cannot_run``)."""

    cannot_run = True


def _jax():
    try:
        import jax  # noqa: PLC0415 — the one lazy import of the mechanism
    except Exception as e:  # noqa: BLE001
        raise XlaCacheUnavailable(f"XLA executable cache unavailable: jax import failed ({e!r})") from None
    return jax


def stack_identity(device=None) -> Dict[str, str]:
    """``{jax, jaxlib, device_kind, platform_version}`` of the running process (the stack part of every cache key)."""
    jax = _jax()
    try:
        import jaxlib  # noqa: PLC0415
        jaxlib_v = getattr(jaxlib, "__version__", None) or getattr(getattr(jax, "lib", None), "__version__", "unknown")
    except Exception:  # noqa: BLE001
        jaxlib_v = getattr(getattr(jax, "lib", None), "__version__", "unknown")
    dev = device if device is not None else jax.devices()[0]
    try:
        pv = str(getattr(dev.client, "platform_version", "unknown"))
    except Exception:  # noqa: BLE001
        pv = "unknown"
    return {"jax": str(jax.__version__), "jaxlib": str(jaxlib_v), "device_kind": str(getattr(dev, "device_kind", "unknown")), "platform_version": pv}


def config_hash(config: Mapping[str, Any], stack: Mapping[str, str]) -> str:
    """sha256[:16] of the canonical text of ``config`` (:func:`_canonical`: sorted keys, JSON scalars, ``repr`` for other leaves) plus the stack mapping."""
    text = _canonical(config) + "|" + _canonical(stack)
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def persistent_cache_env(root: str, cache_key: str, sub: str = "jax") -> Dict[str, str]:
    """The environment of JAX's persistent compilation cache under ``<root>/<cache_key>/<sub>``: ``JAX_COMPILATION_CACHE_DIR`` plus the thresholds
    that make every compilation eligible (min compile time 0 s, min entry size -1)."""
    env = {"JAX_COMPILATION_CACHE_DIR": os.path.join(root, cache_key, sub)}
    env.update(PERSISTENT_CACHE_ENV)
    return env


def enable_persistent_cache(directory: str) -> Dict[str, Any]:
    """Turn on JAX's persistent compilation cache in-process (``jax.config``) with the same thresholds; returns what was set. A jax without the
    options raises :class:`XlaCacheUnavailable` naming the missing option."""
    jax = _jax()
    os.makedirs(directory, exist_ok=True)
    applied: Dict[str, Any] = {}
    for opt, val in (("jax_compilation_cache_dir", directory), ("jax_persistent_cache_min_compile_time_secs", 0.0),
                     ("jax_persistent_cache_min_entry_size_bytes", -1)):
        try:
            jax.config.update(opt, val)
            applied[opt] = val
        except Exception as e:  # noqa: BLE001
            if opt == "jax_compilation_cache_dir":
                raise XlaCacheUnavailable(f"jax {jax.__version__} has no config option {opt!r} ({e!r})") from None
            applied[opt] = f"unavailable ({e!r})"
    return applied


class ExecutableCache:
    """Serialized XLA executables per shape bucket under one directory (module contract). ``config``: the adapter's mapping of everything the
    executable depends on besides argument shapes; ``device``: the jax device the executables run on (default ``jax.devices()[0]``);
    ``reload_after_store``: execute the artefact just written (writer == readers) rather than the in-memory compile; ``exact``: the line's
    tier (module contract — True: a miss raises :class:`XlaCacheUnavailable` at the event; False: a miss is the word ``cache=miss(n)``)."""

    def __init__(self, name: str, directory: str, config: Mapping[str, Any], *, device=None, reload_after_store: bool = True,
                 log: Optional[Callable[[str], None]] = None, verbose: bool = True, exact: bool = True):
        self.name = str(name)
        self.directory = str(directory)
        self.config = dict(config)
        self.exact = bool(exact)
        self._device = device
        self.reload_after_store = bool(reload_after_store)
        self._log = log
        self.verbose = bool(verbose)
        self._stack: Optional[Dict[str, str]] = None
        self._hash: Optional[str] = None
        self._mem: Dict[str, Any] = {}
        self.records: List[dict] = []
        self.stats_ = {"loads": 0, "stores": 0, "compiles": 0, "unstored": 0, "stale": 0, "load_failures": 0, "load_s": 0.0, "compile_s": 0.0, "hits_mem": 0, "refused": 0}

    # ------------------------------------------------------------------------------------------------------------- naming
    def _say(self, line: str) -> None:
        text = f"[opt_core.xla_cache:{self.name}] {line}"
        if self._log is not None:
            self._log(text)
        else:
            print(text, file=sys.stderr, flush=True)

    @property
    def device(self):
        if self._device is None:
            self._device = _jax().devices()[0]
        return self._device

    @property
    def stack(self) -> Dict[str, str]:
        if self._stack is None:
            self._stack = stack_identity(self.device)
        return self._stack

    @property
    def key(self) -> str:
        if self._hash is None:
            self._hash = config_hash(self.config, self.stack)
        return self._hash

    def path(self, tag: str) -> str:
        """``<directory>/<config+stack hash>/<tag>.xla_exe``."""
        return os.path.join(self.directory, self.key, f"{tag}.xla_exe")


    def _miss(self, counter: str, tag: str, text: str) -> None:
        """One cache miss of the tier switch (module contract): counted and printed; under ``exact`` raised as the lever's refusal."""
        self.stats_[counter] += 1
        self._say(text)
        if self.exact:
            raise XlaCacheUnavailable(f"{self.name}: {text} — under exact the executable must come from the stored artefact "
                                      f"({counter}={self.stats_[counter]}); the mode refuses by name")
    # ------------------------------------------------------------------------------------------------------------- serialize api
    @staticmethod
    def _ser():
        try:
            from jax.experimental import serialize_executable  # noqa: PLC0415
        except Exception as e:  # noqa: BLE001
            raise XlaCacheUnavailable(f"jax.experimental.serialize_executable unavailable ({e!r})") from None
        return serialize_executable

    def _load_kwargs(self, ser) -> dict:
        kw = {}
        params = inspect.signature(ser.deserialize_and_load).parameters
        if "backend" in params:
            kw["backend"] = self.device.client
        if "execution_devices" in params:
            kw["execution_devices"] = [self.device]
        return kw

    def load(self, tag: str):
        """The stored executable for ``tag`` or None (missing, stale by stack, or unreadable — each a named event)."""
        p = self.path(tag)
        if not os.path.exists(p):
            return None
        ser = self._ser()
        t0 = time.perf_counter()
        try:
            with open(p, "rb") as f:
                why = cache_refusal(p, os.fstat(f.fileno()))
                head, body = (b"", b"") if why else (f.readline(80), f.read())
            if why is None and head != DIGEST_HEAD + hashlib.sha256(body).hexdigest().encode() + b"\n":
                why = "its first line is not the sha256 of the rest (truncated, corrupted, or written without one); fix: none needed, it is rewritten now"
            if why is not None:                                      # nothing was unpickled: named once, then exactly the absent file
                self.stats_["refused"] += 1
                self.records.append({"tag": tag, "event": "refused", "path": p, "why": why})
                print(f"[opt_core.xla_cache:{self.name}] REFUSED cache file {p}: {why} — not unpickled, treated as absent (compiled and stored again)",
                      file=sys.stderr, flush=True)
                return None
            payload = pickle.loads(body)
            rec_stack = payload.get("stack", {})
            if rec_stack and rec_stack != self.stack:
                self.stats_["stale"] += 1
                self._say(f"stale({tag}): recorded stack {rec_stack} != running {self.stack}; ignored, recompiling")
                return None
            if hashlib.sha256(payload["serialized"]).hexdigest() != payload.get("sha256"):
                raise ValueError("sha256 of the serialized bytes does not match the record (truncated or corrupted file)")
            compiled = ser.deserialize_and_load(payload["serialized"], payload["in_tree"], payload["out_tree"], **self._load_kwargs(ser))
        except XlaCacheUnavailable:
            raise
        except Exception as e:  # noqa: BLE001
            if is_oom(e):                                            # RESOURCE_EXHAUSTED while loading: the caller's to see (recompiling needs no less)
                raise
            self._miss("load_failures", tag, f"reloaded-failed({tag}): {e!r}; recompiling")
            return None
        dt = time.perf_counter() - t0
        self.stats_["loads"] += 1
        self.stats_["load_s"] += dt
        self.records.append({"tag": tag, "event": "loaded", "seconds": round(dt, 3), "path": p, "sha256": payload.get("sha256", "")[:16]})
        if self.verbose:
            self._say(f"loaded {tag} from {p} in {dt:.1f} s (sha256 {payload.get('sha256', '')[:16]})")
        return compiled

    def store(self, tag: str, compiled) -> Tuple[Optional[Any], str]:
        """Serialize ``compiled`` to the tag's file atomically; returns ``(reloaded executable or None, event)``."""
        ser = self._ser()
        p = self.path(tag)
        try:
            serialized, in_tree, out_tree = ser.serialize(compiled)
        except Exception as e:  # noqa: BLE001
            if is_oom(e):
                raise
            self._miss("unstored", tag, f"compiled(unstored:{tag}): serialization failed ({e!r}); using the in-memory executable")
            return None, f"compiled(unstored: {e!r})"
        os.makedirs(os.path.dirname(p), mode=0o755, exist_ok=True)       # 0755 / 0644 whatever the umask: what cache_refusal accepts at the next load
        tmp = f"{p}.tmp{os.getpid()}-{int(time.time() * 1e6)}-{os.urandom(4).hex()}"
        digest = hashlib.sha256(serialized).hexdigest()
        try:
            body = pickle.dumps({"serialized": serialized, "in_tree": in_tree, "out_tree": out_tree, "sha256": digest, "tag": tag,
                                 "config": _canonical(self.config), "stack": dict(self.stack), "written_unix": time.time()})
            with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644), "wb") as f:
                f.write(DIGEST_HEAD + hashlib.sha256(body).hexdigest().encode() + b"\n")
                f.write(body)
            os.replace(tmp, p)
        except Exception as e:  # noqa: BLE001 — an unwritable cache is a NAMED event; the in-memory executable is used
            if is_oom(e):
                raise
            try:
                os.unlink(tmp)
            except OSError:
                pass
            self.records.append({"tag": tag, "event": f"compiled(unstored: store failed {e!r})"})
            self._miss("unstored", tag, f"compiled(unstored:{tag}): store failed {e!r}; using the in-memory executable")
            return None, f"compiled(unstored: store failed {e!r})"
        self.stats_["stores"] += 1
        self.records.append({"tag": tag, "event": "stored", "path": p, "sha256": digest[:16], "bytes": len(serialized)})
        if not self.reload_after_store:
            return None, "compiled+stored"
        try:
            re = ser.deserialize_and_load(serialized, in_tree, out_tree, **self._load_kwargs(ser))
            return re, "compiled+stored"
        except Exception as e:  # noqa: BLE001
            if is_oom(e):
                raise
            self._miss("load_failures", tag, f"reloaded-failed({tag}) after store: {e!r}; using the in-memory executable")
            return None, f"compiled+stored(reload failed: {e!r})"

    def load_or_compile(self, tag: str, lower: Callable[[], Any]):
        """The executable for ``tag``: memory hit, else file hit, else ``lower().compile()`` stored and reloaded. Returns ``(executable, event)``."""
        if tag in self._mem:
            self.stats_["hits_mem"] += 1
            return self._mem[tag], "memory"
        exe = self.load(tag)
        if exe is not None:
            self._mem[tag] = exe
            return exe, "loaded"
        t0 = time.perf_counter()
        lowered = lower()
        compiled = lowered.compile() if hasattr(lowered, "compile") else lowered
        dt = time.perf_counter() - t0
        self.stats_["compiles"] += 1
        self.stats_["compile_s"] += dt
        re, event = self.store(tag, compiled)
        exe = re if re is not None else compiled
        self._mem[tag] = exe
        self.records.append({"tag": tag, "event": event, "compile_s": round(dt, 2)})
        if self.verbose:
            self._say(f"{event} {tag} in {dt:.1f} s -> {self.path(tag)}")
        return exe, event

    def warm(self, tags_and_lowers: Mapping[str, Callable[[], Any]]) -> Dict[str, str]:
        """Compile + store every missing tag (the kit's ``warm`` verb); returns ``{tag: event}``."""
        return {tag: self.load_or_compile(tag, lower)[1] for tag, lower in tags_and_lowers.items()}

    # ------------------------------------------------------------------------------------------------------------- census
    def stats(self) -> dict:
        d = dict(self.stats_)
        d.update(name=self.name, directory=self.directory, key=self._hash, stack=self._stack, in_memory=sorted(self._mem), state=self.state())
        return d

    def state(self) -> str:
        s = self.stats_
        if s["loads"]:
            return "loaded"
        if s["stores"]:
            return "stored"
        if s["compiles"]:
            return "compiled"
        return "idle"

    def evidence_fields(self) -> List[Tuple[str, Any]]:
        s = self.stats_
        return [("lever", self.name), ("state", self.state()), ("loads", s["loads"]), ("stores", s["stores"]), ("compiles", s["compiles"]),
                ("unstored", s["unstored"]), ("stale", s["stale"]), ("load_failures", s["load_failures"]), ("key", self._hash),
                ("load_s", round(s["load_s"], 1)), ("compile_s", round(s["compile_s"], 1))]

    def evidence_line(self, tag: str, name: Optional[str] = None) -> str:
        """``[<tag>] LEVER name=<name> state=on impl=xla_exec_cache cache=<loaded|stored|compiled|idle> loads=… stores=… compiles=… unstored=… stale=… …
        [cache=miss(n)]``."""
        fields = self.evidence_fields()
        detail = [(k, v) for k, v in fields if k not in ("lever", "state")]
        head = [("name", name or self.name), ("state", "on"), ("impl", "xla_exec_cache"), ("origin", "core"), ("cache", self.state())]
        return report.with_words(f"{report.prefix(tag)} LEVER {report.kv(*(head + detail))}", self.words())

    def misses(self) -> int:
        """Executables of this process that did not come from the stored artefact: unstored compiles + failed loads."""
        return int(self.stats_["unstored"]) + int(self.stats_["load_failures"])

    def words(self) -> List[str]:
        """``["cache=miss(<n>)"]`` after a miss under ``exact=False`` (module contract), else ``[]`` — the named uncertainty the run
        proceeded under; exit 0."""
        n = self.misses()
        return [report.word("cache", "miss", n)] if n else []
