"""Lever `lowercache` — the design step's two programs (ColabDesign's `fn` = jit(_model) and `grad_fn` = jit(value_and_grad(_model)),
colabdesign/af/model.py `_get_model`) skip jax's TRACE + LOWER + cache-key hash on every model build after the first one of a configuration:
the COMPILED executable of each program is serialized per call signature next to the compile cache and, in any later process — or any later
trajectory of the same process (BindCraft builds a fresh `mk_afdesign_model` per trajectory, so stock re-traces and re-lowers the AlphaFold
graph for every trajectory even when every executable is already compiled) — deserialized and CALLED DIRECTLY
(`jax.experimental.serialize_executable.serialize` / `deserialize_and_load`: the loaded object is a `jax.stages.Compiled`, invoked with the
same pytree arguments the jitted function takes).

What it removes. With `compilecache` warm, the first `grad_fn` / `fn` call of a trajectory still costs jax's Python-side tracing of the
multimer model (haiku), the StableHLO lowering — which for the kit's Pallas kernels includes the kernels' own lowering — and the
persistent-cache key (a hash over the lowered module): measured 19 s (exact) / 33 s (fast) per trajectory at 275 tokens on H100, every
trajectory. With this lever a configuration met before costs the executable load (seconds).

Numerics class: exact. The serialized executable IS the executable the compile produced (the same bytes `compilecache` stores and loads);
a loaded executable replays the design step bit for bit. The one correctness obligation is the KEY: an entry may only be served to a call
whose traced program would be identical. The key (sha256) covers
  * the stack: jax / jaxlib / PJRT-plugin versions, the GPU product (`compilecache`'s stack key), XLA_FLAGS and every JAX_* / XLA_* /
    MODEL_OPT_* / COLABDESIGN_OPT_* / AF_PALLAS_* / NVIDIA_TF32_OVERRIDE variable of the environment whose value is not a path (cache roots and
    tree homes decide no program; the code is pinned by content hash below), jax's matmul-precision / x64 config;
  * the code: opt_core's version, this package's version AND a content hash of every .py file of `colabdesign` (stock, as vendored),
    `colabdesign_opt` and `opt_core` importable in this process (`code_id`) — an edited tree is a new key by construction;
  * the lever set in force (levers.installed(), the mode word) — the levers decide the traced program;
  * the model: `mk_af_model._args`, `protocol`, the lengths, the haiku model config text (`str(cfg)`: subbatch, recycles, dropout, heads…),
    the qualified names of every model callback (pre / post / loss: BindCraft's added losses are traced code), the program tag (fn | grad_fn);
  * the call: the argument pytree structure (dict keys included) and every leaf's shape / dtype / weak-type — exactly what jit specialises on
    (VALUES are traced, never baked: opt weights, temperatures, recycle counts, keys and parameters may all differ call to call).
A call whose signature has no entry is traced, lowered and compiled through the wrapped jit object as stock does (`compilecache` serves the
compile when warm) and its executable stored (write-to-temp + rename; a store failure is counted, never raised). A loaded executable that
refuses the arguments (jax raises TypeError on any abstract mismatch) or an entry that fails to deserialize is discarded for the process and
the call takes the traced path (`fallbacks`). `COLABDESIGN_OPT_LOWERCACHE=relower` additionally traces + lowers on every LOAD and compares the
lowered StableHLO text's sha256 with the entry's (`relower=same:<n>` | `relower=DIFFERENT:<n>` — a mismatch discards the entry and takes the
traced path; it is the key's self-test, run in the kit's GPU test, and costs the trace + lower it exists to remove).

The directory (never a flag of the kit): `<compilecache's directory>/../lowered/` — i.e. `<XDG_CACHE_HOME | ~/.cache>/colabdesign_opt/pcc/
<stack key>/lowered/<key>/{executable.bin, trees.pkl, meta.json}` (or beside an adopted JAX_COMPILATION_CACHE_DIR). One entry per (program,
signature); `meta.json` names the human-readable key material. A directory that cannot be created / written steps the lever aside by name
(`LowerCacheError`: state=skipped reason=cannot_run). `executable.bin` and `trees.pkl` are both pickles, so each is one line `sha256:<hex>` over
the rest of the file, then the bytes, and nothing of an entry is unpickled before both lines match and neither file nor the entry directory
is writable by group or other or owned by another account (this uid or root; uid 0 accepts any owner — a container's root reading a
bind-mounted host directory). An entry that fails is REFUSED by name on stderr (what, why, the fix; `why=refused/<program>:…` on the LEVER
line) and is then the absent entry: traced, compiled, stored again (an entry whose bytes do not match is removed first, so the store lands).
The sha256 line catches truncation and corruption, not a writer of the directory — hence the owner / mode rule.

Evidence: ONE LEVER line at process exit (the package's exit printer), install fields + the census of this process:

    [colabdesign-opt] LEVER name=lowercache state=on impl=serialize_executable@kit origin=kit numerics=exact dir=<path> calls=<n> memo_hits=<n>
        loads=<n> stores=<n> traced=<n> fallbacks=<n> load_s=<s> retrace_s=<s> traced_s=<s> store_s=<s> mb_stored=<n>
        entries=<at install>-><at exit> relower=<off|same:n|DIFFERENT:n> why=<none|where:Exception:text;…> source=exit pid=<pid>

`calls` = first calls of a (model, program, signature) this process handled (later calls of the same signature go straight to the executable
and are not counted); `memo_hits` = served from this process's memo (an earlier model build met the key: BindCraft's trajectories 2..N);
`loads` = served from disk (their seconds in `load_s`), `traced` = traced + lowered + compiled here (`traced_s`, the cost a
load removes), `stores` = entries written; `calls>0` with the line present = applied.

Protocol (registry.py): NUMERICS, REFUSALS, install() / installed() / uninstall() / off_line(reason) / evidence(); wraps
`colabdesign.af.model.mk_af_model._get_model` (composes with `nosub`, which calls `_get_model` itself: every program it builds is wrapped).

    from colabdesign_opt import lowercache
    lowercache.install()            # idempotent; before the first model is built
    lowercache.evidence()           # {"installed", "dir", "calls", "loads", "stores", "traced", "fallbacks", "load_s", "traced_s", ...}
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import sys
import threading
import time
from typing import Any, Callable, Dict, MutableMapping, Optional

import opt_core
from opt_core import gates as _gates
from opt_core import report as _report

from . import compilecache_jax as _cc
from .names import TAG

LEVER = "lowercache"
NUMERICS = "exact"                                            # registry.LEVERS[lowercache].numerics: a loaded executable IS the compiled executable (bitwise)
IMPL = "serialize_executable@kit"
SUBDIR = "lowered"                                            # <compilecache dir>/../lowered
ENV_MODE = "COLABDESIGN_OPT_LOWERCACHE"                       # unset | relower
ENV_PREFIXES = ("JAX_", "XLA_", "MODEL_OPT_", "COLABDESIGN_OPT_", "AF_PALLAS_", "NVIDIA_TF32_OVERRIDE", "CUDA_VISIBLE_DEVICES")
# Variables under ENV_PREFIXES whose VALUE is a filesystem location and never a numerics/compile setting: excluded from the key BY NAME (so a
# composite variable such as XLA_FLAGS always stays in the key even when it carries a path). The traced code is pinned by `code_id`, the caches
# by the stack key; where they live on disk does not change the program.
ENV_PATH_NAMES = ("MODEL_OPT_JIT_ROOT", "MODEL_OPT_JIT_IMAGE", "MODEL_OPT_HOME", "MODEL_OPT_TREE", "COLABDESIGN_OPT_HOME", "JAX_COMPILATION_CACHE_DIR")
ENV_PATH_SUFFIXES = ("_DIR", "_ROOT", "_HOME", "_PATH", "_FILE")


def env_in_key(name: str) -> bool:
    """True when an environment variable of this NAME enters the executable key (prefix in scope, not a path-valued name, not this lever's word)."""
    if not name.startswith(ENV_PREFIXES) or name == ENV_MODE:
        return False
    if name in ENV_PATH_NAMES or name.endswith(ENV_PATH_SUFFIXES):
        return False
    return True
PROGRAMS = ("fn", "grad_fn")
# XLA-FFI custom-call targets a persisted executable may name must be REGISTERED in the process before the executable is loaded (jax's own
# compile cache never meets this: it loads after tracing, and tracing a kernel registers its target). Per lever that ships such a kernel: the
# registrar to call (module:function; idempotent, cached by the kernel package) — called once, before the first load, when that lever is installed.
FFI_REGISTRARS = {"txla": "opt_core.kernels.triattn_xla._launch:load"}
FILES = {"exe": "executable.bin", "trees": "trees.pkl", "meta": "meta.json"}
DIGEST_HEAD = b"sha256:"                                           # line 1 of executable.bin and of trees.pkl: DIGEST_HEAD + <64 hex> + b"\n", over every byte after it

_LOCK = threading.Lock()
_ZERO = {"calls": 0, "memo_hits": 0, "loads": 0, "stores": 0, "traced": 0, "fallbacks": 0, "store_errors": 0, "load_s": 0.0, "retrace_s": 0.0, "traced_s": 0.0, "store_s": 0.0, "relower_same": 0, "relower_diff": 0, "bytes_stored": 0}
_STATE: dict = {"installed": False, "info": None, "orig": None, "surface": None, "counts": dict(_ZERO), "static": None, "relower": False, "reasons": []}
_MEMO: Dict[str, Any] = {}
_TRACED_TAGS: set = set()                                          # programs traced once in this process for the trace-time census (Persisted._first_call)                                     # key -> loaded / compiled executable: every later model build of this PROCESS reuses it (no disk, no trace)


class LowerCacheError(RuntimeError):
    """The lever cannot be put in force in this process (directory not creatable / writable, jax's serialization module or ColabDesign
    absent): it steps aside by name (``cannot_run``) and the mode runs the rest of its set."""
    cannot_run = True


REFUSALS = (LowerCacheError,)


# ------------------------------------------------------------------------------------------------------------------ the directory

def resolve_dir(environ: Optional[MutableMapping[str, str]] = None) -> str:
    """The entries' directory: a sibling `lowered/` of compilecache's executable directory (same stack key, same adoption rule)."""
    where = _cc.resolve_dir(os.environ if environ is None else environ)
    return os.path.join(os.path.dirname(where["dir"].rstrip(os.sep)), SUBDIR)


def _writable_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, f".probe.{os.getpid()}")
        with open(probe, "w") as fh:
            fh.write("x")
        os.remove(probe)
    except OSError as e:
        raise _gates.cannot_run(LowerCacheError(f"{LEVER}: directory {path!r} is not writable ({e})")) from e


def _entries(path: str) -> int:
    try:
        return sum(1 for n in os.listdir(path) if os.path.isfile(os.path.join(path, n, FILES["meta"])))
    except OSError:
        return 0


# ------------------------------------------------------------------------------------------------------------------ the key

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _code_id() -> Dict[str, str]:
    """Content hash per traced-code package importable here (colabdesign, colabdesign_opt, opt_core): sha256 over (relative path, file sha)
    of every .py file under the package directory (tests excluded). An edited file is a new id."""
    out = {}
    for name in ("colabdesign", "colabdesign_opt", "opt_core"):
        mod = sys.modules.get(name)
        if mod is None:
            try:
                mod = __import__(name)
            except Exception:  # noqa: BLE001
                out[name] = "absent"; continue
        root = os.path.dirname(getattr(mod, "__file__", "") or "")
        h = hashlib.sha256(); n = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in ("tests", "__pycache__"))
            for fn in sorted(filenames):
                if fn.endswith(".py"):
                    p = os.path.join(dirpath, fn)
                    try:
                        with open(p, "rb") as fh:
                            h.update(os.path.relpath(p, root).encode()); h.update(hashlib.sha256(fh.read()).digest()); n += 1
                    except OSError:
                        h.update(b"unreadable:" + p.encode())
        out[name] = f"{h.hexdigest()[:16]}:{n}"
    return out


def _static_context() -> Dict[str, Any]:
    """Everything of the PROCESS that decides the traced program besides the model and the call (computed once)."""
    if _STATE["static"] is not None:
        return _STATE["static"]
    import jax  # noqa: PLC0415
    try:
        import jaxlib  # noqa: PLC0415
        jaxlib_v = getattr(jaxlib, "__version__", "?")
    except Exception:  # noqa: BLE001
        jaxlib_v = "?"
    try:
        dev = jax.devices()[0]; device = f"{dev.platform}:{getattr(dev, 'device_kind', '?')}"
    except Exception:  # noqa: BLE001
        device = "?"
    cfg = {}
    for name in ("jax_default_matmul_precision", "jax_enable_x64", "jax_disable_jit", "jax_debug_nans", "jax_numpy_rank_promotion", "jax_default_dtype_bits"):
        try:
            cfg[name] = str(getattr(jax.config, name))
        except Exception:  # noqa: BLE001
            cfg[name] = "?"
    env = {k: v for k, v in sorted(os.environ.items()) if env_in_key(k) and k != _cc.ENV_DIR}   # path-valued variables excluded BY NAME (env_in_key); every other JAX_/XLA_/MODEL_OPT_/… variable — XLA_FLAGS included — is key material
    try:
        from . import levers as _levers  # noqa: PLC0415
        lever_state = {"mode": (_levers._STATE.get("mode") if hasattr(_levers, "_STATE") else None), "installed": list(_levers.installed())}
    except Exception:  # noqa: BLE001
        lever_state = {"mode": None, "installed": "?"}
    from . import __version__ as kit_version  # noqa: PLC0415
    ctx = {"jax": jax.__version__, "jaxlib": jaxlib_v, "stack_key": (_STATE["info"] or {}).get("key"), "device": device, "jax_config": cfg, "env": env,
           "opt_core": opt_core.__version__, "colabdesign_opt": kit_version, "code_id": _code_id(), "levers": lever_state}
    _STATE["static"] = ctx
    return ctx


def _model_context(af, cfg, tag: str) -> Dict[str, Any]:
    """What of THIS model object decides the traced program (values that jit bakes because `_model` closes over them)."""
    def names(fns):
        return [f"{getattr(f, '__module__', '?')}.{getattr(f, '__qualname__', repr(f))}" for f in (fns or [])]
    cb = getattr(af, "_callbacks", {}) or {}
    model_cb = cb.get("model", {}) if isinstance(cb, dict) else {}
    return {"program": tag, "args": getattr(af, "_args", None), "protocol": getattr(af, "protocol", None),
            "lengths": {k: getattr(af, k, None) for k in ("_len", "_lengths", "_target_len", "_binder_len", "_num")},
            "callbacks": {k: names(v) for k, v in (model_cb.items() if isinstance(model_cb, dict) else [])},
            "cfg_sha": _sha(str(cfg))}


def _signature(args) -> str:
    """The abstract signature jit specialises on: pytree structure + every leaf's dtype / shape / weak type."""
    import jax  # noqa: PLC0415
    leaves, tree = jax.tree_util.tree_flatten(args)
    parts = [str(tree)]
    for x in leaves:
        try:
            a = jax.core.get_aval(x)
            parts.append(f"{a.dtype.name}{list(a.shape)}{'~' if getattr(a, 'weak_type', False) else ''}")
        except Exception:  # noqa: BLE001 — a non-array leaf (jit would treat it the same way every call): its type and value
            parts.append(f"{type(x).__name__}:{x!r}")
    return "|".join(parts)


def key_of(model_ctx: Dict[str, Any], signature: str) -> Dict[str, Any]:
    material = {"static": _static_context(), "model": model_ctx, "signature_sha": _sha(signature)}
    text = json.dumps(material, sort_keys=True, default=str)
    return {"key": _sha(text), "material": material}


# ------------------------------------------------------------------------------------------------------------------ the store

def _entry_dir(key: str) -> str:
    return os.path.join(_STATE["info"]["dir"], key)


def _register_ffi_targets() -> None:
    """Register the custom-call targets of the installed kernel levers (FFI_REGISTRARS) once per process, before any executable is loaded."""
    if _STATE.get("ffi_done"):
        return
    _STATE["ffi_done"] = True
    try:
        from . import levers as _levers  # noqa: PLC0415
        live = set(_levers.installed())
    except Exception:  # noqa: BLE001
        live = set(FFI_REGISTRARS)
    for lever, spec in FFI_REGISTRARS.items():
        if lever not in live:
            continue
        mod, fn = spec.split(":")
        try:
            getattr(__import__(mod, fromlist=[fn]), fn)()
            _STATE.setdefault("ffi_registered", []).append(lever)
        except Exception as e:  # noqa: BLE001 — a kernel that cannot register refuses by name at its own first use; the load falls back if it needed it
            with _LOCK:
                if len(_STATE["reasons"]) < 8: _STATE["reasons"].append(f"ffi/{lever}:{type(e).__name__}:{str(e)[:80]}")


class Refused(Exception):
    """An entry that must not be unpickled (the reason and the fix) — named on stderr, then exactly the absent entry. ``rewrite``: its bytes
    are not what was stored, so :func:`_load` removes it and the traced program's store lands in its place."""

    def __init__(self, why: str, rewrite: bool = False):
        super().__init__(why)
        self.rewrite = rewrite


def _checked(path: str) -> bytes:
    """The bytes after ``path``'s sha256 line, or :class:`Refused` — nothing of an entry is unpickled before this holds for both of its pickles.
    Neither the file nor its directory may be writable by group or other, and both must belong to this uid or to root (uid 0 accepts any
    owner: a container's root reading a bind-mounted host directory); then line 1 must be the sha256 of the rest."""
    uid, d = os.geteuid(), os.path.dirname(os.path.abspath(path))
    with open(path, "rb") as fh:
        for what, p, s in (("directory", d, os.stat(d)), ("file", path, os.fstat(fh.fileno()))):
            if s.st_mode & 0o022:
                raise Refused(f"{what} {p} is writable by group or other (mode {s.st_mode & 0o7777:04o}); fix: chmod go-w {p}, or name a cache root only you can write (MODEL_OPT_JIT_ROOT)")
            if uid != 0 and s.st_uid not in (uid, 0):
                raise Refused(f"{what} {p} belongs to uid {s.st_uid}, not to this process (uid {uid}) or root; fix: name a cache root of your own (MODEL_OPT_JIT_ROOT), or chown {p}")
        head, body = fh.readline(80), fh.read()
    if head != DIGEST_HEAD + hashlib.sha256(body).hexdigest().encode() + b"\n":
        raise Refused(f"file {path}: its first line is not the sha256 of the rest (truncated, corrupted, or written without one); fix: none needed, the entry is removed and rewritten now", rewrite=True)
    return body


def _load(key: str):
    """(compiled, meta) from disk or (None, None); :class:`Refused` for an entry that must not be unpickled."""
    d = _entry_dir(key)
    meta_p, exe_p, trees_p = (os.path.join(d, FILES[k]) for k in ("meta", "exe", "trees"))
    if not (os.path.isfile(meta_p) and os.path.isfile(exe_p) and os.path.isfile(trees_p)):
        return None, None
    from jax.experimental.serialize_executable import deserialize_and_load  # noqa: PLC0415
    try:
        blob, trees = _checked(exe_p), _checked(trees_p)          # both pickles, before either is unpickled
    except Refused as e:
        if e.rewrite:                                             # _store's os.replace keeps an existing entry: this one leaves first
            for n in FILES.values():
                try: os.remove(os.path.join(d, n))
                except OSError: pass
            try: os.rmdir(d)
            except OSError: pass
        raise
    _register_ffi_targets()
    with open(meta_p) as fh:
        meta = json.load(fh)
    in_tree, out_tree = pickle.loads(trees)
    return deserialize_and_load(blob, in_tree, out_tree), meta


def _store(key: str, compiled, meta: Dict[str, Any]) -> int:
    from jax.experimental.serialize_executable import serialize  # noqa: PLC0415
    blob, in_tree, out_tree = serialize(compiled)
    d = _entry_dir(key); tmp = f"{d}.tmp.{os.getpid()}.{threading.get_ident()}"
    os.makedirs(tmp, mode=0o755, exist_ok=True)                   # 0755 / 0644 whatever the umask: what _checked accepts at the next load
    for name, body in ((FILES["exe"], blob), (FILES["trees"], pickle.dumps((in_tree, out_tree), protocol=pickle.HIGHEST_PROTOCOL))):
        with os.fdopen(os.open(os.path.join(tmp, name), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644), "wb") as fh:
            fh.write(DIGEST_HEAD + hashlib.sha256(body).hexdigest().encode() + b"\n")
            fh.write(body)
    with open(os.path.join(tmp, FILES["meta"]), "w") as fh:
        json.dump(meta, fh, sort_keys=True, default=str, indent=1)
    try:
        os.replace(tmp, d)                                    # atomic on POSIX when d does not exist; if a concurrent writer won, keep theirs
    except OSError:
        for n in FILES.values():
            try: os.remove(os.path.join(tmp, n))
            except OSError: pass
        try: os.rmdir(tmp)
        except OSError: pass
    return len(blob)


def _same_outputs(a, b) -> bool:
    """True iff two output pytrees are leaf-for-leaf bitwise identical (structure and dtypes included)."""
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    la, ta = jax.tree_util.tree_flatten(a); lb, tb = jax.tree_util.tree_flatten(b)
    if ta != tb or len(la) != len(lb):
        return False
    return all(np.asarray(x).dtype == np.asarray(y).dtype and np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


def _note_fallback(where: str, exc: BaseException) -> None:
    with _LOCK:
        _STATE["counts"]["fallbacks"] += 1
        if len(_STATE["reasons"]) < 8:
            _STATE["reasons"].append(f"{where}:{type(exc).__name__}:{str(exc).strip().splitlines()[0][:120] if str(exc).strip() else ''}")


def _hlo_sha(lowered) -> str:
    try:
        return _sha(lowered.as_text())
    except Exception:  # noqa: BLE001
        return "unavailable"


class Persisted:
    """A jitted ColabDesign program whose compiled executable persists per call signature (see the module doc). Calls with a signature
    seen before in this object go straight to the loaded / compiled executable; the first call of a signature loads or traces."""

    def __init__(self, jitted: Callable, tag: str, model_ctx: Dict[str, Any]):
        self._jitted, self._tag, self._ctx = jitted, tag, model_ctx
        self._by_sig: Dict[str, Any] = {}                      # signature -> Compiled | False (= take the jit path for this signature)
        self._keys: Dict[str, str] = {}                        # signature -> key

    def __getattr__(self, name):                              # lower / eval_shape / … of the wrapped jit object stay reachable
        return getattr(self._jitted, name)

    def __call__(self, *args):
        sig = _signature(args)
        exe = self._by_sig.get(sig)
        if exe is None:
            exe = self._first_call(sig, args)
            self._by_sig[sig] = exe
        if exe is False:
            return self._jitted(*args)
        try:
            return exe(*args)
        except TypeError as e:                                # the executable refuses these arguments: never again for this signature (nor from the memo)
            _note_fallback(f"call/{self._tag}", e)
            _MEMO.pop(self._keys.get(sig, ""), None)
            self._by_sig[sig] = False
            return self._jitted(*args)

    def _first_call(self, sig: str, args):
        k = key_of(self._ctx, sig); key = k["key"]; self._keys[sig] = key
        with _LOCK:
            _STATE["counts"]["calls"] += 1
            memo = _MEMO.get(key)
        if memo is not None and not _STATE["relower"]:            # met earlier in THIS process (a previous trajectory's model): nothing to load or trace
            with _LOCK:
                _STATE["counts"]["memo_hits"] += 1
            return memo
        t0 = time.perf_counter()
        try:
            exe, meta = _load(key)
        except Refused as e:                                      # named once on stderr and in the line's why=; from here on exactly the absent entry
            exe, meta = None, None
            with _LOCK:
                if len(_STATE["reasons"]) < 8: _STATE["reasons"].append(f"refused/{self._tag}:{str(e).split(';')[0][:120]}")
            print(f"[{TAG}] {LEVER}: REFUSED cache entry: {e} — not unpickled, treated as absent (the program is traced and stored again)", file=sys.stderr, flush=True)
        except Exception as e:  # noqa: BLE001 — a corrupt / foreign entry: trace instead, overwrite below
            exe, meta = None, None
            _note_fallback(f"load/{self._tag}", e)
        if exe is not None:
            dt = time.perf_counter() - t0
            t1 = time.perf_counter()
            if _STATE["relower"]:                                 # the self-check: lower + compile afresh too and compare the FIRST CALL's outputs bit for bit
                try:
                    ref = self._jitted.lower(*args).compile()
                    same = _same_outputs(exe(*args), ref(*args))
                    with _LOCK:
                        _STATE["counts"]["relower_same" if same else "relower_diff"] += 1
                    if not same:
                        with _LOCK:
                            if len(_STATE["reasons"]) < 8: _STATE["reasons"].append(f"relower/{self._tag}:outputs_differ:the_fresh_executable_serves")
                        exe = ref                                  # serve the fresh one; the line names the difference
                except Exception as e:  # noqa: BLE001
                    _note_fallback(f"relower/{self._tag}", e); exe = None
            elif self._tag not in _TRACED_TAGS:                   # a loaded executable never ran the model's Python: trace the program ONCE per process so every lever
                try:                                              # that counts at trace time (the kernels' served / fallback census) reads as in a traced run
                    if hasattr(self._jitted, "trace"):
                        self._jitted.trace(*args)
                    _TRACED_TAGS.add(self._tag)
                except Exception as e:  # noqa: BLE001 — census only; the executable is unaffected
                    with _LOCK:
                        if len(_STATE["reasons"]) < 8: _STATE["reasons"].append(f"retrace/{self._tag}:{type(e).__name__}:{str(e)[:80]}")
            with _LOCK:
                _STATE["counts"]["retrace_s"] += time.perf_counter() - t1
            if exe is not None:
                with _LOCK:
                    _STATE["counts"]["loads"] += 1; _STATE["counts"]["load_s"] += dt; _MEMO[key] = exe
                return exe
        t1 = time.perf_counter()
        try:
            lowered = self._jitted.lower(*args)
            compiled = lowered.compile()
        except Exception as e:  # noqa: BLE001 — whatever jit itself would do with these arguments, let it (the traced path raises stock's error)
            _note_fallback(f"trace/{self._tag}", e)
            return False
        dt = time.perf_counter() - t1
        with _LOCK:
            _STATE["counts"]["traced"] += 1; _STATE["counts"]["traced_s"] += dt
        meta = {"key": key, "program": self._tag, "hlo_sha": _hlo_sha(lowered), "traced_s": round(dt, 2), "signature": sig[:4000],
                "material": k["material"], "pid": os.getpid(), "stored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        t2 = time.perf_counter()
        try:
            n = _store(key, compiled, meta)
            with _LOCK:
                _STATE["counts"]["stores"] += 1; _STATE["counts"]["bytes_stored"] += n; _STATE["counts"]["store_s"] += time.perf_counter() - t2
        except Exception as e:  # noqa: BLE001
            with _LOCK:
                _STATE["counts"]["store_errors"] += 1
                if len(_STATE["reasons"]) < 8: _STATE["reasons"].append(f"store/{self._tag}:{type(e).__name__}:{str(e)[:120]}")
        with _LOCK:
            _MEMO[key] = compiled
        return compiled


# ------------------------------------------------------------------------------------------------------------------ install

def _wrap_get_model(orig):
    def _get_model(self, cfg, callback=None, *a, **kw):
        out = orig(self, cfg, callback, *a, **kw) if (callback is not None or a or kw) else orig(self, cfg)   # ColabDesign: (cfg, callback=None); a surface without the parameter is called as it defines
        try:
            for tag in PROGRAMS:
                if tag in out and not isinstance(out[tag], Persisted):
                    out[tag] = Persisted(out[tag], tag, _model_context(self, cfg, tag))
        except Exception as e:  # noqa: BLE001 — never break a model build: the programs stay jax's
            _note_fallback("wrap", e)
        return out
    _get_model._lowercache_orig = orig
    return _get_model


def install(environ: Optional[MutableMapping[str, str]] = None, *, surface: Optional[type] = None) -> dict:
    """Wrap `mk_af_model._get_model` (or `surface._get_model`: the class ColabDesign's `_get_model` lives on — tests hand their own) so every design
    program built from here on persists / loads its executable. Idempotent. Returns the record {"dir", "entries_install", "relower", "patched"}."""
    with _LOCK:
        if _STATE["installed"]:
            return dict(_STATE["info"])
        environ = os.environ if environ is None else environ
        try:
            import jax  # noqa: F401, PLC0415
            from jax.experimental import serialize_executable as _se  # noqa: F401, PLC0415
        except Exception as e:  # noqa: BLE001
            raise _gates.cannot_run(LowerCacheError(f"{LEVER}: jax's serialize_executable is not importable ({e!r})")) from e
        if surface is None:
            try:
                from colabdesign.af import model as _afm  # noqa: PLC0415
            except Exception as e:  # noqa: BLE001
                raise _gates.cannot_run(LowerCacheError(f"{LEVER}: colabdesign.af.model is not importable ({e!r})")) from e
            surface = _afm.mk_af_model
        if not callable(getattr(surface, "_get_model", None)):
            raise _gates.cannot_run(LowerCacheError(f"{LEVER}: {getattr(surface, '__name__', surface)!r} defines no `_get_model` to wrap"))
        d = resolve_dir(environ)
        _writable_dir(d)
        key = None
        try:
            key = _cc.resolve_dir(environ).get("key")
        except Exception:  # noqa: BLE001
            pass
        orig = surface._get_model
        if getattr(orig, "_lowercache_orig", None) is None:
            surface._get_model = _wrap_get_model(orig)
        _STATE["orig"] = orig; _STATE["surface"] = surface
        info = {"dir": d, "entries_install": _entries(d), "relower": (environ.get(ENV_MODE, "").strip().lower() == "relower"), "patched": f"{getattr(surface, '__name__', 'surface')}._get_model", "key": key}
        _STATE.update(installed=True, info=info, relower=info["relower"])
        _register_exit_line()
        return dict(info)


def _register_exit_line() -> None:
    try:
        from .kernels import register_exit_line  # noqa: PLC0415
        register_exit_line(LEVER, exit_line)
    except (ImportError, ValueError):
        _report.register_exit_tally(TAG + "." + LEVER, exit_line)


def installed() -> bool:
    return bool(_STATE["installed"])


def uninstall() -> None:
    """Restore `mk_af_model._get_model`; models already built keep their wrapped programs; entries stay on disk."""
    with _LOCK:
        if not _STATE["installed"]:
            return
        surface = _STATE.get("surface")
        try:
            cur = surface._get_model if surface is not None else None
            orig = getattr(cur, "_lowercache_orig", None)
            if orig is not None:
                surface._get_model = orig
        except Exception:  # noqa: BLE001
            pass
        _STATE.update(installed=False, info=None, orig=None, surface=None)


def off_line(reason: str) -> str:
    return _report.lever_line(TAG, LEVER, "off", reason=reason, impl=IMPL, origin="kit", numerics=NUMERICS)


# ------------------------------------------------------------------------------------------------------------------ evidence

def evidence() -> dict:
    info = _STATE["info"] or {}
    out = {"installed": bool(_STATE["installed"]), **info, **_STATE["counts"]}
    out["entries_now"] = _entries(info["dir"]) if info.get("dir") else 0
    out["reasons"] = list(_STATE.get("reasons") or [])
    return out


def _relower_word(ev: dict) -> str:
    if ev.get("relower_diff"):
        return f"DIFFERENT:{ev['relower_diff']}"
    if ev.get("relower"):
        return f"same:{ev.get('relower_same', 0)}"
    return "off"


def line_of(ev: Optional[dict] = None) -> str:
    ev = evidence() if ev is None else ev
    if not ev.get("installed"):
        return _report.lever_line(TAG, LEVER, "off", impl=IMPL, origin="kit", numerics=NUMERICS)
    return _report.lever_line(
        TAG, LEVER, "on", impl=IMPL, origin="kit", numerics=NUMERICS, dir=_cc._blank_free(ev["dir"]),
        calls=ev["calls"], memo_hits=ev.get("memo_hits", 0), loads=ev["loads"], stores=ev["stores"], traced=ev["traced"], fallbacks=ev["fallbacks"] + ev.get("store_errors", 0),
        load_s=f"{ev['load_s']:.1f}", retrace_s=f"{ev.get('retrace_s', 0.0):.1f}", traced_s=f"{ev['traced_s']:.1f}", store_s=f"{ev['store_s']:.1f}", mb_stored=f"{ev.get('bytes_stored', 0) / 1e6:.0f}",
        entries=f"{ev['entries_install']}->{ev['entries_now']}", relower=_relower_word(ev), why=_cc._blank_free(";".join(ev.get("reasons") or []) or "none"), source="exit", pid=os.getpid())


def exit_line() -> str:
    return line_of()


def reset_for_tests() -> None:
    uninstall()
    with _LOCK:
        _STATE.update(installed=False, info=None, orig=None, counts=dict(_ZERO), static=None, relower=False, reasons=[], ffi_done=False, ffi_registered=[]); _MEMO.clear(); _TRACED_TAGS.clear()
