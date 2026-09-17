"""programs — the model PROGRAM of a design, prepared ahead of its call (L7) and kept across processes (L13). Runs INSIDE the driver.

A "program" is the compiled XLA executable of the jitted AlphaFold apply for one input signature (the pytree structure and the shape /
dtype of every argument: one per distinct residue count and lever set). Stock jax builds it lazily inside the first ``apply(...)`` call
of each new signature: TRACE (haiku runs the Python model code) -> LOWER (StableHLO) -> COMPILE (XLA; or a load from the persistent
compilation cache, the ccache lever) -> run. jax's ahead-of-time API gives the same executable without the call
(``jit(f).trace(*a).lower().compile()`` — the object jit itself would run: the same output bytes), so:

  * L7 ``-precompile N`` (predict_pdb.py precompile_shapes): the programs of every distinct length of the run are prepared BEFORE the loop
    on N host threads — no forward is run, nothing is discarded, the featurized inputs are kept for the loop; the loop calls the prepared
    executables (``AF2_runner.model_program``: one per signature for the rest of the process).
  * L13 ``-program_cache DIR`` (:class:`Store`): each program compiled in this process is SERIALIZED under DIR
    (``jax.experimental.serialize_executable``: the executable's own bytes + its argument layout) with a JSON companion (the output pytree
    skeleton, the trace-time kernel census of the levers inside it, sizes, the stack); a later process with the same key LOADS it — no trace,
    no lowering, no compilation — and runs it. The key (:func:`static_key` + the input signature) names everything that decides the program:
    the code of the checkout (every .py under the driver directory: the patched alphafold, the driver), of this package and of the shared
    core (their files' digests), the lever arguments of the run, jax / jaxlib / the CUDA plugin / haiku / numpy / python versions, the device
    kind, ``XLA_FLAGS`` and every other ``XLA_*`` / ``JAX_*`` variable that is not the allocator's or the compilation cache's location. A key
    that does not match is a cold program by name (traced, compiled, stored), never a stale load; a file that does not load is named
    (``load_failed:<Type>`` | ``load_failed:custom_call_unregistered``) and the program is traced instead — the run never depends on the store.
    A stored program is one line ``sha256:<hex>`` over the rest of the file, then the executable's bytes (``deserialize_and_load`` unpickles
    them): nothing is deserialized before that line matches and :func:`refusal` passes the file and its directory. A file that fails either
    is REFUSED by name on stderr (what, why, the fix; ``load_refused`` on the L13 line) and is then the absent file — traced, compiled, stored
    again. The line catches truncation and corruption, not a writer of the directory — hence the owner / mode rule.
The key also carries the tree's / kit's environment words (MODEL_OPT_*, AF2IG_OPT_* minus cache roots) and the PROVIDER decision
(``providers``: the triattn_xla bridge on + row + length rows | aside + reason) — a Pallas-provider process and a bridge-provider process have two keys.
Numerics: none. The loaded executable is the one this stack compiled from this code with these flags: the outputs are the bytes the tracing
process wrote (exact stays bitwise with the stock line under ``--det 1``: cold store, warm store, or no store).
Kernel-lever evidence across processes: the fast line's kernel levers (L8, L10, L11) are evidenced by their TRACE-TIME census (calls served
onto the kernel while haiku traced the model). A loaded program was traced in an earlier process: its census travelled with it (the JSON
companion, recorded around that trace under a lock) and is REPLAYED into this process's ledgers when the program loads (``replayed:L<n>``
shape keys), so the census of the run still says which kernels are inside the executables it ran — named as replayed, never invented.
"""
from __future__ import annotations

import hashlib
import json
import re
import os
import sys
import threading
import time
from typing import Callable, Optional, Tuple

FORMAT = "af2ig-program/1"
SUFFIX, META_SUFFIX = ".xser", ".json"
DIGEST_HEAD = b"sha256:"                                     # line 1 of a stored program: DIGEST_HEAD + <64 hex> + b"\n", over every byte after it (the serialized executable)
TRACE_LOCK = threading.Lock()                     # one model trace at a time (the census delta of a trace must be its own); lowering / compilation run in parallel
IO_ARGS = ("pdbdir", "outpdbdir", "scorefilename", "checkpoint_name", "af2_dir", "timers", "program_cache", "precompile", "sort_by_length", "prefetch", "overlap_output",   # + the host-scheduling levers (L15 -prefetch, L16 -overlap_output; 0.7.3): which thread featurises / writes never changes a traced program
          
           "host_outputs", "device_params", "debug", "fast", "seed", "runlist")          # arguments that never enter a program (paths, scheduling, host-side handling)
ENV_SKIP_PREFIXES = ("XLA_PYTHON_CLIENT_", "JAX_COMPILATION_CACHE_DIR", "JAX_PERSISTENT_CACHE_", "JAX_PLATFORMS", "JAX_DEBUG_LOG", "JAX_LOG_COMPILES",
                     "JAX_TRACEBACK")                                 # allocator / cache location / logging: not program-deciding


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tree_digest(root: str, suffixes=(".py",)) -> Tuple[str, int]:
    """sha256 over the sorted (relpath, file digest) of every file under ``root`` with one of ``suffixes``; (digest, n files)."""
    items = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", ".git"))
        for f in sorted(files):
            if f.endswith(tuple(suffixes)):
                p = os.path.join(base, f)
                with open(p, "rb") as fh:
                    items.append(f"{os.path.relpath(p, root)}={_sha(fh.read())}")
    return _sha("\n".join(items).encode()), len(items)


def signature(args) -> str:
    """The input signature of a call: the pytree structure and every leaf's shape / dtype / weak type — read off the leaves themselves (jax
    Arrays carry .weak_type; numpy arrays are never weak; Python scalars are weak by type), at least as fine as the key jit builds its
    programs on, so two calls with one signature always share one program and a signature never names a program of other shapes."""
    import jax
    leaves, treedef = jax.tree_util.tree_flatten(args)
    parts = [str(treedef)]
    for x in leaves:
        shape, dtype = getattr(x, "shape", None), getattr(x, "dtype", None)
        if shape is not None and dtype is not None:
            parts.append(f"{tuple(int(d) for d in shape)}:{dtype}:{int(bool(getattr(x, 'weak_type', False)))}")
        elif isinstance(x, (bool, int, float, complex)):
            parts.append(f"():{type(x).__name__}:1")
        else:
            parts.append(f"{type(x).__module__}.{type(x).__name__}:{x!r}"[:200])
    return _sha("|".join(parts).encode())


def static_key(args_ns: dict, code_root: str) -> Tuple[str, dict]:
    """(key, facts): everything but the input signature that decides a program (module docstring)."""
    import importlib.metadata as md
    import jax
    import numpy
    facts = {}
    facts["code"], facts["n_code_files"] = tree_digest(code_root)
    here = os.path.dirname(os.path.abspath(__file__))
    facts["af2ig_opt"], facts["n_pkg_files"] = tree_digest(here)
    try:
        import opt_core
        facts["opt_core"], facts["n_core_files"] = tree_digest(os.path.dirname(os.path.abspath(opt_core.__file__)))
        facts["opt_core_version"] = getattr(opt_core, "__version__", "?")
    except ImportError:
        facts["opt_core"] = "absent"

    def ver(d):
        try:
            return md.version(d)
        except md.PackageNotFoundError:
            return "absent"
    facts["stack"] = {d: ver(d) for d in ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "dm-haiku", "numpy", "nvidia-cudnn-cu12", "nvidia-cublas-cu12")}
    facts["python"] = sys.version.split()[0]
    try:
        dev = jax.devices()[0]
        facts["device"] = f"{dev.platform}:{getattr(dev, 'device_kind', '?')}"
        cc = getattr(dev, "compute_capability", None)
        if cc: facts["device"] += f":cc{cc}"
    except Exception as e:  # noqa: BLE001
        facts["device"] = f"unknown:{type(e).__name__}"
    facts["args"] = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else repr(v)) for k, v in sorted(dict(args_ns).items()) if k not in IO_ARGS}
    facts["env"] = {k: _env_word(k, v) for k, v in sorted(os.environ.items()) if (k.startswith(("XLA_", "JAX_", "MODEL_OPT_", "AF2IG_OPT_")) or k in ("AF2_SUBBATCH_SIZE", "NVIDIA_TF32_OVERRIDE"))
                    and not k.startswith(ENV_SKIP_PREFIXES) and not k.endswith(ENV_SKIP_SUFFIXES)}      # the tree's and the kit's own words too (MODEL_OPT_LEVERS_OFF — the core honours `triattn_xla[:row]` at trace time —, AF2IG_OPT_TRIATTN_XLA_MIN_LEN, precision words …); cache ROOTS / DIRS never (a path must not split keys across boxes)
    facts["providers"] = provider_facts(facts.get("device", ""))                                        # the kernel providers this process will trace with (the triattn_xla bridge: on + row + length rows | aside + reason) — two provider states, two keys
    key = _sha(json.dumps({k: facts[k] for k in KEY_FACTS}, sort_keys=True).encode())
    return key, facts


KEY_FACTS = ("code", "af2ig_opt", "opt_core", "stack", "python", "device", "args", "env", "providers")   # the facts hashed into the static key (module docstring)
ENV_SKIP_SUFFIXES = ("_JIT_ROOT", "_CACHE_DIR", "_ROOT", "_DIR")                                            # AF2IG_OPT_JIT_ROOT, MODEL_OPT_JIT_ROOT, AF2IG_OPT_CACHE_DIR …: where things live, not what is traced


def _env_word(k: str, v: str) -> str:
    """An environment value as keyed: MODEL_OPT_LEVERS_OFF as its sorted word set (order / repeats do not split keys), everything else verbatim."""
    if k == "MODEL_OPT_LEVERS_OFF":
        return ",".join(sorted({w for w in re.split(r"[,\s]+", v or "") if w}))
    return v


def provider_facts(device_word: str = "") -> dict:
    """The provider binding of this process, WITHOUT arrays : the tier word the kernel adapters hand opt_core's provider (``AF2IG_OPT_TIER``), L19's
    operand dtype word and the provider package version — two processes that would trace different rows never share a program key (the opt_core version is a
    key fact of its own; a provider cell change ships as a version)."""
    try:
        from af2ig_opt import modes, triattn
        return {"tier": modes.tier_word(), "core_dtype": triattn.core_dtype(), "binding": "opt_core.kernels.pallas.serve"}
    except Exception as e:  # noqa: BLE001
        return {"tier": "unknown", "reason": type(e).__name__}


class Refused(Exception):
    """A stored program that must not be deserialized (the reason and the fix) — named on stderr, then exactly the absent file."""


def refusal(path: str, st) -> Optional[str]:
    """Why the stored program ``path`` (``st`` = its ``os.fstat``) must not be deserialized — reason + fix — or None. ``deserialize_and_load`` unpickles,
    so neither the file nor its directory may be writable by group or other, and both must belong to this uid or to root; uid 0 accepts any
    owner (a container's root reading a bind-mounted host directory)."""
    uid, d = os.geteuid(), os.path.dirname(os.path.abspath(path))
    for what, p, s in (("directory", d, os.stat(d)), ("file", path, st)):
        if s.st_mode & 0o022:
            return f"{what} {p} is writable by group or other (mode {s.st_mode & 0o7777:04o}); fix: chmod go-w {p}, or name a cache root only you can write (AF2IG_OPT_JIT_ROOT)"
        if uid != 0 and s.st_uid not in (uid, 0):
            return f"{what} {p} belongs to uid {s.st_uid}, not to this process (uid {uid}) or root; fix: name a cache root of your own (AF2IG_OPT_JIT_ROOT), or chown {p}"
    return None


def load_failed_word(e: BaseException) -> str:
    """The L13 event word of a program that did not load: ``load_failed:custom_call_unregistered`` when the stored executable names an FFI / custom-call target this
    process has not registered (a bridge program met by a process without the bridge — cannot happen across correct keys, named if it does), else ``load_failed:<Type>``."""
    txt = str(e)
    if "No registered implementation" in txt or "custom call" in txt.lower() and "not found" in txt.lower():
        return "load_failed:custom_call_unregistered"
    return f"load_failed:{type(e).__name__}"


# ---------------------------------------------------------------- pytree skeletons (JSON) for the output structure of a stored program
def _skel(x):
    if isinstance(x, dict):
        return {"__d__": {str(k): _skel(v) for k, v in x.items()}, "__k__": [[str(k), type(k).__name__] for k in x.keys()]}
    if isinstance(x, tuple):
        return {"__t__": [_skel(v) for v in x]}
    if isinstance(x, list):
        return {"__l__": [_skel(v) for v in x]}
    if x is None:
        return {"__n__": 1}
    return 0


def _unskel(s):
    if isinstance(s, dict):
        if "__d__" in s:
            keys = {k: (int(k) if t == "int" else k) for k, t in s["__k__"]}
            return {keys[k]: _unskel(v) for k, v in s["__d__"].items()}
        if "__t__" in s: return tuple(_unskel(v) for v in s["__t__"])
        if "__l__" in s: return [_unskel(v) for v in s["__l__"]]
        if "__n__" in s: return None
    return 0


def aot_compile(apply_fn: Callable, args: tuple, snapshot: Optional[Callable[[], dict]] = None) -> Tuple[Callable, dict]:
    """Trace, lower and compile ``apply_fn`` for ``args`` ahead of the call (jax AOT); (compiled, {source, t_trace, t_lower, t_compile,
    census_before, census_after}). The trace runs under TRACE_LOCK (one at a time) with the kernel-census ``snapshot`` read INSIDE the lock
    right before and right after it — so the delta is this trace's own even when several programs are prepared on parallel threads (read
    outside the lock, a thread's 'after' also counted the traces of the threads that ran while it compiled); lowering and compilation do not."""
    rec = {"source": "traced"}
    t0 = time.time()
    with TRACE_LOCK:
        t1 = time.time()
        rec["census_before"] = snapshot() if snapshot else {}
        tr = getattr(apply_fn, "trace", None)
        if tr is not None:
            traced = tr(*args); t2 = time.time(); lowered = traced.lower()
        else:                                                     # a jax without jit(...).trace: lower() traces and lowers in one step
            traced = None; lowered = apply_fn.lower(*args); t2 = time.time()
        t3 = time.time()
        rec["census_after"] = snapshot() if snapshot else {}
    rec["t_trace_wait"], rec["t_trace"], rec["t_lower"] = round(t1 - t0, 3), round(t2 - t1, 3), round(t3 - t2, 3) if traced is not None else None
    t4 = time.time(); compiled = lowered.compile(); rec["t_compile"] = round(time.time() - t4, 3)
    return compiled, rec


class Store:
    """L13: the serialized programs under ``directory`` (module docstring). One per driver process."""

    def __init__(self, directory: str, args_ns: dict, code_root: str, census_snapshot: Optional[Callable[[], dict]] = None,
                 census_replay: Optional[Callable[[dict, int, str], None]] = None, process_local: Optional[str] = None):
        self.dir = os.path.abspath(directory)
        os.makedirs(self.dir, mode=0o755, exist_ok=True)           # 0755 / 0644 whatever the umask: what refusal() accepts at the next load
        t = time.time()
        self.key, self.facts = static_key(args_ns, code_root)
        self.t_key = round(time.time() - t, 3)
        self.snapshot, self.replay = census_snapshot, census_replay
        self.lock = threading.Lock()
        self.counts = {"loaded": 0, "traced": 0, "stored": 0, "load_failed": 0, "store_failed": 0, "store_skipped": 0}
        self.events = []                                           # load_failed:<Type> / store_failed:<Type> words, in order
        self.process_local = process_local                         # a reason word when this process's programs cannot run in another process (the triattn_xla bridge launches through FFI launch SPECS registered at trace time — a loaded program names spec ids the loading process never registered: `unknown spec id`): nothing is loaded or stored, by name; every program is traced
        self.bytes_loaded = self.bytes_stored = 0

    # -- census
    def probe(self) -> dict:
        return {"dir": self.dir, "static_key": self.key, "t_key": self.t_key, "n_code_files": self.facts.get("n_code_files"), "jax": self.facts["stack"].get("jax"),
                "device": self.facts.get("device"), "entries": len([f for f in os.listdir(self.dir) if f.endswith(SUFFIX)])}

    def census(self) -> dict:
        with self.lock:
            return dict(self.counts, events=list(self.events), dir=self.dir, static_key=self.key[:16], bytes_loaded=self.bytes_loaded, bytes_stored=self.bytes_stored,
                        entries=len([f for f in os.listdir(self.dir) if f.endswith(SUFFIX)]))

    def _paths(self, sig: str):
        k = _sha((self.key + ":" + sig).encode())
        return k, os.path.join(self.dir, k + SUFFIX), os.path.join(self.dir, k + META_SUFFIX)

    # -- the program of a signature
    def program(self, apply_fn: Callable, args: tuple, sig: str, L: int) -> Tuple[Callable, dict]:
        """(callable, record) for ``args``: LOADED from the store when its key is there, else traced / lowered / compiled here and STORED.
        record: {source: loaded|traced, key, file, bytes, t_load | t_trace t_lower t_compile t_store, census (traced) | census_replayed (loaded), event}."""
        import jax
        k, path, meta_path = self._paths(sig)
        rec = {"key": k[:16], "L_compiled": int(L)}
        if self.process_local:                                     # this process's programs are process-local (FFI launch specs): traced, never loaded / stored — named on the L13 line
            rec["event"] = f"process_local:{self.process_local}"
        if not self.process_local and os.path.isfile(path) and os.path.isfile(meta_path):
            t = time.time()
            try:
                from jax.experimental.serialize_executable import deserialize_and_load
                with open(meta_path, encoding="utf-8") as fh:
                    meta = json.load(fh)
                if meta.get("format") != FORMAT:
                    raise ValueError(f"format {meta.get('format')!r} is not {FORMAT}")
                with open(path, "rb") as fh:                       # nothing is deserialized before the owner / mode rule and the sha256 line both hold
                    why = refusal(path, os.fstat(fh.fileno()))
                    head, payload = (b"", b"") if why else (fh.readline(80), fh.read())
                if why is None and head != DIGEST_HEAD + _sha(payload).encode() + b"\n":
                    why = "its first line is not the sha256 of the rest (truncated, corrupted, or written without one); fix: none needed, it is rewritten now"
                if why is not None:
                    raise Refused(why)
                in_tree = jax.tree_util.tree_structure((tuple(args), {}))
                out_tree = jax.tree_util.tree_structure(_unskel(meta["out_skeleton"]))
                compiled = deserialize_and_load(payload, in_tree, out_tree)
                rec.update(source="loaded", t_load=round(time.time() - t, 3), bytes=len(payload), file=os.path.basename(path), traced_by=meta.get("traced_by"),
                           census_replayed=meta.get("census") or {})
                if self.replay is not None and meta.get("census"):
                    self.replay(meta["census"], int(L), k[:16])
                with self.lock:
                    self.counts["loaded"] += 1; self.bytes_loaded += len(payload)
                return compiled, rec
            except Refused as e:                                   # named once on stderr and on the L13 line; from here on exactly the absent file
                with self.lock:
                    self.events.append("load_refused")
                rec["event"] = f"load_refused: {e}"
                print(f"[af2ig-opt] program_cache: REFUSED cache file {path}: {e} — not deserialized, treated as absent (the program is traced and stored again)",
                      file=sys.stderr, flush=True)
            except Exception as e:  # noqa: BLE001 — a file that does not load is named and the program is traced instead
                word = load_failed_word(e)                                   # named; the program is traced instead — a load never ends the run
                with self.lock:
                    self.counts["load_failed"] += 1; self.events.append(word)
                rec["event"] = f"{word}: {str(e)[:160]}"
                print(f"program_cache: {rec['event']} ({os.path.basename(path)}) — tracing the program instead", flush=True)
        compiled, arec = aot_compile(apply_fn, args, self.snapshot)  # the census is read inside TRACE_LOCK around the trace: the delta is this program's own
        before, after = arec.pop("census_before", {}), arec.pop("census_after", {})
        rec.update(arec)
        rec["census"] = census_delta(before, after)
        with self.lock:
            self.counts["traced"] += 1
        t = time.time()
        if self.process_local:
            with self.lock:
                self.counts["store_skipped"] += 1
                if f"store_skipped:{self.process_local}" not in self.events: self.events.append(f"store_skipped:{self.process_local}")
            return compiled, rec
        try:
            from jax.experimental.serialize_executable import serialize
            payload, in_tree, out_tree = serialize(compiled)
            skel = _skel(jax.tree_util.tree_unflatten(out_tree, [0] * out_tree.num_leaves))
            meta = {"format": FORMAT, "L_compiled": int(L), "signature": sig, "static_key": self.key, "stack": self.facts["stack"], "device": self.facts.get("device"),
                    "args": self.facts["args"], "env": self.facts["env"], "bytes": len(payload), "out_skeleton": skel, "census": rec["census"],
                    "traced_by": {"pid": os.getpid(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "t_trace": arec.get("t_trace"), "t_lower": arec.get("t_lower"),
                                  "t_compile": arec.get("t_compile")}}
            tmp = path + f".tmp{os.getpid()}"
            with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644), "wb") as fh:
                fh.write(DIGEST_HEAD + _sha(payload).encode() + b"\n")
                fh.write(payload)
            os.replace(tmp, path)
            tmpm = meta_path + f".tmp{os.getpid()}"
            with open(tmpm, "w", encoding="utf-8") as fh:
                json.dump(meta, fh)
            os.replace(tmpm, meta_path)
            rec.update(t_store=round(time.time() - t, 3), bytes=len(payload), file=os.path.basename(path))
            with self.lock:
                self.counts["stored"] += 1; self.bytes_stored += len(payload)
        except Exception as e:  # noqa: BLE001 — a program that cannot be stored is named; the run goes on with the compiled program in memory
            word = f"store_failed:{type(e).__name__}"
            with self.lock:
                self.counts["store_failed"] += 1; self.events.append(word)
            rec["event"] = f"{word}: {str(e)[:160]}"
            print(f"program_cache: {rec['event']}", flush=True)
        return compiled, rec


def census_delta(before: dict, after: dict) -> dict:
    """Per census kind: the calls this trace added — served, fallback (by reason), errors."""
    out = {}
    for kind, a in (after or {}).items():
        b = (before or {}).get(kind) or {}
        d = {"served": int(a.get("served") or 0) - int(b.get("served") or 0), "fallback": int(a.get("fallback") or 0) - int(b.get("fallback") or 0),
             "errors": int(a.get("errors") or 0) - int(b.get("errors") or 0)}
        fa, fb = a.get("fallback_by") or {}, b.get("fallback_by") or {}
        d["fallback_by"] = {r: int(fa.get(r) or 0) - int(fb.get(r) or 0) for r in fa if int(fa.get(r) or 0) - int(fb.get(r) or 0)}
        ea, eb = a.get("errors_by") or {}, b.get("errors_by") or {}
        d["errors_by"] = {r: int(ea.get(r) or 0) - int(eb.get(r) or 0) for r in ea if int(ea.get(r) or 0) - int(eb.get(r) or 0)}
        out[kind] = d
    return out


def line(store: Optional["Store"], tag: str = "af2ig-opt") -> str:
    """The L13 LEVER line at exit (the driver prints it; the package's own line is built from the timer records)."""
    if store is None:
        return f"[{tag}] LEVER name=L13 state=off reason=not_requested flag=-program_cache"
    c = store.census()
    ev = ",".join(c["events"]) or "none"
    return (f"[{tag}] LEVER name=L13 state=on impl=serialize_executable origin=kit flag=-program_cache dir={c['dir']} key={c['static_key']} "
            f"loaded={c['loaded']} traced={c['traced']} stored={c['stored']} load_failed={c['load_failed']} store_failed={c['store_failed']} store_skipped={c.get('store_skipped', 0)} process_local={getattr(store, 'process_local', None) or 'none'} events={ev} "
            f"bytes_loaded={c['bytes_loaded']} bytes_stored={c['bytes_stored']} entries={c['entries']}")
