"""opt_core.mem.peak — the ALLOCATOR PEAK of a pass, recorded beside the sampled device high-water (never instead of it).

Two values per pass; each is what it says, neither stands in for the other:

    device high-water    max nvidia-smi memory.used over the pass window — the process footprint, pools included. The launcher's sampler
                         (smi100.py beside the pass) is the source; this module never samples nvidia-smi: it records the window
                         markers [t0, t1] and, when the sampler's rows are handed to it, reads the high-water over that window (`smi_high_water`).
    allocator peak       what the process's allocator handed out at its peak — torch: `torch.cuda.max_memory_allocated` (and
                         `max_memory_reserved`) per device, the caching allocator's own maxima (exact, not sampled); JAX:
                         `device.memory_stats()["peak_bytes_in_use"]`, the BFC allocator's high-water mark inside its pool (exact), with
                         `bytes_in_use` (instantaneous) and `bytes_limit` beside, read from a daemon thread at >= 2 Hz.

Why both. Under JAX's default client settings (XLA_PYTHON_CLIENT_PREALLOCATE=true, a fixed fraction of the card) nvidia-smi shows the
PREALLOCATED POOL — the same value for every input size — so the device high-water of a JAX pass is the pool, not a peak; the allocator
peak is the process's true peak under that env. Under torch the two differ by the caching allocator's reserve and fragmentation. A table
that renders one column must say which it is; this module writes both into one record so the reader can.

Rules (each one a test): READ-ONLY — the module imports neither torch nor jax (it reads them from `sys.modules` once the process has
imported them itself), initialises no backend (JAX: waits for `jax._src.xla_bridge.backends_are_initialized()`; torch: waits for
`torch.cuda.is_initialized()`), resets no counter, and sets no environment variable: the XLA / allocator env is recorded VERBATIM; a
caller's expectation that disagrees with the process env is a named refusal (`PeakRefusal`), never an override of a kit's pins. Sampling
never raises into the process: an exception is recorded (`error`, `n_errors`) and the thread carries on. An `atexit` reader alone is too
late for JAX (the backend is torn down first and reads zero) — the maxima live in memory and the final write at exit carries them.

The mechanism — the backend-ready gate, the read-only daemon thread, the periodic atomic rewrite — has this file as its one source.

Use:
    in-process     probe = peak.start("<out>/peak_mem.json"); ...the pass...; peak.stop()          (the exit hook writes too)
    as a hook      python -m opt_core.mem.peak hook <dir>   → <dir>/peak.py (this file, verbatim), <dir>/peak.sha256 (checked at boot: a
                   copy that differs from its sha is refused, one stderr line, no record) and the two LOADERS, each the same two statements:
                     <dir>/peakhook.pth       THE LOADER: a .pth line for the model interpreter's site-packages (site
                                              initialisation, before any sitecustomize); the launcher copies it there (a box-local write)
                                              and puts <dir> LAST on PYTHONPATH — PYTHONPATH=$PYTHONPATH:<dir> — so the .pth imports `peak`
                                              from there and nothing of the kit's or a det recipe's own path order (its det_site/
                                              sitecustomize) is shadowed: the interpreter imports ONE sitecustomize, and a hook dir put
                                              FIRST would silently replace a kit's own and unarm the lever under the kit's label.
                     <dir>/sitecustomize.py   the same two statements for an interpreter whose sitecustomize slot NO kit or recipe owns
                                              (a stock arm; <dir> still LAST on PYTHONPATH) — never beside a kit's own sitecustomize.
                                              The record says which loader ran (`loader`: sitecustomize | pth | api).
                   A .pth fires ONLY in the interpreter whose site-packages holds it: `python -m opt_core.mem.peak install <dir> <python>...`
                   copies it into the purelib of EVERY interpreter that runs a model process (a kit with two venvs = two copies; a driver
                   under the launcher venv and a model process under the kit's venv are two interpreters) and proves each one by a run that
                   must leave a record. The record names its interpreter (`executable`, `prefix`, `argv`): a pass whose records all come
                   from a driver (no backend ever ready) is refused by the reduce by name — `error` says which processes the probe rode.
                   A forked child (multiprocessing 'fork', os.fork) restarts the sampling with its own record (`forked_from` = the parent).
                   then  OPT_PEAK_MEM_PATH=<local pass dir>/peak_mem.json <the model process>  — the launcher's route for a STOCK arm: the
                   instrument is carried verbatim like smi100.py, imported under its own name (`peak`, never `opt_core.*`) from a directory
                   outside the kit tree, so the stock proof holds unchanged. Every process of the pass writes its own
                   <out>/peak_mem.<pid>-<start ms>.json (OPT_PEAK_MEM_PER_PROCESS=1, the hook's default; pid + start time: a reused pid never
                   overwrites an earlier process's record) — to a box-LOCAL directory, never loose onto shared storage; the pass record is
    the reduce     python -m opt_core.mem.peak reduce <out> [--smi <smi100 csv>] [--pack]  → <out>/peak_mem.json (exit 3 + one stderr line
                   when the pass is refused by name — no recorded process saw a backend — the document still written; a sampler csv that is
                   not there, an unreadable or a malformed record are NAMED on the document — `smi_error`, `unreadable`, `malformed` —
                   never a traceback; exit 2 only when no record is readable at all): the max per field over
                   the processes and devices, the window = the union, smi_high_water_gb over that window from the sampler's rows; --pack
                   folds the per-process records (and their sample series) into <out>/peak_mem.records.tar.gz and removes the loose files
                   (born-packed: one record + one archive per pass).
    the env        the XLA / allocator env is snapshotted twice — at start (interpreter start under a loader) and again at the first tick a
                   backend is ready; the record's `xla_env` / `torch_env` / `xla_effective` are the backend-ready snapshot (the env the
                   backend was created under: a lever that exports XLA_PYTHON_CLIENT_PREALLOCATE=false or PYTORCH_CUDA_ALLOC_CONF in-process
                   before its first device call is seen), `xla_env_at_start` beside; `env_source` names which (`backend_ready`, else
                   `start` when no backend became ready) and `env_changed_before_backend` says whether they differ. `xla_effective` (the
                   pool verdict) is derived ONLY in a process that created a JAX backend (`backend.jax_backend_ready_at` set): a torch-only
                   process records {applies: false, pool: false} — the JAX client defaults describe no pool there.
    the loaders    a `python -I` route ignores PYTHONPATH: the .pth's `import peak` then fails (one stderr line, no record) — the launcher
                   chooses the loader for its engine; the record's `loader` field is the evidence of which one ran.
    the cadence    `hz` is the nominal cadence, `hz_effective` the one observed (samples − 1 over the sampled span): a starved thread is visible.

Environment (read, never written): OPT_PEAK_MEM_PATH (the hook's output path), OPT_PEAK_MEM_HZ (sampling cadence, default 2, floor 2),
OPT_PEAK_MEM_FLUSH_S (rewrite cadence, default 5), OPT_PEAK_MEM_PER_PROCESS (1 = <stem>.<pid>.json), OPT_PEAK_MEM_SERIES (N > 0 = keep
the last N samples and write them beside the record as <stem>.samples.csv).

The record (peak_mem.json; every *_gb is GB = 1e9 bytes, the bytes beside it; None = not observed in this process):
    {schema, device, device_name, peak_alloc_gb, peak_alloc_kind, torch_max_allocated_gb, torch_max_reserved_gb, jax_peak_bytes_in_use_gb,
     jax_bytes_in_use_max_gb, jax_bytes_limit_gb, smi_high_water_gb, xla_env, xla_effective, torch_env, window: [t0, t1], devices: [...],
     samples, hz, backend: {torch_loaded, torch_cuda_ready_at, jax_loaded, jax_backend_ready_at}, pid, argv, instrument: {file, sha256},
     error, n_errors, stopped}
Standard library only; no intra-package import (the carried copy runs alone, like smi100.py).
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "opt_core.mem.peak/1"
GB = 1e9
MIB = 2 ** 20
HZ_FLOOR = 2.0                                    # the cadence: never slower than 2 samples per second
ENV_PATH, ENV_HZ, ENV_FLUSH, ENV_PER_PROCESS, ENV_SERIES = ("OPT_PEAK_MEM_PATH", "OPT_PEAK_MEM_HZ", "OPT_PEAK_MEM_FLUSH_S",
                                                            "OPT_PEAK_MEM_PER_PROCESS", "OPT_PEAK_MEM_SERIES")
XLA_NAMES = ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_ALLOCATOR",
             "XLA_FLAGS", "TF_FORCE_UNIFIED_MEMORY", "JAX_PLATFORMS")   # the named client settings; every other XLA_* / JAX_* name rides along verbatim
XLA_PREFIXES = ("XLA_", "JAX_")
TORCH_NAMES = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_NO_CUDA_MEMORY_CACHING")
JAX_DEFAULT_PREALLOCATE, JAX_DEFAULT_MEM_FRACTION, JAX_DEFAULT_ALLOCATOR = True, 0.75, "default"   # the client library's documented defaults
SITECUSTOMIZE = ("# the peak instrument's loader, sitecustomize form (written by `python -m opt_core.mem.peak hook`): the process imports the\n"
                 "# carried copy beside this file under its own name and boots the probe when the launcher names an output path; nothing else.\n"
                 "import peak\n"
                 "peak.boot('sitecustomize')\n")
_PTH_BODY = "try:\n import peak; peak.boot('pth')\nexcept Exception as e: sys.stderr.write('[peak] REFUSED: pth loader: %r; no record\\n' % (e,))"
PTH = "import sys; exec(%r)\n" % _PTH_BODY   # the same two statements as ONE .pth line (site initialisation; `peak` from the hook dir on PYTHONPATH); guarded: a `python -I` route (PYTHONPATH ignored) costs one stderr line, never a traceback
PTH_NAME = "peakhook.pth"


class PeakRefusal(RuntimeError):
    """A named refusal: the caller's expectation of the allocator env disagrees with the process env (never overridden)."""


# ------------------------------------------------------------------------------------------------------------------ the env, verbatim
def xla_env(environ: Optional[Mapping[str, str]] = None) -> dict:
    """The XLA / JAX client environment of this process, verbatim: the named settings plus every XLA_* / JAX_* name, present names only."""
    e = os.environ if environ is None else environ
    return {k: e[k] for k in sorted(e) if k in XLA_NAMES or k.startswith(XLA_PREFIXES)}


def torch_env(environ: Optional[Mapping[str, str]] = None) -> dict:
    e = os.environ if environ is None else environ
    return {k: e[k] for k in TORCH_NAMES if k in e}


MEM_FRACTION_PRECEDENCE = ("XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_MEM_FRACTION")   # jaxlib/xla_client.py (jaxlib 0.10.2, the JAX pinned stack): XLA_CLIENT_MEM_FRACTION is read first; XLA_PYTHON_CLIENT_MEM_FRACTION is the deprecated name, read when the former is unset. BOTH SET is not a warning: jaxlib raises ValueError at CUDA plugin initialisation (xla_client.py:182-188 @0.10.2, :232-238 @0.5.3) and jax falls back to the CPU backend silently (measured on jaxlib 0.10.2) — xla_effective() reports it as conflict=True and a lever exporting either name refuses when the other is present; both recorded verbatim
POOL_ALLOCATORS = ("default", "bfc")               # jaxlib/xla_client.py:126-142: XLA_PYTHON_CLIENT_ALLOCATOR ∈ default | platform | bfc | cuda_async | vmm — the BFC kinds (default, bfc) preallocate the pool; platform / cuda_async / vmm allocate on demand


NO_JAX_EFFECTIVE = {"applies": False, "pool": False, "preallocate": None, "mem_fraction": None, "allocator": None, "conflict": False, "source": {},
                    "note": "no JAX backend in this process: the device high-water tracks the peak"}


def xla_effective(env: Mapping[str, str], jax_backend: bool = True) -> dict:
    """What the JAX client library resolves from the verbatim env: preallocate (default true), the pool fraction (MEM_FRACTION_PRECEDENCE: the
    first name present, else 0.75) and the allocator (default 'default' = the pooled BFC allocator); `source` names env or default per key so
    a reader can say 'the pool' with its provenance. APPLIES ONLY to a process that created a JAX backend (`jax_backend`): without one the
    client defaults describe nothing — {applies: false, pool: false}, never a pool verdict on a torch-only process. Both MEM_FRACTION names set
    is `conflict: true` (the CUDA plugin does not initialise; jax falls back to CPU): pool false, the note says so."""
    if not jax_backend:
        return dict(NO_JAX_EFFECTIVE)
    src = {}
    pre = env.get("XLA_PYTHON_CLIENT_PREALLOCATE")
    if pre is None:
        preallocate, src["preallocate"] = JAX_DEFAULT_PREALLOCATE, "default"
    else:
        preallocate, src["preallocate"] = pre.strip().lower() not in ("false", "0", "no", "off"), "env"
    frac_name = next((n for n in MEM_FRACTION_PRECEDENCE if n in env), None)
    conflict = all(n in env for n in MEM_FRACTION_PRECEDENCE)          # both names set: the CUDA plugin does not initialise (CPU fallback) — named, never a pool verdict
    if frac_name is None:
        fraction, src["mem_fraction"] = JAX_DEFAULT_MEM_FRACTION, "default"
    else:
        try:
            fraction = float(env[frac_name])
        except ValueError:
            fraction = None
        src["mem_fraction"] = frac_name
    alloc = env.get("XLA_PYTHON_CLIENT_ALLOCATOR")
    allocator, src["allocator"] = (JAX_DEFAULT_ALLOCATOR, "default") if alloc is None else (alloc.strip().lower(), "env")
    pool = bool(preallocate) and allocator in POOL_ALLOCATORS and not conflict
    if conflict:
        note = ("XLA_CLIENT_MEM_FRACTION and XLA_PYTHON_CLIENT_MEM_FRACTION are both set: jaxlib raises ValueError at CUDA plugin initialisation "
                "and jax runs on the CPU backend — no device pool, no device peak; unset one name")
    elif pool:
        note = ("the device high-water of this process is the preallocated pool (mem_fraction of the card), not a peak; the allocator "
                "peak (jax_peak_bytes_in_use_gb) is the peak")
    else:
        note = "no preallocated pool: the device high-water tracks the peak"
    return {"applies": True, "preallocate": preallocate, "mem_fraction": fraction, "allocator": allocator, "pool": pool, "conflict": conflict,
            "source": src, "note": note}


def check_xla_env(expected: Mapping[str, str], environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """The names whose process value differs from `expected` (an absent name differs from any expected value) — the caller's expectation
    is checked, never applied."""
    e = os.environ if environ is None else environ
    return [f"{k}: process={e.get(k)!r} expected={v!r}" for k, v in sorted(expected.items()) if e.get(k) != v]


# ------------------------------------------------------------------------------------------------------------------ small primitives
def _write_atomic(path: str, doc) -> str:
    d = os.path.dirname(os.path.abspath(path)); os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True); fh.write("\n")
    os.replace(tmp, path)
    return path


def _gb(b) -> Optional[float]:
    return None if b is None else round(b / GB, 3)


def instrument() -> dict:
    """This file, named and hashed — the instrument every record and every launch record of a row it rides names."""
    p = os.path.abspath(__file__)
    try:
        sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
    except OSError:
        sha = None
    return {"file": p, "sha256": sha}


def per_process_path(path: str, pid: Optional[int] = None, t_start: Optional[float] = None) -> str:
    """<stem>.<pid>-<start ms>.json beside `path` — one record per process of a pass, keyed by pid AND start time (a reused pid never
    overwrites an earlier process's record)."""
    stem, ext = os.path.splitext(path)
    return f"{stem}.{os.getpid() if pid is None else pid}-{int((time.time() if t_start is None else t_start) * 1000)}{ext or '.json'}"


def _jax_backend_ready() -> bool:
    """True once the process has initialised its own JAX backend (jax._src.xla_bridge.backends_are_initialized()); never initialises one."""
    xb = sys.modules.get("jax._src.xla_bridge")
    if xb is None:
        return False
    f = getattr(xb, "backends_are_initialized", None)
    if callable(f):
        try:
            return bool(f())
        except Exception:
            return False
    return any(getattr(xb, name, None) for name in ("_backends", "_backend_lock_initialized"))


def _torch_cuda_ready():
    """The imported torch module once its CUDA context exists (torch.cuda.is_initialized()), else None; never initialises it."""
    t = sys.modules.get("torch")
    cu = getattr(t, "cuda", None) if t is not None else None
    try:
        return t if cu is not None and cu.is_initialized() else None
    except Exception:
        return None


# ------------------------------------------------------------------------------------------------------------------ the probe
class Probe:
    """One process's probe: a daemon thread sampling the counters at `hz`, the running maxima per device, the record rewritten every
    `flush_s` seconds and at stop / exit. `record()` is the document; `write()` puts it at `path`."""

    def __init__(self, path: str, hz: float = HZ_FLOOR, flush_s: float = 5.0, series: int = 0, loader: str = "api"):
        self.path = path; self.loader = loader; self.forked_from = None; self.stem_path = None   # stem_path: the path before per_process_path (a forked child derives its own)
        self.hz = max(float(hz), HZ_FLOOR)
        self.flush_s = float(flush_s)
        self.series_n = int(series)
        self.t0 = time.time(); self.t1 = self.t0
        self.samples = 0; self.n_errors = 0; self.error = None; self.stopped = False
        self.torch_dev = {}; self.jax_dev = {}                # "cuda:0" → maxima
        self.torch_cuda_ready_at = None; self.jax_backend_ready_at = None
        self.series: List[Tuple[float, Optional[int], Optional[int]]] = []
        self.xla_env_at_start = xla_env(); self.torch_env_at_start = torch_env(); self.instrument = instrument()
        self.xla_env_ready = None; self.torch_env_ready = None; self.env_ready_at = None   # re-snapshotted at the first tick a backend is ready (L1)
        self.t_first_sample = None
        self._lock = threading.Lock(); self._stop = threading.Event(); self._thread = None; self._last_write = 0.0

    # -- sampling (read-only; every exception recorded, none raised)
    def _snapshot_env_at_ready(self, now: float) -> None:
        """The env the backend was created under: taken once, at the first tick a backend is ready (a lever exporting an allocator variable
        in-process before its first device call is seen here, not at interpreter start)."""
        if self.env_ready_at is None:
            self.xla_env_ready = xla_env(); self.torch_env_ready = torch_env(); self.env_ready_at = now

    def tick(self) -> None:
        now = time.time(); t_alloc = j_in_use = None
        try:
            t = _torch_cuda_ready()
            if t is not None:
                if self.torch_cuda_ready_at is None: self.torch_cuda_ready_at = now; self._last_write = 0.0   # the backend is seen: the next loop iteration writes (a short-lived worker leaves its evidence)
                self._snapshot_env_at_ready(now)
                for i in range(t.cuda.device_count()):
                    k = f"cuda:{i}"; d = self.torch_dev.setdefault(k, {"device": k, "kind": "torch", "name": None, "max_allocated_bytes": 0, "max_reserved_bytes": 0, "samples": 0})
                    if d["name"] is None:
                        try: d["name"] = t.cuda.get_device_name(i)
                        except Exception: d["name"] = "?"
                    a = int(t.cuda.max_memory_allocated(i)); r = int(t.cuda.max_memory_reserved(i))
                    d["max_allocated_bytes"] = max(d["max_allocated_bytes"], a); d["max_reserved_bytes"] = max(d["max_reserved_bytes"], r); d["samples"] += 1
                    t_alloc = a if t_alloc is None else max(t_alloc, a)
            if "jax" in sys.modules and _jax_backend_ready():
                if self.jax_backend_ready_at is None: self.jax_backend_ready_at = now; self._last_write = 0.0
                self._snapshot_env_at_ready(now)
                jax = sys.modules["jax"]
                for dev in jax.local_devices():
                    st = dev.memory_stats() or {}
                    k = str(dev); d = self.jax_dev.setdefault(k, {"device": k, "kind": "jax", "name": getattr(dev, "device_kind", None), "peak_bytes_in_use": 0, "bytes_in_use_max": 0, "bytes_limit": None, "samples": 0})
                    d["peak_bytes_in_use"] = max(d["peak_bytes_in_use"], int(st.get("peak_bytes_in_use") or 0))
                    b = int(st.get("bytes_in_use") or 0); d["bytes_in_use_max"] = max(d["bytes_in_use_max"], b)
                    if st.get("bytes_limit") is not None: d["bytes_limit"] = int(st["bytes_limit"])
                    d["samples"] += 1
                    j_in_use = b if j_in_use is None else max(j_in_use, b)
        except Exception as e:                                  # never disturb the process
            self.n_errors += 1; self.error = repr(e)[:400]
        with self._lock:
            self.samples += 1; self.t1 = now
            if self.t_first_sample is None: self.t_first_sample = now
            if self.series_n > 0:
                self.series.append((round(now, 3), j_in_use, t_alloc))
                if len(self.series) > self.series_n: del self.series[: len(self.series) - self.series_n]

    def _loop(self) -> None:
        period = 1.0 / self.hz
        while not self._stop.wait(period):
            self.tick()
            if time.time() - self._last_write >= self.flush_s:
                try:
                    self.write()
                except Exception as e:                          # a failed periodic write (a full disk, a vanished dir) never ends the sampling; the last write at stop names it
                    self.n_errors += 1; self.error = f"write: {e!r}"[:400]; self._last_write = time.time()

    # -- the document
    def record(self) -> dict:
        with self._lock:
            tdev = [dict(d) for d in self.torch_dev.values()]; jdev = [dict(d) for d in self.jax_dev.values()]
            t1, samples, t_first = self.t1, self.samples, self.t_first_sample
        t_alloc = max((d["max_allocated_bytes"] for d in tdev), default=None); t_res = max((d["max_reserved_bytes"] for d in tdev), default=None)
        j_peak = max((d["peak_bytes_in_use"] for d in jdev), default=None); j_use = max((d["bytes_in_use_max"] for d in jdev), default=None)
        j_lim = max((d["bytes_limit"] for d in jdev if d["bytes_limit"] is not None), default=None)
        cands = [(t_alloc, "torch_max_allocated", tdev), (j_peak, "jax_peak_bytes_in_use", jdev)]
        best = max((c for c in cands if c[0] is not None), key=lambda c: c[0], default=None)
        if best is None: device = device_name = kind = None; peak = None
        else:
            peak, kind, devs = best
            key = "max_allocated_bytes" if kind == "torch_max_allocated" else "peak_bytes_in_use"
            top = max(devs, key=lambda d: d[key]); device, device_name = top["device"], top["name"]
        env_ready = self.xla_env_ready is not None
        xenv = self.xla_env_ready if env_ready else self.xla_env_at_start; tenv = self.torch_env_ready if env_ready else self.torch_env_at_start
        span = (t1 - t_first) if (t_first is not None and samples > 1) else None
        return {"schema": SCHEMA, "device": device, "device_name": device_name, "peak_alloc_gb": _gb(peak), "peak_alloc_kind": kind,
                "torch_max_allocated_gb": _gb(t_alloc), "torch_max_allocated_bytes": t_alloc, "torch_max_reserved_gb": _gb(t_res), "torch_max_reserved_bytes": t_res,
                "jax_peak_bytes_in_use_gb": _gb(j_peak), "jax_peak_bytes_in_use": j_peak, "jax_bytes_in_use_max_gb": _gb(j_use), "jax_bytes_in_use_max": j_use,
                "jax_bytes_limit_gb": _gb(j_lim), "jax_bytes_limit": j_lim, "smi_high_water_gb": None, "smi_samples_in_window": None,
                "xla_env": dict(xenv), "xla_effective": xla_effective(xenv, jax_backend=self.jax_backend_ready_at is not None), "torch_env": dict(tenv), "xla_env_at_start": dict(self.xla_env_at_start),
                "torch_env_at_start": dict(self.torch_env_at_start), "env_source": "backend_ready" if env_ready else "start", "env_ready_at": self.env_ready_at,
                "env_changed_before_backend": (env_ready and (xenv != self.xla_env_at_start or tenv != self.torch_env_at_start)),
                "window": [round(self.t0, 3), round(t1, 3)], "devices": tdev + jdev, "samples": samples, "hz": self.hz,
                "hz_effective": (round((samples - 1) / span, 3) if span and span > 0 else None), "loader": self.loader,
                "backend": {"torch_loaded": "torch" in sys.modules, "torch_cuda_ready_at": self.torch_cuda_ready_at, "jax_loaded": "jax" in sys.modules,
                            "jax_backend_ready_at": self.jax_backend_ready_at},
                "pid": os.getpid(), "argv": [a[:160] for a in sys.argv[:12]], "executable": sys.executable, "prefix": sys.prefix, "forked_from": self.forked_from,
                "instrument": dict(self.instrument), "error": self.error, "n_errors": self.n_errors, "stopped": self.stopped}

    def write(self) -> str:
        self._last_write = time.time()
        _write_atomic(self.path, self.record())
        if self.series_n > 0:
            with self._lock: rows = list(self.series)
            stem, _ = os.path.splitext(self.path); tmp = f"{stem}.samples.csv.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write("epoch,jax_bytes_in_use,torch_allocated_bytes\n")
                for t, j, a in rows: fh.write(f"{t:.3f},{'' if j is None else j},{'' if a is None else a}\n")
            os.replace(tmp, f"{stem}.samples.csv")
        return self.path

    def start(self) -> "Probe":
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="opt_core.mem.peak", daemon=True); self._thread.start()
        return self

    def stop(self, final_tick: bool = True) -> dict:
        """End the window: one last sample (guarded — after a backend teardown it reads nothing new), the final write, the record."""
        self._stop.set()
        if self._thread is not None: self._thread.join(timeout=max(2.0, 3.0 / self.hz))
        if final_tick: self.tick()
        self.stopped = True; self.write()
        return self.record()


_PROBE: Optional[Probe] = None


def start(path: Optional[str] = None, *, hz: Optional[float] = None, flush_s: Optional[float] = None, per_process: Optional[bool] = None,
          series: Optional[int] = None, xla_env_expected: Optional[Mapping[str, str]] = None, loader: str = "api") -> Probe:
    """Start this process's probe (one per process — a second call returns the running one). `path` defaults to OPT_PEAK_MEM_PATH (refused by
    name when neither is given); `per_process` (default: the env's OPT_PEAK_MEM_PER_PROCESS, else False for an explicit path, True for the env
    path) writes <stem>.<pid>.json. `xla_env_expected` is CHECKED against the process env — a difference is a PeakRefusal, never an override."""
    global _PROBE
    if _PROBE is not None:
        return _PROBE
    if xla_env_expected:
        conflicts = check_xla_env(xla_env_expected)
        if conflicts:
            raise PeakRefusal("allocator env conflict (the process env is never overridden): " + "; ".join(conflicts))
    env_path = os.environ.get(ENV_PATH)
    if path is None and not env_path:
        raise PeakRefusal(f"no output path: pass `path` or set {ENV_PATH}")
    out = path or env_path
    if per_process is None:
        pp = os.environ.get(ENV_PER_PROCESS)
        per_process = (pp not in ("0", "false", "no", "")) if pp is not None else (path is None)
    stem = out
    if per_process: out = per_process_path(out)
    hz = float(os.environ.get(ENV_HZ, HZ_FLOOR)) if hz is None else hz
    flush_s = float(os.environ.get(ENV_FLUSH, 5.0)) if flush_s is None else flush_s
    series = int(os.environ.get(ENV_SERIES, 0) or 0) if series is None else series
    _PROBE = Probe(out, hz=hz, flush_s=flush_s, series=series, loader=loader); _PROBE.stem_path = stem if per_process else None
    _PROBE.write()                                            # the window opens on disk at once: an aborted process still leaves its markers
    atexit.register(_at_exit)
    _register_at_fork()
    return _PROBE.start()


def boot(loader: str) -> Optional[Probe]:
    """The loaders' entry (sitecustomize | pth): start the probe when the launcher names an output path (OPT_PEAK_MEM_PATH), after checking
    this copy against the `peak.sha256` beside it when one exists — a copy that differs from its sha is REFUSED (one stderr line, no record:
    the launcher's 'no probe record' bucket names the pass); a refusal or any failure here never reaches the process (nothing raised)."""
    try:
        if not os.environ.get(ENV_PATH):
            return None
        sha_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "peak.sha256")
        if os.path.exists(sha_file):
            want = open(sha_file, encoding="utf-8").read().split()[0]
            got = instrument()["sha256"]
            if want != got:
                sys.stderr.write(f"[peak] REFUSED: {__file__} sha256 {got} != peak.sha256 {want} — the carried instrument is not the one named; no record\n"); return None
        return start(loader=loader)
    except Exception as e:
        try: sys.stderr.write(f"[peak] REFUSED: {e!r}; no record\n")
        except Exception: pass
        return None


def probe() -> Optional[Probe]:
    return _PROBE


def stop() -> Optional[dict]:
    """Stop this process's probe (the final write); None when none runs."""
    return None if _PROBE is None else (_PROBE.record() if _PROBE.stopped else _PROBE.stop())


def _at_exit() -> None:
    try:
        if _PROBE is not None and not _PROBE.stopped: _PROBE.stop(final_tick=True)
    except Exception as e:                                     # fail loud: a final write that cannot land (the record dir gone) says so — the last periodic write is what stands
        try: sys.stderr.write(f"[peak] REFUSED: at-exit write of {getattr(_PROBE, 'path', '?')} in pid {os.getpid()}: {e!r}; the last periodic write stands\n")
        except Exception: pass


_FORK_HOOKED = False


def _after_fork_in_child() -> None:
    """A forked child (multiprocessing 'fork', os.fork) inherits the probe object but not its daemon thread: restart the sampling here with
    the child's own record (`<stem>.<child pid>-<start>.json` when per-process, else the same path — a forked child of an explicit-path probe
    would overwrite its parent's record, so its record goes beside under the child's pid), its own window and maxima, `forked_from` = the
    parent's pid. A child that leaves through os._exit() (multiprocessing workers) skips atexit: its periodic writes (every flush_s) are
    its record. Never raises into the child."""
    global _PROBE
    try:
        p = _PROBE
        if p is None or p.stopped:
            return
        base = p.stem_path if p.stem_path else os.path.splitext(p.path)[0] + ".fork.json"
        q = Probe(per_process_path(base), hz=p.hz, flush_s=p.flush_s, series=p.series_n, loader=p.loader); q.stem_path = base
        q.forked_from = os.getppid(); q.xla_env_at_start = dict(p.xla_env_at_start); q.torch_env_at_start = dict(p.torch_env_at_start)
        _PROBE = q; q.write(); q.start()
    except Exception as e:                                     # fail loud: a child whose restart fails says so (one line), never silently records nothing
        try: sys.stderr.write(f"[peak] REFUSED: fork restart in pid {os.getpid()} (parent {os.getppid()}): {e!r}; no record for this child\n")
        except Exception: pass


def _register_at_fork() -> None:
    global _FORK_HOOKED
    if not _FORK_HOOKED and hasattr(os, "register_at_fork"):
        try:
            os.register_at_fork(after_in_child=_after_fork_in_child); _FORK_HOOKED = True
        except Exception as e:                                 # fail loud: forked children of this process will carry no record
            try: sys.stderr.write(f"[peak] REFUSED: os.register_at_fork: {e!r}; forked children of pid {os.getpid()} leave no record\n")
            except Exception: pass


# ------------------------------------------------------------------------------------------------------------------ the sampler's rows + the reduce
def smi_rows(path: str) -> List[Tuple[float, float]]:
    """The launcher sampler's csv (smi100.py: `epoch,util_gpu_pct,util_mem_pct,mem_used_mib,…`, one row per sample) as [(epoch, mem_used_mib)];
    rows whose stamp or memory field is not numeric are skipped — the same rows the pass reduce reads for peak_mem_gb."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            f = line.strip().split(",")
            if len(f) < 4: continue
            try: out.append((float(f[0]), float(f[3])))
            except ValueError: continue
    return out


def smi_high_water(rows: Iterable[Tuple[float, float]], t0: float, t1: float) -> Tuple[Optional[float], int]:
    """(max mem_used_mib over the samples stamped inside [t0, t1], the count of those samples) — in MiB; the caller converts."""
    inside = [m for t, m in rows if t0 <= t <= t1]
    return (max(inside) if inside else None), len(inside)


def read(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def reduce(records: Sequence[Mapping], smi: Optional[Iterable[Tuple[float, float]]] = None) -> dict:
    """N per-process records → the pass record: every maximum is the max over the records (and their devices), the window the union, the
    envs carried when the records agree (a disagreement is named in `xla_env_conflict`, never averaged away), `smi_high_water_gb` the
    sampler's high-water over the union window when the rows are given. Refused by name on an empty list."""
    recs = [dict(r) for r in records]
    if not recs:
        raise PeakRefusal("reduce over zero records — a pass without a probe record has no allocator peak (name it, never zero)")
    def mx(key):
        v = [r.get(key) for r in recs if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
        return max(v) if v else None
    t0 = min(r["window"][0] for r in recs); t1 = max(r["window"][1] for r in recs)
    brecs = [r for r in recs if (r.get("backend") or {}).get("torch_cuda_ready_at") is not None or (r.get("backend") or {}).get("jax_backend_ready_at") is not None]   # the records that saw a backend
    t_alloc, t_res, j_peak, j_use, j_lim = (mx(k) for k in ("torch_max_allocated_bytes", "torch_max_reserved_bytes", "jax_peak_bytes_in_use", "jax_bytes_in_use_max", "jax_bytes_limit"))
    cands = [(t_alloc, "torch_max_allocated"), (j_peak, "jax_peak_bytes_in_use")]
    best = max((c for c in cands if c[0] is not None), key=lambda c: c[0], default=(None, None))
    key = {"torch_max_allocated": "torch_max_allocated_bytes", "jax_peak_bytes_in_use": "jax_peak_bytes_in_use"}.get(best[1])
    top = max(recs, key=lambda r: r.get(key) or 0) if key else recs[0]
    jrecs = [r for r in recs if isinstance(r.get("jax_peak_bytes_in_use"), int) or (r.get("backend") or {}).get("jax_backend_ready_at")]   # the processes that created a JAX backend
    jtop = max(jrecs, key=lambda r: r.get("jax_peak_bytes_in_use") or 0) if jrecs else None                                                # the XLA fields come from the record holding the JAX peak
    envs = [json.dumps(r.get("xla_env") or {}, sort_keys=True) for r in recs]
    smi_mib, n_in = (smi_high_water(list(smi), t0, t1) if smi is not None else (None, None))
    devices = [dict(d, pid=r.get("pid")) for r in recs for d in (r.get("devices") or [])]
    return {"schema": SCHEMA, "device": top.get("device"), "device_name": top.get("device_name"), "peak_alloc_gb": _gb(best[0]), "peak_alloc_kind": best[1],
            "torch_max_allocated_gb": _gb(t_alloc), "torch_max_allocated_bytes": t_alloc, "torch_max_reserved_gb": _gb(t_res), "torch_max_reserved_bytes": t_res,
            "jax_peak_bytes_in_use_gb": _gb(j_peak), "jax_peak_bytes_in_use": j_peak, "jax_bytes_in_use_max_gb": _gb(j_use), "jax_bytes_in_use_max": j_use,
            "jax_bytes_limit_gb": _gb(j_lim), "jax_bytes_limit": j_lim,
            "smi_high_water_gb": None if smi_mib is None else round(smi_mib * MIB / GB, 3), "smi_high_water_mib": smi_mib, "smi_samples_in_window": n_in,
            "xla_env": dict((jtop or top).get("xla_env") or {}), "xla_effective": xla_effective((jtop or {}).get("xla_env") or {}, jax_backend=jtop is not None),
            "xla_env_conflict": sorted(set(envs)) if len(set(envs)) > 1 else None, "jax_records": len(jrecs),
            "torch_env": dict(top.get("torch_env") or {}), "window": [t0, t1], "devices": devices, "samples": sum(int(r.get("samples") or 0) for r in recs),
            "hz": min((r.get("hz") for r in recs if r.get("hz")), default=None), "records": len(recs), "pids": [r.get("pid") for r in recs],
            "instrument": top.get("instrument"), "instruments": sorted({(r.get("instrument") or {}).get("sha256") or "?" for r in recs}),
            "loaders": sorted({r.get("loader") or "?" for r in recs}), "env_source": top.get("env_source"), "env_changed_before_backend": top.get("env_changed_before_backend"),
            "hz_effective": min((r.get("hz_effective") for r in recs if r.get("hz_effective")), default=None),
            "errors": [r.get("error") for r in recs if r.get("error")], "n_errors": sum(int(r.get("n_errors") or 0) for r in recs),
            "backend_records": len(brecs), "executables": sorted({r.get("executable") or "?" for r in recs}),
            "error": (None if brecs else
                      "no recorded process loaded torch or jax: the probe rode " + ", ".join(sorted({os.path.basename((r.get("argv") or ["?"])[0]) for r in recs}))[:200] +
                      " under " + ", ".join(sorted({r.get("executable") or "?" for r in recs}))[:200] + " — the model process ran without a loader (a .pth fires only in the"
                      " interpreter whose site-packages holds it: `peak.py install <hook> <model python>`; the sitecustomize form needs the hook dir on that process's PYTHONPATH)"),
            "unstopped": [r.get("pid") if r.get("pid") is not None else (r.get("file") or "?") for r in recs if not r.get("stopped")]}


def per_process_files(out_dir: str, stem: str = "peak_mem") -> List[str]:
    """The per-process records of a pass in `out_dir`: <stem>.<pid>-<start ms>.json (and the older <stem>.<pid>.json), never the pass record."""
    return sorted(f for f in os.listdir(out_dir) if f.startswith(stem + ".") and f.endswith(".json") and f != stem + ".json" and ".samples" not in f and ".records" not in f)


def reduce_dir(out_dir: str, smi_csv: Optional[str] = None, out: Optional[str] = None, stem: str = "peak_mem", pack: bool = False) -> Tuple[str, dict]:
    """<out_dir>/<stem>.<pid>-<start ms>.json (every process of the pass) → <out_dir>/<stem>.json (or `out`); refused by name without a
    record. `pack`: the per-process records and their sample series are folded into <out_dir>/<stem>.records.tar.gz and the loose files
    removed (the pass keeps one record + one archive; nothing loose lands on shared storage) — `packed` names the archive on the record."""
    files = per_process_files(out_dir, stem)
    if not files:
        raise PeakRefusal(f"{out_dir}: no {stem}.<pid>-<start>.json — the pass carries no probe record")
    recs, unreadable, malformed = [], [], []                   # every file lands in exactly one of the three; the reduce runs over the readable, well-formed ones
    for f in files:
        try:
            r = read(os.path.join(out_dir, f))
        except (OSError, ValueError) as e:
            unreadable.append({"file": f, "error": repr(e)[:200]}); continue
        w = r.get("window") if isinstance(r, dict) else None
        if not (isinstance(w, (list, tuple)) and len(w) == 2 and all(isinstance(x, (int, float)) for x in w)):
            malformed.append({"file": f, "error": "no numeric window [t0, t1]"}); continue
        recs.append(dict(r, file=f))
    if not recs:
        raise PeakRefusal(f"{out_dir}: {len(files)} {stem}.<pid>-<start>.json but none readable — " + "; ".join(f"{u['file']}: {u['error']}" for u in unreadable + malformed)[:600])
    smi, smi_error = None, None
    if smi_csv:
        try:
            smi = smi_rows(smi_csv)
        except OSError as e:                                   # the launcher named a sampler csv that is not there: the allocator side is still, the smi side is named absent
            smi_error = f"{smi_csv}: {e.strerror or repr(e)}"
    doc = reduce(recs, smi=smi)
    doc["files"] = files; doc["unreadable"] = unreadable; doc["malformed"] = malformed; doc["smi_csv"] = smi_csv; doc["smi_error"] = smi_error; doc["packed"] = None
    if pack:
        import tarfile
        series = sorted(f for f in os.listdir(out_dir) if f.startswith(stem + ".") and f.endswith(".samples.csv"))
        arc = os.path.join(out_dir, stem + ".records.tar.gz"); tmp = arc + f".{os.getpid()}.tmp"
        with tarfile.open(tmp, "w:gz") as tf:
            for f in files + series: tf.add(os.path.join(out_dir, f), arcname=f)
        os.replace(tmp, arc)
        for f in files + series: os.remove(os.path.join(out_dir, f))
        doc["packed"] = os.path.basename(arc); doc["packed_entries"] = files + series
    return _write_atomic(out or os.path.join(out_dir, stem + ".json"), doc), doc


def write_hook(hook_dir: str) -> dict:
    """<dir>/peak.py (this file, byte for byte) + <dir>/peak.sha256 + the two loaders <dir>/sitecustomize.py and <dir>/peakhook.pth — the
    carried instrument a launcher puts on a model process's PYTHONPATH (a directory outside the kit tree); the .pth goes into the model
    interpreter's site-packages when the sitecustomize slot is owned by a kit or recipe."""
    os.makedirs(hook_dir, exist_ok=True)
    src = os.path.abspath(__file__); dst = os.path.join(hook_dir, "peak.py")
    shutil.copyfile(src, dst)
    ins = instrument()
    with open(os.path.join(hook_dir, "peak.sha256"), "w", encoding="utf-8") as fh: fh.write(f"{ins['sha256']}  peak.py\n")
    with open(os.path.join(hook_dir, "sitecustomize.py"), "w", encoding="utf-8") as fh: fh.write(SITECUSTOMIZE)
    with open(os.path.join(hook_dir, PTH_NAME), "w", encoding="utf-8") as fh: fh.write(PTH)
    return {"dir": hook_dir, "peak_py": dst, "sha256": ins["sha256"], "sitecustomize": os.path.join(hook_dir, "sitecustomize.py"), "pth": os.path.join(hook_dir, PTH_NAME)}


def install_pth(hook_dir: str, pythons: Sequence[str], prove: bool = True) -> List[dict]:
    """Copy <hook_dir>/peakhook.pth into the site-packages (sysconfig purelib) of EACH interpreter that runs a model process — a .pth fires
    only in the interpreter whose site-packages holds it (a kit with two venvs needs two copies; a site-packages dir spliced onto sys.path by
    the kit at runtime is not a site dir and its .pth files are never processed). Box-local writes; the hook dir must be on that process's
    PYTHONPATH (last). `prove` runs each interpreter once with the loader armed and reads the record it leaves: [{python, purelib, pth, proved,
    executable, loader, error}] — an interpreter that leaves no record is named, never assumed."""
    out = []
    pth_src = os.path.join(hook_dir, PTH_NAME)
    for py in pythons:
        e = {"python": py, "purelib": None, "pth": None, "proved": None, "executable": None, "loader": None, "error": None}
        try:
            r = subprocess.run([py, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                e["error"] = f"purelib query failed rc={r.returncode}: {r.stderr.strip()[:200]}"; out.append(e); continue
            e["purelib"] = r.stdout.strip(); e["pth"] = os.path.join(e["purelib"], PTH_NAME)
            shutil.copyfile(pth_src, e["pth"])
            if prove:
                with tempfile.TemporaryDirectory() as td:
                    env = dict(os.environ); env["PYTHONPATH"] = (env["PYTHONPATH"] + os.pathsep if env.get("PYTHONPATH") else "") + os.path.abspath(hook_dir)
                    env[ENV_PATH] = os.path.join(td, "peak_mem.json"); env[ENV_PER_PROCESS] = "1"
                    r2 = subprocess.run([py, "-c", "import sys, time; time.sleep(0.1); print(sys.executable)"], capture_output=True, text=True, timeout=120, env=env)
                    recs = [read(os.path.join(td, f)) for f in per_process_files(td)]
                    e["proved"] = bool(recs) and r2.returncode == 0
                    if recs:
                        e["executable"] = recs[0].get("executable"); e["loader"] = recs[0].get("loader")
                    else:
                        e["error"] = f"the loader left no record under {py} (rc={r2.returncode}; stderr: {r2.stderr.strip()[-200:]})"
        except Exception as ex:                                # a read-only site-packages, a missing interpreter: named, the next one still tried
            e["error"] = repr(ex)[:300]
        out.append(e)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="opt_core.mem.peak", description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("reduce", help="the per-process records of a pass → its peak_mem.json"); r.add_argument("out_dir"); r.add_argument("--smi", default=None, help="the pass's smi100.py csv (the high-water over the window)"); r.add_argument("--out", default=None)
    r.add_argument("--pack", action="store_true", help="fold the per-process records + sample series into <stem>.records.tar.gz and remove the loose files")
    h = sub.add_parser("hook", help="write the carried instrument + the two loaders into a directory"); h.add_argument("hook_dir")
    i = sub.add_parser("install", help="copy <hook_dir>/peakhook.pth into each interpreter's site-packages and prove the loader there"); i.add_argument("hook_dir"); i.add_argument("pythons", nargs="+")
    i.add_argument("--no-prove", action="store_true", help="copy only")
    s = sub.add_parser("show", help="one line per record"); s.add_argument("paths", nargs="+")
    a = ap.parse_args(argv)
    if a.verb == "reduce":
        try:
            p, doc = reduce_dir(a.out_dir, smi_csv=a.smi, out=a.out, pack=a.pack)
        except PeakRefusal as e:
            sys.stderr.write(f"[peak] refused: {e}\n"); return 2
        sys.stdout.write(f"[peak] {p}: peak_alloc_gb={doc['peak_alloc_gb']} ({doc['peak_alloc_kind']}) smi_high_water_gb={doc['smi_high_water_gb']} records={doc['records']} window={doc['window']} packed={doc['packed']}"
                         + (f" smi_error={doc['smi_error']}" if doc.get("smi_error") else "") + (f" unreadable={[u['file'] for u in doc['unreadable']]}" if doc.get("unreadable") else "")
                         + (f" malformed={[m['file'] for m in doc['malformed']]}" if doc.get("malformed") else "") + "\n")
        if doc.get("error"):                                   # a refused pass (no recorded process saw a backend): the document is written, the reason is on stderr, the exit says so — like `install`
            sys.stderr.write(f"[peak] REFUSED {p}: {doc['error']}\n"); return 3
        return 0
    if a.verb == "hook":
        d = write_hook(a.hook_dir); sys.stdout.write(f"[peak] hook {d['dir']}: peak.py sha256 {d['sha256']}\n"); return 0
    if a.verb == "install":
        rows = install_pth(a.hook_dir, a.pythons, prove=not a.no_prove); bad = 0
        for e in rows:
            ok = e["error"] is None and (e["proved"] is not False)
            bad += 0 if ok else 1
            sys.stdout.write(f"[peak] install {e['python']}: pth={e['pth']} proved={e['proved']} loader={e['loader']} executable={e['executable']}" + (f" ERROR: {e['error']}" if e["error"] else "") + "\n")
        return 0 if not bad else 3
    for p in a.paths:
        d = read(p)
        sys.stdout.write(f"{p}: peak_alloc_gb={d.get('peak_alloc_gb')} ({d.get('peak_alloc_kind')}) torch_max_allocated_gb={d.get('torch_max_allocated_gb')} "
                         f"jax_peak_bytes_in_use_gb={d.get('jax_peak_bytes_in_use_gb')} jax_bytes_limit_gb={d.get('jax_bytes_limit_gb')} smi_high_water_gb={d.get('smi_high_water_gb')} "
                         f"pool={(d.get('xla_effective') or {}).get('pool')} window={d.get('window')} samples={d.get('samples')}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
