"""Lever `compilecache` — the design step's two XLA executables (ColabDesign's `fn` = jit(_model) and `grad_fn` = jit(value_and_grad(_model)),
colabdesign/af/model.py `_get_model`) persist across processes: JAX's persistent compilation cache, placed and keyed by the shared mechanism
`opt_core.jax_design.pcc` (origin=core), installed in the kit arm BEFORE the first compile of the process.

What it changes: nothing the step computes. A process whose (program, token count, flags, stack) was compiled before on this machine LOADS
the serialized executable instead of compiling it (what remains of the first step's cost is jax's own trace + lower + cache-key hash +
executable load, not XLA compilation). The first process of a new shape compiles as stock does
and writes the entry (cost: ~1 s of serialization). Numerics class: exact — a loaded executable is the compiled executable, byte for
byte (a warm process replays a cold one's design steps bit for bit, in both tiers).

The directory (never a flag of the kit):
  1. `JAX_COMPILATION_CACHE_DIR` when the environment sets it — upstream jax's own, documented variable: adopted as the cache directory as
     given (`dir_source=JAX_COMPILATION_CACHE_DIR`; jax keys every entry by program, flags, jaxlib and platform version itself);
  2. else `<XDG_CACHE_HOME | ~/.cache>/colabdesign_opt/pcc/<stack key>/xla` (`dir_source=default`), the stack key being pcc's
     `jax<v>-jaxlib<v>-cuda12plugin<v>-<gpu slug>` — one directory per (jax, jaxlib, PJRT plugin, GPU product), so a wheel or card
     change is a cold directory by name, never a stale load.
A directory that cannot be created / written steps the lever aside by name (`CompileCacheError`: state=skipped reason=cannot_run) — no silent
uncached run under the lever's name. Store-everything thresholds (pcc.STORE_ALL: every executable is written whatever its compile time or size).

XLA autotuning: this lever appends NOTHING to `XLA_FLAGS` (no `--xla_gpu_load/dump_autotune_results_*` pin). On this stack jax couples
XLA's per-fusion autotune results cache to the persistent cache directory by itself (`jax_persistent_cache_enable_xla_caches`, default
`xla_gpu_per_fusion_autotune_cache_dir` -> `<dir>/xla_gpu_per_fusion_autotune_cache_dir`), and the design program's cold compiles are
bitwise-reproducible process to process (upstream disables Triton GEMM; cuBLAS algorithm selection is not applied on sm_80+; two cold
processes compile executables that produce identical bytes) — the line names it: `autotune=jax_xla_cache`.

Evidence: ONE LEVER line at process exit (`opt_core.report.register_exit_tally`), counting jax's own cache events
(`jax.monitoring`: compile requests that consulted the cache, hits, misses) and the directory's entries at install and at exit:

    [colabdesign-opt] LEVER name=compilecache state=on impl=jax_design.pcc@<opt_core version> origin=core numerics=exact dir=<path> dir_source=<default|JAX_COMPILATION_CACHE_DIR>
        key=<stack key|adopted> autotune=jax_xla_cache requests=<n> hits=<n> misses=<n> compiled_s=<s> saved_s=<s> entries=<at install>-><at exit>
        reinit=<0|1> source=exit pid=<pid>

`requests>0` with the line present = applied (a design process always compiles: requests=0 classifies `missing`, fail-closed). `compiled_s` =
the seconds this process spent in backend compiles (jax.monitoring backend_compile_duration; a cache load counts its load time), `saved_s` = jax's
own estimate of compile seconds the hits saved (compile_time_saved_sec): a cold-populating design process prints misses>0 and saved_s~0, a warm
one hits=requests, a small compiled_s and a saved_s near the cold process's compiled_s.
`reinit=1` names the one repair this module makes: jax had already initialised its (empty) cache state before install ran — the state was
reset so the directory takes effect (install order puts this lever first; the word says when that did not hold).

Protocol (registry.py): NUMERICS, REFUSALS, install() / installed() / uninstall() / off_line(reason) / evidence(); the exit line goes through
the package's one exit printer (kernels.register_exit_line).

    from colabdesign_opt import compilecache_jax
    info = compilecache_jax.install()          # idempotent; before the first jit compile of the process
    compilecache_jax.evidence()                # {"installed", "dir", "dir_source", "key", "requests", "hits", "misses", "compiled_s", "saved_s", "entries_install", "entries_now", "reinit"}
"""
from __future__ import annotations

import os
import re
import threading
from typing import Mapping, MutableMapping, Optional

import opt_core
from opt_core import gates as _gates
from opt_core import report as _report
from opt_core.jax_design import pcc as _pcc

from .names import TAG

LEVER = "compilecache"
NUMERICS = "exact"                                            # registry.LEVERS[compilecache].numerics: a loaded executable IS the compiled executable (bitwise)
IMPL = f"jax_design.pcc@{opt_core.__version__}"
ENV_DIR = _pcc.CACHE_DIR_VAR                                  # JAX_COMPILATION_CACHE_DIR — upstream jax's variable, adopted when set
ENV_XDG = "XDG_CACHE_HOME"
DEFAULT_CACHE_HOME = os.path.join("~", ".cache")              # when XDG_CACHE_HOME is unset (the XDG base-directory default)
SUBDIR = os.path.join("colabdesign_opt", "pcc")               # <cache home>/colabdesign_opt/pcc/<stack key>/xla
AUTOTUNE_WORD = "jax_xla_cache"                               # what pins XLA's autotune results here: jax's coupling of the per-fusion cache to the directory
EVENT_REQUESTS = "/jax/compilation_cache/compile_requests_use_cache"
EVENT_HITS = "/jax/compilation_cache/cache_hits"
EVENT_MISSES = "/jax/compilation_cache/cache_misses"
DURATION_COMPILE = "/jax/core/compile/backend_compile_duration"      # seconds spent in backend compiles (a cache load counts its load time)
DURATION_SAVED = "/jax/compilation_cache/compile_time_saved_sec"      # jax's own estimate per hit: the entry's recorded compile time minus its retrieval time

_LOCK = threading.Lock()
_STATE: dict = {"installed": False, "info": None, "exports": None, "counts": {"requests": 0, "hits": 0, "misses": 0, "compiled_s": 0.0, "saved_s": 0.0}, "listener": False}


class CompileCacheError(RuntimeError):
    """The cache cannot be put in force in this process (directory not creatable / writable, stack key unresolvable, jax absent):
    the lever steps aside by name (``cannot_run``) and the mode runs the rest of its set."""
    cannot_run = True


REFUSALS = (CompileCacheError,)                               # what install() raises when the lever cannot run here (levers.install steps the lever aside by name)
_JAX_DEFAULTS = {"jax_compilation_cache_dir": None, "jax_persistent_cache_min_compile_time_secs": 1.0, "jax_persistent_cache_min_entry_size_bytes": 0}   # jax 0.6.0's own defaults, restored by uninstall()


# ----------------------------------------------------------------------------------------------------------------- the directory

def resolve_dir(environ: Optional[Mapping[str, str]] = None, *, key: Optional[str] = None) -> dict:
    """``{"dir", "dir_source", "key", "root"}`` — rule 1 (JAX_COMPILATION_CACHE_DIR adopted as given, key ``adopted``) else rule 2 (the keyed
    default under the XDG cache home). ``key`` given = no probe (tests); else pcc.key() of the running stack (exact: a missing part raises)."""
    environ = os.environ if environ is None else environ
    adopted = (environ.get(ENV_DIR) or "").strip()
    if adopted:
        d = os.path.abspath(os.path.expanduser(adopted))
        return {"dir": d, "dir_source": ENV_DIR, "key": "adopted", "root": None}
    home = (environ.get(ENV_XDG) or "").strip() or DEFAULT_CACHE_HOME
    root = os.path.abspath(os.path.expanduser(os.path.join(home, SUBDIR)))
    try:
        k = key or _pcc.key()
    except _pcc.PccError as e:
        raise _gates.cannot_run(CompileCacheError(f"{LEVER}: {e}")) from e
    return {"dir": _pcc.cache_dir(root, k), "dir_source": "default", "key": k, "root": root}


def _writable_dir(path: str) -> None:
    """Create ``path`` (parents included) and prove a file can be written there, or raise CompileCacheError naming the remedy."""
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, f".write_probe_{os.getpid()}")
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("ok")
        os.remove(probe)
    except OSError as e:
        raise _gates.cannot_run(CompileCacheError(
            f"{LEVER}: cache directory {path} is not writable ({e.__class__.__name__}: {e}); set {ENV_DIR} (or {ENV_XDG}) to a writable directory")) from e


def _entries(path: str) -> int:
    return int(_pcc.identity_key(path)["n_cache_entries"])


# --------------------------------------------------------------------------------------------------------------------- install

def _listen() -> None:
    """Count jax's own persistent-cache events (jax.monitoring) — registered once per process."""
    if _STATE["listener"]:
        return
    from jax import monitoring  # noqa: PLC0415

    def on_event(event: str, **_kw) -> None:
        c = _STATE["counts"]
        if event == EVENT_REQUESTS:
            c["requests"] += 1
        elif event == EVENT_HITS:
            c["hits"] += 1
        elif event == EVENT_MISSES:
            c["misses"] += 1

    def on_duration(event: str, seconds, **_kw) -> None:
        c = _STATE["counts"]
        if event == DURATION_COMPILE:
            c["compiled_s"] += float(seconds)
        elif event == DURATION_SAVED:
            c["saved_s"] += float(seconds)

    monitoring.register_event_listener(on_event)
    monitoring.register_event_duration_secs_listener(on_duration)
    _STATE["listener"] = True


def install(environ: Optional[MutableMapping[str, str]] = None, *, key: Optional[str] = None) -> dict:
    """Put the persistent compilation cache in force for this process (jax.config + the environment for any child), before the first
    compile. Idempotent. Returns the record ``{"dir", "dir_source", "key", "entries_install", "reinit", "via"}``."""
    with _LOCK:
        if _STATE["installed"]:
            return dict(_STATE["info"])
        environ = os.environ if environ is None else environ
        try:
            import jax  # noqa: PLC0415
            from jax._src import compilation_cache as _cc  # noqa: PLC0415
        except Exception as e:  # noqa: BLE001
            raise _gates.cannot_run(CompileCacheError(f"{LEVER}: jax is not importable ({e!r})")) from e
        where = resolve_dir(environ, key=key)
        _writable_dir(where["dir"])
        exports = _pcc.plan(where["dir"], autotune="off", environ=environ, exact=True)["exports"]     # JAX_COMPILATION_CACHE_DIR + the store-everything thresholds; XLA_FLAGS untouched
        for k, v in exports.items():
            if k != _pcc.XLA_FLAGS_VAR:
                environ[k] = v
        for name, value in _pcc.JAX_CONFIG.items():
            jax.config.update(name, where["dir"] if value is None else value)
        reinit = 0
        if getattr(_cc, "_cache_initialized", False) and getattr(_cc, "_cache", None) is None:   # jax consulted its cache state before we set the directory: reset so the directory takes effect
            _cc.reset_cache()
            reinit = 1
        _listen()
        info = {"dir": where["dir"], "dir_source": where["dir_source"], "key": where["key"], "entries_install": _entries(where["dir"]),
                "reinit": reinit, "via": "jax.config+environ", "autotune": AUTOTUNE_WORD}
        _STATE.update(installed=True, info=info, exports=dict(exports))
        _register_exit_line()
        return dict(info)


def _register_exit_line() -> None:
    """The exit census line through the package's ONE exit printer (colabdesign_opt.kernels.register_exit_line, registry order); if the
    printer is not importable or refuses the id (not in registry.LEVERS), the line goes through the core's tally under its own tag."""
    try:
        from .kernels import register_exit_line  # noqa: PLC0415
        register_exit_line(LEVER, exit_line)
    except (ImportError, ValueError):
        _report.register_exit_tally(TAG + "." + LEVER, exit_line)


def installed() -> bool:
    return bool(_STATE["installed"])


def uninstall(environ: Optional[MutableMapping[str, str]] = None) -> None:
    """Take the cache out of force for compiles after this call (jax.config back to jax's defaults, the exported variables removed, jax's
    cache state reset); entries already written stay on disk; the exit line then reads state=off."""
    with _LOCK:
        if not _STATE["installed"]:
            return
        environ = os.environ if environ is None else environ
        try:
            import jax  # noqa: PLC0415
            from jax._src import compilation_cache as _cc  # noqa: PLC0415
            for name, value in _JAX_DEFAULTS.items():
                jax.config.update(name, value)
            _cc.reset_cache()
        except Exception:  # noqa: BLE001
            pass
        for k in (_STATE.get("exports") or {}):
            if k != _pcc.XLA_FLAGS_VAR:
                environ.pop(k, None)
        _STATE.update(installed=False, info=None, exports=None)


def off_line(reason: str) -> str:
    """The lever's LEVER line with state=off and the reason (an ablated / refused mode prints it)."""
    return _report.lever_line(TAG, LEVER, "off", reason=reason, impl=IMPL, origin="core", numerics=NUMERICS)


# -------------------------------------------------------------------------------------------------------------------- evidence

def evidence() -> dict:
    info = _STATE["info"] or {}
    out = {"installed": bool(_STATE["installed"]), **info, **_STATE["counts"]}
    out["entries_now"] = _entries(info["dir"]) if info.get("dir") else 0
    return out


def line_of(ev: Optional[dict] = None) -> str:
    """The lever's one LEVER line from ``evidence()`` (state=on) — or state=off when the lever was never installed in this process."""
    ev = evidence() if ev is None else ev
    if not ev.get("installed"):
        return _report.lever_line(TAG, LEVER, "off", impl=IMPL, origin="core", numerics=NUMERICS)
    return _report.lever_line(
        TAG, LEVER, "on", impl=IMPL, origin="core", numerics=NUMERICS,
        dir=_blank_free(ev["dir"]), dir_source=ev["dir_source"], key=ev["key"], autotune=ev.get("autotune", AUTOTUNE_WORD),
        requests=ev["requests"], hits=ev["hits"], misses=ev["misses"], compiled_s=f"{ev.get('compiled_s', 0.0):.1f}", saved_s=f"{ev.get('saved_s', 0.0):.1f}",
        entries=f"{ev['entries_install']}->{ev['entries_now']}",
        reinit=ev["reinit"], source="exit", pid=os.getpid())


def exit_line() -> str:
    return line_of()


def _blank_free(text) -> str:
    """A path as ONE token of the line (runs of blanks -> ``_``; the line is split on blanks)."""
    return re.sub(r"\s+", "_", str(text)) or "none"


def reset_for_tests() -> None:
    with _LOCK:
        _STATE.update(installed=False, info=None, exports=None, counts={"requests": 0, "hits": 0, "misses": 0, "compiled_s": 0.0, "saved_s": 0.0})
