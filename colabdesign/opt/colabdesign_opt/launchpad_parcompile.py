"""Lever `parcompile` — XLA compiles the design step's GPU kernels in parallel: the LLVM module of each executable is split into shards that are
optimised and assembled (LLVM -> PTX -> cubin) on a thread pool instead of one thread (`--xla_gpu_enable_llvm_module_compilation_parallelism=true`,
off by default in the pinned XLA; `--xla_gpu_force_compilation_parallelism=<threads>` sizes the pool). HLO optimisation, buffer assignment and
the kernels' code are untouched: each kernel is generated from the same LLVM IR with the same options, shard by shard.

What it changes: cold compile seconds only; steady s/step identical; with `compilecache` warm it changes nothing (nothing compiles). Numerics
class: exact — the executables compute what executables compiled under stock's flags compute, bit for bit — and the flags enter jax's cache key,
so parcompile and non-parcompile executables never alias in the cache.

Mechanism and its one precondition. Upstream ColabDesign assigns `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` wholesale when `colabdesign` is
imported (colabdesign/__init__.py) and XLA reads XLA_FLAGS once, when jax initialises its backends. So `install()` imports `colabdesign` FIRST (its
assignment happens, once), then APPENDS the two flags (upstream's flag kept verbatim, ours after it), and steps aside by name if jax's backends were
already initialised in this process (the flags could not take effect: `ParcompileError`, cannot_run) — install order puts this lever before
anything that touches a device (before `pallas`, whose probe initialises the backends). Threads: min(16, the CPUs this process may run on
(affinity-aware)), at least 2 — printed on the line with the CPU count it was derived from.

    [colabdesign-opt] LEVER name=parcompile state=on impl=xla_llvm_module_parallelism@kit origin=kit numerics=exact threads=<n> cpus=<n> flags=<the two flags> xla_flags_kept=<upstream's> source=install pid=<pid>

Protocol (registry.py): NUMERICS, REFUSALS, install() / installed() / uninstall() / off_line(reason) / evidence().

    from colabdesign_opt import launchpad_parcompile
    launchpad_parcompile.install()       # idempotent; before jax initialises its backends
"""
from __future__ import annotations

import os
import re
import threading
from typing import MutableMapping, Optional

from opt_core import gates as _gates
from opt_core import report as _report

from .names import TAG

LEVER = "parcompile"
NUMERICS = "exact"                                                # registry.LEVERS[parcompile].numerics: the kernels' code is unchanged (the thread count does not enter the generated code)
IMPL = "xla_llvm_module_parallelism@kit"
XLA_FLAGS_VAR = "XLA_FLAGS"
FLAG_SPLIT = "--xla_gpu_enable_llvm_module_compilation_parallelism=true"
FLAG_THREADS = "--xla_gpu_force_compilation_parallelism={threads}"
MIN_THREADS, MAX_THREADS = 2, 16                                  # the pool: the CPUs this process may use, capped — min(16, cpus), at least 2

_LOCK = threading.Lock()
_STATE: dict = {"installed": False, "info": None}


class ParcompileError(RuntimeError):
    """The flags cannot take effect in this process (jax's backends already initialised, jax/colabdesign not importable): the lever steps aside by name (cannot_run)."""
    cannot_run = True


REFUSALS = (ParcompileError,)                                     # what install() raises when the lever cannot run here (levers.install steps the lever aside by name)


CGROUP_ROOT = "/sys/fs/cgroup"
PROC_SELF_CGROUP = "/proc/self/cgroup"


def _cgroup_dirs(root: str = CGROUP_ROOT, proc: str = PROC_SELF_CGROUP) -> list:
    """Candidate cgroup directories of this process, most specific first: the v2 unified path (`0::/path`) and the v1 `cpu` controller path from
    /proc/self/cgroup, each joined under `root` (and under `root/cpu` for v1), then the roots themselves (a container's namespace root)."""
    out = []
    try:
        with open(proc, encoding="utf-8") as fh:
            for ln in fh:
                parts = ln.strip().split(":", 2)
                if len(parts) != 3:
                    continue
                _, ctrls, path = parts
                rel = path.lstrip("/")
                if ctrls == "":                                                     # v2 unified hierarchy
                    out.append(os.path.join(root, rel) if rel else root)
                elif "cpu" in ctrls.split(","):                                    # v1 cpu controller
                    base = os.path.join(root, "cpu")
                    out.append(os.path.join(base, rel) if rel else base)
                    base2 = os.path.join(root, ctrls)
                    out.append(os.path.join(base2, rel) if rel else base2)
    except OSError:
        pass
    out += [root, os.path.join(root, "cpu"), os.path.join(root, "cpu,cpuacct")]
    seen, dirs = set(), []
    for d in out:
        if d not in seen:
            seen.add(d); dirs.append(d)
    return dirs


def cgroup_cpu_quota(dirs: Optional[list] = None) -> Optional[int]:
    """The container's CPU quota in whole CPUs — ceil(quota / period) from cgroup v2 `cpu.max` ("<quota> <period>" | "max <period>") or v1
    `cpu.cfs_quota_us` / `cpu.cfs_period_us` (-1 = none) in the first candidate directory that states one; None when unlimited / unreadable."""
    import math  # noqa: PLC0415
    for d in (_cgroup_dirs() if dirs is None else dirs):
        try:
            with open(os.path.join(d, "cpu.max"), encoding="utf-8") as fh:
                q, per = fh.read().split()[:2]
            if q == "max":
                return None
            if int(per) > 0:
                return max(1, math.ceil(int(q) / int(per)))
        except (OSError, ValueError):
            pass
        try:
            with open(os.path.join(d, "cpu.cfs_quota_us"), encoding="utf-8") as fh:
                q = int(fh.read().strip())
            with open(os.path.join(d, "cpu.cfs_period_us"), encoding="utf-8") as fh:
                per = int(fh.read().strip())
            if q <= 0:
                return None
            if per > 0:
                return max(1, math.ceil(q / per))
        except (OSError, ValueError):
            pass
    return None


def cpu_count() -> int:
    """The CPUs this process may actually use: min(sched_getaffinity (cpuset / taskset aware; os.cpu_count() where the platform lacks it),
    the cgroup CPU quota when one is set) — a container's CPU limit is often a quota, not a cpuset, so both are read; never below 1."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = int(os.cpu_count() or 1)
    q = cgroup_cpu_quota()
    return max(1, min(n, q) if q else n)


def thread_count(cpus: Optional[int] = None) -> int:
    """The compile pool size: min(MAX_THREADS, cpus), at least MIN_THREADS."""
    n = cpu_count() if cpus is None else int(cpus)
    return max(MIN_THREADS, min(MAX_THREADS, n))


def flags(threads: int) -> list:
    return [FLAG_SPLIT, FLAG_THREADS.format(threads=int(threads))]


def compose(current: Optional[str], threads: int) -> str:
    """`current` XLA_FLAGS (upstream's assignment, kept verbatim) with the lever's two flags appended once (a flag already present with the same
    value is not repeated; the same flag with another value already present is left as it is and ours is appended after it — XLA takes the last occurrence)."""
    parts = [p for p in re.split(r"\s+", (current or "").strip()) if p]
    for f in flags(threads):
        if f not in parts:
            parts.append(f)
    return " ".join(parts)


def _backends_initialized() -> bool:
    try:
        from jax._src import xla_bridge as xb  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return False
    try:
        return bool(xb.backends_are_initialized())
    except AttributeError:
        return bool(getattr(xb, "_backends", {}))


def install(environ: Optional[MutableMapping[str, str]] = None, *, threads: Optional[int] = None) -> dict:
    """Append the flags to XLA_FLAGS (after importing colabdesign, whose import assigns the variable) before jax initialises its backends.
    Idempotent. Returns ``{"threads", "flags", "xla_flags_kept", "xla_flags"}``."""
    with _LOCK:
        if _STATE["installed"]:
            return dict(_STATE["info"])
        environ = os.environ if environ is None else environ
        if _backends_initialized():
            raise _gates.cannot_run(ParcompileError(
                f"{LEVER}: jax's backends are already initialised in this process — XLA_FLAGS is read once at backend creation, so the compile-parallelism "
                f"flags cannot take effect; install {LEVER} before anything touches a device"))
        try:
            import colabdesign  # noqa: F401, PLC0415 — its import assigns XLA_FLAGS wholesale (colabdesign/__init__.py); it must happen BEFORE we append
        except Exception as e:  # noqa: BLE001
            raise _gates.cannot_run(ParcompileError(f"{LEVER}: colabdesign is not importable ({e!r})")) from e
        if _backends_initialized():
            raise _gates.cannot_run(ParcompileError(f"{LEVER}: importing colabdesign initialised jax's backends before the flags were appended"))
        kept = environ.get(XLA_FLAGS_VAR, "") if environ is os.environ else environ.get(XLA_FLAGS_VAR, os.environ.get(XLA_FLAGS_VAR, ""))
        n = thread_count() if threads is None else max(MIN_THREADS, int(threads))
        new = compose(kept, n)
        environ[XLA_FLAGS_VAR] = new
        if environ is not os.environ:
            os.environ[XLA_FLAGS_VAR] = new
        info = {"threads": n, "cpus": cpu_count(), "flags": flags(n), "xla_flags_kept": kept, "xla_flags": new}
        _STATE.update(installed=True, info=info)
        info["line"] = _report.emit(line_of())                                  # ONE LEVER line, at install (the flags are in force from here; nothing later can change them)
        _STATE["info"] = info
        return dict(info)


def installed() -> bool:
    return bool(_STATE["installed"])


def uninstall(environ: Optional[MutableMapping[str, str]] = None) -> None:
    """Put XLA_FLAGS back to what install() found (upstream's assignment). XLA read the variable when jax created its backends: if that has
    happened, executables compiled from here on still use the pool — the variable is restored for any child process and the record says off."""
    with _LOCK:
        if not _STATE["installed"]:
            return
        environ = os.environ if environ is None else environ
        kept = (_STATE["info"] or {}).get("xla_flags_kept", "")
        for envm in {id(environ): environ, id(os.environ): os.environ}.values():
            if kept:
                envm[XLA_FLAGS_VAR] = kept
            else:
                envm.pop(XLA_FLAGS_VAR, None)
        _STATE.update(installed=False, info=None)


def off_line(reason: str) -> str:
    """The lever's LEVER line with state=off and the reason (an ablated / refused mode prints it)."""
    return _report.lever_line(TAG, LEVER, "off", reason=reason, impl=IMPL, origin="kit", numerics=NUMERICS)


def evidence() -> dict:
    info = _STATE["info"] or {}
    return {"installed": bool(_STATE["installed"]), **info, "xla_flags_now": os.environ.get(XLA_FLAGS_VAR, ""), "backends_initialized": _backends_initialized()}


def line_of(ev: Optional[dict] = None) -> str:
    ev = evidence() if ev is None else ev
    if not ev.get("installed"):
        return _report.lever_line(TAG, LEVER, "off", impl=IMPL, origin="kit", numerics=NUMERICS)
    return _report.lever_line(TAG, LEVER, "on", impl=IMPL, origin="kit", numerics=NUMERICS, threads=ev["threads"], cpus=ev.get("cpus", "none"), flags=",".join(ev["flags"]),
                              xla_flags_kept=_blank_free(ev["xla_flags_kept"]) or "none", source="install", pid=os.getpid())


def _blank_free(text) -> str:
    return re.sub(r"\s+", ",", str(text or "").strip())


def reset_for_tests() -> None:
    with _LOCK:
        _STATE.update(installed=False, info=None)
