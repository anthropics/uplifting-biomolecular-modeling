"""Subprocesses of a kit: the wall-clock watchdog runner and the child environment.

Contract. :func:`run_logged` runs a command in its own process group with stdout+stderr merged; a reader thread pumps every line to
``on_line`` (and to the log file when given) as it arrives; the deadline is wall clock — ``timeout_s`` after the start the whole process
group is killed whether or not the run is printing, so a silent, hung or stuck-in-a-child run cannot outlive it. The result names the
event: ``timed_out``, ``error="SelftestTimeout"`` and ``reason="<what> exceeded --timeout N s: process group killed at W s"``.
:func:`child_env` builds a child's environment from the caller's minus the kit package's own switches (never exported to a model
process) plus explicit exports. :func:`run_step` is one named step of a pass over :func:`run_logged`: a COMMAND line before, a STEP line
after, and a record ``{step, argv, rc, wall_s, log, ok[, error, reason]}`` — ``rc`` is the child's, ``-1`` on the watchdog's timeout,
``-2`` when the command could not be launched; both are named in ``error``.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Sequence

REAPER_GRACE_S = 5.0


@dataclass
class RunResult:
    rc: int
    timed_out: bool
    wall_s: float
    log_path: Optional[str] = None
    error: Optional[str] = None
    reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.rc == 0 and not self.timed_out


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the child's process group; when the group cannot be signalled (already gone, or not a group leader) kill the child itself."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass


def run_logged(cmd: Sequence[str], *, timeout_s: Optional[float] = None, on_line: Optional[Callable[[str], None]] = None,
               log_path: Optional[str] = None, what: str = "run", env: Optional[Mapping[str, str]] = None, cwd: Optional[str] = None,
               **popen_kw) -> RunResult:
    """Run ``cmd`` (module contract). ``on_line`` receives each line with its newline; ``log_path`` receives the merged transcript."""
    t0 = time.monotonic()
    log = open(log_path, "w", encoding="utf-8") if log_path else None
    proc = subprocess.Popen(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
                            start_new_session=True, env=dict(env) if env is not None else None, cwd=cwd, **popen_kw)

    def pump():
        for line in proc.stdout:
            if log is not None:
                log.write(line)
                log.flush()
            if on_line is not None:
                on_line(line)

    reader = threading.Thread(target=pump, name="run_logged-transcript", daemon=True)
    reader.start()
    timed_out = False
    try:
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(proc)
            rc = proc.wait()
    finally:
        if proc.poll() is None:                                            # an exception while waiting (KeyboardInterrupt): no orphans
            _kill_group(proc)
            proc.wait()
        reader.join(REAPER_GRACE_S if timed_out else None)                 # the pipe closes when the last process of the group is gone
        proc.stdout.close()
        if log is not None:
            log.close()
    wall = time.monotonic() - t0
    if timed_out:
        return RunResult(rc=rc, timed_out=True, wall_s=wall, log_path=log_path, error="SelftestTimeout",
                         reason=f"{what} exceeded --timeout {timeout_s:g} s: process group killed at {wall:.1f} s")
    return RunResult(rc=rc, timed_out=False, wall_s=wall, log_path=log_path)


def child_env(base: Optional[Mapping[str, str]] = None, *, strip_prefixes: Iterable[str] = (), strip_names: Iterable[str] = (),
              export: Optional[Mapping[str, str]] = None) -> dict:
    """The caller's environment (``os.environ`` by default) minus the names under ``strip_prefixes`` and ``strip_names``, plus ``export``."""
    base = os.environ if base is None else base
    prefixes = tuple(strip_prefixes)
    names = set(strip_names)
    env = {k: v for k, v in base.items() if k not in names and not (prefixes and k.startswith(prefixes))}
    if export:
        env.update({k: str(v) for k, v in export.items()})
    return env


RC_TIMEOUT, RC_LAUNCH = -1, -2                                   # run_step's rc when the watchdog fired / the command could not be launched


def run_step(step: str, argv: Sequence[str], *, prefix: str, env: Optional[Mapping[str, str]] = None, log_path: Optional[str] = None,
             timeout_s: Optional[float] = None, cwd: Optional[str] = None, on_line: Optional[Callable[[str], None]] = None,
             stream=None) -> dict:
    """One named step of a pass (module contract). Prints ``<prefix> COMMAND step=<step> argv=<argv joined>`` before and
    ``<prefix> STEP step=<step> rc=<rc> wall_s=<s> log=<log_path>`` after (:func:`opt_core.report.line` / ``emit``)."""
    from .report import emit, line
    emit(line(prefix, "COMMAND", step=step, argv=" ".join(str(a) for a in argv)), stream)
    t0 = time.monotonic()
    rec: dict = {"step": step, "argv": list(argv), "log": log_path}
    try:
        r = run_logged(list(argv), timeout_s=timeout_s, on_line=on_line, log_path=log_path, what=step, env=env, cwd=cwd)
        if r.timed_out:
            rec.update(rc=RC_TIMEOUT, error=f"timeout after {timeout_s:g}s", reason=r.reason)
        else:
            rec.update(rc=r.rc)
    except OSError as e:
        rec.update(rc=RC_LAUNCH, error=f"could not launch: {e}")
    rec["wall_s"] = round(time.monotonic() - t0, 3)
    rec["ok"] = rec["rc"] == 0
    emit(line(prefix, "STEP", step=step, rc=rec["rc"], wall_s=rec["wall_s"], log=log_path), stream)
    return rec
