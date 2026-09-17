"""The kit driver at the process boundary: compose one arm (environment + argv) and run it — a fresh python process per design, started
from the kit's `tools/` directory with the row's environment exported, `public_design_run.py --tag TAG --out OUT` followed by the row's
flags. Every route of the package that produces a design — `design --mode exact`, `design --mode off` (through stock_design.py), `warm` —
goes through ``compose()`` and ``launch()``; nothing here knows a lever value: the environment and the flags come from the resolved row
(modes.resolve), the recipe's variables from det.py. ``run_logged`` is the process runner (own process group, merged pipes, the group
killed at a wall-clock deadline when a timeout is given).
"""
from __future__ import annotations

import os
import sys
from typing import Callable, List, Optional, Tuple

from . import det, stack
from .modes import DRIVER_RELPATH, Resolution


def strip_env(env: dict, prefixes: List[str], exceptions: List[str]) -> Tuple[dict, List[str]]:
    """Drop every variable whose name starts with one of `prefixes` (exceptions kept); returns (new env, the names dropped)."""
    out, dropped = {}, []
    for k, v in env.items():
        if any(k.startswith(p) for p in prefixes) and k not in exceptions:
            dropped.append(k)
            continue
        out[k] = v
    return out, dropped


def compose(res: Resolution, *, seed: int, out_dir: str, tag: str, shape_flags: List[str], settings_flags: List[str], det_level: int,
            features_out: Optional[str] = None, base_env: Optional[dict] = None, python: Optional[str] = None) -> Tuple[List[str], dict, dict]:
    """(argv, env, notes): the driver command for `res` at `seed`, the environment `run_arm` would export — the caller's, stripped of the
    package's switches and of every P1 variable (the row decides; a pre-set one is dropped and noted), plus the row's own, plus the
    recipe's (det.py) — and what was dropped."""
    base_env = dict(os.environ if base_env is None else base_env)
    p = stack.pins()
    prefixes = stack.stock_env_absent(p)
    env, dropped = strip_env(base_env, prefixes, stack.stock_env_exceptions(p))
    for k in stack.PACKAGE_ENV:
        if env.pop(k, None) is not None and k not in dropped:
            dropped.append(k)
    env.update(res.env)
    env = det.apply_env(env, det_level)
    argv = [python or os.path.abspath(sys.executable), os.path.join(stack.kit_home(), DRIVER_RELPATH), "--tag", tag, "--out", out_dir, "--seed", str(seed)]
    argv += list(shape_flags) + list(settings_flags) + list(res.flags)
    if features_out:
        argv += ["--features-out", features_out]
    notes = {"dropped": dropped, "row_env": dict(res.env), "det_env": {k: env[k] for k in det.ENV if k in env}, "cwd": os.path.join(stack.kit_home(), "tools")}
    return argv, env, notes


def run_logged(cmd: List[str], on_line: Callable[[str], None], timeout: Optional[float] = None, **popen_kw) -> Tuple[int, bool]:
    """Run `cmd` in its own process group with stdout+stderr merged, calling ``on_line`` for every line as it arrives, and return
    ``(exit_code, timed_out)``. The deadline is wall clock: ``timeout`` seconds after the start the process group is killed whether or
    not it has printed anything (the core's runner, `opt_core.process.run_logged`)."""
    from opt_core.process import run_logged as core_run_logged                  # the core's runner: own process group, merged pipes, SIGKILL of the group at the deadline
    res = core_run_logged(list(cmd), timeout_s=timeout, on_line=lambda line: on_line(line.rstrip("\n")), **popen_kw)
    return res.rc, res.timed_out


def launch(argv: List[str], env: dict, *, cwd: str, timeout: Optional[float], on_line: Callable[[str], None]) -> Tuple[int, bool]:
    """Run the composed arm in its own process group with a wall-clock deadline (``run_logged``); (exit_code, timed_out)."""
    return run_logged(argv, on_line, timeout=timeout, cwd=cwd, env=env)
