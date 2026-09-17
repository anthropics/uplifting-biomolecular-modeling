"""The subprocess per arm: its environment (the caller's, stripped of every must-be-absent prefix of stock/PINS.json — the package's own
switches included; nothing is added: the arm's mode travels as the launcher's `--mode`), its command line (the arm's launcher,
launch.py), and the run itself: the child's stdout+stderr merged, teed line by line to `run.log` and to this process's stderr (the tree's one
runner, opt_core.process.run_logged).

The arm's paths travel as arguments (the design script's `--params-dir`; the launcher's `--pins`, `--mode`), never
as environment variables: the child's environment carries nothing of the package.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, List, Optional, Tuple

from opt_core import process as _core_process

from . import modes, stack


def strip_env(env: dict, prefixes: List[str], keep: Optional[dict] = None) -> Tuple[dict, List[str]]:
    """Drop every variable whose name starts with one of `prefixes`, then set `keep` (name -> value); returns (env, names dropped)."""
    out, dropped = {}, []
    for k, v in env.items():
        if any(k.startswith(p) for p in prefixes):
            dropped.append(k)
            continue
        out[k] = v
    for k in stack.PACKAGE_ENV:
        if out.pop(k, None) is not None and k not in dropped:
            dropped.append(k)
    out.update(keep or {})
    return out, sorted(dropped)


def compose(res: modes.Resolved, script_args: List[str], *, out_dir: str, base_env: Optional[dict] = None, environ=None) -> Tuple[List[str], dict, dict]:
    """(argv, env, notes) for the arm `res`: the launcher command (its own arguments, then `--` and the design script's) and the stripped environment."""
    base = dict(os.environ if base_env is None else base_env)
    p = stack.pins(environ)
    keep = {}                                                                   # neither arm carries a kit or lever variable: a mode is its lever set (launch.py --mode)
    env, dropped = strip_env(base, stack.stock_env_absent(p, "stock" if res.route == "stock" else "kit"), keep)   # the stock arm also drops jax's cache variables (stock compiles as shipped)
    env["PYTHONDONTWRITEBYTECODE"] = "1"                                       # the kit arm imports the package and the core from the tree: no __pycache__ beside them
    pins_path = stack.pins_path(environ)
    if res.route == "stock":
        argv = [sys.executable, "-s", "-m", "colabdesign_opt.stock_launch", "--pins", pins_path, "--", *script_args]
    else:
        argv = [sys.executable, "-s", "-m", "colabdesign_opt.kit_launch", "--pins", pins_path, "--mode", res.mode, "--", *script_args]
    notes = {"env_dropped": dropped, "env_kept": keep, "argv": argv}
    return argv, env, notes


def launch(argv: List[str], env: dict, *, cwd: str, log_path: str, timeout: Optional[float], echo: Optional[Callable[[str], None]] = None) -> dict:
    """Run the arm: every line to `log_path` and to `echo` (default: this process's stderr). Returns {"exit_code", "timed_out", "wall_s",
    "lines", "stamps" (each line's arrival, seconds since the launch — one per line; report.run_lines reads a design's start from them), "log"}."""
    lines: List[str] = []
    stamps: List[float] = []
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    echo = echo or (lambda s: (sys.stderr.write(s + "\n"), sys.stderr.flush()))
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as fh:
        def on_line(s: str):
            fh.write(s + "\n"); fh.flush()
            lines.append(s); stamps.append(time.perf_counter() - t0)                # arrival ≈ print time: the package's lines are flushed one by one (report.log) into the one merged pipe
            echo(s)
        r = _core_process.run_logged(argv, timeout_s=timeout, on_line=lambda raw: on_line(raw.rstrip("\n")), env=env, cwd=cwd)   # the tree's one runner: own process group, merged streams, wall-clock deadline, group kill
    return {"exit_code": r.rc, "timed_out": r.timed_out, "wall_s": time.perf_counter() - t0, "lines": lines, "stamps": stamps, "log": log_path}
