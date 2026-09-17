"""Mode ``exact`` on the command line — the upstream CLI in ONE clean subprocess with the kit armed:

    python -m e1_opt.kit_score --det 0|1 --variant V --pins-path P -- <E1.tools.score options>

The child arms the kit exactly as the env / API route does (stack.arm: `E1Predictor.__init__` / `E1Scorer.__init__` wrapped so the kit's
`apply(model, size=V)` runs once, on the tool's own model, at its first scorer construction; the KERNELS proof armed innermost, accel.py)
and then runs upstream's module in-process (`runpy.run_module("E1.tools.score")`, the tool's own argv) — the same code a user's
`E1_OPT=exact python -m E1.tools.score …` runs, without depending on the interpreter's start-up hook. The kit prints its KIT / LEVER lines,
the proof prints

    [e1-opt exact] KERNELS route=exact flash_attn=<word> hub_layernorm=<word> flex_attention=<word> site=<stock|kit_attn> upstream_says=<bool>

— an accelerator that is absent or fell back on this host is NAMED by its word there and is the mode's refusal BY NAME (`[e1-opt] NOT
ACTIVE: …`, exit 3): the mode is all of its levers, never a subset under its name; a lever that cannot be put in force is the same refusal
(the kit's KitRefused, worded on the NOT ACTIVE line by stack.py's wrap). `--det 1`: the recipe's environment is exported into the child by
the caller and this runner applies the recipe's seeds and torch switches (det.py) before the tool runs; the autotune pin is the kit's own.
The parent (cli.py) relays the child's stdout / stderr verbatim and reads the kit's lines back (`relay`, `refusal`).
"""
from __future__ import annotations

import argparse
import re
import runpy
import subprocess
import sys
import threading
import time
from typing import List, Optional, Tuple

from . import report
from ._names import MUST_BE_ABSENT_PREFIXES   # the one names module (no re-typing)

TOOL_MODULE = "E1.tools.score"                 # upstream's documented entry (`python -m E1.tools.score …`)
CHILD_PREFIXES = (report.STOCK_PREFIX, report.KIT_PREFIX)   # the children's own line prefixes the relay collects (the KERNELS lines)
RE_KIT_LINE = re.compile(r"^\[e1-opt\] (?:KIT|LEVER)\b")                            # the kit's own lines: KIT and LEVER (one per lever)
KIT_REFUSED = "KitRefused"                    # the kit's fail-loud class (kits/v1 KitRefused; the tests hold the name to the kit's own)
RE_KIT_REFUSED = re.compile(r"^(?:[\w.]+\.)?(?:" + KIT_REFUSED + r"|PinDrift|PinUncensused): (?P<reason>.+)$")   # its line on the child's stderr (an uncaught refusal: rc 1 by traceback)


def command(python: str, tool_args: List[str], *, det: int = 0, variant: Optional[str] = None, pins_path: Optional[str] = None) -> List[str]:
    """The child's argv: this runner, then `--`, then the tool's own options verbatim."""
    return [python, "-m", "e1_opt.kit_score", "--det", str(int(det)), "--variant", str(variant), "--pins-path", str(pins_path), "--", *tool_args]


def relay(cmd: List[str], env: dict, cwd: Optional[str] = None, out=None, timeout: Optional[float] = None, err=None) -> Tuple[int, float, List[str]]:
    """Run `cmd`; relay its stdout line by line to `out` (default: this process's stdout) and its stderr to `err` (default: this
    process's stderr) verbatim; collect the kit's own lines — its stdout lines and, on stderr, the line of its refusal by name
    (`<module>.KitRefused: <reason>` from an uncaught refusal; `refusal()` reads it back). Returns (exit_code, wall_s, kit_lines). The
    wall is the process wall of the command (start -> exit)."""
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    kit_lines, err_lines = [], []

    def pump_err():
        for line in p.stderr:
            err.write(line)
            err.flush()
            s = line.rstrip("\n")
            if RE_KIT_REFUSED.match(s):
                err_lines.append(s)
    t = threading.Thread(target=pump_err, daemon=True)
    t.start()
    try:
        for line in p.stdout:
            out.write(line)
            out.flush()
            s = line.rstrip("\n")
            if RE_KIT_LINE.match(s) or report.RE_NOT_ACTIVE.match(s) or s.startswith(CHILD_PREFIXES) or s.startswith("STOCK_ENV_CHECK:") or s.startswith(MUST_BE_ABSENT_PREFIXES):
                kit_lines.append(s)
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        rc = p.wait()
        rc = rc if rc else 124
    finally:
        t.join()
        for stream in (p.stdout, p.stderr):
            if stream:
                stream.close()
    return rc, time.perf_counter() - t0, kit_lines + err_lines


def refusal(kit_lines: List[str]) -> Optional[str]:
    """The reason of the kit's refusal by name (its `KitRefused: <reason>` line, relayed from the child's stderr), or None."""
    for s in kit_lines:
        m = RE_KIT_REFUSED.match(s)
        if m:
            return m.group("reason")
    return None


# ------------------------------------------------------------------------------------------------------- the runner (child)
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m e1_opt.kit_score", description="the upstream CLI with the kit armed (mode exact)")
    ap.add_argument("--det", type=int, default=0, choices=(0, 1))
    ap.add_argument("--variant", required=True)
    ap.add_argument("--pins-path", required=True)
    ap.add_argument("tool_args", nargs=argparse.REMAINDER, help="-- then the E1.tools.score options")
    a = ap.parse_args(argv)
    if a.tool_args and a.tool_args[0] == "--":
        a.tool_args = a.tool_args[1:]
    return a


def main(argv: Optional[List[str]] = None) -> int:
    a = parse_args(argv)
    from . import stack, stock_score
    pins = stock_score.load_pins_by_path(a.pins_path)                       # under the package's own module name (the kit imports its own copy as engines.e1.kits.pins)
    if a.det:
        from . import det
        det.apply_env(pins, 1)
        det.apply_torch(pins, 1)
    stack.arm("exact", a.variant, det=bool(a.det))                           # the constructors wrapped: the kit applies at the tool's first scorer construction; the KERNELS proof innermost
    sys.argv = [TOOL_MODULE, *a.tool_args]
    try:
        runpy.run_module(TOOL_MODULE, run_name="__main__", alter_sys=True)
    except SystemExit as e:
        code = e.code
        return 0 if code is None else (code if isinstance(code, int) else 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
