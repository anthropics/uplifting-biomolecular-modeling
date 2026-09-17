"""Mode off: the stock atlasfold CLI in a clean subprocess — no ATLASFOLD_OPT*/AFO_*/FPF_*/OPT_CORE_* variable, the kit's opt/ directory
off sys.path, the .pth hook inert (ATLASFOLD_OPT unset) — so 'off' is stock by construction (stock/PINS.json stock_environment). The stock
process loads exactly one kit file, by path: phase_timing.py (standard library only) — the per-item PHASE / PEAK timing lines, no value changed."""
import os
import subprocess
import sys
from typing import List, Sequence

ABSENT_PREFIXES = ("ATLASFOLD_OPT", "AFO_", "FPF_", "OPT_CORE_")


def clean_env(extra: dict = None) -> dict:
    env = {k: v for k, v in os.environ.items() if not any(k.startswith(p) for p in ABSENT_PREFIXES)}
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # <engine>/opt
    pp = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and os.path.abspath(p) != here]
    env["PYTHONPATH"] = os.pathsep.join(pp)
    env.update(extra or {})
    return env


PHASE_TIMING = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase_timing.py")   # standard library only; loaded BY PATH in the stock process (opt/ stays off sys.path)
BOOT = ("import importlib.util, sys; "
        f"_s = importlib.util.spec_from_file_location('afo_phase_timing', {PHASE_TIMING!r}); _m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m); _m.install(); "
        "from atlasfold.cli import main; sys.exit(main())")


def command(stock_argv: Sequence[str]) -> List[str]:
    """The stock console entry with the arguments unchanged; the ONE kit file the stock process holds is phase_timing.py (the per-item PHASE /
    PEAK lines: the stock model's trunk / sampler / confidence seconds between CUDA events — timing only, no value changed)."""
    return [sys.executable, "-c", BOOT, *stock_argv]


def run(stock_argv: Sequence[str], extra_env: dict = None) -> int:
    return subprocess.call(command(stock_argv), env=clean_env(extra_env))
