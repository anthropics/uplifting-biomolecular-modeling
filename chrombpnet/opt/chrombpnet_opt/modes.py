"""The mode table (the one place a mode is defined) and the kit's documented line.

  fast   the kit's documented form: `python tf/pred_bw_fast.py <the stock pred_bw arguments>` with NO extra flags, at the SHIPPED numerics
          (TF32 convolutions, on the tensor cores where the K1 route runs); the kit decides the forward route per (GPU class, mode) from its own
          tables (torch/chrombpnet_k1/arch_tiles.json entries[*].default_route, read HERE-first by tf/chrombpnet_fastkit/fastdefault.py, the class
          table there as its fallback) and prints its own line. Matches stock as shipped to TF32 precision. The package DEFAULT.
  exact  the same documented form with upstream's determinism settings composed by the package (det.py: TF32 off, deterministic ops, seed 0 —
          the seed hook on PYTHONPATH): the K1 convolutions run as fp32 chains in stock's summation order (route k1 precision fp32); refused
          by name on a card whose tables give no K1 route under the recipe (A100 / sm_80, B200, no CUDA device: stack.activate). Bit-for-bit
          equal to `chrombpnet pred_bw` run with the same settings (`--mode off --det 1`).
  off     stock: the upstream console script `chrombpnet pred_bw` (stock setup.py:27) in a clean subprocess with all kit and package
          variables removed and proved absent (stack.STOCK_FORBIDDEN_*).

The stock subcommand the kit covers is `pred_bw` (SUBCOMMAND); every other subcommand runs stock under the environment route.
"""
import os
from typing import Tuple

ENV = "CHROMBPNET_OPT"
ENV_DET = "CHROMBPNET_OPT_DET"                                   # "1" = `--mode off --det 1`: stock with TensorFlow's determinism settings (the reference `exact` equals); the kit modes carry their numerics themselves
MODES = ("off", "exact", "fast")                               # type: Tuple[str, ...]
DEFAULT_MODE = "fast"
DET_MODES = ("exact",)                                          # the kit modes that compose the deterministic recipe (det.py) — by the mode, never by a flag
STOCK_CONSOLE_SCRIPT = "chrombpnet"                             # stock setup.py:27  'chrombpnet = chrombpnet.CHROMBPNET:main'
SUBCOMMAND = "pred_bw"                                          # stock chrombpnet/parsers.py:37
KIT_LINE_RELPATH = os.path.join("tf", "pred_bw_fast.py")        # the script of the documented form


def check_mode(mode: str) -> str:
    """Normalise and validate a mode name; an unknown name raises ValueError."""
    m = (mode or "").strip().lower()
    if m not in MODES:
        raise ValueError("unknown mode {!r} (expected {})".format(mode, "|".join(MODES)))
    return m


def kit_line_script(kit_home: str) -> str:
    """Absolute path of the kit's documented entry script."""
    return os.path.join(kit_home, KIT_LINE_RELPATH)


def documented_line(kit_home: str, args, python: str = None) -> list:
    """The documented line as argv: `python <kit>/tf/pred_bw_fast.py <the stock pred_bw arguments verbatim>` — no flag added."""
    return [python or "python", kit_line_script(kit_home)] + list(args)

