"""``esmc-opt`` — the package's one command (``python -m esmc_opt`` is the same entry).

  check   --variant V [--mode M]: a dry run — resolves and gates the mode on this box (the kits present, the stock package at its pinned
          upstream commit, the stack versions, a visible GPU), applies nothing, prints DRY-RUN (or NOT ACTIVE with the reason).
There is no inference command: ESM C is a Python API and the kit is a drop-in beside it — ``ESMC_OPT=exact`` (or ``esmc_opt.enable``)
and the user's own ``ESMC.from_pretrained`` / ``encode`` / ``logits`` calls (README). Exit codes: 0 the mode would activate (or the mode
is ``off``) · 2 usage · 3 the mode could not be activated (the NOT ACTIVE line names the gate).
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional

from . import modes

EXIT_OK, EXIT_USAGE, EXIT_NOT_ACTIVE = 0, 2, 3


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="esmc-opt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="dry run: resolve the mode on this box, apply nothing")
    c.add_argument("--variant", required=True, help="|".join(modes.VARIANTS))
    c.add_argument("--mode", default=None, help=f"{'|'.join(modes.MODES)} (default: ESMC_OPT, else {modes.DEFAULT_MODE})")
    return ap


def _mode_of(a) -> str:
    return (a.mode or os.environ.get("ESMC_OPT") or modes.DEFAULT_MODE).strip().lower()


def cmd_check(a) -> int:
    from . import stack
    mode = _mode_of(a)
    rep = stack.activate(mode, a.variant, None, strict=False, trigger="cli:check", dry_run=True)
    if mode in modes.STOCK_MODES:
        return EXIT_OK
    return EXIT_OK if not rep.get("reason") else EXIT_NOT_ACTIVE


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    return {"check": cmd_check}[a.cmd](a)
