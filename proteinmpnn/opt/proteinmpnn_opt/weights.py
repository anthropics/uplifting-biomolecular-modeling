"""``run.sh install --weights DIR [--variant V]``: upstream's own files fetched into DIR the way upstream distributes them, then digest-checked
against stock/PINS.json (the digesting is stock/check_pins.py's ``check_weights``). A file already in DIR is kept and only checked; nothing is
deleted or fetched twice.

Variants soluble | vanilla (``--variant`` absent or either name — one checkout serves both): DIR becomes the stock ProteinMPNN checkout. Code and
repository weights are one git repository upstream, so the fetch is ``git clone <upstream.repo> DIR`` then ``git -C DIR checkout <upstream.commit>``
(stock/PINS.json "upstream"; the clone keeps its .git: the .fa header names ``git rev-parse HEAD``). A DIR that already holds protein_mpnn_run.py is
kept as it is (a HEAD other than the pinned commit is reported, as ``run.sh check`` reports it). Both repository weight sets (PINS.json "weights":
soluble_model_weights/v_48_020.pt, vanilla_model_weights/v_48_020.pt) are then digested. DIR is your MPNN_DIR.
Exit 0 with one ``WEIGHTS OK`` line naming the variable to export; exit 1 with a ``WEIGHTS FAILED`` line (the fetch could not run) or one
``WEIGHTS REFUSED`` line per pinned file that is absent or whose sha256 is not stock/PINS.json's (the file is left in place).

    python -m proteinmpnn_opt.weights DIR [--variant soluble|vanilla]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple

from . import stack
from .modes import VARIANTS

ENV_BASE = "MPNN_DIR"


class WeightsError(Exception):
    """The fetch could not run (a tool absent, a directory that cannot take the clone, upstream's command failed)."""


def check_pins_module():
    """stock/check_pins.py as a module: the tree's one digest checker (check_weights, git_head) — standard library only."""
    path = os.path.join(stack.stock_dir(), "check_pins.py")
    spec = importlib.util.spec_from_file_location("proteinmpnn_stock_check_pins", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(cmd: List[str], what: str) -> None:
    """One upstream command; its output relayed on stderr, a non-zero exit a WeightsError naming the command and its last words."""
    r = subprocess.run(cmd, text=True, capture_output=True)
    words = ((r.stdout or "") + (r.stderr or "")).strip()
    if words:
        print(words, file=sys.stderr)
    if r.returncode != 0:
        raise WeightsError(f"{what} failed (rc {r.returncode}): {' '.join(cmd)}" + (f" — {words[-600:]}" if words else ""))


# ------------------------------------------------------------------------------------------------ base variants: the ProteinMPNN checkout
def fetch_checkout(cp, pins: dict, dest: str) -> str:
    """The stock checkout at ``dest``: kept when protein_mpnn_run.py is there, else cloned at the pinned commit. Returns the one line saying which."""
    up = pins["upstream"]
    if os.path.isfile(os.path.join(dest, "protein_mpnn_run.py")):
        head = cp.git_head(dest)
        at = f"HEAD {head[:8]}" if head else "not a git checkout: the .fa header will say git_hash=unknown"
        pin = "" if head == up["commit"] else f"; the pinned commit is {up['commit'][:8]}"
        return f"kept: {dest} already holds protein_mpnn_run.py ({at}{pin})"
    if os.path.isdir(dest) and os.listdir(dest):
        raise WeightsError(f"{dest} is not empty and holds no protein_mpnn_run.py — nothing is cloned over it (name an absent or empty directory, or your ProteinMPNN clone)")
    git = shutil.which("git")
    if not git:
        raise WeightsError(f"git is not on PATH: upstream distributes ProteinMPNN, code and weights, as a git repository ({up['install']})")
    _run([git, "clone", "--quiet", up["repo"], dest], "the clone")
    _run([git, "-C", dest, "checkout", "--quiet", up["commit"]], "the checkout of the pinned commit")
    return f"cloned {up['repo']} into {dest} at {up['commit'][:8]}"


def check_checkout_weights(cp, pins: dict, dest: str) -> Tuple[List[str], List[str]]:
    """(ok, refused): the repository weight sets of PINS.json "weights" under ``dest``, each digested by check_pins.check_weights."""
    ok, bad = [], []
    for key, ent in pins["weights"].items():
        wbad, rec = cp.check_weights(pins, os.path.join(dest, ent["path"]))
        if wbad:
            bad.append(f"{ent['path']}: absent under {dest}")
        elif rec["sha256"] != ent["sha256"]:
            bad.append(f"{ent['path']}: sha256 {rec['sha256'][:12]} is not stock/PINS.json's {ent['sha256'][:12]} (weights.{key}) — left in place")
        else:
            ok.append(ent["path"])
    return ok, bad


# ------------------------------------------------------------------------------------------------ entry
def main(argv: Optional[List[str]] = None, pins: Optional[dict] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m proteinmpnn_opt.weights", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", help="the ProteinMPNN checkout to make or check (your MPNN_DIR; it serves both weight sets)")
    p.add_argument("--variant", default=None, choices=list(VARIANTS), help="which upstream's files (default: the ProteinMPNN checkout, serving soluble and vanilla)")
    a = p.parse_args(argv)
    dest = os.path.abspath(a.dir)
    pins = pins if pins is not None else stack.pins()
    cp = check_pins_module()
    try:
        how = fetch_checkout(cp, pins, dest)
        ok, bad = check_checkout_weights(cp, pins, dest)
        tail = f" — export {ENV_BASE}={dest}"
    except WeightsError as e:
        print(f"WEIGHTS FAILED: {e}", file=sys.stderr)
        return 1
    print(f"weights: {how}", flush=True)
    if bad:
        for line in bad:
            print(f"WEIGHTS REFUSED: {line}", file=sys.stderr)
        return 1
    print(f"WEIGHTS OK: {len(ok)}/{len(ok)} pinned files match stock/PINS.json under {dest}{tail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
