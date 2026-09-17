"""The installed tree: the digest over it that stock/PINS.json pins, and the package's scratch directory.

The installed ``caliby`` / ``chroma`` / ``protpardelle`` trees hold the pinned upstream bytes in every mode: a kit mode loads the kits'
files by import hook (overlay.py) and writes nothing into site-packages. This module is the read side of that fact:

    digest()                   ``stack.tree_digest()``: sha256 over the three installed trees (nothing upstream imported);
    assert_upstream_digest()   the digest against ``stock/PINS.json`` "tree_digest_upstream" — ``NOT STOCK`` names the way back:
                               reinstall the pinned upstream packages (stock/PINS.json "upstream".<package>.install, STOCK.md).

``scratch_dir`` is the package's scratch directory under ``MODEL_OPT_STATE`` (warm's outputs) — nothing is ever written inside the
release tree.
"""
from __future__ import annotations

import os
import time
from typing import Optional, Tuple

from . import stack

REINSTALL = 'reinstall the pinned upstream packages (stock/PINS.json "upstream".<package>.install, STOCK.md) to get the pinned tree back'


class TreeError(RuntimeError):
    """The installed tree is not what it must be, and the message names the way back."""


def scratch_dir(tag: str) -> str:
    d = os.path.join(stack.state_dir(), "scratch", f"{tag}-{os.getpid()}-{int(time.time())}")
    os.makedirs(d, exist_ok=True)
    return d


def digest() -> Tuple[str, dict]:
    return stack.tree_digest()


def assert_upstream_digest(pins: Optional[dict] = None) -> Tuple[str, str]:
    """(digest, pinned) — raises TreeError when the installed tree digest is not stock/PINS.json "tree_digest_upstream"."""
    p = pins or stack.pins()
    want = p["tree_digest_upstream"]["sha256"]
    have, _ = digest()
    if have != want:
        raise TreeError(f"NOT STOCK: installed tree digest {have[:16]} != pinned upstream {want[:16]} (state {stack.tree_state()!r}); {REINSTALL}")
    return have, want
