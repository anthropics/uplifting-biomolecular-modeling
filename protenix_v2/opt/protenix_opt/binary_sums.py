"""binary_sums.py — the kit's shipped kernel binaries are held to a SHA256SUMS line before any loader maps them.

The prebuilt binaries this kit ships under ``$FPF_HOME/third_party/`` (the stream-correct fast-LayerNorm extension of
``fastln_prebuilt*/`` and the sm_90a tri-attention prologue library of ``protenix_fpf_triatt_procuda/prebuilt/``) each sit beside a
``SHA256SUMS`` in the shared core's format (``<sha256>  <name>``; written by the core's ``tools/binary_sums.py --write --package <dir>``).
:func:`binary_refusal` is the shared core's rule (:func:`opt_core.gates.binary_refusal`) applied to a kit binary: the file must be a line
of the SHA256SUMS in its own directory and its bytes must re-hash to that line. A binary that is not listed, whose bytes differ, that
cannot be read, or that has no SHA256SUMS beside it is refused BY NAME — one ``[opt_core] SHA256SUMS: refused <path>: <reason>`` line on
stderr, the core's line — and the caller steps aside exactly as it does for an absent binary (the fast-LayerNorm falls back to its
per-process source rebuild; the prologue lever reports itself unavailable). One verdict per path per process; a binary that passed is
recorded in :data:`opt_core.gates.BINARIES_HELD`.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

from opt_core import gates as _gates

SUMS = _gates.BINARY_SUMS                     # "SHA256SUMS"
_VERDICTS: dict = {}


def binary_refusal(path: str) -> Optional[str]:
    """``None`` when the kit binary at ``path`` is a line of the SHA256SUMS beside it and re-hashes to it; otherwise the reason (one
    clause), after one refusal line on stderr in the core's format. Unlike a file outside any package, a kit binary with no SHA256SUMS
    in its directory is refused too: every binary this kit ships has one."""
    p = os.path.abspath(path)
    if p in _VERDICTS:
        return _VERDICTS[p]
    if os.path.isfile(os.path.join(os.path.dirname(p), SUMS)):
        why = _gates.binary_refusal(p)          # listed and equal -> None (held); unlisted / altered / unreadable -> the reason, printed by name by the core
    else:
        why = f"no {SUMS} in its directory"
        print(f"[opt_core] {SUMS}: refused {p}: {why}", file=sys.stderr, flush=True)
    _VERDICTS[p] = why
    return why


def manifest_binary_refusal(pre_dir: str, key: str = "so") -> Optional[str]:
    """The verdict of :func:`binary_refusal` for the binary a prebuilt directory's ``manifest.json`` names under ``key`` (the file its
    loader is about to map). A directory with no readable manifest, no such entry, or no such file is left to the loader, which refuses
    an absent binary by name itself: ``None``."""
    try:
        with open(os.path.join(pre_dir, "manifest.json"), encoding="utf-8") as fh:
            name = json.load(fh).get(key)
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(name, str) or not name:
        return None
    so = os.path.join(pre_dir, name)
    if not os.path.isfile(so):
        return None
    return binary_refusal(so)
