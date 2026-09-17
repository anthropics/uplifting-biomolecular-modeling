"""fpf_trimul_rows — the ROW-BLOCK members of the triangle multiplicative update for pair widths beside fpf_trimul_v4's (c_z / c_hidden 64 and 384;
128 / 256 compile too), reachable ONLY through the sharded row-block provider `opt_core.mem.rowpair.trimul_fused` (P > 1).  Nothing here is a
whole-plane TriangleMultiplication provider: no entry of `opt_core.kernels.trimul` (TRIMUL_CELLS.json) and no cell of `fpf_trimul_v4/table.json`
reads this package or its table, so every single-GPU line resolves exactly what it resolved without it.

  kernels.py        `_k1r` (LN_in + one gated projection + mask -> channel-major bf16 planes, a- or b-layout through strides, channel-last z read in
                    place) and `_k3r` (the tile epilogue on a column window of the shard in place); torch + triton at module level — imported by the
                    provider at its first served call, never at this package's import.
  table_rows.json   the ROWS-ONLY tile table (:data:`TABLE_PATH`), keyed ``"<cc M.m>|C<c_z>|H<c_hidden>"``: per key the K1 (a-layout), K1 (b-layout)
                    and K3 launch cells, whether the b operand is written in its GEMM layout directly (``b_direct``), the size floor ``min_tokens``
                    of the row-block lever at that (capability, width), ``measured`` (bool) + ``status`` / ``evidence`` words, and per capability
                    the SAFE cells (``safe``).  A key whose ``kernels`` word is ``fpf_trimul_v4`` carries NO tiles: those widths keep the
                    whole-plane unit's own table.json row on the row blocks (today's path) and the entry only sets their floor.
This module is standard library at import (the provider imports it lazily anyway); :func:`select` is the ONE selection statement.

Admission (the provider's gate for a width outside fpf_trimul_v4's): a key that EXISTS and is ``measured`` serves; an existing UNMEASURED key
(``status`` ``PLACEHOLDER …``: tiles seeded from a neighbouring sweep, not yet measured at this unit's shapes) serves only under the engineering
switch :data:`ENV_UNMEASURED` ``=1`` (the tile sweep / an anchor before certification); no key = the provider's decline word ``c=<c_z>/<c_h>``
(the torch statements, counted) — the word every such width answered before this package existed.
"""
import json
import os
import re
from typing import Any, Dict, Mapping, NamedTuple, Optional, Tuple

__version__ = "0.1.0"

TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "table_rows.json")
SCHEMA = "fpf_trimul_rows/v1"
KERNELS_WORD = "fpf_trimul_rows"           # a key served by this package's kernels (its tiles live in the key)
V4_WORD = "fpf_trimul_v4"                  # a key whose row blocks stay on the whole-plane unit's kernels + table.json row (the entry sets the floor only)
ENV_UNMEASURED = "ROWPAIR_TRIMUL_ROWS_UNMEASURED"     # engineering switch: '1' admits PLACEHOLDER (unmeasured) keys — the sweep / an anchor before certification
SAFE_LEVER = "pair_fused:trimul_rows"      # the safety-net lever word of these kernels (their SAFE cells are the table's `safe` member, not opt_core.kernels.safe_settings)
WIDTHS = (64, 128, 256, 384)               # the (c_z | c_hidden) values kernels.py is written and tested for (chunking() also walks 512 = 4 x 128; untested, not admitted)
MAX_CHUNKS = 4                             # resident channel chunks per width (kernels.py _ln_rows / _ln_cols / the chunked dot-accumulate are written for <= 4)
CELL_KEYS = ("BM", "BN", "num_warps", "num_stages")
KEY_RE = re.compile(r"^(?P<cc>\d+\.\d+)\|C(?P<cz>\d+)\|H(?P<ch>\d+)$")
PLACEHOLDER_PREFIX = "PLACEHOLDER"

_CACHE: Dict[str, Tuple[float, dict]] = {}


class RowsUnsupported(ValueError):
    """A width / shape / layout / dtype the row-block kernels do not serve — raised BEFORE any launch, by name (the provider declines that unit, counted)."""


def chunking(width: int) -> Tuple[int, int]:
    """``(CK, NCK)``: the power-of-two channel chunk and chunk count kernels.py walks a width in — ONE chunk when the width is a power of two <= 256
    (64 / 128 / 256: the fpf_trimul_v4 statements), else 128-chunks (384 = 3 x 128), else 64-chunks; RowsUnsupported for anything else or > 4 chunks."""
    width = int(width)
    if width <= 0:
        raise RowsUnsupported("width:%d" % width)
    if width <= 256 and (width & (width - 1)) == 0:
        ck = width
    elif width % 128 == 0:
        ck = 128
    elif width % 64 == 0:
        ck = 64
    else:
        raise RowsUnsupported("width:%d(not a multiple of 64)" % width)
    n = width // ck
    if n > MAX_CHUNKS or ck < 16:
        raise RowsUnsupported("width:%d(chunks %dx%d outside 1..%d x >=16)" % (width, n, ck, MAX_CHUNKS))
    return ck, n


class Selection(NamedTuple):
    """The answer of :func:`select`: ``served`` (bool), ``reason`` (``served`` | ``served:unmeasured`` | ``unmeasured`` | ``no_row`` | ``v4`` |
    ``width``), ``key`` (the table key or None), ``cell`` (the entry dict or None)."""
    served: bool
    reason: str
    key: Optional[str]
    cell: Optional[Mapping[str, Any]]


def cell_key(cc, C_z: int, C_h: int) -> str:
    """``"<cc>|C<c_z>|H<c_h>"`` for a ``"M.m"`` string or a ``(major, minor)`` tuple."""
    if not isinstance(cc, str):
        cc = "%d.%d" % (int(cc[0]), int(cc[1]))
    return "%s|C%d|H%d" % (cc, int(C_z), int(C_h))


def load_table(path: Optional[str] = None) -> dict:
    """The rows table (parsed json; cached per path + mtime).  A missing / unreadable file is an EMPTY table (every width outside fpf_trimul_v4's
    declines by name; the v4 widths keep their default floor) — never an exception at a call site."""
    p = os.path.abspath(path or os.environ.get("FPF_TRIMUL_ROWS_TABLE") or TABLE_PATH)
    try:
        mt = os.path.getmtime(p)
    except OSError:
        return {"schema": SCHEMA, "cells": {}, "safe": {}, "_path": p, "_missing": True}
    hit = _CACHE.get(p)
    if hit is not None and hit[0] == mt:
        return hit[1]
    with open(p, "r", encoding="utf-8") as fh:
        t = json.load(fh)
    t = dict(t)
    t.setdefault("cells", {})
    t.setdefault("safe", {})
    t["_path"] = p
    _CACHE[p] = (mt, t)
    return t


def entry(table: Mapping, cc, C_z: int, C_h: int) -> Optional[Mapping[str, Any]]:
    """The table entry for (cc, c_z, c_h) or None."""
    e = (table.get("cells") or {}).get(cell_key(cc, C_z, C_h))
    return e if isinstance(e, Mapping) else None


def is_measured(e: Optional[Mapping]) -> bool:
    return bool(e) and bool(e.get("measured")) and not str(e.get("status", "")).upper().startswith(PLACEHOLDER_PREFIX)


def allow_unmeasured_env() -> bool:
    return os.environ.get(ENV_UNMEASURED, "").strip().lower() in ("1", "true", "yes", "on")


def select(table: Mapping, cc, C_z: int, C_h: int, *, allow_unmeasured: Optional[bool] = None) -> Selection:
    """The ONE selection statement of the row-block members for a width OUTSIDE fpf_trimul_v4's: served iff the key exists, names this package's
    kernels, carries k1 / k3 cells, both widths are in :data:`WIDTHS`, and is measured (or ``allow_unmeasured`` — default: :data:`ENV_UNMEASURED`).
    Pure function of the table + arguments (+ that one environment variable): identical on every rank."""
    if int(C_z) not in WIDTHS or int(C_h) not in WIDTHS:
        return Selection(False, "width", None, None)
    key = cell_key(cc, C_z, C_h)
    e = entry(table, cc, C_z, C_h)
    if e is None:
        return Selection(False, "no_row", None, None)
    if str(e.get("kernels", KERNELS_WORD)) != KERNELS_WORD:
        return Selection(False, "v4", key, e)
    if not (isinstance(e.get("k1"), Mapping) and isinstance(e.get("k3"), Mapping)):
        return Selection(False, "no_row", key, e)
    if is_measured(e):
        return Selection(True, "served", key, e)
    allow = allow_unmeasured_env() if allow_unmeasured is None else bool(allow_unmeasured)
    if allow:
        return Selection(True, "served:unmeasured", key, e)
    return Selection(False, "unmeasured", key, e)


def tiles(e: Mapping[str, Any]) -> Tuple[dict, dict, dict]:
    """``(k1 a-layout cell, k1 b-layout cell, k3 cell)`` of an entry (``k1b`` defaults to ``k1``); each ``{BM, BN, num_warps, num_stages}`` ints."""
    def norm(c):
        return {k: int(c[k]) for k in CELL_KEYS}
    k1 = norm(e["k1"])
    k1b = norm(e["k1b"]) if isinstance(e.get("k1b"), Mapping) else dict(k1)
    return k1, k1b, norm(e["k3"])


def b_direct(e: Optional[Mapping]) -> bool:
    """Does this entry write the b operand in its GEMM layout directly (K1 b-layout; no transpose copy)?  Default False."""
    return bool(e) and bool(e.get("b_direct", False))


def min_tokens(table: Mapping, cc, C_z: int, C_h: int, default: int) -> int:
    """The size floor of the row-block lever at (cc, c_z, c_h): the entry's ``min_tokens`` when it names one (v4-width entries included), else ``default``."""
    e = entry(table, cc, C_z, C_h)
    if e is not None and e.get("min_tokens") is not None:
        return int(e["min_tokens"])
    return int(default)


def safe_cells(table: Mapping, cc) -> Optional[Mapping[str, Any]]:
    """The SAFE ``{"k1": …, "k3": …}`` cells of these kernels for capability ``cc`` (exact ``"<cc>"`` entry, else ``"*"``), or None (no safe cells:
    a build failure of the tuned cell is the lever's refusal)."""
    safe = table.get("safe") or {}
    w = cc if isinstance(cc, str) else "%d.%d" % (int(cc[0]), int(cc[1]))
    for k in (w, "*"):
        row = safe.get(k)
        if isinstance(row, Mapping) and isinstance(row.get("k1"), Mapping) and isinstance(row.get("k3"), Mapping):
            return row
    return None


def keys(table: Mapping):
    """Every cell key of the table (sorted)."""
    return sorted(k for k in (table.get("cells") or {}) if not str(k).startswith("_"))


def validate(table: Mapping) -> list:
    """Problems of a rows table as strings (empty = well-formed): key grammar, kernels word, integer power-of-two tiles, measured/status agreement."""
    out = []
    if table.get("schema") != SCHEMA:
        out.append("schema %r != %r" % (table.get("schema"), SCHEMA))
    for k in keys(table):
        e = table["cells"][k]
        m = KEY_RE.match(str(k))
        if not m:
            out.append("%s: key grammar (<cc>|C<c_z>|H<c_h>)" % k); continue
        word = str(e.get("kernels", KERNELS_WORD))
        if word not in (KERNELS_WORD, V4_WORD):
            out.append("%s: kernels word %r" % (k, word))
        if word == KERNELS_WORD:
            for name in ("k1", "k3") + (("k1b",) if "k1b" in e else ()):
                c = e.get(name)
                if not isinstance(c, Mapping) or any(not isinstance(c.get(x), int) for x in CELL_KEYS):
                    out.append("%s: %s must carry integer %s" % (k, name, "/".join(CELL_KEYS))); continue
                for x in ("BM", "BN"):
                    v = int(c[x])
                    if v < 16 or v & (v - 1):
                        out.append("%s: %s.%s=%d (power of two >= 16)" % (k, name, x, v))
            if int(m.group("cz")) not in WIDTHS or int(m.group("ch")) not in WIDTHS:
                out.append("%s: widths outside %s" % (k, WIDTHS))
        if not isinstance(e.get("measured"), bool):
            out.append("%s: measured must be a bool" % k)
        elif e["measured"] == str(e.get("status", "")).upper().startswith(PLACEHOLDER_PREFIX):
            out.append("%s: measured=%s contradicts status %r" % (k, e["measured"], e.get("status")))
        if e.get("min_tokens") is not None and (not isinstance(e["min_tokens"], int) or e["min_tokens"] < 0):
            out.append("%s: min_tokens must be a non-negative int" % k)
    for ccw, row in (table.get("safe") or {}).items():
        if str(ccw).startswith("_"):
            continue
        for name in ("k1", "k3"):
            c = row.get(name) if isinstance(row, Mapping) else None
            if not isinstance(c, Mapping) or any(not isinstance(c.get(x), int) for x in CELL_KEYS):
                out.append("safe.%s: %s must carry integer %s" % (ccw, name, "/".join(CELL_KEYS)))
    return out


def describe(table: Optional[Mapping] = None) -> dict:
    """Plain data for a manifest: version, table path, keys with their measured / kernels / min_tokens words."""
    t = table if table is not None else load_table()
    return {"unit": KERNELS_WORD, "version": __version__, "table": t.get("_path"), "schema": t.get("schema"),
            "keys": {k: {"kernels": t["cells"][k].get("kernels", KERNELS_WORD), "measured": bool(t["cells"][k].get("measured")),
                         "min_tokens": t["cells"][k].get("min_tokens"), "b_direct": bool(t["cells"][k].get("b_direct", False))} for k in keys(t)}}
