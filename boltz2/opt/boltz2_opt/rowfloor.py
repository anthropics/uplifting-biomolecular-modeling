"""boltz2_opt.rowfloor — the per-card ROW RULE the bitwise (exact-class) variants of the fused pair-track levers serve under
(``boltz2_opt.transition``, ``boltz2_opt.pairblock``): the one rule both adapters apply; each keeps its own table
``CARD_ROWS = {compute capability: {variant: (min_rows, max_rows, piece_rows)}}`` — a call is served when ``min_rows <= rows < max_rows`` and,
for a call of ``piece_rows`` rows or more, when its TRAILING PIECE ``rows mod piece_rows`` has ``min_rows`` rows or more too.

A fused cell computes its projections in ONE fixed summation order, the order the stock ``Linear`` layers' GEMMs (cuBLAS) use across the row
counts the models run at, so its outputs equal stock's bit for bit there. On a card whose cuBLAS switches to another summation order for one
of the projections' shapes at some row counts (compute capability 8.0 under the pinned torch does — in bands below a few thousand rows; 9.0
does not, at any row count), the bitwise variants cannot equal stock at those counts whatever the cell's launch configuration: there the lever
hands the call to the module's original forward BY NAME, decided from the call's shape before the core is asked, counted on the lever's census:
``below_min_rows`` (a call under the floor), ``above_max_rows`` (a call at or over the measured top), ``trailing_piece`` (see next) — the
rule's numbers printed as ``min_rows=<n> max_rows=<n> piece_rows=<n>``. The same card works a call of ``piece_rows`` rows or more (8.0:
2**20 - 32 = 1,048,544) IN PIECES of that many rows and a trailing piece of ``rows mod piece_rows`` rows, each piece summed as a call of its own
size would be: a trailing piece under the floor meets the low-row bands again (an input of 1,025 tokens: 1,050,625 = 1,048,544 + 2,081 rows —
not bitwise, measured), one over it does not (1,200 tokens: trailing 391,456; 1,400: 911,456 — bitwise, measured; the adapters' tables name every
measured point); so a pieced call is served exactly when its trailing piece clears the floor, else counted ``trailing_piece``,
and the pieced calls' arithmetic rides on the line as ``pieces=<n> trailing=<rows>`` (the last such call's). Rows = the flattened call's leading
dimensions, what the GEMMs see: a pair tensor [B, N, N, C] is B·N·N rows (an input of N tokens runs its Pairformer pair calls at N·N rows).
A card or variant absent from a table has no rule: every supported call is served there and nothing is printed. The Tier-2 variants have none
anywhere: a summation order is inside their tolerance.
"""
from typing import Dict, Optional, Tuple

BELOW_MIN_ROWS = "below_min_rows"     # the census word of a call under the floor (a DECLARED stock path: each adapter lists it in its EXPECTED)
ABOVE_MAX_ROWS = "above_max_rows"     # ... of a call at or over the measured top
TRAILING_PIECE = "trailing_piece"     # ... and of a pieced call whose trailing piece is under the floor
WORDS = (BELOW_MIN_ROWS, ABOVE_MAX_ROWS, TRAILING_PIECE)

PIECE_ROWS_SM80 = 2 ** 20 - 32        # 1,048,544: from this many rows cuBLAS on compute capability 8.0 (the pinned torch) works a Linear's GEMM in pieces of it

Range = Tuple[int, int, int]          # (min_rows, max_rows, piece_rows); piece_rows 0 = the card never pieces a call


def card_cc() -> Optional[str]:
    """'M.m' of this (worker) process's CUDA device — the kit's one probe (msa_kernels.device_cc); None without torch/CUDA (the CPU tests)."""
    from .msa_kernels import device_cc
    return device_cc()


def range_for(table: Dict[str, Dict[str, Range]], variant: Optional[str], cc: Optional[str]) -> Optional[Range]:
    """The (min_rows, max_rows, piece_rows) rule of `variant` on compute capability `cc` in an adapter's CARD_ROWS `table`, else None = no rule."""
    r = (table.get(cc or "") or {}).get(variant or "")
    return (int(r[0]), int(r[1]), int(r[2])) if r is not None else None


def rows_of(x) -> int:
    """The flattened call's row count: the product of x's leading dimensions ([B, N, N, C] -> B·N·N)."""
    c = int(x.shape[-1])
    return int(x.numel()) // c if c else 0


def pieces(rows: int, rng: Optional[Range]) -> Tuple[int, int]:
    """(pieces, trailing) — how the card works a call of `rows` rows under `rng`: (1, rows) for a call it does not piece (no rule, no piece
    size, or fewer rows than one piece); else the number of pieces and the trailing piece's rows, ``rows mod piece_rows`` (0 = whole pieces only)."""
    p = int(rng[2]) if rng is not None else 0
    if not p or rows < p:
        return 1, int(rows)
    k, r = divmod(int(rows), p)
    return k + (1 if r else 0), r


def outside(rows: int, rng: Optional[Range]) -> Optional[str]:
    """The census word when `rows` is not served under `rng` (below_min_rows | above_max_rows | trailing_piece), None when served or no rule.
    A pieced call whose trailing piece is 0 rows (whole pieces only) is not served either: no model call lands there and none was measured."""
    if rng is None:
        return None
    if rows < rng[0]:
        return BELOW_MIN_ROWS
    if rows >= rng[1]:
        return ABOVE_MAX_ROWS
    if rng[2] and rows >= rng[2] and rows % rng[2] < rng[0]:
        return TRAILING_PIECE
    return None


def call_facts(rows: int, rng: Optional[Range]) -> Dict[str, int]:
    """The pieced call's arithmetic as the lever line's facts ({"pieces": n, "trailing": r}); {} for a call the card does not piece."""
    if rng is None or not rng[2] or rows < rng[2]:
        return {}
    n, trailing = pieces(rows, rng)
    return {"pieces": n, "trailing": trailing}


def facts(rng: Optional[Range]) -> Dict[str, int]:
    """The rule as the lever line's / report's facts ({} when no rule: nothing printed)."""
    return {} if rng is None else {"min_rows": rng[0], "max_rows": rng[1], "piece_rows": rng[2]}
