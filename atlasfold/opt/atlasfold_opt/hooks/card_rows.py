"""The per-card ROW RULE the exact-class construction of the fused triangle-attention surround serves under (hooks/triatt_block_exact.py) —
the rule measured for these core kernels at this geometry (c_z = 128 / 4 x 32): the lever keeps its own table
``CARD_ROWS = {(major, minor): (min_rows, max_rows, piece_rows)}``; a call of
``rows`` = B*N*N flattened rows is served when ``min_rows <= rows < max_rows`` and, for a call of ``piece_rows`` rows or more (compute capability 8.0
works a Linear's GEMM in pieces of 2**20 - 32 rows under the pinned torch), when its TRAILING PIECE ``rows mod piece_rows`` has ``min_rows`` rows or
more too; otherwise the stock statement serves the call BY NAME, counted ``below_min_rows`` / ``above_max_rows`` / ``trailing_piece``.  Why: a fused
cell sums its projections in ONE fixed order — cuBLAS's order at the models' row counts; on a card whose cuBLAS switches order in low-row bands
(8.0 does, 9.0 does not at any count) the construction cannot equal stock there whatever the launch configuration.  A card absent from a table has
no rule (nothing printed).  Standard library only."""
from typing import Dict, Optional, Tuple

BELOW_MIN_ROWS = "below_min_rows"
ABOVE_MAX_ROWS = "above_max_rows"
TRAILING_PIECE = "trailing_piece"
WORDS = (BELOW_MIN_ROWS, ABOVE_MAX_ROWS, TRAILING_PIECE)
PIECE_ROWS_SM80 = 2 ** 20 - 32        # 1,048,544

Range = Tuple[int, int, int]          # (min_rows, max_rows, piece_rows); piece_rows 0 = the card never pieces a call


def range_for(table: Dict[tuple, Range], cc) -> Optional[Range]:
    r = table.get(tuple(cc))
    return (int(r[0]), int(r[1]), int(r[2])) if r is not None else None


def rows_of(x) -> int:
    """The flattened call's row count: the product of x's leading dimensions ([B, N, N, C] -> B*N*N)."""
    c = int(x.shape[-1])
    return int(x.numel()) // c if c else 0


def outside(rows: int, rng: Optional[Range]) -> Optional[str]:
    """The census word when `rows` is not served under `rng`, None when served or no rule.  A pieced call whose trailing piece is 0 rows (whole
    pieces only) is not served either: no model call lands there and none was measured."""
    if rng is None:
        return None
    if rows < rng[0]:
        return BELOW_MIN_ROWS
    if rows >= rng[1]:
        return ABOVE_MAX_ROWS
    if rng[2] and rows >= rng[2] and rows % rng[2] < rng[0]:
        return TRAILING_PIECE
    return None


def facts(rng: Optional[Range]) -> Dict[str, int]:
    """The rule as LEVER-line facts ({} when no rule: nothing printed)."""
    return {} if rng is None else {"min_rows": rng[0], "max_rows": rng[1], "piece_rows": rng[2]}
