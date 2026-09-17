"""The sharding floor of the tp line — ONE predicate, evaluated twice: by the launcher on the query's polymer residue count (a lower bound of
the token count, before any rank exists) and by every rank on the featurised token count (``model.transfer_batch_to_device_patched``, before
the rank joins the group or issues any collective; every rank featurises the same query, so the verdict is identical on all of them).

Rule: the run's row partition (``opt_core.mem.rowpair.dist.row_parts(N, P, align)``, ``align`` = the pinned chunk = the layout's row-block
alignment) must give EVERY rank at least one full block of ``BLOCK_UNIT`` rows (16: the finest row-block unit the chunk plan reduces to). The
launcher's chunk plan halves its chunk (down to ``BLOCK_UNIT``) until the partition passes (``tp.chunk_of_run``); a query whose partition still
leaves some rank with fewer rows is not sharded: the launcher serves it unsharded on one card through the stock route (``tp.serve_unsharded``,
the named fallback ``small_n_unsharded``); a rank that finds it on the featurised count exits ``EXIT_BELOW_FLOOR`` before any collective and
the launcher answers that exit the same way.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

BLOCK_UNIT = 16                                              # rows: the finest row-block unit of the layout (tp.CHUNK_FLOOR, the smallest chunk the plan reduces to)
EXIT_BELOW_FLOOR = 70                                        # a rank's exit code for "the featurised query is below the sharding floor" (the launcher serves the query unsharded)
REASON = "small_n_unsharded"                                 # the named fallback (census word `fallbacks=tp:small_n_unsharded=1`)


def row_parts(N: int, world: int, align: int) -> List[Tuple[int, int]]:
    """The run's row partition — the core's own statement (``dist.row_parts``: chunk-aligned, chunks dealt floor/ceil per rank, the last chunk ragged)."""
    from opt_core.mem.rowpair.dist import row_parts as _row_parts
    return _row_parts(int(N), int(world), int(align))


def verdict(N: int, world: int, align: int) -> Dict[str, object]:
    """{"ok", "N", "world", "align", "parts", "rows", "min_rank", "min_rows", "block"}: ``ok`` iff every rank of the partition owns at least
    ``BLOCK_UNIT`` rows (one full block of the finest unit). ``N`` = tokens (or polymer residues, the launcher's lower bound)."""
    N, world, align = int(N), int(world), int(align)
    parts = row_parts(N, world, align) if N > 0 else [(0, 0)] * world
    rows = [e - s for s, e in parts]
    j = min(range(world), key=lambda r: rows[r])
    return {"ok": bool(N > 0 and rows[j] >= BLOCK_UNIT), "N": N, "world": world, "align": align, "parts": parts, "rows": rows,
            "min_rank": j, "min_rows": rows[j], "block": BLOCK_UNIT}


def words(v: Dict[str, object], unit: str = "tokens") -> str:
    """The ONE sentence both tiers print for a query below the floor."""
    return (f"{v['N']} {unit} at P={v['world']} in row blocks aligned to the chunk {v['align']} give rank rows {v['rows']}: rank {v['min_rank']} would own "
            f"{v['min_rows']} rows, fewer than one full block of {v['block']} (the sharding floor: every rank owns at least {v['block']} rows)")


class BelowFloor(SystemExit):
    """Raised in a rank whose featurised query is below the floor: exits the rank with ``EXIT_BELOW_FLOOR`` before any collective."""

    def __init__(self, v: Dict[str, object]):
        super().__init__(EXIT_BELOW_FLOOR)
        self.verdict = v


def check_featurised(batch, world: int, align: int, log=None) -> Dict[str, object]:
    """The rank tier: the verdict on the featurised token count of ``batch`` (``token_mask``'s last dimension). Below the floor: one named line
    through ``log`` and :class:`BelowFloor` (SystemExit ``EXIT_BELOW_FLOOR``) — the caller has not joined the group yet, so no peer waits on it."""
    tm = batch["token_mask"] if isinstance(batch, dict) else batch.get("token_mask")
    v = verdict(int(tm.shape[-1]), world, align)
    if not v["ok"]:
        if log is not None:
            log(f"below the sharding floor: the featurised query's {words(v)}; this rank exits {EXIT_BELOW_FLOOR} before any collective — "
                f"the launcher serves the query unsharded ({REASON})")
        raise BelowFloor(v)
    return v
