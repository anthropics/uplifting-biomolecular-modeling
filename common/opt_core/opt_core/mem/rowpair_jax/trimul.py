"""Triangle multiplication's cross-row contraction on a row-sharded pair block.

The stock sub-layer (``TriangleMultiplication``) is row-local — layer norms, the (gated)
projections, the output projection and gate act per pair element — EXCEPT its one einsum, whose second index ranges over rows other devices hold.
:func:`contract` is that einsum for a row-sharded block, for the four stock equation strings:

  ================  ============  ===========================================  =========================================================
  equation          direction     operands here (row-sharded input layout)     collectives (schedule ``gather``)
  ================  ============  ===========================================  =========================================================
  ``ikc,jkc->ijc``  outgoing      a, b: ``[N/P, N, C]`` (row dim 0)             all_gather(b) over rows → einsum(a_local, b_full)
  ``cik,cjk->cij``  outgoing      a, b: ``[C, N/P, N]`` (row dim 1)             same, channel-first
  ``kjc,kic->ijc``  incoming      a, b: ``[N/P(k), N, C]`` — the ROW index is k  all_gather(a) → ``[N(k), N(j)]``; all_to_all(b) → the column
                                                                                 block ``[N(k), N/P(i)]``; einsum complete over k
  ``ckj,cki->cij``  incoming      a, b: ``[C, N/P(k), N]``                       same, channel-first
  ================  ============  ===========================================  =========================================================

The result is this device's OUTPUT rows (``[N/P(i), N(j), C]`` / ``[C, N/P(i), N(j)]``). Numerics class ``complete_contraction``: every output
element is the dense-length sum over k of the dense operands — no partial sums across devices — so the value differs from the single-device einsum
only through the accumulation order the GEMM library picks for the local shape (M = N/P instead of N): a stated tolerance, tier-2 by construction,
identical when the library's order happens to coincide (small shapes on CPU typically do). Schedules: ``gather`` holds ONE full ``[N, N, C]``-class
operand per device during the call (the memory floor of the simple schedule); ``ring`` runs P steps, each contracting against one column block
received by ``ppermute`` and writing one column block of the output — transient one block, P collectives, the SAME numerics class (each element's
sum is still complete; the GEMM shape is ``[N/P × N]·[N × N/P]``). The kit picks the schedule (``ring`` where the full operand does not fit) and names
it on the lever line (``schedule=``). Inside a ``shard_map`` region only (the axis name must be bound). Standard library at import.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from .. import MemLeverRefused
from . import LEVER, _lazy
from . import shard as _shard

SCHEDULES = ("gather", "ring")
DEFAULT = "ring"                                   # the schedule: b row-blocks travel once around the ring (bounded transient); "gather" is the named alternative
# equation -> (direction, row_dim, col_dim) of the OPERAND layout (row-sharded input); the output layout equals the operand layout with i at row_dim
EQUATIONS: Dict[str, Tuple[str, int, int]] = {
    "ikc,jkc->ijc": ("outgoing", 0, 1),
    "kjc,kic->ijc": ("incoming", 0, 1),
    "cik,cjk->cij": ("outgoing", 1, 2),
    "ckj,cki->cij": ("incoming", 1, 2),
}
NUMERICS_CLASS = "complete_contraction"


def classify(equation: str, lever: str = LEVER) -> Tuple[str, int, int]:
    """``(direction, row_dim, col_dim)`` for a stock equation string; anything else is refused by name (a sub-layer whose equation the mechanism does
    not cover is never contracted by guess)."""
    eq = str(equation).replace(" ", "")
    if eq not in EQUATIONS:
        raise MemLeverRefused(lever, f"refused: triangle-multiplication equation {equation!r} is not one of {','.join(EQUATIONS)}")
    return EQUATIONS[eq]


def contract(equation: str, a: Any, b: Any, axis: str, schedule: str = DEFAULT, precision: Any = None, lever: str = LEVER):
    """The einsum ``equation`` of the stock sub-layer for ROW-SHARDED operands ``a``, ``b`` (this device's row blocks, the stock's own layout), inside a
    shard_map region over ``axis``; returns this device's output rows. ``precision`` is passed to ``jnp.einsum`` when given (the stock's argument,
    if it has one). See the module table for the collectives per equation and schedule."""
    direction, row_dim, col_dim = classify(equation, lever)
    eq = str(equation).replace(" ", "")
    if schedule not in SCHEDULES:
        raise MemLeverRefused(lever, f"refused: trimul schedule {schedule!r} is not one of {','.join(SCHEDULES)}")
    if a.ndim != 3 or b.ndim != 3 or a.shape != b.shape:
        raise MemLeverRefused(lever, f"refused: trimul operands must be two rank-3 blocks of one shape (got {tuple(a.shape)} and {tuple(b.shape)})")
    jnp = _lazy.jnp(lever)
    kw = {} if precision is None else {"precision": precision}
    if schedule == "gather":
        if direction == "outgoing":                                     # out[i,j] = sum_k a[i,k] b[j,k]: i local; b over ALL rows j
            b_full = _shard.gather(b, axis, dim=row_dim)
            return jnp.einsum(eq, a, b_full, **kw)
        a_full = _shard.gather(a, axis, dim=row_dim)                    # incoming: out[i,j] = sum_k a[k,j] b[k,i]; the row index is k
        b_col = _shard.rows_to_cols(b, axis, row_dim, col_dim)          # b: [k local, i all] → [k all, i local]
        return jnp.einsum(eq, a_full, b_col, **kw)                      # complete over k → out[i local, j all]
    return _ring(eq, direction, row_dim, col_dim, a, b, axis, kw, lever)


def _ring(eq: str, direction: str, row_dim: int, col_dim: int, a: Any, b: Any, axis: str, kw: dict, lever: str):
    """P steps; step t contracts against the column block that started on device (me - t) mod P and writes output column block (me - t) mod P."""
    jax = _lazy.jax(lever)
    jnp = _lazy.jnp(lever)
    p = _shard.axis_size(axis)
    me = jax.lax.axis_index(axis)
    n_loc = int(a.shape[row_dim])
    n_all = int(a.shape[col_dim])
    if n_loc * p != n_all:
        raise MemLeverRefused(lever, f"refused: ring trimul needs square blocks: rows {n_loc} x P {p} != cols {n_all}")
    if direction == "outgoing":
        # out[i, j in blk s] = sum_k a[i,k] b[j in s, k]: the moving operand is b's ROW block (all k); output columns of block s
        mover = b                                                       # [.. N/P rows(j) .., N cols(k) ..] — my rows of b to start
        fixed = a
        def partial(mv):                                                # → out block [i local, j block] in the operand layout
            return jnp.einsum(eq, fixed, mv, **kw)
    else:
        # out[i local, j in blk s] = sum_k a[k, j in s] b[k, i local]: both operands as COLUMN blocks over all k; a's column block moves
        mover = _shard.rows_to_cols(a, axis, row_dim, col_dim)          # [N(k), N/P(j mine)]
        fixed = _shard.rows_to_cols(b, axis, row_dim, col_dim)          # [N(k), N/P(i mine)]
        def partial(mv):
            return jnp.einsum(eq, mv, fixed, **kw)                      # → [i local, j block]
    out = None
    for t in range(p):
        blk = partial(mover)                                            # this step's column block of my output rows
        src = (me - t) % p                                              # which device's block I hold at step t
        if out is None:
            shape = list(blk.shape)
            shape[col_dim] = n_all
            out = jnp.zeros(tuple(shape), blk.dtype)
        out = jax.lax.dynamic_update_slice_in_dim(out, blk, src * n_loc, axis=col_dim)
        if t + 1 < p:
            mover = _shard.ppermute_shift(mover, axis, shift=1)         # pass my current block to me+1; receive from me-1
    return out


def transient_bytes(n_rows: int, channels: int, itemsize: int, n_gpu: int, schedule: str = DEFAULT, direction: str = "outgoing") -> Dict[str, int]:
    """The per-device transient of one :func:`contract` call beyond its inputs and output (block = ``N*N*C*itemsize / P``): ``gather`` outgoing = one full
    operand; ``gather`` incoming = one full operand + one column block; ``ring`` outgoing = the moving block's double buffer (2) + the step's partial (1) =
    3 blocks; ``ring`` incoming = two column re-layouts (2) + the moving block's double buffer (1 more) + the step's partial (1) = 4 blocks. Arithmetic for
    the kit's schedule choice and its memory table."""
    n, c, b, p = int(n_rows), int(channels), int(itemsize), int(n_gpu)
    full = n * n * c * b
    block = full // max(p, 1)
    if direction not in ("outgoing", "incoming"):
        raise ValueError(f"transient_bytes: direction {direction!r} is not outgoing|incoming")
    if schedule == "gather":
        extra = block if direction == "incoming" else 0
        return {"full_operand": full, "column_block": extra, "total": full + extra}
    if schedule == "ring":
        k = 3 if direction == "outgoing" else 4
        return {"blocks": k, "block_bytes": block, "total": k * block}
    raise ValueError(f"transient_bytes: schedule {schedule!r} is not one of {','.join(SCHEDULES)}")
