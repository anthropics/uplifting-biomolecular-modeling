"""Triangle attention on a row-sharded pair block: the plumbing around the kit's attention kernel.

Starting node (``TriangleAttention`` per-row / ``GridSelfAttention(transpose=False)``): for pair row ``i`` the queries,
keys and values all come from row ``i`` — ROW-LOCAL — and the only cross-row input is the non-batched bias ``b[h, j, k] = Linear(LN(x[j, k]))``
over ALL rows: each device projects its own rows and :func:`bias_full` all_gathers them (``[N/P, N, H]`` → ``[N, N, H]``; the stock's transpose to
``[H, N, N]`` stays the stock's line). Ending node (``per_column`` / ``transpose=True``): the stock computes the same attention on ``x^T``; the row
block of ``x^T`` is this device's COLUMN block of ``x`` — :func:`enter_transposed` (one all_to_all + swapaxes) yields it as ``[N/P, N, C]`` local rows
of the transposed tensor, the row attention runs unchanged, :func:`exit_transposed` undoes it. The bias of the ending node is projected from the
transposed local block and gathered the same way. Masks: :func:`mask_rows` / :func:`mask_cols` cut the replicated ``[N, N]`` pair mask to the
block (a symmetric mask makes them transposes of each other; the functions do not assume it). Numerics: ``moves_bytes`` for every function here;
the attention arithmetic is the kit's kernel on the same rows (class ``row_local``) — its row-chunk loop (``subbatch``) runs on the local rows.

Per-shard kernels. A fused kernel (F1's Pallas attention, tokamax) called inside the region sees LOCAL shapes: :func:`kernel_gate` decides from the
kernel's declared block constraints whether it can run on the local extent and returns ``(ok, reason)`` — the adapter serves the kernel when
``ok`` and otherwise the stock jnp path, printing ``kernel=<name|jnp> kernel_reason=<reason>`` on the lever line (a NAMED state, never a silent
switch). Standard library at import.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from . import LEVER, _lazy
from . import shard as _shard

NUMERICS_CLASS = "moves_bytes"


def bias_full(bias_local: Any, axis: str, row_dim: int = 0):
    """The non-batched attention bias over ALL pair rows from this device's rows: all_gather along ``row_dim`` (``[N/P, N, H]`` → ``[N, N, H]``, or
    ``[H, N/P, N]`` → ``[H, N, N]`` with ``row_dim=1``)."""
    return _shard.gather(bias_local, axis, dim=row_dim)


def enter_transposed(x_rows: Any, axis: str, row_dim: int = 0, col_dim: int = 1):
    """This device's row block of ``x^T`` (``[N/P, N, …]``) from its row block of ``x``: the ending-node attention then runs as a starting-node
    attention on the result. One all_to_all (:func:`shard.transpose_block`)."""
    return _shard.transpose_block(x_rows, axis, row_dim, col_dim)


def exit_transposed(y_rows_t: Any, axis: str, row_dim: int = 0, col_dim: int = 1):
    """Undo :func:`enter_transposed` on the attention output: the row block of ``y`` from the row block of ``y^T`` (the same operation — a block
    transpose is an involution across the mesh)."""
    return _shard.transpose_block(y_rows_t, axis, row_dim, col_dim)


def mask_rows(pair_mask_full: Any, axis: str, n_loc: int):
    """My rows of the replicated ``[N, N]`` pair mask (``[N/P, N]``)."""
    return _shard.local_block(pair_mask_full, axis, n_loc, dim=0)


def mask_cols(pair_mask_full: Any, axis: str, n_loc: int):
    """My columns of the replicated ``[N, N]`` pair mask, transposed to ``[N/P, N]`` — the row mask of ``x^T`` for the ending node."""
    jnp = _lazy.jnp()
    return jnp.swapaxes(_shard.local_block(pair_mask_full, axis, n_loc, dim=1), 0, 1)


def kernel_gate(n_loc: int, n_cols: int, constraints: Optional[Mapping[str, Any]] = None) -> Tuple[bool, str]:
    """Whether a fused attention kernel can serve the LOCAL block ``[n_loc rows, n_cols keys]``. ``constraints`` is the kernel's declaration (the F1
    serve layer's, passed through by the adapter): ``min_rows`` / ``min_cols`` (smallest extent it serves), ``row_multiple`` / ``col_multiple`` (block
    divisibility it requires when it does not mask ragged blocks), ``max_cols``. Returns ``(True, "fits")`` or ``(False, <the first violated key as
    key=value:extent>)`` for the lever line. No constraints → ``(True, "unconstrained")``."""
    c = dict(constraints or {})
    if not c:
        return True, "unconstrained"
    checks = (("min_rows", lambda v: n_loc >= int(v), n_loc), ("min_cols", lambda v: n_cols >= int(v), n_cols),
              ("row_multiple", lambda v: n_loc % int(v) == 0, n_loc), ("col_multiple", lambda v: n_cols % int(v) == 0, n_cols),
              ("max_cols", lambda v: n_cols <= int(v), n_cols), ("max_rows", lambda v: n_loc <= int(v), n_loc))
    for key, ok, extent in checks:
        if key in c and c[key] is not None and not ok(c[key]):
            return False, f"{key}={c[key]}:{extent}"
    unknown = sorted(k for k in c if k not in {k for k, _, _ in checks})
    if unknown:
        return False, f"unknown_constraint={','.join(unknown)}"
    return True, "fits"


def kernel_fields(name: str, ok: bool, reason: str) -> dict:
    """``{"kernel": <name or 'jnp'>, "kernel_reason": <reason>}`` — the lever-line fragment of the per-shard kernel decision."""
    return {"kernel": str(name) if ok else "jnp", "kernel_reason": str(reason).replace(" ", "_")}
