"""Sharding a whole tensor into row shards and back, applying a ROW-LOCAL sub-module to a shard (no communication), and the row-block /
storage helpers the row-sharded statements share.

A row-local statement (pair transition, LayerNorm, dropout-free residual add, a per-row linear / one-hot embedding, the recycling
``LN(z_prev) + linear``) returns on rows ``[r0, r1)`` exactly the rows it would return on the whole tensor, so under sharding it runs
on the shard unchanged; :func:`local_rows` only bounds its transient by evaluating it in row blocks (it composes
:func:`opt_core.mem.torch_rowchunk.rowchunk_tensor` — one row-chunking primitive in the core).

API (``layout`` = :class:`opt_core.mem.rowpair.dist.Layout`):
    shard_rows(x_full, layout, dim=0)          this rank's rows of a whole tensor (a VIEW: ``narrow``) — the entry seam of a sharded region
    unshard_rows(x_shard, layout, dim=0)       the whole tensor on every rank (``all_gather_rows``; a copy) — the exit seam; ``gather_rows`` is the
                                               same function (short spelling)
    unshard_rows_to_rank0(x_shard, layout, dim=0)   the whole tensor on rank 0 only (None elsewhere) — the output seam; ``gather_rows_to(x, layout,
                                               dim, dst=0)`` is the same function (``dst != 0`` refused by name)
    local_rows(fn, x_shard, rows=None, ...)    ``out[i] = fn(x_shard[i])`` in blocks of ``rows`` local rows (``rows=None``: one call); ``inplace``
                                               form ``local_rows_(fn, z_shard, rows)`` writes ``fn``'s rows back into ``z_shard``
    shard_mask(mask_full, layout)              the row shard of an ``[N, N]`` (or ``[*, N, N]``) pair mask
    choose_block_rows(N, C, elem_bytes=4, *, rows=, budget_bytes=, env=, default_mb=512, cap_elems=2**31-1, align=1, n_max=, bytes_per_row=)
                                               THE rows-per-block chooser of the family -> ``(rows, source)``: ``source`` = given | budget |
                                               env:<NAME> | default (+cap, +align, +n_max suffixes when those bit); every row-streaming
                                               statement (pair-init rows, OPM rows, template rows, z_cond rows, head logits rows, transposes)
                                               takes its block size here so the census prints one grammar
    row_block_size(N, C, elem_bytes=4, target_bytes=None, cap_elems=2**31-1)   ``choose_block_rows(...)[0]`` (``target_bytes`` = ``budget_bytes``)
    row_blocks(n, bs)                          ``[(b0, b1), ...]`` covering ``range(n)`` in steps of ``bs``
    iter_row_blocks(layout, block_rows=None, unit=None)   this rank's row blocks ON THE CHUNK GRID: ``[(b0, b1, g0, g1), ...]`` local ``[b0, b1)`` = global
                                               ``[g0, g1)``, every block start a multiple of ``unit`` (``layout.align``, else ``chunk_align()``),
                                               ``block_rows`` rounded down to that unit (at least one unit), last block ragged
    produce_rows_(out_loc, layout, fn, op="set"|"add", block_rows=None, row_dim=-3)   THE row-production loop of the family: for each block
                                               ``out_loc[.., b0:b1, ...] (=|+=) fn(g0, g1)`` (``fn`` returns the ``b1-b0`` rows of the dense
                                               statement for global rows ``[g0, g1)``); ``block_rows`` defaults to :func:`row_block_size` of one
                                               output row; returns ``out_loc``. Pair-init rows, outer-product-mean rows, template rows, z_cond
                                               rows are all this loop with their own ``fn``.
    bcast_small(t, src=0)                      broadcast a small replicated tensor in place (``comm().bcast_``)
    ln_rows_guarded(ln, x, rows_dim=-3, block=None)   a whole-shard LayerNorm evaluated in row blocks once the operand reaches
                                               ``ROWPAIR_LN_GUARD_ELEMS`` elements (default 2^31; 0 disables): ``F.layer_norm`` on CUDA fp32 tensors
                                               with >= 2^32 elements returned wrong rows beyond the boundary (torch 2.7) — LN rows are independent,
                                               so per-row arithmetic is unchanged
    owns_whole_storage(t) / release_storage_(t) / regrow_storage_(t, nbytes=None)   park a shard's device memory (``storage.resize_(0)``) and
                                               re-attach it, for tensors that span their whole storage
    ctx / default_ctx / chunk_align            re-exported from :mod:`.dist`; transpose_shard / transpose_shard_streamed / transpose_shard_inplace_ /
                                               transpose_band / ring_pass re-exported from :mod:`.ring`; ring_blocks / allreduce_ from :mod:`.dist`
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from . import RowpairRefused
from ._torch import torch
from .dist import (Layout, all_gather_rows, allreduce_, chunk_align, comm, ctx, default_ctx, env_float, gather_rows_to_rank0, is_dist, ring_blocks,
                   row_parts)
from .ring import ring_pass, transpose_band, transpose_shard, transpose_shard_inplace_, transpose_shard_streamed

__all__ = ["shard_rows", "unshard_rows", "gather_rows", "unshard_rows_to_rank0", "gather_rows_to", "local_rows", "local_rows_", "shard_mask",
           "choose_block_rows", "row_block_size", "row_blocks", "iter_row_blocks", "produce_rows_", "PRODUCE_OPS", "BLOCK_SOURCES", "bcast_small", "ln_rows_guarded", "owns_whole_storage", "storage_resizable", "release_storage_", "regrow_storage_",
           "ctx", "default_ctx", "chunk_align", "row_parts", "Layout", "transpose_shard", "transpose_shard_streamed", "transpose_shard_inplace_",
           "transpose_band", "ring_pass", "ring_blocks", "allreduce_", "ROWBLK_MB", "LN_GUARD_ELEMS"]

ROWBLK_MB = 512.0                 # row_block_size's default target per block (ROWPAIR_ROWBLK_MB)
PRODUCE_OPS = ("set", "add")
BLOCK_SOURCES = ("given", "budget", "env", "default")   # choose_block_rows' primary sources (census grammar: source[+cap][+align][+n_max])
LN_GUARD_ELEMS = 2 ** 31          # ln_rows_guarded's default operand size from which LN runs in row blocks (ROWPAIR_LN_GUARD_ELEMS)


def shard_rows(x_full, layout: Layout, dim: int = 0):
    """Rows ``[r0, r1)`` of ``x_full`` along ``dim`` (a view). ``x_full.shape[dim]`` must be ``layout.N`` (else refused by name)."""
    d = dim % x_full.dim()
    if int(x_full.shape[d]) != layout.N:
        raise RowpairRefused(f"shard_rows: dim {dim} has {int(x_full.shape[d])} rows, layout.N={layout.N}")
    if layout.replicated or layout.P == 1:
        return x_full
    return x_full.narrow(d, layout.r0, layout.R)


def unshard_rows(x_shard, layout: Layout, dim: int = 0):
    """The whole tensor on every rank: ``all_gather_rows`` along ``dim`` (moved to dim 0 and back when ``dim != 0``: then a view — call ``.contiguous()`` to own it)."""
    if layout.replicated or layout.P == 1 or not is_dist():
        return x_shard
    d = dim % x_shard.dim()
    if d == 0:
        return all_gather_rows(x_shard.contiguous(), layout)
    return all_gather_rows(x_shard.movedim(d, 0).contiguous(), layout).movedim(0, d)


gather_rows = unshard_rows


def unshard_rows_to_rank0(x_shard, layout: Layout, dim: int = 0):
    """The whole tensor on rank 0 (None on other ranks)."""
    if layout.replicated or layout.P == 1 or not is_dist():
        return x_shard if layout.rank == 0 else None
    d = dim % x_shard.dim()
    out = gather_rows_to_rank0(x_shard.movedim(d, 0).contiguous() if d else x_shard.contiguous(), layout)
    if out is None:
        return None
    return out.movedim(0, d) if d else out


def gather_rows_to(x_shard, layout: Layout, dim: int = 0, dst: int = 0):
    """:func:`unshard_rows_to_rank0` (short spelling); only ``dst == 0`` (the output rank) is supported."""
    if int(dst) != 0:
        raise RowpairRefused(f"gather_rows_to: dst={dst}: only rank 0 (the output rank) receives whole tensors")
    return unshard_rows_to_rank0(x_shard, layout, dim)


def local_rows(fn: Callable, x_shard, rows: Optional[int] = None, *, row_dim: int = 0, out=None, out_dtype=None, ledger=None,
               key: Optional[str] = None, lever: str = "rowpair"):
    """A row-local ``fn`` over the shard: ``rows=None`` -> ``fn(x_shard)``; else ``opt_core.mem.torch_rowchunk.rowchunk_tensor`` in blocks
    of ``rows`` along ``row_dim`` into ONE preallocated output (``out`` / ``out_dtype`` as there)."""
    if rows is None or int(rows) >= int(x_shard.shape[row_dim]):
        y = fn(x_shard)
        if ledger is not None and key:
            ledger.count(key)
        return y if out is None else out.copy_(y)
    from ..torch_rowchunk import rowchunk_tensor
    rd = row_dim % x_shard.dim()
    return rowchunk_tensor(fn, x_shard, int(rows), row_dim=rd - x_shard.dim(), out=out, out_dtype=out_dtype, ledger=ledger, key=key,
                           lever=lever)


def local_rows_(fn: Callable, z_shard, rows: Optional[int] = None, *, add: bool = False, ledger=None, key: Optional[str] = None):
    """In place: ``z_shard[i0:i1] = fn(z_shard[i0:i1])`` (or ``+=`` with ``add=True``) per local row block — the residual form of a
    row-local sub-module without a second shard-sized tensor. Returns ``z_shard``."""
    from .dist import zadd, zblocks, zlen, zrows, zwrite
    R = zlen(z_shard)
    step = R if rows is None else max(1, int(rows))
    for i0, i1 in zblocks(R, step):
        y = fn(zrows(z_shard, i0, i1))
        if add:
            zadd(z_shard, i0, i1, y)
        else:
            zwrite(z_shard, i0, i1, y)
        del y
    if ledger is not None and key:
        ledger.count(key)
    return z_shard


def shard_mask(mask_full, layout: Layout):
    """The row shard of a pair mask ``[N, N]`` / ``[*, N, N]`` (rows on dim -2)."""
    return shard_rows(mask_full, layout, dim=-2)


# ----------------------------------------------------------------------------------------------------------------- row blocks
def choose_block_rows(N: Optional[int] = None, C: Optional[int] = None, elem_bytes: int = 4, *, rows: Optional[int] = None,
                      budget_bytes: Optional[int] = None, env: Optional[str] = "ROWPAIR_ROWBLK_MB", default_mb: float = ROWBLK_MB,
                      cap_elems: Optional[int] = 2 ** 31 - 1, align: int = 1, n_max: Optional[int] = None,
                      bytes_per_row: Optional[int] = None, elems_per_row: Optional[int] = None) -> Tuple[int, str]:
    """THE rows-per-block chooser: how many rows of a ``[rows, N, C]``-class slab one streaming step handles, and WHY (``source``, printed in the
    schedule census by the caller). One row costs ``bytes_per_row`` (default ``N * C * elem_bytes``) and ``elems_per_row`` (default ``N * C``)
    elements. Primary source, first that applies: ``rows`` (``given``) | ``budget_bytes`` (``budget``: e.g. a fraction of
    :func:`opt_core.mem.rowpair.dist.agreed_free_bytes` — the caller passes the AGREED value so ranks meet in the same collectives) | the MiB
    named by ``env`` when that variable is set (``env:<NAME>``) | ``default_mb`` MiB (``default``). Then, in order: ``cap_elems`` (rows * elems_per_row
    stays below it — the int32-indexing rule; suffix ``+cap`` when it bites), ``align`` (rounded DOWN to a multiple, at least one ``align``; suffix
    ``+align``), ``n_max`` (no block longer than the longest shard; suffix ``+n_max``). At least 1. A pure function of its arguments / the
    environment: identical on every rank given identical arguments."""
    if elems_per_row is None:
        if N is None or C is None:
            if bytes_per_row is None:
                raise RowpairRefused("choose_block_rows: give N and C, or bytes_per_row (+ elems_per_row for the int32 cap)")
            elems_per_row = max(1, int(bytes_per_row) // max(1, int(elem_bytes)))
        else:
            elems_per_row = max(1, int(N) * int(C))
    if bytes_per_row is None:
        bytes_per_row = int(elems_per_row) * max(1, int(elem_bytes))
    bytes_per_row = max(1, int(bytes_per_row))
    if rows is not None:
        b, source = max(1, int(rows)), "given"
    elif budget_bytes is not None:
        b, source = max(1, int(budget_bytes) // bytes_per_row), "budget"
    elif env and str(__import__("os").environ.get(env, "")).strip():
        b, source = max(1, int(env_float(env, default_mb) * 2 ** 20) // bytes_per_row), f"env:{env}"
    else:
        b, source = max(1, int(float(default_mb) * 2 ** 20) // bytes_per_row), "default"
    if cap_elems is not None:
        capped = max(1, int(cap_elems) // max(1, int(elems_per_row)))
        if capped < b:
            b, source = capped, source + "+cap"
    a = max(1, int(align))
    if a > 1:
        nb = max(a, (b // a) * a)
        if nb != b:
            b, source = nb, source + "+align"
    if n_max is not None and int(n_max) >= 1 and b > int(n_max):
        b, source = int(n_max), source + "+n_max"
    return int(b), source


def row_block_size(N: int, C: int, elem_bytes: int = 4, target_bytes: Optional[int] = None, cap_elems: int = 2 ** 31 - 1) -> int:
    """Rows per streaming block for ``[rows, N, C]`` slabs: :func:`choose_block_rows` with ``budget_bytes=target_bytes`` (None: ``ROWPAIR_ROWBLK_MB``,
    default 512 MiB) and the int32 cap; the int only."""
    return choose_block_rows(N, C, elem_bytes, budget_bytes=target_bytes, cap_elems=cap_elems)[0]


def row_blocks(n: int, bs: int) -> List[Tuple[int, int]]:
    """``[(b0, b1), ...]`` covering ``range(n)`` in steps of ``bs``."""
    bs = max(1, int(bs))
    return [(b0, min(int(n), b0 + bs)) for b0 in range(0, int(n), bs)]


def iter_row_blocks(layout: Layout, block_rows: Optional[int] = None, unit: Optional[int] = None) -> List[Tuple[int, int, int, int]]:
    """This rank's row blocks on the chunk grid: ``[(b0, b1, g0, g1), ...]`` with local rows ``[b0, b1)`` = global rows ``[g0, g1) = [r0+b0, r0+b1)``.
    ``unit`` (default ``layout.align``, else :func:`opt_core.mem.rowpair.dist.chunk_align`) is the engine's chunk size: every block start is a
    multiple of it (``r0`` is, by the layout) so a per-block engine call walks the same chunk grid as the dense statement; ``block_rows``
    (default: the whole shard) is rounded DOWN to a multiple of ``unit`` (at least one unit); the last block is the ragged remainder."""
    u = int(unit) if unit is not None else int(layout.align or chunk_align())
    u = max(1, u)
    if layout.r0 % u:
        raise RowpairRefused(f"iter_row_blocks: rank rows start at {layout.r0}, not a multiple of the chunk unit {u} (build the Layout with align={u})")
    R = layout.n_loc
    if block_rows is None or int(block_rows) >= R:
        bs = max(1, R)                                                       # one block: the whole shard (its start r0 is on the grid)
    else:
        bs = max(u, (max(1, int(block_rows)) // u) * u)                      # block starts stay on the grid; the last block is ragged
    return [(b0, b1, layout.r0 + b0, layout.r0 + b1) for b0, b1 in row_blocks(R, bs)]


def produce_rows_(out_loc, layout: Layout, fn: Callable[[int, int], object], *, op: str = "set", block_rows: Optional[int] = None,
                  row_dim: int = -3, unit: Optional[int] = None, ledger=None, key: Optional[str] = None):
    """The row-production loop: for each block of :func:`iter_row_blocks` ``out_loc.narrow(row_dim, b0, b1-b0) (= | +=) fn(g0, g1)``, where
    ``fn(g0, g1)`` returns the rows ``[g0, g1)`` (global) of the DENSE statement — so ``out_loc`` ends equal to rows ``r0:r1`` of the dense result
    (``op="set"``) or incremented by them (``op="add"``), with only one block's transient alive at a time. ``block_rows=None``: about
    :func:`choose_block_rows` of one output row (``ROWPAIR_ROWBLK_MB``, default 512 MiB, int32 cap, aligned to the unit); the value and its source
    are recorded in the schedule census (``produce_block_rows``, ``produce_block_source``). Returns ``out_loc``. No communication; P == 1 runs the
    same loop over all rows."""
    if op not in PRODUCE_OPS:
        raise RowpairRefused(f"produce_rows_: op {op!r} not in {PRODUCE_OPS}")
    rd = row_dim % out_loc.dim()
    if int(out_loc.shape[rd]) != layout.n_loc:
        raise RowpairRefused(f"produce_rows_: out_loc has {int(out_loc.shape[rd])} rows on dim {row_dim}, layout.n_loc={layout.n_loc}")
    per_row = max(1, out_loc.numel() // max(1, layout.n_loc))
    u = int(unit) if unit is not None else int(layout.align or chunk_align())
    block_rows, source = choose_block_rows(elems_per_row=per_row, elem_bytes=out_loc.element_size(), rows=block_rows, align=u, n_max=layout.n_max)
    blocks = iter_row_blocks(layout, block_rows, u)
    from .evidence import record_schedule
    record_schedule(produce_block_rows=block_rows, produce_block_source=source)
    for b0, b1, g0, g1 in blocks:
        y = fn(g0, g1)
        dst = out_loc.narrow(rd, b0, b1 - b0)
        if tuple(y.shape) != tuple(dst.shape):
            raise RowpairRefused(f"produce_rows_: fn({g0}, {g1}) returned {tuple(y.shape)}, expected {tuple(dst.shape)}")
        if op == "set":
            dst.copy_(y)
        else:
            dst.add_(y)
        del y
    if ledger is not None and key:
        ledger.count(key)
    return out_loc


def bcast_small(t, src: int = 0):
    """Broadcast a small replicated tensor in place; returns it."""
    comm().bcast_(t, src)
    return t


def ln_rows_guarded(ln: Callable, x, rows_dim: int = -3, block: Optional[int] = None, limit: Optional[int] = None):
    """``ln(x)`` — evaluated in row blocks along ``rows_dim`` once ``x.numel()`` reaches ``limit`` (``ROWPAIR_LN_GUARD_ELEMS``, default 2^31; 0 =
    never block). Identical per-row arithmetic (LayerNorm rows are independent)."""
    lim = int(env_float("ROWPAIR_LN_GUARD_ELEMS", float(LN_GUARD_ELEMS))) if limit is None else int(limit)
    if lim <= 0 or x.numel() < lim or x.dim() < 3:
        return ln(x)
    n = x.shape[rows_dim]
    per_row = x.numel() // max(n, 1)
    if block is None:
        block = max(1, (lim // 2) // max(per_row, 1))
    out = torch.empty_like(x)
    idx = [slice(None)] * x.dim()
    for i0 in range(0, n, block):
        idx[rows_dim] = slice(i0, min(n, i0 + block))
        out[tuple(idx)] = ln(x[tuple(idx)])
    return out


# ----------------------------------------------------------------------------------------------------------------- storage parking
def owns_whole_storage(t) -> bool:
    """True if ``t`` is contiguous and spans its entire storage (offset 0, nbytes == numel*esize) — only then may its storage be released and
    re-grown behind the caller's back without touching any other tensor (given :func:`storage_resizable`)."""
    return (t.is_contiguous() and t.storage_offset() == 0
            and t.untyped_storage().nbytes() == t.numel() * t.element_size())


def storage_resizable(t) -> bool:
    """True if ``t``'s storage was allocated by torch and can be ``resize_``d (released / re-grown). False for FOREIGN memory torch only views —
    a zero-copy DLPack import of another framework's buffer, ``torch.from_numpy`` — whose bytes torch neither frees nor may overwrite behind the
    owner's back: such a shard is never parked and never embedded in place (the words ``storage_not_resizable`` / ``foreign_storage``)."""
    st = t.untyped_storage()
    if not hasattr(st, "resizable"):
        raise RowpairRefused(f"storage_resizable: torch {getattr(torch, '__version__', '?')} has no UntypedStorage.resizable() — the park / in-place levers "
                             "cannot tell torch-owned from foreign storage on this torch (they refuse rather than guess)")
    return bool(st.resizable())


def release_storage_(t) -> int:
    """Free ``t``'s device (or CPU) memory now: synchronize (pending kernels / collectives that read it), then ``storage.resize_(0)``. The tensor
    object, its shape and strides stay valid; :func:`regrow_storage_` re-attaches fresh (uninitialised) memory. Returns the released bytes."""
    if not owns_whole_storage(t):
        raise RowpairRefused("release_storage_: tensor must own its whole storage (contiguous, offset 0, no larger base)")
    if not storage_resizable(t):
        raise RowpairRefused("release_storage_: the tensor's storage is not resizable (foreign memory: a DLPack / numpy view)")
    if t.is_cuda:
        torch.cuda.synchronize()
    nbytes = t.untyped_storage().nbytes()
    t.untyped_storage().resize_(0)
    return int(nbytes)


def regrow_storage_(t, nbytes: Optional[int] = None):
    """Re-allocate storage released by :func:`release_storage_` (contents undefined until written); returns ``t``."""
    if nbytes is None:
        nbytes = t.numel() * t.element_size()
    t.untyped_storage().resize_(int(nbytes))
    return t
