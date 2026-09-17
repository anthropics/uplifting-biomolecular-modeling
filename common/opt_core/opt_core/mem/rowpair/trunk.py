"""The row-sharded TRUNK of a pair-representation model: the pair representation is BORN as this rank's row shard and carried as a shard
across the recycling iterations — nothing ``N x N x C`` is ever whole on a rank.

Layout. ``z_loc = z[..., r0:r1, :, :]`` (``[*, R, N, C]``; rows on dim -3, ``(r0, r1) = layout.bounds[rank]`` of :class:`.dist.Layout`). The
single representation ``s``, the input single ``s_input``, the MSA representation ``m`` and every atom tensor are REPLICATED by design (the
engine's own tensors, identical on every rank; the schedule census says ``trunk_replicated=s,s_input,m``). Every statement below equals the
dense statement element for element; a GEMM inside an engine callable runs at ``M = rows`` instead of ``M = N`` (bit-exact iff the kernel is
M-invariant: CPU fp32 at one thread is; cuBLAS was on every stack tested, though not by contract). No dtype is changed anywhere.

Row statements — pure functions of GLOBAL rows ``[g0, g1)``; ``(0, N)`` IS the dense statement (no layout, no communication):
    same_rows(x, g0, g1)                          ``x[..., g0:g1, None] == x[..., None, :]``                       -> bool ``[*, rows, N]``
    reloffset_rows(pos, g0, g1, clip, condition)  relative-offset bins ``where(condition, clamp(pos_i - pos_j + clip, 0, 2 clip), 2 clip + 1)``
                                                  for rows i in ``[g0, g1)``                                        -> int64 ``[*, rows, N]``
    relpos_onehot_rows(pos, g0, g1, clip, ...)    the one-hot of those bins over ``2 clip + 2`` classes — the LAZY relpos: only the row slab
                                                  ``[*, rows, N, 2 clip + 2]`` exists, never ``[N, N, bins]``
    outer_sum_rows(a, b, g0, g1)                  ``a[..., g0:g1, None, :] + b[..., None, :, :]`` (the two single projections' outer sum) -> ``[*, rows, N, C]``
    feature_rows(x, g0, g1, device)               the row slab of a replicated pair FEATURE ``[*, N, N(, F)]`` on ``device`` — the feature may live
                                                  in host memory (``ROWPAIR``-agnostic: the adapter decides), only ``rows x N`` moves per block
    pair_mask_rows(token_mask, g0, g1)            ``token_mask[..., g0:g1, None] * token_mask[..., None, :]``

Schedules — a SHARDED layout is required (``P == 1`` / replicated / no group is refused BY NAME through :func:`.dist.require_sharded`: at
``--n_gpu 1`` the adapter installs nothing and the engine's own trunk runs):
    (block sizes)                                 every schedule's rows per block come from THE chooser :func:`.shard.choose_block_rows`:
                                                  ``rows=`` given > the row count in ``ROWPAIR_INIT_ROWS`` / ``ROWPAIR_RECYCLE_ROWS`` >
                                                  ``ROWPAIR_ROWBLK_MB`` MiB of ``[rows, N, channels]`` transients > 512 MiB; value + source are
                                                  printed in the schedule census (``trunk_init_rows[_source]``, ``trunk_recycle_rows[_source]``);
                                                  the blocks are :func:`.shard.produce_rows_`'s (starts on the engine chunk grid, ``Layout.align``
                                                  / ``ROWPAIR_CHUNK_ALIGN``)
    init_pair_shard(layout, rows_fn, like, rows=) seam "pair init": ``z_loc[..., i0:i1, :, :] = rows_fn(r0 + i0, r0 + i1)`` per local row block — the
                                                  engine's z_init statement (outer sum + relpos linear + bond linear ...) evaluated on ROWS only;
                                                  ``like [*, N, C]`` fixes the shard's leading dims / C / dtype / device
    recycle_shard_(z_loc, zinit, update_fn, layout, rows=)  seam "recycling": in place ``z_loc[b] = zinit.block(b) + update_fn(z_loc[b])`` per row block
                                                  (``update_fn`` = the engine's ``linear(layer_norm(z_prev))``); ``z_loc is None`` (cycle 0, stock
                                                  ``z_prev = zeros_like(z_init)``): ``update_fn`` runs on a ZEROS ROW BLOCK and a fresh shard is
                                                  allocated — no zeros shard ever exists
    ShardPark(t, park=, ...)                       a shard parked in pinned host memory (``ROWPAIR_PARK_ZINIT``) serving row blocks to the device:
                                                  ``block(i0, i1)`` · ``full()`` · ``release()`` · ``record()``; the device storage is released at
                                                  park time; pinned buffers come from the package's ONE pool (:class:`opt_core.mem.torch_hostpair.PinPool`);
                                                  a refused page-lock is answered pageable and NAMED (``where=host_pageable:<kind>``), or refused
                                                  under ``ROWPAIR_PARK_STRICT=1``; ``census_words(key)`` = ``<key> <key>_gib <key>_release`` for the schedule
                                                  census (``release`` = storage_resized_to_0 | not_released:shared_storage |
                                                  not_released:storage_not_resizable | kept | resident)
    park_pool(pin=, strict=, pin_max_bytes=)       the ONE constructor of a park's private pool (``ROWPAIR_PARK_PIN_MAX_GB``, ``ROWPAIR_PARK_STRICT``):
                                                  :class:`ShardPark` and :class:`.heads.ZTrunkPlan` answer a locked-memory cap with the same words
    guard_replicated(t, name)                      a replicated tensor proven bit-identical on all ranks (:func:`.dist.allreduce_checksum`)
    guard_rng_replicated(name, device)             the torch RNG STREAM proven identical on all ranks (checksums of the generator states) before a
                                                  replicated random draw (the MSA subsample of the MSA-module embedder)
    run_trunk_sharded(layout, ...)                 the trunk DRIVER: owns the ``z_loc`` carry across ``n_cycles``; every engine sub-module enters as
                                                  a CALLABLE on the shard (template embedder, MSA module, pairformer are installed by the adapter
                                                  behind those callables); ends with NO gather (``gather="none"``: the consumers are row-local) or
                                                  ONE (``"all"`` / ``"rank0"``)

    HostSlabLease / park_host(t) / host_slab_lease  ``ROWPAIR_HOST_SLAB=lease``: the family's u-sized host copies — the z parks
                                                  (:class:`ShardPark`, :class:`.template.ParkedStorage`, :class:`.heads.ZTrunkPlan`) and the
                                                  triangle-multiplication row mirror (:func:`.trimul.host_mirror`) — share ONE process-wide (per rank)
                                                  exact-size pinned slab handed to ONE holder at a time (:func:`host_alloc` on a lease); a second
                                                  concurrent holder gets a NAMED private buffer (census ``host_slab_second``) or is refused under
                                                  ``ROWPAIR_HOST_SLAB_STRICT=1``; unset = today's private buffers (census ``host_slab=private``)

Environment (all optional; every resolved value is printed in the schedule census via :func:`.evidence.record_schedule`):
    ROWPAIR_PARK_ZINIT=0|1|recompute  park the z_init shard in host memory between cycles (default 0), or (``recompute``) hold NO copy of it
                                  anywhere: a row block of z_init re-runs the init statements on the init grid (:meth:`ShardPark.recompute`); census
                                  park_z_init=<where> park_z_init_gib park_z_init_release zinit_park=resident|parked|recompute
    ROWPAIR_HOST_SLAB=lease       the u-sized host copies share one leased slab per process (see above; default unset = private buffers); census
                                  host_slab=private|lease host_slab_gib host_slab_holder host_slab_leases host_slab_second host_slab_second_gib
    ROWPAIR_HOST_SLAB_STRICT=0|1  a second concurrent holder of the leased slab is refused by name (default 0: a NAMED private buffer)
    ROWPAIR_PARK_STRICT=0|1       a page-lock a park's pool cannot honour is a refusal by name instead of a named pageable buffer (default 0; :func:`park_strict`)
    ROWPAIR_PARK_PIN_MAX_GB=<g>   pinned budget of a park's private pool (default: unbudgeted; :func:`park_pool` — the one constructor, shared with :class:`.heads.ZTrunkPlan`)
    ROWPAIR_INIT_ROWS=<n>         rows per block of the pair-init schedule (default: ROWPAIR_ROWBLK_MB-derived)
    ROWPAIR_RECYCLE_ROWS=<n>      rows per block of the recycling schedule (default: ROWPAIR_ROWBLK_MB-derived)
    ROWPAIR_TRUNK_GUARD=rng|off   the replicated-draw guard before the MSA callable of every cycle (default rng; ``off`` is printed as such)
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Callable, Dict, Optional, Sequence, Tuple

from . import RowpairRefused
from ._torch import torch
from .dist import checksum, comm, is_dist, RowpairRefused, Layout, allreduce_checksum, env_int, require_sharded, env_flag
from .evidence import record_schedule
from .shard import choose_block_rows, iter_row_blocks, owns_whole_storage, produce_rows_, release_storage_, storage_resizable, unshard_rows, unshard_rows_to_rank0

__all__ = ["same_rows", "reloffset_rows", "relpos_onehot_rows", "outer_sum_rows", "feature_rows", "pair_mask_rows",
           "init_pair_shard", "recycle_shard_", "ShardPark", "park_pool", "park_strict", "host_alloc", "guard_replicated", "guard_replicated_params",
           "guard_rng_replicated", "run_trunk_sharded", "TrunkOut", "INIT_TRANSIENT_CHANNELS", "GATHERS", "GUARDS", "REPLICATED",
           "ENV_PARK_ZINIT", "ENV_PARK_STRICT", "ENV_PARK_PIN_MAX_GB",
           "ENV_HOST_SLAB", "ENV_HOST_SLAB_STRICT", "HOST_SLAB_MODES", "host_slab_mode", "HostSlabLease", "LeasedHost", "park_host",
           "host_slab_lease", "host_slab_leases", "host_slab_bytes", "release_host_slabs", "release_free_host_slabs",
           "ZINIT_RECOMPUTE", "ZINIT_PARK_MODES", "zinit_park_mode", "init_grid", "census_token",
           "store_rows", "ENV_HOST_TRIM", "HOST_TRIM_MODES", "host_trim_mode", "host_trim", "HOST_TRIM_STATS", "DevicePeak", "ENV_TEMPL_ORDER_NAME"]

LEVER = "rowpair.trunk"
ENV_PARK_ZINIT = "ROWPAIR_PARK_ZINIT"              # 1: park the z_init shard in host memory between cycles; recompute: no copy anywhere (ShardPark.recompute)
ENV_PARK_STRICT = "ROWPAIR_PARK_STRICT"            # 1: a page-lock a park's pool cannot honour is a refusal by name (else a named pageable buffer)
ENV_PARK_PIN_MAX_GB = "ROWPAIR_PARK_PIN_MAX_GB"    # pinned budget of a park's private pool (unset: unbudgeted)
ENV_PARK_CHUNK_GIB = "ROWPAIR_PARK_CHUNK_GIB"      # GiB per host chunk of a park's exact-size host copy (default 8; 0 = one allocation of the whole size)
ENV_PARK_ZINIT_HOLD = "ROWPAIR_PARK_ZINIT_HOLD"    # 1: hold the z_init park to trunk exit (else released right after the final cycle's recycle)
ENV_HOST_SLAB = "ROWPAIR_HOST_SLAB"                # lease: the u-sized host copies share ONE process-wide (per rank) slab, one holder at a time
ENV_HOST_SLAB_STRICT = "ROWPAIR_HOST_SLAB_STRICT"  # 1: a second concurrent holder of the leased slab is refused by name (else a NAMED private buffer)
HOST_SLAB_MODES = ("private", "lease")
ENV_HOST_TRIM = "ROWPAIR_HOST_TRIM"                # the template->MSA seam trim of torch's cached-free pinned host blocks: unset = on under
HOST_TRIM_MODES = ("auto", "0", "1")               # ROWPAIR_HOST_SLAB=lease / off otherwise (auto); 0 = never; 1 = always (also under private pools)
ENV_TEMPL_ORDER_NAME = "ROWPAIR_TEMPL_ORDER"       # template.ENV_TEMPL_ORDER (named here too: the trunk driver brackets the template stage's device peak when it is SET)
ZINIT_RECOMPUTE = "recompute"                      # the ROWPAIR_PARK_ZINIT word of the copy-free z_init placement
ZINIT_PARK_MODES = ("resident", "parked", ZINIT_RECOMPUTE)
INIT_TRANSIENT_CHANNELS = 3          # rows=None for init_pair_shard: budget [rows, N, 3 C] fp32-equivalent (the block + the two addends) per block
GATHERS = ("none", "all", "rank0")
GUARDS = ("rng", "off")
REPLICATED = "s,s_input,m"           # the replicated-by-design tensors of the trunk (census field trunk_replicated)


def _env_flag(name: str, default: str = "0") -> bool:
    return env_flag(name, str(default).strip().lower() not in ("", "0", "false", "no", "off"))


_TOKEN_OPEN = re.compile(r"\s*\(\s*")
_TOKEN_WS = re.compile(r"\s+")


def census_token(text) -> str:
    """A census VALUE as ONE ``\\S+`` token: a kit's LEVER-line writer refuses values with blanks by contract, and the host-slab
    holders are named like ``ParkedStorage(z_trunk (confidence))`` / ``trimul row mirror`` in the log lines. Rule: the first ``(`` -> ``:``,
    later ``(`` -> ``.``, ``)`` dropped, runs of whitespace -> ``_``; e.g. ``ParkedStorage:z_trunk.confidence``, ``trimul_row_mirror``,
    ``ParkedStorage:z_shard.template_pair_stack``. Log lines keep the readable names; only census values pass through here."""
    t = str(text).strip()
    t = _TOKEN_OPEN.sub(":", t, count=1)
    t = _TOKEN_OPEN.sub(".", t).replace(")", "")
    t = _TOKEN_WS.sub("_", t)
    return t or "-"


def _numel(shape: Sequence[int]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def host_slab_mode() -> str:
    """``ROWPAIR_HOST_SLAB``: unset / ``0`` / ``private`` -> ``private`` (today: every park and the row mirror takes its own host buffer);
    ``lease`` / ``1`` -> ``lease`` (:class:`HostSlabLease`). Any other word is refused by name."""
    v = os.environ.get(ENV_HOST_SLAB, "").strip().lower()
    if v in ("", "0", "private", "off", "no", "false"):
        return "private"
    if v in ("lease", "1", "on", "yes", "true"):
        return "lease"
    raise RowpairRefused(f"{ENV_HOST_SLAB}={v!r}: one of {HOST_SLAB_MODES} is required", LEVER)


def zinit_park_mode(park=None) -> str:
    """The z_init placement word — ``resident`` | ``parked`` | ``recompute``: ``park`` given wins (``True`` / ``False`` / ``"recompute"``), else
    ``ROWPAIR_PARK_ZINIT`` (``recompute`` -> recompute; any other set flag word -> parked by :func:`.dist.env_flag`'s rule; unset / 0 -> resident).
    ``parked`` is the INTENT: :class:`ShardPark` keeps a CPU shard resident unless ``force_host`` (parking host to host frees nothing) — unchanged."""
    if park is None:
        if os.environ.get(ENV_PARK_ZINIT, "").strip().lower() == ZINIT_RECOMPUTE:
            return ZINIT_RECOMPUTE
        return "parked" if _env_flag(ENV_PARK_ZINIT) else "resident"
    if isinstance(park, str):
        w = park.strip().lower()
        if w == ZINIT_RECOMPUTE:
            return ZINIT_RECOMPUTE
        if w in ("1", "true", "yes", "on", "parked"):
            return "parked"
        if w in ("", "0", "false", "no", "off", "resident"):
            return "resident"
        raise RowpairRefused(f"z_init park word {park!r}: one of {ZINIT_PARK_MODES} (or 0/1) is required", LEVER)
    return "parked" if bool(park) else "resident"


# ================================================================================================================ row statements (pure)
def _check_rows(x, g0: int, g1: int, what: str, dim: int = -1) -> None:
    n = int(x.shape[dim])
    if not (0 <= int(g0) <= int(g1) <= n):
        raise RowpairRefused(f"{what}: rows [{g0}, {g1}) outside [0, {n}] of shape {tuple(x.shape)}", LEVER)


def same_rows(x, g0: int, g1: int):
    """``(x[..., :, None] == x[..., None, :])[..., g0:g1, :]`` for a per-token id ``x[*, N]`` (same chain / same entity / same residue)."""
    _check_rows(x, g0, g1, "same_rows")
    return x[..., g0:g1, None] == x[..., None, :]


def reloffset_rows(pos, g0: int, g1: int, clip: int, condition=None):
    """The relative-offset bin of every pair (i, j), rows ``i in [g0, g1)`` only: ``d = clamp(pos_i - pos_j + clip, 0, 2 clip)``;
    where ``condition`` (bool ``[*, rows, N]``, e.g. :func:`same_rows` of the chain id) is False the bin is ``2 clip + 1``. int64 ``[*, rows, N]``.
    The per-element statement of a relative-position (``relpos``) input embedder; the caller's one-hot / linear follow on the slab."""
    _check_rows(pos, g0, g1, "reloffset_rows")
    clip = int(clip)
    offset = pos[..., g0:g1, None] - pos[..., None, :]
    d = torch.clamp(offset + clip, min=0, max=2 * clip)
    if condition is not None:
        d = torch.where(condition, d, (2 * clip + 1) * torch.ones_like(d))
    return d.to(torch.int64)


def relpos_onehot_rows(pos, g0: int, g1: int, clip: int, condition=None, dtype=None, one_hot: Optional[Callable] = None):
    """One-hot relative-position features of rows ``[g0, g1)``: ``one_hot(reloffset_rows(...), 2 clip + 2)`` -> ``[*, rows, N, 2 clip + 2]`` in
    ``dtype`` (default float32). ``one_hot(bins_int64, n_classes)`` defaults to ``torch.nn.functional.one_hot``; an engine whose stock uses
    another binning function passes it (values are 0 / 1 either way). Only the row slab is ever built (the lazy relpos)."""
    d = reloffset_rows(pos, g0, g1, clip, condition)
    n = 2 * int(clip) + 2
    oh = torch.nn.functional.one_hot(d, n) if one_hot is None else one_hot(d, n)
    return oh.to(dtype=dtype or torch.float32)


def outer_sum_rows(a, b, g0: int, g1: int):
    """``(a[..., :, None, :] + b[..., None, :, :])[..., g0:g1, :, :]`` for two per-token projections ``[*, N, C]`` -> ``[*, rows, N, C]``."""
    _check_rows(a, g0, g1, "outer_sum_rows", dim=-2)
    return a[..., g0:g1, :].unsqueeze(-2) + b.unsqueeze(-3)


def feature_rows(x, g0: int, g1: int, device=None, *, row_dim: int = -2, non_blocking: bool = False):
    """Rows ``[g0, g1)`` (dim ``row_dim``: -2 for ``[*, N, N]``, -3 for ``[*, N, N, F]``) of a replicated pair feature, on ``device`` (the
    feature may be host-resident: a contiguous row slab is what moves). A view when no move is needed."""
    _check_rows(x, g0, g1, "feature_rows", dim=row_dim)
    slab = x.narrow(row_dim % x.dim(), int(g0), int(g1) - int(g0))
    if device is None or slab.device == torch.device(device):
        return slab
    return slab.to(device, non_blocking=non_blocking)


def pair_mask_rows(token_mask, g0: int, g1: int):
    """``(token_mask[..., :, None] * token_mask[..., None, :])[..., g0:g1, :]``."""
    _check_rows(token_mask, g0, g1, "pair_mask_rows")
    return token_mask[..., g0:g1, None] * token_mask[..., None, :]


# ================================================================================================================ block schedule
def _rows_for(what: str, N: int, channels: int, elem_bytes: int, rows: Optional[int], rows_env: str, n_max: Optional[int] = None) -> int:
    """Rows per block of the trunk schedule ``what`` through THE chooser (:func:`.shard.choose_block_rows` over ``[rows, N, channels]`` slabs):
    ``rows`` given > the row COUNT named by ``$rows_env`` (``ROWPAIR_INIT_ROWS`` / ``ROWPAIR_RECYCLE_ROWS``) > ``ROWPAIR_ROWBLK_MB`` > 512 MiB;
    the value and its source land in the schedule census as ``<what>`` / ``<what>_source``. P-invariant (no rank-local reading enters)."""
    source_env = None
    if rows is None and env_int(rows_env, 0) > 0:
        rows, source_env = env_int(rows_env, 0), f"env:{rows_env}"
    n, source = choose_block_rows(int(N), int(channels), int(elem_bytes), rows=rows, n_max=n_max)
    if source_env is not None:
        source = source.replace("given", source_env, 1)
    record_schedule(**{what: n, what + "_source": source})
    return n


def init_pair_shard(layout: Layout, rows_fn: Callable[[int, int], object], like, rows: Optional[int] = None, *,
                    channels: Optional[int] = None, out=None):
    """The pair representation BORN sharded: ``z_loc[..., i0:i1, :, :] = rows_fn(r0 + i0, r0 + i1)`` for this rank's local row blocks of ``rows``
    rows (``ROWPAIR_INIT_ROWS``; default the ``ROWPAIR_ROWBLK_MB`` budget over ``[rows, N, channels]`` fp32-equivalent transients, ``channels``
    default ``3 C`` = the block and its two addends — an engine whose relpos one-hot slab is wider passes its count). ``rows_fn(g0, g1)`` is the
    engine's z_init statement on GLOBAL rows ``[g0, g1)`` returning ``[*, g1 - g0, N, C]`` (compose :func:`outer_sum_rows`, :func:`relpos_onehot_rows`,
    :func:`feature_rows` and the engine's linears). ``like`` is any tensor ``[*, N, C]`` fixing the leading dims, ``C``, dtype and device of the
    shard (the per-token projection the outer sum reads is the natural choice). Returns the shard ``[*, R, N, C]`` (written into ``out`` when given)."""
    require_sharded(layout, "init_pair_shard", LEVER)
    R, N = layout.R, layout.N
    if like.dim() < 2 or int(like.shape[-2]) != N:
        raise RowpairRefused(f"init_pair_shard: like {tuple(like.shape)} must be [*, {N}, C]", LEVER)
    C = int(like.shape[-1])
    shape = tuple(like.shape[:-2]) + (R, N, C)
    if out is None:
        z_loc = like.new_empty(shape)
    else:
        if tuple(out.shape) != shape or out.dtype != like.dtype:
            raise RowpairRefused(f"init_pair_shard: out {tuple(out.shape)} {out.dtype} is not this rank's shard {shape} {like.dtype}", LEVER)
        z_loc = out
    rows = _rows_for("trunk_init_rows", N, channels or INIT_TRANSIENT_CHANNELS * C, like.element_size(), rows, "ROWPAIR_INIT_ROWS", n_max=layout.Rmax)
    return produce_rows_(z_loc, layout, rows_fn, op="set", block_rows=rows)


def init_grid(layout: Layout, like, rows: Optional[int] = None, *, channels: Optional[int] = None):
    """``(shape, blocks)`` of :func:`init_pair_shard` WITHOUT producing anything: the shard shape ``[*, R, N, C]`` ``like`` fixes and the row blocks
    ``[(b0, b1, g0, g1), ...]`` its :func:`.shard.produce_rows_` loop visits (the same chooser calls in the same order: :func:`_rows_for` — which
    records ``trunk_init_rows`` — then produce_rows_'s ``choose_block_rows(elems_per_row=, rows=, align=)`` + :func:`.shard.iter_row_blocks`). The
    ``recompute`` z_init placement (:meth:`ShardPark.recompute`) re-runs the init statement on exactly these blocks, so every GEMM runs at the shapes
    the parked copy was produced at (bitwise the parked bytes given deterministic kernels). P-invariant like its parts."""
    from .dist import chunk_align
    require_sharded(layout, "init_grid", LEVER)
    R, N = layout.R, layout.N
    if like.dim() < 2 or int(like.shape[-2]) != N:
        raise RowpairRefused(f"init_grid: like {tuple(like.shape)} must be [*, {N}, C]", LEVER)
    C = int(like.shape[-1])
    shape = tuple(int(d) for d in like.shape[:-2]) + (R, N, C)
    rows = _rows_for("trunk_init_rows", N, channels or INIT_TRANSIENT_CHANNELS * C, like.element_size(), rows, "ROWPAIR_INIT_ROWS", n_max=layout.Rmax)
    lead = 1
    for d in shape[:-3]:
        lead *= int(d)
    u = int(layout.align or chunk_align())
    block_rows, source = choose_block_rows(elems_per_row=max(1, lead * N * C), elem_bytes=like.element_size(), rows=rows, align=u, n_max=layout.n_max)
    record_schedule(produce_block_rows=block_rows, produce_block_source=source)
    return shape, iter_row_blocks(layout, block_rows, u)


def recycle_shard_(z_loc, zinit, update_fn: Callable[[object], object], layout: Layout, rows: Optional[int] = None):
    """The recycling statement ``z = z_init + update_fn(z_prev)`` (``update_fn`` = the engine's ``linear(layer_norm(.))``, row-local) on the shard,
    IN PLACE per row block of ``rows`` rows (``ROWPAIR_RECYCLE_ROWS`` / the ``ROWPAIR_ROWBLK_MB`` default over ``[rows, N, C]``): row i of the result
    depends on row i only, so writing block b back into ``z_loc``'s storage before block b+1 is read is exact. ``zinit`` is a :class:`ShardPark`
    (parked or resident) or a plain shard tensor. ``z_loc is None`` is cycle 0 (stock: ``z_prev = zeros_like(z_init)``): ``update_fn`` is
    evaluated on a ZEROS ROW BLOCK (identical per-element statements) and the result shard is allocated fresh — with ``z_init`` parked the device
    then holds one shard plus row blocks, never two shards. Returns the shard (the same tensor object when one was given). Runs in the CALLER's
    grad context: inference calls it under ``torch.no_grad()`` (the in-place per-block writes are not autograd-safe)."""
    require_sharded(layout, "recycle_shard_", LEVER)
    park = zinit if isinstance(zinit, ShardPark) else ShardPark(zinit, park=False, name="z_init")
    shape = tuple(park.shape)
    R, N, C = shape[-3], shape[-2], shape[-1]
    if (R, N) != (layout.R, layout.N):
        raise RowpairRefused(f"recycle_shard_: z_init shard {shape} vs layout rows R={layout.R} N={layout.N}", LEVER)
    first = z_loc is None
    if not first and tuple(z_loc.shape) != shape:
        raise RowpairRefused(f"recycle_shard_: z_loc {tuple(z_loc.shape)} vs z_init shard {shape}", LEVER)
    rows = _rows_for("trunk_recycle_rows", N, C, torch.empty((), dtype=park.dtype).element_size(), rows, "ROWPAIR_RECYCLE_ROWS", n_max=layout.Rmax)
    if first:
        z_loc = torch.empty(shape, dtype=park.dtype, device=park.device)
        nz = max([b1 - b0 for b0, b1, _g0, _g1 in iter_row_blocks(layout, rows)] or [0])       # the largest block produce_rows_ will ask for
        zeros_blk = torch.zeros(shape[:-3] + (nz, N, C), dtype=park.dtype, device=park.device)
    r0 = layout.r0

    def recycled_rows(g0: int, g1: int):                  # z_init[b] + linear(LN(z_prev[b])) — z_prev[b] read BEFORE the block is overwritten
        b0, b1 = g0 - r0, g1 - r0
        src = zeros_blk[..., : b1 - b0, :, :] if first else z_loc[..., b0:b1, :, :]
        upd = update_fn(src)
        return park.block(b0, b1) + upd

    produce_rows_(z_loc, layout, recycled_rows, op="set", block_rows=rows)
    if first:
        del zeros_blk
    return z_loc


# ================================================================================================================ parking a shard on the host
def park_strict(strict: Optional[bool] = None) -> bool:
    """``ROWPAIR_PARK_STRICT`` unless ``strict`` is given: whether a page-lock a park cannot honour is refused by name (True) or answered with a
    NAMED pageable host buffer (False, the default)."""
    return _env_flag(ENV_PARK_STRICT) if strict is None else bool(strict)


def park_pool(*, pin: bool = True, strict: Optional[bool] = None, pin_max_bytes: Optional[int] = None):
    """The private pool of a host park that shares none (:class:`ShardPark`, :class:`.heads.ZTrunkPlan`): a
    :class:`opt_core.mem.torch_hostpair.PinPool` budgeted by ``pin_max_bytes`` (None: ``ROWPAIR_PARK_PIN_MAX_GB``, unset = unbudgeted), page-locking
    when ``pin``, answering a refused page-lock (``pin_alloc_failed`` / ``pin_budget_exceeded`` / ``host_ram_insufficient``) with a pageable buffer
    whose kind the park names — or refusing by name when :func:`park_strict` (``ROWPAIR_PARK_STRICT=1`` / ``strict=True``). The ONE constructor of
    such pools: every park of the family answers a locked-memory cap with the same words."""
    from ..torch_hostpair import GiB, PinPool
    if pin_max_bytes is None:
        g = os.environ.get(ENV_PARK_PIN_MAX_GB, "").strip()
        pin_max_bytes = int(float(g) * GiB) if g else None
    return PinPool(pin_max_bytes, pin=bool(pin), pageable=not park_strict(strict), lever=LEVER)


def host_alloc(pool, t, what: str, say: Optional[Callable[[str], None]] = None, chunk: Optional[int] = None):
    """The ONE host-buffer statement of the family's parks (and of the triangle-multiplication row mirror): an exact-size
    :class:`opt_core.mem.torch_hostpair.HostChunks` copy of ``t``'s bytes from ``pool`` (:class:`opt_core.mem.torch_hostpair.PinPool`) — chunks of
    ``ROWPAIR_PARK_CHUNK_GIB`` GiB (default 8; ``chunk`` bytes when given; 0 = one allocation of the whole size), page-locked for a CUDA tensor; a
    page-lock the pool answers pageable is NAMED (``where = host_pageable:<kind>``, said through ``say``); a pool that refuses (strict) is a
    :class:`RowpairRefused` naming ``what``, the bytes and the kind. Returns ``(host, where, fallback, pinned)`` with ``host`` the (still empty)
    chunked buffer, ``where`` = ``host`` (CPU tensor) | ``host_pinned`` | ``host_pageable:<kind>`` and ``fallback`` = ``<kind>`` or None."""
    from ..torch_hostpair import HostChunks, PinRefused
    from .. import MemLeverRefused
    if isinstance(pool, HostSlabLease):                                        # ROWPAIR_HOST_SLAB=lease: the family's ONE slab, leased
        host = pool.acquire_for(t, what, say=say, chunk=chunk)
        return host, host.where, host.fallback, bool(host.pinned)
    want_pin = bool(t.is_cuda)
    nbytes = int(t.numel() * t.element_size())
    try:
        host = HostChunks(nbytes, pool, pin=want_pin, chunk=chunk, tag=what, lever=LEVER)
    except PinRefused as r:
        raise RowpairRefused(f"{what}: parking {nbytes / 2 ** 30:.2f} GiB refused ({r.kind}: {r.detail}); ROWPAIR_PARK_STRICT=1 forbids the pageable answer",
                             LEVER) from None
    except MemLeverRefused as r:
        raise RowpairRefused(f"{what}: {r.reason}", LEVER) from None
    if host.fallback is not None and say is not None:
        say(f"[park] {what}: page-locking {nbytes / 2 ** 30:.2f} GiB refused ({host.fallback}); PAGEABLE host buffer in use (named)")
    return host, host.where, host.fallback, bool(host.pinned)


def park_host(t, *, strict: Optional[bool] = None, pin_max_bytes: Optional[int] = None):
    """WHERE a u-sized host copy of ``t`` comes from — the one resolver every park of the family and the row mirror call when no pool was handed
    in: ``ROWPAIR_HOST_SLAB`` unset -> :func:`park_pool` (today's private pool, the same construction and arguments); ``lease`` -> this rank's
    :class:`HostSlabLease` (:func:`host_slab_lease`; its slab pool is built by the same :func:`park_pool` call at the first lease). Records the census
    word ``host_slab=private|lease``."""
    mode = host_slab_mode()
    record_schedule(host_slab=mode)
    if mode == "lease":
        return host_slab_lease(t.device, pin=bool(t.is_cuda), strict=strict, pin_max_bytes=pin_max_bytes)
    return park_pool(pin=bool(t.is_cuda), strict=strict, pin_max_bytes=pin_max_bytes)


# ================================================================================================================ the leased host slab
class LeasedHost(object):
    """One holder's handle on the leased slab (:class:`HostSlabLease`): the copy API of :class:`opt_core.mem.torch_hostpair.HostChunks` over the
    slab's first ``nbytes`` bytes (``store`` / ``load`` / ``store_range`` / ``load_range`` — byte copies, bit-exact), its facts (``where`` /
    ``fallback`` / ``pinned`` / ``n_chunks`` / ``pinned_bytes`` / ``alloc_bytes`` — the SLAB's, so the census vocabulary of the parks is unchanged:
    ``host_pinned`` | ``host_pageable:<kind>`` | ``host``), and ``release()`` = hand the slab back (the chunks stay allocated, owned by the lease; a
    CUDA event is recorded so the next holder waits for this holder's in-flight copies). After ``release()`` every copy is refused by name."""

    __slots__ = ("lease", "holder", "nbytes", "slab", "device", "_live", "leased")

    def __init__(self, lease: "HostSlabLease", holder: str, nbytes: int, slab, device):
        self.lease, self.holder, self.nbytes, self.slab, self.device, self._live, self.leased = lease, str(holder), int(nbytes), slab, device, True, True

    # --- the slab's facts (what the holder's census words quote) ------------------------------------------------------------------
    @property
    def where(self) -> str:
        return self.slab.where

    @property
    def fallback(self):
        return self.slab.fallback

    @property
    def pinned(self) -> bool:
        return bool(self.slab.pinned)

    @property
    def n_chunks(self) -> int:
        return int(self.slab.n_chunks) if self._live else 0

    @property
    def pinned_bytes(self) -> int:
        return int(self.slab.pinned_bytes) if self._live else 0

    @property
    def alloc_bytes(self) -> int:
        return int(self.slab.alloc_bytes) if self._live else 0

    @property
    def sizes(self):
        return list(self.slab.sizes)

    @property
    def released(self) -> bool:
        return not self._live

    def _check(self, what: str, offset: int, n: int) -> None:
        if not self._live:
            raise RowpairRefused(f"host slab lease: {self.holder} {what} after release (the slab was handed back)", LEVER)
        if int(offset) < 0 or int(n) < 0 or int(offset) + int(n) > self.nbytes:
            raise RowpairRefused(f"host slab lease: {self.holder} {what} byte range [{offset}, {int(offset) + int(n)}) outside its lease [0, {self.nbytes})", LEVER)

    # --- copies (HostChunks' API) --------------------------------------------------------------------------------------------------
    def store_range(self, src, offset: int = 0, non_blocking: bool = True) -> None:
        self._check("store_range", offset, int(src.numel()) * int(src.element_size()))
        self.slab.store_range(src, offset, non_blocking)

    def load_range(self, dst, offset: int = 0, non_blocking: bool = True):
        self._check("load_range", offset, int(dst.numel()) * int(dst.element_size()))
        return self.slab.load_range(dst, offset, non_blocking)

    def store(self, t, non_blocking: bool = True) -> None:
        n = int(t.numel()) * int(t.element_size())
        if n != self.nbytes:
            raise RowpairRefused(f"host slab lease: {self.holder} store of {n} B into a {self.nbytes} B lease", LEVER)
        self.store_range(t, 0, non_blocking)

    def load(self, t, non_blocking: bool = True):
        n = int(t.numel()) * int(t.element_size())
        if n != self.nbytes:
            raise RowpairRefused(f"host slab lease: {self.holder} load of a {self.nbytes} B lease into {n} B", LEVER)
        return self.load_range(t, 0, non_blocking)

    def release(self) -> None:
        """Hand the slab back to the lease (idempotent). The lease records the hand-over event; the chunks stay allocated."""
        if self._live:
            self._live = False
            self.lease._release(self)


def _cuda_event_factory(device):
    """The default hand-over event of a lease on a CUDA device: ``record()`` puts a :class:`torch.cuda.Event` on the device's current stream,
    ``wait()`` blocks the host until it completed. None for a CPU slab (host copies are synchronous)."""
    if device is None or torch.device(device).type != "cuda" or not torch.cuda.is_available():
        return None

    class _Ev(object):
        __slots__ = ("ev", "dev")

        def __init__(self, dev):
            self.dev = torch.device(dev)
            self.ev = torch.cuda.Event()
            self.ev.record(torch.cuda.current_stream(self.dev))

        def wait(self):
            self.ev.synchronize()

    return lambda: _Ev(device)


class HostSlabLease(object):
    """ONE u-sized host slab per process (per rank) handed to ONE HOLDER AT A TIME — the ``ROWPAIR_HOST_SLAB=lease`` placement of the family's
    host copies that are never needed together: the z_init park (or none: ``ROWPAIR_PARK_ZINIT=recompute``), the template stage's z park
    (:class:`.template.ParkedStorage`), the triangle-multiplication A-block row mirror (:func:`.trimul.host_mirror`, per multi-pass call), the
    confidence stage's z_trunk park (:class:`.heads.ZTrunkPlan`). :func:`host_alloc` on a lease = :meth:`acquire_for`.

    * ``acquire_for(t, holder)``: the slab is FREE -> wait for the previous holder's hand-over event (its in-flight D2H / H2D copies), grow the slab
      when ``t``'s bytes exceed it (an exact-size :class:`opt_core.mem.torch_hostpair.HostChunks` from the lease's pool — :func:`park_pool`, so
      ``ROWPAIR_PARK_CHUNK_GIB`` / ``ROWPAIR_PARK_PIN_MAX_GB`` / ``ROWPAIR_PARK_STRICT`` mean what they mean for a private park; the old chunks go back
      to torch's caching host allocator first), and return a :class:`LeasedHost` over its first ``nbytes`` — census ``host_slab=lease
      host_slab_gib=<slab GiB> host_slab_holder=<holder> host_slab_leases=<n>``.
    * the slab is HELD -> the SECOND concurrent holder gets a private exact-size buffer from a private :func:`park_pool` (today's statement; its
      ``release()`` returns the chunks to torch's cache as today) and the overlap is NAMED: census ``host_slab_second=<holder>+<second>
      host_slab_second_gib=<its GiB> host_slab_seconds=<n>``, a ``[park] host slab: …`` line through ``say``; under ``ROWPAIR_HOST_SLAB_STRICT=1``
      (``strict_second=True``) it is refused by name instead (:class:`RowpairRefused` naming both holders).
    * ``release`` (through :meth:`LeasedHost.release`): the slab becomes free; ``event_factory()`` (default: a CUDA event on the holder's device's
      current stream; None on CPU) is recorded and the NEXT acquire waits on it before handing the bytes over.
    * :meth:`drop`: return the slab's chunks to the pool (a free slab only; :func:`release_free_host_slabs` — the pinned-pool shrink calls it).
    Byte copies only: placement, never arithmetic."""

    def __init__(self, *, pin: bool = True, pool=None, strict: Optional[bool] = None, pin_max_bytes: Optional[int] = None,
                 strict_second: Optional[bool] = None, event_factory=None, device=None, name: str = "host_slab"):
        self.name, self.pin, self.device = str(name), bool(pin), device
        self._pool, self._pool_args = pool, dict(strict=strict, pin_max_bytes=pin_max_bytes)
        self._private_pool = None
        self.strict_second = _env_flag(ENV_HOST_SLAB_STRICT) if strict_second is None else bool(strict_second)
        self._event_factory = event_factory if event_factory is not None else _cuda_event_factory(device)
        self._pending = None                                                   # the last holder's hand-over event (waited on by the next acquire)
        self.slab, self.holder, self.handle = None, None, None
        self.n_leases, self.n_second, self.n_grow, self.n_waits = 0, 0, 0, 0
        self._said_second = set()                                               # (holder, second) pairs whose PRIVATE-buffer line was said (one line per pair)
        self.second_peak_bytes, self.slab_peak_bytes = 0, 0
        self.events: list = []                                                 # (what, holder, bytes) — the lease's own transcript (tests, records)
        self._lock = threading.Lock()

    # --- facts -------------------------------------------------------------------------------------------------------------------
    @property
    def nbytes(self) -> int:
        return int(self.slab.nbytes) if self.slab is not None else 0

    @property
    def pinned_bytes(self) -> int:
        if self.slab is None:
            return 0
        return int(self.slab.pinned_bytes if self.slab.pinned else self.slab.alloc_bytes)

    @property
    def free(self) -> bool:
        return self.holder is None

    def record(self) -> Dict[str, object]:
        return {"name": self.name, "slab_gib": round(self.nbytes / 2 ** 30, 4), "pinned_gib": round(self.pinned_bytes / 2 ** 30, 4), "holder": self.holder,
                "leases": self.n_leases, "seconds": self.n_second, "grows": self.n_grow, "waits": self.n_waits,
                "second_peak_gib": round(self.second_peak_bytes / 2 ** 30, 4), "where": (self.slab.where if self.slab is not None else None)}

    def _pool_for_slab(self):
        if self._pool is None:
            self._pool = park_pool(pin=self.pin, **self._pool_args)
        return self._pool

    # --- the statement -------------------------------------------------------------------------------------------------------------
    def acquire_for(self, t, holder: str, *, say: Optional[Callable[[str], None]] = None, chunk: Optional[int] = None):
        """Lease the slab to ``holder`` for ``t``'s bytes (see the class doc); returns a :class:`LeasedHost` — or, while another holder has it, a
        private :class:`opt_core.mem.torch_hostpair.HostChunks` (named) / a refusal by name (strict)."""
        from ..torch_hostpair import HostChunks, PinRefused
        from .. import MemLeverRefused
        nbytes = int(t.numel()) * int(t.element_size())
        holder = str(holder)
        with self._lock:
            if self.holder is not None:                                        # a SECOND concurrent holder: named private buffer, or refused
                if self.strict_second:
                    raise RowpairRefused(f"host_slab_second: {holder} asked for the host slab ({nbytes / 2 ** 30:.2f} GiB) while {self.holder} holds it; "
                                         f"{ENV_HOST_SLAB_STRICT}=1 refuses a second pinned buffer (unset it: a NAMED private buffer is taken instead)", LEVER)
                if self._private_pool is None:
                    self._private_pool = park_pool(pin=bool(t.is_cuda), **self._pool_args)
                try:
                    host = HostChunks(nbytes, self._private_pool, pin=bool(t.is_cuda), chunk=chunk, tag=holder, lever=LEVER)
                except PinRefused as r:
                    raise RowpairRefused(f"{holder}: parking {nbytes / 2 ** 30:.2f} GiB refused ({r.kind}: {r.detail}); ROWPAIR_PARK_STRICT=1 forbids the pageable answer",
                                         LEVER) from None
                except MemLeverRefused as r:
                    raise RowpairRefused(f"{holder}: {r.reason}", LEVER) from None
                self.n_second += 1
                self.second_peak_bytes = max(self.second_peak_bytes, nbytes)
                self.events.append(("second", f"{self.holder}+{holder}", nbytes))
                record_schedule(host_slab_second=f"{census_token(self.holder)}+{census_token(holder)}", host_slab_second_gib=round(nbytes / 2 ** 30, 3),
                                host_slab_seconds=self.n_second)                # census values are single tokens (a kit's LEVER-line writer refuses blanks)
                pair = (str(self.holder), holder)
                first_pair = pair not in self._said_second                     # one line per (holder, second) pair; the census counts every one (host_slab_seconds)
                self._said_second.add(pair)
                if say is not None and first_pair:
                    say(f"[park] host slab: {holder} needs {nbytes / 2 ** 30:.2f} GiB while {self.holder} holds the slab — a PRIVATE {host.where} buffer "
                        f"({host.n_chunks} chunks) is taken (host_slab_second={self.holder}+{holder}; second #{self.n_second}; said once per pair)")
                if host.fallback is not None and say is not None:
                    say(f"[park] {holder}: page-locking {nbytes / 2 ** 30:.2f} GiB refused ({host.fallback}); PAGEABLE host buffer in use (named)")
                return host
            if self._pending is not None:                                      # hand-over: the previous holder's copies must have landed
                pending, self._pending = self._pending, None
                self.n_waits += 1
                self.events.append(("wait", holder, 0))
                pending.wait()
            if self.slab is None or self.slab.nbytes < nbytes or self.slab.released:
                old = self.slab
                if old is not None and not old.released:
                    old.release()                                              # the smaller slab's chunks return to torch's cache before the larger comes
                    self.n_grow += 1
                    self.events.append(("grow", holder, nbytes))
                try:
                    self.slab = HostChunks(nbytes, self._pool_for_slab(), pin=self.pin and bool(t.is_cuda), chunk=chunk, tag=self.name, lever=LEVER)
                except PinRefused as r:
                    self.slab = None
                    raise RowpairRefused(f"{holder}: leasing {nbytes / 2 ** 30:.2f} GiB refused ({r.kind}: {r.detail}); ROWPAIR_PARK_STRICT=1 forbids the pageable answer",
                                         LEVER) from None
                except MemLeverRefused as r:
                    self.slab = None
                    raise RowpairRefused(f"{holder}: {r.reason}", LEVER) from None
                self.slab_peak_bytes = max(self.slab_peak_bytes, int(self.slab.nbytes))
                if self.slab.fallback is not None and say is not None:
                    say(f"[park] {self.name}: page-locking {nbytes / 2 ** 30:.2f} GiB refused ({self.slab.fallback}); PAGEABLE host slab in use (named)")
                if say is not None:
                    say(f"[park] host slab: {nbytes / 2 ** 30:.2f} GiB {self.slab.where} in {self.slab.n_chunks} chunks (pinned_gib={self.slab.pinned_bytes / 2 ** 30:.2f}) "
                        f"leased first to {holder}")
            self.holder = holder
            self.handle = LeasedHost(self, holder, nbytes, self.slab, t.device)
            self.n_leases += 1
            self.events.append(("lease", holder, nbytes))
            record_schedule(host_slab="lease", host_slab_gib=round(self.slab.nbytes / 2 ** 30, 3), host_slab_pinned_gib=round(self.pinned_bytes / 2 ** 30, 3),
                            host_slab_holder=census_token(holder), host_slab_leases=self.n_leases)
            return self.handle

    def _release(self, handle: LeasedHost) -> None:
        with self._lock:
            if self.handle is not handle:                                      # a stale handle (the lease moved on): nothing to hand back
                return
            self.events.append(("release", handle.holder, handle.nbytes))
            self.holder, self.handle = None, None
            if self._event_factory is not None:
                self._pending = self._event_factory()

    def drop(self) -> int:
        """Return a FREE slab's chunks to the pool (torch's caching host allocator; the pinned-pool shrink then hands them to the OS). Returns the
        bytes dropped (0 while held or when empty). The next acquire allocates afresh."""
        with self._lock:
            if self.holder is not None or self.slab is None or self.slab.released:
                return 0
            if self._pending is not None:
                pending, self._pending = self._pending, None
                pending.wait()
            n = self.pinned_bytes
            self.slab.release()
            self.slab = None
            self.events.append(("drop", "-", n))
            return int(n)


_LEASES: Dict[Tuple[str, int], HostSlabLease] = {}
_LEASE_LOCK = threading.Lock()


def host_slab_lease(device=None, *, pin: bool = True, strict: Optional[bool] = None, pin_max_bytes: Optional[int] = None) -> HostSlabLease:
    """This rank's :class:`HostSlabLease` for ``device`` (created on first use; one per ``(device, rank)`` — one per process in a P-process run,
    one per rank-thread under :func:`opt_core.testing.run_ranks`). ``strict`` / ``pin_max_bytes`` reach the slab's pool at creation only."""
    key = (str(device), int(comm().rank))
    with _LEASE_LOCK:
        lease = _LEASES.get(key)
        if lease is None:
            lease = _LEASES[key] = HostSlabLease(pin=pin, strict=strict, pin_max_bytes=pin_max_bytes, device=device, name="host_slab")
        return lease


def host_slab_leases() -> Dict[Tuple[str, int], HostSlabLease]:
    """A copy of the lease registry (``{(device, rank): lease}``)."""
    with _LEASE_LOCK:
        return dict(_LEASES)


def host_slab_bytes() -> Tuple[int, int, int]:
    """``(leases with a slab, held, pinned bytes)`` over this process's leases (the pinned-pool shrink line quotes it)."""
    with _LEASE_LOCK:
        live = [l for l in _LEASES.values() if l.slab is not None]
        return len(live), sum(1 for l in live if not l.free), sum(int(l.pinned_bytes) for l in live)


def release_free_host_slabs() -> int:
    """Drop every FREE leased slab of this process (:meth:`HostSlabLease.drop`); returns the bytes returned to torch's cache. Held slabs stay."""
    with _LEASE_LOCK:
        leases = list(_LEASES.values())
    return sum(l.drop() for l in leases)


def release_host_slabs() -> int:
    """Forget every lease of this process (tests / between requests): free slabs are dropped, the registry cleared; returns how many leases went."""
    with _LEASE_LOCK:
        leases = list(_LEASES.values())
        _LEASES.clear()
    for l in leases:
        l.drop()
    return len(leases)


HOST_TRIM_STATS = {"calls": 0, "returned_bytes": 0, "seconds": 0.0, "skipped": 0}


def host_trim_mode() -> str:
    """``ROWPAIR_HOST_TRIM``: unset / ``auto`` -> ``auto`` (trim iff :func:`host_slab_mode` is ``lease``); ``0`` / off words -> ``0`` (never);
    ``1`` / on words -> ``1`` (always). Any other word is refused by name."""
    v = os.environ.get(ENV_HOST_TRIM, "").strip().lower()
    if v in ("", "auto"):
        return "auto"
    if v in ("0", "off", "no", "false"):
        return "0"
    if v in ("1", "on", "yes", "true"):
        return "1"
    raise RowpairRefused(f"{ENV_HOST_TRIM}={v!r}: one of {HOST_TRIM_MODES} is required", LEVER)


def host_trim(where: str, log: Optional[Callable[[str], None]] = None, *, force: Optional[bool] = None) -> Optional[Dict[str, object]]:
    """Return torch's CACHED-FREE pinned host blocks to the OS at a stage seam: one :func:`opt_core.mem.torch_hostpair.host_cache_trim`
    call (``torch._C._host_emptyCache``: live pinned tensors untouched, no tensor value can change). Gated: ``force`` given wins; else
    :func:`host_trim_mode` — ``auto`` trims only under ``ROWPAIR_HOST_SLAB=lease`` (the line whose pinned host is meant to be 1 u: the template
    stage's PRIVATE second buffers — the c=64 tri-mult row mirror of a multi-pass template pair stack, parked u slabs — were handed back to torch's
    cache at stage exit and would otherwise stay page-locked through the MSA module, the pairformer and the sampler until :func:`.heads.pinned_shrink`),
    ``0`` never, ``1`` always. When it does not run NOTHING is recorded (default census byte-identical). When it runs: census ``host_trim=<where>:<GiB
    returned>`` (``<where>:unavailable`` on a torch without the binding — said, never silent), ``host_trim_s``, ``host_trim_calls``; one ``[park] host
    trim`` line through ``log``. Cost: the trim itself (~0.1 s per 10 GiB) plus the LATER re-pin of whoever page-locks that much again (the next
    park under private pools; nobody under the lease, whose slab is live and therefore kept). Returns the trim record (None when gated off)."""
    if force is None:
        mode = host_trim_mode()
        run = (mode == "1") or (mode == "auto" and host_slab_mode() == "lease")
    else:
        run = bool(force)
    if not run:
        HOST_TRIM_STATS["skipped"] += 1
        return None
    from ..torch_hostpair import GiB, host_cache_trim
    t0 = time.time()
    rec = dict(host_cache_trim())
    dt = time.time() - t0
    rb, ra = rec.get("reserved_before"), rec.get("reserved_after")
    returned = (int(rb) - int(ra)) if isinstance(rb, int) and isinstance(ra, int) and rb >= 0 and ra >= 0 else None
    HOST_TRIM_STATS["calls"] += 1
    HOST_TRIM_STATS["seconds"] += dt
    if returned is not None:
        HOST_TRIM_STATS["returned_bytes"] += max(0, returned)
    if not rec.get("available", False):
        word = f"{where}:unavailable"
    elif returned is None:                                                            # the binding ran but this torch has no host allocator statistics:
        rss_b, rss_a = rec.get("rss_before"), rec.get("rss_after")                    # quote the resident-set drop instead (pinned pages count in VmRSS)
        drop = (float(rss_b) - float(rss_a)) if isinstance(rss_b, (int, float)) and isinstance(rss_a, (int, float)) else None
        word = f"{where}:rss-{drop:.3f}" if drop is not None else f"{where}:done"
    else:
        word = f"{where}:{returned / GiB:.3f}"
    rec.update({"where": str(where), "word": word, "seconds": round(dt, 3), "returned_bytes": returned})
    record_schedule(host_trim=word, host_trim_s=round(HOST_TRIM_STATS["seconds"], 3), host_trim_calls=HOST_TRIM_STATS["calls"])
    if log is not None:
        rss_b, rss_a = rec.get("rss_before"), rec.get("rss_after")
        log(f"[park] host trim at {where}: cached-free pinned host returned "
            f"{('%.2f GiB' % (returned / GiB)) if returned is not None else 'n/a'} (reserved "
            f"{('%.2f' % (rb / GiB)) if isinstance(rb, int) and rb >= 0 else 'n/a'} -> {('%.2f' % (ra / GiB)) if isinstance(ra, int) and ra >= 0 else 'n/a'} GiB; "
            f"VmRSS {rss_b} -> {rss_a} GiB) in {dt:.2f}s call={rec.get('call')} ok={rec.get('ok')}"
            + (f" error={rec.get('error')}" if rec.get("error") else ""))
    return rec


class DevicePeak(object):
    """The device high-water of ONE stage without touching torch's process-wide peak counters: at construction it reads
    ``memory_allocated`` / ``max_memory_allocated`` of ``device`` and (``sample_hz`` > 0, CUDA) starts a daemon thread sampling ``memory_allocated``
    at that rate; :meth:`mark` samples at named points; :meth:`close` stops the sampler and says ``{"peak_gib", "kind", "at", "entry_gib", "exit_gib",
    "marks", "samples"}`` where ``kind`` is ``exact`` when the process high-water ROSE during the stage (``peak_gib`` = the new ``max_memory_allocated``:
    the stage set it) or ``sampled`` (the high-water predates the stage: ``peak_gib`` = the largest sampled / marked value, a lower bound within one
    sampling period of the stage's own peak; ``at`` names the mark or ``sampler``). A CPU device answers zeros with ``kind="cpu"``."""

    __slots__ = ("device", "cuda", "entry", "hw0", "best", "at", "marks", "samples", "_stop", "_thread", "_lock", "cur", "phases")

    def __init__(self, device, sample_hz: float = 20.0):
        import threading
        self.device = device
        self.cuda = bool(getattr(device, "type", None) == "cuda" and torch.cuda.is_available())
        self.entry = self._alloc()
        self.hw0 = int(torch.cuda.max_memory_allocated(self.device)) if self.cuda else 0
        self.best, self.at, self.marks, self.samples = self.entry, "entry", 0, 0
        self.cur, self.phases = "entry", {}                                   # phase(name): samples / marks are also folded into a per-phase maximum
        self._lock = threading.Lock()
        self._stop, self._thread = None, None
        if self.cuda and sample_hz and float(sample_hz) > 0:
            self._stop = threading.Event()
            period = 1.0 / float(sample_hz)

            def run():
                while not self._stop.wait(period):
                    v = self._alloc()
                    with self._lock:
                        self.samples += 1
                        if v > self.best:
                            self.best, self.at = v, f"sampler@{self.cur}"
                        if v > self.phases.get(self.cur, -1):
                            self.phases[self.cur] = v
            self._thread = threading.Thread(target=run, name="rowpair-devicepeak", daemon=True)
            self._thread.start()

    def _alloc(self) -> int:
        return int(torch.cuda.memory_allocated(self.device)) if self.cuda else 0

    def mark(self, name: str, extra_bytes: int = 0) -> int:
        """Sample now (+ ``extra_bytes`` the caller knows are about to be / were just allocated inside a callee); returns the sampled bytes."""
        v = self._alloc() + int(extra_bytes)
        with self._lock:
            self.marks += 1
            if v > self.best:
                self.best, self.at = v, str(name)
            if v > self.phases.get(self.cur, -1):
                self.phases[self.cur] = v
        return v

    def phase(self, name: str) -> None:
        """Name the sub-phase the following samples / marks belong to (:meth:`close` reports ``phases`` = {name: max GiB})."""
        v = self._alloc()
        with self._lock:
            self.cur = str(name)
            if v > self.phases.get(self.cur, -1):
                self.phases[self.cur] = v

    def close(self, name: str = "exit") -> Dict[str, object]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None
        now = self.mark(name)
        hw1 = int(torch.cuda.max_memory_allocated(self.device)) if self.cuda else 0
        g = float(2 ** 30)
        if not self.cuda:
            return {"peak_gib": 0.0, "kind": "cpu", "at": self.at, "entry_gib": 0.0, "exit_gib": 0.0, "marks": self.marks, "samples": 0}
        base = {"entry_gib": round(self.entry / g, 3), "exit_gib": round(now / g, 3), "marks": self.marks, "samples": self.samples,
                "process_high_water_gib": round(hw1 / g, 3), "sampled_peak_gib": round(self.best / g, 3), "sampled_at": self.at,
                "phases": {k: round(v / g, 3) for k, v in self.phases.items()}}
        if hw1 > self.hw0:
            return dict(base, peak_gib=round(hw1 / g, 3), kind="exact", at="allocator_high_water")
        return dict(base, peak_gib=round(self.best / g, 3), kind="sampled", at=self.at)


def park_rows_offset(shape: Sequence[int], lead_index: int, i0: int) -> int:
    """Byte offset of row ``i0`` (dim -3) of leading index ``lead_index`` inside the flat bytes of a contiguous ``[*, R, N, C]`` tensor of ``shape``
    (element size folded in by the caller: multiply by it)."""
    R, N, C = int(shape[-3]), int(shape[-2]), int(shape[-1])
    return (int(lead_index) * R + int(i0)) * N * C


def load_rows(host, shape: Sequence[int], elem_size: int, i0: int, i1: int, stage, non_blocking: bool = False):
    """Fill the contiguous device tensor ``stage`` ``[*, i1 - i0, N, C]`` with rows ``i0:i1`` of the ``[*, R, N, C]`` tensor of ``shape`` whose bytes
    ``host`` (:class:`opt_core.mem.torch_hostpair.HostChunks`) holds: one byte-range copy per leading index (one in all for a ``[1, R, N, C]`` / ``[R, N, C]``
    shard). Returns ``stage``."""
    lead = 1
    for d in tuple(shape)[:-3]:
        lead *= int(d)
    rows = int(i1) - int(i0)
    flat = stage.reshape((lead, rows) + tuple(int(d) for d in tuple(shape)[-2:])) if lead != 1 else None
    for li in range(lead):
        dst = stage if lead == 1 else flat[li]
        host.load_range(dst, park_rows_offset(shape, li, i0) * int(elem_size), non_blocking=non_blocking)
    return stage


def store_rows(host, shape: Sequence[int], elem_size: int, i0: int, i1: int, src, non_blocking: bool = False):
    """The inverse of :func:`load_rows`: write the contiguous device (or host) tensor ``src`` ``[*, i1 - i0, N, C]`` into rows ``i0:i1`` of the
    ``[*, R, N, C]`` tensor of ``shape`` whose bytes ``host`` (:class:`opt_core.mem.torch_hostpair.HostChunks` / :class:`LeasedHost`) holds: one
    byte-range copy per leading index. Byte copies only (bit-exact). With ``non_blocking=False`` (the default) the call returns after the bytes
    landed (a CUDA source is synchronized), so the caller may free ``src`` at once. Returns ``src``. (The z-parked template order
    writes the closed z row blocks back into the park with it.)"""
    lead = 1
    for d in tuple(shape)[:-3]:
        lead *= int(d)
    rows = int(i1) - int(i0)
    if int(src.shape[-3]) != rows or tuple(int(d) for d in tuple(src.shape)[-2:]) != tuple(int(d) for d in tuple(shape)[-2:]):
        raise RowpairRefused(f"store_rows: src {tuple(src.shape)} is not rows {i0}:{i1} of a {tuple(shape)} tensor", LEVER)
    flat = src.reshape((lead, rows) + tuple(int(d) for d in tuple(shape)[-2:])) if lead != 1 else None
    for li in range(lead):
        blk = src if lead == 1 else flat[li]
        host.store_range(blk, park_rows_offset(shape, li, i0) * int(elem_size), non_blocking=bool(non_blocking))
    if not non_blocking and getattr(src, "is_cuda", False):
        torch.cuda.synchronize()
    return src


class ShardPark(object):
    """A row shard (``[*, R, N, C]``, or any tensor whose dim -3 is the row dim) held either RESIDENT on its device (``park=False``) or PARKED in
    host memory (``park=True``; ``park=None`` reads ``ROWPAIR_PARK_ZINIT``), serving contiguous row blocks to the compute device. Parking is a
    GPU mechanism (a CPU tensor stays resident: the pinned / pageable / strict branches below apply to CUDA tensors only). Parking a CUDA
    tensor: a host buffer from the package's pinned pool (:class:`opt_core.mem.torch_hostpair.PinPool`; ``pool=`` to share one, else a private
    pool budgeted by ``ROWPAIR_PARK_PIN_MAX_GB``), one D2H copy, then the tensor's DEVICE STORAGE IS RELEASED (``storage.resize_(0)`` through
    :func:`.shard.release_storage_` when the tensor owns its whole storage — every view the caller still holds becomes empty; ``record()["device_release"]``
    names the outcome) so the shard's bytes return to the allocator at park time, not when the last reference dies. A page-lock the pool cannot
    honour (``pin_alloc_failed`` under a locked-memory cap, ``pin_budget_exceeded``, ``host_ram_insufficient``) is answered with a PAGEABLE host
    buffer that is named (``where = "host_pageable:<kind>"`` in the record and the census) — or refused by name under ``ROWPAIR_PARK_STRICT=1`` /
    ``strict=True``. A CPU tensor parks into a plain host copy only with ``force_host=True`` (else it stays resident: parking host to host frees
    nothing). ``block(i0, i1)`` -> a device tensor ``[*, i1 - i0, N, C]`` (a view when resident, a fresh staging copy when parked); ``full()`` ->
    the whole shard on the device; ``release()`` drops host and device references (the pool's bytes return to its budget)."""

    __slots__ = ("name", "park", "shape", "dtype", "device", "host", "dev", "where", "fallback", "device_release", "pinned", "nbytes", "park_s",
                 "_pool", "_own_pool", "_log", "mode", "_rows_fn", "_grid", "recompute_s", "n_recompute")

    def __init__(self, t, park: Optional[bool] = None, *, name: str = "z_init", pool=None, force_host: bool = False, release_device: bool = True,
                 strict: Optional[bool] = None, pin_max_bytes: Optional[int] = None, log: Optional[Callable[[str], None]] = None):
        self.name = str(name)
        self.park = _env_flag(ENV_PARK_ZINIT) if park is None else bool(park)
        self.shape, self.dtype, self.device = tuple(t.shape), t.dtype, t.device
        self.nbytes = int(t.numel()) * int(t.element_size())
        self.host, self.dev = None, None
        self.where, self.fallback, self.device_release, self.pinned, self.park_s = "device", None, None, False, 0.0
        self._pool, self._own_pool, self._log = None, False, log
        self.mode, self._rows_fn, self._grid, self.recompute_s, self.n_recompute = "resident", None, None, 0.0, 0
        if t.dim() < 3:
            raise RowpairRefused(f"ShardPark({self.name}): a shard [*, R, N, C] is required, got shape {self.shape}", LEVER)
        if not (self.park and (t.is_cuda or force_host)):
            self.park = False
            self.dev = t
            return
        self.mode = "parked"
        if not t.is_contiguous():
            raise RowpairRefused(f"ShardPark({self.name}): only a contiguous shard is parked (strides {tuple(t.stride())})", LEVER)
        strict = park_strict(strict)
        t0 = time.time()
        self.host = self._host_buffer(t, pool, strict, pin_max_bytes)
        self.host.store(t)
        if t.is_cuda:
            torch.cuda.synchronize(t.device)
        self.park_s = round(time.time() - t0, 3)
        self._say(f"[park] {self.name} {self.shape} {self.nbytes / 2 ** 30:.2f} GiB -> {self.where} host in {self.park_s:.1f}s "
                  f"chunks={self.host.n_chunks} pinned_gib={self.host.pinned_bytes / 2 ** 30:.2f}")
        if t.is_cuda and release_device:
            self._release_device(t)
        elif t.is_cuda:
            self.device_release = "kept"

    @classmethod
    def recompute(cls, layout: Layout, rows_fn: Callable[[int, int], object], like, *, rows: Optional[int] = None, channels: Optional[int] = None,
                  name: str = "z_init", log: Optional[Callable[[str], None]] = None) -> "ShardPark":
        """The COPY-FREE placement (``ROWPAIR_PARK_ZINIT=recompute``): no z_init tensor exists anywhere — neither on the device nor on the
        host. ``block(i0, i1)`` re-runs ``rows_fn`` (the engine's z_init statement on GLOBAL rows, :func:`init_pair_shard`'s callable) on the blocks
        of the INIT GRID (:func:`init_grid`: the row blocks :func:`init_pair_shard` would have produced the shard with) that intersect ``[i0, i1)``
        and returns the rows (one block: a slice of it; several: their concatenation) in ``like``'s dtype — the bytes the parked copy would have
        served, bitwise, given deterministic kernels (same statements at the same shapes; the dtype cast is the one ``produce_rows_``'s ``copy_``
        applies). ``full()`` = every block. Costs the init statement once per served block range (+ the partial init blocks at a range's edges);
        holds zero bytes between calls. ``release()`` drops the callable. Census: ``where = recompute``, ``<key>_release = not_materialized``."""
        self = cls.__new__(cls)
        self.name = str(name)
        shape, grid = init_grid(layout, like, rows, channels=channels)
        self.park, self.mode = False, ZINIT_RECOMPUTE
        self.shape, self.dtype, self.device = tuple(shape), like.dtype, like.device
        self.nbytes = 1
        for d in self.shape:
            self.nbytes *= int(d)
        self.nbytes *= int(like.element_size())
        self.host, self.dev = None, None
        self.where, self.fallback, self.device_release, self.pinned, self.park_s = ZINIT_RECOMPUTE, None, "not_materialized", False, 0.0
        self._pool, self._own_pool, self._log = None, False, log
        self._rows_fn, self._grid, self.recompute_s, self.n_recompute = rows_fn, list(grid), 0.0, 0
        self._say(f"[park] {self.name} {self.shape} {self.nbytes / 2 ** 30:.2f} GiB -> recompute (no copy: a row block re-runs the init statement on "
                  f"{len(self._grid)} init-grid blocks of {self._grid[0][1] - self._grid[0][0] if self._grid else 0} rows)")
        return self

    # --- internals ---------------------------------------------------------------------------------------------------------------
    def _say(self, msg: str) -> None:
        if self._log is not None:
            self._log(msg)

    def _host_buffer(self, t, pool, strict: bool, pin_max_bytes: Optional[int]):
        if pool is None:
            pool = park_host(t, strict=strict, pin_max_bytes=pin_max_bytes)     # ROWPAIR_HOST_SLAB unset: park_pool(pin=, strict=, pin_max_bytes=) as before
            self._own_pool = True
        self._pool = pool
        host, self.where, self.fallback, self.pinned = host_alloc(pool, t, f"ShardPark({self.name})", self._say)
        return host

    def _recomputed(self, i0: int, i1: int):
        """Rows ``[i0, i1)`` re-produced from the init statement on the init grid (see :meth:`recompute`)."""
        if self._rows_fn is None:
            raise RowpairRefused(f"ShardPark({self.name}).block: released", LEVER)
        t0 = time.time()
        R, N, C = int(self.shape[-3]), int(self.shape[-2]), int(self.shape[-1])
        lead = tuple(self.shape[:-3])
        pieces = []
        for b0, b1, g0, g1 in self._grid:
            if b1 <= i0 or b0 >= i1:
                continue
            y = self._rows_fn(g0, g1)
            if tuple(int(d) for d in y.shape[-3:]) != (b1 - b0, N, C) or int(y.numel()) != (b1 - b0) * N * C * max(1, _numel(lead)):
                raise RowpairRefused(f"ShardPark({self.name}).block: rows_fn({g0}, {g1}) returned {tuple(y.shape)}, expected [*{lead}, {b1 - b0}, {N}, {C}]", LEVER)
            y = y.reshape(lead + (b1 - b0, N, C))
            if y.dtype != self.dtype:
                y = y.to(self.dtype)                                           # produce_rows_'s dst.copy_(y) cast, the one the parked bytes went through
            lo, hi = max(int(i0), b0) - b0, min(int(i1), b1) - b0
            pieces.append(y if (lo, hi) == (0, b1 - b0) else y[..., lo:hi, :, :])
            del y
        if not pieces:
            raise RowpairRefused(f"ShardPark({self.name}).block({i0}, {i1}): no init-grid block covers it (grid of {len(self._grid)} blocks over {R} rows)", LEVER)
        out = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-3)
        self.n_recompute += 1
        self.recompute_s += time.time() - t0
        return out

    def _release_device(self, t) -> None:
        before = int(torch.cuda.memory_allocated(t.device))
        if not owns_whole_storage(t):
            self.device_release = "not_released:shared_storage"
        elif not storage_resizable(t):
            self.device_release = "not_released:storage_not_resizable"      # foreign memory (a DLPack view of another framework's buffer)
        else:
            release_storage_(t)
            self.device_release = "storage_resized_to_0"
        torch.cuda.empty_cache()
        after = int(torch.cuda.memory_allocated(t.device))
        self._say(f"[park] {self.name} device copy {self.device_release}: memory_allocated {before / 2 ** 30:.2f} -> {after / 2 ** 30:.2f} GiB "
                  f"(delta {(after - before) / 2 ** 30:+.2f} GiB), reserved {torch.cuda.memory_reserved(t.device) / 2 ** 30:.2f} GiB")

    # --- service -----------------------------------------------------------------------------------------------------------------
    def block(self, i0: int, i1: int):
        """Local rows ``[i0, i1)`` on the compute device: a view of the resident shard, or a contiguous staging copy of the parked slab."""
        if not (0 <= int(i0) <= int(i1) <= int(self.shape[-3])):
            raise RowpairRefused(f"ShardPark({self.name}).block({i0}, {i1}): outside [0, {self.shape[-3]}]", LEVER)
        if self.mode == ZINIT_RECOMPUTE:
            return self._recomputed(int(i0), int(i1))
        if not self.park:
            if self.dev is None:
                raise RowpairRefused(f"ShardPark({self.name}).block: released", LEVER)
            return self.dev[..., i0:i1, :, :]
        if self.host is None:
            raise RowpairRefused(f"ShardPark({self.name}).block: released", LEVER)
        stage = torch.empty(self.shape[:-3] + (int(i1) - int(i0),) + self.shape[-2:], dtype=self.dtype, device=self.device)
        return load_rows(self.host, self.shape, stage.element_size(), int(i0), int(i1), stage, non_blocking=False)

    def full(self):
        """The whole shard on the compute device (the resident tensor itself, or one H2D copy of the parked buffer)."""
        if self.mode == ZINIT_RECOMPUTE:
            return self._recomputed(0, int(self.shape[-3]))
        if not self.park:
            if self.dev is None:
                raise RowpairRefused(f"ShardPark({self.name}).full: released", LEVER)
            return self.dev
        if self.host is None:
            raise RowpairRefused(f"ShardPark({self.name}).full: released", LEVER)
        out = torch.empty(self.shape, dtype=self.dtype, device=self.device)
        return self.host.load(out, non_blocking=False)

    def release(self) -> None:
        """Drop the host buffer (its bytes return to the pool's budget) and the resident reference."""
        if self.host is not None:
            self.host.release()
        self.host, self.dev = None, None
        if self.mode == ZINIT_RECOMPUTE and self._rows_fn is not None:
            self._say(f"[park] {self.name} recompute released: {self.n_recompute} row-block calls re-ran the init statement in {self.recompute_s:.2f}s")
            record_schedule(zinit_recompute_calls=self.n_recompute, zinit_recompute_s=round(self.recompute_s, 3))
        self._rows_fn = None

    def record(self) -> Dict[str, object]:
        """``{name, parked, where (device|host|host_pinned|host_pageable:<kind>), fallback, pinned, gib, park_s, device_release}`` — the park's
        row of the trunk record (:meth:`census_words` puts ``where`` / ``gib`` / the release word in the schedule census as ``park_<name>*``)."""
        return {"name": self.name, "parked": bool(self.park), "where": self.where, "fallback": self.fallback, "pinned": bool(self.pinned),
                "gib": round(self.nbytes / 2 ** 30, 4), "park_s": self.park_s, "device_release": self.device_release,
                "chunks": self.chunks, "pinned_gib": self.pinned_gib}

    @property
    def chunks(self) -> int:
        """Host chunks the park holds (0 when resident / released)."""
        return int(self.host.n_chunks) if self.host is not None else 0

    @property
    def pinned_gib(self) -> float:
        """What the park's page-locked chunks occupy in torch's caching host allocator (each chunk rounded to the next power of two), GiB."""
        return round(self.host.pinned_bytes / 2 ** 30, 3) if self.host is not None else 0.0

    def census_words(self, key: str) -> Dict[str, object]:
        """``{<key>: where, <key>_gib: the shard's GiB (host bytes while parked), <key>_release: the device copy's fate}`` (+ ``<key>_chunks`` /
        ``<key>_pinned_gib``: the host copy's chunk count / page-locked GiB, when parked) for :func:`.evidence.record_schedule` — ``release`` = ``storage_resized_to_0`` | ``not_released:shared_storage`` |
        ``not_released:storage_not_resizable`` | ``kept`` (parked, device copy kept) | ``resident`` (not parked: the device tensor IS the shard)."""
        release = self.device_release if self.device_release is not None else ("kept" if self.park else "resident")
        words = {key: self.where, key + "_gib": round(self.nbytes / 2 ** 30, 3), key + "_release": release}
        if self.park:                                                        # a parked shard says what its host copy occupies
            words.update({key + "_chunks": self.chunks, key + "_pinned_gib": self.pinned_gib})
        return words


# ================================================================================================================ replicated guards
def guard_replicated(t, name: str):
    """Prove a REPLICATED tensor is bit-identical on every rank (:func:`.dist.allreduce_checksum`; a mismatch raises :class:`RowpairRefused`
    naming ``name`` and the per-rank checksums). Returns the checksums."""
    return allreduce_checksum(t, name=name, raise_on_mismatch=True)


def guard_replicated_params(named_tensors, name: str) -> int:
    """Prove a module's PARAMETERS (and buffers) are replicated: every rank must hold bit-identical weights, or the row statements mix
    rows computed from different models (a per-rank random init, a nondeterministic checkpoint load, a rank-local finetune are silent
    wrong structures otherwise). ``named_tensors``: an iterable of ``(param_name, tensor)`` — ``module.named_parameters()`` /
    ``module.state_dict().items()`` — or a mapping. ONE object all-gather per call: the per-tensor :func:`.dist.checksum` digests travel as
    one list, so a mismatch names WHICH tensors differ (``RowpairRefused``: ``replicated parameters '<name>' differ across ranks: [...]``).
    An adapter calls it once per bound module at install time (weights are replicated tensors too). Returns the number of tensors
    checked; without a group nothing is compared."""
    items = list(named_tensors.items()) if hasattr(named_tensors, "items") else [(str(k), v) for k, v in named_tensors]
    local = [(k, checksum(v.detach())) for k, v in items]
    if not is_dist():
        return len(local)
    allv = comm().allgather_obj(local)
    names0 = [k for k, _ in allv[0]]
    for q, lst in enumerate(allv):
        if [k for k, _ in lst] != names0:
            raise RowpairRefused(f"replicated parameters {name!r}: rank {q} lists different tensor names ({len(lst)} vs {len(names0)})")
    bad = [names0[i] for i in range(len(names0)) if any(lst[i][1] != allv[0][i][1] for lst in allv)]
    if bad:
        raise RowpairRefused(f"replicated parameters {name!r} differ across ranks: {bad[:12]}{' …' if len(bad) > 12 else ''} "
                             f"({len(bad)} of {len(names0)} tensors)")
    return len(names0)


def guard_rng_replicated(name: str = "trunk.rng", device=None) -> Dict[str, object]:
    """Prove the torch RNG STREAM is at the same state on every rank before a replicated random draw: the CPU default generator's state and,
    when ``device`` is a CUDA device (or CUDA is current), that device's generator state are checksummed and all-gathered
    (:func:`guard_replicated`); a rank out of lockstep raises :class:`RowpairRefused` naming ``<name>.cpu`` / ``<name>.cuda``. O(KB) per call.
    Returns ``{"cpu": checksums, "cuda": checksums | None}``. Without a group: the local checksums, nothing raised."""
    out: Dict[str, object] = {"cpu": guard_replicated(torch.get_rng_state(), f"{name}.cpu"), "cuda": None}
    dev = None
    if device is not None and torch.device(device).type == "cuda":
        dev = torch.device(device)
    elif device is None and torch.cuda.is_available() and torch.cuda.is_initialized():
        dev = torch.device("cuda", torch.cuda.current_device())
    if dev is not None:
        out["cuda"] = guard_replicated(torch.cuda.get_rng_state(dev), f"{name}.cuda")
    return out


# ================================================================================================================ the trunk driver
class TrunkOut(object):
    """The driver's result: ``s`` (the replicated single representation as the callables left it), ``z`` (this rank's shard ``[*, R, N, C]`` when
    ``sharded``; the whole pair tensor after ``gather="all"``, on rank 0 only after ``gather="rank0"`` — None elsewhere), ``layout``,
    ``start_cycle`` / ``resumed`` (checkpoint resume), ``record`` (block sizes and sources, the park's row, the guard, the gather, timings)."""

    __slots__ = ("s", "z", "layout", "sharded", "start_cycle", "resumed", "record")

    def __init__(self, s, z, layout, sharded, start_cycle, resumed, record):
        self.s, self.z, self.layout, self.sharded, self.start_cycle, self.resumed, self.record = s, z, layout, sharded, start_cycle, resumed, record

    def __iter__(self):                       # s, z, out = run_trunk_sharded(...)
        return iter((self.s, self.z, self))


def run_trunk_sharded(layout: Layout, *, n_cycles: int,
                      init_rows_fn: Callable[[int, int], object],
                      init_like,
                      recycle_update_fn: Callable[[object], object],
                      s_init=None,
                      single_recycle_fn: Optional[Callable[[object, int], object]] = None,
                      template_fn: Optional[Callable[[object, int], object]] = None,
                      msa_fn: Optional[Callable[[object, int], object]] = None,
                      pairstack_fn: Optional[Callable[[object, object, int], Tuple[object, object]]] = None,
                      s_input=None,
                      init_rows: Optional[int] = None, init_channels: Optional[int] = None, recycle_rows: Optional[int] = None,
                      park_zinit: Optional[bool] = None, pool=None,
                      guard: Optional[str] = None, gather: str = "none", ckpt=None,
                      on_cycle_start: Optional[Callable[[int, bool], None]] = None,
                      census_fn: Optional[Callable[[str], None]] = None, log: Optional[Callable[[str], None]] = None) -> TrunkOut:
    """The recycling trunk with the pair representation row-sharded from birth (inference). Control flow, stock names on the left::

        z_init = input_embedder(...)              ->  z_init_loc = init_pair_shard(layout, init_rows_fn, init_like, init_rows)  [*, R, N, C]
                                                      zinit = ShardPark(z_init_loc, park_zinit)                             (ROWPAIR_PARK_ZINIT)
        s = zeros_like(s_init); z = zeros_like    ->  s = zeros_like(s_init) (replicated); z_loc = None (NO zeros shard)
        for cycle in range(n_cycles):
            z = z_init + linear(LN(z))            ->  z_loc = recycle_shard_(z_loc, zinit, recycle_update_fn, layout, recycle_rows)  (in place)
            z = z + template_embedder(z, ...)     ->  z_loc = template_fn(z_loc, cycle)                                    (rows, in place)
            m = msa_embedder(...) ; z = msa(m, z) ->  guard_rng_replicated() ; z_loc = msa_fn(z_loc, cycle)                (m replicated: the draw is
                                                      from a proven-identical RNG stream; OPM rows / pair averaging on local rows inside msa_fn)
            s = s_init + linear_s(LN_s(s))        ->  s = single_recycle_fn(s, cycle)                                      (replicated statement)
            s, z = pairformer(s, z, ...)          ->  s, z_loc = pairstack_fn(s, z_loc, cycle)
            (checkpoint)                          ->  ckpt.maybe_save_cycle(...) / ckpt.save_trunk_final(...) when ckpt is given
        return s, z                               ->  TrunkOut(s, z_loc)  — gather="none": the shard (its consumers are row-local);
                                                      "all": ONE all-gather of the rows; "rank0": the whole tensor on rank 0 only

    Every ``*_fn`` is the ADAPTER's callable closing over the engine's modules and batch (this module imports no engine): ``init_rows_fn(g0, g1)``
    -> z_init rows ``[*, g1-g0, N, C]`` (``init_like [*, N, C]`` fixes lead dims / C / dtype / device, see :func:`init_pair_shard`); ``recycle_update_fn(z_rows)`` -> ``linear(LN(z_rows))``; ``template_fn(z_loc, cycle)`` / ``msa_fn(z_loc,
    cycle)`` -> the shard with that module's update added (in place allowed); ``single_recycle_fn(s, cycle)`` -> s; ``pairstack_fn(s, z_loc, cycle)``
    -> ``(s, z_loc)``; a None callable is skipped. ``s_init`` / ``s_input`` are replicated (``s_input`` is only handed to ``ckpt``). ``guard``:
    ``"rng"`` (default, ``ROWPAIR_TRUNK_GUARD``) proves the RNG stream identical across ranks before every ``msa_fn`` call; ``"off"`` is printed.
    ``ckpt`` duck-types :class:`.ckpt.TrunkCheckpointer` (``try_resume(layout=, num_cycles=, device=)`` -> None | ``{"tag", "cycle", "s", "z_loc",
    "s_input"}``; ``phase(name)``; ``maybe_save_cycle(cycle, layout=, s=, z_loc=)``; ``save_trunk_final(layout=, s_input=, s=, z_loc=, cycle=)``): a
    ``trunk_final`` resume returns without computing; a cycle-k resume continues at k+1 (the RNG state is the checkpointer's to restore).
    ``on_cycle_start(cycle, is_final)`` and ``census_fn(tag)`` are optional engine hooks (autocast-cache clearing; residency census lines);
    ``log(text)`` receives the park / timing lines. Refused by name at ``P == 1`` (the engine's own trunk runs at ``--n_gpu 1``)."""
    require_sharded(layout, "run_trunk_sharded", LEVER)
    if gather not in GATHERS:
        raise RowpairRefused(f"run_trunk_sharded: gather={gather!r} not in {GATHERS}", LEVER)
    guard = (os.environ.get("ROWPAIR_TRUNK_GUARD", "rng").strip().lower() or "rng") if guard is None else str(guard)
    if guard not in GUARDS:
        raise RowpairRefused(f"run_trunk_sharded: guard={guard!r} not in {GUARDS} (ROWPAIR_TRUNK_GUARD)", LEVER)
    n_cycles = int(n_cycles)
    if n_cycles < 1:
        raise RowpairRefused(f"run_trunk_sharded: n_cycles={n_cycles} < 1", LEVER)
    say = log if log is not None else (lambda _m: None)
    census = census_fn if census_fn is not None else (lambda _t: None)
    rec: Dict[str, object] = {"P": layout.P, "rank": layout.rank, "rows": f"{layout.r0}:{layout.r1}", "n_cycles": n_cycles, "guard": guard,
                              "gather": gather, "replicated": REPLICATED, "timings_s": {}}
    record_schedule(trunk_guard=guard, trunk_gather=gather, trunk_replicated=REPLICATED)

    # ---- resume before any compute: trunk_final -> nothing to do; cycle k -> continue at k + 1
    res = None
    if ckpt is not None:
        res = ckpt.try_resume(layout=layout, num_cycles=n_cycles, device=None if s_init is None else s_init.device)
        if res is not None and res.get("tag") == "trunk_final":
            say("trunk skipped: resumed from trunk_final")
            rec["resumed"] = "trunk_final"
            return _finish(res.get("s"), res.get("z_loc"), layout, gather, n_cycles, True, rec)

    t0 = time.time()
    if zinit_park_mode(park_zinit) == ZINIT_RECOMPUTE:                                                    # ROWPAIR_PARK_ZINIT=recompute:
        zinit = ShardPark.recompute(layout, init_rows_fn, init_like, rows=init_rows, channels=init_channels, name="z_init", log=say)   # no z_init tensor anywhere;
        rec["timings_s"]["init"] = round(time.time() - t0, 3)                                            # cycle k's recycle re-runs the init statement per block
        census("trunk.after_init")
    else:
        z_init_loc = init_pair_shard(layout, init_rows_fn, init_like, init_rows, channels=init_channels)  # seam: pair init on local rows only
        rec["timings_s"]["init"] = round(time.time() - t0, 3)
        census("trunk.after_init")
        zinit = ShardPark(z_init_loc, park=park_zinit, name="z_init", pool=pool, log=say)               # ROWPAIR_PARK_ZINIT: host copy + device release
        del z_init_loc
    rec["park"] = zinit.record()
    record_schedule(**zinit.census_words("park_z_init"), zinit_park=zinit.mode)
    census("trunk.after_park")

    s = torch.zeros_like(s_init) if s_init is not None else None
    z_loc = None                                                                                          # cycle 0: no zeros shard (recycle_shard_)
    start_cycle = 0
    if res is not None:
        start_cycle = int(res["cycle"]) + 1
        s, z_loc = res.get("s", s), res["z_loc"]
        rec["resumed"] = f"cycle{res['cycle']}"
        say(f"trunk resumed after cycle {res['cycle']} -> continuing at cycle {start_cycle}/{n_cycles}")

    for cycle in range(start_cycle, n_cycles):
        is_final = cycle == n_cycles - 1
        first = cycle == start_cycle
        tc = time.time()
        if on_cycle_start is not None:
            on_cycle_start(cycle, is_final)

        z_loc = recycle_shard_(z_loc, zinit, recycle_update_fn, layout, recycle_rows)               # seam: recycling, in place on the shard
        if is_final and not _env_flag(ENV_PARK_ZINIT_HOLD):
            zinit.release()                                                                          # its last reader was this recycle
            record_schedule(park_z_init_freed=f"after_cycle{cycle}_recycle")
            say(f"[park] z_init released after the final cycle's recycle (cycle {cycle})")
        _phase(ckpt, f"cycle{cycle}.recycle"); census(f"cycle{cycle}.recycle") if first else None

        if template_fn is not None:
            tprobe = DevicePeak(z_loc.device) if os.environ.get(ENV_TEMPL_ORDER_NAME, "").strip() else None   # bracket the stage's device
            z_loc = template_fn(z_loc, cycle)                                                        # seam: template embedder adds on rows
            if tprobe is not None:                                                                   # peak only when ROWPAIR_TEMPL_ORDER is SET (unset:
                tp = tprobe.close()                                                                  # no probe, no census word — today's bytes)
                record_schedule(templ_stage_peak_gib=tp["peak_gib"], templ_stage_peak_kind=tp["kind"], templ_stage_peak_at=tp["at"],
                                templ_stage_entry_gib=tp["entry_gib"], templ_stage_exit_gib=tp["exit_gib"])
                say(f"[templ] cycle {cycle} template stage device peak {tp['peak_gib']:.2f} GiB ({tp['kind']} at {tp['at']}; sampled max "
                    f"{tp.get('sampled_peak_gib', 0.0):.2f} GiB over {tp.get('samples', 0)} samples + {tp['marks']} marks; process high-water "
                    f"{tp.get('process_high_water_gib', 0.0):.2f} GiB); entry {tp['entry_gib']:.2f} exit {tp['exit_gib']:.2f} GiB")
            host_trim(f"templ.cycle{cycle}", say)                                                    # under ROWPAIR_HOST_SLAB=lease return the
            _phase(ckpt, f"cycle{cycle}.template"); census(f"cycle{cycle}.template") if first else None   # stage's cached-free pinned host (gated; unset = no-op)

        if msa_fn is not None:
            if guard == "rng":
                guard_rng_replicated(f"trunk.cycle{cycle}.rng", device=zinit.device)                # the MSA subsample draw: replicated by proof
            z_loc = msa_fn(z_loc, cycle)                                                             # seams: MSA embedder (replicated m) + MSA module
            _phase(ckpt, f"cycle{cycle}.msa"); census(f"cycle{cycle}.msa") if first else None

        if single_recycle_fn is not None:
            s = single_recycle_fn(s, cycle)                                                          # replicated: s = s_init + linear_s(LN(s))
        if pairstack_fn is not None:
            s, z_loc = pairstack_fn(s, z_loc, cycle)                                                 # seams: the pairformer on the shard
            _phase(ckpt, f"cycle{cycle}.pairformer"); census(f"cycle{cycle}.pairformer") if first else None

        if ckpt is not None:
            if is_final:
                ckpt.save_trunk_final(layout=layout, s_input=s_input, s=s, z_loc=z_loc, cycle=cycle)
            else:
                ckpt.maybe_save_cycle(cycle, layout=layout, s=s, z_loc=z_loc)
        rec["timings_s"][f"cycle{cycle}"] = round(time.time() - tc, 3)

    if zinit.host is not None or zinit.dev is not None:
        zinit.release()
        record_schedule(park_z_init_freed="trunk_exit")
    return _finish(s, z_loc, layout, gather, start_cycle, res is not None, rec)


def _phase(ckpt, name: str) -> None:
    if ckpt is not None and hasattr(ckpt, "phase"):
        ckpt.phase(name)


def _finish(s, z_loc, layout: Layout, gather: str, start_cycle: int, resumed: bool, rec: Dict[str, object]) -> TrunkOut:
    """The ONE optional exit gather of the driver (``none``: the shard as is)."""
    if gather == "none" or z_loc is None:
        return TrunkOut(s, z_loc, layout, True, start_cycle, resumed, rec)
    if gather == "all":
        return TrunkOut(s, unshard_rows(z_loc, layout, dim=-3), layout, False, start_cycle, resumed, rec)
    return TrunkOut(s, unshard_rows_to_rank0(z_loc, layout, dim=-3), layout, False, start_cycle, resumed, rec)
