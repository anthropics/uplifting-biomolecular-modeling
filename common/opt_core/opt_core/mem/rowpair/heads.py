"""The pair-conditioned output heads under row sharding — distogram / PAE / PDE logits and the confidence head's pair input are produced
ROW BLOCK BY ROW BLOCK from the pair shard; an ``[N, N, bins]`` logits tensor and a per-sample ``[S, N, N, C]`` pair representation never
exist (at most ``rows x N x bins`` per rank per statement). No engine name here: the head linears / LayerNorms / the confidence
pairformer's embedding statements enter as CALLABLES that are per-(i, j) maps (a linear over the channel dim, ``LN`` over the channel dim,
a one-hot distance embedding of row i against all j ...), so evaluating them on a row block — or on a COLUMN slab — returns exactly the
elements the dense statement returns there.

Layout of the operands: rows are dim ``-3`` (``z[..., rows, N, C]``); any leading dims (batch 1, samples S) ride along unchanged. The trunk
pair shard is ``[1, R, N, C]`` (or ``[R, N, C]``); the confidence head's per-sample pair representation is ``[S, R, N, C]`` — R local rows,
never N.

API (every generator MUST be exhausted on every rank: the symmetrised form runs one all-to-all window per step, in the same order on
every rank; ``rows`` is the row block — see :func:`conf_rows`)::

    conf_rows(N, bins, rows=None, lead=1)             the row block of the head schedules and its census source (:func:`.shard.choose_block_rows`):
                                                      ``rows`` given -> ``given``; ``ROWPAIR_CONF_ROWS`` (row count) -> ``env:…``; else 128 rows ->
                                                      ``default``; then the int32 rule ``rows*N*bins*lead < 2**31`` (``+cap``). A function of
                                                      (N, bins, lead, env) only => identical on every rank by construction (no free-memory term:
                                                      the symmetrised schedule is a collective). Recorded: ``conf_rows``, ``conf_rows_source``.
    logit_rows(fn, z_shard, layout, rows=)            yields ``(i0, i1, fn(z_shard[..., i0:i1, :, :]))`` per LOCAL row block — the plain per-row
                                                      head (PAE). No communication; kept beside the symmetrised form so both heads walk one
                                                      printed schedule.
    sym_logit_rows(fn, z_shard, layout, rows=)        the symmetrised head ``L + L.transpose(-2, -3)`` with ``L = fn(z)`` (PDE, distogram):
                                                      yields ``(i0, i1, fn(z_shard[..., i0:i1, :, :]) + LT)`` where ``LT[..., w, j, :] =
                                                      fn(z)[..., j, r0+i0+w, :]`` is assembled from the COLUMN slab ``fn(z_q[..., :, window, :])``
                                                      of every rank q by ONE all-to-all window (:func:`.dist.alltoall_window`; "column-slab
                                                      exchange"): per element identical to the dense statement, pure data movement for the
                                                      transpose term. Transient per step: ``N x rows x F`` (the slabs) + the a2a staging.
    embed_rows(fn, z_shard, layout, rows=, lead_out=, inplace=)   the head's pair INPUT from the trunk shard, per row block:
                                                      ``out[..., i0:i1, :, :] = fn(z_shard[..., i0:i1, :, :], g0, g1)`` (``g0, g1`` = the block's
                                                      GLOBAL rows, for statements that slice replicated per-token operands). ``out`` is
                                                      ``[*lead_out, R, N, C]`` — R rows: the per-sample representation of S samples costs
                                                      ``S/P`` pair-equivalents, never a full clone. ``inplace=True`` (lead numel unchanged)
                                                      overwrites the trunk shard's own storage block by block (row i's result depends on row i
                                                      only, so the in-place form is exact) — zero extra pair memory when S == 1. ``z_shard``
                                                      may be a duck-typed source with ``.zrows(i0, i1) -> [*lead, w, N, C]`` and ``.shape /
                                                      .dtype / .device`` (a host-parked trunk shard served block by block: :class:`ZTrunkPlan`).
    ZTrunkPlan(z_shard, passes=, free=, park=, ...)   the PLACEMENT of the trunk shard across the confidence stage's passes (pass i = one
                                                      ``embed_rows`` call producing the ``[k_i, R, N, C]`` pair input of k_i samples + that pass's
                                                      pair stack and heads), decided per pass, ONCE, here: ``inplace`` (``ROWPAIR_FREE_ZTRUNK``: the
                                                      pair input overwrites the shard's storage — one sample-equivalent per pass at the shard's LAST
                                                      use), else ``parked`` (``ROWPAIR_CONF_PARK_ZTRUNK``: the shard copied to (pinned) host memory and
                                                      its device storage RELEASED before the pair input is allocated, row blocks served from the host,
                                                      the same storage re-grown on request — :class:`.template.ParkedStorage`), else ``resident``.
                                                      ``begin(i, lead_out) -> (src, inplace)`` feeds ``embed_rows``; ``end(i, device_needed_next=)``;
                                                      every pass's form is a census word (``conf_ztrunk``). Device bytes: one shard-equivalent per
                                                      sample of the pass under either lever, instead of the trunk shard + the pass's pair input.
The row-block loops are :func:`.shard.iter_row_blocks` / :func:`.shard.produce_rows_` (``unit=1``: head statements are per-(i, j) maps, any
block boundary is exact); the symmetrised head walks ``ceil(Rmax / rows)`` WINDOWS instead (a rank-invariant count — it is a collective).

The structural n_gpu=1 rule: each entry point REFUSES a P == 1 / replicated / no-group layout by name (:func:`.dist.require_sharded`) unless
``allow_unsharded=True`` — a kit that ships row-blocked heads as its OWN single-device memory lever (the dense ``[N, N, bins]`` logits are
44 GB at N = 13.5k) says so and names that lever in its line; the statements are then the dense ones evaluated in row blocks (the
transpose term of the symmetrised head is ``fn`` on this process's own column slab).

Levers read from the environment (explicit arguments win; every outcome in the census)::

    ROWPAIR_CONF_ROWS=<rows>          the head row block (default 128 rows, int32-capped)
    ROWPAIR_FREE_ZTRUNK=0|1           ZTrunkPlan: the confidence pair input may overwrite the trunk shard's storage at its last use (default 0)
    ROWPAIR_CONF_PARK_ZTRUNK=0|1      ZTrunkPlan: the trunk shard is parked on the host, device storage released, while a pass's pair input lives
                                      (default 0); ROWPAIR_PARK_STRICT / ROWPAIR_PARK_PIN_MAX_GB govern its pool as they govern the trunk's parks

Numerics: ``fn`` on a row block / column slab = the dense elements (per-(i, j) maps; launch M differs); the transpose exchange, the
in-place embed and the host park are data movement (bit-exact: the three placements of :class:`ZTrunkPlan` give identical outputs). REPLICATED by design (named, not sharded): the single representation the embedding callable
closes over, per-token coordinates, the head weights.
"""
from __future__ import annotations
import time
import os

from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from . import RowpairRefused
from ._torch import torch
from .dist import Layout, alltoall_window, env_flag, env_int, is_dist, require_sharded
from .shard import iter_row_blocks, owns_whole_storage, produce_rows_, storage_resizable

__all__ = ["ENV_CONF_ROWS", "ENV_FREE_ZTRUNK", "ENV_CONF_PARK_ZTRUNK", "CONF_ROWS_DEFAULT", "conf_rows", "logit_rows", "sym_logit_rows",
           "embed_rows", "ZTrunkPlan", "ZTRUNK_FORMS"]

ENV_CONF_ROWS = "ROWPAIR_CONF_ROWS"
ENV_FREE_ZTRUNK = "ROWPAIR_FREE_ZTRUNK"              # 1: the confidence pair input may overwrite the trunk shard's storage at its last use
ENV_CONF_PARK_ZTRUNK = "ROWPAIR_CONF_PARK_ZTRUNK"    # 1: the trunk shard is parked on the host (device storage released) while a pass's pair input lives
ZTRUNK_FORMS = ("inplace", "parked", "resident")     # the per-pass placements ZTrunkPlan decides among (first word of its census words)
LEVER = "rowpair.heads"


# ----------------------------------------------------------------------------------------------------------------- schedule
CONF_ROWS_DEFAULT = 128                                   # the head row block when nothing else binds (int32 cap below it)


def conf_rows(N: int, bins: int, rows: Optional[int] = None, lead: int = 1, *, record: bool = True) -> Tuple[int, str]:
    """``(rows, source)``: the row block of the head schedules, by :func:`.shard.choose_block_rows` (THE chooser): ``rows`` given -> ``given``;
    env ``ROWPAIR_CONF_ROWS`` (a ROW COUNT) -> ``env:ROWPAIR_CONF_ROWS``; else ``CONF_ROWS_DEFAULT`` = 128 rows (``default``); then the int32 rule
    ``rows * N * bins * lead < 2**31`` (``+cap``). A function of its arguments / the environment only — identical on every rank (the
    symmetrised schedule is a collective every rank walks in step); recorded as ``conf_rows`` / ``conf_rows_source``."""
    from .shard import choose_block_rows
    epr = max(1, int(N) * max(1, int(bins)) * max(1, int(lead)))
    env_rows = env_int(ENV_CONF_ROWS, 0)
    given = int(rows) if (rows is not None and int(rows) > 0) else (env_rows if env_rows > 0 else None)
    r, src = choose_block_rows(rows=given, env=None, elems_per_row=epr, elem_bytes=4, default_mb=CONF_ROWS_DEFAULT * epr * 4 / 2 ** 20,
                               cap_elems=2 ** 31 - 1)
    if given is not None and not (rows is not None and int(rows) > 0):
        src = src.replace("given", "env:" + ENV_CONF_ROWS, 1)
    if record:
        from .evidence import record_schedule
        record_schedule(conf_rows=r, conf_rows_source=src)
    return r, src


def _rows_dim_checks(what: str, z_shard, layout: Layout, allow_unsharded: bool) -> int:
    """Common preconditions: the structural n_gpu=1 rule and ``z_shard.shape[-3] == layout.R``. Returns R."""
    if not allow_unsharded:
        require_sharded(layout, what)
    if len(tuple(z_shard.shape)) < 3:
        raise RowpairRefused(f"{what}: z_shard must be [..., R, N, C] (rows on dim -3); got shape {tuple(z_shard.shape)}")
    R = int(z_shard.shape[-3])
    if R != layout.R or int(z_shard.shape[-2]) != layout.N:
        raise RowpairRefused(f"{what}: z_shard is {tuple(z_shard.shape)} but layout has R={layout.R} local rows of N={layout.N}")
    return R


def _refuse_parked(what: str, z_shard) -> None:
    """The logits heads read row blocks AND column slabs of the device shard: a parked source (rows only) is refused by name."""
    if hasattr(z_shard, "zrows") and not torch.is_tensor(z_shard):
        raise RowpairRefused(f"{what}: a parked shard serves rows only; take the distogram / PAE / contact rows from the device shard before "
                             "ZTrunkPlan.begin / park_now releases it (plan.decide says when it will)")


def _lead_numel(shape: Sequence[int]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


# ----------------------------------------------------------------------------------------------------------------- plain per-row head
def logit_rows(fn: Callable, z_shard, layout: Layout, rows: Optional[int] = None, *, bins: int = 64, allow_unsharded: bool = False):
    """Yields ``(i0, i1, fn(z_shard[..., i0:i1, :, :]))`` for this rank's LOCAL row blocks (ascending, covering ``[0, R)``). ``fn`` is a
    per-(i, j) map ``[..., w, N, C] -> [..., w, N, F]`` (a head's LayerNorm + linear). ``bins`` sizes the default row block (int32 rule)."""
    _refuse_parked("logit_rows", z_shard)
    R = _rows_dim_checks("logit_rows", z_shard, layout, allow_unsharded)
    lead = _lead_numel(z_shard.shape[:-3])
    rb, _ = conf_rows(layout.N, bins, rows, lead)
    for i0, i1, _g0, _g1 in iter_row_blocks(layout, rb, unit=1):
        yield i0, i1, fn(z_shard[..., i0:i1, :, :])


# ----------------------------------------------------------------------------------------------------------------- symmetrised head
def sym_logit_rows(fn: Callable, z_shard, layout: Layout, rows: Optional[int] = None, *, bins: int = 64, chunks: Optional[int] = None,
                   allow_unsharded: bool = False):
    """The symmetrised head ``L + L.transpose(-2, -3)``, ``L = fn(z)``, on row shards. Yields ``(i0, i1, logits)`` with
    ``logits[..., w, j, :] = fn(z)[.., r0+i0+w, j, :] + fn(z)[..., j, r0+i0+w, :]`` for this rank's LOCAL row blocks; blocks with
    ``i1 == i0`` are not yielded but their exchange step still runs (COLLECTIVE: every rank walks ``ceil(Rmax / rows)`` all-to-all windows in
    the same order — exhaust the generator on every rank). ``fn``: per-(i, j) map ``[..., a, b, C] -> [..., a, b, F]`` applied to row blocks
    AND column slabs of the shard. ``chunks``: channel groups staging each all-to-all (:func:`.dist.alltoall_window`)."""
    _refuse_parked("sym_logit_rows", z_shard)
    R = _rows_dim_checks("sym_logit_rows", z_shard, layout, allow_unsharded)
    lead_shape = tuple(int(s) for s in z_shard.shape[:-3])
    L = _lead_numel(lead_shape)
    rb, _ = conf_rows(layout.N, bins, rows, L)
    sharded = not (layout.P == 1 or layout.replicated or not is_dist())
    if not sharded:
        # the single-device lever (allow_unsharded): my column slab IS the whole column block
        for i0, i1, _g0, _g1 in iter_row_blocks(layout, rb, unit=1):
            local = fn(z_shard[..., i0:i1, :, :])                                   # [..., w, N, F]
            colT = fn(z_shard[..., :, i0:i1, :]).transpose(-2, -3)                 # [..., N, w, F] -> [..., w, N, F]
            yield i0, i1, local + colT
        return
    nd = len(lead_shape)
    probe = fn(z_shard[..., 0:min(R, 1), 0:1, :])                                     # one element: the head's F and dtype
    F, dtype = int(probe.shape[-1]), probe.dtype
    del probe
    perm_fold = [nd, nd + 1] + list(range(nd)) + [nd + 2]                            # [*lead, R, W, F] -> [R, W, *lead, F]

    def provider(c0: int, c1: int):
        # my rows, the destination's column window: fn on the column slab, lead dims folded into channels (data movement)
        t = fn(z_shard[..., :, c0:c1, :])                                             # [*lead, R, W, F]
        return t.permute(*perm_fold).reshape(R, c1 - c0, L * F).contiguous()

    for off in range(0, layout.Rmax, rb):                                            # identical window schedule on every rank
        i0, i1 = min(R, off), min(R, off + rb)
        local = fn(z_shard[..., i0:i1, :, :]) if i1 > i0 else None                  # [*lead, w, N, F]
        colT = alltoall_window(provider, layout, off, rb, C=L * F, dtype=dtype, device=z_shard.device, out_layout="rows", chunks=chunks)
        # colT[w, j, l*F + f] = fn(z)[l][j, r0+off+w, f]   -> [*lead, w, N, F]
        Wv = int(colT.shape[0])
        if Wv != i1 - i0:
            raise RowpairRefused(f"sym_logit_rows: window rows {Wv} != local block {i1 - i0} (off={off}, R={R}, layout={layout!r})")
        if Wv == 0:
            del colT
            continue
        colT = colT.view((Wv, layout.N) + lead_shape + (F,))
        perm = list(range(2, 2 + nd)) + [0, 1, 2 + nd]
        colT = colT.permute(*perm)
        yield i0, i1, local + colT
        del colT, local


# ----------------------------------------------------------------------------------------------------------------- per-sample pair input
def embed_rows(fn: Callable, z_shard, layout: Layout, rows: Optional[int] = None, *, lead_out: Optional[Sequence[int]] = None,
               inplace: bool = False, out=None, out_dtype=None, bins: int = 0, allow_unsharded: bool = False):
    """The confidence head's pair input built per row block from the trunk shard: ``out[..., i0:i1, :, :] = fn(z_blk, g0, g1)`` with
    ``z_blk = z_shard[..., i0:i1, :, :]`` and ``(g0, g1) = (r0+i0, r0+i1)`` the block's GLOBAL rows. ``fn`` returns ``[*lead_out, w, N, C]``
    (``lead_out`` defaults to ``z_shard``'s leading dims; a per-sample statement returns S on a leading dim).

    ``inplace=True``: the trunk shard's storage is overwritten (requires ``numel(lead_out) == numel(lead_in)``, a contiguous shard TENSOR and
    ``out_dtype`` == the shard's dtype) and returned viewed as ``[*lead_out, R, N, C]`` — exact because row block i's result reads row block i
    only. Otherwise a new ``[*lead_out, R, N, C]`` tensor of ``out_dtype`` (default: the shard's dtype; ``out`` if given) — R local rows per
    sample, never N. ``rows``: the row block (``ROWPAIR_CONF_ROWS`` / int32 rule via :func:`conf_rows` with ``bins = C * 3 + bins`` as the
    transient width per row element). Which of the two a confidence stage uses, per pass: :class:`ZTrunkPlan`."""
    R = _rows_dim_checks("embed_rows", z_shard, layout, allow_unsharded)
    lead_in = tuple(int(s) for s in z_shard.shape[:-3])
    lead_out = lead_in if lead_out is None else tuple(int(s) for s in lead_out)
    N, C = int(z_shard.shape[-2]), int(z_shard.shape[-1])
    rb, _ = conf_rows(layout.N, C * 3 + int(bins), rows, max(1, _lead_numel(lead_out)))
    out_dtype = z_shard.dtype if out_dtype is None else out_dtype
    if inplace:
        if hasattr(z_shard, "zrows"):
            raise RowpairRefused("embed_rows: inplace needs the shard tensor itself (a parked / duck-typed source has no device storage to reuse)")
        if _lead_numel(lead_out) != _lead_numel(lead_in):
            raise RowpairRefused(f"embed_rows: inplace needs numel(lead_out)={_lead_numel(lead_out)} == numel(lead_in)={_lead_numel(lead_in)} "
                                 f"(a per-sample representation of S>1 samples cannot reuse the single trunk shard)")
        if not z_shard.is_contiguous():
            raise RowpairRefused("embed_rows: inplace needs a contiguous shard")
        if out_dtype != z_shard.dtype:
            raise RowpairRefused(f"embed_rows: inplace needs out_dtype == the shard's dtype ({out_dtype} vs {z_shard.dtype}: a pair input of another dtype "
                                 "cannot live in the shard's storage)")
        if not storage_resizable(z_shard):
            raise RowpairRefused("embed_rows: inplace refused on a shard over foreign storage (a DLPack / numpy view torch does not own)")
        out = z_shard.view(lead_out + (R, N, C))
    elif out is None:
        out = torch.empty(lead_out + (R, N, C), dtype=out_dtype, device=z_shard.device)
    elif tuple(out.shape) != lead_out + (R, N, C) or out.dtype != out_dtype:
        raise RowpairRefused(f"embed_rows: out is {tuple(out.shape)} {out.dtype}, expected {lead_out + (R, N, C)} {out_dtype}")
    r0 = layout.r0
    blk = (lambda g0, g1: z_shard.zrows(g0 - r0, g1 - r0)) if hasattr(z_shard, "zrows") else (lambda g0, g1: z_shard[..., g0 - r0:g1 - r0, :, :])

    def rows(g0: int, g1: int):
        y = fn(blk(g0, g1), g0, g1)
        if y.dtype != out.dtype:
            if torch.promote_types(y.dtype, out.dtype) != out.dtype:          # a narrowing store would round the confidence rows: refused, never silent
                raise RowpairRefused(f"embed_rows: fn returned {y.dtype} rows for a {out.dtype} pair input (rows {g0}:{g1}) — a silent cast would round them; "
                                     f"pass out_dtype={y.dtype} (ZTrunkPlan.begin(out_dtype=) declines the in-place form by name then)")
            _note_widening(y.dtype, out.dtype)                                 # an exact widening store, named in the census
        return y
    return produce_rows_(out, layout, rows, op="set", block_rows=rb, unit=1)


def _note_widening(src, dst) -> None:
    from .evidence import record_schedule
    record_schedule(conf_embed_dtype=f"{src}->{dst}".replace("torch.", ""))


# ----------------------------------------------------------------------------------------------------------------- the trunk shard across the confidence passes

ENV_POOL_SHRINK = "ROWPAIR_POOL_SHRINK"        # 0: keep the A-block row mirror and torch's cached-free pinned blocks through the confidence heads; default 1 = the pinned-pool shrink below


def pinned_shrink(say: Optional[Callable[[str], None]] = None, *, plan: Optional["ZTrunkPlan"] = None, where: str = "confidence-pairformer -> heads") -> Dict[str, object]:
    """The PINNED-POOL SHRINK at the confidence-pairformer -> heads seam of one rank: (1) release the triangle-multiplication A-block ROW
    MIRROR (:func:`.trimul.release_host_mirror`: the heads never read it; the next pass's pairformer re-establishes it through
    :func:`.trimul.host_mirror`, one re-pin), (2) return torch's CACHED-FREE pinned host blocks to the OS
    (:func:`opt_core.mem.torch_hostpair.host_cache_trim`; a torch without the binding is NAMED, nothing pretended), (3) say
    ``[pool] pinned shrink: kept X GiB in k chunks (<what is live>), returned Y GiB (row mirror M GiB + cached-free C GiB) via <call>; VmRSS a -> b GiB``.
    ``kept`` = the confidence stage's z_trunk park when ``plan`` holds one (its chunks stay: the heads read the trunk rows through the last pass).
    ``ROWPAIR_POOL_SHRINK=0`` disables it (one line says so). Placement only: no tensor value changes. Returns the record (also on the schedule
    census as ``pool_shrink=on|off|unavailable pool_shrink_returned_gib= pool_shrink_kept_gib= pool_shrink_mirror_gib= pool_shrink_cached_free_gib=<GiB|n/a> pool_shrink_stats=host_memory_stats|none pool_shrink_call=<binding>``; the binding probed is torch's private ``torch._C._host_emptyCache`` first (present on torch 2.7.1+cu128 and 2.10.0+cu128), then ``_cuda_hostEmptyCache`` / ``torch.cuda[.memory].host_empty_cache``; the line names which)."""
    from .evidence import record_schedule
    from .trimul import host_mirror_bytes, release_host_mirror
    from .trunk import release_free_host_slabs
    from ..torch_hostpair import GiB, host_cache_trim, proc_mem
    say = say or (lambda m: None)
    kept_chunks, kept_bytes, kept_what = 0, 0, "nothing parked"
    parked = plan.parked if plan is not None else None
    host = getattr(parked, "host", None)
    if host is not None:
        kept_chunks, kept_bytes = int(host.n_chunks), int(host.pinned_bytes if host.pinned else host.alloc_bytes)
        kept_what = f"z_trunk park {getattr(parked, 'where', '?')}"
    if os.environ.get(ENV_POOL_SHRINK, "1").strip() == "0":
        rec = {"pool_shrink": "off", "kept_gib": round(kept_bytes / GiB, 3), "kept_chunks": kept_chunks}
        say(f"[pool] pinned shrink: off ({ENV_POOL_SHRINK}=0) at the {where} seam; kept {kept_bytes / GiB:.2f} GiB in {kept_chunks} chunks ({kept_what}) "
            f"+ the A-block row mirror + torch's cached-free pinned blocks (VmRSS {proc_mem().get('VmRSS', float('nan')):.2f} GiB)")
        record_schedule(pool_shrink="off")
        return rec
    m0 = proc_mem()
    n_mir, mir_chunks, mir_bytes = host_mirror_bytes()
    t0 = time.time()
    release_host_mirror()
    slab_bytes = release_free_host_slabs()                                       # ROWPAIR_HOST_SLAB=lease: a FREE leased slab goes too (nothing after
    trim = host_cache_trim()                                                     # this seam holds it: the park was retired / the mirror handed it back); 0 else
    dt = time.time() - t0
    m1 = proc_mem()
    rss0, rss1 = m0.get("VmRSS"), m1.get("VmRSS")
    returned = (rss0 - rss1) if (rss0 is not None and rss1 is not None) else None
    cached = None
    if trim.get("reserved_before") not in (None, -1) and trim.get("reserved_after") not in (None, -1):
        cached = (int(trim["reserved_before"]) - int(trim["reserved_after"])) / GiB - mir_bytes / GiB - slab_bytes / GiB   # what the trim freed beyond the mirror's (and a free leased slab's) own blocks
    word = "on" if trim.get("ok") else "unavailable"
    call = trim.get("call") or "none"
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"
    say(f"[pool] pinned shrink: kept {kept_bytes / GiB:.2f} GiB in {kept_chunks} chunks ({kept_what}), returned "
        f"{(returned if returned is not None else float('nan')):.2f} GiB (row mirror {mir_bytes / GiB:.2f} GiB in {mir_chunks} chunks"
        f"{' + cached-free n/a (no host_memory_stats in this torch)' if cached is None else f' + cached-free {max(cached, 0.0):.2f} GiB'}) via {call}"
        f"{'' if trim.get('ok') else ' — TRIM UNAVAILABLE: ' + str(trim.get('error'))} at the {where} seam in {dt:.1f}s; "
        f"VmRSS {rss0 if rss0 is not None else '?'} -> {rss1 if rss1 is not None else '?'} GiB, VmLck {m0.get('VmLck', '?')} -> {m1.get('VmLck', '?')} GiB at {stamp}"
        + (f"; leased host slab (free) returned {slab_bytes / GiB:.2f} GiB" if slab_bytes else ""))
    rec = {"pool_shrink": word, "call": call, "kept_gib": round(kept_bytes / GiB, 3), "kept_chunks": kept_chunks, "mirror_gib": round(mir_bytes / GiB, 3),
           "returned_gib": None if returned is None else round(returned, 3), "rss_before_gib": rss0, "rss_after_gib": rss1, "seconds": round(dt, 2), "error": trim.get("error")}
    rec["cached_free_gib"] = None if cached is None else round(max(cached, 0.0), 3)
    if slab_bytes:
        rec["slab_gib"] = round(slab_bytes / GiB, 3)
        record_schedule(pool_shrink_slab_gib=rec["slab_gib"])
    record_schedule(pool_shrink=word, pool_shrink_call=call, pool_shrink_kept_gib=rec["kept_gib"], pool_shrink_mirror_gib=rec["mirror_gib"],
                    pool_shrink_returned_gib=rec["returned_gib"], pool_shrink_cached_free_gib=("n/a" if cached is None else rec["cached_free_gib"]),
                    pool_shrink_stats=("host_memory_stats" if cached is not None else "none"))
    return rec

class ZTrunkPlan(object):
    """The placement of the trunk pair shard across the confidence stage's ``passes`` passes — the ONE statement of "which mechanism when"
    for the trunk shard a confidence head embeds (adapters bind it; none re-derives it). Pass ``i`` = one :func:`embed_rows` call producing the
    ``[k_i, R, N, C]`` pair input of ``k_i`` samples, that pass's pair stack and heads; between passes other statements may READ the device
    shard whole (a sample loop's next diffusion roll-out). Precondition: nothing WRITES the trunk shard between ``begin(0)`` and the last ``end``.

    The per-pass form (:meth:`decide`, pure; :meth:`begin` acts on it), first match:
      * a live park from an earlier pass keeps serving                                               -> ``parked:<where>``
      * ``free`` and ``numel(lead_out) == numel(lead_in)`` and ``out_dtype == z_shard.dtype`` and last use and a contiguous shard
                                                                                                      -> ``inplace``: ``src`` is the shard,
        ``embed_rows(..., inplace=True)`` overwrites its storage block by block (zero copies, zero extra bytes)
      * ``park`` and the shard owns its whole storage (:func:`.shard.owns_whole_storage`)            -> ``parked:<where>``: a
        :class:`.template.ParkedStorage` copies the shard to the host and RELEASES its device storage inside :meth:`begin` — before
        :func:`embed_rows` allocates the pass's ``[k_i, R, N, C]`` output (that ordering is the saving) — and serves ``.zrows(i0, i1)``;
        ``where`` = ``host_pinned`` | ``host_pageable:<kind>`` (a refused page-lock answered pageable AND named; refused by name instead under
        ``ROWPAIR_PARK_STRICT=1`` / ``strict=True``) | ``host`` (a CPU shard)
      * otherwise                                                                                     -> ``resident`` (both levers off) or
        ``resident:<reason>[+<reason>]`` naming why a set lever did not engage: ``free_declined:samples`` (``numel(lead_out) != numel(lead_in)``:
        a per-sample representation of several samples cannot reuse the one shard) | ``free_declined:dtype`` (the pass's pair input has another
        dtype than the shard — a bf16 trunk shard under autocast embedded in fp32 — and cannot live in its storage) | ``free_declined:not_last``
        (a later statement reads the trunk rows) | ``free_declined:not_contiguous`` | ``free_declined:foreign_storage`` (the shard views memory
        torch does not own — a zero-copy DLPack import of another framework's buffer: never written behind the owner's back) |
        ``not_parked:shared_storage`` (the shard is a view into a larger storage: releasing it would free bytes other tensors own) |
        ``not_parked:storage_not_resizable`` (foreign memory cannot be released: :func:`.shard.storage_resizable`).
    Device bytes while a pass's pair input lives: ``k_i`` shard-equivalents under either lever (the trunk shard's own bytes are reused or
    released), ``1 + k_i`` resident. Host bytes: one shard while parked (``host_gib``). Numerics: placement only — the three forms give
    bit-identical pair inputs.

    ``free`` / ``park``: None reads ``ROWPAIR_FREE_ZTRUNK`` / ``ROWPAIR_CONF_PARK_ZTRUNK`` (default 0); explicit values win. ``pool``: a
    :class:`opt_core.mem.torch_hostpair.PinPool` to share (the trunk's); None = a private :func:`.trunk.park_pool` (``ROWPAIR_PARK_PIN_MAX_GB``,
    ``ROWPAIR_PARK_STRICT``) built at the first park. ``z_shard``: the shard TENSOR handed to :func:`embed_rows` (``[R, N, C]`` /
    ``[1, R, N, C]``; rows on dim -3).

    Protocol (every rank runs the same passes)::

        plan = ZTrunkPlan(z_shard, passes=n)
        ...the distogram head's rows / column slabs from the device shard (sym_logit_rows refuses a parked source by name)...
        for i in range(n):
            plan.park_now()                                   # roll-out entry: parks when ROWPAIR_CONF_PARK_ZTRUNK (else a no-op word)
            z_cond = diffusion.pair_cond_rows(embed_fn, plan.source(), layout, ...)     # reads .zrows of the park, or the tensor
            ...the roll-out of pass i...
            src, inplace = plan.begin(i, lead_out=(k_i,), out_dtype=torch.float32)
            zc = embed_rows(fn, src, layout, lead_out=(k_i,), inplace=inplace, out_dtype=torch.float32, ...)
            ...the pass's pair stack / heads on zc...; del zc
            plan.end(i, device_needed_next=<a statement before begin(i+1) — or after the stage — reads the device shard whole>)

    :meth:`end`: parked + ``device_needed_next=True`` -> :meth:`.template.ParkedStorage.restore` re-grows the SAME storage and copies the rows
    back (every view the engine holds is valid again; the host copy is released, the next parked pass copies out again); parked +
    ``device_needed_next=False`` -> the park stays live for the next pass (no new copy), or, after the last pass, the host copy is dropped and the
    device storage stays released (``consumed``). ``inplace`` / ``resident``: no data moves. ``consumed`` is True once the device tensor has stopped
    holding the trunk rows (overwritten in place, or released and not restored): the adapter must not hand that tensor to a reader. Census
    (:func:`.evidence.record_schedule`, printed on the SCHEDULE / LEVER lines): ``conf_ztrunk`` = the per-pass words joined by ``,`` (``-`` for
    a pass not begun), ``conf_ztrunk_free`` / ``conf_ztrunk_park`` = the levers as read (0|1), ``conf_ztrunk_passes``, ``conf_ztrunk_host_gib`` = the
    largest host copy held."""

    __slots__ = ("z", "name", "passes", "free", "park", "strict", "words", "consumed", "host_bytes", "closed", "_pool", "_own_pool", "_parked",
                 "_lead_in", "_log", "_forms", "retired")

    def __init__(self, z_shard, *, passes: int = 1, free: Optional[bool] = None, park: Optional[bool] = None, pool=None,
                 strict: Optional[bool] = None, name: str = "z_trunk", log: Optional[Callable[[str], None]] = None):
        self.name = str(name)
        if not torch.is_tensor(z_shard) or z_shard.dim() < 3:
            raise RowpairRefused(f"ZTrunkPlan({self.name}): the trunk shard TENSOR [..., R, N, C] is required (got {type(z_shard).__name__}"
                                 f"{'' if not torch.is_tensor(z_shard) else ' ' + str(tuple(z_shard.shape))})", LEVER)
        self.passes = int(passes)
        if self.passes < 1:
            raise RowpairRefused(f"ZTrunkPlan({self.name}): passes={passes} < 1", LEVER)
        self.z = z_shard
        self.free = env_flag(ENV_FREE_ZTRUNK) if free is None else bool(free)
        self.park = env_flag(ENV_CONF_PARK_ZTRUNK) if park is None else bool(park)
        self.strict = strict
        self.words: List[Optional[str]] = [None] * self.passes
        self._forms: List[Optional[str]] = [None] * self.passes
        self.consumed, self.host_bytes, self.closed = False, 0, False
        self._pool, self._own_pool, self._parked, self._log = pool, pool is None, None, log
        self.retired: List[Optional[str]] = [None] * self.passes              # retire(i)'s word per pass; None = not retired
        self._lead_in = _lead_numel(tuple(z_shard.shape[:-3]))
        self._record()

    # --- internals ---------------------------------------------------------------------------------------------------------------
    def _say(self, msg: str) -> None:
        if self._log is not None:
            self._log(msg)

    def _index(self, i: int, what: str) -> int:
        i = int(i)
        if not 0 <= i < self.passes:
            raise RowpairRefused(f"ZTrunkPlan({self.name}).{what}({i}): pass index outside [0, {self.passes})", LEVER)
        if self.closed:
            raise RowpairRefused(f"ZTrunkPlan({self.name}).{what}({i}): the plan is closed (every pass ended)", LEVER)
        return i

    def _record(self) -> None:
        from .evidence import record_schedule
        record_schedule(conf_ztrunk=",".join(w if w is not None else "-" for w in self.words), conf_ztrunk_free=int(self.free),
                        conf_ztrunk_park=int(self.park), conf_ztrunk_passes=self.passes, conf_ztrunk_host_gib=round(self.host_gib, 3))

    @property
    def host_gib(self) -> float:
        """The largest host copy held so far, GiB (``host_bytes`` exact)."""
        return self.host_bytes / 2 ** 30

    # --- the statement -------------------------------------------------------------------------------------------------------------
    def decide(self, i: int, lead_out: Optional[Sequence[int]] = None, last_use: Optional[bool] = None, out_dtype=None) -> str:
        """PURE: the form word pass ``i`` gets — ``inplace`` | ``parked`` | ``resident[:<reason>[+<reason>]]`` — from the levers, ``lead_out``
        (the pass's leading dims; None = the shard's own) against the shard's leading numel, ``out_dtype`` (the pass's pair-input dtype; None =
        the shard's) against the shard's dtype, ``last_use`` (None: ``i == passes - 1``), the shard's contiguity / storage ownership, and whether
        a park is already live (the device tensor holds the rows only when no park is live: a live park keeps serving — never
        restore-then-inplace). No side effect: adapters call it BEFORE :meth:`begin` to learn whether the device tensor goes away this pass."""
        i = self._index(i, "decide")
        if self._parked is not None:
            return "parked"
        lead = self._lead_in if lead_out is None else _lead_numel(tuple(int(v) for v in lead_out))
        last = (i == self.passes - 1) if last_use is None else bool(last_use)
        reasons = []
        if self.free:
            if lead != self._lead_in:
                reasons.append("free_declined:samples")
            elif out_dtype is not None and out_dtype != self.z.dtype:
                reasons.append("free_declined:dtype")
            elif not last:
                reasons.append("free_declined:not_last")
            elif not self.z.is_contiguous():
                reasons.append("free_declined:not_contiguous")
            elif not storage_resizable(self.z):
                reasons.append("free_declined:foreign_storage")
            else:
                return "inplace"
        if self.park:
            if not owns_whole_storage(self.z):
                reasons.append("not_parked:shared_storage")
            elif not storage_resizable(self.z):
                reasons.append("not_parked:storage_not_resizable")
            else:
                return "parked"
        return "resident" + (":" + "+".join(reasons) if reasons else "")

    def begin(self, i: int, lead_out: Optional[Sequence[int]] = None, last_use: Optional[bool] = None, out_dtype=None):
        """Act on :meth:`decide` for pass ``i``; returns ``(src, inplace)`` for :func:`embed_rows` (``embed_rows(fn, src, layout, lead_out=...,
        inplace=inplace, out_dtype=...)``): ``src`` is the shard tensor (``inplace`` / ``resident``) or the live :class:`.template.ParkedStorage`
        (``parked``: the device storage is ALREADY released when this returns). Records the pass's census word."""
        i = self._index(i, "begin")
        if self.words[i] is not None:
            raise RowpairRefused(f"ZTrunkPlan({self.name}).begin({i}): pass {i} already begun ({self.words[i]})", LEVER)
        form = self.decide(i, lead_out, last_use, out_dtype)
        if form == "inplace":
            src, inplace, word = self.z, True, "inplace"
            self.consumed = True
        elif form == "parked":
            if self._parked is None:
                self._park_now(f"begin({i})")
            src, inplace, word = self._parked, False, "parked:" + self._parked.where
            self.consumed = True
        else:
            src, inplace, word = self.z, False, form
        self._forms[i], self.words[i] = form.split(":", 1)[0], word
        self._record()
        self._say(f"[conf] {self.name} pass {i + 1}/{self.passes}: {word} (free={int(self.free)} park={int(self.park)} "
                  f"lead_out={tuple(lead_out) if lead_out is not None else None} out_dtype={out_dtype})")
        return src, inplace

    def _park_now(self, what: str) -> None:
        from .template import ParkedStorage
        if self._pool is None:
            from .trunk import park_host                                       # ROWPAIR_HOST_SLAB unset: park_pool(pin=, strict=) as before; =lease: this
            self._pool = park_host(self.z, strict=self.strict)                 # rank's HostSlabLease
        parked = ParkedStorage(self.z, name=f"{self.name} (confidence)", log=self._log, pool=self._pool)
        if not parked.ok:                                                      # the caller proved ownership: a park that did not happen is a defect, loud
            raise RowpairRefused(f"ZTrunkPlan({self.name}).{what}: {parked.how}", LEVER)
        self._parked = parked
        self.host_bytes = max(self.host_bytes, int(parked.nbytes))
        self.consumed = True

    def park_now(self) -> str:
        """Park the shard NOW — before a pass's :meth:`begin`, at a ROLL-OUT ENTRY: the diffusion conditioning then reads the trunk rows through
        :meth:`source` (``.zrows``) while the shard's device storage is already released (one shard-equivalent off the roll-out's peak). Acts only
        when the park lever is set and the shard owns a whole resizable storage; returns the word (``parked:<where>``, or ``resident[:<reason>]``
        saying why nothing moved). Idempotent while a park is live; the following :meth:`begin` finds the live park (``parked``); :meth:`end` with
        ``device_needed_next=True`` restores as usual. Recorded as ``conf_ztrunk_entry``."""
        if self.closed:
            raise RowpairRefused(f"ZTrunkPlan({self.name}).park_now: the plan is closed (every pass ended)", LEVER)
        if self._parked is not None:
            word = "parked:" + self._parked.where
        elif not self.park:
            word = "resident"
        elif not owns_whole_storage(self.z):
            word = "resident:not_parked:shared_storage"
        elif not storage_resizable(self.z):
            word = "resident:not_parked:storage_not_resizable"
        else:
            self._park_now("park_now")
            word = "parked:" + self._parked.where
        from .evidence import record_schedule
        record_schedule(conf_ztrunk_entry=word)
        self._record()
        self._say(f"[conf] {self.name} roll-out entry: {word}")
        return word

    def retire(self, i: int, *, where: str = "embed") -> str:
        """Drop the trunk park EARLY — right after pass ``i``'s :func:`embed_rows` built its pair input, BEFORE that pass's pair
        stack runs — when ``i`` is the plan's LAST pass: nothing reads the trunk rows again (the distogram / contact rows were taken at entry, the
        heads read the pass's pair input), so the host copy's u bytes leave before the pair stack's row mirror asks for its u (under
        ``ROWPAIR_HOST_SLAB=lease`` the mirror then leases the slab the park just handed back: peak pinned 1u instead of 2u). Returns and records the
        word: ``pass<i>@<where>`` (dropped now) | ``kept:not_last`` (an earlier pass: the next roll-out's conditioning reads the park — nothing
        moves; ``ROWPAIR_CONF_ZTRUNK_SPILL`` is the designed lever for those) | ``nothing_parked`` (``inplace`` / ``resident``: no host copy exists).
        After a drop the plan is ``consumed``; :meth:`end` of that pass then only closes the plan, and ``end(i, device_needed_next=True)`` is
        refused by name (the rows are gone). Census ``ztrunk_retired=<word>`` (+ ``conf_ztrunk_retired``, the family-prefixed twin). Idempotent."""
        from .evidence import record_schedule
        i = self._index(i, "retire")
        if self.words[i] is None:
            raise RowpairRefused(f"ZTrunkPlan({self.name}).retire({i}): pass {i} was not begun", LEVER)
        if self.retired[i] is not None:
            return self.retired[i]
        if i != self.passes - 1:
            word = "kept:not_last"
        elif self._parked is None:
            word = "nothing_parked"
        else:
            gib = self._parked.nbytes / 2 ** 30
            self._parked.drop()
            self._parked = None
            self.consumed = True
            word = f"pass{i}@{where}"
            self._say(f"[conf] {self.name} pass {i + 1}/{self.passes}: park RETIRED at {where} ({gib:.2f} GiB host released before the pass's pair stack)")
        self.retired[i] = word
        record_schedule(ztrunk_retired=word, conf_ztrunk_retired=word)
        self._record()
        return word

    def source(self):
        """What to read trunk rows from RIGHT NOW: the live :class:`.template.ParkedStorage` (``.zrows``) while parked, else the shard tensor —
        the ``z_shard`` argument of :func:`.diffusion.pair_cond_rows` / :func:`embed_rows`. Refused by name once the rows are gone for good."""
        if self._parked is not None:
            return self._parked
        if self.consumed:
            raise RowpairRefused(f"ZTrunkPlan({self.name}).source: the device tensor does not hold the trunk rows any more (embedded in place or dropped)", LEVER)
        return self.z

    def end(self, i: int, *, device_needed_next: bool = False) -> None:
        """Close pass ``i`` (call it after the pass's pair input is dead). ``device_needed_next``: a statement before the next :meth:`begin` —
        or, for the last pass, after the stage — reads the device shard WHOLE: a parked shard is restored into its own storage (host copy
        released); an in-place pass cannot honour it (refused by name: the rows were overwritten — pass ``last_use=False`` to :meth:`begin`).
        Without it a park stays live for the next pass; after the last pass its host copy is dropped and the device storage stays released.
        The last pass's ``end`` closes the plan (:meth:`close`)."""
        i = self._index(i, "end")
        if self.words[i] is None:
            raise RowpairRefused(f"ZTrunkPlan({self.name}).end({i}): pass {i} was not begun", LEVER)
        last = i == self.passes - 1
        if self._forms[i] == "inplace" and device_needed_next:
            raise RowpairRefused(f"ZTrunkPlan({self.name}).end({i}, device_needed_next=True): pass {i} embedded the trunk shard IN PLACE — its rows "
                                 "are overwritten (begin(i, last_use=False) keeps them)", LEVER)
        if device_needed_next and self.retired[i] is not None and self.retired[i].startswith("pass"):
            raise RowpairRefused(f"ZTrunkPlan({self.name}).end({i}, device_needed_next=True): pass {i}'s park was RETIRED ({self.retired[i]}) — the trunk "
                                 "rows are gone (do not retire a pass whose device shard is read again)", LEVER)
        if self._parked is not None:
            if device_needed_next:
                self._parked.restore()
                self._parked = None
                self.consumed = False
                self._say(f"[conf] {self.name} pass {i + 1}/{self.passes}: restored to {self.z.device} (device_needed_next)")
            elif last:
                self._parked.drop()
                self._parked = None
        if last:
            self.close()
        self._record()

    def close(self) -> None:
        """Drop any live host copy (device storage stays released: ``consumed``) and mark the plan closed. Idempotent; the last :meth:`end` calls it."""
        if self._parked is not None:
            self._parked.drop()
            self._parked = None
            self.consumed = True
        self.closed = True

    @property
    def parked(self):
        """The live :class:`.template.ParkedStorage`, or None."""
        return self._parked

    def record(self) -> Dict[str, object]:
        """``{name, passes, free, park, words, forms, consumed, host_bytes, host_gib}`` — the plan's row of an adapter's record."""
        return {"name": self.name, "passes": self.passes, "free": bool(self.free), "park": bool(self.park), "words": list(self.words),
                "forms": list(self._forms), "consumed": bool(self.consumed), "host_bytes": int(self.host_bytes), "host_gib": round(self.host_gib, 4)}
