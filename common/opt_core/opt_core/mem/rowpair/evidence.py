"""The evidence of the row-sharded pair stack: the ``n_gpu=P sharding=<rowpair|none>`` fields the kit's ACTIVE / EXIT line carries (ONE
producer of that token text — the kit imports it), per-rank peak-memory fields for the memory ladder, the family's LEVER lines, and
rank-0-only emission.

API:
    sharding_value(n_gpu)                         ``rowpair`` when ``n_gpu > 1`` else ``none``
    fields(n_gpu, **more)                         ``[("n_gpu", P), ("sharding", rowpair|none), ...]`` — positional pairs for ``opt_core.report.kv`` /
                                                  ``report.line`` in THIS order (the launcher matches ``n_gpu=P sharding=rowpair``)
    fields_text(n_gpu, **more)                    the same rendered: ``n_gpu=2 sharding=rowpair``
    peak_fields(peaks_gib)                        ``peak_alloc_gib_max=<max over ranks> peak_alloc_gib_ranks=r0:<gib>,r1:<gib>`` from ``{rank: gib}``
    gather_peaks(device=None)                     every rank's ``torch.cuda.max_memory_allocated`` in GiB gathered to all ranks: ``{rank: gib}``
                                                  (collective; ``{0: gib}`` at P == 1; ``{}`` without CUDA)
    lever_line(tag, state, n_gpu, layout=, reason=, strategy=, **evidence)
                                                  the family's ``[tag] LEVER name=<LEVER> state=… impl=opt_core.mem.rowpair@<ver> origin=core strategy=<LEVER>
                                                  n_gpu=P sharding=… [N= P= B= R= …]`` via ``opt_core.report.lever_line``; sub-levers pass ``name=``
    emit_rank0(text)                              ``opt_core.report.emit`` on rank 0 (or a P == 1 run) only; returns the text everywhere
    rank_failed_line(tag, exc)                    ``[tag] ROWPAIR event=<rank_failed|rank_timeout> rank=<r> exitcode=<c> log=<path>`` for a :class:`launch.RankFailed`
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from . import LEVER, SHARDING, SUB_LEVERS, active_fields, active_pairs, check_n_gpu
from . import sharding_value as _sharding_value

__all__ = ["LEVER_NAME", "IMPL", "sharding_value", "fields", "fields_text", "peak_fields", "gather_peaks", "lever_line", "emit_rank0",
           "rank_failed_line", "rows_census", "arch_fields", "arch_state", "STRATEGY_IDS", "record_schedule", "reset_schedule",
           "schedule", "schedule_fields"]

LEVER_NAME = LEVER
STRATEGY_IDS = dict(SUB_LEVERS, tensor_parallel=LEVER)  # the strategy id of the TP lever and of every sub-lever (the LEVER lines' strategy= words)


def _impl() -> str:
    try:
        from ... import __version__ as v  # opt_core.__version__
    except Exception:  # noqa: BLE001
        v = "unknown"
    return f"opt_core.mem.rowpair@{v}"


IMPL = None  # rendered lazily by _impl() (the version file is the core's)


def sharding_value(n_gpu) -> str:
    return _sharding_value(n_gpu, SHARDING)


def fields(n_gpu, **more) -> List[Tuple[str, object]]:
    """The positional pairs ``[("n_gpu", P), ("sharding", "rowpair"|"none")] + more`` for a kit's ACTIVE / EXIT line (``opt_core.mem.ngpu``)."""
    return active_pairs(n_gpu, SHARDING) + list(more.items())


def fields_text(n_gpu, **more) -> str:
    if not more:
        return active_fields(n_gpu, SHARDING)
    from ...report import kv
    return kv(*fields(n_gpu, **more))


def peak_fields(peaks_gib: Dict[int, float]) -> List[Tuple[str, object]]:
    """``[("peak_alloc_gib_max", x), ("peak_alloc_gib_ranks", {"r0": .., "r1": ..})]`` from ``{rank: GiB}`` (empty dict -> ``none`` values)."""
    if not peaks_gib:
        return [("peak_alloc_gib_max", None), ("peak_alloc_gib_ranks", None)]
    mx = max(float(v) for v in peaks_gib.values())
    ranks = ",".join(f"r{int(r)}:{float(v):.2f}" for r, v in sorted(peaks_gib.items()))
    return [("peak_alloc_gib_max", round(mx, 2)), ("peak_alloc_gib_ranks", ranks)]


def gather_peaks(device=None) -> Dict[int, float]:
    """``{rank: torch.cuda.max_memory_allocated(device) / 2**30}`` over all ranks (``comm().allgather_obj``; every rank gets the dict)."""
    from ._torch import torch
    from .dist import comm, is_dist, world
    if not torch.cuda.is_available():
        return {}
    gib = float(torch.cuda.max_memory_allocated(device)) / 2 ** 30
    if not is_dist():
        return {0: gib}
    P, r = world()
    allv = comm().allgather_obj((r, gib))
    return {int(rr): float(g) for rr, g in allv}


def lever_line(tag: str, state: str, n_gpu, *, name: Optional[str] = None, layout=None, reason: Optional[str] = None,
               strategy: Optional[str] = LEVER, peaks_gib: Optional[Dict[int, float]] = None, with_schedule: bool = True, **evidence) -> str:
    """The family's LEVER line (``opt_core.report.lever_line`` grammar): ``name`` defaults to :data:`LEVER_NAME`; ``impl`` / ``origin=core`` are
    filled here; ``n_gpu`` / ``sharding`` lead the evidence, then the layout facts (``N P B R Rmax`` of this rank) when given, then per-rank
    peaks, then ``evidence`` in the caller's order."""
    from ...report import lever_line as core_lever_line
    pairs = fields(n_gpu)
    if layout is not None:
        f = layout.facts()
        pairs += [("N", f["N"]), ("P", f["P"]), ("B", f["B"]), ("rank", f["rank"]), ("rows", f"{f['r0']}:{f['r1']}"), ("R", f["R"]), ("Rmax", f["Rmax"])]
    if peaks_gib is not None:
        pairs += peak_fields(peaks_gib)
    if with_schedule:
        pairs += schedule_fields()
    return core_lever_line(tag, name or LEVER_NAME, state, *pairs, reason=reason, impl=_impl(), origin="core", strategy=strategy, **evidence)


def rows_census(layout) -> List[Tuple[str, object]]:
    """Collective: every rank's ``(r0, r1)`` all-gathered; ``[("ranks", P_alive), ("rows_tiled", true|false), ("row_bounds", "0:512,512:1024")]``
    — the proof in the run record that the ranks' rows tile ``0:N`` exactly once (P processes exiting 0 proves nothing)."""
    from .dist import comm, is_dist
    mine = (int(layout.rank), int(layout.r0), int(layout.r1))
    if is_dist():
        allv = comm().allgather_obj(mine)
    else:
        allv = [mine]
    allv = sorted(allv)
    tiled = allv[0][1] == 0 and allv[-1][2] == layout.N and all(allv[i][2] == allv[i + 1][1] for i in range(len(allv) - 1))
    return [("ranks", len(allv)), ("rows_tiled", bool(tiled)), ("row_bounds", ",".join(f"{a}:{b}" for _, a, b in allv))]


def arch_state(index: int = 0, lever: str = LEVER, strict: bool = False) -> dict:
    """``opt_core.arch.lever_state(lever, current_sm(index))``: ``{"state", "reason", "evidence" {sm, card_support|card_reason}, "verdict"}`` —
    the words the family's LEVER line carries on this card (the registry is the ONE reader; the family declares its levers at import)."""
    from ... import arch
    sm, _probe = arch.current_sm(int(index))
    return arch.lever_state(lever, sm, strict=strict)


def arch_fields(index: int = 0, lever: str = LEVER) -> List[Tuple[str, object]]:
    """``[("sm", smXY|none), ("card_support", word)]`` of ``lever`` on device ``index`` from the arch registry."""
    st = arch_state(index, lever)
    ev = dict(st["evidence"])
    return [("sm", ev.get("sm", st["verdict"].sm or "none")), ("card_support", ev.get("card_support", ev.get("card_reason", st["verdict"].word)))]


_SCHEDULE: Dict[str, object] = {}


def record_schedule(**facts) -> None:
    """Called by the primitives: the numerics-relevant schedule of this process's most recent calls (``trimul_RA``, ``trimul_RB``,
    ``tiles_source`` = budget | fixed, ``triatt_qrows``, ``a2a_chunks``, ``transpose_block_rows``, ``transpose_form`` = all | streamed:<peers> |
    inplace:<peers>, ``produce_block_rows``, ``layout_B``). A tile schedule derived from
    free device memory varies machine to machine; it is therefore PRINTED (``schedule_fields``, carried by :func:`lever_line` by default) — a kit pins
    ``RA`` / ``RB`` (arguments or ``ROWPAIR_TRIMUL_ROWS_A/B``) for reproducible draws and the line then says ``tiles_source=fixed``."""
    for k, v in facts.items():
        _SCHEDULE[str(k)] = v


def schedule() -> Dict[str, object]:
    """A copy of the recorded schedule facts (empty before any primitive ran)."""
    return dict(_SCHEDULE)


def reset_schedule() -> Dict[str, object]:
    """Clear the process-global schedule census (between requests / test cases); returns the facts it held."""
    old = dict(_SCHEDULE)
    _SCHEDULE.clear()
    return old


def schedule_fields() -> List[Tuple[str, object]]:
    """The recorded schedule as evidence pairs in a fixed key order (absent keys omitted)."""
    order = ("layout_B", "layout_align", "trimul_RA", "trimul_RB", "tiles_source", "a2a_chunks", "transpose_block_rows", "transpose_form", "produce_block_rows",
             "triatt_qrows", "conf_finish", "conf_rows")
    return [(k, _SCHEDULE[k]) for k in order if k in _SCHEDULE] + [(k, v) for k, v in sorted(_SCHEDULE.items()) if k not in order]



def schedule_line(tag: str, rank: int, world: int) -> str:
    """``[tag] SCHEDULE rank=q P=world k=v k=v …`` — every ``record_schedule`` word of THIS process (:func:`schedule_fields`) on one line: the
    as-run block sizes with their sources, sub-block / ring / host-gather choices and the replicated-by-design names. The worker launcher
    prints it per rank after the entry returns, so across-P census diffs can name the schedule; a kit that prints its own EXIT line calls
    :func:`schedule_fields` after the forward instead."""
    from ...report import line, prefix
    return line(prefix(tag), "SCHEDULE", rank=int(rank), P=int(world), **{k: v for k, v in schedule_fields()})

def memstats_fields() -> List[Tuple[str, object]]:
    """The CUDA caching-allocator census of THIS process as evidence pairs — ``allocated_peak_mb`` ``active_peak_mb`` ``reserved_peak_mb``
    ``inactive_split_peak_mb`` ``alloc_retries`` ``ooms`` (``torch.cuda.memory_stats``) + ``staging_peak_mb`` (the largest wire-buffer set a
    schedule held at once, :class:`opt_core.mem.rowpair.dist.Staging`). Empty when CUDA is absent or was never initialised in this process (nothing to report)."""
    from ._torch import torch
    if torch is None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return []
    s = torch.cuda.memory_stats()
    mb = lambda key: int(s.get(key, 0)) >> 20  # noqa: E731
    from .dist import comm, is_dist
    staging_peak_mb = -(-comm().staging.peak_bytes // (1 << 20)) if is_dist() else 0
    return [("allocated_peak_mb", mb("allocated_bytes.all.peak")), ("active_peak_mb", mb("active_bytes.all.peak")),
            ("reserved_peak_mb", mb("reserved_bytes.all.peak")), ("inactive_split_peak_mb", mb("inactive_split_bytes.all.peak")),
            ("alloc_retries", int(s.get("num_alloc_retries", 0))), ("ooms", int(s.get("num_ooms", 0))), ("staging_peak_mb", int(staging_peak_mb))]


def memstats_line(tag: str, rank: int, world: int) -> str:
    """``[tag] MEMSTATS rank=q P=world allocated_peak_mb=… active_peak_mb=… reserved_peak_mb=… inactive_split_peak_mb=… alloc_retries=… ooms=…
    staging_peak_mb=…`` — :func:`memstats_fields` on one line; ``""`` when there is nothing to report (CPU). ``active_peak ≫ allocated_peak`` names
    freed blocks a communication stream still held (the scoped wire buffers of :class:`opt_core.mem.rowpair.dist.Staging` free
    nothing in flight, which keeps the two equal up to transients)."""
    fields = memstats_fields()
    if not fields:
        return ""
    from ...report import line, prefix
    return line(prefix(tag), "MEMSTATS", rank=int(rank), P=int(world), **{k: v for k, v in fields})


def emit_rank0(text: str, stream=None) -> str:
    """Print ``text`` (``opt_core.report.emit``) only on the output rank (rank 0, or any P == 1 process); return it everywhere."""
    from ...report import emit
    from .launch import is_output_rank
    if is_output_rank():
        emit(text, stream)
    return text


def rank_failed_line(tag: str, exc) -> str:
    """``[tag] ROWPAIR event=<event> rank=<r> exitcode=<code|none> log=<path|none>`` for a :class:`opt_core.mem.rowpair.launch.RankFailed`."""
    from ...report import line, prefix
    return line(prefix(tag), "ROWPAIR", event=getattr(exc, "event", "rank_failed"), rank=getattr(exc, "rank", None),
                exitcode=getattr(exc, "exitcode", None), log=getattr(exc, "log", None))
