"""L1 — the communication surface, the row layout, and the bit-preserving data movement of a row-sharded pair tensor (tensors in, tensors
out; no arithmetic on payloads except integer checksums). torch / torch.distributed are reached lazily (:mod:`._torch`).

Conventions:
  * A pair tensor ``X[N, N, C]`` is ROW-SHARDED: rank q holds ``X[q0:q1, :, :]`` (contiguous), ``(q0, q1) = Layout.bounds[q]``. A kit whose
    pair tensor carries a leading batch dim passes ``z[0]`` (a view) to the dim-0 primitives of this module, or uses the ``[*, n_loc, N, C]``
    primitives of :mod:`.ring` (``dim=-3``).
  * Two row partitions, ONE type (:class:`Layout`): the P-INVARIANT GRID (``align=None``, the default: global rows grouped in blocks of ``B``;
    rank q owns blocks ``[q*bpr, (q+1)*bpr)``; trailing ranks may own fewer / zero rows) and the BALANCED CHUNK-ALIGNED partition
    (``align=k``: the ``ceil(N/k)`` chunks of ``k`` rows are dealt floor/ceil per rank, earlier ranks get the extra chunk, the last chunk is
    the ragged remainder — :func:`row_parts`). Under both every rank boundary is a multiple of ``B`` (``B == align`` under the aligned
    policy), so an engine chunk grid whose chunk size divides ``B`` never straddles a rank boundary (:meth:`Layout.chunks`).
  * THE COMM SURFACE: every collective of this package goes through the object :func:`comm` returns (``bcast_ p2p p2p_start allreduce_
    barrier allgather_obj broadcast_obj all_gather_into gather all_to_all_single checksum split_counts local``) — op modules never call
    ``torch.distributed``. Backends: ``nccl`` / ``gloo`` (the default process group :func:`init_from_env` creates), ``threaded`` (CPU tests:
    ranks are threads of one process sharing a :class:`ThreadHub`, :func:`opt_core.testing.run_ranks`), ``solo`` (P == 1: every method is
    the equality / a no-op and allocates nothing).
  * ``P == 1`` (or no initialised group) is always supported: equality semantics, no communication.

API:
    comm() / get()                                    the :class:`Comm` of the calling rank (thread-local under ``threaded``); ``.rank .world .P
                                                      .backend .active .stats``; ``with comm().local():`` makes ``.active`` False (rank-local
                                                      sections that call stock chunked code on shards)
    host_signal(key, value="1") / host_wait(key, timeout_s) -> str   a HOST-side key/value signal between ranks backed by the group's c10d
                                                      store (``file://$ROWPAIR_STORE`` under :mod:`.launch`): a rank idling while another process
                                                      works waits here on the CPU — no posted NCCL recv / barrier spinning a kernel. solo: signal
                                                      is kept in-process, wait returns immediately; threaded: the hub's table. A failed
                                                      wait is :class:`RowpairRefused` worded by ``host_wait_refusal(key, timeout_s,
                                                      waited_s, exc)``: ``no rank signalled it within <T>s`` (a timeout) or ``the store
                                                      failed after <t>s of a <T>s wait, before its timeout`` (the store unreachable —
                                                      under tcp://, its hosting rank exited; failures within ``HOST_WAIT_SLACK_S`` of the
                                                      deadline count as timeouts)
    init_from_env(backend, timeout_s, device_index)   the group from ``ROWPAIR_{RANK,WORLD,LOCAL_RANK,ADDR,PORT}`` (set by :mod:`.launch`;
                                                      torchrun's names as fallback): NCCL on CUDA (device bound and made current), gloo otherwise; returns
                                                      ``(P, rank, device)``; ``WORLD_SIZE`` unset/1 -> no group. ``destroy(failure=None)`` ends it ->
                                                      ``none|destroyed``: on a healthy process the library's destroy_process_group in the calling
                                                      thread, unbounded (it may wait for the peers' leave); on a FAILURE path (non-zero
                                                      ``exit_code``, ``failure()``, an exception propagating or already reported) bounded by
                                                      ``ROWPAIR_ABORT_TIMEOUT_S`` (60 s) — on overrun one ``RANKLEAVE`` line, pending EXIT tallies
                                                      and ``os._exit(exit_code() else 70)`` there and then. Joining a group registers that leave
                                                      ``atexit`` and arms the EXIT GUARD, both live on failure paths only: ``ROWPAIR_EXIT_GRACE_S``
                                                      (90 s) after the main thread ended a FAILED process still in its exit hooks / threads is ended
                                                      (``RANKSTUCK`` + thread stacks, ``os._exit``); a healthy process is never touched
    Layout(N, P, rank, B=128, align=None)             the row partition: ``r0 r1 R Rmax bounds replicated B align policy`` + the short aliases
                                                      ``parts n_loc n_max rows_of(q)`` the row-sharded statements spell, ``rows(q) nrows(q)
                                                      chunks(chunk) local_blocks(rows) owner(i)``; :meth:`Layout.checked` refuses BY NAME a
                                                      partition that replicates (grid: ``N < P*B``) or leaves a rank with zero rows;
                                                      :func:`layout_here` / :func:`ctx` build it from the calling rank's comm
    row_parts(N, P, align)                            the balanced chunk-aligned partition (pure function)
    is_dist() / barrier() / world()                   group predicates (thread-aware)
    all_gather_rows(x_shard, layout)                  ``[R, ...] -> [N, ...]`` on every rank
    gather_rows_to_rank0(x_shard, layout)             ``[R, ...] -> [N, ...]`` on rank 0 only (None elsewhere)
    gather_cat_to_rank0(piece, counts)                variable-length row concat to rank 0
    alltoall_window(provider, layout, off, width, …)  the generalised distributed transpose window (``rows`` / ``gemm_a`` / ``gemm_b`` layouts)
    transpose_shards(z_shard, layout, chunks, …)      ``zT_shard[i-r0, j, :] = Z[j, i, :]`` — whole shard (optionally in output row windows)
    transpose_blocks(z_shard, layout, step)           the same, streamed: yields ``(i0, i1, zT_blk)`` per local row block
    ring_blocks(x_shard, layout, rows=, dim=)         every rank's block streams around the ring: yields ``(q, x_q)`` once per rank,
                                                      double-buffered ``p2p_start``; MUST be exhausted on every rank
    allreduce_(t, op)                                 in-place sum / max / min over ranks (the comm surface's, module-level spelling)
    allreduce_checksum(t, name)                       assert a REPLICATED tensor is bit-identical on all ranks (integer fingerprint)
    checksum(t) / broadcast_obj(obj, src) / agreed_free_bytes(device) / env_int / env_float
    zrows / zwrite / zadd / zmeta / zlen / zblocks    storage-agnostic shard accessors (a plain tensor, or a duck-typed store with those methods)

Replicated by design (never sharded by this package): the single representation, the MSA representation, atom tensors — the modules that
consume them name them in their docstrings and the schedule census.

Environment knobs (data movement only): ``ROWPAIR_A2A_CHUNKS`` (channel groups staging an all-to-all, default 4: the send+recv transient
is 2/chunks of the moved tensor), ``ROWPAIR_TRANSPOSE_BLOCK_ROWS`` (0 = whole shard per exchange), ``ROWPAIR_ZROWS_BLOCK`` (128),
``ROWPAIR_CHUNK_ALIGN`` (the aligned policy's ``align`` for :func:`default_ctx`; the adapter sets it from the engine's pinned chunk size),
``ROWPAIR_VERBOSE`` (comm log lines, default 1). No knob of this module selects a replicated or gathered path by size: every such
choice is an explicit argument an adapter passes.
"""
from __future__ import annotations

import bisect
import contextlib
import os
import sys
import threading
import time
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from . import RowpairRefused
from ._torch import dist, torch

__all__ = ["Comm", "ThreadHub", "comm", "get", "register_thread_comm", "unregister_thread_comm", "host_signal", "host_wait", "host_wait_refusal", "HOST_KEY_PREFIX",
           "init_from_env", "destroy", "exit_code", "failure", "arm_exit_guard", "rankleave_line", "rankstuck_line", "ABORT_TIMEOUT_S", "EXIT_GRACE_S", "EXIT_TEARDOWN_STUCK", "ENV_ABORT_TIMEOUT", "ENV_EXIT_GRACE", "Layout", "row_parts",
           "layout_here", "ctx", "default_ctx", "chunk_align", "clear_layout_cache", "require_sharded", "B_CANDIDATES", "is_dist", "barrier", "world",
           "all_gather_rows", "gather_rows_to_rank0", "gather_rows_to_rank0_host", "gather_cat_to_rank0_host", "host_gather_cols", "gather_cat_to_rank0", "transpose_shards", "transpose_blocks", "alltoall_window", "ring_blocks",
           "allreduce_", "allreduce_checksum", "checksum", "broadcast_obj", "agreed_free_bytes", "zrows", "zwrite", "zadd", "zmeta", "zlen", "zblocks",
           "zclone", "zalloc_like", "ZBLOCK", "env_int", "env_float", "env_flag", "POLICIES", "Staging", "release_staging",
           "ENV_RANK_THREADS", "RANK_THREAD_VARS", "visible_cpus", "resolve_cpu_threads", "rank_thread_env", "apply_rank_threads", "cpu_threads"]

ZBLOCK = 128
B_CANDIDATES = (128, 64, 32, 16)          # Layout.auto tries these block sizes, largest first
POLICIES = ("grid", "aligned")            # Layout.policy: the P-invariant B-grid | the balanced chunk-aligned partition
BACKENDS = ("nccl", "gloo", "threaded", "solo")
_REDUCE_OPS = ("sum", "max", "min")
HOST_KEY_PREFIX = "rowpair/host/"             # host_signal / host_wait keys in the group's c10d store
HOST_WAIT_S = 1800.0
ENV_P2P_CHECK = "ROWPAIR_P2P_CHECK"          # "1": every point-to-point site checks its transfers are matched across ranks before posting (Comm.check_p2p)
ENV_P2P_SYNC = "ROWPAIR_P2P_SYNC"            # "1": every point-to-point wait is followed by a current-stream synchronize (events retire deterministically)


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    if not str(v).strip():
        return int(default)
    try:
        return int(str(v).strip())
    except ValueError:
        raise ValueError(f"{name}={v!r}: an integer is required") from None



def env_flag(name: str, default: bool = False) -> bool:
    """A 0/1 environment flag by its FULL name: unset or empty -> ``default``; ``0 / false / no / off`` (any case) -> False; anything else -> True.
    The one flag parser of the family."""
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return bool(default)
    return v not in ("0", "false", "no", "off")

def env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "")
    if not str(v).strip():
        return float(default)
    try:
        return float(str(v).strip())
    except ValueError:
        raise ValueError(f"{name}={v!r}: a number is required") from None


# ----------------------------------------------------------------------------------------------------------------- the per-rank CPU-thread cap
ENV_RANK_THREADS = "ROWPAIR_RANK_THREADS"      # unset | inherit | auto | <n>: the intra-op / OpenMP thread count of ONE rank process of a P-rank launch
RANK_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def visible_cpus() -> int:
    """The CPUs this process may run on (the affinity mask; ``os.cpu_count()`` where the platform has no mask), at least 1."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, int(os.cpu_count() or 1))


def resolve_cpu_threads(spec, local_world: Optional[int] = None, cpus: Optional[int] = None) -> Tuple[Optional[int], str]:
    """``(n | None, source)`` of a thread-count spec: ``None`` / ``""`` / ``inherit`` / ``0`` -> ``(None, "inherit")`` (touch nothing: today's
    behaviour); ``auto`` -> ``(max(1, cpus // local_world), "auto")`` (``cpus`` = :func:`visible_cpus`, ``local_world`` = the ranks of this node,
    default this process's world); an integer ``n >= 1`` (or its decimal string) -> ``(n, "given")``. Anything else is a ``ValueError`` naming
    ``ROWPAIR_RANK_THREADS``. Why the lever exists: P rank processes that each inherit the container's ``OMP_NUM_THREADS`` (= every core) run
    ``P x cores`` OpenMP threads on ``cores`` cores the moment they enter a threaded CPU op together (e.g. 8 x 64 on a 64-core host)."""
    if spec is None:
        return None, "inherit"
    if isinstance(spec, bool):
        raise ValueError(f"{ENV_RANK_THREADS}={spec!r}: inherit | auto | <n >= 1> is required")
    if isinstance(spec, int):
        if spec < 0:
            raise ValueError(f"{ENV_RANK_THREADS}={spec!r}: inherit | auto | <n >= 1> is required")
        return (None, "inherit") if spec == 0 else (int(spec), "given")
    word = str(spec).strip().lower()
    if word in ("", "inherit", "0"):
        return None, "inherit"
    if word == "auto":
        n_cpu = int(cpus) if cpus is not None else visible_cpus()
        if local_world is None:
            from .launch import world_size
            local_world = world_size()
        return max(1, n_cpu // max(1, int(local_world))), "auto"
    try:
        n = int(word)
    except ValueError:
        raise ValueError(f"{ENV_RANK_THREADS}={spec!r}: inherit | auto | <n >= 1> is required") from None
    if n < 1:
        raise ValueError(f"{ENV_RANK_THREADS}={spec!r}: inherit | auto | <n >= 1> is required")
    return n, "given"


def rank_thread_env(spec, local_world: int, cpus: Optional[int] = None) -> Dict[str, str]:
    """The environment DELTA of one rank process for the cap ``spec`` (:func:`resolve_cpu_threads`): ``{}`` for inherit, else
    ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS`` = ``n`` and ``ROWPAIR_RANK_THREADS`` = ``n`` (RESOLVED, so the rank
    applies exactly the count its launcher chose and names it in its census). A launcher lays it over the rank's environment BEFORE the
    interpreter starts (OpenMP reads its variable once, at first use)."""
    n, _src = resolve_cpu_threads(spec, local_world=local_world, cpus=cpus)
    if n is None:
        return {}
    out = {k: str(n) for k in RANK_THREAD_VARS}
    out[ENV_RANK_THREADS] = str(n)
    return out


def apply_rank_threads(spec=None, local_world: Optional[int] = None) -> Tuple[Optional[int], str]:
    """In a RANK process: resolve ``spec`` (default: this process's ``ROWPAIR_RANK_THREADS``), and when it names a count set
    ``torch.set_num_threads(n)``, export the three thread variables into this process's environment (children and lazily initialised BLAS
    pools inherit them) and record the census words ``rank_threads=<n> rank_threads_source=<auto|given>``. ``(None, "inherit")`` touches
    nothing and records nothing (today's behaviour, byte for byte)."""
    if spec is None:
        spec = os.environ.get(ENV_RANK_THREADS)
    n, src = resolve_cpu_threads(spec, local_world=local_world)
    if n is None:
        return None, src
    for k in RANK_THREAD_VARS:
        os.environ[k] = str(n)
    os.environ[ENV_RANK_THREADS] = str(n)
    torch.set_num_threads(int(n))
    from .evidence import record_schedule
    record_schedule(rank_threads=int(n), rank_threads_source=src)
    return int(n), src


@contextlib.contextmanager
def cpu_threads(n=None, local_world: Optional[int] = None):
    """``with cpu_threads(n):`` — the torch intra-op thread count for the scope (restored on exit): ``None`` / ``inherit`` / ``0`` = no-op,
    ``auto`` = ``cpus // local_world``, an integer = that many. The scope form of the lever for a rank's CPU-heavy section (the confidence
    reducer's index build / finish); the process-wide form is :func:`apply_rank_threads` / the launcher's ``rank_env(cpu_threads=)``."""
    k, _src = resolve_cpu_threads(n, local_world=local_world)
    if k is None:
        yield None
        return
    prev = int(torch.get_num_threads())
    torch.set_num_threads(int(k))
    try:
        yield int(k)
    finally:
        torch.set_num_threads(prev)


# ----------------------------------------------------------------------------------------------------------------- the comm surface
def record_p2p_check() -> None:
    """The schedule census names the point-to-point check when it ran (``p2p_check=on``)."""
    from .evidence import record_schedule                                    # noqa: WPS433 (evidence imports dist)
    record_schedule(p2p_check="on")


class Staging(object):
    """The WIRE BUFFERS of one comm, SCOPED to one call / one ring pass: every point-to-point / all-to-all schedule (``ring_blocks``,
    ``ring_pass``, the shard transposes, ``transpose_band``, ``alltoall_window``) takes its send / receive staging — including a copy of the
    OWN block it sends, so no caller tensor ever goes on the wire — from :meth:`buffer` between :meth:`enter` and :meth:`exit` of its site;
    buffers are reused across the steps / chunks inside that scope and FREED at ``exit`` after the current stream has been synchronized (nccl),
    i.e. only once every transfer that touched them has completed. Nothing the process group ``recordStream``-ed is therefore ever freed with a
    transfer in flight (a freed block whose transfer event has not retired stays 'active' in the caching allocator and pushes the next
    allocation into new memory), no buffer outlives its call, and the footprint is one call's wire set (``ring_blocks`` / ``ring_pass``: two
    padded blocks + the own copy; ``transpose_shard``: (P-1) send + (P-1) receive blocks, ``peers_in_flight`` pairs for the streamed / in-place
    forms; ``transpose_band``: one pair per peer window; ``alltoall_window``: one ``[P, Rmax, W, C/chunks]`` pair). Census (P > 1 only):
    ``staging_peak_mb`` (the largest wire set held at once) ``staging_allocs`` ``staging_syncs``."""

    def __init__(self, comm_obj=None):
        self._bufs: Dict[tuple, object] = {}
        self._live: Dict[str, int] = {}
        self._comm = comm_obj                                                    # its ``backend`` decides whether exit() synchronizes a CUDA stream
        self.allocs = 0
        self.syncs = 0
        self.peak_bytes = 0

    def enter(self, site: str) -> str:
        """A scope token for ``site``; nested / concurrent scopes of one site (two live rings) get distinct slots. Pair with :meth:`exit`."""
        depth = self._live.get(site, 0)
        self._live[site] = depth + 1
        return site if depth == 0 else f"{site}#{depth}"

    def buffer(self, token: str, slot: str, shape, dtype, device):
        """A contiguous tensor of ``shape`` from the scope's storage of ``(token, slot, dtype, device)`` (allocated on first use in the scope,
        re-viewed afterwards; a larger request re-allocates)."""
        shape = [int(s) for s in shape]
        numel = 1
        for s in shape:
            numel *= s
        dev = torch.device(device)
        key = (token, slot, dtype, str(dev))
        t = self._bufs.get(key)
        if t is None or t.numel() < numel:
            t = torch.empty(max(1, numel), dtype=dtype, device=dev)
            self._bufs[key] = t
            self.allocs += 1
            nb = self.nbytes()
            if nb > self.peak_bytes:
                self.peak_bytes = nb
            self._record()
        return t[:numel].view(shape)

    def exit(self, token: str) -> None:
        """End of the scope: synchronize the current stream (nccl: every transfer that used the scope's buffers has completed) and free them."""
        mine = [k for k in self._bufs if k[0] == token]
        if mine:
            if getattr(self._comm, "backend", "") == "nccl":
                torch.cuda.current_stream().synchronize()
                self.syncs += 1
            for k in mine:
                del self._bufs[k]
            self._record()
        base = token.split("#", 1)[0]
        n = self._live.get(base, 0) - 1
        if n <= 0:
            self._live.pop(base, None)
        else:
            self._live[base] = n

    def nbytes(self) -> int:
        return int(sum(int(t.numel()) * int(t.element_size()) for t in self._bufs.values()))

    def stats(self) -> Dict[str, int]:
        return {"bufs": len(self._bufs), "bytes": self.nbytes(), "allocs": int(self.allocs), "syncs": int(self.syncs), "peak_bytes": int(self.peak_bytes)}

    def release(self, reason: str = "explicit") -> int:
        """Drop every buffer of every scope (comm teardown); returns the bytes released."""
        n = self.nbytes()
        if self._bufs:
            self._bufs.clear()
            self._record()
        return n

    def _record(self) -> None:
        from .evidence import record_schedule                                    # noqa: WPS433 (evidence imports dist)
        record_schedule(staging_peak_mb=-(-int(self.peak_bytes) // (1 << 20)), staging_allocs=int(self.allocs), staging_syncs=int(self.syncs))


def release_staging(reason: str = "explicit") -> int:
    """Release the calling rank's wire buffers (:class:`Staging`); 0 outside a scope (every site frees its own at exit) and at P == 1."""
    if not is_dist():
        return 0
    return comm().staging.release(reason)


def _p2p_sync(cm) -> None:
    """``ROWPAIR_P2P_SYNC=1``: after a point-to-point wait on nccl, synchronize the current stream so the transfers' events have retired."""
    if getattr(cm, "backend", "") == "nccl" and env_flag(ENV_P2P_SYNC, False):
        torch.cuda.current_stream().synchronize()
        from .evidence import record_schedule                                    # noqa: WPS433
        record_schedule(p2p_sync="on")


class _Handle(object):
    """Returned by :meth:`Comm.p2p_start`; ``wait()`` blocks until the transfers are complete (idempotent)."""

    __slots__ = ("_works", "_comm", "_nbytes", "_t0", "_done")

    def __init__(self, works, comm_obj, nbytes: int):
        self._works, self._comm, self._nbytes, self._t0 = works, comm_obj, int(nbytes), time.perf_counter()
        self._done = works is None

    def wait(self) -> None:
        if self._done:
            return
        for w in self._works:
            w.wait()
        _p2p_sync(self._comm)
        self._comm._acct(self._nbytes, self._t0)
        self._done = True


def _nbytes_of(tensors) -> int:
    return int(sum(int(t.numel()) * int(t.element_size()) for t in tensors))


def _check_contiguous(*groups) -> None:
    for g in groups:
        for t in g:
            if not t.is_contiguous():
                raise RowpairRefused(f"comm: a {tuple(t.shape)} tensor handed to a collective is not contiguous")


HOST_WAIT_SLACK_S = 2.0                       # a store wait that fails this much before its deadline did not time out: the store itself failed


def host_wait_refusal(key: str, timeout_s: float, waited_s: float, exc: BaseException) -> str:
    """The refusal text of a failed store wait in :meth:`Comm.host_wait`. Two cases, told apart by the store's words and by WHEN it failed:
    the wait ran out its timeout — ``no rank signalled it within <T>s`` (the peer is slow or never reached its signal; under the launcher's
    ``file://`` rendezvous a peer that died before signalling also surfaces here, at the timeout) — or the wait failed more than
    :data:`HOST_WAIT_SLACK_S` before its deadline with no timeout in the store's words: the store itself is unreachable — under a ``tcp://``
    rendezvous that is the rank hosting the store having exited before it signalled."""
    name = type(exc).__name__
    text = " ".join(str(exc).split())
    timed_out = "timeout" in name.lower() or "timeout" in text.lower() or "timed out" in text.lower() or waited_s >= float(timeout_s) - HOST_WAIT_SLACK_S
    if timed_out:
        return f"host_wait({key!r}): no rank signalled it within {float(timeout_s):.1f}s ({name})"
    return (f"host_wait({key!r}): the store failed after {waited_s:.1f}s of a {float(timeout_s):.1f}s wait, before its timeout — the store is "
            f"unreachable (under a tcp:// rendezvous: the rank hosting it exited before signalling) ({name}: {text[:200]})")


class Comm(object):
    """The communication surface of one rank. This base class IS the ``solo`` backend (P == 1): every method is the equality / a no-op and
    allocates nothing. Subclasses: the default process group (:class:`_GroupComm`, nccl | gloo) and the threaded test backend
    (:class:`_ThreadComm`). Every rank issues the same sequence of collectives (NCCL discipline; the threaded hub enforces it by rendezvous).

    Attributes: ``rank``, ``world`` (== ``P``), ``backend``, ``verbose`` (``ROWPAIR_VERBOSE``), ``stats`` (``comm_calls comm_bytes comm_s`` +
    free-form counters op modules add), ``active`` (``world > 1`` and not inside :meth:`local`)."""

    backend = "solo"

    def __init__(self, rank: int = 0, world: int = 1):
        self.rank, self.world = int(rank), int(world)
        self._local_depth = 0
        self.verbose = env_int("ROWPAIR_VERBOSE", 1) != 0
        self.stats: Dict[str, object] = {"comm_calls": 0, "comm_bytes": 0, "comm_s": 0.0}
        self._signals: Dict[str, str] = {}
        self.staging = Staging(self)                                         # the scoped wire buffers of this rank (P > 1 schedules)

    # ---- state
    @property
    def P(self) -> int:
        return self.world

    @property
    def active(self) -> bool:
        return self.world > 1 and self._local_depth == 0

    @contextlib.contextmanager
    def local(self):
        """Inside ``with comm().local():`` every TP-aware statement behaves as single-device stock (``active`` is False): use around
        rank-local sections that call an engine's own chunked code on a shard."""
        self._local_depth += 1
        try:
            yield self
        finally:
            self._local_depth -= 1

    @property
    def device(self):
        """The device payloads of this comm's collectives live on (``cuda:<current>`` for nccl, cpu otherwise)."""
        return torch.device("cpu")

    @property
    def has_all_to_all(self) -> bool:
        """True when :meth:`all_to_all_single` with split sizes is native (nccl, threaded); gloo composes it from p2p (equal blocks only)."""
        return True

    def _acct(self, nbytes: int, t0: float) -> None:
        s = self.stats
        s["comm_calls"] = int(s["comm_calls"]) + 1
        s["comm_bytes"] = int(s["comm_bytes"]) + int(nbytes)
        s["comm_s"] = float(s["comm_s"]) + (time.perf_counter() - t0)

    def count(self, key: str, n: int = 1) -> None:
        """``stats[key] += n`` (op modules' free-form counters: split calls, trimul calls ...)."""
        self.stats[key] = int(self.stats.get(key, 0)) + int(n)

    def log(self, msg: str) -> None:
        """``[rowpair r<rank>] msg`` on stderr when ``verbose``."""
        if self.verbose:
            sys.stderr.write(f"[rowpair r{self.rank}] {msg}\n")
            sys.stderr.flush()

    # ---- partition helpers (pure functions of shapes -> identical on all ranks)
    def split_counts(self, n_items: int) -> List[Tuple[int, int]]:
        """Balanced contiguous partition of ``n_items`` indices over ranks: ``[(k0, k1)] * P``."""
        P, n = self.world, int(n_items)
        return [((r * n) // P, ((r + 1) * n) // P) for r in range(P)]

    # ---- collectives: solo semantics
    def bcast_(self, t, src: int):
        """In-place broadcast of a contiguous tensor from rank ``src``; returns ``t``."""
        _check_contiguous((t,))
        return t

    def check_p2p(self, sends: dict, recvs: dict, what: str = "") -> None:
        """``ROWPAIR_P2P_CHECK=1`` (default off): BEFORE a point-to-point exchange is posted, every rank publishes its ``(peer, numel, dtype)``
        send and receive sets (one object all-gather — every rank of the group must reach the site, which the rendezvous discipline already
        requires) and the exchange is REFUSED BY NAME on every rank when any send→q has no receive←sender on q with the same numel and dtype, or
        any receive has no matching send: an unmatched transfer is a device-side hang otherwise (a fused NCCL group never completes). ``what``
        names the call site. Off: no collective, no cost."""
        if self.world <= 1 or not env_flag(ENV_P2P_CHECK, False):
            return
        mine = {"rank": int(self.rank), "what": str(what),
                "send": sorted((int(q), int(t.numel()), str(t.dtype)) for q, t in sends.items()),
                "recv": sorted((int(q), int(t.numel()), str(t.dtype)) for q, t in recvs.items())}
        allv = self.allgather_obj(mine)
        record_p2p_check()
        bad = []
        for e in allv:
            r = int(e["rank"])
            for q, n, dt in e["send"]:
                peer = allv[q] if 0 <= q < len(allv) else None
                if peer is None or (r, n, dt) not in {tuple(x) for x in peer["recv"]}:
                    bad.append(f"rank {r} sends {n} x {dt} to rank {q} [{e['what']}] but rank {q} posts no matching receive"
                               + (f" (its receives: {peer['recv']} [{peer['what']}])" if peer is not None else " (no such rank)"))
            for q, n, dt in e["recv"]:
                peer = allv[q] if 0 <= q < len(allv) else None
                if peer is None or (r, n, dt) not in {tuple(x) for x in peer["send"]}:
                    bad.append(f"rank {r} expects {n} x {dt} from rank {q} [{e['what']}] but rank {q} posts no matching send"
                               + (f" (its sends: {peer['send']} [{peer['what']}])" if peer is not None else " (no such rank)"))
        sites = sorted({str(e["what"]) for e in allv})
        if len(sites) > 1:
            bad.append(f"ranks are at DIFFERENT p2p sites in the same step: {[(int(e['rank']), e['what']) for e in allv]}")
        if bad:
            import traceback
            where = "".join(traceback.format_stack(limit=8)[:-1])
            raise RowpairRefused(f"p2p[{what}] (ROWPAIR_P2P_CHECK): unmatched point-to-point transfers — " + "; ".join(bad[:6])
                                 + (f" (+{len(bad) - 6} more)" if len(bad) > 6 else "") + f"\ncall site (rank {self.rank}):\n{where}")

    def check_replicated_arg(self, value, what: str) -> None:
        """``ROWPAIR_P2P_CHECK=1``: refuse by name unless the picklable ``value`` (a windows / shapes / rows list a p2p schedule is keyed on) is
        IDENTICAL on every rank — a rank-dependent schedule argument is the unmatched-transfer class. Off: no collective."""
        if self.world <= 1 or not env_flag(ENV_P2P_CHECK, False):
            return
        allv = self.allgather_obj(value)
        record_p2p_check()
        if any(v != allv[0] for v in allv):
            raise RowpairRefused(f"{what} (ROWPAIR_P2P_CHECK): the schedule argument differs across ranks (must be identical on every rank): "
                                 + "; ".join(f"rank {q}: {v}" for q, v in enumerate(allv))[:2000])

    def p2p(self, sends: dict, recvs: dict, what: str = "") -> None:
        """``sends`` / ``recvs``: ``{peer_rank: contiguous tensor}``. All listed transfers complete before return; a rank with nothing to move
        still CALLS (rendezvous discipline). ``what`` names the call site (:meth:`check_p2p`)."""
        if sends or recvs:
            raise RowpairRefused(f"comm(solo): p2p with peers {sorted(set(sends) | set(recvs))} at world size 1")

    def p2p_start(self, sends: dict, recvs: dict, what: str = "") -> _Handle:
        """Asynchronous :meth:`p2p`: returns a handle with ``.wait()``; buffers stay alive and unmodified until then."""
        self.p2p(sends, recvs, what)
        return _Handle(None, self, 0)

    def allreduce_(self, t, op: str = "sum"):
        """In-place reduction over ranks (``sum`` | ``max`` | ``min``); returns ``t``."""
        if op not in _REDUCE_OPS:
            raise RowpairRefused(f"comm: allreduce op {op!r} not in {_REDUCE_OPS}")
        return t

    def barrier(self) -> None:
        return None

    def allgather_obj(self, obj) -> list:
        """``[obj_0, ..., obj_{P-1}]`` (picklable objects) on every rank."""
        return [obj]

    def broadcast_obj(self, obj, src: int = 0):
        """``obj`` of rank ``src`` on every rank."""
        return obj

    def all_gather_into(self, out, buf) -> None:
        """``out[q*u:(q+1)*u] = buf of rank q`` for every q (``u = buf.shape[0]``; equal blocks along dim 0; both contiguous)."""
        _check_contiguous((out, buf))
        out.copy_(buf)

    def gather(self, buf, slots: Optional[list], dst: int = 0) -> None:
        """Rank ``dst``: ``slots[q].copy_(buf of rank q)`` for every q (``slots`` = P tensors shaped like ``buf``); other ranks pass ``slots=None``."""
        _check_contiguous((buf,))
        slots[0].copy_(buf)

    def all_to_all_single(self, out, inp, out_sizes: Optional[Sequence[int]] = None, in_sizes: Optional[Sequence[int]] = None) -> None:
        """``torch.distributed.all_to_all_single`` semantics along dim 0 (equal blocks when the size lists are None)."""
        _check_contiguous((out, inp))
        out.copy_(inp)

    def checksum(self, t, tag: str = "") -> bool:
        """Debug predicate: True iff a REPLICATED tensor's (fp64 sum, |max|) agree on all ranks (logs on rank 0 or on mismatch).
        The raising form is :func:`allreduce_checksum` (integer fingerprint)."""
        if self.world <= 1:
            return True
        with torch.no_grad():
            s = torch.stack([t.double().sum(), t.double().abs().max()]).to(t.device)
            mn = self.allreduce_(s.clone(), "min")
            mx = self.allreduce_(s.clone(), "max")
            same = bool(torch.equal(mn, mx))
            if self.rank == 0 or not same:
                self.log(f"checksum[{tag}] identical_across_ranks={same} sum={s[0].item():.10e} maxabs={s[1].item():.6e} shape={tuple(t.shape)}")
            return same

    def report(self) -> str:
        s = self.stats
        line = (f"comm stats: backend={self.backend} world={self.world} calls={s['comm_calls']} bytes={int(s['comm_bytes']) / 2 ** 30:.2f} GiB "
                f"host_wait={float(s['comm_s']):.2f} s " + " ".join(f"{k}={v}" for k, v in sorted(s.items()) if k not in ("comm_calls", "comm_bytes", "comm_s")))
        self.log(line)
        return line

    # ---- host-side signals (CPU only; never a device collective)
    def host_signal(self, key: str, value: str = "1") -> None:
        """Publish ``key = value`` for the other ranks' :meth:`host_wait` (solo: kept in this process)."""
        self._signals[str(key)] = str(value)

    def host_wait(self, key: str, timeout_s: float = HOST_WAIT_S) -> str:
        """Block on the HOST until some rank signalled ``key``; returns its value. solo: returns immediately (the value if this process
        signalled it, else ``""`` — no peer exists that could). A timeout is :class:`RowpairRefused` naming the key."""
        return self._signals.get(str(key), "")

    # ---- p2p-composed forms subclasses without a native collective use (identical on every rank: decided by backend, never by an error)
    def _all_gather_into_p2p(self, out, buf) -> None:
        u, me = int(buf.shape[0]), self.rank
        out[me * u:(me + 1) * u].copy_(buf)
        self.p2p({q: buf for q in range(self.world) if q != me}, {q: out[q * u:(q + 1) * u] for q in range(self.world) if q != me}, "all_gather_into(p2p)")

    def _gather_p2p(self, buf, slots, dst: int) -> None:
        me = self.rank
        if me == dst:
            slots[me].copy_(buf)
            self.p2p({}, {q: slots[q] for q in range(self.world) if q != me}, "gather(p2p)")
        else:
            self.p2p({dst: buf}, {}, "gather(p2p)")

    def _all_to_all_p2p(self, out, inp, out_sizes, in_sizes) -> None:
        P, me = self.world, self.rank
        if in_sizes is None:
            in_sizes = [int(inp.shape[0]) // P] * P
        if out_sizes is None:
            out_sizes = [int(out.shape[0]) // P] * P
        io = [0]
        for n in in_sizes:
            io.append(io[-1] + int(n))
        oo = [0]
        for n in out_sizes:
            oo.append(oo[-1] + int(n))
        out[oo[me]:oo[me + 1]].copy_(inp[io[me]:io[me + 1]])
        sends = {q: inp[io[q]:io[q + 1]] for q in range(P) if q != me and io[q + 1] > io[q]}
        recvs = {q: out[oo[q]:oo[q + 1]] for q in range(P) if q != me and oo[q + 1] > oo[q]}
        self.p2p(sends, recvs, "all_to_all_single(p2p)")

    def __repr__(self) -> str:
        return f"Comm(backend={self.backend}, rank={self.rank}, world={self.world}, active={self.active})"


class _GroupComm(Comm):
    """The default ``torch.distributed`` process group (nccl on CUDA, gloo on CPU). Collective FORMS are decided by the backend identically on
    every rank (gloo has no all-to-all and no ``all_gather_into_tensor``: the list / p2p forms), never by catching a collective's error."""

    def __init__(self):
        super().__init__(int(dist.get_rank()), int(dist.get_world_size()))
        self.backend = str(dist.get_backend())

    @property
    def device(self):
        if self.backend == "nccl":
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    @property
    def has_all_to_all(self) -> bool:
        return self.backend == "nccl"

    def _op(self, op: str):
        if op not in _REDUCE_OPS:
            raise RowpairRefused(f"comm: allreduce op {op!r} not in {_REDUCE_OPS}")
        return {"sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX, "min": dist.ReduceOp.MIN}[op]

    def bcast_(self, t, src: int):
        _check_contiguous((t,))
        t0 = time.perf_counter()
        if t.dtype == torch.bool:                                            # NCCL has no bool: the uint8 view travels
            dist.broadcast(t.view(torch.uint8), src=int(src))
        else:
            dist.broadcast(t, src=int(src))
        self._acct(_nbytes_of((t,)), t0)
        return t

    def _ops(self, sends: dict, recvs: dict) -> list:
        _check_contiguous(sends.values(), recvs.values())
        return ([dist.P2POp(dist.isend, t, int(peer)) for peer, t in sends.items()]
                + [dist.P2POp(dist.irecv, t, int(peer)) for peer, t in recvs.items()])

    def p2p(self, sends: dict, recvs: dict, what: str = "") -> None:
        t0 = time.perf_counter()
        self.check_p2p(sends, recvs, what)
        ops = self._ops(sends, recvs)
        if ops:                                                              # a rank with nothing to move posts no group (its peers' group does not name it)
            for w in dist.batch_isend_irecv(ops):
                w.wait()
            _p2p_sync(self)
        self._acct(_nbytes_of(sends.values()), t0)

    def p2p_start(self, sends: dict, recvs: dict, what: str = "") -> _Handle:
        self.check_p2p(sends, recvs, what)
        ops = self._ops(sends, recvs)
        if not ops:
            return _Handle(None, self, 0)
        return _Handle(dist.batch_isend_irecv(ops), self, _nbytes_of(sends.values()))

    def allreduce_(self, t, op: str = "sum"):
        t0 = time.perf_counter()
        dist.all_reduce(t, op=self._op(op))
        self._acct(_nbytes_of((t,)), t0)
        return t

    def barrier(self) -> None:
        dist.barrier()

    def allgather_obj(self, obj) -> list:
        allv = [None] * self.world
        dist.all_gather_object(allv, obj)
        return allv

    def broadcast_obj(self, obj, src: int = 0):
        box = [obj if self.rank == int(src) else None]
        dist.broadcast_object_list(box, src=int(src))
        return box[0]

    def all_gather_into(self, out, buf) -> None:
        _check_contiguous((out, buf))
        t0 = time.perf_counter()
        if self.backend == "nccl":
            dist.all_gather_into_tensor(out, buf)
        else:                                                                # gloo: the list form (decided by backend, not by an error)
            u = int(buf.shape[0])
            dist.all_gather([out[q * u:(q + 1) * u] for q in range(self.world)], buf)
        self._acct(_nbytes_of((buf,)), t0)

    def gather(self, buf, slots: Optional[list], dst: int = 0) -> None:
        _check_contiguous((buf,))
        t0 = time.perf_counter()
        dist.gather(buf, gather_list=slots if self.rank == int(dst) else None, dst=int(dst))
        self._acct(_nbytes_of((buf,)), t0)

    def _store(self):
        st = dist.distributed_c10d._get_default_store()
        if st is None:
            raise RowpairRefused("host_signal/host_wait: the process group has no c10d store")
        return st

    def host_signal(self, key: str, value: str = "1") -> None:
        self._store().set(HOST_KEY_PREFIX + str(key), str(value))

    def host_wait(self, key: str, timeout_s: float = HOST_WAIT_S) -> str:
        import datetime
        st = self._store()
        k = HOST_KEY_PREFIX + str(key)
        t0 = time.monotonic()
        try:
            st.wait([k], datetime.timedelta(seconds=float(timeout_s)))
        except Exception as exc:  # noqa: BLE001 — the store's timeout / connection error types vary by torch version
            raise RowpairRefused(host_wait_refusal(key, float(timeout_s), time.monotonic() - t0, exc)) from None
        v = st.get(k)
        return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)

    def all_to_all_single(self, out, inp, out_sizes=None, in_sizes=None) -> None:
        _check_contiguous((out, inp))
        t0 = time.perf_counter()
        if self.backend == "nccl":
            dist.all_to_all_single(out, inp, output_split_sizes=None if out_sizes is None else list(out_sizes),
                                   input_split_sizes=None if in_sizes is None else list(in_sizes))
            self._acct(_nbytes_of((inp,)), t0)
        else:
            self._all_to_all_p2p(out, inp, out_sizes, in_sizes)


# ----------------------------------------------------------------------------------------------------------------- threaded backend (CPU tests)
class ThreadHub(object):
    """Shared by the P rank-threads of one process (:func:`opt_core.testing.run_ranks`): collectives rendezvous through a barrier + a slot
    table. Every rank must issue the same sequence of collectives (the discipline NCCL requires, made an error here: a rank that skips a
    collective deadlocks the barrier, which the runner aborts and reports)."""

    def __init__(self, P: int):
        self.P = int(P)
        self.bar = threading.Barrier(self.P)
        self.slots: Dict[tuple, object] = {}
        self.lock = threading.Lock()
        self.seq = [0] * self.P
        self.signals: Dict[str, str] = {}
        self.cond = threading.Condition()

    def _next(self, rank: int) -> int:
        self.seq[rank] += 1
        return self.seq[rank]

    def barrier(self) -> None:
        self.bar.wait()

    def bcast_(self, t, src: int, rank: int) -> None:
        k = self._next(rank)
        if rank == src:
            with self.lock:
                self.slots[("b", k)] = t
        self.bar.wait()
        if rank != src:
            s = self.slots[("b", k)]
            if tuple(s.shape) != tuple(t.shape) or s.dtype != t.dtype:
                self.bar.abort()
                raise RowpairRefused(f"threaded bcast step {k}: rank {src} broadcasts {tuple(s.shape)} {s.dtype}, rank {rank} holds {tuple(t.shape)} {t.dtype}")
            t.copy_(s)
        self.bar.wait()
        if rank == src:
            with self.lock:
                del self.slots[("b", k)]

    def p2p(self, sends: dict, recvs: dict, rank: int) -> None:
        k = self._next(rank)
        with self.lock:
            for peer, t in sends.items():
                self.slots[("p", k, rank, int(peer))] = t
        self.bar.wait()
        for peer, t in recvs.items():
            key = ("p", k, int(peer), rank)
            if key not in self.slots:
                self.bar.abort()
                raise RowpairRefused(f"threaded p2p step {k}: rank {rank} expects a block from rank {peer} that was never sent")
            src = self.slots[key]
            if tuple(src.shape) != tuple(t.shape) or src.dtype != t.dtype:
                self.bar.abort()
                raise RowpairRefused(f"threaded p2p step {k}: rank {peer}->{rank} sends {tuple(src.shape)} {src.dtype}, receiver expects {tuple(t.shape)} {t.dtype}")
            t.copy_(src)
        self.bar.wait()
        with self.lock:
            for peer in sends:
                del self.slots[("p", k, rank, int(peer))]

    def allreduce_(self, t, op: str, rank: int) -> None:
        k = self._next(rank)
        with self.lock:
            self.slots[("a", k, rank)] = t.clone()
        self.bar.wait()
        vals = [self.slots[("a", k, r)] for r in range(self.P)]           # fixed rank order -> identical result on every rank
        acc = vals[0].clone().to(t.device)
        for v in vals[1:]:
            v = v.to(acc.device)
            acc = acc + v if op == "sum" else (torch.maximum(acc, v) if op == "max" else torch.minimum(acc, v))
        t.copy_(acc)
        self.bar.wait()
        with self.lock:
            del self.slots[("a", k, rank)]

    def host_signal(self, key: str, value: str) -> None:
        with self.cond:
            self.signals[key] = value
            self.cond.notify_all()

    def host_wait(self, key: str, timeout_s: float) -> str:
        deadline = time.monotonic() + float(timeout_s)
        with self.cond:
            while key not in self.signals:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise RowpairRefused(f"host_wait({key!r}): no rank signalled it within {float(timeout_s):.1f}s")
                self.cond.wait(min(left, 0.05))
            return self.signals[key]

    def allgather_obj(self, obj, rank: int) -> list:
        k = self._next(rank)
        with self.lock:
            self.slots[("o", k, rank)] = obj
        self.bar.wait()
        out = [self.slots[("o", k, r)] for r in range(self.P)]
        self.bar.wait()
        with self.lock:
            del self.slots[("o", k, rank)]
        return out


class _ThreadComm(Comm):
    """One rank-thread's comm on a :class:`ThreadHub` (CPU tests). ``p2p_start`` completes synchronously (a no-op handle)."""

    backend = "threaded"

    def __init__(self, rank: int, world: int, hub: ThreadHub):
        super().__init__(rank, world)
        self._hub = hub

    def bcast_(self, t, src: int):
        _check_contiguous((t,))
        t0 = time.perf_counter()
        self._hub.bcast_(t, int(src), self.rank)
        self._acct(_nbytes_of((t,)), t0)
        return t

    def p2p(self, sends: dict, recvs: dict, what: str = "") -> None:
        _check_contiguous(sends.values(), recvs.values())
        t0 = time.perf_counter()
        self.check_p2p(sends, recvs, what)
        self._hub.p2p(sends, recvs, self.rank)                               # every rank participates (rendezvous), even with nothing to move
        self._acct(_nbytes_of(sends.values()), t0)

    def p2p_start(self, sends: dict, recvs: dict, what: str = "") -> _Handle:
        self.p2p(sends, recvs, what)
        return _Handle(None, self, 0)

    def allreduce_(self, t, op: str = "sum"):
        if op not in _REDUCE_OPS:
            raise RowpairRefused(f"comm: allreduce op {op!r} not in {_REDUCE_OPS}")
        t0 = time.perf_counter()
        self._hub.allreduce_(t, op, self.rank)
        self._acct(_nbytes_of((t,)), t0)
        return t

    def barrier(self) -> None:
        self._hub.barrier()

    def allgather_obj(self, obj) -> list:
        return self._hub.allgather_obj(obj, self.rank)

    def broadcast_obj(self, obj, src: int = 0):
        return self._hub.allgather_obj(obj, self.rank)[int(src)]

    def all_gather_into(self, out, buf) -> None:
        _check_contiguous((out, buf))
        self._all_gather_into_p2p(out, buf)

    def gather(self, buf, slots, dst: int = 0) -> None:
        _check_contiguous((buf,))
        self._gather_p2p(buf, slots, int(dst))

    def host_signal(self, key: str, value: str = "1") -> None:
        self._hub.host_signal(str(key), str(value))

    def host_wait(self, key: str, timeout_s: float = HOST_WAIT_S) -> str:
        return self._hub.host_wait(str(key), float(timeout_s))

    def all_to_all_single(self, out, inp, out_sizes=None, in_sizes=None) -> None:
        _check_contiguous((out, inp))
        self._all_to_all_p2p(out, inp, out_sizes, in_sizes)


# ----------------------------------------------------------------------------------------------------------------- comm resolution
_TLS = threading.local()          # threaded backend: the calling thread's comm
_SOLO = None
_GROUP = None
_LOCK = threading.Lock()


def register_thread_comm(rank: int, world: int, hub: ThreadHub) -> Comm:
    """Bind a threaded-backend comm to the CALLING thread (:func:`opt_core.testing.run_ranks` does this per rank-thread)."""
    _TLS.comm = _ThreadComm(int(rank), int(world), hub)
    return _TLS.comm


def unregister_thread_comm() -> None:
    if hasattr(_TLS, "comm"):
        _TLS.comm.staging.release("unregister")
        del _TLS.comm


def _group_live() -> bool:
    try:
        return bool(dist.is_available() and dist.is_initialized())
    except RowpairRefused:                                                   # torch absent: no group
        return False


def comm() -> Comm:
    """The comm of the calling rank: the thread's registered threaded comm, else the default process group's (world > 1), else ``solo``."""
    global _SOLO, _GROUP
    c = getattr(_TLS, "comm", None)
    if c is not None:
        return c
    if _group_live() and int(dist.get_world_size()) > 1:
        g = _GROUP
        if g is None or g.world != int(dist.get_world_size()) or g.rank != int(dist.get_rank()):
            with _LOCK:
                g = _GROUP = _GroupComm()
        return g
    if _SOLO is None:
        with _LOCK:
            if _SOLO is None:
                _SOLO = Comm(0, 1)
    return _SOLO


get = comm                                                                   # the short spelling the row-sharded statements use: ``tp = dist.get()``


# ----------------------------------------------------------------------------------------------------------------- process group
def init_from_env(backend: Optional[str] = None, timeout_s: float = 1800.0, device_index: Optional[int] = None,
                  force_cpu: bool = False):
    """Initialise torch.distributed from the rank environment (``ROWPAIR_{RANK,WORLD,LOCAL_RANK,ADDR,PORT}`` set by :mod:`.launch`, rendezvous
    ``file://$ROWPAIR_STORE`` (default under the launcher) else ``tcp://addr:port``). ``backend=None``: NCCL when CUDA is used, else gloo
    (gloo = CPU tensors: the P2P forms this module uses under gloo do not take CUDA tensors). Returns ``(P, rank, device)``. NCCL on CUDA machines (device = ``cuda:<device_index or LOCAL_RANK>``, made current
    BEFORE the group exists so NCCL binds it), gloo otherwise (CPU tests; ``force_cpu``). With ``WORLD_SIZE`` unset or 1 no group is
    created (P == 1 semantics). The collective timeout is ``timeout_s``: a peer that died mid-collective is an error, not a hang."""
    global _GROUP
    from .launch import ENV_NCCL_TIMEOUT, init_method, local_rank, rank as env_rank, world_size
    rank_ = env_rank()
    P = world_size()
    if os.environ.get(ENV_NCCL_TIMEOUT, "").strip():
        timeout_s = float(os.environ[ENV_NCCL_TIMEOUT])
    local = local_rank() if device_index is None else int(device_index)
    use_cuda = (not force_cpu) and backend != "gloo" and bool(torch.cuda.is_available())   # an explicit gloo group = CPU tensors, no device binding
    if backend is None:
        backend = "nccl" if use_cuda else "gloo"
    if use_cuda:
        n = int(torch.cuda.device_count())
        if local >= n:
            raise RowpairRefused(f"rank {rank_}: device index {local} but {n} CUDA device(s) visible")
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if P > 1 and not dist.is_initialized():
        import datetime
        kw = dict(backend=backend, timeout=datetime.timedelta(seconds=float(timeout_s)), rank=rank_, world_size=P,
                  init_method=init_method())
        if backend == "nccl":
            try:
                dist.init_process_group(device_id=device, **kw)
            except TypeError:  # older torch without device_id
                dist.init_process_group(**kw)
        else:
            dist.init_process_group(**kw)
        with _LOCK:
            _GROUP = None
        arm_exit_guard()                                                      # a FAILED rank's wait before finalisation (exit hooks, non-daemon threads) is bounded by the grace
        _register_leave_at_exit()                                             # a FAILED rank gets the bounded leave on every path out of the interpreter; a healthy one is untouched
    return P, rank_, device


ABORT_TIMEOUT_S = 60.0                    # ``ROWPAIR_ABORT_TIMEOUT_S``: the bound of :func:`destroy` on a FAILURE path — the library's graceful teardown gets this long, then the process ends
EXIT_GRACE_S = 90.0                       # ``ROWPAIR_EXIT_GRACE_S``: how long a FAILED rank process may outlive its main thread before the exit guard ends it
EXIT_TEARDOWN_STUCK = 70                  # the status a failed rank ends with when its teardown stuck and the kit registered no verdict (:func:`exit_code`)
ENV_ABORT_TIMEOUT = "ROWPAIR_ABORT_TIMEOUT_S"
ENV_EXIT_GRACE = "ROWPAIR_EXIT_GRACE_S"
_EXIT = {"code": None, "failure": False, "armed": False, "leave_registered": False}


def exit_code(code: Optional[int]) -> None:
    """Register the status this rank process ends with when the core has to force its exit on a FAILURE path (the exit guard; the hard exit
    after a leave that overran) — the adapter registers it once its verdict is known; unregistered: :data:`EXIT_TEARDOWN_STUCK`. Clamped to a
    process status (0..255; a non-zero code never becomes 0). A non-zero code also marks the process failed (:func:`failure`)."""
    if code is None:
        _EXIT["code"] = None
        return
    c = int(code) & 0xFF
    _EXIT["code"] = c if (c != 0 or int(code) == 0) else 1


def failure(flag: bool = True) -> None:
    """Mark this rank process FAILED (or, with ``False``, not): the leave is then the bounded one and the exit guard is live (:func:`destroy`)."""
    _EXIT["failure"] = bool(flag)


def _failure_known(in_caller: bool = True) -> bool:
    """Whether this process is on a FAILURE path: a non-zero registered :func:`exit_code`, :func:`failure`, an exception in flight in the calling
    thread (a leave inside ``finally:`` / ``except:`` while it propagates; ``in_caller`` only), or an uncaught exception the interpreter has
    already reported (``sys.last_exc`` / ``sys.last_value`` — a leave in an exit hook after the command raised). A registered ``exit_code(0)``
    vetoes the last two (a kit that registered success is healthy). A healthy process is none of these, whatever its peers are doing. The
    in-flight test reads the calling thread's ACTIVE exception, which includes an outer ``except`` handler the leave is called under and a
    generator's finaliser: a healthy leave belongs outside any active handler — or passes ``failure=False`` / registers ``exit_code(0)``."""
    if _EXIT["failure"] or _EXIT["code"] not in (None, 0):
        return True
    if _EXIT["code"] == 0:                                                    # a registered success is healthy, whatever handler is active or was reported
        return False
    if in_caller:
        e = sys.exc_info()[1]
        if e is not None and not (isinstance(e, SystemExit) and e.code in (None, 0)):
            return True
    return getattr(sys, "last_exc", None) is not None or getattr(sys, "last_value", None) is not None


def _exit_status() -> int:
    return EXIT_TEARDOWN_STUCK if _EXIT["code"] is None else int(_EXIT["code"])


def rankleave_line(tag: str, rank: int, destroy_word: str, verdict: str, code) -> str:
    """``[<tag>] RANKLEAVE rank=<r> destroy=timeout:<s>s verdict=abandoned exit=<code>`` — the one line :func:`destroy` prints, on a FAILURE path,
    when the collective library's graceful teardown overran its bound and the process is ended (silent when it returned in time; never on a
    healthy path); the exit hook's form for a teardown that RAISED is ``destroy=raised:<Exc> verdict=none exit=-``."""
    return f"[{tag}] RANKLEAVE rank={int(rank)} destroy={destroy_word} verdict={verdict} exit={code}"


def rankstuck_line(tag: str, rank: int, alive_s: float, code: int, reason: str) -> str:
    """``[<tag>] RANKSTUCK rank=<r> reason=teardown_stuck alive_s=<g> exit=<code>`` — the line before a FAILED rank process is ended by ``os._exit``
    because its interpreter teardown did not finish (thread stacks follow it)."""
    return f"[{tag}] RANKSTUCK rank={int(rank)} reason={reason} alive_s={float(alive_s):g} exit={int(code)}"


def _env_seconds(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    try:
        return float(v) if v else float(default)
    except ValueError:
        return float(default)


def _who():
    """``(tag, rank)`` for the exit lines — the launch words when importable this late, else the environment."""
    try:
        from .launch import _tag, rank as env_rank
        return _tag(), int(env_rank())
    except BaseException:  # noqa: BLE001 — interpreter shutdown may have torn the module down; the words still come out
        try:
            return os.environ.get("ROWPAIR_TAG", "rowpair"), int(os.environ.get("RANK", os.environ.get("ROWPAIR_RANK", "0")) or 0)
        except BaseException:  # noqa: BLE001
            return "rowpair", 0


def _run_bounded(fn, bound_s: float, name: str):
    """Run ``fn()`` in a daemon thread; wait ``bound_s``. Returns ``(finished, exc)`` — a call still running is left behind (daemon). When no
    thread can be started (the interpreter is finalising) the call is made inline instead."""
    box = {}

    def call():
        try:
            fn()
        except BaseException as e:  # noqa: BLE001 — handed to the caller
            box["exc"] = e
    t = threading.Thread(target=call, name=name, daemon=True)
    try:
        t.start()
    except RuntimeError:                                                      # no thread can start this late in the interpreter's exit: the call is
        call()                                                                # made inline (unbounded) rather than skipped
        return True, box.get("exc")
    t.join(max(0.0, float(bound_s)))
    return (not t.is_alive()), box.get("exc")


def _hard_exit(line: str) -> None:
    """End a FAILED process NOW: the traceback of the exception in flight (it would otherwise be lost), ``line``, the kit's EXIT tallies not yet
    printed (``report.forced_exit``), stdio flushed, ``os._exit`` with :func:`_exit_status`. Nothing after the call runs — in particular not the
    collective library's or the interpreter's own teardown, which must not touch a communicator whose teardown was left running."""
    from ...report import forced_exit
    code = _exit_status()
    try:
        e = sys.exc_info()[1]
        if e is not None and not isinstance(e, SystemExit):
            import traceback
            traceback.print_exception(type(e), e, e.__traceback__)
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    except BaseException:  # noqa: BLE001
        pass
    forced_exit(code)


def _leave_at_exit() -> None:
    """The exit hook :func:`init_from_env` registers when it joins a group. On a FAILURE path (:func:`_failure_known`) it is the bounded
    :func:`destroy` before finalisation reaches the library's destructors (which block on a wedged communicator the way its teardown does);
    on a healthy process it does nothing — the kit's own leave, or none, exactly as without it."""
    if not _failure_known(in_caller=False):
        return
    try:
        destroy(failure=True)
    except Exception as e:  # noqa: BLE001 — an exit hook has no caller to hand it to: one named line instead of the interpreter's traceback
        tag, r = _who()
        sys.stderr.write(rankleave_line(tag, r, f"raised:{type(e).__name__}", "none", "-") + "\n")
        sys.stderr.flush()


def _register_leave_at_exit() -> bool:
    if _EXIT["leave_registered"]:
        return False
    import atexit
    atexit.register(_leave_at_exit)
    _EXIT["leave_registered"] = True
    return True


def arm_exit_guard(grace_s: Optional[float] = None) -> bool:
    """Arm (once per process) the EXIT GUARD: a daemon thread that, once the MAIN thread has ended, gives a FAILED process (:func:`_failure_known`:
    a non-zero registered status, :func:`failure`, an uncaught exception) ``ROWPAIR_EXIT_GRACE_S`` (else ``grace_s``, else :data:`EXIT_GRACE_S`,
    never less than the leave's bound ``ROWPAIR_ABORT_TIMEOUT_S`` + 15 s unless set explicitly) to exit, then ends it loudly: one
    ``RANKSTUCK reason=teardown_stuck`` line, every thread's stack (``faulthandler``), the pending EXIT tallies, ``os._exit`` with the status of
    :func:`exit_code` (unregistered: :data:`EXIT_TEARDOWN_STUCK`). It covers the waits of a failed rank BEFORE the interpreter finalises — a
    non-daemon thread that never ends, an exit hook that blocks. A HEALTHY process is never touched, however long its exit takes (a rank
    writing outputs for minutes after its command returned, peers still in their epilogue). :func:`init_from_env` arms it when it joins a
    group. Returns whether this call armed it."""
    if _EXIT["armed"]:
        return False
    _EXIT["armed"] = True
    import faulthandler
    from ...report import forced_exit                                         # imported now: the guard acts late in the interpreter's life
    grace = _env_seconds(ENV_EXIT_GRACE, EXIT_GRACE_S if grace_s is None else grace_s)
    if grace_s is None and not os.environ.get(ENV_EXIT_GRACE, "").strip():
        grace = max(grace, _env_seconds(ENV_ABORT_TIMEOUT, ABORT_TIMEOUT_S) + 15.0)
    main = threading.main_thread()

    def watch():
        main.join()
        t0 = time.monotonic()
        while time.monotonic() - t0 < grace:
            time.sleep(0.5)
        if not _failure_known(in_caller=False):                               # a healthy process: stand down, whatever is still running
            return
        code = _exit_status()
        try:
            tag, r = _who()
            sys.stderr.write(rankstuck_line(tag, r, time.monotonic() - t0, code, "teardown_stuck") + "\n")
            sys.stderr.flush()
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        except BaseException:  # noqa: BLE001
            pass
        finally:
            forced_exit(code)                                                 # the pending EXIT tallies, the flush, os._exit

    try:
        threading.Thread(target=watch, name="rowpair-exit-guard", daemon=True).start()
    except RuntimeError:
        _EXIT["armed"] = False
        return False
    return True


def destroy(timeout_s: Optional[float] = None, failure: Optional[bool] = None) -> str:
    """Leave the default group if one is initialised and return how: ``none`` (no group) or ``destroyed``. On a HEALTHY process this is the
    collective library's own ``destroy_process_group`` in the calling thread, however long it takes — it can wait for the peers' leave (a rank
    that finished early waits here while rank 0 writes its outputs), and that wait is correct. On a FAILURE path — ``failure=True``, else
    :func:`_failure_known`: a non-zero registered :func:`exit_code`, :func:`failure`, an exception propagating through the caller, an uncaught
    one already reported — the same call runs in a daemon thread bounded by ``ROWPAIR_ABORT_TIMEOUT_S`` (else ``timeout_s``, else
    :data:`ABORT_TIMEOUT_S`, 60 s): a failed rank's communicator can be WEDGED (it raised with a collective in flight its peers will never
    match) and the library then waits for that collective — tens of minutes, holding every device of the run, because the launcher only learns
    a rank's status when the rank exits. In time: ``destroyed``, silent. On overrun the process ENDS there: the propagating exception's
    traceback, one :func:`rankleave_line`, the kit's pending EXIT tallies, ``os._exit`` with :func:`_exit_status` — it never re-enters the
    library's or the interpreter's teardown (the communicator's own teardown is still running in the thread left behind; nothing else may
    touch it). An exception raised by ``destroy_process_group`` propagates as before. :func:`init_from_env` registers :func:`_leave_at_exit`
    when it joins a group so a failed rank gets this bounded leave on every path out of the interpreter; a kit's own call makes it a no-op."""
    global _GROUP
    if _GROUP is not None:
        _GROUP.staging.release("destroy")                                     # the wire buffers die with the group
    verdict = "none"
    if _group_live():
        failed = _failure_known() if failure is None else bool(failure)
        if not failed:
            dist.destroy_process_group()                                      # the healthy leave: the library's teardown in this thread, unbounded
        else:
            bound = _env_seconds(ENV_ABORT_TIMEOUT, ABORT_TIMEOUT_S if timeout_s is None else timeout_s)
            done, exc = _run_bounded(dist.destroy_process_group, bound, "rowpair-destroy")
            if not done:
                tag, r = _who()
                _hard_exit(rankleave_line(tag, r, f"timeout:{bound:g}s", "abandoned", _exit_status()))
                with _LOCK:                                                   # (reached only where a test stands in for the hard exit)
                    _GROUP = None
                return "abandoned"
            if exc is not None:
                with _LOCK:
                    _GROUP = None
                raise exc
        verdict = "destroyed"
    with _LOCK:
        _GROUP = None
    return verdict


def is_dist() -> bool:
    """True iff the calling rank belongs to a group of world size > 1 (the default process group, or a threaded test hub)."""
    return comm().world > 1


def barrier() -> None:
    comm().barrier()


def world() -> Tuple[int, int]:
    """``(P, rank)`` of the calling rank's comm, or from the rank environment without a group."""
    c = comm()
    if c.world > 1:
        return c.world, c.rank
    from .launch import rank as env_rank, world_size
    return world_size(), env_rank()


def allreduce_(t, op: str = "sum"):
    """In place ``t = op over ranks (t)`` (``sum`` | ``max`` | ``min``); returns ``t``. The equality at P == 1."""
    return comm().allreduce_(t, op)


def host_signal(key: str, value: str = "1") -> None:
    """:meth:`Comm.host_signal` of the calling rank's comm."""
    comm().host_signal(key, value)


def host_wait(key: str, timeout_s: float = HOST_WAIT_S) -> str:
    """:meth:`Comm.host_wait` of the calling rank's comm (CPU-side wait; the value signalled)."""
    return comm().host_wait(key, timeout_s)


# ----------------------------------------------------------------------------------------------------------------- Layout
def row_parts(N: int, P: int, align: int = 1) -> List[Tuple[int, int]]:
    """The BALANCED chunk-aligned contiguous row partition of ``N`` rows over ``P`` ranks: every boundary a multiple of ``align``; the
    ``n_chunks = ceil(N/align)`` chunks are dealt floor/ceil(n_chunks/P) per rank (earlier ranks get the extra one); the last chunk is the
    ragged remainder. A rank is empty only when ``n_chunks < P`` (:meth:`Layout.checked` refuses that by name)."""
    N, P, align = int(N), int(P), max(1, int(align))
    n_chunks = -(-N // align)
    base, extra = divmod(n_chunks, P)
    parts, s = [], 0
    for r in range(P):
        k = base + (1 if r < extra else 0)
        e = min(N, s + k * align)
        parts.append((s, e))
        s = e
    return parts


class Layout(object):
    """The row partition of ``N`` pair rows over ``P`` ranks, seen from ``rank``.

    ``align=None`` (policy ``grid``): the P-invariant grid. ``nb = ceil(N/B)`` blocks of ``B`` rows; ``bpr = ceil(nb/P)`` blocks per rank; rank q
    owns global rows ``[q0, q1) = [min(N, q*bpr*B), min(N, (q+1)*bpr*B))``; ``replicated`` is True iff ``N < P*B`` (the grid cannot shard —
    bounds become ``(0, N)`` on every rank and every primitive is the dense statement); trailing ranks may own ZERO rows when N is small
    relative to ``P*B``.
    ``align=k`` (policy ``aligned``): :func:`row_parts` — balanced, every boundary a multiple of ``k``, last rank ragged; ``B == k``;
    never ``replicated``; a rank owns zero rows only when ``ceil(N/k) < P``.

    Attributes: ``N P rank B align policy nb bpr r0 r1 R (= r1-r0) Rmax (= max_q R_q) bounds`` (list of ``(q0, q1)``), ``replicated``.
    Short aliases the row-sharded statements use: ``parts`` (= bounds), ``n_loc`` (= R), ``n_max`` (= Rmax), ``rows_of(q)``
    (= bounds[q]). The data-movement primitives handle ``R == 0``; a KIT refuses such a plan by name with :meth:`checked` (``B`` / ``align``
    is then chosen smaller, or P smaller, by the user — never silently)."""

    def __init__(self, N: int, P: int, rank: int, B: int = ZBLOCK, align: Optional[int] = None):
        if not (int(N) >= 1 and int(P) >= 1 and 0 <= int(rank) < int(P) and int(B) >= 1 and (align is None or int(align) >= 1)):
            raise ValueError(f"Layout(N={N}, P={P}, rank={rank}, B={B}, align={align}): need N>=1, P>=1, 0<=rank<P, B>=1, align>=1")
        self.N, self.P, self.rank = int(N), int(P), int(rank)
        self.align = None if align is None else int(align)
        if self.align is None:
            self.policy = "grid"
            self.B = int(B)
            self.nb = -(-self.N // self.B)
            self.bpr = -(-self.nb // self.P)
            self.Rmax = self.bpr * self.B
            self.bounds: List[Tuple[int, int]] = [
                (min(self.N, q * self.Rmax), min(self.N, (q + 1) * self.Rmax)) for q in range(self.P)
            ]
            self.replicated = self.N < self.P * self.B
            if self.replicated:
                self.bounds = [(0, self.N) for _ in range(self.P)]
                self.Rmax = self.N
        else:
            self.policy = "aligned"
            self.B = self.align
            self.nb = -(-self.N // self.B)
            self.bpr = -(-self.nb // self.P)
            self.bounds = row_parts(self.N, self.P, self.align)
            self.Rmax = max(q1 - q0 for q0, q1 in self.bounds)
            self.replicated = False
        self.r0, self.r1 = self.bounds[self.rank]
        self.R = self.r1 - self.r0
        self._starts = [q0 for q0, _ in self.bounds]

    # ---- short aliases the row-sharded statements spell
    @property
    def parts(self) -> List[Tuple[int, int]]:
        return self.bounds

    @property
    def n_loc(self) -> int:
        return self.R

    @property
    def n_max(self) -> int:
        return self.Rmax

    def rows_of(self, q: int) -> Tuple[int, int]:
        return self.bounds[int(q)]

    @property
    def padded_is_global(self) -> bool:
        """True iff rank q's rows start at global row ``q*Rmax`` for every q (the grid): a shard padded to ``Rmax`` and concatenated in rank
        order then has global row i at position i (no unpad copy after an all-gather)."""
        return all(q0 == min(self.N, q * self.Rmax) for q, (q0, _) in enumerate(self.bounds)) and not self.replicated

    @classmethod
    def auto(cls, N: int, P: int, rank: int, candidates: Sequence[int] = B_CANDIDATES, chunk: Optional[int] = None,
             lever: str = "rowpair", align: Optional[int] = None) -> "Layout":
        """The layout of a SHARDED run with the block size CHOSEN: the first ``B`` of ``candidates`` (largest first; only multiples of
        ``chunk`` when the engine's chunk size must divide B) whose grid neither replicates nor leaves a rank with zero rows. Every bin of
        the memory ladder (1000 ... 16000 tokens) x P in {2, 4, 8} has a plan with the default candidates (128, 64, 32, 16); a shape with
        none is refused BY NAME listing the candidates tried. ``P == 1`` returns ``cls(N, 1, 0, candidates[0])`` (equality grid).
        ``align=k`` selects the aligned policy instead (nothing to choose: :meth:`checked` with that align)."""
        if align is not None:
            return cls.checked(N, P, rank, B=int(align), lever=lever, align=int(align))
        cands = [int(b) for b in candidates if chunk is None or int(b) % int(chunk) == 0]
        if not cands:
            raise RowpairRefused(f"Layout.auto: no block size in {tuple(candidates)} is a multiple of chunk={chunk}", lever)
        if int(P) == 1:
            return cls(N, 1, 0, cands[0])
        tried = []
        for B in cands:
            lay = cls(N, P, rank, B)
            if not lay.replicated and all(lay.nrows(q) > 0 for q in range(lay.P)):
                return lay
            tried.append(f"B={B}:{'replicated' if lay.replicated else 'empty_rank'}")
        raise RowpairRefused(f"N={int(N)} P={int(P)}: no valid row grid among {','.join(tried)} (choose a smaller n_gpu)", lever)

    @classmethod
    def checked(cls, N: int, P: int, rank: int, B: int = ZBLOCK, lever: str = "rowpair", align: Optional[int] = None) -> "Layout":
        """The layout of a SHARDED run: refuses by name (``RowpairRefused``) when ``P > 1`` and the partition replicates (grid: ``N < P*B``) or
        leaves some rank with zero rows. P == 1 always passes."""
        lay = cls(N, P, rank, B, align)
        if lay.P > 1:
            if lay.replicated:
                raise RowpairRefused(f"N={lay.N} < P*B={lay.P}*{lay.B}: the row grid cannot shard (choose a smaller B or n_gpu)", lever)
            empty = [q for q in range(lay.P) if lay.nrows(q) == 0]
            if empty:
                raise RowpairRefused(f"N={lay.N} P={lay.P} B={lay.B} align={lay.align}: ranks {empty} own zero rows (choose a smaller "
                                     f"{'align' if lay.align else 'B'} or n_gpu)", lever)
        return lay

    def rows(self, q: Optional[int] = None) -> slice:
        q0, q1 = self.bounds[self.rank if q is None else q]
        return slice(q0, q1)

    def nrows(self, q: Optional[int] = None) -> int:
        q0, q1 = self.bounds[self.rank if q is None else q]
        return q1 - q0

    def chunks(self, chunk: int, q: Optional[int] = None) -> Iterator[Tuple[int, int]]:
        """GLOBAL chunk grid ``range(0, N, chunk)`` clipped to rank q's rows: yields ``(c0, c1)`` global with ``q0 <= c0 < c1 <= q1``. If
        ``chunk`` divides ``B`` these are exactly the engine's chunks that fall inside the rank's rows."""
        q0, q1 = self.bounds[self.rank if q is None else q]
        c = (q0 // chunk) * chunk
        while c < q1:
            c0, c1 = max(c, q0), min(c + chunk, q1)
            if c1 > c0:
                yield (c0, c1)
            c += chunk

    def local_blocks(self, rows_per_block: int, q: Optional[int] = None) -> Iterator[Tuple[int, int]]:
        """LOCAL row sub-blocks ``[i0, i1)`` (local coordinates) of rank q's shard."""
        R = self.nrows(q)
        i = 0
        rows_per_block = max(1, int(rows_per_block))
        while i < R:
            yield (i, min(R, i + rows_per_block))
            i += rows_per_block

    def owner(self, i: int) -> int:
        """The rank owning global row ``i``."""
        if self.replicated:
            return self.rank
        if self.policy == "grid":
            return min(self.P - 1, int(i) // self.Rmax)
        q = bisect.bisect_right(self._starts, int(i)) - 1
        while q + 1 < self.P and self.bounds[q][1] <= int(i):                # skip empty ranks
            q += 1
        return max(0, q)

    def facts(self) -> dict:
        """The evidence facts of this rank's shard: ``N P rank B align policy r0 r1 R Rmax``."""
        return {"N": self.N, "P": self.P, "rank": self.rank, "B": self.B, "align": self.align, "policy": self.policy,
                "r0": self.r0, "r1": self.r1, "R": self.R, "Rmax": self.Rmax}

    def __repr__(self) -> str:
        return (f"Layout(N={self.N}, P={self.P}, rank={self.rank}, B={self.B}, align={self.align}, policy={self.policy}, r0={self.r0}, "
                f"r1={self.r1}, R={self.R}, Rmax={self.Rmax}, replicated={self.replicated})")


def require_sharded(layout: "Layout", what: str, lever: str = "rowpair") -> "Layout":
    """The structural n_gpu=1 rule of the L2 statements: a P == 1 / replicated / no-group layout is REFUSED BY NAME — at ``--n_gpu 1`` a
    kit's adapter installs nothing and the engine's single-GPU statement runs unchanged (no tiled schedule of this package may run at P=1,
    so P=1 numerics are the engine's by construction, not by care)."""
    if layout.P == 1 or layout.replicated or not is_dist():
        raise RowpairRefused(f"{what}: refused at n_gpu=1 (P={layout.P} replicated={layout.replicated} group={'yes' if is_dist() else 'no'}): "
                             "the adapter installs nothing at n_gpu=1; the engine's own statement runs", lever)
    return layout


def layout_here(N: int, B: int = ZBLOCK, checked: bool = True, lever: str = "rowpair", align: Optional[int] = None) -> Layout:
    """The layout of THIS rank for ``N`` rows from the calling rank's comm (``P = world size``, or 1 without a group)."""
    P, r = world() if is_dist() else (1, 0)
    return Layout.checked(N, P, r, B, lever, align) if checked else Layout(N, P, r, B, align)


def chunk_align() -> int:
    """The aligned policy's default ``align``: ``ROWPAIR_CHUNK_ALIGN`` (the engine's pinned chunk size, set by the adapter), 1 when unset."""
    return max(1, env_int("ROWPAIR_CHUNK_ALIGN", 1))


_CTX: Dict[tuple, Layout] = {}


def ctx(N: int, align: int = 1, lever: str = "rowpair") -> Layout:
    """The aligned-policy :class:`Layout` of pair size ``N`` for the calling rank (cached per ``(N, align, rank, P)``; refused by name when a
    rank would own zero rows) — the sharding context of the row-sharded statements."""
    P, r = world() if is_dist() else (1, 0)
    k = (int(N), int(align), r, P)
    lay = _CTX.get(k)
    if lay is None:
        lay = _CTX[k] = Layout.checked(int(N), P, r, B=int(align), lever=lever, align=int(align))
    return lay


def default_ctx(N: int) -> Layout:
    """:func:`ctx` on the programme grid (``align = chunk_align()``)."""
    return ctx(N, chunk_align())


def clear_layout_cache() -> None:
    _CTX.clear()


# ----------------------------------------------------------------------------------------------------------------- shard accessors
def zrows(z, i0: int, i1: int):
    """Rows ``i0:i1`` (LOCAL indices) of the shard, ``[i1-i0, N, C]``; a VIEW for tensors, the store's own read otherwise."""
    return z.zrows(i0, i1) if hasattr(z, "zrows") else z[i0:i1]


def zwrite(z, i0: int, i1: int, x) -> None:
    """Copy rows back (no-op when ``x`` is the view itself)."""
    if hasattr(z, "zwrite"):
        return z.zwrite(i0, i1, x)
    dst = z[i0:i1]
    if x.data_ptr() == dst.data_ptr() and x.shape == dst.shape and x.stride() == dst.stride():
        return None
    dst.copy_(x)
    return None


def zadd(z, i0: int, i1: int, delta) -> None:
    """In place ``z[i0:i1] += delta``."""
    if hasattr(z, "zadd"):
        return z.zadd(i0, i1, delta)
    z[i0:i1] += delta
    return None


def zmeta(z):
    """``(R, N, C, dtype, device)`` of a shard (tensor or store duck-type)."""
    if hasattr(z, "zmeta"):
        return z.zmeta()
    return z.shape[0], z.shape[1], z.shape[2], z.dtype, z.device


def zlen(z) -> int:
    """Number of LOCAL rows of a shard (tensor or store-like with ``.R`` / ``__len__``)."""
    return int(z.shape[0]) if hasattr(z, "shape") else int(getattr(z, "R", None) or len(z))


def zclone(z):
    """A private copy of a shard (``.clone()``)."""
    return z.clone()


def zalloc_like(z, dtype=None, rows: Optional[int] = None):
    """A new shard container shaped like ``z`` (``rows`` overrides dim 0)."""
    shape = list(z.shape)
    if rows is not None:
        shape[0] = int(rows)
    return torch.empty(shape, dtype=dtype or z.dtype, device=z.device)


def zblocks(R: int, step: int = ZBLOCK) -> Iterator[Tuple[int, int]]:
    """LOCAL row blocks ``(i0, i1)`` of size ``step`` (tail shorter)."""
    for i0 in range(0, int(R), int(step)):
        yield i0, min(int(R), i0 + int(step))


# ----------------------------------------------------------------------------------------------------------------- internal helpers
def _pad_rows(x, rows: int, dim: int = 0):
    """Contiguous tensor with ``rows`` entries along ``dim``, leading entries = x (bit copy), padding zeroed. No copy if x already has
    that shape and is contiguous."""
    n = x.shape[dim]
    if n == rows and x.is_contiguous():
        return x
    shape = list(x.shape)
    shape[dim] = rows
    buf = x.new_empty(shape)
    if n > 0:
        buf.narrow(dim, 0, n).copy_(x)
    if rows > n:
        buf.narrow(dim, n, rows - n).zero_()
    return buf


def require_schedule_dtype(t, dtype, device, what: str) -> None:
    """A COMMUNICATED operand (a slab that enters the ring / all-to-all / p2p) carries the pair schedule's dtype and device — the ones every
    receiving rank sizes its wire buffers with (``z_shard.dtype``): a producer returning another dtype (an autocast projection) on the ranks that
    own a sub-block, while a rank owning none posts the schedule-dtype empty slab, is transfers of different byte counts that never complete on the
    device. Refused by name (``what`` names the operand); ``dtype`` / ``device`` None skip that half of the check; local epilogue callables
    (``out`` / ``gate``) are not constrained (torch adds their result into z). The one statement of the rule: the trimul b slabs, the all-to-all
    windows and the ring-pass blocks call it."""
    bad_dtype = dtype is not None and t.dtype != dtype
    bad_device = device is not None and t.device != torch.device(device)
    if bad_dtype or bad_device:
        want_dev = torch.device(device) if device is not None else t.device
        raise RowpairRefused(f"{what} is {t.dtype} on {t.device} but the pair schedule is {dtype if dtype is not None else t.dtype} on {want_dev} — a "
                             f"communicated operand carries the schedule dtype and device every peer sizes its receive buffers with: proj must return "
                             f"z's dtype and device (autocast callers: cast the projection to z.dtype); out / gate may return any dtype torch adds into z")


def _p2p_exchange(sends: Sequence[Tuple[object, int]], recvs: Sequence[Tuple[object, int]], what: str = "p2p_exchange") -> None:
    """``sends`` = ``[(tensor, dst)]``, ``recvs`` = ``[(tensor, src)]`` (at most one tensor per peer and direction) through :meth:`Comm.p2p`."""
    s = {int(d): t for t, d in sends}
    r = {int(q): t for t, q in recvs}
    if len(s) != len(sends) or len(r) != len(recvs):
        raise RowpairRefused("comm: _p2p_exchange takes at most one tensor per peer and direction")
    comm().p2p(s, r, what)


def _all_to_all_single(out, inp) -> None:
    """P equal blocks along dim 0 exchanged all-to-all through the comm surface (nccl: ``all_to_all_single``; gloo / threaded: the batched
    isend/irecv exchange — decided by the BACKEND, identically on every rank, never by catching a collective's error)."""
    comm().all_to_all_single(out, inp)


# ----------------------------------------------------------------------------------------------------------------- gathers
def all_gather_rows(x_shard, layout: Layout):
    """``x_shard[R_rank, ...] -> full [N, ...]`` on every rank (rank order). Grid policy: pad to ``Rmax``, ``all_gather_into``, return the
    leading-N view (global row i lands at padded position i, so no unpad copy is needed). Aligned policy (padded positions are not global
    positions): ONE full-size allocation and P in-place broadcasts of the rank regions (contiguous along dim 0). Bit-preserving."""
    if layout.replicated or not is_dist() or layout.P == 1:
        return x_shard
    if int(x_shard.shape[0]) != layout.R:
        raise RowpairRefused(f"all_gather_rows: shard has {int(x_shard.shape[0])} rows, layout.R={layout.R}")
    rest = list(x_shard.shape[1:])
    c = comm()
    x_shard = x_shard.contiguous()
    if layout.padded_is_global:
        buf = _pad_rows(x_shard, layout.Rmax, 0)
        out = x_shard.new_empty([layout.P * layout.Rmax] + rest)
        c.all_gather_into(out, buf)
        return out[: layout.N]
    out = x_shard.new_empty([layout.N] + rest)
    for q, (q0, q1) in enumerate(layout.bounds):
        if q1 <= q0:
            continue
        reg = out[q0:q1]
        if q == layout.rank:
            reg.copy_(x_shard)
        c.bcast_(reg, src=q)
    return out


def gather_rows_to_rank0(x_shard, layout: Layout):
    """``x_shard[R_rank, ...] -> full [N, ...]`` on rank 0 (None elsewhere). ONE collective (``gather`` of the ``Rmax``-padded shard: every
    rank calls it, in the same order as every other collective of the run); only rank 0 allocates the full tensor."""
    if layout.replicated or not is_dist() or layout.P == 1:
        return x_shard if layout.rank == 0 else None
    if not (x_shard.shape[0] == layout.R):
        raise RowpairRefused(f"rowpair.dist: violated: x_shard.shape[0] == layout.R")
    return gather_cat_to_rank0(x_shard, [r1 - r0 for r0, r1 in layout.bounds])


def host_gather_cols(shape: Sequence[int], elem_bytes: int, block_bytes: int) -> int:
    """Columns (dim 1) per block of :func:`gather_rows_to_rank0_host` so that one assembled block ``[N, w, *rest]`` stays under
    ``block_bytes``: ``w = clamp(block_bytes // (N * prod(rest) * elem_bytes), 1, K)``; a rank-1 ``[N]`` tensor is one block (returns 0).
    Identical on every rank (shapes and the budget are)."""
    shape = [int(s) for s in shape]
    if len(shape) < 2:
        return 0
    N, K = shape[0], shape[1]
    inner = 1
    for s in shape[2:]:
        inner *= s
    per_col = max(1, N * inner * int(elem_bytes))
    return max(1, min(K, int(block_bytes) // per_col))


def _host_out(x, full_shape, out, pin: bool, what: str):
    if out is None:
        return torch.empty(full_shape, dtype=x.dtype, device="cpu", pin_memory=bool(pin) and torch.cuda.is_available())
    if [int(s) for s in out.shape] != list(full_shape) or out.dtype != x.dtype or out.device.type != "cpu":
        raise RowpairRefused(f"rowpair.dist: {what}: out must be a host {list(full_shape)} {x.dtype} tensor; got {tuple(out.shape)} {out.dtype} {out.device}")
    return out


def _host_copy_blocks(x, *, block_bytes: int, out=None, pin: bool = True, what: str = "host_copy"):
    """Rank-local: ``x[N, K, *rest]`` copied into a (pinned) host tensor in column blocks of :func:`host_gather_cols` columns."""
    from .evidence import record_schedule
    shape = [int(s) for s in x.shape]
    out = _host_out(x, shape, out, pin, what)
    w = host_gather_cols(shape, x.element_size(), int(block_bytes))
    grouped = is_dist()                                                       # the census names the SHARDED schedule; a no-group (n_gpu=1) copy adds no word
    if w == 0:
        if grouped:
            record_schedule(host_gather_cols=0, host_gather_blocks=1)
        out.copy_(x)
        return out
    K = shape[1]
    if grouped:
        record_schedule(host_gather_cols=int(w), host_gather_blocks=int((K + w - 1) // w))
    for c0 in range(0, K, w):
        c1 = min(K, c0 + w)
        out[:, c0:c1].copy_(x[:, c0:c1])
    return out


def gather_cat_to_rank0_host(piece, counts: Sequence[int], *, block_bytes: int = 64 << 20, out=None, pin: bool = True):
    """:func:`gather_cat_to_rank0` with a HOST destination and a bounded device transient: rank q's ``piece[counts[q], K, *rest]`` rows are
    concatenated (rank order) into a ``[sum(counts), K, *rest]`` HOST tensor on rank 0 — pinned when ``pin`` and CUDA is present, or the
    caller's host ``out`` — walking dim 1 in blocks of :func:`host_gather_cols` columns (one ``gather`` per block, every rank, the same count on
    every rank; rank 0's device transient = one assembled block ``[total, w, *rest]`` + the P padded pieces). Returns ``out`` on rank 0,
    ``None`` elsewhere. A rank-1 piece is one block. P == 1 / no group: the block-wise device->host copy of ``piece``. Bit-preserving.
    Census: ``host_gather_cols host_gather_blocks``."""
    from .evidence import record_schedule
    P_, rank_ = world() if is_dist() else (1, 0)
    piece = piece.contiguous()
    shape = [int(s) for s in piece.shape]
    if P_ == 1:
        return _host_copy_blocks(piece, block_bytes=block_bytes, out=out, pin=pin, what="gather_cat_to_rank0_host")
    counts = [int(c) for c in counts]
    if len(counts) != P_ or shape[0] != counts[rank_]:
        raise RowpairRefused(f"rowpair.dist: gather_cat_to_rank0_host: piece rows {shape[0]} vs counts {counts} (P={P_}, rank={rank_})")
    full_shape = [sum(counts)] + shape[1:]
    rank0 = rank_ == 0
    if rank0:
        out = _host_out(piece, full_shape, out, pin, "gather_cat_to_rank0_host")
    w = host_gather_cols(full_shape, piece.element_size(), int(block_bytes))   # from the FULL shape: identical on every rank
    if w == 0:                                                                # rank-1: one block
        full = gather_cat_to_rank0(piece, counts)
        record_schedule(host_gather_cols=0, host_gather_blocks=1)
        if rank0:
            out.copy_(full)
            return out
        return None
    K = shape[1]
    record_schedule(host_gather_cols=int(w), host_gather_blocks=int((K + w - 1) // w))
    for c0 in range(0, K, w):
        c1 = min(K, c0 + w)
        full = gather_cat_to_rank0(piece[:, c0:c1].contiguous(), counts)    # [total, w, *rest] on rank 0's device, None elsewhere
        if rank0:
            out[:, c0:c1].copy_(full)
        del full
    return out if rank0 else None


def gather_rows_to_rank0_host(x_shard, layout: Layout, *, block_bytes: int = 64 << 20, out=None, pin: bool = True):
    """A row-sharded tensor ``x_shard[R_rank, K, *rest]`` (this rank's rows of a full ``[N, K, *rest]``: PAE / PDE / contact matrices,
    distogram logits, any per-row-pair head output) assembled into a HOST tensor on rank 0 (pinned when ``pin`` and CUDA is present, or the
    caller's ``out[N, K, *rest]``) WITHOUT the full tensor ever existing on a device: :func:`gather_cat_to_rank0_host` over the layout's
    per-rank row counts (column blocks of :func:`host_gather_cols` columns; ``block_bytes`` bounds rank 0's device transient). Returns ``out``
    on rank 0, ``None`` elsewhere. P == 1 / no group / a replicated layout: rank 0's block-wise device->host copy of the (already whole)
    tensor, no collective. Bit-preserving (data movement only). Census: ``host_gather_cols host_gather_blocks``."""
    shape = [int(s) for s in x_shard.shape]
    if layout.replicated or not is_dist() or layout.P == 1:
        if shape[0] != layout.N:
            raise RowpairRefused(f"rowpair.dist: gather_rows_to_rank0_host: whole tensor expected ({layout.facts()}), got rows={shape[0]} != N={layout.N}")
        if layout.rank != 0:
            return None
        return _host_copy_blocks(x_shard.contiguous(), block_bytes=block_bytes, out=out, pin=pin, what="gather_rows_to_rank0_host")
    if shape[0] != layout.R:
        raise RowpairRefused(f"rowpair.dist: gather_rows_to_rank0_host: x_shard.shape[0]={shape[0]} != layout.R={layout.R}")
    return gather_cat_to_rank0_host(x_shard, [r1 - r0 for r0, r1 in layout.bounds], block_bytes=block_bytes, out=out, pin=pin)


def gather_cat_to_rank0(piece, counts: Sequence[int], device=None):
    """Variable-length row concat to rank 0: rank q contributes ``piece`` with ``counts[q]`` rows (counts known on every rank). Returns the
    concatenation over ranks (rank order) on rank 0, None elsewhere. ONE collective per call (``gather`` of the pieces padded to
    ``max(counts)`` rows) — every rank calls it, ranks with zero rows included, so it is ordered with the run's other collectives on the
    group's stream (no pairwise send/recv whose matching depends on issue order across peers). Bit-preserving."""
    if device is not None:
        piece = piece.to(device)
    P, rank_ = world() if is_dist() else (1, 0)
    piece = piece.contiguous()
    if not (piece.shape[0] == counts[rank_]):
        raise RowpairRefused("rowpair.dist: violated: " + repr(((piece.shape, counts, rank_))))
    if P == 1:
        return piece
    if len(counts) != P:
        raise RowpairRefused(f"rowpair.dist: gather_cat_to_rank0: len(counts)={len(counts)} != P={P}")
    maxc = max(int(c) for c in counts)
    rest = tuple(piece.shape[1:])
    if maxc == 0:
        return piece.new_empty((0,) + rest) if rank_ == 0 else None
    buf = _pad_rows(piece, maxc, 0).contiguous()                             # [maxc, ...] on every rank
    c = comm()
    if rank_ == 0:
        slots = [piece.new_empty((maxc,) + rest) for _ in range(P)]
        c.gather(buf, slots, dst=0)
        total = int(sum(int(n) for n in counts))
        full = piece.new_empty((total,) + rest)
        o = 0
        for q in range(P):
            n = int(counts[q])
            if n > 0:
                full[o: o + n].copy_(slots[q][:n])
            o += n
        return full
    c.gather(buf, None, dst=0)
    return None


# ----------------------------------------------------------------------------------------------------------------- all-to-all transposes
def alltoall_window(provider: Callable[[int, int], object], layout: Layout, off: int = 0, width: Optional[int] = None, *, C: int,
                    dtype, device, out_layout: str = "rows", chunks: Optional[int] = None, out=None):
    """Generalised distributed transpose of a row-sharded ``X[N, N, C]``.

    ``provider(c0, c1)`` must return THIS rank's block ``X[r0:r1, c0:c1, :]`` as a tensor ``[R, c1-c0, C]`` (a view, or computed just in
    time); it is called once per destination rank with that destination's column window. For destination d the window is global columns
    ``[d0+off, min(d1, d0+off+width))`` (``(d0, d1) = bounds[d]``); ``width=None`` means d's whole row range (``off`` must be 0).

    Every rank receives the data of ITS window rows ``i in [r0+off, min(r1, r0+off+width))`` (``W_v`` rows) over all ``j in [0, N)``::

        out_layout "rows"   : out[w, j, c] = X[j, r0+off+w, c]  -> [W_v, N, C]
        out_layout "gemm_a" : out[c, w, j] = X[j, r0+off+w, c]  -> [C, W_v, N]
        out_layout "gemm_b" : out[c, j, w] = X[j, r0+off+w, c]  -> [C, N, W_v]

    Bit-preserving. Staged in ``chunks`` channel groups (``ROWPAIR_A2A_CHUNKS``, default 4): transient = provider blocks + out +
    2/chunks * (W*N*C) send/recv."""
    P, rank_, N = layout.P, layout.rank, layout.N
    if width is None:
        if not (off == 0):
            raise RowpairRefused(f"rowpair.dist: violated: off == 0")
        width = layout.Rmax
    W = int(width)

    def win(q):
        q0, q1 = layout.bounds[q]
        return min(q1, q0 + off), min(q1, q0 + off + W)

    my_a, my_b = win(rank_)
    Wv = my_b - my_a
    if out is None:
        shape = {"rows": (Wv, N, C), "gemm_a": (C, Wv, N), "gemm_b": (C, N, Wv)}[out_layout]
        out = torch.empty(shape, dtype=dtype, device=device)

    def place(src_q: int, blk, c0: int, c1: int):
        # blk: [R_src (j range = bounds[src_q]), Wv, c1-c0]
        j0, j1 = layout.bounds[src_q]
        if j1 <= j0 or Wv == 0:
            return
        if out_layout == "rows":
            out[:, j0:j1, c0:c1].copy_(blk.transpose(0, 1))
        elif out_layout == "gemm_a":
            out[c0:c1, :, j0:j1].copy_(blk.permute(2, 1, 0))
        else:
            out[c0:c1, j0:j1, :].copy_(blk.permute(2, 0, 1))

    if layout.replicated or not is_dist() or P == 1:
        if Wv > 0:
            place(rank_, provider(my_a, my_b), 0, C)
        return out

    blocks = []
    for d in range(P):
        a, b = win(d)
        blocks.append(provider(a, b) if b > a else None)
    nch = int(chunks if chunks is not None else env_int("ROWPAIR_A2A_CHUNKS", 4))
    nch = max(1, min(nch, C))
    cs = -(-C // nch)
    Rmax = layout.Rmax
    st = comm().staging                                                       # one send / recv pair reused across the channel chunks, freed at exit
    site = st.enter("alltoall_window")
    for c0 in range(0, C, cs):
        c1 = min(C, c0 + cs)
        send = st.buffer(site, "send", (P, Rmax, W, c1 - c0), dtype, device).zero_()
        for d in range(P):
            blk = blocks[d]
            if blk is not None and blk.shape[0] > 0 and blk.shape[1] > 0:
                require_schedule_dtype(blk, dtype, send.device, "alltoall_window: the window producer's block")   # never a silent cast on the wire
                send[d, : blk.shape[0], : blk.shape[1], :].copy_(blk[..., c0:c1])
        recv = st.buffer(site, "recv", (P, Rmax, W, c1 - c0), dtype, device)
        _all_to_all_single(recv, send)
        del send
        for s in range(P):
            Rs = layout.nrows(s)
            if Rs > 0:
                place(s, recv[s, :Rs, :Wv, :], c0, c1)
        del recv
    st.exit(site)
    del blocks
    return out


def transpose_shards(z_shard, layout: Layout, chunks: Optional[int] = None, *, block_rows: Optional[int] = None, out=None):
    """``z_shard[R, N, C]`` (rows ``r0:r1`` of Z) -> ``zT_shard[R, N, C]`` with ``zT_shard[i-r0, j, :] = Z[j, i, :]`` (all_to_all over padded
    ``[P, Rmax, W, C/chunks]`` blocks + local block transpose; bit-preserving).

    ``z_shard`` / ``out`` may be plain tensors or store objects (accessed only via :func:`zrows` / :func:`zwrite` in row blocks). ``block_rows``
    (``ROWPAIR_TRANSPOSE_BLOCK_ROWS``, default 0 = whole shard in one exchange) streams the transpose in output row windows of that many rows
    (use multiples of 128) so that a full copy of the shard is never materialised at once: transient ~= 3 * block_rows * N * C elements
    (+ 2/chunks of one window). (The ``[*, n_loc, N, C]`` tensor form built on peer exchanges is :func:`opt_core.mem.rowpair.ring.transpose_shard`.)"""
    R, N, C, dtype, device = zmeta(z_shard)
    if layout.replicated or not is_dist() or layout.P == 1:
        zt = zrows(z_shard, 0, R).transpose(0, 1).contiguous()
        if out is None:
            return zt
        zwrite(out, 0, zt.shape[0], zt)
        return out
    if not (R == layout.R and N == layout.N):
        raise RowpairRefused("rowpair.dist: violated: " + repr((((R, N, C), repr(layout)))))
    if block_rows is None:
        block_rows = env_int("ROWPAIR_TRANSPOSE_BLOCK_ROWS", 0)
    rb = env_int("ROWPAIR_ZROWS_BLOCK", 128)  # read granularity for zrows
    from .evidence import record_schedule
    record_schedule(a2a_chunks=int(chunks) if chunks else env_int("ROWPAIR_A2A_CHUNKS", 4), transpose_block_rows=int(block_rows))

    def provider(c0, c1):
        # my rows, columns c0:c1, read through zrows in row blocks
        if c1 <= c0:
            return torch.empty((R, 0, C), dtype=dtype, device=device)
        if not hasattr(z_shard, "zrows") and torch.is_tensor(z_shard):
            return z_shard[:, c0:c1, :]
        blk = torch.empty((R, c1 - c0, C), dtype=dtype, device=device)
        for i0 in range(0, R, rb):
            i1 = min(R, i0 + rb)
            blk[i0:i1].copy_(zrows(z_shard, i0, i1)[:, c0:c1, :])
        return blk

    if not block_rows or block_rows >= layout.Rmax:
        res = alltoall_window(provider, layout, 0, None, C=C, dtype=dtype, device=device, out_layout="rows", chunks=chunks)
        if out is None:
            return res
        zwrite(out, 0, res.shape[0], res)
        return out
    if out is None:
        out = torch.empty((R, N, C), dtype=dtype, device=device)
    for w0 in range(0, layout.Rmax, block_rows):      # same schedule on every rank
        w1 = min(layout.Rmax, w0 + block_rows)
        res = alltoall_window(provider, layout, w0, w1 - w0, C=C, dtype=dtype, device=device, out_layout="rows", chunks=chunks)
        if res.shape[0] > 0:
            zwrite(out, w0, w0 + res.shape[0], res)
        del res
    return out


def transpose_blocks(z_shard, layout: Layout, step: int = ZBLOCK):
    """Block-streamed transpose. Yields ``(i0, i1, zT_blk)`` for this rank's LOCAL row blocks ``[i0, i1)`` (size ``step``, ascending; tail
    shorter) with ``zT_blk[i - i0, j, :] = z[j, r0 + i, :]`` for ALL global j. ``zT_blk`` is a transposed view of a ``[N, i1-i0, C]`` column
    block assembled by ONE ``all_to_all_single`` per round (variable split sizes, no padding); peak transient is ``O(N * step * C)`` — a full
    transposed copy of the shard never exists. Pure data movement (bit-preserving). Collective: every rank runs ``max_q ceil(R_q / step)``
    rounds. (A comm without a native all-to-all — gloo — all-gathers once and slices column blocks.)"""
    P, R, N, r0 = layout.P, layout.R, layout.N, layout.r0
    if layout.replicated or not is_dist() or P == 1:
        full = zrows(z_shard, 0, R)
        for (i0, i1) in zblocks(R, step):
            yield i0, i1, full[:, i0:i1].transpose(0, 1)
        return
    probe = zrows(z_shard, 0, min(1, R))
    C = tuple(probe.shape[2:])
    Cn = 1
    for c in C:
        Cn *= int(c)
    n_rounds = max(-(-layout.nrows(q) // step) for q in range(P))
    cm = comm()
    if not cm.has_all_to_all:  # gloo: all-gather once, slice column blocks
        mine = zrows(z_shard, 0, R)
        full = all_gather_rows(mine.contiguous(), layout)
        for (i0, i1) in zblocks(R, step):
            yield i0, i1, full[:, r0 + i0: r0 + i1].transpose(0, 1)
        return
    zr = zrows(z_shard, 0, R)                              # [R, N, C] view of my rows

    def _blk(q, t):
        q0, q1 = layout.bounds[q]
        g0 = q0 + t * step
        g1 = min(q1, g0 + step)
        return g0, max(0, g1 - g0)

    for t in range(n_rounds):
        in_pieces, in_sizes, out_sizes = [], [], []
        _, my_b = _blk(layout.rank, t)                     # width of MY t-th block (0 if none)
        for q in range(P):
            g0, b_q = _blk(q, t)
            if b_q > 0 and R > 0:
                in_pieces.append(zr[:, g0:g0 + b_q].reshape(R * b_q, Cn))   # copy: (row j, col) row-major
            in_sizes.append(R * b_q)
            out_sizes.append(layout.nrows(q) * my_b)
        inp = torch.cat(in_pieces, 0) if in_pieces else probe.new_zeros((0, Cn))
        out = probe.new_empty((N * my_b, Cn))
        cm.all_to_all_single(out, inp.contiguous(), out_sizes=out_sizes, in_sizes=in_sizes)
        del inp, in_pieces
        if my_b > 0:
            i0 = t * step
            colblock = out.view((N, my_b) + C)                # z[:, r0+i0 : r0+i0+my_b, :] (rank-ordered rows)
            yield i0, i0 + my_b, colblock.transpose(0, 1)
        del out


# ----------------------------------------------------------------------------------------------------------------- ring streaming
def ring_blocks(x_shard, layout: Layout, *, rows: Optional[Tuple[int, int]] = None, dim: int = 0) -> Iterator[Tuple[int, object]]:
    """Stream every rank's block around the ring; yields ``(q, x_q)`` for EVERY rank q exactly once: t=0 the own block, then
    ``q = (rank - t) % P``.

    ``dim``          : the row dimension of ``x_shard`` (negative allowed: ``-3`` for a ``[*, n_loc, N, C]`` shard).
    ``rows=None``    : ``x_shard`` is the rank's whole shard along ``dim`` (``R_rank`` entries); blocks travel padded to ``Rmax``; ``x_q`` has
                     ``R_q`` entries.
    ``rows=(j0,RB)`` : ``x_shard`` is the rank's LOCAL sub-slab ``[j0:j0+RB]`` (clipped to R) along ``dim``; blocks travel padded to ``RB``;
                     ``x_q`` has ``clip(R_q - j0, 0, RB)`` entries (may be 0 -> still yielded).
    Double-buffered ``p2p_start`` on padded buffers (exactly P-1 exchanges; the padded own copy + one padded receive buffer resident, from the
    comm's :class:`Staging` scope of this pass: freed at exhaustion once every transfer completed; the caller's tensor never travels). The
    generator MUST be run
    to exhaustion on every rank (the prefetch for step t+1 is posted before x_t is yielded). The consumer must be done READING x_q before
    calling next() and must not modify it (on CUDA ordinary stream ordering is enough: the later irecv into that buffer is ordered after
    kernels already enqueued on the current stream)."""
    P, rank_ = layout.P, layout.rank
    dim = dim % x_shard.dim()
    if rows is None:
        unit = layout.Rmax
        valid = lambda q: layout.nrows(q)  # noqa: E731
    else:
        j0, RB = int(rows[0]), int(rows[1])
        unit = RB
        valid = lambda q: max(0, min(layout.nrows(q) - j0, RB))  # noqa: E731
    if layout.replicated or not is_dist() or P == 1:
        yield rank_, x_shard
        return
    if not (x_shard.shape[dim] == valid(rank_)):
        raise RowpairRefused("rowpair.dist: violated: " + repr(((tuple(x_shard.shape), dim, valid(rank_), rows))))
    cm = comm()
    st = cm.staging                              # the pass's wire buffers: own copy + two padded recv buffers, freed at exit once every transfer completed
    site = st.enter("ring_blocks")
    try:
        pshape = list(x_shard.shape)
        pshape[dim] = unit
        own = st.buffer(site, "own", pshape, x_shard.dtype, x_shard.device)     # the padded own copy (bit copy of the rows, zeroed padding):
        n = int(x_shard.shape[dim])                                              # the caller's tensor itself never goes on the wire
        if n > 0:
            own.narrow(dim, 0, n).copy_(x_shard)
        if unit > n:
            own.narrow(dim, n, unit - n).zero_()
        cur = own
        bufs = [st.buffer(site, "A", pshape, x_shard.dtype, x_shard.device), None]      # recv targets alternate: step t receives into bufs[t % 2]
        nxt_rank, prv_rank = (rank_ + 1) % P, (rank_ - 1) % P
        for t in range(P):
            q = (rank_ - t) % P
            h = None
            if t < P - 1:
                if bufs[t % 2] is None:          # t == 1: the own copy has been sent (waited) and consumed: it is the second recv buffer
                    bufs[1] = own
                tgt = bufs[t % 2]
                if not (tgt.data_ptr() != cur.data_ptr()):
                    raise RowpairRefused(f"rowpair.dist: violated: tgt.data_ptr() != cur.data_ptr()")
                h = cm.p2p_start({nxt_rank: cur}, {prv_rank: tgt}, "ring_blocks")
            yield q, cur.narrow(dim, 0, valid(q))
            if h is not None:
                h.wait()
                cur = bufs[t % 2]
    finally:
        st.exit(site)


# ----------------------------------------------------------------------------------------------------------------- replicated checks
def _int_view(t):
    t = t.detach().contiguous()
    if t.dtype == torch.bool:
        return t.to(torch.int64).reshape(-1)
    nb = t.element_size()
    if nb == 1:
        return t.view(torch.uint8).to(torch.int64).reshape(-1)
    if nb == 2:
        return t.view(torch.int16).to(torch.int64).reshape(-1)
    if nb == 4:
        return t.view(torch.int32).to(torch.int64).reshape(-1)
    return t.view(torch.int64).reshape(-1)


def _xor_fold(v) -> int:
    v = v.reshape(-1)
    while v.numel() > 1:
        n = v.numel()
        if n % 2:
            v = torch.cat([v, v.new_zeros(1)])
            n += 1
        v = torch.bitwise_xor(v[: n // 2], v[n // 2:])
    return int(v.item()) if v.numel() else 0


CHECKSUM_BLOCK_ELEMS = 1 << 24                                              # elements per weighting block (bounded int64 temporaries)
_MIX1 = -7046029254386353131                                                # 0x9E3779B97F4A7C15 as a signed int64
_MIX2 = -4658895280553007687                                                # 0xBF58476D1CE4E5B9 as a signed int64


def _position_weights(i0: int, i1: int, device):
    """Per-position int64 weights ``h(i)`` for flat indices ``[i0, i1)``: a wrapping multiply / xor-shift mix of the index (NOT linear in i,
    so weights do not separate into a row term plus a column term). Integer ops only: identical on every device and rank."""
    w = torch.arange(i0, i1, dtype=torch.int64, device=device)
    w.mul_(_MIX1)
    w.bitwise_xor_(w >> 29)
    w.mul_(_MIX2)
    w.bitwise_xor_(w >> 32)
    return w


def _position_weighted_sum(iv) -> int:
    """``sum_i h(i) * iv_i`` (wrapping int64 per block, folded to 64 bits across blocks) over the flat integer view: a PERMUTATION of the
    elements changes it — two MSA one-hot subsamples of equal size hold the same multiset of values (and equal row and column sums), so the
    plain sum, the xor-fold and any row-plus-column-separable weighting cannot tell them apart; this term can."""
    n = int(iv.numel())
    acc = 0
    for i0 in range(0, n, CHECKSUM_BLOCK_ELEMS):
        i1 = min(n, i0 + CHECKSUM_BLOCK_ELEMS)
        w = _position_weights(i0, i1, iv.device)
        acc = (acc + int((iv[i0:i1] * w).sum().item())) & 0xFFFFFFFFFFFFFFFF
        del w
    return acc


def checksum(t) -> Tuple[int, int, int, int]:
    """``(numel, wrapping int64 sum of the integer view, xor-fold, position-weighted sum)`` — a cheap bit-exact fingerprint of a tensor.
    The first three terms are permutation-invariant; the fourth (:func:`_position_weighted_sum`) is not, so a re-ordered or re-sampled
    tensor with the same multiset of values (e.g. a one-hot MSA subsample) has a different fingerprint."""
    iv = _int_view(t)
    if iv.numel() == 0:
        return 0, 0, 0, 0
    return int(t.numel()), int(iv.sum().item()), _xor_fold(iv), _position_weighted_sum(iv)


def allreduce_checksum(t, name: str = "tensor", raise_on_mismatch: bool = True):
    """Assert a REPLICATED tensor is bit-identical on all ranks (:func:`checksum` all-gathered). Returns the per-rank checksums; a
    mismatch raises :class:`RowpairRefused` naming ``name`` (a replicated input that differs across ranks is a broken run, never tolerated)."""
    local = checksum(t)
    if not is_dist():
        return [local]
    allv = comm().allgather_obj(local)
    if raise_on_mismatch and not all(x == allv[0] for x in allv):
        raise RowpairRefused(f"replicated tensor {name!r} differs across ranks: checksums {allv}")
    return allv


def agreed_free_bytes(device=None) -> Optional[int]:
    """This rank's free device bytes (:func:`opt_core.mem.budget.device_free_bytes`) at world size 1 or without a group — and, at world
    size > 1, the MINIMUM over ranks (one int64 ``allreduce_(min)``; a rank without CUDA counts as unbounded; None when no rank has CUDA).
    Every free-bytes-derived schedule of this package (``trimul`` RA / RB, row block sizes) derives from THIS value, so ranks under different
    memory pressure still run one (pass, slab, ring-step) schedule and meet in the same collectives. Called at points every rank reaches."""
    from ..budget import device_free_bytes
    free = device_free_bytes(device)
    if not is_dist():
        return free
    unbounded = 2 ** 62
    c = comm()
    t = torch.tensor([unbounded if free is None else int(free)], dtype=torch.int64, device=c.device)
    c.allreduce_(t, "min")
    v = int(t.item())
    return None if v >= unbounded else v


def broadcast_obj(obj, src: int = 0):
    """Broadcast a picklable object from ``src``; returns it on every rank."""
    if not is_dist():
        return obj
    return comm().broadcast_obj(obj, src)
