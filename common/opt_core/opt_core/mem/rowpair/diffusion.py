"""The diffusion module under row sharding — the CONDITIONED pair tensor ``z_cond[*, R, N, c]`` is built from this rank's rows of the
trunk pair shard and stays sharded for the whole roll-out; the diffusion transformer attends with token queries = this rank's rows
(pair bias rows local), keys / values from the replicated token activation, and re-replicates the updated rows once per block; the
atom attention (sequence-local windows, ``O(N_atom x c_atom_pair)``) stays REPLICATED by design and reads its token-pair input from a
diagonal BAND of the sharded conditioned pair (plus the few full rows an out-of-band pair needs, broadcast by their owners), so no
``[N, N, c]`` tensor is ever whole on a rank. Engine-free: the conditioning embedder, the transitions, the pair-bias projection, the
attention kernel and the block epilogue are the caller's CALLABLES (an adapter binds its engine's modules); this module holds the row
layout, the block schedule and the communication (all through :mod:`opt_core.mem.rowpair.dist` / :mod:`.shard` / :mod:`.bcast`;
every row block walks the engine's chunk grid: :func:`opt_core.mem.rowpair.shard.produce_rows_` / :func:`~opt_core.mem.rowpair.shard.iter_row_blocks`).

Statements (L2: a P == 1 / replicated / no-group layout is refused by name — at ``--n_gpu 1`` the engine's own diffusion module runs):

    pair_cond_rows(embed_fn, z_shard, layout, c_out, rows, transitions)
                                        seam 'diffusion conditioning (pair)': ``out[i] = embed_fn(z_shard[i-block], g0, g1)`` per local row
                                        block (the engine's ``linear(LN(cat([z_rows, relpos_rows(g0, g1)])))`` — relative-position rows enter
                                        through the callable, lazily per block: :func:`opt_core.mem.rowpair.trunk.relpos_onehot_rows` is the
                                        family's one producer), then every ``t`` of ``transitions`` IN PLACE per block:
                                        ``out[i-block] += t(out[i-block], g0, g1)`` (row-local; the same per-element add as ``z = z + t(z)``).
                                        Computed ONCE per roll-out by the adapter and reused by every step and sample ('conditioning-once':
                                        the statement is step-invariant, so once == per-step recompute, value for value). ``z_shard`` may be a
                                        host-parked shard serving ``.zrows(i0, i1)`` (:class:`opt_core.mem.rowpair.template.ParkedStorage`):
                                        the trunk shard's device storage is then released for the whole roll-out; same rows, same output.
    pair_bias_rows(bias_fn, z_shard, layout, rows)
                                        the per-block pair bias of the diffusion transformer for local query rows: ``bias_fn(z_rows) ->
                                        [*, rows, N, H]`` (the engine's ``linear(LN(z_rows))``) written as ``[*, H, R, N]`` contiguous.
    PairBiasCache(enabled)              the per-block biases depend on ``z_cond`` only: cached across the steps x samples of a roll-out when
                                        the schedule says they fit (``diff_bias_cache=on|off`` is printed); ``get(i, compute)``.
    dit_block_sharded(fns, a, s, bias, layout, q_rows)
                                        ONE diffusion-transformer block: ``x = fns.norm(a, s)`` (replicated, all rows), ``kv = fns.kv(x)`` (once),
                                        ``o[q] = fns.attn(x[q], kv, bias[:, q], (g0, g1))`` over local query-row SUB-BLOCKS of ``q_rows`` rows
                                        (bounds the ``S x H x q x N`` logits transient; the softmax over all N keys is whole per query row),
                                        ``a_rows = fns.update(a, o, s, (r0, r1))`` (gate, residual, conditioned transition: row-local engine
                                        statements on local rows), then ONE all_gather of the rows -> replicated ``a``. The local-query call is
                                        :func:`opt_core.mem.rowpair.transition.apb_local_queries` (the family's one attention-pair-bias seam).
    diffusion_transformer_sharded(blocks, a, s, z_cond_shard, layout, schedule, bias_cache)
                                        the block loop: collectives = one all_gather of ``[*, N, c_token]`` per block per call (a 24-block
                                        transformer over 200 steps = 4800 all_gathers per sample batch).
    gather_band_rows(x_loc, layout, W, extra_rows) -> (band, extras)
                                        the band op on an already projected shard ``x_loc [*, R, N, C]``: ``band [*, N, 2W+1, C]`` replicated
                                        (``band[i, t] = x[i, i+t-W]``) by one all_gather of the local band rows + the named full rows
                                        ``extras [*, E, N, C]`` broadcast by their owners — never an ``[N, N, C]`` all-gather.
    band_plan(q_idx, k_idx, pair_valid, N) / pair_band_rows(proj_fn, z_shard, layout, plan, rows) / band_lookup(band, extras, plan)
                                        the atom-attention pair input: ``zp = proj_fn(z_cond rows)`` is row-local; ``band[i, t] = zp[i, i+t-W]``
                                        for ``|t| <= W`` is built from local rows and all_gathered (``[*, N, 2W+1, C]``, W = the largest
                                        |k_idx - q_idx| over valid pairs, capped); every row an out-of-band valid pair needs is fetched WHOLE
                                        from its owner (``extras [*, E, N, C]``, E named and capped — typically the padding row 0), so
                                        ``band_lookup`` returns exactly ``zp[b, q_idx, k_idx]`` for ANY atom->token map.
    sync_replicated(t, name, mode)      the per-step proof that a replicated draw (the initial noise, the per-step noise, the augmentation) IS
                                        replicated: ``guard`` = the family's replication proof :func:`trunk.guard_replicated`, mismatch refused by name;
                                        ``bcast`` = rank 0's tensor to every rank (:func:`bcast.broadcast_tensordict`) then guard; ``off`` = the
                                        engine's identically seeded generators are trusted (RECORDED as ``diff_noise_sync=off``, never silent).
    DiffusionSchedule.decide(layout, ...)
                                        the row blocks of the statements above, decided ONCE per roll-out from explicit arguments, the
                                        ``ROWPAIR_DIFF_*_ROWS`` pins, or the per-rank work budget (``ROWPAIR_DIFF_WORK_GB``, default 8 GB;
                                        ``auto`` = a share of the free bytes AGREED across ranks, :func:`dist.agreed_free_bytes`, one collective)
                                        through :func:`shard.choose_block_rows`, and RECORDED (:func:`evidence.record_schedule`: ``diff_cond_rows
                                        diff_bias_rows diff_q_rows diff_band_rows diff_bias_cache diff_rows_source diff_work_bytes diff_replicated``).

Replicated by design (named here and in the schedule census): the single conditioning ``s`` (depends on the noise level: every step,
``O(N)``), the token activation ``a[*, S, N, c_token]`` between blocks, the atom tensors and coordinates, the atom attention encoder /
decoder. Memory per rank (u = ``N^2 c elt / P``): z_cond shard 1 u resident for the roll-out (+ the trunk shard it was built from, the
caller's to free); transients ``rows x N x (c_in + c)`` (conditioning), ``rows x N x (c + 2H)`` (bias), ``S x H x q_rows x N x 4``
(logits); bias cache ``n_blocks x H x Rmax x N`` when on; band ``N x (2W+1) x C`` + extras ``E x N x C`` replicated (``O(N)``).

Numerics class (API.md table): conditioning / bias / band projection are row-local (per-element identical to the dense statement,
launch M = local rows); the attention is whole per query row (bit-exact iff the kernel is query-count invariant); the gathers, the band
assembly and the extras broadcast are data movement (bit-exact). Every rank issues the engine's RNG calls in the engine's order (the
roll-out loop is the engine's), so identically seeded ranks draw identical noise; :func:`sync_replicated` proves or enforces it.
"""
from __future__ import annotations

import os
from collections import namedtuple
from typing import Callable, List, Optional, Sequence, Tuple

from . import RowpairRefused
from ._torch import torch
from .dist import Layout, agreed_free_bytes, env_int, is_dist, require_sharded
from .evidence import record_schedule


def census_token(text) -> str:
    """:func:`opt_core.mem.rowpair.trunk.census_token` (one ``\S+`` token per census VALUE; imported at call time: no import-order change here)."""
    from .trunk import census_token as _census_token
    return _census_token(text)
from .shard import choose_block_rows, iter_row_blocks, produce_rows_, unshard_rows
from .transition import apb_local_queries

__all__ = ["ENV_COND_ROWS", "ENV_BIAS_ROWS", "ENV_Q_ROWS", "ENV_BAND_ROWS", "ENV_BIAS_CACHE", "ENV_BAND_W", "ENV_BAND_EXTRA_MAX",
           "ENV_NOISE_SYNC", "ENV_WORK_GB", "ENV_BIAS_CACHE_GB", "WORK_GB_DEFAULT", "BIAS_CACHE_GB_DEFAULT", "WORK_FRAC", "BIAS_CACHE_FRAC",
           "INT32_MAX", "BAND_EXTRA_MAX_DEFAULT", "DiffusionSchedule", "pair_cond_rows",
           "pair_bias_rows", "PairBiasCache", "DiTBlockFns", "dit_block_sharded", "diffusion_transformer_sharded", "BandPlan", "band_plan",
           "gather_band_rows", "pair_band_rows", "band_lookup", "sync_replicated", "NOISE_SYNC_MODES",
           "ENV_DIFF_BIAS", "DIFF_BIAS_WORDS", "dit_bias_word", "DitBias", "pair_bias_rows_into", "dit_bias_census",
           "ATTN_CORE_WORDS", "DitKV", "dit_rows_core", "dit_attention_rows", "dit_rows_census", "cast16"]

INT32_MAX = 2 ** 31 - 1
WORK_GB_DEFAULT = 8.0                             # per-rank transient budget the row blocks are sized from (ROWPAIR_DIFF_WORK_GB; 'auto' = WORK_FRAC of agreed free)
BIAS_CACHE_GB_DEFAULT = 24.0                      # per-rank budget of the pair-bias cache decision (ROWPAIR_DIFF_BIAS_CACHE_GB; 'auto' = BIAS_CACHE_FRAC)
WORK_FRAC = 0.10                                  # 'auto': share of the AGREED free bytes one row-block transient may take (freed per block)
BIAS_CACHE_FRAC = 0.30                            # 'auto': share of the AGREED free bytes the per-block pair-bias cache may take
ROW_MULTIPLE = 4                                  # budget-derived row blocks are multiples of 4 rows (block starts keep their address class)
ENV_WORK_GB = "ROWPAIR_DIFF_WORK_GB"              # the transient budget in GB (float) | auto
ENV_BIAS_CACHE_GB = "ROWPAIR_DIFF_BIAS_CACHE_GB"  # the bias-cache budget in GB (float) | auto
ENV_COND_ROWS = "ROWPAIR_DIFF_COND_ROWS"          # pins the conditioning row block          (printed diff_rows_source=...cond:env...)
ENV_BIAS_ROWS = "ROWPAIR_DIFF_BIAS_ROWS"          # pins the pair-bias projection row block
ENV_Q_ROWS = "ROWPAIR_DIFF_Q_ROWS"                # pins the query-row sub-block of the transformer attention (0 = all local rows)
ENV_BAND_ROWS = "ROWPAIR_DIFF_BAND_ROWS"          # pins the band projection row block
ENV_BIAS_CACHE = "ROWPAIR_DIFF_BIAS_CACHE"        # 0 | 1 | auto (default auto: on iff n_blocks*H*Rmax*N*elt <= ROWPAIR_DIFF_BIAS_CACHE_GB, default 24 GB)
ENV_BAND_W = "ROWPAIR_DIFF_BAND_W"                # the ADAPTERS' name for band_plan(max_w=) (the core reads no env for it: R-TP-3, a gather-size lever is explicit)
ENV_BAND_EXTRA_MAX = "ROWPAIR_DIFF_BAND_EXTRA_MAX"  # the ADAPTERS' name for band_plan(max_extra_rows=)
BAND_EXTRA_MAX_DEFAULT = 256                      # band_plan(max_extra_rows=None): more full rows than this is refused by name
ENV_NOISE_SYNC = "ROWPAIR_DIFF_NOISE_SYNC"        # guard | bcast | off (default guard)
NOISE_SYNC_MODES = ("guard", "bcast", "off")
ENV_DIFF_BIAS = "ROWPAIR_DIFF_BIAS"                # engine | ln_proj | ln_proj_fp32 (default engine): the transformer's pair-bias PRODUCER when a block carries DiTBlockFns.bias_into
DIFF_BIAS_WORDS = ("engine", "ln_proj", "ln_proj_fp32")    # ln_proj: bf16 x bf16 MMA (stock's rounding point under a bf16 trunk; bf16-class on fp32 rows); ln_proj_fp32: fp32 IEEE dot (fp32-exact, ~24x the MMA's time at c=128 — slower than the engine statement: an opt-in yardstick, not a speed lever)
ATTN_CORE_WORDS = ("kernel",)                     # DiffusionSchedule.decide(attn_core=): the adapter bound a flash attention core on the query rows (no logits transient)


# ----------------------------------------------------------------------------------------------------------------- schedule
def _env_rows(env: Optional[str]) -> Optional[int]:
    if not env or not os.environ.get(env, "").strip() or os.environ.get(env, "").strip().lower() == "auto":
        return None
    try:
        v = env_int(env, 0)
    except ValueError as e:
        raise RowpairRefused(f"diffusion: {e}") from None
    if v < 0:
        raise RowpairRefused(f"diffusion: {env}={v}: a non-negative row count is required (0 = all local rows)")
    return v


def _env_gb(env: str, default: float) -> Optional[float]:
    """GB float of ``env`` (``default`` when unset); None when the pin says ``auto`` (= derive from the agreed free bytes)."""
    v = os.environ.get(env, "").strip().lower()
    if not v:
        return float(default)
    if v == "auto":
        return None
    try:
        f = float(v)
    except ValueError:
        raise RowpairRefused(f"diffusion: {env}={v!r}: a number of GB or 'auto' is required") from None
    if f < 0:
        raise RowpairRefused(f"diffusion: {env}={v!r}: a non-negative number of GB is required")
    return f


def _choose_rows(bytes_per_row: int, elems_per_row: int, Rmax: int, *, rows: Optional[int], env: Optional[str],
                 budget_bytes: Optional[int]) -> Tuple[int, str]:
    """One row block through THE chooser (:func:`opt_core.mem.rowpair.shard.choose_block_rows`): ``rows`` given (0 = the whole shard) ->
    ``given``; else this statement's row-count pin ``env`` (0 = the whole shard) -> ``env:<NAME>``; else the work budget -> ``budget`` (multiples
    of ``ROW_MULTIPLE``); no budget -> the chooser's default MiB target -> ``default``. Always: the int32 launch cap, clamp to ``Rmax``."""
    pinned = None if rows is not None else _env_rows(env)
    given = rows if rows is not None else pinned
    if given is not None:
        if int(given) < 0:
            raise RowpairRefused(f"diffusion: rows={given}: a non-negative row count is required (0 = all local rows)")
        given = int(given) or int(Rmax)
    r, source = choose_block_rows(rows=given, budget_bytes=None if given is not None else budget_bytes, bytes_per_row=int(bytes_per_row),
                                  elems_per_row=int(elems_per_row), cap_elems=INT32_MAX, align=1 if given is not None else ROW_MULTIPLE,
                                  n_max=int(Rmax), env=None)
    if rows is None and pinned is not None:
        source = f"env:{env}" + source[len("given"):]
    return r, source


class DiffusionSchedule(object):
    """The row blocks of one roll-out, identical on every rank: ``cond_rows`` (conditioning), ``bias_rows`` (pair-bias projection),
    ``q_rows`` (attention query sub-block), ``band_rows`` (band projection), ``bias_cache`` (bool). Built by :meth:`decide`; ``fields()``
    are the evidence pairs (also recorded through :func:`opt_core.mem.rowpair.evidence.record_schedule`)."""

    __slots__ = ("cond_rows", "bias_rows", "q_rows", "band_rows", "bias_cache", "sources", "budget_bytes", "bias_cache_bytes", "attn_core", "q_rows_stock")

    def __init__(self, cond_rows: int, bias_rows: int, q_rows: int, band_rows: int, bias_cache: bool, sources: dict,
                 budget_bytes: Optional[int] = None, bias_cache_bytes: int = 0, attn_core: Optional[str] = None,
                 q_rows_stock: Optional[int] = None):
        self.cond_rows, self.bias_rows, self.q_rows, self.band_rows = int(cond_rows), int(bias_rows), int(q_rows), int(band_rows)
        self.bias_cache = bool(bias_cache)
        self.sources = dict(sources)
        self.budget_bytes = budget_bytes
        self.bias_cache_bytes = int(bias_cache_bytes)
        self.attn_core = attn_core                                                   # None (today: the engine statement in q sub-blocks) | 'kernel'
        self.q_rows_stock = int(q_rows_stock) if q_rows_stock is not None else self.q_rows   # the byte-model query block the STOCK statement fits (a kernel's fallback chunks by it)

    @classmethod
    def decide(cls, layout: Layout, *, c_z: int, c_in: int, c_cond: int, H: int, S: int, n_blocks: int, c_pair: int = 0, elt: int = 4,
               cond_rows: Optional[int] = None, bias_rows: Optional[int] = None, q_rows: Optional[int] = None,
               band_rows: Optional[int] = None, bias_cache: Optional[bool] = None, budget_bytes: Optional[int] = None,
               record: bool = True, attn_core: Optional[str] = None) -> "DiffusionSchedule":
        """Decide every block size ONCE per roll-out, at a point every rank reaches. Each row block: the explicit argument (``given``), else its
        ``ROWPAIR_DIFF_*_ROWS`` pin (``env``), else the largest block whose transient fits the work budget — ``ROWPAIR_DIFF_WORK_GB`` (default
        ``WORK_GB_DEFAULT`` = 8 GB per rank: ``work:default`` / ``work:env``; ``auto`` = ``WORK_FRAC`` of the free bytes AGREED across ranks, one
        :func:`dist.agreed_free_bytes` collective: ``work:budget``; no CUDA on any rank: ``work:all`` and the chooser's default MiB target)
        through THE chooser :func:`opt_core.mem.rowpair.shard.choose_block_rows` (int32 cap, ``ROW_MULTIPLE`` alignment, clamp to Rmax). Byte model per
        row (x ``N x elt``): conditioning ``max(2 c_in + 3 (c_in - c_z) + c_cond, 9 c_cond)`` (embed transient vs a 2x-expansion transition),
        bias ``c_cond + 2 H``, band ``c_cond + 2 c_pair``, attention ``4 S H`` (logits + softmax + copies). The bias cache (``n_blocks H Rmax N
        elt`` bytes) is on iff it fits ``ROWPAIR_DIFF_BIAS_CACHE_GB`` (default 24 GB; ``auto`` = ``BIAS_CACHE_FRAC`` of the agreed free bytes),
        or as pinned by ``ROWPAIR_DIFF_BIAS_CACHE=0|1`` / ``bias_cache=``. ``budget_bytes`` (tests) replaces the free-bytes reading.
        ``attn_core='kernel'`` (an adapter that bound :func:`dit_attention_rows` and whose core was admitted, :func:`dit_rows_core`): the query
        block has no ``[S, H, q, N]`` logits transient, so ``q_rows`` = all local rows (source ``q:kernel``) unless given / pinned; the byte-model
        block is kept as ``q_rows_stock`` (the stock statement's chunk when the kernel refuses a call by name). ``attn_core=None``: today's fields,
        value for value."""
        N, Rmax = layout.N, layout.Rmax
        mode = os.environ.get(ENV_BIAS_CACHE, "auto").strip().lower() or "auto"
        if mode not in ("0", "1", "auto", "on", "off"):
            raise RowpairRefused(f"DiffusionSchedule: {ENV_BIAS_CACHE}={mode!r}: one of 0 | 1 | auto")
        if attn_core is not None and attn_core not in ATTN_CORE_WORDS:
            raise RowpairRefused(f"DiffusionSchedule: attn_core={attn_core!r}: one of {ATTN_CORE_WORDS} or None")
        rows_open = any(v is None and _env_rows(e) is None for v, e in
                        ((cond_rows, ENV_COND_ROWS), (bias_rows, ENV_BIAS_ROWS), (q_rows, ENV_Q_ROWS), (band_rows, ENV_BAND_ROWS)))
        cache_open = bias_cache is None and mode == "auto"
        work_gb, cache_gb = _env_gb(ENV_WORK_GB, WORK_GB_DEFAULT), _env_gb(ENV_BIAS_CACHE_GB, BIAS_CACHE_GB_DEFAULT)
        need_free = budget_bytes is None and ((rows_open and work_gb is None) or (cache_open and cache_gb is None))
        free = agreed_free_bytes() if need_free else None                        # ONE collective ('auto' budgets only); None = no CUDA on any rank
        srcs = {}
        if budget_bytes is not None:
            work, srcs["work"] = max(0, int(int(budget_bytes) * WORK_FRAC)), "given"
        elif work_gb is not None:
            work, srcs["work"] = int(work_gb * 1e9), ("env" if os.environ.get(ENV_WORK_GB, "").strip() else "default")
        else:
            work, srcs["work"] = (None if free is None else max(0, int(int(free) * WORK_FRAC))), ("all" if free is None else "budget")
        if budget_bytes is not None:
            cache_budget, srcs["cache_budget"] = int(int(budget_bytes) * BIAS_CACHE_FRAC), "given"
        elif cache_gb is not None:
            cache_budget, srcs["cache_budget"] = int(cache_gb * 1e9), ("env" if os.environ.get(ENV_BIAS_CACHE_GB, "").strip() else "default")
        else:
            cache_budget, srcs["cache_budget"] = (None if free is None else int(int(free) * BIAS_CACHE_FRAC)), ("all" if free is None else "budget")
        c_rel = max(0, int(c_in) - int(c_z))
        cr, srcs["cond"] = _choose_rows(N * elt * max(2 * c_in + 3 * c_rel + c_cond, 9 * c_cond), N * max(c_in, 2 * c_cond), Rmax,
                                        rows=cond_rows, env=ENV_COND_ROWS, budget_bytes=work)
        br, srcs["bias"] = _choose_rows(N * elt * (c_cond + 2 * H), N * max(c_cond, H), Rmax, rows=bias_rows, env=ENV_BIAS_ROWS, budget_bytes=work)
        qr, srcs["q"] = _choose_rows(N * elt * 4 * max(1, S) * H, N * max(1, S) * H, Rmax, rows=q_rows, env=ENV_Q_ROWS, budget_bytes=work)
        qr_stock = qr
        if attn_core == "kernel" and q_rows is None and _env_rows(ENV_Q_ROWS) is None:
            qr, srcs["q"] = Rmax, "kernel"                                         # a flash core holds no logits: one launch per block over all local query rows
        dr, srcs["band"] = _choose_rows(N * elt * (c_cond + 2 * max(1, c_pair)), N * max(c_cond, 1), Rmax, rows=band_rows, env=ENV_BAND_ROWS,
                                        budget_bytes=work)
        cache_bytes = int(n_blocks) * int(H) * Rmax * N * elt
        if bias_cache is not None:
            on, srcs["bias_cache"] = bool(bias_cache), "given"
        elif mode in ("1", "on"):
            on, srcs["bias_cache"] = True, "env"
        elif mode in ("0", "off"):
            on, srcs["bias_cache"] = False, "env"
        elif cache_budget is None:
            on, srcs["bias_cache"] = True, "all"                                   # no CUDA on any rank and no GB pin: nothing to budget against
        else:
            on, srcs["bias_cache"] = cache_bytes <= int(cache_budget), srcs["cache_budget"]
        if attn_core is None:
            sch = cls(cr, br, qr, dr, on, srcs, budget_bytes=work, bias_cache_bytes=cache_bytes if on else 0)
        else:
            sch = cls(cr, br, qr, dr, on, srcs, budget_bytes=work, bias_cache_bytes=cache_bytes if on else 0, attn_core=attn_core, q_rows_stock=qr_stock)
        if record:
            record_schedule(**dict(sch.fields()))
        return sch

    def fields(self) -> List[Tuple[str, object]]:
        src = ",".join(f"{k}:{self.sources[k]}" for k in ("work", "cond", "bias", "q", "band", "bias_cache") if k in self.sources)
        out = [("diff_cond_rows", self.cond_rows), ("diff_bias_rows", self.bias_rows), ("diff_q_rows", self.q_rows),
               ("diff_band_rows", self.band_rows), ("diff_bias_cache", "on" if self.bias_cache else "off"), ("diff_rows_source", src),
               ("diff_work_bytes", self.budget_bytes if self.budget_bytes is not None else "none"),
               ("diff_replicated", "s,a,atoms,atom_attention")]
        if self.attn_core is not None:                                               # a NEW pair, present only when an adapter bound a rows core
            out.append(("diff_attn_core", f"{self.attn_core}:q_stock={self.q_rows_stock}"))
        return out

    def __repr__(self) -> str:
        return "DiffusionSchedule(" + ", ".join(f"{k}={v}" for k, v in self.fields()) + ")"


# ----------------------------------------------------------------------------------------------------------------- checks
def _check_shard(z_shard, layout: Layout, what: str) -> Tuple[int, int]:
    if len(tuple(z_shard.shape)) < 3 or int(z_shard.shape[-3]) != layout.R or int(z_shard.shape[-2]) != layout.N:
        raise RowpairRefused(f"{what}: z shard {tuple(z_shard.shape)} vs layout rows R={layout.R} N={layout.N} (rows on dim -3, columns on dim -2)")
    return layout.R, layout.N


def _all_rows(layout: Layout, rows: Optional[int]) -> int:
    """A statement's row block: ``rows`` >= 1 as given (clipped to the shard); None / 0 = the whole shard as one block."""
    R = max(1, int(layout.R))
    return R if not rows or int(rows) <= 0 else min(R, int(rows))


# ----------------------------------------------------------------------------------------------------------------- seam: pair conditioning
def pair_cond_rows(embed_fn: Callable[[object, int, int], object], z_shard, layout: Layout, *, c_out: int, rows: Optional[int] = None,
                   transitions: Sequence[Callable[[object, int, int], object]] = (), out=None):
    """The conditioned pair rows of this rank, ``[*, R, N, c_out]`` in ``z_shard``'s dtype / device. ``embed_fn(z_rows, g0, g1) -> [*, rows, N,
    c_out]`` is the engine's conditioning embedder for the GLOBAL rows ``[g0, g1)`` of a block (``z_rows = z_shard[..., g0-r0:g1-r0, :, :]``;
    the relative-position rows of the block are built INSIDE the callable — ``opt_core.mem.rowpair.trunk.relpos_onehot_rows`` or the engine's own —
    so no ``[N, N, c_rel]`` one-hot exists); each ``t(x_rows, g0, g1) -> [*, rows, N, c_out]`` of ``transitions`` is then ADDED in place block by
    block (the engine's pair transition with its mask rows ``[g0, g1)``). ``rows``: the block (``DiffusionSchedule.cond_rows``; None = one
    block). ``out``: a preallocated ``[*, R, N, c_out]`` to fill (else allocated here). ``z_shard`` may be a host-PARKED shard — a duck-typed
    source with ``.zrows(i0, i1) -> [*, rows, N, C]`` and ``.shape / .dtype / .device`` (:class:`.template.ParkedStorage`: the trunk shard's device
    storage released for the roll-out, its rows staged block by block): the same rows, hence the same output."""
    require_sharded(layout, "pair_cond_rows")
    R, N = _check_shard(z_shard, layout, "pair_cond_rows")
    lead = tuple(int(s) for s in z_shard.shape[:-3])
    r0 = layout.r0
    if out is None:
        out = torch.empty(lead + (R, N, int(c_out)), dtype=z_shard.dtype, device=z_shard.device)
    elif tuple(int(s) for s in out.shape) != lead + (R, N, int(c_out)):
        raise RowpairRefused(f"pair_cond_rows: out {tuple(out.shape)} vs {lead + (R, N, int(c_out))}")
    rows = _all_rows(layout, rows)
    blk = (lambda g0, g1: z_shard.zrows(g0 - r0, g1 - r0)) if hasattr(z_shard, "zrows") else (lambda g0, g1: z_shard[..., g0 - r0:g1 - r0, :, :])
    produce_rows_(out, layout, lambda g0, g1: embed_fn(blk(g0, g1), g0, g1), op="set", block_rows=rows)
    for t in transitions:
        produce_rows_(out, layout, lambda g0, g1, t=t: t(out[..., g0 - r0:g1 - r0, :, :], g0, g1), op="add", block_rows=rows)
    return out


# ----------------------------------------------------------------------------------------------------------------- seam: transformer
def pair_bias_rows(bias_fn: Callable[[object], object], z_shard, layout: Layout, *, rows: Optional[int] = None, out=None):
    """``[*, H, R, N]`` contiguous: ``bias_fn(z_rows) -> [*, rows, N, H]`` (the engine's ``linear(LN(z_rows))`` of one transformer block) per
    local row block, heads moved first (the layout the attention adds to its ``[*, H, q, N]`` logits)."""
    require_sharded(layout, "pair_bias_rows")
    R, N = _check_shard(z_shard, layout, "pair_bias_rows")
    lead = tuple(int(s) for s in z_shard.shape[:-3])
    for b0, b1, g0, g1 in iter_row_blocks(layout, _all_rows(layout, rows)):
        y = bias_fn(z_shard[..., b0:b1, :, :])
        if tuple(int(s) for s in y.shape[:-1]) != lead + (b1 - b0, N):
            raise RowpairRefused(f"pair_bias_rows: bias_fn returned {tuple(y.shape)} for rows [{g0}, {g1}) (expected {lead + (b1 - b0, N)} + (H,))")
        H = int(y.shape[-1])
        if out is None:
            out = y.new_empty(lead + (H, R, N))
        out[..., :, b0:b1, :].copy_(y.movedim(-1, -3))                                # heads first: written per block, no shard-sized permute copy
        del y
    return out


def dit_bias_word(word: Optional[str] = None) -> str:
    """The transformer pair-bias PRODUCER word: ``word`` as given, else ``ROWPAIR_DIFF_BIAS``, else ``engine`` (today's statement). One of
    :data:`DIFF_BIAS_WORDS` (``engine`` | ``ln_proj`` = the kernel's bf16 MMA | ``ln_proj_fp32`` = its fp32 IEEE dot); anything else is
    :class:`RowpairRefused` by name."""
    w = (word if word is not None else os.environ.get(ENV_DIFF_BIAS, "")).strip().lower() or "engine"
    if w not in DIFF_BIAS_WORDS:
        raise RowpairRefused(f"{ENV_DIFF_BIAS}={w!r}: one of {DIFF_BIAS_WORDS}")
    return w


_DIT_BIAS_CENSUS = {}                              # producer word served -> row blocks written ('ln_proj', 'engine', 'engine:<refusal>' ...)


def dit_bias_census(reset: bool = False) -> str:
    """``dit_bias=<word>:<n>[,<word>:<n>]`` — the row blocks :func:`pair_bias_rows_into` wrote per producer word this process (``engine:<reason>``
    = the kernel producer stepped aside by name for those blocks). Recorded through :func:`evidence.record_schedule` as it changes."""
    if reset:
        _DIT_BIAS_CENSUS.clear()
    return ",".join(f"{k}:{v}" for k, v in _DIT_BIAS_CENSUS.items()) or "none"


def _count_bias(word: str, n: int = 1) -> None:
    _DIT_BIAS_CENSUS[word] = _DIT_BIAS_CENSUS.get(word, 0) + int(n)
    record_schedule(dit_bias=dit_bias_census())


class DitBias(object):
    """The pair-bias producer of ONE transformer block as WEIGHTS — the optional ``bias_into`` field of :class:`DiTBlockFns`:
    ``engine(z_rows) -> [*, rows, N, H]`` (today's ``bias`` callable: the engine's ``linear(LN(z_rows))`` — the reference statement and the
    fallback by name), ``ln_weight`` / ``ln_bias`` (``[c]`` or None), ``weight`` (``linear.weight [H, c]``), ``linear_bias`` (``[H]`` or None),
    ``eps``. :func:`pair_bias_rows_into` writes each row block head-major straight into its ``[*, H, rows, N]`` slice by the producer
    :func:`dit_bias_word` names: ``engine`` — ``slice.copy_(engine(z_rows).movedim(-1, -3))``, value for value :func:`pair_bias_rows`;
    ``ln_proj`` | ``ln_proj_fp32`` — :func:`opt_core.kernels.ln_proj.pair_bias` with ``out=`` the slice (LayerNorm + projection + head-major write in ONE pass over
    the rows: no ``[rows, N, H]`` intermediate, no permute copy; numerics = that kernel's: LN statistics fp32, the normalised row rounded to bf16 into a
    bf16 x bf16 MMA with fp32 accumulation (stock's rounding point under a bf16 trunk; bf16-class — max 1.5e-2 / rms 2.4e-3 on unit-rms planes — vs an
    fp32 engine's LN -> Linear), one output rounding; ``ln_proj_fp32`` — the same kernel packed ``dot_fp32`` (fp32 row x fp32 weights, IEEE dot: fp32-exact,
    ~24x the MMA's time at c = 128, slower than the engine statement — a yardstick, not a speed lever); either refused by name (import,
    ``linear_bias``, the kernel's ``Unsupported(reason)``: not_cuda / dtype / width / heads / ...) -> the engine statement for that block, counted
    ``engine:<reason>`` in :func:`dit_bias_census`. Widths: ``opt_core.kernels.ln_proj.ROWS_SERVED_C_PAIR`` (the x1 pair-bias widths 64 | 128 plus
    256, packed through ``pack_pair_bias_weights_rows`` — the x1 set ``SERVED_C_PAIR`` and its packer are untouched)."""

    __slots__ = ("engine", "ln_weight", "ln_bias", "weight", "linear_bias", "eps", "_packed", "_off")

    def __init__(self, engine: Callable[[object], object], ln_weight=None, ln_bias=None, weight=None, linear_bias=None, eps: float = 1e-5):
        self.engine = engine
        self.ln_weight, self.ln_bias, self.weight, self.linear_bias, self.eps = ln_weight, ln_bias, weight, linear_bias, float(eps)
        self._packed = {}                                                             # device -> (ln_proj module, packed weights)
        self._off = None                                                              # the sticky refusal word once the kernel producer cannot serve this block

    def engine_into(self, z_rows, out_slice) -> None:
        y = self.engine(z_rows)
        if tuple(int(s) for s in y.shape[:-1]) != tuple(int(s) for s in out_slice.shape[:-3]) + (int(out_slice.shape[-2]), int(out_slice.shape[-1])):
            raise RowpairRefused(f"pair_bias_rows_into: engine returned {tuple(y.shape)} for a slice {tuple(out_slice.shape)} ([*, rows, N, H] expected)")
        out_slice.copy_(y.movedim(-1, -3))

    def _ln_proj(self, device, dot_fp32: bool = False):
        got = self._packed.get((device, bool(dot_fp32)))
        if got is None:
            if self.weight is None:
                raise _IntoRefused("no_weight")
            if self.linear_bias is not None:
                raise _IntoRefused("linear_bias")                                     # ln_proj.pair_bias projects without a bias term
            try:
                from ...kernels.apb import carried_module, Refusal
                try:
                    LP = carried_module("ln_proj")
                except Refusal as r:
                    raise _IntoRefused("import") from r
                pack = getattr(LP, "pack_pair_bias_weights_rows", None) or LP.pack_pair_bias_weights   # the ROWS packer: the x1 widths + c 256 (ln_proj.ROWS_SERVED_C_PAIR)
                packed = pack(None if self.ln_weight is None else self.ln_weight.detach(),
                              None if self.ln_bias is None else self.ln_bias.detach(), self.weight.detach(), self.eps, device, dot_fp32=bool(dot_fp32))
            except _IntoRefused:
                raise
            except ImportError as e:
                raise _IntoRefused("import") from e
            except Exception as e:                                                    # noqa: BLE001 — OOM propagates (never rerouted); the packer's Unsupported(reason) / anything else is a refusal by name
                from ...oom import is_oom
                if is_oom(e): raise                                                   # noqa: E701
                raise _IntoRefused(getattr(e, "reason", type(e).__name__)) from e
            got = (LP, packed)
            self._packed[(device, bool(dot_fp32))] = got
        return got

    def ln_proj_into(self, z_rows, out_slice, dot_fp32: bool = False) -> None:
        """``out_slice [*, H, rows, N] <- Linear(LN(z_rows [*, rows, N, c]))`` head-major in place through the kernel, or :class:`_IntoRefused`.
        ``dot_fp32``: the ``ln_proj_fp32`` word (fp32 IEEE dot; fp32-exact and slow) instead of the bf16 MMA."""
        if self._off is not None:
            raise _IntoRefused(self._off)
        z, o = z_rows, out_slice
        while z.dim() > 4 and int(z.shape[0]) == 1:
            z = z[0]
        while o.dim() > z.dim() and int(o.shape[0]) == 1:
            o = o[0]
        if z.dim() == 4 and int(z.shape[0]) == 1 and o.dim() == 4:
            z, o = z[0], o[0]
        if z.dim() not in (3, 4) or o.dim() != z.dim():
            raise _IntoRefused("lead_dims")
        LP, packed = self._ln_proj(z.device, dot_fp32=bool(dot_fp32))       # ln_proj: bf16 x bf16 MMA, fp32 accumulate (bf16-class on fp32 rows); ln_proj_fp32: the fp32 row and weights through an IEEE fp32 dot
        try:
            LP.pair_bias(z, packed, out_layout="bhij", out_dtype=o.dtype, out=o)
        except Exception as e:                                                        # noqa: BLE001 — OOM propagates (never rerouted); Unsupported(reason) / a compile or launch failure is a refusal by name
            from ...oom import is_oom
            if is_oom(e): raise                                                       # noqa: E701
            reason = getattr(e, "reason", None) or ("launch:" + type(e).__name__)
            if reason in ("not_cuda", "dtype", "width", "heads", "out_dtype", "device", "packed_kind") or reason.startswith("launch:"):
                self._off = reason                                                    # static for this block: never re-probed
            raise _IntoRefused(reason) from e


class _IntoRefused(Exception):
    """A kernel producer stepped aside by name for a row block (``reason``); :func:`pair_bias_rows_into` runs the engine statement for it."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = str(reason)


def pair_bias_rows_into(spec, z_shard, layout: Layout, *, rows: Optional[int] = None, out=None, heads: Optional[int] = None, dtype=None,
                        probe_fn: Optional[Callable[[object], object]] = None, word: Optional[str] = None):
    """``[*, H, R, N]``: the per-block pair bias of the diffusion transformer for this rank's query rows, each local row block written head-major
    STRAIGHT INTO its ``out[..., :, b0:b1, :]`` slice (beside :func:`pair_bias_rows`, which materialises ``[*, rows, N, H]`` and copies it permuted).
    ``spec``: a :class:`DitBias` (weights: the producer is :func:`dit_bias_word` — ``engine`` (default) | ``ln_proj``, the engine statement the
    fallback by name per block, census :func:`dit_bias_census`) or a callable ``into_fn(z_rows [*, rows, N, c], out_slice [*, H, rows, N]) -> None``
    (an adapter's own in-place producer; counted ``adapter``). ``out``: a preallocated ``[*, H, R, N]``; else allocated once from ``heads`` /
    ``dtype`` — either may be None when ``probe_fn`` (the engine statement, e.g. ``DiTBlockFns.bias``) is given: it is evaluated on ONE pair element
    (``z_shard[..., :1, :1, :]``) to read the heads and the dtype the engine produces under the caller's autocast (the dtype :func:`pair_bias_rows`
    allocates), so the engine word is value-for-value :func:`pair_bias_rows`. ``rows``: the block (``DiffusionSchedule.bias_rows``; None = one block)."""
    require_sharded(layout, "pair_bias_rows_into")
    R, N = _check_shard(z_shard, layout, "pair_bias_rows_into")
    lead = tuple(int(s) for s in z_shard.shape[:-3])
    is_spec = isinstance(spec, DitBias)
    if not is_spec and not callable(spec):
        raise RowpairRefused(f"pair_bias_rows_into: spec must be a DitBias or a callable into_fn(z_rows, out_slice), got {type(spec).__name__}")
    if out is None:
        if heads is None or dtype is None:
            probe = probe_fn if probe_fn is not None else (spec.engine if is_spec else None)
            if probe is None:
                raise RowpairRefused("pair_bias_rows_into: heads= and dtype= (or probe_fn= / a DitBias engine statement to probe on one pair element) "
                                     "are needed to allocate [*, H, R, N]")
            y = probe(z_shard[..., :1, :1, :])
            heads = int(y.shape[-1]) if heads is None else heads
            dtype = y.dtype if dtype is None else dtype
            del y
        out = torch.empty(lead + (int(heads), R, N), dtype=dtype, device=z_shard.device)
    elif tuple(int(s) for s in out.shape[:-3]) != lead or int(out.shape[-2]) != R or int(out.shape[-1]) != N:
        raise RowpairRefused(f"pair_bias_rows_into: out {tuple(out.shape)} vs [*={lead}, H, R={R}, N={N}]")
    w = dit_bias_word(word) if is_spec else "adapter"
    for b0, b1, g0, g1 in iter_row_blocks(layout, _all_rows(layout, rows)):
        z_rows, sl = z_shard[..., b0:b1, :, :], out[..., :, b0:b1, :]
        if not is_spec:
            spec(z_rows, sl)
            _count_bias("adapter")
        elif w in ("ln_proj", "ln_proj_fp32"):
            try:
                spec.ln_proj_into(z_rows, sl, dot_fp32=(w == "ln_proj_fp32"))
                _count_bias(w)
            except _IntoRefused as r:
                spec.engine_into(z_rows, sl)                                          # the engine statement for this block, by name
                _count_bias("engine:" + census_token(r.reason))
        else:
            spec.engine_into(z_rows, sl)
            _count_bias("engine")
        del z_rows, sl
    return out


class PairBiasCache(object):
    """The per-block pair biases of a roll-out (they depend on ``z_cond`` only): ``get(i, compute)`` returns the cached ``[*, H, R, N]`` of
    block ``i`` when ``enabled`` (computing it on first use), else ``compute()`` every call. One object per roll-out (``clear()`` between
    roll-outs if reused). ``nbytes`` = bytes held."""

    __slots__ = ("enabled", "store", "hits", "misses")

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self.store = {}
        self.hits = 0
        self.misses = 0

    def get(self, i: int, compute: Callable[[], object]):
        if not self.enabled:
            self.misses += 1
            return compute()
        if i not in self.store:
            self.misses += 1
            self.store[i] = compute()
        else:
            self.hits += 1
        return self.store[i]

    def clear(self) -> None:
        self.store.clear()

    @property
    def nbytes(self) -> int:
        return int(sum(int(t.numel()) * int(t.element_size()) for t in self.store.values()))


DiTBlockFns = namedtuple("DiTBlockFns", ["norm", "kv", "attn", "update", "bias", "bias_into"], defaults=(None,))
DiTBlockFns.__doc__ = """The engine callables of ONE diffusion-transformer block (an adapter builds one per block from its module):
    norm(a, s) -> x                         the block's pre-attention normalisation of the replicated activation (AdaLN(a, s) or LN(a)), all rows
    kv(x) -> kv                             everything the attention derives from ALL rows once per block (key / value projections, the mask
                                            bias) — an opaque object handed to ``attn``; ``lambda x: x`` for a monolithic attention
    attn(x_q, kv, bias_q, (g0, g1)) -> o    the attention output rows for GLOBAL query rows [g0, g1): query projection / scaling of ``x_q``
                                            ``[*, q, c]``, logits vs ``kv`` + ``bias_q [*, H, q, N]`` (+ the engine's mask bias), softmax over all
                                            N keys, PV, gating with ``x_q``, output projection -> ``[*, q, c]``
    update(a, o_rows, s, (r0, r1)) -> a'    the rest of the block on this rank's rows: ``a[..., r0:r1, :]`` (+ the AdaLN-zero output gate from
                                            ``s`` rows) + ``o_rows``, then the conditioned transition residual -> ``[*, R, c]``
    bias(z_rows) -> [*, rows, N, H]         the block's pair-bias projection of conditioned pair rows (``linear(LN(z_rows))``)
    bias_into                               OPTIONAL (default None: ``bias`` through :func:`pair_bias_rows`, today's path): a :class:`DitBias` (the
                                            same statement as weights — :func:`pair_bias_rows_into` picks the producer by ``ROWPAIR_DIFF_BIAS``)
                                            or an in-place ``into_fn(z_rows, out_slice [*, H, rows, N])``"""


def dit_block_sharded(fns: DiTBlockFns, a, s, bias, layout: Layout, *, q_rows: Optional[int] = None, gather: bool = True):
    """One transformer block with local query rows. ``a``: replicated ``[*, N, c]``; ``s``: the replicated conditioning (or None); ``bias``:
    this rank's ``[*, H, R, N]`` (from :func:`pair_bias_rows` / :class:`PairBiasCache`). Returns the replicated updated ``a`` (``gather=True``:
    one all_gather of the ``R`` rows) or this rank's rows ``[*, R, c]``."""
    require_sharded(layout, "dit_block_sharded")
    R, r0 = layout.R, layout.r0
    if int(a.shape[-2]) != layout.N:
        raise RowpairRefused(f"dit_block_sharded: a {tuple(a.shape)} vs layout.N={layout.N} (token rows on dim -2)")
    if bias.dim() < 3 or int(bias.shape[-2]) != R or int(bias.shape[-1]) != layout.N:
        raise RowpairRefused(f"dit_block_sharded: bias {tuple(bias.shape)} must be this rank's [*, H, R={R}, N={layout.N}]")
    x = fns.norm(a, s)                                                                # replicated: K/V of every rank come from all rows

    def attn_rows(x_q, x_all, bias_loc):                                              # apb_local_queries hands us (x[r0:r1], x, bias)
        kv = fns.kv(x_all)                                                            # once per block per rank
        out = None
        for b0, b1, g0, g1 in iter_row_blocks(layout, _all_rows(layout, q_rows)):                          # query sub-blocks bound the S x H x q x N logits
            o = fns.attn(x_q[..., b0:b1, :], kv, bias_loc[..., :, b0:b1, :], (g0, g1))
            if int(o.shape[-2]) != b1 - b0:
                raise RowpairRefused(f"dit_block_sharded: attn returned {tuple(o.shape)} for query rows [{g0}, {g1}) (rows on dim -2)")
            if out is None:
                out = o.new_empty(tuple(int(v) for v in o.shape[:-2]) + (R, int(o.shape[-1])))
            out[..., b0:b1, :].copy_(o)
            del o
        del kv
        return out

    o_rows = apb_local_queries(attn_rows, x, bias, layout, gather=False)             # [*, R, c]: queries = local rows, keys = all rows
    del x
    a_rows = fns.update(a, o_rows, s, (r0, layout.r1))
    del o_rows
    if int(a_rows.shape[-2]) != R:
        raise RowpairRefused(f"dit_block_sharded: update returned {tuple(a_rows.shape)}; this rank's R={R} rows belong on dim -2")
    if not gather:
        return a_rows
    return unshard_rows(a_rows.contiguous(), layout, dim=-2)                          # ONE collective per block: a is replicated again


def diffusion_transformer_sharded(blocks: Sequence[DiTBlockFns], a, s, z_cond_shard, layout: Layout, *,
                                  schedule: Optional[DiffusionSchedule] = None, q_rows: Optional[int] = None,
                                  bias_rows: Optional[int] = None, bias_cache: Optional[PairBiasCache] = None):
    """The diffusion transformer over a row-sharded conditioned pair: for block ``i``, ``bias_i = cache.get(i, pair_bias_rows(blocks[i].bias,
    z_cond_shard))`` then :func:`dit_block_sharded`. ``q_rows`` / ``bias_rows`` default to ``schedule``'s. Returns the replicated ``a``."""
    require_sharded(layout, "diffusion_transformer_sharded")
    _check_shard(z_cond_shard, layout, "diffusion_transformer_sharded")
    if schedule is not None:
        q_rows = schedule.q_rows if q_rows is None else q_rows
        bias_rows = schedule.bias_rows if bias_rows is None else bias_rows
    cache = bias_cache if bias_cache is not None else PairBiasCache(enabled=False)
    for i, fns in enumerate(blocks):
        into = getattr(fns, "bias_into", None)
        if into is None:
            bias = cache.get(i, lambda fns=fns: pair_bias_rows(fns.bias, z_cond_shard, layout, rows=bias_rows))
        else:                                                                          # the block's producer writes head-major into the [*, H, R, N] slices
            bias = cache.get(i, lambda fns=fns, into=into: pair_bias_rows_into(into, z_cond_shard, layout, rows=bias_rows, probe_fn=fns.bias))
        a = dit_block_sharded(fns, a, s, bias, layout, q_rows=q_rows, gather=True)
        del bias
    return a


# ----------------------------------------------------------------------------------------------------------------- the rows attention slot
DitKV = namedtuple("DitKV", ["k", "v", "key_mask", "stock"])
DitKV.__doc__ = """What an adapter's ``kv(x)`` returns when its ``attn`` is :func:`dit_attention_rows`: ``k`` / ``v`` ``[.., S, H, N, D]`` (leading
singleton dims allowed; the engine's projections of ALL rows, cast ONCE per block to the 16-bit compute dtype with :func:`cast16` when the roll-out
is fp32 and the core word is a tier word — the documented cast policy), ``key_mask`` ``[.., N]`` keep-mask (nonzero = attend; the same for every
sample, or ``[S, N]``; None = no mask) and ``stock`` = whatever the engine's own attention callable takes as its ``kv`` (today's object: the
fallback by name runs the engine statement unchanged)."""

_DIT_ROWS_CENSUS = {}                              # '<row>[:variant]' served | 'stock:<refusal kind>' -> attention calls


def dit_rows_census(reset: bool = False) -> str:
    """``dit_rows=<row>:<n>[,stock:<event>:<n>]`` — the query-block attention calls :func:`dit_attention_rows` served per row word this process, and
    the calls the engine statement served after a refusal BY NAME (``stock:<kind>``). Recorded through :func:`evidence.record_schedule` as it changes."""
    if reset:
        _DIT_ROWS_CENSUS.clear()
    return ",".join(f"{k}:{v}" for k, v in _DIT_ROWS_CENSUS.items()) or "none"


def _count_rows(word: str, n: int = 1) -> None:
    _DIT_ROWS_CENSUS[word] = _DIT_ROWS_CENSUS.get(word, 0) + int(n)
    record_schedule(dit_rows=dit_rows_census())


def cast16(t, dtype=None):
    """The rows slot's operand cast: an fp32 tensor -> ``dtype`` (default bfloat16) once; 16-bit tensors (and None) are returned as they are.
    The documented policy under a TIER word (``big`` | ``fast``): k / v once per block by the adapter's ``kv``, q once per query block inside
    :func:`dit_attention_rows`; fp32 statistics and accumulation stay inside the kernel (:mod:`opt_core.kernels.apb_attn`)."""
    if t is None or t.dtype != torch.float32:
        return t
    return t.to(dtype if dtype is not None else torch.bfloat16)


def _lead_strip(t, nd: int):
    """Leading singleton dims stripped down to ``nd`` dims (a view); :class:`RowpairRefused` when a non-singleton leading dim remains."""
    while t.dim() > nd and int(t.shape[0]) == 1:
        t = t[0]
    if t.dim() != nd:
        raise RowpairRefused(f"dit_attention_rows: operand {tuple(t.shape)} does not reduce to {nd} dims (only leading singleton dims are stripped)")
    return t


def dit_rows_core(core_word: Optional[str], *, dtype, heads: int, head_dim: int, samples: int = 1, kind: str = "dit", device=None,
                  record: bool = True):
    """The STATIC admission of the rows attention core an adapter is about to bind, once per roll-out (identical on every rank: a pure function
    of the word, the process's device class and the operand class): ``(attn_core, selection, reason)``. ``attn_core`` is ``'kernel'`` — pass it to
    :meth:`DiffusionSchedule.decide` so the query block is all local rows — when :func:`opt_core.kernels.apb.select_rows` lands the word on a flash
    row (``apb_attn`` | ``sba`` | ``sdpa``), else None (a refusal by name, or the materialised ``naive`` row: today's byte-model query blocks).
    ``core_word`` None / '' / 'engine' / 'off': nothing bound (``(None, None, 'engine')``). Recorded ``dit_rows_core=<describe>``."""
    w = (core_word or "").strip().lower()
    if w in ("", "engine", "off", "none", "stock"):
        if record:
            record_schedule(dit_rows_core="engine")
        return None, None, "engine"
    from ...kernels import apb as APB
    dt = str(dtype).replace("torch.", "").replace("bfloat16", "bf16").replace("float16", "fp16").replace("float32", "fp32")
    try:
        sel = APB.select_rows(None, dt, kind, None, None, word=w, samples=int(samples), heads=int(heads), head_dim=int(head_dim), device=device)
    except APB.Refusal as r:
        if record:
            record_schedule(dit_rows_core=census_token(f"refused:{r.kind}->engine"))
        return None, None, f"refused:{r.kind}"
    if record:
        record_schedule(dit_rows_core=rows_core_token(sel))
    core = "kernel" if sel.row in ("apb_attn", "sba") or (sel.row == "sdpa" and (sel.variant or "auto") != "math") else None
    return core, sel, sel.reason


def rows_core_token(sel) -> str:
    """The census VALUE for a decided rows selection as ONE token (``dit_rows_core=``): ``<row>[:<variant>]@<word>:cell=<key|->[.nearest]:<reason>``
    — never :func:`opt_core.kernels.apb.describe` (a multi-word LEVER body). E.g. ``apb_attn@big:cell=-:rows:beyond_measured(no_rows_cell:
    ditrows_h16d48:S5:cc9.0:fp32)->apb_attn:by_name,cast:bf16``."""
    arm = str(sel.row) + ((":" + str(sel.variant)) if getattr(sel, "variant", None) else "")
    cell = str(sel.cell) if getattr(sel, "cell", None) else "-"
    if getattr(sel, "cell", None) and not getattr(sel, "size_measured", False):
        cell += ".nearest"
    return census_token("%s@%s:cell=%s:%s" % (arm, sel.word, cell, sel.reason or "-"))


def dit_attention_rows(*, q_fn: Callable[[object], object], out_fn: Callable[[object, object], object], stock_fn: Callable, num_heads: int,
                       core_word: str = "big", scale: Optional[float] = None, inf: float = 1e9, stock_q_rows: Optional[int] = None,
                       kind: str = "dit", selection=None, sticky: bool = True):
    """The ``attn`` callable of a :class:`DiTBlockFns` whose query-block attention runs on the ROWS face
    :func:`opt_core.kernels.apb.pair_bias_attention_rows` (rectangular: ``q`` = this rank's query rows, keys / values = all ``N`` rows, the pair bias
    the block's ``[H, q, N]`` rows read in place) — beside the engine statement, which stays the fallback BY NAME:

        q_fn(x_q) -> q [.., S, H, q, D]            the engine's query projection (+ its scaling: pass ``scale=1.0`` when q arrives pre-scaled, None = D**-0.5)
        out_fn(o [.., S, q, H*D], x_q) -> [.., q, c] the engine's epilogue (sigmoid gate from ``x_q``, output projection: e.g. ``mha._wrap_up``)
        stock_fn(x_q, kv.stock, bias_q, (g0, g1))   today's ``attn`` (the engine statement) — served, in chunks of ``stock_q_rows`` query rows (the
                                                   schedule's ``q_rows_stock``), when the face refuses the call by name (``stock:<kind>`` in
                                                   :func:`dit_rows_census`; ``sticky``: a refusal holds for the rest of this callable's calls)
        kv                                         a :class:`DitKV` from the adapter's ``kv(x)``

    ``core_word``: a row word (``apb_attn`` | ``sba[:tf32|ieee|tf32x3]`` | ``sdpa[:…]`` | ``naive``) or a tier word (``big`` | ``fast``: the measured
    rows cell's winner, else ``apb_attn`` BY NAME flagged ``beyond_measured`` — :func:`opt_core.kernels.apb.select_rows`); ``exact`` / ``faithful``
    are refused by name there (every flash row is tolerance-class vs the fp32 materialised statement). Numerics: the served row's class (``apb_attn``:
    16-bit operands — fp32 q is cast per query block to k's dtype, see :func:`cast16` — fp32 online-softmax statistics and fp32 accumulation, one
    output rounding; the output is handed to ``out_fn`` in ``q_fn``'s dtype). P-invariant: per-query arithmetic does not depend on the query count."""
    from ...kernels import apb as APB
    state = {"sel": selection, "off": None}
    H = int(num_heads)

    def _stock(x_q, kv, bias_q, rows, kind_word):
        g0, g1 = rows
        qn = int(x_q.shape[-2])
        step = int(stock_q_rows) if stock_q_rows and int(stock_q_rows) > 0 else qn
        stock_kv = kv.stock if isinstance(kv, DitKV) else kv
        outs = []
        for c0 in range(0, qn, step):
            c1 = min(qn, c0 + step)
            outs.append(stock_fn(x_q[..., c0:c1, :], stock_kv, bias_q[..., :, c0:c1, :], (g0 + c0, g0 + c1)))
            _count_rows("stock:" + census_token(kind_word))
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=-2)

    def attn(x_q, kv, bias_q, rows):
        if state["off"] is not None:
            return _stock(x_q, kv, bias_q, rows, state["off"])
        if not isinstance(kv, DitKV):
            raise RowpairRefused("dit_attention_rows: kv(x) must return a DitKV(k, v, key_mask, stock) under the rows slot")
        try:
            q = q_fn(x_q)                                                              # [.., S, H, q, D]
            q4, k4, v4 = _lead_strip(q, 4), _lead_strip(kv.k, 4), _lead_strip(kv.v, 4)
            qdt = q4.dtype
            if q4.dtype != k4.dtype:
                q4 = q4.to(k4.dtype)                                                   # q per query block to the compute dtype k / v carry (cast16 policy)
            b = bias_q
            while b.dim() > 3 and int(b.shape[0]) == 1:
                b = b[0]                                                               # [H, q, N] (shared) or [S, H, q, N]
            km = kv.key_mask
            if km is not None:
                while km.dim() > 2 and int(km.shape[0]) == 1:
                    km = km[0]
            o, sel = APB.pair_bias_attention_rows(q4, k4, v4, b, km, word=core_word, scale=scale, inf=inf, selection=state["sel"], kind=kind)
        except APB.Refusal as r:
            if sticky:
                state["off"] = r.kind
            return _stock(x_q, kv, bias_q, rows, r.kind)
        state["sel"] = sel
        _count_rows(census_token(sel.row + ((":" + sel.variant) if sel.variant else "")))
        S_, qn = int(q4.shape[0]), int(q4.shape[2])
        del q4
        o = o.reshape(tuple(int(d) for d in q.shape[:-4]) + (S_, qn, H * int(q.shape[-1])))   # [.., S, q, H*D]
        del q
        if o.dtype != qdt:
            o = o.to(qdt)
        return out_fn(o, x_q)

    attn.core_word = core_word
    attn.state = state
    return attn


# ----------------------------------------------------------------------------------------------------------------- seam: atom-attention band
class BandPlan(object):
    """The geometry of the atom-attention token-pair lookup: ``W`` band half-width (tokens), ``w_need`` the width the valid pairs need,
    ``extra_rows`` (sorted global token rows fetched whole), ``slot[N]`` (row -> extra index or -1), ``d_idx`` (band offset of every (q, k)
    slot, clamped), ``in_band`` (bool per slot), ``q_idx`` / ``k_idx`` as given (``[F, nb, nq]`` / ``[F, nb, nk]`` long), ``N``."""

    __slots__ = ("W", "w_need", "extra_rows", "slot", "d_idx", "in_band", "q_idx", "k_idx", "N")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw[k])

    def facts(self) -> dict:
        return {"diff_band_W": self.W, "diff_band_w_need": self.w_need, "diff_band_extra_rows": len(self.extra_rows)}


def band_plan(q_idx, k_idx, pair_valid, N: int, *, max_w: Optional[int] = None, max_extra_rows: Optional[int] = None,
              record: bool = True) -> BandPlan:
    """``q_idx [F, nb, nq]`` / ``k_idx [F, nb, nk]`` (long): the token row / column every (query slot, key slot) of the engine's atom blocks
    reads from the token-pair tensor; ``pair_valid [F, nb, nq, nk]`` (bool): the slots whose value is actually used (real query atom AND real
    key atom — padded slots read row 0 in the stock statement and are masked afterwards). ``W = min(max |k - q| over valid slots, max_w)``;
    ``max_w=None`` (default) = as wide as the valid slots need, so NO row travels whole; a cap ``max_w`` (the adapter's explicit argument — e.g.
    ``n_key - 1``, read from its ``ROWPAIR_DIFF_BAND_W``) sends the rows of valid slots outside the band whole (``extra_rows``: exact for any
    map, each an ``[N, C]`` broadcast); more than ``max_extra_rows`` (explicit; default ``BAND_EXTRA_MAX_DEFAULT`` = 256) is refused by name (an
    atom->token map that is not token-ordered would need O(N) full rows: not this mechanism). The core reads no environment here (a size that
    decides what is gathered whole is the adapter's explicit argument). Replicated inputs -> identical plan on every rank."""
    if q_idx.dim() != 3 or k_idx.dim() != 3 or tuple(pair_valid.shape) != tuple(q_idx.shape) + (int(k_idx.shape[-1]),):
        raise RowpairRefused(f"band_plan: q_idx {tuple(q_idx.shape)} k_idx {tuple(k_idx.shape)} pair_valid {tuple(pair_valid.shape)}: "
                             "expected [F, nb, nq], [F, nb, nk], [F, nb, nq, nk]")
    d = k_idx.long().unsqueeze(-2) - q_idx.long().unsqueeze(-1)                       # [F, nb, nq, nk]
    valid = pair_valid.bool()
    w_need = int(d.abs()[valid].max().item()) if bool(valid.any()) else 0
    W = w_need if max_w is None else min(w_need, max(0, int(max_w)))
    in_band = d.abs() <= W
    oob = (~in_band) & valid
    extra_rows = sorted(set(q_idx.long().unsqueeze(-1).expand_as(d)[oob].tolist())) if bool(oob.any()) else []
    emax = BAND_EXTRA_MAX_DEFAULT if max_extra_rows is None else int(max_extra_rows)
    if len(extra_rows) > int(emax):
        raise RowpairRefused(f"band_plan: {len(extra_rows)} full pair rows needed outside the |k-q|<={W} band (max_extra_rows={emax}): the "
                             f"atom->token map is not token-ordered enough for this W; pass a larger max_w / max_extra_rows ({ENV_BAND_W} / "
                             f"{ENV_BAND_EXTRA_MAX} are the adapters' names for them) or shard differently")
    slot = torch.full((int(N),), -1, dtype=torch.long, device=q_idx.device)
    for e, r in enumerate(extra_rows):
        slot[r] = e
    plan = BandPlan(W=int(W), w_need=w_need, extra_rows=extra_rows, slot=slot, d_idx=(d + W).clamp_(0, 2 * W), in_band=in_band,
                    q_idx=q_idx.long(), k_idx=k_idx.long(), N=int(N))
    if record:
        record_schedule(**plan.facts())
    return plan


def _band_block(x_rows, g0: int, W: int, N: int, tt):
    """The band slots of a block of FULL rows: ``out[..., i, t, :] = x_rows[..., i - g0, i + t - W, :]`` for global rows ``i`` in the block, 0
    where the column ``i + t - W`` falls outside ``[0, N)`` (never read). Pure indexing (bit copy)."""
    lead = tuple(int(s) for s in x_rows.shape[:-3])
    rb, C = int(x_rows.shape[-3]), int(x_rows.shape[-1])
    cols = torch.arange(g0, g0 + rb, device=x_rows.device)[:, None] + tt[None, :] - W        # [rb, 2W+1] absolute column of each slot
    valid = ((cols >= 0) & (cols < N)).view((1,) * len(lead) + (rb, 2 * W + 1, 1))
    idx = cols.clamp(0, N - 1).view((1,) * len(lead) + (rb, 2 * W + 1, 1)).expand(lead + (rb, 2 * W + 1, C))
    g = torch.gather(x_rows, -2, idx)
    return torch.where(valid, g, torch.zeros((), dtype=g.dtype, device=g.device))


def _bcast_extra_rows(mine: dict, extra_rows: Sequence[int], layout: Layout, like, N: int, C: int):
    """``[*, E, N, C]``: row ``extra_rows[e]`` broadcast WHOLE by its owner (``mine[i]`` holds this rank's owned rows) to every rank."""
    from .bcast import broadcast_tensordict
    lead = tuple(int(s) for s in like.shape[:-3])
    extras = like.new_empty(lead + (len(extra_rows), N, C))
    for e, i in enumerate(extra_rows):
        src = layout.owner(int(i))
        if src == layout.rank and mine.get(int(i)) is None:
            raise RowpairRefused(f"band extras: rank {layout.rank} owns row {i} but did not produce it")
        got = broadcast_tensordict(mine[int(i)] if src == layout.rank else None, src=src)      # the owner's row to every rank (bit copy)
        extras[..., e, :, :].copy_(got.to(like.device))
        del got
    return extras


def gather_band_rows(x_loc, layout: Layout, W: int, extra_rows: Optional[Sequence[int]] = None, *, rows: Optional[int] = None):
    """``(band, extras)`` from an already projected row shard ``x_loc [*, R, N, C]`` (small C — the engine's atom-pair projection of the
    conditioned pair): ``band [*, N, 2W+1, C]`` REPLICATED with ``band[..., i, t, :] = x[i, i + t - W, :]`` (0 outside ``[0, N)``), assembled
    from local rows (in blocks of ``rows``) and all_gathered — never an ``[N, N, C]`` gather; ``extras [*, E, N, C]`` = the full rows
    ``extra_rows`` (global indices, any owner) broadcast by their owners, None when ``extra_rows`` is empty. Collectives: one all_gather + E
    broadcasts. ``W`` and ``extra_rows`` must be identical on every rank (:func:`band_plan` derives both from replicated index tensors)."""
    require_sharded(layout, "gather_band_rows")
    R, N = _check_shard(x_loc, layout, "gather_band_rows")
    W = int(W)
    if W < 0:
        raise RowpairRefused(f"gather_band_rows: W={W} must be >= 0")
    lead = tuple(int(s) for s in x_loc.shape[:-3])
    C = int(x_loc.shape[-1])
    tt = torch.arange(2 * W + 1, device=x_loc.device)
    band_loc = x_loc.new_empty(lead + (R, 2 * W + 1, C))
    for b0, b1, g0, g1 in iter_row_blocks(layout, _all_rows(layout, rows)):
        band_loc[..., b0:b1, :, :] = _band_block(x_loc[..., b0:b1, :, :], g0, W, N, tt)
    band = unshard_rows(band_loc, layout, dim=-3)                                      # [*, N, 2W+1, C] on every rank
    del band_loc
    extra_rows = [int(i) for i in (extra_rows or [])]
    if not extra_rows:
        return band, None
    mine = {i: x_loc[..., i - layout.r0, :, :].contiguous() for i in extra_rows if layout.r0 <= i < layout.r1}
    return band, _bcast_extra_rows(mine, extra_rows, layout, band, N, C)


def pair_band_rows(proj_fn: Callable[[object], object], z_shard, layout: Layout, plan: "BandPlan", *, rows: Optional[int] = None):
    """The FUSED form of :func:`gather_band_rows`: ``zp = proj_fn(z_rows) -> [*, rows, N, C]`` (the engine's row-local atom-pair projection)
    is produced per block of ``rows`` local rows and folded into the band at once, so the projected shard ``[R, N, C]`` never exists either;
    ``plan`` (:func:`band_plan`) supplies ``W`` and ``extra_rows``. Returns ``(band [*, N, 2W+1, C], extras [*, E, N, C] | None)``."""
    require_sharded(layout, "pair_band_rows")
    R, N = _check_shard(z_shard, layout, "pair_band_rows")
    if plan.N != N:
        raise RowpairRefused(f"pair_band_rows: plan.N={plan.N} vs layout.N={N}")
    W = plan.W
    lead = tuple(int(s) for s in z_shard.shape[:-3])
    tt = torch.arange(2 * W + 1, device=z_shard.device)
    mine = {int(i): None for i in plan.extra_rows if layout.r0 <= int(i) < layout.r1}
    band_loc, C = None, None
    for b0, b1, g0, g1 in iter_row_blocks(layout, _all_rows(layout, rows)):
        zp = proj_fn(z_shard[..., b0:b1, :, :])                                       # [*, rb, N, C]
        if tuple(int(s) for s in zp.shape[:-1]) != lead + (b1 - b0, N):
            raise RowpairRefused(f"pair_band_rows: proj_fn returned {tuple(zp.shape)} for rows [{g0}, {g1})")
        C = int(zp.shape[-1])
        if band_loc is None:
            band_loc = zp.new_empty(lead + (R, 2 * W + 1, C))
        band_loc[..., b0:b1, :, :] = _band_block(zp, g0, W, N, tt)
        for i in mine:
            if g0 <= i < g1:
                mine[i] = zp[..., i - g0, :, :].contiguous()                           # the exact full row, kept for the broadcast
        del zp
    band = unshard_rows(band_loc, layout, dim=-3)                                      # [*, N, 2W+1, C] on every rank
    del band_loc
    if not plan.extra_rows:
        return band, None
    return band, _bcast_extra_rows(mine, plan.extra_rows, layout, band, N, C)


def band_lookup(band, extras, plan: BandPlan):
    """``zp[b, q_idx, k_idx]`` for every (block, query slot, key slot): ``[F, nb, nq, nk, C]`` read from ``band`` (in band) or ``extras``
    (out of band), with ``F`` = the flattened leading dims of ``band [*, N, 2W+1, C]``. Slots the plan marks invalid hold band values of
    unspecified pairs (the engine masks them, as its own gather of padded indices does)."""
    F_ = int(plan.q_idx.shape[0])
    band_f = band.reshape((F_,) + tuple(int(s) for s in band.shape[-3:]))
    bidx = torch.arange(F_, device=band.device, dtype=torch.long).view(-1, 1, 1, 1)
    out = band_f[bidx, plan.q_idx.unsqueeze(-1), plan.d_idx]                          # [F, nb, nq, nk, C]
    if extras is not None:
        ex_f = extras.reshape((F_,) + tuple(int(s) for s in extras.shape[-3:]))
        sl = plan.slot[plan.q_idx].clamp(min=0)                                        # [F, nb, nq]
        e = ex_f[bidx, sl.unsqueeze(-1), plan.k_idx.unsqueeze(-2)]                     # [F, nb, nq, nk, C]
        out = torch.where(plan.in_band.unsqueeze(-1), out, e)
        del e
    elif bool(((~plan.in_band) & (plan.slot[plan.q_idx].unsqueeze(-1) >= 0)).any()):
        raise RowpairRefused("band_lookup: the plan names extra rows but extras is None")
    return out


# ----------------------------------------------------------------------------------------------------------------- replicated draws
_SYNC_COUNTS = {"guard": 0, "bcast": 0, "off": 0}


def sync_replicated(t, name: str = "diffusion_noise", *, mode: Optional[str] = None):
    """Prove / enforce that a tensor every rank drew for itself (initial noise, per-step noise, random augmentation) is IDENTICAL on all
    ranks. ``mode`` (default ``ROWPAIR_DIFF_NOISE_SYNC``, else ``guard``): ``guard`` -> :func:`trunk.guard_replicated` (a mismatch is
    :class:`RowpairRefused` naming ``name``; returns ``t``); ``bcast`` -> rank 0's tensor replaces ``t`` on every rank
    (:func:`bcast.broadcast_tensordict`), then the guard; ``off`` -> ``t`` unchanged (identically seeded per-rank generators trusted). The
    mode and the per-mode call counts are recorded (``diff_noise_sync``, ``diff_noise_sync_calls``). Without a group: ``t``."""
    m = (mode or os.environ.get(ENV_NOISE_SYNC, "guard")).strip().lower() or "guard"
    if m not in NOISE_SYNC_MODES:
        raise RowpairRefused(f"sync_replicated({name}): mode {m!r} is not one of {NOISE_SYNC_MODES} ({ENV_NOISE_SYNC})")
    _SYNC_COUNTS[m] += 1
    record_schedule(diff_noise_sync=m, diff_noise_sync_calls=",".join(f"{k}:{v}" for k, v in _SYNC_COUNTS.items() if v))
    if not is_dist():
        return t
    if m == "off":
        return t
    if m == "bcast":
        from .bcast import broadcast_tensordict
        got = broadcast_tensordict(t.contiguous(), src=0)
        if got is not t:
            t = got.to(t.device) if got.device != t.device else got
    from .trunk import guard_replicated                   # the family's ONE replicated-tensor guard
    guard_replicated(t, name)
    return t
