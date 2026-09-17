# The finishing statements below reproduce the Protenix confidence head's single-device statements (Protenix: Copyright ByteDance and
# affiliates, Apache License 2.0, http://www.apache.org/licenses/LICENSE-2.0); this module is a derived work under those terms, and the
# licence text is carried beside it as confidence.PROTENIX_LICENSE.
"""The pair-conditioned confidence head under row sharding — engine-agnostic row-block reductions of AF3-style confidence summaries
(tensors in, tensors out; key names, dict layout and file writing are the engine adapter's).

The confidence pairformer runs on the primitives of :mod:`.trimul` / :mod:`.triatt` / :mod:`.transition` like the trunk. What is
specific to the head: the PAE / PDE logits ``[N, N, b]`` are produced per row and must never be materialised whole for large N, yet
every AF3-style summary statistic is a function of per-(i, j) expected values::

    E[i, j]    = sum_b softmax(logits[i, j, :])[b] * center[b]            (expected PAE / PDE)
    TM_d[i, j] = sum_b softmax(pae_logits[i, j, :])[b] * w_d[b],           w_d[b] = 1 / (1 + (center[b] / d0(N_d))^2)

where d0 is the TM-score normalisation and ``N_d`` the size of the token subset ("context") the statistic is evaluated on::

    FULL          N_d = N            -> ptm  (mean_j TM, max over rows with a frame); iptm (mean over j in a different chain, max over framed rows)
    CHAIN a       N_d = |a|          -> chain_ptm[a]
    PAIR (a, b)   N_d = |a| + |b|    -> chain_pair_iptm[a, b] (a < b; symmetric fill) -> chain_iptm[a] (mean over pairs touching a whose first
                                        chain has a frame) -> chain_pair_iptm_global (ligand-aware mix of chain_iptm)
    gpde family   sum_ij pde_ij * contact_ij / sum_ij contact_ij over (all | chain a x chain a | chain a x chain b)

:meth:`RowBlockReducer.consume` ``(c0, c1, pae_logits_rows, pde_logits_rows, contact_rows)`` is called by the owning rank for consecutive
row blocks ``[c0, c1)`` of its shard; only ``[c, N, b]`` tensors exist at any time. :meth:`RowBlockReducer.finalize` ``(bounds)`` (collective;
every rank calls it with the list of per-rank ``(q0, q1)`` row bounds — ``Layout.bounds``) returns the finished statistics on rank 0.

Two finishing regimes (both produce the same keys):

  ``finish="exact"``   keep the fp32 row pieces ``TM_d[i, cols(d)]`` (about 3N floats per row in total), gather each context's ``[N_d, N_d]``
                       matrix to rank 0 (one context at a time, freed immediately) and apply the single-device post-contraction statements
                       verbatim on tensors of the SAME shape the single-device code uses -> the reductions run in the same order -> bit-exact
                       identical results whenever the per-element steps (softmax over b, ``(p*w).sum(-1)``, ``p @ centers``) are shape-invariant
                       (the family's tests state where). Rank-0 transient: ``max_d N_d^2`` fp32 (<= ``4 N^2`` bytes) + the ``[N, N]`` fp32
                       expected-PDE and contact matrices for the gpde family.
  ``finish="rowsum"``  reduce every piece to per-row sums on the owning rank and gather only ``O(N * n_ctx)`` scalars + ``[C, C]`` tables;
                       finish with sum / count. Summation order differs from the single-device code -> |diff| ~ 1e-7 (a STATED tolerance
                       class, never asserted bit-exact), minimal traffic / memory.

int32 rule: callers choose the row block ``c`` such that ``c * N * b < 2**31`` (:func:`max_row_chunk`; asserted).

API:
    tm_bin_centers(min_bin, max_bin, no_bins) / tm_d0(N) / tm_bin_weight(N_d, centers_cpu, device)
    expected_value_rows(logits_rows, centers_dev)        ``(prob [1, c, N, b], prob @ centers [1, c, N])`` fp32, leading batch dim kept so the
                                                         statements are literally the single-device ones
    ChainIndex(asym_id, has_frame, token_is_ligand)      chain bookkeeping (ids remapped contiguous); ``.contexts(centers_cpu, device)``
    ContextReducer(contexts, r0, r1, bounds=, form=auto|block|sub)   the ONE row-piece / gather mechanism: ``.consume(i0, i1, probs)`` (LOCAL rows,
                                                         probs ``[*lead, w, N, bins]``) / ``.gather(k)`` / ``.finalize(finish)`` -> per-context
                                                         ``[*lead, n_D, n_D]`` on rank 0 for the engine's own finishing statements
    RowBlockReducer(chains, r0, r1, device, pae_bins=, pde_bins=, finish=, eps=, keep_value_rows=)   ``.consume(...)`` / ``.finalize(bounds)``
    gpde_from_full(token_pair_pde, contact_probs, chains, eps)   the single-device gpde statements on full matrices
    max_row_chunk(N, bins=64, cap=128)                   the largest row block on the 128-grid divisors honouring the int32 rule
    per_row_outputs_to_rank0(rows_shard, layout)         ``[R, ...] -> [N, ...]`` on rank 0 (plddt / per-token rows the writer needs whole)
    gather_single_rows(s_shard, layout)                  a row-sharded single-representation update back to replicated ``[N, c_s]``
"""
from __future__ import annotations

import functools
import math  # noqa: F401  (kept for engine finishing statements handed this module's namespace)
from dataclasses import dataclass, field  # noqa: F401
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import RowpairRefused
from ._torch import dist, torch
from .dist import Layout, all_gather_rows, gather_cat_to_rank0, gather_cat_to_rank0_host, gather_rows_to_rank0, is_dist, world

__all__ = ["FULL", "CHAIN", "PAIR", "FINISHES", "AF3_SUMMARIES", "tm_bin_centers", "tm_d0", "tm_bin_weight", "expected_value_rows", "Context", "ChainIndex",
           "ContextReducer", "RowBlockReducer", "gpde_from_full", "max_row_chunk", "per_row_outputs_to_rank0", "gather_single_rows", "gather_cat_to_rank0",
           "FastContextReducer", "BothContextReducer", "context_reducer_for_layout", "conf_reducer_word", "context_labels", "ptm_contexts_from_labels", "CONF_REDUCERS",
           "ENV_CONF_REDUCER", "ENV_CONF_GATHER_MB"]

FINISHES = ("exact", "rowsum")
# The AF3-family summary set (the OUTPUT CONTRACT of the OpenFold3 head; an engine with another summary
# definition uses the mechanism — contexts, row pieces, finishers — with its own statistic table, never this list bent to fit):
AF3_SUMMARIES = ("ptm", "iptm", "chain_ptm", "chain_pair_iptm", "chain_iptm", "chain_pair_iptm_global", "gpde", "chain_gpde", "chain_pair_gpde")


def _no_grad(fn):
    """``torch.no_grad()`` applied at CALL time (the module imports torch lazily)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return fn(*args, **kwargs)
    return wrapper


FULL, CHAIN, PAIR = "full", "chain", "pair"


# ----------------------------------------------------------------------------- TM helpers
def tm_bin_centers(min_bin: float, max_bin: float, no_bins: int) -> torch.Tensor:
    """Bin centres exactly as the AF3-style head computes them (CPU fp32 linspace + half width)."""
    bin_width = (max_bin - min_bin) / no_bins
    boundaries = torch.linspace(start=min_bin, end=max_bin - bin_width, steps=no_bins)
    return boundaries + 0.5 * bin_width


def tm_d0(N: int) -> float:
    """TM-score normalisation constant d0(N) = 1.24 (max(N,19) - 15)^(1/3) - 1.8."""
    return 1.24 * (max(N, 19) - 15) ** (1 / 3) - 1.8


def tm_bin_weight(N_d: int, centers_cpu: torch.Tensor, device) -> torch.Tensor:
    """w_d[b] = 1 / (1 + (center_b / d0(N_d))^2), computed on CPU fp32 then moved (engine order)."""
    return (1 / (1 + (centers_cpu / tm_d0(N_d)) ** 2)).to(device)


def expected_value_rows(logits_rows: torch.Tensor, centers_dev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """softmax over bins -> (prob [1,c,N,b], prob @ centers [1,c,N]); fp32; leading batch dim kept
    so the statements are literally the single-device ones."""
    prob = torch.nn.functional.softmax(logits_rows.unsqueeze(0), dim=-1)
    return prob, prob @ centers_dev


# ----------------------------------------------------------------------------- chains / contexts
@dataclass
class Context:
    kind: str
    a: int = -1
    b: int = -1
    mask: Optional[torch.Tensor] = None   # bool [N] (None for FULL)
    N_d: int = 0
    w: Optional[torch.Tensor] = None      # [b] on device


class ChainIndex:
    """Chain bookkeeping shared by all reductions (remap to contiguous ids like the engine does)."""

    def __init__(self, asym_id: torch.Tensor, has_frame: torch.Tensor,
                 token_is_ligand: Optional[torch.Tensor] = None):
        asym_raw = asym_id.long()
        uniq = torch.unique(asym_raw)
        if len(uniq) != int(asym_raw.max().item()) + 1:
            remap = {old.item(): new for new, old in enumerate(uniq)}
            asym = torch.tensor([remap[x.item()] for x in asym_raw], dtype=torch.long, device=asym_raw.device)
        else:
            asym = asym_raw
        self.N = int(asym.shape[0])
        self.asym_raw = asym_raw
        self.asym = asym
        self.C = int(len(torch.unique(asym)))
        self.masks = [asym == a for a in range(self.C)]
        self.sizes = [int(m.sum().item()) for m in self.masks]
        self.has_frame = has_frame.bool()
        self.chain_has_frame = [bool((self.masks[a] & self.has_frame).any().item()) for a in range(self.C)]
        if token_is_ligand is None:
            token_is_ligand = torch.zeros_like(asym, dtype=torch.bool)
        self.token_is_ligand = token_is_ligand.bool()
        self.chain_is_ligand = [
            bool((self.token_is_ligand[self.masks[a]].sum() >= self.masks[a].sum() // 2).item())
            for a in range(self.C)
        ]

    def contexts(self, centers_cpu: torch.Tensor, device) -> List[Context]:
        ctxs = [Context(FULL, N_d=self.N, w=tm_bin_weight(self.N, centers_cpu, device))]
        for a in range(self.C):
            ctxs.append(Context(CHAIN, a=a, mask=self.masks[a], N_d=self.sizes[a],
                                w=tm_bin_weight(self.sizes[a], centers_cpu, device)))
        for a in range(self.C):
            for b in range(a + 1, self.C):
                n = self.sizes[a] + self.sizes[b]
                ctxs.append(Context(PAIR, a=a, b=b, mask=self.masks[a] | self.masks[b], N_d=n,
                                    w=tm_bin_weight(n, centers_cpu, device)))
        return ctxs

    def contexts_lite(self, centers_cpu: torch.Tensor, device) -> List[Context]:
        """The SAME context list as :meth:`contexts` (kinds, ``a`` / ``b``, ``N_d``, weights of identical VALUES) without the ``C²`` device
        masks (``mask=None`` on every context: the structure is ``kind / a / b`` over ``self.asym``, the label vector a
        :class:`FastContextReducer` takes as ``labels=``) and with ``w`` cached per distinct ``N_d`` (``tm_bin_weight`` is a function of
        ``N_d`` only) — at most ``#distinct N_d`` host->device copies instead of ``1 + C + C(C-1)/2``. Contexts of equal size share one weight
        TENSOR, so a reducer's ``block`` form makes one pass per size (the ``sub`` form's per-context statement is unchanged by sharing)."""
        w_by_n: Dict[int, object] = {}

        def w(n: int):
            t = w_by_n.get(int(n))
            if t is None:
                t = w_by_n[int(n)] = tm_bin_weight(int(n), centers_cpu, device)
            return t
        ctxs = [Context(FULL, N_d=self.N, w=w(self.N))]
        for a in range(self.C):
            ctxs.append(Context(CHAIN, a=a, mask=None, N_d=self.sizes[a], w=w(self.sizes[a])))
        for a in range(self.C):
            for b in range(a + 1, self.C):
                ctxs.append(Context(PAIR, a=a, b=b, mask=None, N_d=self.sizes[a] + self.sizes[b], w=w(self.sizes[a] + self.sizes[b])))
        return ctxs


# ----------------------------------------------------------------------------- structured contexts from ONE label vector (no per-context mask / sync)
ENV_CONF_REDUCER = "ROWPAIR_CONF_REDUCER"            # classic (default) | fast | both — read by conf_reducer_word / context_reducer_for_layout
ENV_CONF_GATHER_MB = "ROWPAIR_CONF_GATHER_MB"        # FastContextReducer.finalize: rank-0 byte budget per batched gather (MiB, default 512)
CONF_REDUCERS = ("classic", "fast", "both")


def context_labels(asym_id, token_mask=None):
    """The ONE index source of :class:`FastContextReducer`: ``labels [N]`` int64 = ``asym_id`` where ``token_mask`` (None = every token) else
    ``-1`` (a token in no context). Same device as ``asym_id``; no host sync."""
    asym_l = asym_id.reshape(-1).long()
    if token_mask is None:
        return asym_l.clone()
    tm = token_mask.reshape(-1).to(device=asym_l.device).bool()
    return torch.where(tm, asym_l, torch.full_like(asym_l, -1))


def conf_reducer_word(reducer: Optional[str] = None) -> str:
    """The context-reducer choice in force: ``reducer`` when given, else ``ROWPAIR_CONF_REDUCER``, else ``classic``; one of
    ``CONF_REDUCERS`` (``classic`` | ``fast`` | ``both``) or :class:`RowpairRefused` naming the lever."""
    import os
    word = (reducer if reducer is not None else os.environ.get(ENV_CONF_REDUCER, "")).strip().lower() or "classic"
    if word not in CONF_REDUCERS:
        raise RowpairRefused(f"{ENV_CONF_REDUCER}={word!r}: one of {CONF_REDUCERS} is required")
    return word


def ptm_contexts_from_labels(asym_id, token_mask, make: Callable, *, want_full: bool = True, want_chain_ptm: bool = True, want_chain_pair: bool = True,
                             masks: Optional[bool] = None, reducer: Optional[str] = None):
    """The AF3-family TM context LIST an engine's ``compute_ptm`` / ``compute_chain_ptm`` / ``compute_chain_pair_iptm`` evaluate, built from
    ONE label vector without a per-context ``.item()``: ``make(key, n, chains, mask)`` is the ENGINE's context factory (its own ``bin_weight`` /
    ``num_tokens_considered`` statements fed the integer count ``n``, cached per ``n`` exactly as its masked form caches per ``int(mask.sum())``
    — so equal-size contexts share one weight tensor the same way) and receives, in this order::

        ("full", n_full, None, mask)                            n_full = #tokens with token_mask            (want_full)
        (("chain", key_a), n_a, (a,), mask)   for a ascending    key_a = the element of ``torch.unique(asym_id).tolist()`` (the engine dtype's
                                                                Python type, i.e. what ``aid.item()`` gives), a = int label  (want_chain_ptm)
        (("pair", i, j), n_i + n_j, (a_i, a_j), mask)  for i < j i, j index ``unique_chains``                 (want_chain_pair)

    where the counts are of MASKED-IN tokens per label from ONE ``bincount`` (asym ids must be non-negative integer-valued) and ``mask`` is the
    context's ``[N]`` bool device mask (the masked statements' ``token_mask & (asym == a)`` / pair unions — elementwise device ops, no sync)
    when ``masks`` is true, else ``None``. ``masks=None`` decides by the reducer in force (:func:`conf_reducer_word` of ``reducer`` /
    ``ROWPAIR_CONF_REDUCER``): the ``fast`` reducer needs none (no ``C²`` masks are built), ``classic`` / ``both`` need them. Returns
    ``(contexts, unique_chains, labels, n_by_chain)``: ``unique_chains`` = ``torch.unique(asym_id.long()).tolist()`` (ints, over ALL tokens like
    the masked statements' ``torch.unique(asym_l)``), ``labels`` = :func:`context_labels` (hand it to the reducer as ``labels=``), ``n_by_chain``
    ``{label: count}``. Host syncs: 3 (two ``tolist`` of C values, one of the counts), whatever the context count."""
    if masks is None:
        masks = conf_reducer_word(reducer) != "fast"
    asym_flat = asym_id.reshape(-1)
    asym_l = asym_flat.long()
    tm = torch.ones_like(asym_l, dtype=torch.bool) if token_mask is None else token_mask.reshape(-1).to(device=asym_l.device).bool()
    labels = torch.where(tm, asym_l, torch.full_like(asym_l, -1))
    chain_keys = torch.unique(asym_flat).tolist()                            # sorted; the engine dtype's Python type (float ids -> 1.0, like aid.item())
    unique_chains = [int(v) for v in torch.unique(asym_l).tolist()]          # sorted ints (the pair statements' torch.unique(asym_l).tolist())
    if len(chain_keys) != len(unique_chains):
        raise RowpairRefused(f"ptm_contexts_from_labels: asym_id has {len(chain_keys)} distinct values but {len(unique_chains)} distinct integer labels (non-integral chain ids)")
    if unique_chains and unique_chains[0] < 0:
        raise RowpairRefused(f"ptm_contexts_from_labels: negative chain id {unique_chains[0]} (labels must be non-negative; -1 marks a masked token)")
    cnt = torch.bincount(asym_l[tm], minlength=(unique_chains[-1] + 1) if unique_chains else 1).tolist()
    n_by_chain = {a: (int(cnt[a]) if a < len(cnt) else 0) for a in unique_chains}
    n_full = int(sum(n_by_chain.values()))
    chain_masks = [(asym_l == a) & tm for a in unique_chains] if masks else None   # C masks (not C²): the pair masks below are unions made on demand
    ctxs = []
    if want_full:
        ctxs.append(make("full", n_full, None, (tm.clone() if masks else None)))
    if want_chain_ptm:
        for ci, (key_a, a) in enumerate(zip(chain_keys, unique_chains)):
            ctxs.append(make(("chain", key_a), n_by_chain[a], (a,), (chain_masks[ci] if masks else None)))
    if want_chain_pair:
        C = len(unique_chains)
        for i in range(C):
            ni = n_by_chain[unique_chains[i]]
            for j in range(i + 1, C):
                ctxs.append(make(("pair", i, j), ni + n_by_chain[unique_chains[j]], (unique_chains[i], unique_chains[j]),
                                 ((chain_masks[i] | chain_masks[j]) if masks else None)))
    return ctxs, unique_chains, labels, n_by_chain



# ----------------------------------------------------------------------------- generic row-block context reducer (the ONE piece/gather mechanism)
class ContextReducer:
    """Row-block accumulator of the per-context matrices ``T_D[i, j] = sum_b probs[i, j, b] * w_D[b]`` for ``i, j`` in a token subset D
    ("context"), fed row block by row block by the rank owning the rows; :meth:`gather` assembles ONE context's ``[*lead, n_D, n_D]`` matrix
    on rank 0 (rows / columns in ascending token order = the dense ``T[..., mask, :][..., mask]``) so the engine's FINISHING statements (row
    means, has-frame masking, max, per-chain tables ...) run verbatim on the same-shape tensor its single-device code uses. Engine-free: a
    context is any object with ``.mask`` (bool ``[N]`` on any device; ``None`` = all N tokens) and ``.w`` (the ``[bins]`` weight tensor on
    the compute device, from the ENGINE's own d0 / bin-centre statements); keys and kinds are the caller's (:class:`Context` carries them for
    the AF3 family). This is the one piece / gather implementation of the package: :class:`RowBlockReducer` (the AF3-family summaries) and
    every engine adapter's reducer are built on it.

    Statement ``form`` (per element the same products and the same 64-term sum; they differ in which tensor the sum runs on):
      ``"sub"``    ``(probs[..., rows_D, :, :][..., cols_D, :] * w_D).sum(-1)`` — the sub-block first (the alternative statement order of the same sum)
      ``"block"``  ``(probs * w_D).sum(-1)`` ONCE per distinct weight over the whole row block, then rows / columns selected (the OpenFold3
                   statement order; contexts of equal ``N_d`` share one pass)
      ``"auto"``   ``block`` when the contexts sharing a weight select at least half the block's work, else ``sub``
    Weight sharing (``block`` form): contexts carrying the SAME weight tensor object (an engine caches ``w`` per ``N_d``), or the same explicit
    ``ctx.group``, share one multiply-reduce pass over the block.

    API::

        ContextReducer(contexts, r0, r1, *, bounds=None, form="auto", allow_unsharded=False)   rows ``[r0, r1)`` are this rank's; ``bounds`` =
                                             the per-rank ``(q0, q1)`` list (``Layout.bounds``) — or given later to :meth:`gather` / :meth:`finalize`
        ContextReducer.for_layout(contexts, layout, **kw)
        .consume(i0, i1, probs)              LOCAL rows ``[i0, i1)`` (ascending, consecutive, covering ``[0, R)`` before any gather); ``probs``
                                             ``[*lead, i1-i0, N, bins]`` (softmax already applied by the engine's statement)
        .pieces(i0, i1, probs)               the statement alone (no bookkeeping): yields ``(k, rows_local, T_piece [*lead, r', n_D])``
        .gather(k, bounds=None)              COLLECTIVE — context k's ``[*lead, n_D, n_D]`` on rank 0, None elsewhere; frees k's pieces everywhere
        .finalize(finish=None, bounds=None)  COLLECTIVE — ``{k: finish(k, ctx, T) if finish else T}`` on rank 0 (one matrix alive at a time), None elsewhere
        .counts(k, bounds=None)              context k's rows per rank (CPU ints); ``.n(k)``; ``.local_rows(k)``
    Memory: pieces ``~ sum_D n_D`` floats per local row (about 3N per row for the AF3 context set); rank-0 transient ``max_D n_D^2 * lead``.
    """

    FORMS = ("auto", "block", "sub")

    def __init__(self, contexts: Sequence[object], r0: int, r1: int, *, bounds: Optional[Sequence[Tuple[int, int]]] = None,
                 form: str = "auto", allow_unsharded: bool = False):
        if form not in self.FORMS:
            raise RowpairRefused(f"ContextReducer: form={form!r} not in {self.FORMS}")
        if not allow_unsharded and not is_dist():
            raise RowpairRefused("ContextReducer: refused at n_gpu=1 (no group): the adapter installs nothing at n_gpu=1; a kit that ships the "
                                 "reducer as its own single-device memory lever says so (allow_unsharded=True) and names that lever in its line")
        self.ctxs = list(contexts)
        self.r0, self.r1, self.R = int(r0), int(r1), int(r1) - int(r0)
        self.bounds = [tuple(map(int, b)) for b in bounds] if bounds is not None else None
        self.form = form
        self._next = 0
        self._lead: Optional[Tuple[int, ...]] = None
        self._meta = []                      # per context: dict(idx_cpu, n, loc (ascending local rows, list), group)
        self._idx_dev: Dict[Tuple[int, str], object] = {}
        self._pieces: Dict[int, List[object]] = {k: [] for k in range(len(self.ctxs))}
        groups: Dict[object, int] = {}
        for k, ctx in enumerate(self.ctxs):
            mask = getattr(ctx, "mask", None)
            if mask is None:
                N = self._N_of(ctx)
                idx_cpu = torch.arange(N, dtype=torch.long)
            else:
                idx_cpu = mask.detach().to("cpu").bool().nonzero(as_tuple=False).flatten().to(torch.long)
            loc = idx_cpu[(idx_cpu >= self.r0) & (idx_cpu < self.r1)] - self.r0
            gkey = getattr(ctx, "group", None)
            gkey = ("group", gkey) if gkey is not None else ("w", id(ctx.w))     # contexts sharing one weight TENSOR share one "block" pass
            groups.setdefault(gkey, len(groups))
            self._meta.append(dict(idx_cpu=idx_cpu, n=int(idx_cpu.numel()), loc=[int(x) for x in loc.tolist()], group=groups[gkey]))

    @classmethod
    def for_layout(cls, contexts: Sequence[object], layout: Layout, **kw) -> "ContextReducer":
        return cls(contexts, layout.r0, layout.r1, bounds=layout.bounds, **kw)

    def _N_of(self, ctx) -> int:
        N = getattr(ctx, "N_d", None) or getattr(ctx, "N", None)
        if not N:
            raise RowpairRefused("ContextReducer: a context with mask=None must carry N_d (the token count)")
        return int(N)

    # -- bookkeeping views
    def n(self, k: int) -> int:
        return self._meta[k]["n"]

    def local_rows(self, k: int) -> List[int]:
        return self._meta[k]["loc"]

    def counts(self, k: int, bounds: Optional[Sequence[Tuple[int, int]]] = None) -> List[int]:
        b = self._bounds(bounds)
        idx = self._meta[k]["idx_cpu"]
        return [int(((idx >= int(q0)) & (idx < int(q1))).sum()) for (q0, q1) in b]

    def _bounds(self, bounds):
        b = bounds if bounds is not None else self.bounds
        if b is None:
            if not is_dist():
                return [(self.r0, self.r1)]
            raise RowpairRefused("ContextReducer: per-rank row bounds are required for a gather (bounds= at construction or at the call)")
        return b

    def _idx_on(self, k: int, device):
        key = (k, str(device))
        t = self._idx_dev.get(key)
        if t is None:
            t = self._meta[k]["idx_cpu"].to(device)
            self._idx_dev[key] = t
        return t

    @staticmethod
    def _rows_t(rows: List[int], i0: int, device):
        return torch.tensor([x - i0 for x in rows], dtype=torch.long, device=device)

    # -- the statement
    @_no_grad
    def pieces(self, i0: int, i1: int, probs):
        """Yields ``(k, rows_local, T_piece)`` for every context with rows in LOCAL block ``[i0, i1)``: ``rows_local`` the context's local row
        indices inside the block (ascending python ints), ``T_piece`` ``[*lead, len(rows_local), n_D]``. No state is touched."""
        import bisect
        i0, i1 = int(i0), int(i1)
        w_rows = i1 - i0
        if int(probs.shape[-3]) != w_rows:
            raise RowpairRefused(f"ContextReducer: probs has {int(probs.shape[-3])} rows for block [{i0},{i1})")
        N = int(probs.shape[-2])
        dev = probs.device
        by_group: Dict[int, List[Tuple[int, List[int]]]] = {}
        for k, m in enumerate(self._meta):
            loc = m["loc"]
            lo, hi = bisect.bisect_left(loc, i0), bisect.bisect_left(loc, i1)
            if hi <= lo:
                continue
            by_group.setdefault(m["group"], []).append((k, loc[lo:hi]))
        for g, items in by_group.items():
            w = self.ctxs[items[0][0]].w
            form = self.form
            if form == "auto":
                work = sum(len(rows) * self._meta[k]["n"] for k, rows in items)
                form = "block" if 2 * work >= w_rows * N else "sub"
            if form == "block":
                T = (probs * w).sum(dim=-1)                                                   # [*lead, w, N]  (the engine statement, once per weight)
                for k, rows in items:
                    piece = T if len(rows) == w_rows else T.index_select(-2, self._rows_t(rows, i0, dev))
                    if self._meta[k]["n"] != N:
                        piece = piece.index_select(-1, self._idx_on(k, dev))
                    yield k, rows, piece
                del T
            else:
                for k, rows in items:
                    sub = probs if len(rows) == w_rows else probs.index_select(-3, self._rows_t(rows, i0, dev))
                    if self._meta[k]["n"] != N:
                        sub = sub.index_select(-2, self._idx_on(k, dev))                        # [*lead, r', n_D, bins]
                    yield k, rows, (sub * w).sum(dim=-1)                                        # (the engine statement on the sub-block)
                    del sub

    @_no_grad
    def consume(self, i0: int, i1: int, probs) -> None:
        i0, i1 = int(i0), int(i1)
        if not (i0 == self._next and 0 <= i0 < i1 <= self.R):
            raise RowpairRefused(f"ContextReducer.consume: rows [{i0},{i1}) out of order (next expected {self._next}, R={self.R})")
        lead = tuple(int(s) for s in probs.shape[:-3])
        if self._lead is None:
            self._lead = lead
        elif self._lead != lead:
            raise RowpairRefused(f"ContextReducer.consume: leading dims {lead} differ from the first block's {self._lead}")
        self._dtype, self._device = probs.dtype, probs.device
        for k, rows, T in self.pieces(i0, i1, probs):
            self._pieces[k].append(T)
        self._next = i1

    # -- collectives
    @_no_grad
    def gather(self, k: int, bounds: Optional[Sequence[Tuple[int, int]]] = None):
        """COLLECTIVE. Context k's ``[*lead, n_D, n_D]`` matrix on rank 0 (rows in rank order = ascending token order), None on other ranks.
        Every rank calls it for the same k in the same order; k's pieces are freed on every rank."""
        if self._next != self.R:
            raise RowpairRefused(f"ContextReducer.gather: local rows [{self._next},{self.R}) were never consumed")
        counts = self.counts(k, bounds)
        n = self._meta[k]["n"]
        if sum(counts) != n:
            raise RowpairRefused(f"ContextReducer.gather: per-rank bounds cover {sum(counts)} of context {k}'s {n} rows (bounds must partition [0, N) once)")
        lst = self._pieces[k]
        self._pieces[k] = []
        if lst:
            mine = torch.cat(lst, dim=-2)
        else:
            lead = self._lead if self._lead is not None else ()
            dt = getattr(self, "_dtype", None) or torch.float32
            dev = getattr(self, "_device", None) or self.ctxs[k].w.device
            mine = torch.zeros(tuple(lead) + (0, n), dtype=dt, device=dev)
        del lst
        piece = mine.movedim(-2, 0).contiguous()                                             # [n_q, *lead, n_D]
        del mine
        full = gather_cat_to_rank0(piece, counts)
        del piece
        if full is None:
            return None
        return full.movedim(0, -2)                                                          # [*lead, n_D, n_D]

    @_no_grad
    def finalize(self, finish: Optional[Callable] = None, bounds: Optional[Sequence[Tuple[int, int]]] = None):
        """COLLECTIVE. ``{k: finish(k, ctx, T) if finish else T}`` on rank 0 over all contexts in order (one gathered matrix alive at a time
        when ``finish`` reduces it); None on other ranks."""
        P, rank_ = world() if is_dist() else (1, 0)
        out = {} if rank_ == 0 else None
        for k, ctx in enumerate(self.ctxs):
            T = self.gather(k, bounds)
            if rank_ == 0:
                out[k] = finish(k, ctx, T) if finish is not None else T
            del T
        return out

    def index(self, k: int, device=None):
        """Context k's ascending token index (long) on ``device`` (default cpu) — the ``mask.nonzero()`` this reducer took at construction
        (identity -> ``arange(N)``): what an engine's finishing statements select with instead of the boolean mask (same elements, same order,
        no sync). The one method every reducer of :func:`context_reducer_for_layout` offers, so a kit's finish has one form under every lever."""
        return self._idx_on(k, device if device is not None else "cpu")


# ----------------------------------------------------------------------------- structured row-block context reducer (no per-context mask / nonzero / sync)
_FULL_LABELS = None                                  # the label tuple of a FULL context (every labelled token)


class FastContextReducer:
    """API-compatible with :class:`ContextReducer` (same constructor arguments, ``for_layout``, ``n``, ``local_rows``, ``counts``, ``pieces``,
    ``consume``, ``gather``, ``finalize``) for STRUCTURED contexts: every context is a union of whole "chains" (labels) of ONE label vector
    ``labels [N]`` (int; ``-1`` = a token in no context — the engine's token mask). A context's label set is read from, in order: ``ctx.chains``
    (tuple of label values; ``None`` = FULL), the core :class:`Context` ``kind / a / b`` (FULL | CHAIN a | PAIR a, b over ``labels`` =
    ``ChainIndex.asym``; any other kind without a mask = FULL), the AF3 kits' ``ctx.key`` (``"full"`` | ``("chain", label)``; ``("pair", i, j)``
    keys index ``unique_chains``, so pair contexts must carry ``chains``), else ``mask is None`` -> FULL. A context with a mask but no structure
    is REFUSED BY NAME (:class:`ContextReducer` serves arbitrary subsets, e.g. an engine-defined interface set). ``labels=None``: every context
    must be FULL (``N`` from ``N_d`` / the mask length) — e.g. RF3's single all-token context.

    What changes against :class:`ContextReducer` — the VALUES do not (asserted bitwise by the family's tests, both forms, P in {2, 3}):
      * the index bookkeeping is ``O(N + n_ctx)`` host integer work from one ``labels.tolist()`` (positions per label, per-rank counts by
        bisection, the local rows cut into label RUNS) — no per-context ``[N]`` mask, ``nonzero``, device->host copy or OpenMP region (the term
        that cost ~90 ms per context under 8 ranks x 64 OpenMP threads);
      * form ``block``: per row block the engine statement ``T = (probs * w).sum(-1)`` once per weight group (verbatim), then per label run ONE
        packed ``T_rows.index_select(-1, cols[label, group])`` gathering the columns of EVERY member context (full, chain a, pairs (a, .)) — the
        per-context pieces are column slices of the packed piece (pure copies of the same elements: bitwise by construction); ~``G x (1 + runs)``
        launches per block instead of ~``2C``, no Python scan over ``n_ctx``, no per-block row-list copies;
      * form ``sub``: the classic per-context statement ``(probs[..., rows_D, :, :][..., cols_D, :] * w_D).sum(-1)`` on tensors of the SAME shape,
        values and dtype (rows by ``narrow`` / ``cat`` of the runs, columns by a per-context device index concatenated ON THE DEVICE from per-label
        indices — ``C`` host->device copies, not ``n_ctx``): bitwise by construction on every device (the CUDA reduce configuration depends on the
        output count, so no packing across contexts is done in this form);
      * ``finalize`` moves the contexts to rank 0 in BATCHES of ``gather_bytes`` (one :func:`gather_cat_to_rank0` per batch instead of one per
        context; a context larger than the budget — FULL — travels alone as today) and rebuilds each ``[*lead, n_D, n_D]`` with the sizes AND
        strides :meth:`ContextReducer.gather` produces, so an engine's ``finish(k, ctx, T)`` sees an indistinguishable tensor;
      * ``cpu_threads``: the torch intra-op thread count while building / finishing (:func:`dist.cpu_threads`; the per-rank cap proper is the
        launcher's ``ROWPAIR_RANK_THREADS``).
    Extra API: ``index(k, device)`` (context k's ascending token index — what ``mask.nonzero()`` gave finishing statements, for sync-free
    ``index_select`` closes), ``labels_of(k)``, ``stats`` (``index_s consume_s finalize_s gathers packed_selects group_passes``).
    Memory: as :class:`ContextReducer` (the packed pieces hold the same columns; packed column indices live per (label, group) this rank's rows
    touch and are dropped when their last member context is gathered)."""

    FORMS = ("auto", "block", "sub")

    def __init__(self, contexts: Sequence[object], r0: int, r1: int, *, bounds: Optional[Sequence[Tuple[int, int]]] = None,
                 form: str = "auto", allow_unsharded: bool = False, labels=None, gather_bytes: int = 512 << 20, cpu_threads=None):
        import time
        if form not in self.FORMS:
            raise RowpairRefused(f"FastContextReducer: form={form!r} not in {self.FORMS}")
        if not allow_unsharded and not is_dist():
            raise RowpairRefused("FastContextReducer: refused at n_gpu=1 (no group): the adapter installs nothing at n_gpu=1; a kit that ships the "
                                 "reducer as its own single-device memory lever says so (allow_unsharded=True) and names that lever in its line")
        self.ctxs = list(contexts)
        self.r0, self.r1, self.R = int(r0), int(r1), int(r1) - int(r0)
        self.bounds = [tuple(map(int, b)) for b in bounds] if bounds is not None else None
        self.form, self.gather_bytes, self.cpu_threads = form, int(gather_bytes), cpu_threads
        self._next, self._lead, self._dtype, self._device = 0, None, None, None
        self.stats = {"index_s": 0.0, "consume_s": 0.0, "finalize_s": 0.0, "gathers": 0, "packed_selects": 0, "group_passes": 0, "context_selects": 0}
        from .dist import cpu_threads as _cpu_threads
        t0 = time.perf_counter()
        with _cpu_threads(cpu_threads):
            self._build(labels)
        self.stats["index_s"] = time.perf_counter() - t0

    @classmethod
    def for_layout(cls, contexts: Sequence[object], layout: Layout, **kw) -> "FastContextReducer":
        return cls(contexts, layout.r0, layout.r1, bounds=layout.bounds, **kw)

    # ------------------------------------------------------------------ structure (host integers only)
    @staticmethod
    def context_labels_of(ctx):
        """The label tuple of ``ctx`` (sorted ints) or ``None`` = FULL; :class:`RowpairRefused` for a masked context without structure."""
        ch = getattr(ctx, "chains", None)
        if ch is not None:
            return tuple(sorted(set(int(v) for v in ch)))
        kind = getattr(ctx, "kind", None)
        if kind is not None:
            if kind == CHAIN:
                return (int(ctx.a),)
            if kind == PAIR:
                return tuple(sorted((int(ctx.a), int(ctx.b))))
            if kind == FULL or getattr(ctx, "mask", None) is None:
                return _FULL_LABELS                                          # FULL and engine kinds over all tokens (e.g. RF3's one context)
            raise RowpairRefused(f"FastContextReducer: context kind={kind!r} carries a mask but no chain structure; the classic ContextReducer serves arbitrary subsets")
        key = getattr(ctx, "key", None)
        if (isinstance(key, str) and key == FULL) or getattr(ctx, "mask", None) is None:
            return _FULL_LABELS
        if isinstance(key, tuple) and len(key) >= 2 and key[0] == CHAIN:
            return (int(key[1]),)
        raise RowpairRefused(f"FastContextReducer: context {key!r} carries a mask but no chain structure (`chains=`): pair keys index unique_chains, "
                             "not labels — build the contexts with confidence.ptm_contexts_from_labels, or use the classic ContextReducer (ROWPAIR_CONF_REDUCER=classic)")

    def _build(self, labels) -> None:
        ctxs = self.ctxs
        # -- the ONE label vector (one device->host copy), positions per label (ascending), the labelled token count
        if labels is not None:
            if hasattr(labels, "detach"):
                lab_list = [int(v) for v in labels.detach().reshape(-1).to("cpu").tolist()]
            else:
                lab_list = [int(v) for v in labels]
            self.N = len(lab_list)
        else:
            N = None
            for ctx in ctxs:
                m = getattr(ctx, "mask", None)
                N = int(m.numel()) if m is not None else (getattr(ctx, "N_d", None) or getattr(ctx, "N", None))
                if N:
                    break
            if not N:
                raise RowpairRefused("FastContextReducer: labels=None and no context tells N (mask / N_d)")
            self.N = int(N)
            lab_list = None
        pos: Dict[int, List[int]] = {}
        if lab_list is None:
            pos[0] = list(range(self.N))
            self._all_valid = True
        else:
            for i, v in enumerate(lab_list):
                if v >= 0:
                    lst = pos.get(v)
                    if lst is None:
                        pos[v] = [i]
                    else:
                        lst.append(i)
            self._all_valid = all(v >= 0 for v in lab_list)
        self._pos = pos
        self._nlab = {v: len(p) for v, p in pos.items()}
        self._n_valid = sum(self._nlab.values())
        # -- contexts: label tuple, size, weight group (the SAME grouping rule as ContextReducer: explicit .group else the weight TENSOR's identity)
        groups: Dict[object, int] = {}
        self._labs, self._n, self._g = [], [], []
        by_label: Dict[int, List[int]] = {}
        fulls: List[int] = []
        for k, ctx in enumerate(ctxs):
            labs = self.context_labels_of(ctx)
            if labs is not _FULL_LABELS and lab_list is None:
                raise RowpairRefused(f"FastContextReducer: context {k} names chains {labs} but labels= was not given")
            if labs is _FULL_LABELS:
                n = self._n_valid
                fulls.append(k)
            else:
                n = sum(self._nlab.get(v, 0) for v in labs)
                for v in labs:
                    by_label.setdefault(v, []).append(k)
            gkey = getattr(ctx, "group", None)
            gkey = ("group", gkey) if gkey is not None else ("w", id(ctx.w))
            groups.setdefault(gkey, len(groups))
            self._labs.append(labs)
            self._n.append(int(n))
            self._g.append(groups[gkey])
        self._fulls, self._by_label = fulls, by_label
        self._w_of_group: Dict[int, object] = {}
        for k in range(len(ctxs)):
            self._w_of_group.setdefault(self._g[k], ctxs[k].w)
        # -- this rank's rows cut into label runs (label constant, ascending), unlabelled rows skipped
        if lab_list is None:
            self._runs = [(0, self.R, 0)] if self.R > 0 else []
        else:
            loc = lab_list[self.r0:self.r1]
            runs, i = [], 0
            while i < len(loc):
                j, v = i + 1, loc[i]
                while j < len(loc) and loc[j] == v:
                    j += 1
                if v >= 0:
                    runs.append((i, j, int(v)))
                i = j
            self._runs = runs
        self._run_starts = [r[0] for r in self._runs]
        self._labels_here = sorted(set(v for (_lo, _hi, v) in self._runs))     # labels with rows on THIS rank (a FULL context's packs live under them)
        # -- lazily built caches: pack members / offsets per (label, group); packed and per-context device indices; per-bounds counts
        self._members: Dict[Tuple[int, int], List[int]] = {}
        self._off: Dict[Tuple[int, int], Dict[int, int]] = {}
        self._L: Dict[Tuple[int, int], int] = {}
        self._refs: Dict[Tuple[int, int], int] = {}
        self._cols_dev: Dict[Tuple[int, int, str], object] = {}
        self._lab_dev: Dict[Tuple[int, str], object] = {}
        self._kidx_dev: Dict[Tuple[int, str], object] = {}
        self._idx_cpu: Dict[int, Optional[List[int]]] = {}
        self._counts_cache: Dict[tuple, Dict[int, List[int]]] = {}
        self._groups_of_label: Dict[int, List[int]] = {}
        self._devs = set()
        # -- storage: packed pieces per (label, group) with their row runs (block form); per-context pieces with their first local row (sub form)
        self._store: Dict[Tuple[int, int], List[Tuple[int, int, object]]] = {}
        self._kstore: Dict[int, List[Tuple[int, object]]] = {}

    def labels_of(self, k: int):
        return self._labs[k]

    def n(self, k: int) -> int:
        return self._n[k]

    def _groups_for(self, v: int) -> List[int]:
        gs = self._groups_of_label.get(v)
        if gs is None:
            seen, gs = set(), []
            for k in sorted(self._fulls + self._by_label.get(v, [])):
                g = self._g[k]
                if g not in seen:
                    seen.add(g)
                    gs.append(g)
            self._groups_of_label[v] = gs
        return gs

    def _members_of(self, v: int, g: int) -> List[int]:
        key = (v, g)
        m = self._members.get(key)
        if m is None:
            m = sorted([k for k in self._fulls if self._g[k] == g] + [k for k in self._by_label.get(v, []) if self._g[k] == g])
            self._members[key] = m
            off, o = {}, 0
            for k in m:
                off[k] = o
                o += self._n[k]
            self._off[key], self._L[key] = off, o
            self._refs[key] = len(m)
        return m

    def _idx_list(self, k: int) -> Optional[List[int]]:
        """Context k's ascending global token index as a host list; ``None`` = the identity over ``[0, N)``."""
        if k in self._idx_cpu:
            return self._idx_cpu[k]
        labs = self._labs[k]
        if labs is _FULL_LABELS:
            lst = None if self._all_valid else sorted(x for p in self._pos.values() for x in p)
        elif len(labs) == 1:
            lst = self._pos.get(labs[0], [])
        else:
            ps = sorted((p for p in (self._pos.get(v, []) for v in labs) if p), key=lambda p: p[0])
            if all(ps[i][-1] < ps[i + 1][0] for i in range(len(ps) - 1)):
                lst = [x for p in ps for x in p]                                 # disjoint ascending ranges: the concatenation is sorted
            else:
                lst = sorted(x for p in ps for x in p)
        self._idx_cpu[k] = lst
        return lst

    def index(self, k: int, device=None):
        """Context k's ascending token index (long, on ``device``) — the ``mask.nonzero()`` of the masked grammar, without the mask or a sync:
        concatenated ON THE DEVICE from the per-label indices (cached per label: at most ``C`` host->device copies however many contexts ask)."""
        dev = device if device is not None else (self._device or "cpu")
        t = self._context_index_dev(k, dev, cache=False)
        return torch.arange(self.N, dtype=torch.long, device=dev) if t is None else t

    def _label_index_dev(self, v: int, device):
        key = (v, str(device))
        t = self._lab_dev.get(key)
        if t is None:
            t = torch.tensor(self._pos.get(v, []), dtype=torch.long, device=device)
            self._lab_dev[key] = t
            self._devs.add(str(device))
        return t

    def _context_index_dev(self, k: int, device, cache: bool = True):
        """Context k's column index on ``device`` (``None`` = identity), concatenated ON THE DEVICE from the per-label indices."""
        key = (k, str(device))
        if key in self._kidx_dev:
            return self._kidx_dev[key]
        labs = self._labs[k]
        if labs is _FULL_LABELS:
            if self._all_valid:
                t = None
            else:
                fkey = ("full", str(device))
                t = self._lab_dev.get(fkey)
                if t is None:
                    t = torch.tensor(self._idx_list(k), dtype=torch.long, device=device)
                    self._lab_dev[fkey] = t
                    self._devs.add(str(device))
        elif len(labs) == 1:
            t = self._label_index_dev(labs[0], device)
        else:
            ps = sorted(((v, self._pos.get(v, [])) for v in labs if self._pos.get(v)), key=lambda vp: vp[1][0])
            parts = [self._label_index_dev(v, device) for v, _p in ps]
            if not parts:
                t = torch.zeros((0,), dtype=torch.long, device=device)
            else:
                t = torch.cat(parts) if len(parts) > 1 else parts[0]
                if not all(ps[i][1][-1] < ps[i + 1][1][0] for i in range(len(ps) - 1)):
                    t = torch.sort(t).values                                     # interleaved labels: ascending token order, as the mask gave it
        if cache:
            self._kidx_dev[key] = t
            self._devs.add(str(device))
        return t

    def _cols(self, v: int, g: int, device):
        """Packed column index of (label v, group g) on ``device``: the member contexts' ascending indices concatenated in member order
        (``False`` = one FULL identity member: no select)."""
        key = (v, g, str(device))
        t = self._cols_dev.get(key)
        if t is None:
            m = self._members_of(v, g)
            if len(m) == 1 and self._idx_list(m[0]) is None:
                t = False
            else:
                parts = []
                for k in m:
                    lst = self._idx_list(k)
                    parts.append(torch.arange(self.N, dtype=torch.long) if lst is None else torch.tensor(lst, dtype=torch.long))
                t = (torch.cat(parts) if len(parts) > 1 else parts[0]).to(device)
            self._cols_dev[key] = t
            self._devs.add(str(device))
        return t

    def counts(self, k: int, bounds: Optional[Sequence[Tuple[int, int]]] = None) -> List[int]:
        import bisect
        b = self._bounds(bounds)
        key = tuple(b)
        tab = self._counts_cache.get(key)
        if tab is None:
            tab = {v: [bisect.bisect_left(p, q1) - bisect.bisect_left(p, q0) for (q0, q1) in b] for v, p in self._pos.items()}
            self._counts_cache[key] = tab
        labs = self._labs[k]
        out = [0] * len(b)
        for v in (self._pos.keys() if labs is _FULL_LABELS else labs):
            c = tab.get(v)
            if c is not None:
                for q in range(len(b)):
                    out[q] += c[q]
        return out

    def _bounds(self, bounds):
        b = bounds if bounds is not None else self.bounds
        if b is None:
            if not is_dist():
                return [(self.r0, self.r1)]
            raise RowpairRefused("FastContextReducer: per-rank row bounds are required for a gather (bounds= at construction or at the call)")
        return [tuple(map(int, x)) for x in b]

    def local_rows(self, k: int) -> List[int]:
        labs = self._labs[k]
        rows: List[int] = []
        for lo, hi, v in self._runs:
            if labs is _FULL_LABELS or v in labs:
                rows.extend(range(lo, hi))
        return rows

    # ------------------------------------------------------------------ the statement
    def _block_runs(self, i0: int, i1: int) -> List[Tuple[int, int, int]]:
        """The label runs of local block ``[i0, i1)``: ``[(lo, hi, label)]`` clipped to the block, ascending."""
        import bisect
        out = []
        j = max(0, bisect.bisect_right(self._run_starts, i0) - 1)
        while j < len(self._runs):
            lo, hi, v = self._runs[j]
            if lo >= i1:
                break
            a, b = max(lo, i0), min(hi, i1)
            if b > a:
                out.append((a, b, v))
            j += 1
        return out

    def _active(self, runs) -> List[Tuple[int, List[Tuple[int, int]]]]:
        """The contexts with rows in a block (ascending k) and, per context, its row runs ``[(lo, hi)]`` inside the block (ascending)."""
        per_k: Dict[int, List[Tuple[int, int]]] = {}
        if runs:
            for k in self._fulls:
                per_k[k] = [(lo, hi) for (lo, hi, _v) in runs]
        for (lo, hi, v) in runs:
            for k in self._by_label.get(v, []):
                per_k.setdefault(k, []).append((lo, hi))
        return sorted(per_k.items())

    def _plan(self, i0: int, i1: int, probs):
        """Per weight group active in block ``[i0, i1)`` (in the classic first-appearance order): ``(g, form, [(k, runs_k)])``."""
        w_rows, N = i1 - i0, int(probs.shape[-2])
        runs = self._block_runs(i0, i1)
        active = self._active(runs)
        by_group: Dict[int, List[Tuple[int, List[Tuple[int, int]]]]] = {}
        for k, rk in active:
            by_group.setdefault(self._g[k], []).append((k, rk))
        plan = []
        for g, items in by_group.items():
            form = self.form
            if form == "auto":                                                   # the classic rule: block when the group's contexts select at least half the block's work
                work = sum(sum(hi - lo for lo, hi in rk) * self._n[k] for k, rk in items)
                form = "block" if 2 * work >= w_rows * N else "sub"
            plan.append((g, form, items))
        return runs, plan

    def _rows_view(self, probs, i0: int, i1: int, rk: List[Tuple[int, int]], dim: int):
        """The rows ``rk`` (runs, ascending) of a block tensor along ``dim``: the tensor itself, one ``narrow`` view, or a ``cat`` of narrows."""
        merged: List[List[int]] = []
        for lo, hi in rk:                                                       # adjacent runs (a label boundary, no unlabelled row between) are one slice
            if merged and merged[-1][1] == lo:
                merged[-1][1] = hi
            else:
                merged.append([lo, hi])
        if len(merged) == 1 and merged[0][0] == i0 and merged[0][1] == i1:
            return probs
        if len(merged) == 1:
            return probs.narrow(dim, merged[0][0] - i0, merged[0][1] - merged[0][0])
        return torch.cat([probs.narrow(dim, lo - i0, hi - lo) for lo, hi in merged], dim=dim)

    def _sub_piece(self, k: int, rk, i0: int, i1: int, probs, w):
        """Form ``sub``, context k: the classic statement on the same-shape tensor (rows by narrow / cat, columns by the device index)."""
        sub = self._rows_view(probs, i0, i1, rk, -3)
        idx = self._context_index_dev(k, probs.device)
        if idx is not None:
            sub = sub.index_select(-2, idx)                                      # [*lead, r', n_D, bins]
            self.stats["context_selects"] += 1
        return (sub * w).sum(dim=-1)                                             # (the engine statement on the sub-block)

    def _packed(self, g: int, items, runs, i0: int, i1: int, probs, w):
        """Form ``block``, group g: ``T = (probs * w).sum(-1)`` once, then per label run of the block that carries the group ONE packed column
        gather. Returns ``[(lo, hi, v, packed [*lead, hi-lo, L(v, g)])]`` in run order."""
        T = (probs * w).sum(dim=-1)                                              # [*lead, w, N]  (the engine statement, once per weight)
        self.stats["group_passes"] += 1
        full_in_g = any(self._g[k] == g for k in self._fulls)
        out = []
        for (lo, hi, v) in runs:
            if not full_in_g and not any(self._g[k] == g for k in self._by_label.get(v, [])):
                continue                                                         # no member of g has rows in this run
            cols = self._cols(v, g, probs.device)
            rows = T if (lo == i0 and hi == i1) else T.narrow(-2, lo - i0, hi - lo)
            if cols is False:
                piece = rows
            else:
                piece = rows.index_select(-1, cols)                              # ONE gather for every member context of (v, g): copies of T's elements
                self.stats["packed_selects"] += 1
            out.append((lo, hi, v, piece))
        del T
        return out

    @_no_grad
    def pieces(self, i0: int, i1: int, probs):
        """Yields ``(k, rows_local, T_piece [*lead, len(rows_local), n_D])`` for every context with rows in LOCAL block ``[i0, i1)`` — the values,
        shapes and yield order of :meth:`ContextReducer.pieces` (block form: column slices of the packed pieces, concatenated over the context's
        runs in ascending row order). No state is touched (device index caches aside)."""
        i0, i1 = int(i0), int(i1)
        if int(probs.shape[-3]) != i1 - i0:
            raise RowpairRefused(f"FastContextReducer: probs has {int(probs.shape[-3])} rows for block [{i0},{i1})")
        runs, plan = self._plan(i0, i1, probs)
        for g, form, items in plan:
            w = self._w_of_group[g]
            if form == "block":
                packs = {(lo, v): piece for (lo, hi, v, piece) in self._packed(g, items, runs, i0, i1, probs, w)}
                labels_at = {lo: v for (lo, hi, v) in runs}
                for k, rk in items:
                    parts = []
                    for lo, hi in rk:
                        v = labels_at[lo]
                        parts.append(packs[(lo, v)].narrow(-1, self._off[(v, g)][k], self._n[k]))
                    rows = [r for lo, hi in rk for r in range(lo, hi)]
                    yield k, rows, (parts[0] if len(parts) == 1 else torch.cat(parts, dim=-2))
                del packs
            else:
                for k, rk in items:
                    yield k, [r for lo, hi in rk for r in range(lo, hi)], self._sub_piece(k, rk, i0, i1, probs, w)

    @_no_grad
    def consume(self, i0: int, i1: int, probs) -> None:
        import time
        t0 = time.perf_counter()
        i0, i1 = int(i0), int(i1)
        if not (i0 == self._next and 0 <= i0 < i1 <= self.R):
            raise RowpairRefused(f"FastContextReducer.consume: rows [{i0},{i1}) out of order (next expected {self._next}, R={self.R})")
        if int(probs.shape[-3]) != i1 - i0:
            raise RowpairRefused(f"FastContextReducer: probs has {int(probs.shape[-3])} rows for block [{i0},{i1})")
        lead = tuple(int(s) for s in probs.shape[:-3])
        if self._lead is None:
            self._lead = lead
        elif self._lead != lead:
            raise RowpairRefused(f"FastContextReducer.consume: leading dims {lead} differ from the first block's {self._lead}")
        self._dtype, self._device = probs.dtype, probs.device
        runs, plan = self._plan(i0, i1, probs)
        for g, form, items in plan:
            w = self._w_of_group[g]
            if form == "block":
                for (lo, hi, v, piece) in self._packed(g, items, runs, i0, i1, probs, w):
                    self._members_of(v, g)
                    self._store.setdefault((v, g), []).append((lo, hi, piece))
            else:
                for k, rk in items:
                    self._kstore.setdefault(k, []).append((rk[0][0], self._sub_piece(k, rk, i0, i1, probs, w)))
        self._next = i1
        self.stats["consume_s"] += time.perf_counter() - t0

    # ------------------------------------------------------------------ collectives
    def _mine(self, k: int):
        """This rank's rows of context k ``[*lead, R_k, n_k]`` in ascending local row order (packed-store column slices and per-context pieces)."""
        labs, g, n = self._labs[k], self._g[k], self._n[k]
        segs = list(self._kstore.get(k, []))
        for v in (self._labels_here if labs is _FULL_LABELS else labs):
            lst = self._store.get((v, g))
            if not lst:
                continue
            off = self._off[(v, g)][k]
            for (lo, hi, piece) in lst:
                segs.append((lo, piece.narrow(-1, off, n)))
        segs.sort(key=lambda s: s[0])
        if not segs:
            lead = self._lead if self._lead is not None else ()
            dt = self._dtype or torch.float32
            dev = self._device or self.ctxs[k].w.device
            return torch.zeros(tuple(lead) + (0, n), dtype=dt, device=dev)
        return segs[0][1] if len(segs) == 1 else torch.cat([p for _lo, p in segs], dim=-2)

    def _release(self, k: int) -> None:
        """Drop context k's per-context pieces and its reference on every pack it belongs to (a pack goes when its last member was gathered)."""
        self._kstore.pop(k, None)
        for d in list(self._devs):
            self._kidx_dev.pop((k, d), None)
        labs, g = self._labs[k], self._g[k]
        for v in (self._labels_here if labs is _FULL_LABELS else labs):
            key = (v, g)
            if key in self._refs:
                self._refs[key] -= 1
                if self._refs[key] <= 0:
                    self._store.pop(key, None)
                    self._refs.pop(key, None)
                    for d in list(self._devs):
                        self._cols_dev.pop((v, g, d), None)

    @_no_grad
    def gather(self, k: int, bounds: Optional[Sequence[Tuple[int, int]]] = None):
        """COLLECTIVE (API parity with :meth:`ContextReducer.gather`): context k's ``[*lead, n_D, n_D]`` on rank 0, None elsewhere; ONE collective."""
        if self._next != self.R:
            raise RowpairRefused(f"FastContextReducer.gather: local rows [{self._next},{self.R}) were never consumed")
        counts = self.counts(k, bounds)
        n = self._n[k]
        if sum(counts) != n:
            raise RowpairRefused(f"FastContextReducer.gather: per-rank bounds cover {sum(counts)} of context {k}'s {n} rows (bounds must partition [0, N) once)")
        mine = self._mine(k)
        piece = mine.movedim(-2, 0).contiguous()                                 # [n_q, *lead, n_D]
        del mine
        self._release(k)
        full = gather_cat_to_rank0(piece, counts)
        self.stats["gathers"] += 1
        del piece
        if full is None:
            return None
        return full.movedim(0, -2)                                              # [*lead, n_D, n_D]

    @_no_grad
    def finalize(self, finish: Optional[Callable] = None, bounds: Optional[Sequence[Tuple[int, int]]] = None):
        """COLLECTIVE. ``{k: finish(k, ctx, T) if finish else T}`` on rank 0 over all contexts in order, None elsewhere. Contexts travel in
        BATCHES of at most ``gather_bytes`` of ``[*lead, n_D, n_D]`` (a larger context alone): one collective per batch; every ``T`` has the
        sizes and strides :meth:`ContextReducer.gather` returns (``[n_D, *lead, n_D]`` contiguous, ``movedim(0, -2)``)."""
        import time
        t0 = time.perf_counter()
        if self._next != self.R:
            raise RowpairRefused(f"FastContextReducer.finalize: local rows [{self._next},{self.R}) were never consumed")
        P, rank_ = world() if is_dist() else (1, 0)
        b = self._bounds(bounds)
        if len(b) != P:
            if P == 1:
                b = [(self.r0, self.r1)]
            else:
                raise RowpairRefused(f"FastContextReducer.finalize: {len(b)} bounds for P={P}")
        out = {} if rank_ == 0 else None
        K = len(self.ctxs)
        lead = tuple(self._lead) if self._lead is not None else ()
        L = 1
        for s in lead:
            L *= int(s)
        esz = torch.empty((), dtype=self._dtype or torch.float32).element_size()
        from .dist import cpu_threads as _cpu_threads
        with _cpu_threads(self.cpu_threads):
            k = 0
            while k < K:
                batch, byt = [], 0
                while k < K:
                    bk = self._n[k] * self._n[k] * L * esz
                    if batch and byt + bk > self.gather_bytes:
                        break
                    batch.append(k)
                    byt += bk
                    k += 1
                    if byt >= self.gather_bytes:
                        break
                cnts = {kk: self.counts(kk, b) for kk in batch}
                for kk in batch:
                    if sum(cnts[kk]) != self._n[kk]:
                        raise RowpairRefused(f"FastContextReducer.finalize: bounds cover {sum(cnts[kk])} of context {kk}'s {self._n[kk]} rows (bounds must partition [0, N) once)")
                parts = []
                for kk in batch:
                    mine = self._mine(kk)                                        # [*lead, R_k, n_k]
                    parts.append(mine.movedim(-2, 0).contiguous().reshape(-1))   # [R_k, *lead, n_k] flat: ContextReducer.gather's per-rank layout
                    del mine
                    self._release(kk)
                if parts:
                    flat = parts[0] if len(parts) == 1 else torch.cat(parts)
                else:
                    flat = torch.zeros((0,), dtype=self._dtype or torch.float32, device=self._device or "cpu")
                del parts
                lens = [sum(cnts[kk][q] * L * self._n[kk] for kk in batch) for q in range(P)]
                if int(flat.numel()) != lens[rank_]:
                    raise RowpairRefused(f"FastContextReducer.finalize: rank {rank_} packed {int(flat.numel())} elements != expected {lens[rank_]}")
                full = gather_cat_to_rank0(flat, lens)                           # ONE collective per batch
                self.stats["gathers"] += 1
                del flat
                if rank_ == 0:
                    seg = [0] * P
                    for q in range(1, P):
                        seg[q] = seg[q - 1] + lens[q - 1]
                    o_in = [0] * P
                    for kk in batch:
                        nk = self._n[kk]
                        rows = []
                        for q in range(P):
                            c = cnts[kk][q]
                            if c > 0:
                                s0 = seg[q] + o_in[q]
                                rows.append(full[s0:s0 + c * L * nk].view((c,) + lead + (nk,)))
                            o_in[q] += c * L * nk
                        if not rows:
                            Tk = full.new_zeros((0,) + lead + (nk,))
                        else:
                            Tk = rows[0] if len(rows) == 1 else torch.cat(rows, dim=0)   # [n_k, *lead, n_k] contiguous (rank order = ascending token order)
                        T = Tk.movedim(0, -2)                                    # [*lead, n_k, n_k]: ContextReducer.gather's sizes AND strides
                        out[kk] = finish(kk, self.ctxs[kk], T) if finish is not None else T
                        del T, Tk, rows
                del full
        self.stats["finalize_s"] = time.perf_counter() - t0
        if is_dist():                                                            # census (P > 1 only): the batched-gather count and the finalize wall of this rank
            from .evidence import record_schedule
            record_schedule(conf_gathers=int(self.stats["gathers"]), conf_finalize_s=round(float(self.stats["finalize_s"]), 3))
        return out


class BothContextReducer:
    """``ROWPAIR_CONF_REDUCER=both`` — the in-model identity check (debug lever): a :class:`ContextReducer` and a :class:`FastContextReducer` over
    the same contexts, driven side by side; every block's pieces (per context: local rows and ``torch.equal`` of the values) on every rank, and
    every gathered context matrix (values, sizes, strides) on rank 0, are compared; the CLASSIC results are what the caller gets (``finish`` runs
    once, on the classic matrix). Mismatches are logged (``comm().log``), counted into the census word ``conf_reducer_mismatch``, and — after the
    last collective of :meth:`finalize`, so no peer is left inside one — raised as :class:`RowpairRefused` on the rank that saw them. Costs two
    statement evaluations per reducer per block and two collectives per context (``gather`` per context on both: one matrix pair alive at a time).
    ``index`` / ``labels_of`` / ``stats`` are the fast reducer's."""

    def __init__(self, classic: ContextReducer, fast: FastContextReducer):
        self.classic, self.fast = classic, fast
        self.ctxs = classic.ctxs
        self.r0, self.r1, self.R, self.bounds = classic.r0, classic.r1, classic.R, classic.bounds
        self.mismatch = {"n": 0, "counts": 0, "local_rows": 0, "pieces": 0, "T": 0, "strides": 0}
        self._notes: List[str] = []
        for k in range(len(self.ctxs)):
            if classic.n(k) != fast.n(k):
                self._miss("n", f"context {k}: n classic={classic.n(k)} fast={fast.n(k)}")
            if classic.local_rows(k) != fast.local_rows(k):
                self._miss("local_rows", f"context {k}: local rows differ")

    def _miss(self, kind: str, note: str) -> None:
        self.mismatch[kind] += 1
        if len(self._notes) < 20:
            self._notes.append(note)
        from .dist import comm
        comm().log(f"[confidence] ROWPAIR_CONF_REDUCER=both MISMATCH {kind}: {note}")   # a stderr line when verbose; the census word and the final refusal carry the count regardless

    @property
    def stats(self):
        return self.fast.stats

    def n(self, k: int) -> int:
        return self.classic.n(k)

    def local_rows(self, k: int) -> List[int]:
        return self.classic.local_rows(k)

    def labels_of(self, k: int):
        return self.fast.labels_of(k)

    def index(self, k: int, device=None):
        return self.fast.index(k, device)

    def counts(self, k: int, bounds=None) -> List[int]:
        a, b = self.classic.counts(k, bounds), self.fast.counts(k, bounds)
        if a != b:
            self._miss("counts", f"context {k}: counts classic={a} fast={b}")
        return a

    def _compare_pieces(self, i0: int, i1: int, probs) -> list:
        pa = [(k, rows, T) for k, rows, T in self.classic.pieces(i0, i1, probs)]
        pb = {k: (rows, T) for k, rows, T in self.fast.pieces(i0, i1, probs)}
        ka = [k for k, _r, _t in pa]
        if set(ka) != set(pb):
            self._miss("pieces", f"block [{i0},{i1}): active contexts classic={sorted(ka)[:8]}... fast={sorted(pb)[:8]}...")
        for k, rows, T in pa:
            got = pb.get(k)
            if got is None or got[0] != rows or tuple(got[1].shape) != tuple(T.shape) or not torch.equal(got[1], T):
                self._miss("pieces", f"block [{i0},{i1}) context {k}: piece differs")
        return pa

    @_no_grad
    def pieces(self, i0: int, i1: int, probs):
        for k, rows, T in self._compare_pieces(int(i0), int(i1), probs):
            yield k, rows, T

    @_no_grad
    def consume(self, i0: int, i1: int, probs) -> None:
        self._compare_pieces(int(i0), int(i1), probs)
        self.classic.consume(i0, i1, probs)
        self.fast.consume(i0, i1, probs)

    @_no_grad
    def gather(self, k: int, bounds=None):
        Ta = self.classic.gather(k, bounds)
        Tb = self.fast.gather(k, bounds)
        if Ta is not None:
            if Tb is None or tuple(Ta.shape) != tuple(Tb.shape) or not torch.equal(Ta, Tb):
                self._miss("T", f"context {k}: gathered matrix differs")
            elif tuple(Ta.stride()) != tuple(Tb.stride()):
                self._miss("strides", f"context {k}: strides classic={tuple(Ta.stride())} fast={tuple(Tb.stride())}")
        del Tb
        return Ta

    def total_mismatches(self) -> int:
        return int(sum(self.mismatch.values()))

    @_no_grad
    def finalize(self, finish: Optional[Callable] = None, bounds=None):
        import time
        t0 = time.perf_counter()
        P, rank_ = world() if is_dist() else (1, 0)
        out = {} if rank_ == 0 else None
        for k, ctx in enumerate(self.ctxs):
            T = self.gather(k, bounds)
            if rank_ == 0:
                out[k] = finish(k, ctx, T) if finish is not None else T
            del T
        n_bad = self.total_mismatches()
        if is_dist():
            from .evidence import record_schedule
            record_schedule(conf_reducer_mismatch=n_bad, conf_gathers=2 * len(self.ctxs), conf_finalize_s=round(time.perf_counter() - t0, 3))
        if n_bad:
            raise RowpairRefused(f"ROWPAIR_CONF_REDUCER=both: FastContextReducer differs from ContextReducer on this rank — {dict(self.mismatch)}; first: {self._notes[:3]}")
        return out


def context_reducer_for_layout(contexts: Sequence[object], layout: Layout, *, form: str = "auto", labels=None, reducer: Optional[str] = None,
                               gather_bytes: Optional[int] = None, cpu_threads=None, allow_unsharded: bool = False):
    """The ONE factory a kit's confidence head calls for its TM-context reducer: ``reducer`` (else ``ROWPAIR_CONF_REDUCER``, default
    ``classic``) selects :class:`ContextReducer` (``classic`` — today's object, verbatim), :class:`FastContextReducer` (``fast`` — needs
    structured contexts: ``labels=`` + ``ctx.chains`` / core kinds; refused by name otherwise) or :class:`BothContextReducer` (``both`` — the
    in-model identity check). ``gather_bytes`` (else ``ROWPAIR_CONF_GATHER_MB``, default 512 MiB) and ``cpu_threads`` reach the fast reducer.
    Census (P > 1 only): ``conf_reducer=<word> conf_ctx=<n contexts> conf_index_s=<index build seconds> rank_threads=<torch intra-op threads>``
    (``rank_threads`` only when the launcher recorded none); the reducers add ``conf_gathers`` / ``conf_finalize_s`` (and ``conf_reducer_mismatch``
    under ``both``) at ``finalize``."""
    import time
    word = conf_reducer_word(reducer)
    if gather_bytes is None:
        from .dist import env_float
        gather_bytes = int(env_float(ENV_CONF_GATHER_MB, 512.0) * (1 << 20))
    t0 = time.perf_counter()
    if word == "classic":
        red = ContextReducer.for_layout(contexts, layout, form=form, allow_unsharded=allow_unsharded)
    elif word == "fast":
        red = FastContextReducer.for_layout(contexts, layout, form=form, allow_unsharded=allow_unsharded, labels=labels, gather_bytes=gather_bytes, cpu_threads=cpu_threads)
    else:
        red = BothContextReducer(ContextReducer.for_layout(contexts, layout, form=form, allow_unsharded=allow_unsharded),
                                 FastContextReducer.for_layout(contexts, layout, form=form, allow_unsharded=allow_unsharded, labels=labels, gather_bytes=gather_bytes,
                                                               cpu_threads=cpu_threads))
    index_s = time.perf_counter() - t0
    if is_dist():
        from .evidence import record_schedule, schedule
        facts = {"conf_reducer": word, "conf_ctx": len(red.ctxs), "conf_index_s": round(index_s, 3)}
        if "rank_threads" not in schedule():
            facts["rank_threads"] = int(torch.get_num_threads())
        record_schedule(**facts)
    return red


# ----------------------------------------------------------------------------- reducer
class RowBlockReducer:
    """Per-sample accumulator of row-block statistics on the rank owning global rows [r0, r1)."""

    def __init__(self, chains: ChainIndex, r0: int, r1: int, device, *,
                 pae_bins=(0.0, 32.0, 64), pde_bins=(0.0, 32.0, 64),
                 finish: str = "exact", eps: float = 1e-8,
                 keep_value_rows: bool = True, stats: Optional[Sequence[str]] = None, allow_unsharded: bool = False,
                 host_block_bytes: int = 64 << 20):
        if finish not in FINISHES:
            raise RowpairRefused(f"RowBlockReducer: finish={finish!r} not in {FINISHES}")
        if not allow_unsharded and not is_dist():
            raise RowpairRefused("RowBlockReducer: refused at n_gpu=1 (no group): the adapter installs nothing at n_gpu=1; a kit that ships the "
                                 "reducer as its own single-device memory lever says so (allow_unsharded=True) and names that lever in its line")
        self.stats = tuple(AF3_SUMMARIES if stats is None else stats)
        bad = [s for s in self.stats if s not in AF3_SUMMARIES]
        if bad:
            raise RowpairRefused(f"RowBlockReducer: stats {bad} are not in the AF3-family summary set {AF3_SUMMARIES}")
        from .evidence import record_schedule
        record_schedule(conf_finish=finish)
        self.chains, self.r0, self.r1, self.R = chains, int(r0), int(r1), int(r1 - r0)
        self.N, self.finish, self.eps = chains.N, finish, eps
        self.device = torch.device(device)
        self.pae_bins, self.pde_bins = tuple(pae_bins), tuple(pde_bins)
        self.centers_pae_cpu = tm_bin_centers(*self.pae_bins)
        self.centers_pde_cpu = tm_bin_centers(*self.pde_bins)
        self.centers_pae = self.centers_pae_cpu.to(self.device)
        self.centers_pde = self.centers_pde_cpu.to(self.device)
        self.ctxs = chains.contexts(self.centers_pae_cpu, self.device)
        self.keep_value_rows = keep_value_rows
        self.host_block_bytes = int(host_block_bytes)                          # rank 0's device transient per column block of the f16 collect
        N, R = self.N, self.R
        if keep_value_rows:
            self.pae_rows_f16 = torch.empty((R, N), dtype=torch.float16, device=self.device)
            self.pde_rows_f16 = torch.empty((R, N), dtype=torch.float16, device=self.device)
            self.contact_rows_f16 = torch.empty((R, N), dtype=torch.float16, device=self.device)
        # the TM-context pieces / gathers are the package's ONE mechanism (ContextReducer), with this head's statement order ("sub")
        self._tm = ContextReducer(self.ctxs, self.r0, self.r1, form="sub", allow_unsharded=True)
        if finish == "exact":
            self.pde_rows_f32 = torch.empty((R, N), dtype=torch.float32, device=self.device)
            self.contact_rows_f32 = torch.empty((R, N), dtype=torch.float32, device=self.device)
        else:
            C = chains.C
            self.rs_num: Dict[int, List[torch.Tensor]] = {k: [] for k in range(len(self.ctxs))}
            self.rs_full_mean_num: List[torch.Tensor] = []     # sum_j TM_N[i, j]            (ptm)
            self.gp_num = torch.zeros((R, C), dtype=torch.float32, device=self.device)  # sum_{j in x} pde*contact
            self.gp_den = torch.zeros((R, C), dtype=torch.float32, device=self.device)  # sum_{j in x} contact
        self._next = self.r0

    # ------------------------------------------------------------------ consume one row block
    @_no_grad
    def consume(self, c0: int, c1: int, pae_logits_rows: torch.Tensor, pde_logits_rows: torch.Tensor,
                contact_rows: torch.Tensor) -> None:
        """pae/pde logits rows [c, N, b] (fp32), contact rows [c, N] (fp32) for global rows [c0, c1)."""
        if not (c0 == self._next and self.r0 <= c0 < c1 <= self.r1):
            raise RowpairRefused("rowpair.confidence: violated: " + repr(((c0, c1, self._next, self.r0, self.r1))))
        c = c1 - c0
        if not (pae_logits_rows.shape[0] == c and pae_logits_rows.shape[1] == self.N):
            raise RowpairRefused(f"rowpair.confidence: violated: pae_logits_rows.shape[0] == c and pae_logits_rows.shape[1] == self.N")
        if not (pae_logits_rows.numel() < 2 ** 31):
            raise RowpairRefused("rowpair.confidence: violated: " + repr(("int32 rule: row block too large")))
        lo, hi = c0 - self.r0, c1 - self.r0
        dev_type = "cuda" if pae_logits_rows.is_cuda else "cpu"
        with torch.autocast(device_type=dev_type, enabled=False):
            pae_logits_rows = pae_logits_rows.to(torch.float32)
            pde_logits_rows = pde_logits_rows.to(torch.float32)
            contact_rows = contact_rows.to(torch.float32)
            # expected values (engine statement: prob = softmax(logits); score = prob @ centers)
            pde_prob, pde_val = expected_value_rows(pde_logits_rows, self.centers_pde)   # [1,c,N,b], [1,c,N]
            del pde_prob
            pae_prob, pae_val = expected_value_rows(pae_logits_rows, self.centers_pae)
            if self.keep_value_rows:
                self.pae_rows_f16[lo:hi] = pae_val[0].to(torch.float16)
                self.pde_rows_f16[lo:hi] = pde_val[0].to(torch.float16)
                self.contact_rows_f16[lo:hi] = contact_rows.to(torch.float16)
            if self.finish == "exact":
                self.pde_rows_f32[lo:hi] = pde_val[0]
                self.contact_rows_f32[lo:hi] = contact_rows
            else:
                pc = pde_val[0] * contact_rows                                  # [c, N]
                for x in range(self.chains.C):
                    m = self.chains.masks[x]
                    self.gp_num[lo:hi, x] = pc[:, m].sum(dim=-1)
                    self.gp_den[lo:hi, x] = contact_rows[:, m].sum(dim=-1)
            # TM contexts: T = (pae_prob[:, rows_D][:, :, cols_D] * w_D).sum(-1)  [1, n, N_d]  (engine statement; ContextReducer form "sub")
            if self.finish == "exact":
                self._tm.consume(lo, hi, pae_prob)                              # pieces kept per context; gathered in finalize
            else:
                for k, rows, T in self._tm.pieces(lo, hi, pae_prob):
                    ctx = self.ctxs[k]
                    grow = torch.tensor(rows, dtype=torch.long, device=self.chains.asym.device) + self.r0   # the piece's GLOBAL rows
                    if ctx.kind == FULL:
                        rows_asym, cols_asym = self.chains.asym_raw[grow], self.chains.asym_raw
                    else:
                        rows_asym, cols_asym = self.chains.asym[grow], self.chains.asym[ctx.mask]
                    if ctx.kind == FULL:
                        self.rs_full_mean_num.append(T[0].sum(dim=-1))
                    if ctx.kind == CHAIN:
                        self.rs_num[k].append(T[0].sum(dim=-1))
                    else:  # FULL (iptm) and PAIR: different-chain columns only
                        is_diff = cols_asym[None, :] != rows_asym[:, None]      # [n, N_d]
                        self.rs_num[k].append((T[0] * is_diff).sum(dim=-1))
                    del T
            del pae_prob, pae_val, pde_val
        self._next = c1

    # ------------------------------------------------------------------ helpers
    def _cat_or_empty(self, lst: List[torch.Tensor], tail_shape: Tuple[int, ...]) -> torch.Tensor:
        if len(lst):
            return torch.cat(lst, dim=0)
        return torch.empty((0,) + tuple(tail_shape), dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------ single-device finishing statements
    def _finish_ptm_T(self, T: torch.Tensor, has_frame: torch.Tensor) -> torch.Tensor:
        """T [1, N_d, N_d] -> ptm  (T.mean(-1)[..., has_frame].max(-1))"""
        if int(has_frame.sum()) == 0:
            return torch.zeros(size=T.shape[:-2], device=T.device)
        return T.mean(dim=-1)[..., has_frame].max(dim=-1).values

    def _finish_iptm_T(self, T: torch.Tensor, has_frame: torch.Tensor, asym: torch.Tensor) -> torch.Tensor:
        if int(has_frame.sum()) == 0:
            return torch.zeros(size=T.shape[:-2], device=T.device)
        is_diff_chain = asym[None, :] != asym[:, None]
        iptm = (T * is_diff_chain).sum(dim=-1) / (self.eps + is_diff_chain.sum(dim=-1))
        return iptm[..., has_frame].max(dim=-1).values

    def _finish_rowsum(self, num: torch.Tensor, den, has_frame: torch.Tensor) -> torch.Tensor:
        """num [N_d] per-row sums, den scalar or [N_d] -> max over framed rows of num/den (TIER-2 order)."""
        if int(has_frame.sum()) == 0 or num.numel() == 0:
            return torch.zeros((1,), device=self.device)
        v = (num / den).unsqueeze(0)
        return v[..., has_frame].max(dim=-1).values

    # ------------------------------------------------------------------ finalize (collective)
    @_no_grad
    def _selected(self, out: dict) -> dict:
        """The finished dict restricted to the requested summaries (``stats``): non-summary keys (full matrices, per-row pieces) pass."""
        return {k: v for k, v in out.items() if k not in AF3_SUMMARIES or k in self.stats}

    def finalize(self, bounds: Sequence[Tuple[int, int]], *, collect_full: bool = True) -> Optional[dict]:
        """Collective over all ranks (each holding the reducer of its own shard, same sample).

        bounds: per-rank (q0, q1) global row ranges in rank order (contiguous, ascending).
        Returns on rank 0 a dict with batch-dim-1 tensors:
          ptm [1], iptm [1], chain_ptm [1,C], chain_iptm [1,C], chain_pair_iptm [1,C,C],
          chain_pair_iptm_global [1,C,C], gpde [1], chain_gpde [1,C], chain_pair_gpde [1,C,C],
          and (exact) token_pair_pde_f32 [1,N,N], contact_probs_f32 [N,N] on device,
          and (collect_full) token_pair_pae_f16 / token_pair_pde_f16 / contact_probs_f16 [N,N] in (pinned) HOST memory, assembled in
          column blocks of ``host_block_bytes`` (:func:`opt_core.mem.rowpair.dist.gather_cat_to_rank0_host`; never whole on a device).
        Other ranks return None.
        """
        if not (self._next == self.r1):
            raise RowpairRefused("rowpair.confidence: violated: " + repr((f"rows [{self._next},{self.r1}) were never consumed")))
        P, rank = world() if is_dist() else (1, 0)
        ch, dev = self.chains, self.device
        C = ch.C
        is0 = rank == 0
        out = {}
        batch = (1,)
        chain_pair_iptm = torch.zeros(size=batch + (C, C)).to(dev)
        chain_ptm = torch.zeros(size=batch + (C,)).to(dev)
        full_counts = [q1 - q0 for (q0, q1) in bounds]
        if sum(full_counts) != self.N or (P > 1 and len(full_counts) != P):
            raise RowpairRefused(f"RowBlockReducer.finalize: bounds {list(bounds)} do not cover N={self.N} once over P={P} ranks")

        # ---- TM family, one context at a time
        for k, ctx in enumerate(self.ctxs):
            counts = self._tm.counts(k, bounds)
            if ctx.kind == FULL:
                hf, asym_sub = ch.has_frame, ch.asym_raw
            else:
                hf, asym_sub = ch.has_frame[ctx.mask], ch.asym[ctx.mask]
            if self.finish == "exact":
                Tm = self._tm.gather(k, bounds)                                  # [1, N_d, N_d] on rank 0 == single-device token_token_ptm; pieces freed everywhere
                if is0:
                    if ctx.kind == FULL:
                        out["ptm"] = self._finish_ptm_T(Tm, hf)
                        out["iptm"] = self._finish_iptm_T(Tm, hf, asym_sub)
                    elif ctx.kind == CHAIN:
                        chain_ptm[:, ctx.a] = self._finish_ptm_T(Tm, hf)
                    else:
                        chain_pair_iptm[:, ctx.a, ctx.b] = self._finish_iptm_T(Tm, hf, asym_sub)
                del Tm
            else:
                num = gather_cat_to_rank0(self._cat_or_empty(self.rs_num[k], ()), counts)
                if ctx.kind == FULL:
                    mnum = gather_cat_to_rank0(self._cat_or_empty(self.rs_full_mean_num, ()), counts)
                if is0:
                    if ctx.kind == FULL:
                        out["ptm"] = self._finish_rowsum(mnum, float(ctx.N_d), hf)
                        sizes = torch.tensor(ch.sizes, device=dev)
                        nd = ctx.N_d - sizes[ch.asym]                       # different-chain column count per row
                        out["iptm"] = self._finish_rowsum(num, self.eps + nd, hf)
                    elif ctx.kind == CHAIN:
                        chain_ptm[:, ctx.a] = self._finish_rowsum(num, float(ctx.N_d), hf)
                    else:
                        sizes = torch.tensor(ch.sizes, device=dev)
                        nd = ctx.N_d - sizes[asym_sub]
                        chain_pair_iptm[:, ctx.a, ctx.b] = self._finish_rowsum(num, self.eps + nd, hf)
        if is0:
            # symmetric fill, chain_iptm, chain_pair_iptm_global: single-device loops verbatim
            for a1 in range(C):
                for a2 in range(C):
                    if a1 > a2:
                        chain_pair_iptm[:, a1, a2] = chain_pair_iptm[:, a2, a1]
            chain_iptm = torch.zeros(size=batch + (C,)).to(dev)
            for aid in range(C):
                pairs = [(i, j) for i in range(C) for j in range(C)
                         if (i == aid or j == aid) and (i != j) and ch.chain_has_frame[i]]
                vals = [chain_pair_iptm[:, i, j] for (i, j) in pairs]
                if len(vals) > 0:
                    chain_iptm[:, aid] = torch.stack(vals, dim=-1).mean(dim=-1)
            chain_pair_iptm_global = torch.zeros(size=batch + (C, C)).to(dev)
            for a1 in range(C):
                for a2 in range(C):
                    if a1 == a2:
                        continue
                    if ch.chain_is_ligand[a1]:
                        chain_pair_iptm_global[:, a1, a2] = chain_iptm[:, a1]
                    elif ch.chain_is_ligand[a2]:
                        chain_pair_iptm_global[:, a1, a2] = chain_iptm[:, a2]
                    else:
                        chain_pair_iptm_global[:, a1, a2] = (chain_iptm[:, a1] + chain_iptm[:, a2]) * 0.5
            out.update({"chain_ptm": chain_ptm, "chain_iptm": chain_iptm,
                        "chain_pair_iptm": chain_pair_iptm, "chain_pair_iptm_global": chain_pair_iptm_global})

        # ---- gpde family
        if self.finish == "exact":
            pde_full = gather_cat_to_rank0(self.pde_rows_f32, full_counts)
            contact_full = gather_cat_to_rank0(self.contact_rows_f32, full_counts)
            if is0:
                pde_full = pde_full.unsqueeze(0)                                 # [1, N, N]
                out["token_pair_pde_f32"] = pde_full
                out["contact_probs_f32"] = contact_full                          # [N, N]
                out.update(gpde_from_full(pde_full, contact_full, ch, self.eps))
        else:
            num = gather_cat_to_rank0(self.gp_num, full_counts)                  # [N, C]
            den = gather_cat_to_rank0(self.gp_den, full_counts)
            if is0:
                out["gpde"] = (num.sum() / den.sum()).reshape(1)
                chain_gpde = torch.zeros(size=batch + (C,), device=dev)
                chain_pair_gpde = torch.zeros(size=batch + (C, C), device=dev)
                for a in range(C):
                    ra = ch.masks[a]
                    chain_gpde[:, a] = num[ra, a].sum() / (den[ra, a].sum() + self.eps)
                    for b in range(C):
                        if a == b:
                            continue
                        if b < a:
                            chain_pair_gpde[:, a, b] = chain_pair_gpde[:, b, a]
                            continue
                        chain_pair_gpde[:, a, b] = num[ra, b].sum() / (den[ra, b].sum() + self.eps)
                out["chain_gpde"], out["chain_pair_gpde"] = chain_gpde, chain_pair_gpde

        # ---- full matrices for the full_data replacement (fp16, HOST on rank 0 — assembled in column blocks, never whole on a device)
        if collect_full and self.keep_value_rows:
            for name, rows in (("token_pair_pae_f16", self.pae_rows_f16), ("token_pair_pde_f16", self.pde_rows_f16),
                               ("contact_probs_f16", self.contact_rows_f16)):
                g = gather_cat_to_rank0_host(rows, full_counts, block_bytes=self.host_block_bytes)
                if is0:
                    out[name] = g
                del g
        return self._selected(out) if is0 else None


# ----------------------------------------------------------------------------- gpde on full matrices
def gpde_from_full(token_pair_pde: torch.Tensor, contact_probs: torch.Tensor, ch: ChainIndex, eps: float = 1e-8) -> dict:
    """Single-device gpde statements on full matrices: token_pair_pde [1,N,N], contact_probs [N,N]."""
    out = {"gpde": (token_pair_pde * contact_probs).sum(dim=[-1, -2]) / contact_probs.sum(dim=[-1, -2])}
    C = ch.C
    batch_shape = token_pair_pde.shape[:-2]
    device = token_pair_pde.device

    def _cal(m1, m2):
        mc = contact_probs[..., m1, :][..., m2]
        mp = token_pair_pde[..., m1, :][..., m2]
        return (mp * mc).sum(dim=(-1, -2)) / (mc.sum(dim=(-1, -2)) + eps)

    chain_gpde = torch.zeros(size=batch_shape + (C,), device=device)
    for a in range(C):
        chain_gpde[..., a] = _cal(ch.masks[a], ch.masks[a])
    chain_pair_gpde = torch.zeros(size=batch_shape + (C, C), device=device)
    for a in range(C):
        for b in range(C):
            if a == b:
                continue
            if b < a:
                chain_pair_gpde[..., a, b] = chain_pair_gpde[..., b, a]
                continue
            chain_pair_gpde[..., a, b] = _cal(ch.masks[a], ch.masks[b])
    out["chain_gpde"], out["chain_pair_gpde"] = chain_gpde, chain_pair_gpde
    return out


def max_row_chunk(N: int, bins: int = 64, cap: int = 128) -> int:
    """Largest row block <= cap honouring the int32 rule (c * N * bins < 2**31) on a 128-grid divisor."""
    c = min(cap, max(1, (2 ** 31 - 1) // (N * bins)))
    for d in (128, 64, 32, 16, 8, 4, 2, 1):
        if d <= c:
            return d
    return 1


# ----------------------------------------------------------------------------- output seams of the head
def per_row_outputs_to_rank0(rows_shard, layout: Layout):
    """Per-row head outputs (plddt rows, per-token PAE rows an engine writes) ``[R, ...] -> [N, ...]`` on rank 0, None elsewhere."""
    return gather_rows_to_rank0(rows_shard.contiguous(), layout)


def gather_single_rows(s_shard, layout: Layout):
    """A row-sharded single-representation update ``[R, c]`` (or ``[*, R, c]`` with rows on dim -2) back to replicated ``[N, c]`` on every rank."""
    if s_shard.dim() == 2:
        return all_gather_rows(s_shard.contiguous(), layout)
    return all_gather_rows(s_shard.movedim(-2, 0).contiguous(), layout).movedim(0, -2)
