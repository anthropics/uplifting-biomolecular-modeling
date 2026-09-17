"""Deterministic replacement for atomic scatter-add / scatter-mean over an index (the atom -> token aggregation of all-atom models).

SAME FUNCTION, DIFFERENT REDUCTION ORDER — state exactly that wherever it ships:
  stock   ``out[i] = sum/mean_{j: idx[j]==i} src[j]`` via CUDA atomicAdd (``scatter_add_`` / ``index_reduce``): the summation order over j
          is undefined, so a few % of elements differ by a few ulp run to run — the run-to-run floor of every sampler that uses it.
  here    the same sum/mean as a *segment* reduction with a FIXED order: (1) stable-sort the index (a no-op permutation when each
          segment's rows are contiguous, the usual atom layout), (2) build a ``[dim_size, max_rows_per_segment]`` gather table from the
          index alone (cached per index tensor: equality + storage + version, so it is built once per structure), (3) gather the rows and
          ``torch.sum`` over the padded axis in fp32 (one fixed tree per (shape, dtype, device): bit-exact repeatable run to run and under
          CUDA-graph replay), (4) for a mean divide by the per-segment count clamped at 1 (as the upstreams do).
  precision  accumulation in fp32 whatever the source dtype (``accumulate="src"`` keeps the source dtype). For fp32 sources this is the
          stock precision class with a different order; for bf16 sources it is MORE precise than bf16 atomics — a precision change at
          bf16 call sites, which the kit names.
It is a qualification instrument first (applied to BOTH arms of an equality row under the kit's ``--det`` level, never to one) and only a
lever where a kit's mode table says so with its class.

Entry points (torch imported lazily; the module imports nothing heavy):
  :func:`segment_reduce_rows` ``(src [..., N, C], index [N], dim_size, reduce)`` -> ``[..., dim_size, C]`` — the primitive.
  :func:`scatter_mean_into` ``(zeros [..., I, C], dim, index [N], source [..., N, C])`` — the ``index_reduce(..., "mean", include_self=False)``
          call shape (an all-zero ``zeros`` by contract).
  :func:`scatter_reduce` ``(src, index, dim, out, dim_size, reduce, fallback)`` — the torch_scatter-style call shape (``sum|add|mean`` along
          ``dim = -2`` with a 1-D index or a ``[..., N]`` index of identical rows).
Layouts outside these (another dim, ``out=`` given, ragged multi-row index) are NOT served: the call goes to the ``fallback`` the adapter
passed (the stock function) and is COUNTED in :func:`census` under ``fallbacks`` with its reason — the adapter prints the census as its
activation line at exit (``calls``, ``served``, ``fallbacks``, ``fallback_reasons``, ``table_builds``); with no fallback given an unserved
layout raises :class:`LayoutUnsupported`. Nothing falls back silently.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Optional

from . import PrecisionError, require_torch

MAX_TABLES = 64                      # gather tables kept (LRU); one per distinct index tensor
_TABLES = OrderedDict()
_STATS = {"calls": 0, "served": 0, "fallbacks": 0, "table_builds": 0}
_FALLBACK_REASONS: dict = {}


class LayoutUnsupported(PrecisionError):
    """The call's layout is outside the served set and no stock fallback was given."""

    event = "det_scatter_layout"


def census() -> dict:
    """The activation-evidence fields: ``calls served fallbacks fallback_reasons table_builds tables``."""
    out = dict(_STATS)
    out["fallback_reasons"] = dict(_FALLBACK_REASONS) or None
    out["tables"] = len(_TABLES)
    return out


def reset() -> None:
    """Drop the cached tables and zero the census (tests; a kit calls it between items only if it reports per item)."""
    _TABLES.clear()
    _FACTS.clear()
    _FALLBACK_REASONS.clear()
    for k in _STATS:
        _STATS[k] = 0


def _fallback(reason: str, fallback: Optional[Callable], *args):
    _STATS["fallbacks"] += 1
    _FALLBACK_REASONS[reason] = _FALLBACK_REASONS.get(reason, 0) + 1
    if fallback is None:
        raise LayoutUnsupported("deterministic scatter does not serve this layout (%s) and no fallback was given" % reason, layout=reason)
    return fallback(*args)


def _table(index, dim_size: int, torch):
    """index 1-D integer [N] -> (table [dim_size, max_per] long rows of src, valid [dim_size, max_per] bool, counts [dim_size]).
    Cached on (id, data_ptr, numel, version, dtype, device, dim_size); the entry holds the index tensor so its id cannot be recycled."""
    key = (id(index), index.data_ptr(), index.numel(), index._version, index.dtype, str(index.device), int(dim_size))
    ent = _TABLES.get(key)
    if ent is not None and ent[0] is index:
        _TABLES.move_to_end(key)
        return ent[1]
    _STATS["table_builds"] += 1
    idx = index.reshape(-1).long()
    n = idx.numel()
    sorted_idx, order = torch.sort(idx, stable=True)                       # fixed within-segment order = original order
    if n > 0:                                                              # host syncs happen here, once per NEW table, never per call
        lo, hi = int(sorted_idx[0].item()), int(sorted_idx[-1].item())
        if lo < 0 or hi >= dim_size:
            raise LayoutUnsupported("index values must lie in [0, %d) (got min %d, max %d) — the atomic op would raise here too"
                                    % (dim_size, lo, hi), layout="index_out_of_range")
    counts = torch.bincount(sorted_idx, minlength=dim_size)[:dim_size]
    max_per = max(int(counts.max().item()) if n > 0 else 1, 1)
    starts = torch.cumsum(counts, 0) - counts
    ar = torch.arange(max_per, device=idx.device)
    pos = starts[:, None] + ar[None, :]
    valid = ar[None, :] < counts[:, None]
    pos = torch.where(valid, pos, torch.zeros_like(pos)).clamp_(max=max(n - 1, 0))
    table = order[pos] if n > 0 else pos
    built = (table, valid, counts)
    _TABLES[key] = (index, built)
    while len(_TABLES) > MAX_TABLES:
        _TABLES.popitem(last=False)
    return built


def segment_reduce_rows(src, index, dim_size: int, reduce: str = "mean", accumulate: str = "fp32", torch=None):
    """``src [..., N, C]``, integer ``index [N]`` -> ``[..., dim_size, C]``; ``reduce`` in ``mean|sum``; ``accumulate`` in ``fp32|src``.
    Deterministic for a given (shape, dtype, device); result cast back to ``src.dtype``."""
    torch = require_torch(torch)
    if reduce not in ("mean", "sum"):
        raise ValueError("reduce must be 'mean' or 'sum', not %r" % (reduce,))
    if accumulate not in ("fp32", "src"):
        raise ValueError("accumulate must be 'fp32' or 'src', not %r" % (accumulate,))
    if index.dim() != 1 or index.numel() != src.shape[-2]:
        raise LayoutUnsupported("segment_reduce_rows needs a 1-D index of length N=src.shape[-2] (got index %s, src %s)"
                                % (tuple(index.shape), tuple(src.shape)), layout="index_shape")
    lead = src.shape[:-2]
    c = src.shape[-1]
    if index.numel() == 0:
        return src.new_zeros(tuple(lead) + (int(dim_size), c))
    table, valid, counts = _table(index, int(dim_size), torch)
    g = src[..., table.reshape(-1), :].reshape(*(tuple(lead) + (table.shape[0], table.shape[1], c)))
    acc = torch.float32 if accumulate == "fp32" else src.dtype
    g = g.to(acc) * valid.to(acc)[..., None]
    s = g.sum(dim=-2)
    if reduce == "mean":
        s = s / counts.clamp(min=1).to(s.dtype)[:, None]
    return s.to(src.dtype)


_FACTS = OrderedDict()          # per index tensor: (rows_identical, max+1) — the host syncs a call shape needs, paid once


def _facts(index, torch):
    key = (id(index), index.data_ptr(), index.numel(), index._version, index.dtype, str(index.device), tuple(index.shape))
    ent = _FACTS.get(key)
    if ent is not None and ent[0] is index:
        _FACTS.move_to_end(key)
        return ent[1]
    if index.dim() == 1:
        same = True
    else:
        flat = index.reshape(-1, index.shape[-1])
        same = bool((flat == flat[:1]).all().item()) if flat.shape[0] > 1 else True
    size = (int(index.max().item()) + 1) if index.numel() else 0
    facts = (same, size)
    _FACTS[key] = (index, facts)
    while len(_FACTS) > MAX_TABLES:
        _FACTS.popitem(last=False)
    return facts


def scatter_mean_into(zeros, dim: int, index, source, fallback: Optional[Callable] = None, accumulate: str = "fp32", torch=None):
    """The ``zeros.index_reduce(dim, index, source, "mean", include_self=False)`` call shape: ``zeros [..., I, C]`` all-zero by contract,
    ``dim`` the row axis (``-2`` / ``source.dim()-2``), ``index`` 1-D ``[N]``, ``source [..., N, C]``. Other layouts -> ``fallback(zeros,
    dim, index, source)`` (counted) or :class:`LayoutUnsupported`."""
    torch = require_torch(torch)
    _STATS["calls"] += 1
    nd = source.dim()
    d = dim + nd if dim < 0 else dim
    if d != nd - 2:
        return _fallback("dim", fallback, zeros, dim, index, source)
    if index.dim() != 1:
        return _fallback("index_rank", fallback, zeros, dim, index, source)
    if zeros.shape[-1] != source.shape[-1]:
        return _fallback("channels", fallback, zeros, dim, index, source)
    lead_z, lead_s = tuple(zeros.shape[:-2]), tuple(source.shape[:-2])
    if lead_s != lead_z and len(lead_s) != 0:
        return _fallback("lead_shape", fallback, zeros, dim, index, source)
    out = segment_reduce_rows(source, index, zeros.shape[-2], "mean", accumulate, torch)
    _STATS["served"] += 1
    if lead_s != lead_z:                                                   # source [N, C] into zeros [..., I, C]: the same rows for every lead
        out = out.expand(lead_z + tuple(out.shape))
    return out.to(zeros.dtype)


def scatter_reduce(src, index, dim: int = -1, out=None, dim_size: Optional[int] = None, reduce: str = "sum",
                   fallback: Optional[Callable] = None, accumulate: str = "fp32", torch=None):
    """The torch_scatter-style call shape ``scatter(src, index, dim, out, dim_size, reduce)`` for ``reduce`` in ``sum|add|mean`` along the
    row axis (``dim = -2`` / ``src.dim()-2``) with ``out=None`` and an index that is 1-D ``[N]`` or ``[..., N]`` with identical rows (a
    broadcast atom -> token map). Anything else -> ``fallback(src, index, dim, out, dim_size, reduce)`` (counted) or
    :class:`LayoutUnsupported`. ``dim_size=None`` -> ``index.max()+1``. The host syncs a call needs (rows-identical check, max, gather table) are paid ONCE per
    index tensor and cached, so steady-state calls are sync-free and CUDA-graph capturable after one warm call."""
    torch = require_torch(torch)
    _STATS["calls"] += 1
    d = dim + src.dim() if dim < 0 else dim
    if reduce not in ("sum", "add", "mean"):
        return _fallback("reduce_" + str(reduce), fallback, src, index, dim, out, dim_size, reduce)
    if out is not None:
        return _fallback("out_given", fallback, src, index, dim, out, dim_size, reduce)
    if d != src.dim() - 2:
        return _fallback("dim", fallback, src, index, dim, out, dim_size, reduce)
    if index.shape[-1] != src.shape[-2]:
        return _fallback("index_length", fallback, src, index, dim, out, dim_size, reduce)
    same_rows, size = _facts(index, torch)
    if not same_rows:
        return _fallback("index_rows_differ", fallback, src, index, dim, out, dim_size, reduce)
    idx1 = index if index.dim() == 1 else index.reshape(-1, index.shape[-1])[0]
    if dim_size is None:
        dim_size = size
    res = segment_reduce_rows(src, idx1, int(dim_size), "mean" if reduce == "mean" else "sum", accumulate, torch)
    _STATS["served"] += 1
    return res
