"""det_segment_reduce.py — deterministic, CACHE-FREE, SYNC-FREE (CUDA-graph-safe) replacement for the atomic scatter-mean / scatter-add
over an atom->token index (Protenix `scatter_utils.scatter`; `foundry_scatter_mean` is the same routine in the
`scatter_mean(zeros, dim, index, source)` calling convention).

NOTHING derived from the index is cached: the gather table is rebuilt ON DEVICE on every call with STATIC shapes and NO host
synchronisation, so (a) eager results depend only on the values of (src, index), never on process history or allocator addresses
(a table cached by the index tensor's address would be hit, stale, once the caching allocator re-hands a freed address to the next
item's index tensor of equal numel), and (b) inside a captured graph the table build is part of the graph and follows the static
index buffer on replay (a cached table would bake the capturing item's segmentation into the graph).

SAME FUNCTION, DIFFERENT (FIXED) REDUCTION ORDER (label exactly that):
  stock: out[i] = mean_{j: idx[j]==i} src[j] with CUDA atomicAdd (order undefined -> run-to-run ulp noise).
  here:  (1) stable-sort the index on device (uniquely defined permutation -> value-determined), (2) per-token [start, count)
         via searchsorted (no atomics, no bincount, no .item()), (3) a [dim_size, MAX_PER] slot table (MAX_PER = static bound,
         default 32 >= any Protenix token: protein residues <= 14 heavy atoms, nucleotides <= 23, ligands 1 atom/token),
         (4) gather src rows slot-chunk by slot-chunk and reduce in fp32 with an EXPLICIT fixed binary tree inside each chunk of
         8 slots and a sequential sum across chunks (element-wise IEEE adds only -> the result is a pure function of the values;
         it does not depend on MAX_PER either, up to the sign of exact-zero outputs), (5) 'mean' divides by count.clamp(min=1).
  Padding slots contribute exact +0.0 through torch.where (multiplying by the mask instead would let NaN/Inf from a neighbouring
  row leak as NaN*0).  Same precision class as stock (fp32 accumulation); not bitwise equal to the atomic path.

HEALTH FLAGS (device-side, sticky, sync-free): slot overflow (a token with > MAX_PER atoms -> result WRONG for that token),
index rows differ across leading dims (capture only; eager falls back to stock), index out of range. Eager + STRICT
(default): checked with one host sync per call and raised immediately. Under CUDA-graph capture no sync is possible: call
`det_segment_reduce.assert_healthy()` after warm-up / after replays.

Constants: MAX_PER 32 slots per token (multiple of 8), STRICT on, ACCUMULATE fp32; PROTENIX_DET_SCATTER is read by the
     scatter_utils shim, unchanged.
Entry points: foundry_scatter_mean(zeros, dim, index, source); protenix_scatter(src, index, dim, out, dim_size, reduce, _stock).
"""
from __future__ import annotations
import torch
from torch import Tensor

__version__ = "1.1"
ACCUMULATE = "fp32"
MAX_PER = 32
STRICT = True
CHUNK = 8                                                     # slots per gather chunk; part of the (fixed) reduction-order definition
STATS = dict(calls=0, fallbacks=0, table_builds=0, strict_checks=0, captured_calls=0)
_FLAGS: dict = {}                                             # device -> int32[3] sticky flags: [slot_overflow, index_rows_differ, index_out_of_range]
_FLAG_NAMES = ("slot_overflow(token with more atoms than MAX_PER=%d slots)" % MAX_PER,
               "index_rows_differ(across leading dims; only row 0 was used — unsupported layout under capture)",
               "index_out_of_range(index < 0 or >= dim_size; those atoms were dropped)")


def _capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def init(device) -> Tensor:
    """Allocate the sticky health flags for `device` (call once in eager before any CUDA-graph capture; the first eager call does it)."""
    device = torch.device(device)
    t = _FLAGS.get(device)
    if t is None:
        t = torch.zeros(3, dtype=torch.int32, device=device)
        _FLAGS[device] = t
    return t


def health(reset: bool = False) -> dict:
    """Host read of the sticky flags (synchronises). {device: {flag_name: count}}."""
    out = {}
    for dev, t in _FLAGS.items():
        v = t.tolist()
        out[str(dev)] = {n.split("(")[0]: int(x) for n, x in zip(_FLAG_NAMES, v)}
        if reset:
            t.zero_()
    return out


def assert_healthy(reset: bool = True) -> dict:
    """Raise if any sticky flag fired since the last reset (use after graph warm-up/replays, where per-call checks are impossible)."""
    h = health(reset=reset)
    bad = {d: {k: v for k, v in f.items() if v} for d, f in h.items() if any(f.values())}
    if bad:
        raise RuntimeError(f"det_segment_reduce v{__version__}: health flags fired {bad}. Meaning: " + " | ".join(_FLAG_NAMES))
    return h


def build_table(index: Tensor, dim_size: int, max_per: int | None = None):
    """index 1-D int [N] (any order) -> (table [dim_size, max_per] int64 rows of src in original order within each token,
    valid [dim_size, max_per] bool, counts [dim_size] int64, overflow 0-d bool, out_of_range 0-d bool).
    Device-only, static shapes, no host synchronisation, nothing cached -> safe inside CUDA-graph capture."""
    max_per = MAX_PER if max_per is None else int(max_per)
    idx = index.reshape(-1)
    if idx.dtype != torch.int64:
        idx = idx.long()
    N = idx.numel()
    dev = idx.device
    sorted_idx, order = torch.sort(idx, stable=True)                      # stable => atoms of a token keep their original order
    tok = torch.arange(dim_size, device=dev, dtype=torch.int64)
    starts = torch.searchsorted(sorted_idx, tok, right=False)
    counts = torch.searchsorted(sorted_idx, tok, right=True) - starts     # exact integer counts without atomics / bincount / .item()
    ar = torch.arange(max_per, device=dev, dtype=torch.int64)
    valid = ar[None, :] < counts[:, None]
    pos = starts[:, None] + ar[None, :]
    pos = torch.where(valid, pos, torch.zeros_like(pos)).clamp_(min=0, max=max(N - 1, 0))
    table = order[pos]
    overflow = (counts > max_per).any()
    oor = (sorted_idx[:1] < 0).any() | (sorted_idx[-1:] >= dim_size).any()
    return table, valid, counts, overflow, oor


def segment_reduce_rows(src: Tensor, index: Tensor, dim_size: int, reduce: str = "mean", max_per: int | None = None) -> Tensor:
    """src [..., N, C], index [N] int -> out [..., dim_size, C]; reduce in {'mean','sum'}; deterministic, fp32 accumulate,
    fixed value-only-dependent order; no cache, no host sync."""
    max_per = MAX_PER if max_per is None else int(max_per)
    assert max_per % CHUNK == 0, "max_per must be a multiple of 8"
    lead = tuple(src.shape[:-2]); N, C = int(src.shape[-2]), int(src.shape[-1]); dim_size = int(dim_size)
    acc_dtype = torch.float32 if ACCUMULATE == "fp32" else src.dtype
    if N == 0 or dim_size == 0:
        return torch.zeros(*lead, dim_size, C, dtype=src.dtype, device=src.device)
    table, valid, counts, overflow, oor = build_table(index, dim_size, max_per)
    STATS["table_builds"] += 1
    fl = init(src.device)
    fl[0] |= overflow.to(torch.int32); fl[2] |= oor.to(torch.int32)       # sticky, device-side (captured graphs update them on replay too)
    d = src.dim() - 2
    vshape = (1,) * len(lead) + (dim_size, CHUNK, 1)
    acc = None
    for k0 in range(0, max_per, CHUNK):                                    # sequential over chunks (fixed order)
        tix = table[:, k0:k0 + CHUNK].reshape(-1)                          # [dim_size*CHUNK]
        g = src.index_select(d, tix).reshape(*lead, dim_size, CHUNK, C)
        if g.dtype != acc_dtype:
            g = g.to(acc_dtype)
        g = torch.where(valid[:, k0:k0 + CHUNK].reshape(vshape), g, g.new_zeros(()))   # padding slots -> exact +0.0 (no NaN*0 leak)
        # fixed binary tree over the 8 slots: (0+1),(2+3),(4+5),(6+7) -> (01+23),(45+67) -> all   [element-wise IEEE adds only]
        g = g[..., 0::2, :] + g[..., 1::2, :]
        g = g[..., 0::2, :] + g[..., 1::2, :]
        g = g[..., 0, :] + g[..., 1, :]
        acc = g if acc is None else acc + g
    if reduce == "mean":
        acc = acc / counts.clamp(min=1).to(acc.dtype)[:, None]
    elif reduce not in ("sum", "add"):
        raise ValueError(f"unsupported reduce={reduce!r}")
    return acc.to(src.dtype)


def _strict_check(device) -> None:
    """Eager-only per-call health check (one host sync). Skipped under CUDA-graph capture (use assert_healthy() after replays)."""
    if not STRICT or _capturing():
        return
    STATS["strict_checks"] += 1
    t = _FLAGS.get(torch.device(device))
    if t is None:
        return
    v = t.tolist()
    if any(v):
        t.zero_()
        fired = [n for n, x in zip(_FLAG_NAMES, v) if x]
        raise RuntimeError(f"det_segment_reduce v{__version__}: " + " | ".join(fired))


# ---------------------------------------------------------------- foundry ----------------------------------------------------------------
def foundry_scatter_mean(zeros: Tensor, dim: int, index: Tensor, source: Tensor) -> Tensor:
    """Drop-in for foundry.utils.torch.scatter_mean (zeros (...,I,C), dim=-2 or source.dim()-2, index 1-D [N], source (...,N,C))."""
    STATS["calls"] += 1
    ndim = source.dim(); d = dim + ndim if dim < 0 else dim
    if d != ndim - 2 or index.dim() != 1 or zeros.shape[-1] != source.shape[-1] or not source.is_floating_point():
        STATS["fallbacks"] += 1
        return zeros.index_reduce(dim, index, source, "mean", include_self=False)
    if _capturing():
        STATS["captured_calls"] += 1
    out = segment_reduce_rows(source, index, zeros.shape[-2], "mean")
    _strict_check(source.device)
    if zeros.shape[:-2] != out.shape[:-2]:
        out = out.expand(zeros.shape) if out.dim() == zeros.dim() else out.reshape(zeros.shape)
    return (zeros + out) if zeros.requires_grad else out.to(zeros.dtype)      # zeros is all-zero by contract (include_self=False)


# ---------------------------------------------------------------- Protenix ---------------------------------------------------------------
def protenix_scatter(src: Tensor, index: Tensor, dim: int = -1, out: Tensor | None = None, dim_size: int | None = None, reduce: str = "sum",
                     _stock=None) -> Tensor:
    """Drop-in for protenix.utils.scatter_utils.scatter for reduce in (sum, add, mean) along dim=-2 with an atom->token index
    ([N_atom] or [..., N_atom] with identical rows, as produced by the featuriser and broadcast in aggregate_atom_to_token).
    dim_size (= n_token) should be passed (Protenix's aggregate_atom_to_token does); if it is None it is inferred with ONE host
    sync in eager and refused under CUDA-graph capture."""
    STATS["calls"] += 1
    d = dim + src.dim() if dim < 0 else dim
    ok = (reduce in ("sum", "add", "mean") and out is None and d == src.dim() - 2 and index.shape[-1] == src.shape[-2]
          and src.is_floating_point())
    capturing = _capturing()
    idx1 = None
    if ok:
        if index.dim() == 1:
            idx1 = index
        else:
            flat = index.reshape(-1, index.shape[-1])
            idx1 = flat[0]
            if flat.shape[0] > 1:
                differ = (flat != flat[:1]).any()
                if capturing:                                              # cannot decide on the host: use row 0, record a sticky flag
                    init(src.device)[1] |= differ.to(torch.int32)
                elif bool(differ.item()):                                  # eager: differing index rows -> the stock implementation answers
                    ok = False
    if not ok:
        STATS["fallbacks"] += 1
        assert _stock is not None, "unsupported layout for the deterministic path and no stock fallback given"
        return _stock(src, index, dim, out, dim_size, reduce)
    if dim_size is None:
        if capturing:
            raise RuntimeError("det_segment_reduce v%s: dim_size (n_token) must be passed explicitly inside a CUDA-graph capture — no host "
                               "sync is possible there. Protenix's aggregate_atom_to_token passes n_token; check the call site. "
                               "(Slot bound: MAX_PER=%d.)" % (__version__, MAX_PER))
        dim_size = int(idx1.max().item()) + 1                             # eager only; same semantics as torch_scatter
    if capturing:
        STATS["captured_calls"] += 1
    res = segment_reduce_rows(src, idx1, int(dim_size), "mean" if reduce == "mean" else "sum")
    _strict_check(src.device)
    return res


def enabled(env: str = "DET_SCATTER") -> bool:
    return True
