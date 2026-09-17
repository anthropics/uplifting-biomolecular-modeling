"""Row-chunked evaluation of row-independent pair-tensor sub-modules (torch; imported lazily inside the functions).

Mechanism. A pair tensor ``[.., N, N, C]`` passed whole through a pointwise / row-wise sub-module (LayerNorm -> Linear -> SwiGLU
transition; a conditioner's ``cat + LayerNorm + Linear + transitions`` prologue; per-layer pair-bias projections; a distogram head)
allocates its intermediates for all N rows at once (a transition holds LN(x) plus up to three ``[.., N, N, 4C]`` temporaries). Evaluated
over row blocks ``[.., r0:r1, :, :]`` and written into ONE preallocated output, the transient is ``rows x N x 4C`` and the result
tensor is allocated once (no ``torch.cat`` second copy). Per-element arithmetic and dtypes are the caller's; only the M (row count) of
each kernel launch changes — see the NUMERICS clause of :mod:`opt_core.mem` for what that means for the class of a line.

API (every function raises :class:`opt_core.mem.MemLeverRefused` on a failed precondition; CUDA OOM propagates unchanged):

    torch_module(lever)                         the lazily imported ``torch`` (refusal names the lever when torch is missing)
    is_oom(exc)                                 True for torch's CUDA out-of-memory exceptions (never convert those into anything)
    pair_ok(x, min_tokens, row_dim, col_dim)    the pair-shape guard an adapter tests BEFORE choosing the lever over the stock call:
                                                a tensor, square on (row_dim, col_dim), N >= min_tokens, no autograd bookkeeping
    rowchunk_apply(fn_rows, n, rows, ...)       out[r0:r1] = fn_rows(r0, r1) for every row block, out allocated once on the first
                                                block (its dtype, or ``out_dtype`` with pieces cast on write); returns out
    rowchunk_tensor(fn, x, rows, ...)           rowchunk_apply over ``x.narrow(row_dim, ...)`` blocks: out[rows] = fn(x[rows])
    rowchunk_concat(fns_rows, n, rows, ...)     L producers x row blocks written into ONE ``[.., N, .., sum(widths)]`` tensor
                                                (layer-major), e.g. the per-layer pair biases of a diffusion transformer
    PatchSet(lever)                             rebinding of class / module attributes with the originals kept: replace, original,
                                                restore, names — the adapter's install record for its report
    FreeList(lever, poison=False)               lifetime-only release of dead tensors (CUDA or CPU) at a seam: register(candidates,
                                                protect) then release(where, ledger) -> bytes; storage aliasing a protected tensor is
                                                skipped; poison=True keeps + NaN-fills instead (the adapter's audit of the 'dead' claim)
    free_storage(t, poison=False)               release (or poison) one tensor's storage -> bytes
"""
from __future__ import annotations

import threading
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import Ledger, MemLeverRefused, row_blocks
from .patchset import PatchSet                            # the framework-free install record lives in mem.patchset; served here too

__all__ = ["torch_module", "is_oom", "pair_ok", "rowchunk_apply", "rowchunk_tensor", "rowchunk_concat", "PatchSet", "FreeList",
           "free_storage"]

_TORCH = None


def torch_module(lever: str = "mem"):
    """``torch``, imported on first use; a missing torch is a named refusal of ``lever`` (this submodule serves a torch engine only)."""
    global _TORCH
    if _TORCH is None:
        try:
            import torch  # noqa: WPS433 — lazy by contract (opt_core imports nothing heavy at module top)
        except Exception as exc:  # noqa: BLE001
            raise MemLeverRefused(lever, f"torch is not importable in this interpreter ({type(exc).__name__}: {exc})") from None
        _TORCH = torch
    return _TORCH


from ..oom import is_oom  # noqa: E402,F401 — the core's one out-of-memory classifier, re-exported under this module's name; a memory lever re-raises these unchanged


def pair_ok(x: Any, min_tokens: int, row_dim: int = -3, col_dim: int = -2) -> bool:
    """The pair-shape guard: ``x`` is a tensor with at least 3 dims, square on (``row_dim``, ``col_dim``) — ``[B, N, N, C]`` and
    ``[N, N, C]`` both pass with the defaults — ``N >= min_tokens``, and no autograd bookkeeping is needed (grad mode off or the tensor
    does not require grad). False means 'the adapter calls the stock attribute and counts a stock call'; it is a declared guard, not a
    refusal."""
    torch = torch_module()
    if not torch.is_tensor(x) or x.dim() < 3:
        return False
    n_r, n_c = int(x.shape[row_dim]), int(x.shape[col_dim])
    if n_r != n_c or n_r < int(min_tokens):
        return False
    if torch.is_grad_enabled() and bool(getattr(x, "requires_grad", False)):
        return False
    return True


def _narrow_len(r0: int, r1: int) -> int:
    if r1 <= r0:
        raise ValueError(f"empty row block ({r0}, {r1})")
    return r1 - r0


def rowchunk_apply(fn_rows: Callable[[int, int], Any], n: int, rows: int, *, row_dim: int = -3, out: Any = None,
                   out_dtype: Any = None, ledger: Optional[Ledger] = None, key: Optional[str] = None, lever: str = "rowchunk") -> Any:
    """Evaluate ``fn_rows(r0, r1)`` for every row block of ``range(n)`` (``rows`` per block) and write each piece into
    ``out.narrow(row_dim, r0, r1 - r0)``. ``out`` is allocated once, on the first piece, as ``torch.empty`` with the piece's shape except
    ``n`` on ``row_dim``, on the piece's device, in the piece's dtype — or in ``out_dtype`` (pieces are cast on write: e.g. one fp32
    tensor assembled from bf16 pieces, so a later per-step ``.float()`` of the whole returns the same storage). A given ``out`` is held
    to that shape / device (dtype may differ: cast on write). Every piece must have the same shape off ``row_dim`` and ``r1 - r0`` on it;
    a mismatch raises :class:`MemLeverRefused` (the producer is not row-independent in shape — not a lever for this primitive).
    ``ledger.count(key)`` once per call when both are given. Returns ``out`` (``n == 0``: ValueError)."""
    torch = torch_module(lever)
    blocks = row_blocks(n, rows)
    if not blocks:
        raise ValueError("rowchunk_apply: n == 0")
    ref_shape: Optional[Tuple[int, ...]] = None
    for r0, r1 in blocks:
        piece = fn_rows(r0, r1)
        if not torch.is_tensor(piece):
            raise MemLeverRefused(lever, f"fn_rows({r0}, {r1}) returned {type(piece).__name__}, not a tensor")
        rd = row_dim % piece.dim()
        if int(piece.shape[rd]) != _narrow_len(r0, r1):
            raise MemLeverRefused(lever, f"fn_rows({r0}, {r1}) returned {tuple(piece.shape)}: dim {row_dim} is not the row block")
        shp = tuple(int(s) for i, s in enumerate(piece.shape) if i != rd)
        if ref_shape is None:
            ref_shape = shp
            full = list(piece.shape)
            full[rd] = int(n)
            if out is None:
                out = torch.empty(tuple(full), dtype=(out_dtype or piece.dtype), device=piece.device)
            else:
                if tuple(int(s) for s in out.shape) != tuple(full) or out.device != piece.device:
                    raise MemLeverRefused(lever, f"out {tuple(out.shape)}@{out.device} does not match pieces {tuple(full)}@{piece.device}")
        elif shp != ref_shape:
            raise MemLeverRefused(lever, f"fn_rows({r0}, {r1}) returned off-row shape {shp}, first block gave {ref_shape}")
        out.narrow(rd, r0, r1 - r0).copy_(piece)
        del piece
    if ledger is not None and key:
        ledger.count(key)
    return out


def rowchunk_tensor(fn: Callable[[Any], Any], x: Any, rows: int, *, row_dim: int = -3, out: Any = None, out_dtype: Any = None,
                    ledger: Optional[Ledger] = None, key: Optional[str] = None, lever: str = "rowchunk") -> Any:
    """``out[rows] = fn(x[rows])`` over row blocks of ``x`` along ``row_dim`` (a view: ``x.narrow``, no copy). ``fn`` sees a block with
    every other dim whole — the form under which a row-independent module returns the rows it would have returned on the full tensor."""
    torch_module(lever)
    rd = row_dim % x.dim()
    n = int(x.shape[rd])
    return rowchunk_apply(lambda r0, r1: fn(x.narrow(rd, r0, r1 - r0)), n, rows, row_dim=rd - x.dim(), out=out, out_dtype=out_dtype,
                          ledger=ledger, key=key, lever=lever)


def rowchunk_concat(fns_rows: Sequence[Callable[[int, int], Any]], n: int, rows: int, *, row_dim: int = -3, cat_dim: int = -1,
                    out_dtype: Any = None, widths: Optional[Sequence[int]] = None, ledger: Optional[Ledger] = None,
                    key: Optional[str] = None, lever: str = "rowchunk") -> Any:
    """``L = len(fns_rows)`` producers, each evaluated per row block, written into ONE tensor whose ``cat_dim`` is the concatenation of
    the producers' widths (producer ``i`` occupies ``[off_i : off_i + width_i]``) — the result ``torch.cat([f(all rows) for f], cat_dim)``
    would give, without the L full-size temporaries or the cat copy. Order is producer-major (all row blocks of producer 0, then 1, ...).
    ``widths``: per-producer widths on ``cat_dim``; when omitted every producer must have the first piece's width (else refused).
    ``out_dtype``: allocate in this dtype and cast pieces on write (e.g. fp32 assembled from autocast bf16 pieces); the ledger records
    ``<key>_dtype=<out>_from_<piece>`` once. Returns the tensor."""
    torch = torch_module(lever)
    L = len(fns_rows)
    if L == 0:
        raise ValueError("rowchunk_concat: no producers")
    blocks = row_blocks(n, rows)
    if not blocks:
        raise ValueError("rowchunk_concat: n == 0")
    out = None
    offs: List[int] = []
    rd = cd = 0
    for li, fn_rows in enumerate(fns_rows):
        for r0, r1 in blocks:
            piece = fn_rows(r0, r1)
            if not torch.is_tensor(piece):
                raise MemLeverRefused(lever, f"producer {li} returned {type(piece).__name__}, not a tensor")
            if out is None:
                rd, cd = row_dim % piece.dim(), cat_dim % piece.dim()
                if rd == cd:
                    raise ValueError("rowchunk_concat: row_dim and cat_dim are the same dim")
                w0 = int(piece.shape[cd])
                ws = [int(w) for w in widths] if widths is not None else [w0] * L
                if len(ws) != L:
                    raise ValueError(f"rowchunk_concat: {len(ws)} widths for {L} producers")
                offs = [sum(ws[:i]) for i in range(L)] + [sum(ws)]
                full = list(piece.shape)
                full[rd], full[cd] = int(n), offs[-1]
                dt = out_dtype or piece.dtype
                out = torch.empty(tuple(full), dtype=dt, device=piece.device)
                if ledger is not None and key:
                    ledger.set(f"{key}_dtype", f"{str(dt).replace('torch.', '')}_from_{str(piece.dtype).replace('torch.', '')}")
            if int(piece.shape[rd]) != _narrow_len(r0, r1):
                raise MemLeverRefused(lever, f"producer {li} rows ({r0},{r1}) returned {tuple(piece.shape)}: dim {row_dim} is not the block")
            if int(piece.shape[cd]) != offs[li + 1] - offs[li]:
                raise MemLeverRefused(lever, f"producer {li} width {int(piece.shape[cd])} != declared {offs[li + 1] - offs[li]}")
            expect = list(out.shape)
            expect[rd], expect[cd] = r1 - r0, int(piece.shape[cd])
            if [int(s) for s in piece.shape] != expect:
                raise MemLeverRefused(lever, f"producer {li} returned {tuple(piece.shape)}, expected {tuple(expect)}")
            out.narrow(rd, r0, r1 - r0).narrow(cd, offs[li], offs[li + 1] - offs[li]).copy_(piece)
            del piece
    if ledger is not None and key:
        ledger.count(key)
    return out


def _storage_ptr(t: Any) -> Optional[int]:
    try:
        return int(t.untyped_storage().data_ptr())
    except Exception:  # noqa: BLE001
        return None


def free_storage(t: Any, poison: bool = False) -> int:
    """Release the storage behind ``t`` (``untyped_storage().resize_(0)``; CUDA or CPU) and return the bytes released (0 if already
    empty). The tensor object stays alive with an empty storage: touching its data afterwards is the caller's bug, so only DEAD tensors go
    here. ``poison=True`` is the audit form of the same claim: the storage is KEPT and overwritten (NaN for floating dtypes, the dtype's
    maximum for integer/bool), so a tensor that was not dead corrupts the run's outputs deterministically — a band / equality run under
    poison is how an adapter's dead-tensor list is reviewed for its engine; returns the bytes poisoned."""
    torch = torch_module("free")
    st = t.untyped_storage()
    nb = int(st.nbytes())
    if nb == 0:
        return 0
    if poison:
        with torch.no_grad():
            if t.dtype.is_floating_point or t.dtype.is_complex:
                t.fill_(float("nan"))
            elif t.dtype == torch.bool:
                t.fill_(True)
            else:
                t.fill_(torch.iinfo(t.dtype).max)
        return nb
    st.resize_(0)
    return nb


class FreeList:
    """Lifetime-only release of dead tensors at a seam (e.g. conditioner outputs that are dead once the sampler returned, released right
    before the confidence head allocates). ``register(candidates, protect)`` walks ``candidates`` (tensors, or dicts / lists / tuples of
    them; CUDA or CPU) and keeps the tensors whose storage does not alias any tensor reachable from ``protect`` (those are skipped and
    counted ``free_skipped_aliased``); ``release(where, ledger)`` frees the registered storages and records ``free_events``, ``free_tensors``,
    ``free_bytes_last``, ``free_bytes_max`` (and ``free_errors`` per storage that could not be released — counted and returned in
    ``errors``, never raised: nothing ran on a wrong value); ``clear()`` drops the registration (call it at the start of every item so a
    failed item never carries pending frees into the next).

    The 'dead' claim itself cannot be checked by construction (a freed tensor read later is undefined behaviour, not an error), so it is
    a REVIEW ITEM of the adapter: the adapter's candidate and ``protect`` lists are cited against the stock forward they wrap, and the kit runs
    its band / equality panel once with ``FreeList(lever, poison=True)`` (the adapter maps its own env knob to it): storages are then kept
    and overwritten with NaN / max instead of released (``free_poisoned`` counted), so a tensor that was still live fails the panel
    deterministically instead of passing silently on reused memory."""

    def __init__(self, lever: str = "free", poison: bool = False):
        self.lever = str(lever)
        self.poison = bool(poison)
        self._lock = threading.Lock()
        self._pending: List[Any] = []
        self.errors: List[str] = []
        self.log: List[Dict[str, Any]] = []

    @staticmethod
    def _walk(obj: Any, visit: Callable[[Any], None]) -> None:
        torch = torch_module()
        if torch.is_tensor(obj):
            visit(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                FreeList._walk(v, visit)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                FreeList._walk(v, visit)

    def register(self, candidates: Any, protect: Any = (), ledger: Optional[Ledger] = None) -> int:
        prot = set()
        self._walk(protect, lambda t: prot.add(_storage_ptr(t)))
        prot.discard(None)
        keep: List[Any] = []
        skipped = [0]

        def _cand(t):
            if not t.is_cuda and t.device.type != "cpu":
                return
            p = _storage_ptr(t)
            if p is None or p in prot:
                skipped[0] += 1
                return
            keep.append(t)

        self._walk(candidates, _cand)
        with self._lock:
            self._pending = keep
        if ledger is not None and skipped[0]:
            ledger.count("free_skipped_aliased", skipped[0])
        return len(keep)

    def release(self, where: str, ledger: Optional[Ledger] = None) -> int:
        with self._lock:
            pend, self._pending = self._pending, []
        if not pend:
            return 0
        nbytes = n = 0
        for t in pend:
            try:
                nb = free_storage(t, poison=self.poison)
                if nb:
                    nbytes += nb
                    n += 1
            except Exception as exc:  # noqa: BLE001
                txt = "".join(traceback.format_exception_only(type(exc), exc)).strip()[:300]
                self.errors.append(f"{where}: {txt}")
                if ledger is not None:
                    ledger.count("free_errors")
        if ledger is not None:
            ledger.count("free_events")
            ledger.count("free_tensors", n)
            ledger.set("free_bytes_last", nbytes)
            ledger.peak("free_bytes_max", nbytes)
            if self.poison:
                ledger.count("free_poisoned", n)
                ledger.set("free_mode", "poison")
        self.log.append({"where": where, "tensors": n, "GiB": round(nbytes / 2 ** 30, 3), "mode": "poison" if self.poison else "release"})
        del pend
        return nbytes

    def clear(self) -> None:
        with self._lock:
            self._pending = []
