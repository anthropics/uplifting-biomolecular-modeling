"""predict_tracks_fast — a drop-in for ``borzoi_pytorch.pytorch_borzoi_helpers.predict_tracks`` (borzoi-pytorch 0.5.1 @
8a05eb15, pytorch_borzoi_helpers.py:4-13) that returns the SAME numpy array (shape, dtype, bytes) from the SAME forwards
(one per fold, batch 1) with the host side of the user's route rewritten.

The stock helper, per fold::

    yh = models[k](seq[None, ...])[:, None, ...].numpy(force=True)[:, :, slices]     # helpers.py:8

copies the FULL (1, 1, 7611, 6144) fp32 array (187,047,936 bytes) to PAGEABLE host memory synchronously — the GPU idles
while the copy drains — and only then slices on the host.  This form:

  * applies the slice ON THE DEVICE before the copy (a slice object → basic indexing; an int list / int array → index_select,
    numpy's fancy-indexing semantics on axis 2 incl. negative indices); any other index form (boolean masks, negative
    steps, ellipsis) falls back to the stock's host slice on the full array (still pinned + overlapped);
  * allocates the (1, n_folds, n, B) result ONCE from a shape-keyed pool of PAGE-LOCKED host buffers (result_pool.ResultPool,
    cudaHostRegister on an exact-size np.empty; refcount recycling — a result the caller still holds is never reused, two live results
    never alias) and copies each fold's device slice on a side CUDA stream STRAIGHT into its slot of that result, the copy of fold k
    overlapped with fold k+1's forward (events order forward → copy; every fold's device tensor is held until its copy event is
    synchronised, so the allocator cannot reuse it early) — no staging buffer, no host memcpy;
  * when the page-locked pool is exhausted (every pooled result still held by the caller) or the index form needs the host slice, falls
    back to pinned STAGING buffers on the side stream + a worker thread landing each fold into its slot of a pre-faulted pooled (or, over
    the cap, fresh) pageable result.

Exactness: a device→host memcpy reorders no bytes; basic/advanced indexing on the device selects the same elements the
host slice selects; the result has the stock's shape and strides (a C-contiguous (1, n_folds, n, B) array viewed .swapaxes(3, 2),
as np.concatenate(...).swapaxes(3, 2) returns).  Hence the returned bytes equal the stock's by construction for every fold.  The
forward itself is untouched (same module call, same autocast context the caller opens).

    from engines.flashzoi.predict_tracks_fast import predict_tracks_fast
    with torch.autocast('cuda', dtype=torch.float16):             # README L27: Flashzoi 'must be run in autocast'
        out = predict_tracks_fast(models, sequence_one_hot, slices)     # == predict_tracks(models, sequence_one_hot, slices)
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["predict_tracks_fast"]

_POOL: dict = {}            # (shape, dtype) -> list of pinned host staging tensors (leased per call, reused across calls) — the fallback landing's buffers


def _pinned(shape, dtype, k: int):
    key = (tuple(shape), dtype)
    lst = _POOL.setdefault(key, [])
    while len(lst) <= k:
        lst.append(torch.empty(tuple(shape), dtype=dtype, pin_memory=True))
    return lst[k]


def _device_index(slices, n: int):
    """(kind, index) — 'slice' for a basic slice with step > 0 (or None), 'gather' for an int sequence/array (negative ints
    normalised like numpy), 'host' for anything else (fall back to the stock host slice on the full array)."""
    if isinstance(slices, slice):
        if slices.step is None or slices.step > 0:
            return "slice", slices
        return "host", None
    if isinstance(slices, (int, np.integer)):
        return "host", None                                          # an int index drops the axis in numpy — keep the stock path
    try:
        idx = np.asarray(slices)
    except Exception:  # noqa: BLE001
        return "host", None
    if idx.ndim == 1 and idx.size and np.issubdtype(idx.dtype, np.integer):
        idx = idx.astype(np.int64)
        if (idx >= n).any() or (idx < -n).any():
            raise IndexError(f"track index out of range for axis of length {n}")
        return "gather", np.where(idx < 0, idx + n, idx)
    return "host", None


_RESULT_POOLS = {}
DIRECT_POOL_KIND = "registered"          # exact-size page-locked result buffers (np.empty + cudaHostRegister)


LAST_CALL = {}      # the per-call stamp: {landing_requested, landing_ran, kind, overlap} — the package reads the landing form that RAN


def _result_pool(kind: str):
    """the shape-keyed pre-faulted result pools (result_pool.ResultPool), one per backing kind, created on first use; cap 4 live results."""
    if kind not in _RESULT_POOLS:
        from .result_pool import ResultPool
        _RESULT_POOLS[kind] = ResultPool(kind=kind, cap=4)
    return _RESULT_POOLS[kind]


HOST_ORDERS = ("stock", "C")     # "stock": the stock's own view (a C-contiguous (1, n_folds, n, B) result seen .swapaxes(3, 2) — bytes AND strides as the
                                 # stock returns them); "C": the same (1, n_folds, B, n) array C-contiguous — each fold reordered ON THE DEVICE before its copy
                                 # (a transpose kernel, no host reorder), for a caller that writes the array out (np.save takes its fast path; the file's
                                 # bytes are the same either way). Values and shape identical in both orders.


def predict_tracks_fast(models, sequence_one_hot, slices, *, copy_threads: int = 4, host_order: str = "stock"):
    """Same signature and returned array as ``pytorch_borzoi_helpers.predict_tracks``; see the module docstring.  The result is a
    pooled PAGE-LOCKED array each fold lands in directly (LAST_CALL['landing_ran'] = 'direct'); when that pool is exhausted or the index
    form needs the host slice, the folds land through pinned staging buffers into a pooled pre-faulted ('pool') or fresh ('staging')
    pageable result.  Same bytes in every form (result_pool.py); `host_order` (HOST_ORDERS) picks the strides: the stock's view, or C order."""
    if host_order not in HOST_ORDERS:
        raise ValueError(f"host_order={host_order!r}: one of {HOST_ORDERS}")
    c_order = host_order == "C"
    if not sequence_one_hot.is_cuda:                                 # CPU models: nothing to overlap — the stock path
        from borzoi_pytorch.pytorch_borzoi_helpers import predict_tracks
        LAST_CALL.update(landing_requested="direct", landing_ran="stock (cpu)", kind=None, overlap=False)
        return predict_tracks(models, sequence_one_hot, slices)
    from concurrent.futures import ThreadPoolExecutor
    main = torch.cuda.current_stream(sequence_one_hot.device)
    side = torch.cuda.Stream(device=sequence_one_hot.device)
    n_folds = len(models)
    pend, futures, out = [], [], None
    out_entry, direct_events = None, []
    LAST_CALL.clear(); LAST_CALL.update(landing_requested="direct", landing_ran=None, kind=None, overlap=True, host_order=host_order)   # the per-call stamp (names the form that RAN)
    direct_ok = True                                                    # decided at fold 0: a pool overflow (every page-locked result still held) or a
                                                                        # host-slicing kind falls back to staging for EVERY fold of this call
    pool = None

    def _land(k, ev, buf, sl):
        ev.synchronize()                                             # this fold's copy has landed in pinned memory
        src = buf.numpy() if sl is None else buf.numpy()[:, :, sl]
        np.copyto(out[:, k:k + 1], src.swapaxes(3, 2) if (c_order and sl is not None) else src)   # the slot of the owned result (first touch + memcpy, off the GPU's path; a host-sliced fold is reordered here)

    try:
        for k in range(n_folds):
            with torch.no_grad():
                y = models[k](sequence_one_hot[None, ...])[:, None, ...]     # (1, 1, T, B) fp32 on the device — the stock's forward
                kind, idx = _device_index(slices, y.shape[2])
                if kind == "slice":
                    ys = y[:, :, idx]
                elif kind == "gather":
                    ys = y.index_select(2, torch.as_tensor(idx, device=y.device))
                else:
                    ys = y
            if c_order and kind != "host":
                ys = ys.transpose(2, 3).contiguous()                    # (1, 1, B, n): the documented array's C order, reordered on the device (a host-sliced fold is reordered at its landing)
                if len(pend) >= 2:                                      # at most two reordered folds in flight: the fold before last has landed before this one is queued
                    pend[-2][0].synchronize(); pend[-2] = (pend[-2][0], None, None)
            sl = None if kind != "host" else slices
            if direct_ok and kind == "host":
                direct_ok = False
            LAST_CALL["kind"] = kind
            if direct_ok:                                                       # the direct landing: no staging buffer, no host memcpy
                from .result_pool import land_direct
                if out_entry is None:
                    out, out_entry = _result_pool(DIRECT_POOL_KIND).take((1, n_folds, ys.shape[2], ys.shape[3]), np.float32)
                    if out_entry["backing"] != "registered":                    # the pool overflowed (every page-locked result still held): stage this call
                        out, out_entry, direct_ok = None, None, False
                if direct_ok:
                    ev_c = land_direct(out_entry, k, ys, side, main); pend.append((ev_c, y, ys)); direct_events.append(ev_c); continue
            if pool is None: pool = ThreadPoolExecutor(max_workers=max(1, copy_threads))
            buf = _pinned(ys.shape, ys.dtype, k)
            ev_f = torch.cuda.Event(); ev_f.record(main)
            with torch.cuda.stream(side):
                side.wait_event(ev_f)
                buf.copy_(ys, non_blocking=True)
                ev_c = torch.cuda.Event(); ev_c.record(side)
            pend.append((ev_c, y, ys))                                # hold the device tensors until the copy has landed
            if out is None:
                n_sel = buf.shape[2] if sl is None else buf.numpy()[:, :, sl].shape[2]
                shape = (1, n_folds, buf.shape[3], n_sel) if (c_order and sl is not None) else ((1, n_folds, n_sel, buf.shape[3]) if not c_order else tuple([1, n_folds] + list(buf.shape[2:])))
                out, out_entry = _result_pool("numpy").take(shape, np.float32)   # a pre-faulted, recycled buffer (over the cap: a fresh np.empty)
            futures.append(pool.submit(_land, k, ev_c, buf, sl))
        if direct_events:
            for ev in direct_events:
                ev.synchronize()                                     # every fold has landed in the pinned result
            pend.clear()
            LAST_CALL["landing_ran"] = "direct"
            return out if c_order else out.swapaxes(3, 2)
        for f in futures:
            f.result()
        pend.clear()
        LAST_CALL["landing_ran"] = "pool" if out_entry is not None else "staging"   # a pooled (pre-faulted) pageable result (over the pool's cap: its fresh np.empty)
        return out if c_order else out.swapaxes(3, 2)                # C order as landed, or the stock's view of its C-contiguous (1, n_folds, n, B) result
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
