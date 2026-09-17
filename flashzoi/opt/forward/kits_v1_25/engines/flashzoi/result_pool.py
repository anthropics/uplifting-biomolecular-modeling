"""RESULT LANDING for predict_tracks_fast:

  (1) ResultPool — a SHAPE-KEYED pool of PRE-FAULTED host result buffers with REFCOUNT recycling.  A buffer is free when nobody but the
      pool references it (sys.getrefcount on the backing object); a result the user still holds is never reused — the pool hands out
      another buffer (allocating one if under the cap) — so the documented drop-in semantics ("the call returns a new array") hold
      exactly: two live results never alias.  Over the cap the call falls back to a fresh np.empty.
  (2) land_direct — the fold's device tensor copied on the SIDE STREAM straight into the owned result's slot (a page-locked, pool-owned
      buffer viewed as the numpy result; the slot [:, k] of a C-contiguous (1, n_folds, n, B) buffer is contiguous, so the copy
      is one cudaMemcpyAsync into page-locked memory) — no staging buffer, no host memcpy; the landing IS the D2H.

Every form returns the SAME bytes and strides (np.empty (1, n_folds, n_sel, B) float32, viewed .swapaxes(3, 2)).  numpy/torch only; Linux."""
from __future__ import annotations
import sys, threading
import numpy as np

try:
    import torch
except Exception:                                                     # the CPU test runs without torch
    torch = None


class _Registered:
    """unregisters the page-locked range when the pool drops the buffer (never while a user view is alive: the pool holds it)."""
    __slots__ = ("arr",)
    def __init__(self, arr): self.arr = arr
    def __del__(self):
        try: torch.cuda.cudart().cudaHostUnregister(self.arr.ctypes.data)
        except Exception: pass


# ------------------------------------------------------------------------------------------------------ (1) the pool
class ResultPool:
    """shape-keyed pool of pre-faulted result buffers; refcount recycling; capped.

    kind: 'numpy'      -> np.empty + first touch (pre-faulted, pageable)             [the landing is np.copyto from the pinned staging buffer]
          'registered' -> np.empty of the EXACT size, page-locked by cudaHostRegister [enables (2): the device->host copy lands in the slot
                          directly, at the exact bound of 748 MB per all-tracks buffer (torch's own pinned allocator would round the block up to 1 GiB)]
    take(shape, dtype) -> (np.ndarray (1, n_folds, n, B) C-contiguous, entry); the ndarray is the result's backing (view it .swapaxes(3, 2)).
    A buffer is FREE when the refcount of its backing object equals the pool's baseline (the pool's own references only); a result the
    user still holds (any view: the ndarray, its .swapaxes view, a slice) keeps the backing alive and busy.  Over `cap` live buffers per
    key the call gets a plain fresh np.empty — never a wait, never an alias."""

    def __init__(self, kind: str = "numpy", cap: int = 4):
        assert kind in ("numpy", "registered"), kind
        self.kind, self.cap = kind, cap; self._pool = {}; self._lock = threading.Lock(); self.stats = dict(hits=0, allocs=0, overflow=0, checks=0)

    def _new(self, shape, dtype):
        if self.kind == "registered":                                     # exact-size page-locked buffer: np.empty + cudaHostRegister (torch's pinned
            assert torch is not None, "registered pool needs torch"        # allocator rounds every block UP to a power of two: 748 MB -> 1 GiB)
            arr = np.empty(tuple(shape), dtype=dtype); arr.reshape(-1)[::1024] = 0
            rc = torch.cuda.cudart().cudaHostRegister(arr.ctypes.data, arr.nbytes, 0)   # cudaHostRegisterDefault
            assert int(rc) == 0, f"cudaHostRegister failed: {rc}"
            return dict(arr=arr, holder=_Registered(arr), backing="registered")
        arr = np.empty(tuple(shape), dtype=dtype); arr.reshape(-1)[::1024] = 0; return dict(arr=arr, holder=None, backing="numpy-prefaulted")

    @staticmethod
    def _owner(entry):
        # numpy collapses a view's .base to the ROOT ndarray of the chain (the first ndarray whose base is not an ndarray: np.empty's own
        # array) -- so the user's views (swapaxes, slices) reference THAT ndarray; its refcount is the pool's evidence
        o = entry["arr"]
        while isinstance(getattr(o, "base", None), np.ndarray): o = o.base
        return o

    @classmethod
    def _refs(cls, entry) -> int: return sys.getrefcount(cls._owner(entry))

    def take(self, shape, dtype=np.float32):
        key = (tuple(int(s) for s in shape), np.dtype(dtype).str)
        with self._lock:
            lst = self._pool.setdefault(key, [])
            for e in lst:
                self.stats["checks"] += 1
                if self._refs(e) <= e["baseline"]:                       # nobody but the pool holds it -> recycle (no zeroing: every byte is overwritten by the landing)
                    self.stats["hits"] += 1; return e["arr"], e
            if len(lst) >= self.cap:                                        # every buffer is still held by the user -> the plain fresh form
                self.stats["overflow"] += 1; arr = np.empty(key[0], dtype=dtype); return arr, dict(arr=arr, holder=None, backing="fresh-np.empty(overflow)", baseline=None)
            e = self._new(key[0], dtype); lst.append(e); e["baseline"] = self._refs(e); self.stats["allocs"] += 1
            return e["arr"], e

    def live(self, shape, dtype=np.float32) -> dict:
        key = (tuple(int(s) for s in shape), np.dtype(dtype).str); lst = self._pool.get(key, [])
        return dict(buffers=len(lst), held=sum(1 for e in lst if self._refs(e) > e["baseline"]), backing=[e["backing"] for e in lst])


# ------------------------------------------------------------------------------------------------------ (2) the direct landing
def land_direct(out_entry, k: int, ys, side, main):
    """copy the fold's device tensor ys (1, 1, n, B) straight into slot k of the pooled PAGE-LOCKED result on the side stream; returns the
    event that marks the landing (the caller waits on it before returning the array)."""
    assert torch is not None and out_entry["backing"] == "registered", out_entry.get("backing")
    t = torch.from_numpy(out_entry["arr"])                                # (1, n_folds, n, B) page-locked; slot [:, k:k+1] contiguous
    ev_f = torch.cuda.Event(); ev_f.record(main)
    with torch.cuda.stream(side):
        side.wait_event(ev_f)
        t[:, k:k + 1].copy_(ys, non_blocking=True)                        # one cudaMemcpyAsync D2H into page-locked memory: the landing IS the copy
        ev_c = torch.cuda.Event(); ev_c.record(side)
    return ev_c
