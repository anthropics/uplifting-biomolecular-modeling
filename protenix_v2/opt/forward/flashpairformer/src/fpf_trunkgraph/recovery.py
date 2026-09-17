"""fpf_trunkgraph.recovery — the ONE failed-capture recovery path of the FlashPairformer bundle.

    from fpf_trunkgraph.recovery import recover_after_failed_capture
    rep = recover_after_failed_capture(graph=g, capture_began=True, capture_ended=False)   # from the `except` of ANY failed capture
    #   ALWAYS capture into an explicit pool (pool = torch.cuda.graph_pool_handle(); g.capture_begin(pool=pool)) and pass pool= here:
    #   after an invalidated capture graph.pool() raises ('without a preceding successful capture'), and without the id the allocator's
    #   captures_underway entry cannot be popped -> the next torch.cuda.empty_cache() hits the INTERNAL ASSERT (observed on torch 2.7.1).
    # rep = {"dangling_capture_ended", "pool": {...}, "generator_recovered", "allocator_healthy", "rng_healthy"}
    # raises RuntimeError if the default CUDA generator is still unusable (raise_if_unhealthy=True, default): the process must not continue,
    # its outputs would no longer be the stock function.

What it does (numerics-free; the stock RNG stream is NOT consumed):
  (1) if the current stream is still capturing (capture_end never ran), end the dangling capture into a throw-away graph;
  (2) take the default CUDA generator out of capture mode: a capture_begin that raised after the generator's capture prologue leaves it flagged
      'capturing' and every later eager RNG call fails with "Offset increment outside graph capture".  A trivial SUCCESSFUL capture (one memset+add,
      no RNG op inside => the philox offset is not advanced) runs the epilogue and clears the flag (same routine as infopt_graphs.core);
  (2b) allocator epilogue of the failed capture (torch 2.7.1): endAllocateToPool(dev, pool) if still underway + releasePool
      exactly once (else the next torch.cuda.empty_cache() dies with 'captures_underway.empty() INTERNAL ASSERT FAILED'); pool id from graph.pool()
      or the explicit `pool=` argument; never for a capture whose capture_end completed;
  (3) health probes: allocator (synchronize + empty_cache + alloc/free) and RNG (save CPU+CUDA generator states -> torch.rand(1,'cuda') -> restore:
      the stock RNG stream is not consumed); if unhealthy, (2)+(3) retried once;
  (4) return the report; raise if still unhealthy and raise_if_unhealthy (the process must not continue producing 'stock' outputs).
No dependency on the rest of fpf_trunkgraph (importable stand-alone by fpf_stackgraph)."""
from __future__ import annotations
import sys
import torch

__all__ = ["recover_after_failed_capture", "release_failed_capture_pool", "end_dangling_capture", "reset_generator_capture_state", "rng_health_probe", "allocator_health_probe", "clear_pending_cuda_error"]


def clear_pending_cuda_error() -> str:
    """cudaStreamEndCapture on an invalidated capture returns an error that torch raises; the runtime's 'last error' slot may still hold a
    (non-sticky) capture error which makes the NEXT unrelated runtime call fail.  torch exposes no cudaGetLastError binding, so we issue cheap runtime
    calls that read-and-reset it (synchronize raises once, then the state is clean).  Returns a short status string."""
    msgs = []
    for i in range(3):
        try:
            torch.cuda.synchronize()
            break
        except Exception as e:  # noqa
            msgs.append(str(e).splitlines()[0][:80])
    try:
        torch.cuda.current_stream().query()
    except Exception as e:  # noqa
        msgs.append("query:" + str(e)[:60])
    return "clean" if not msgs else ("cleared:" + " | ".join(msgs))


def end_dangling_capture() -> bool:
    """If the CURRENT stream is still in capture mode (an exception skipped capture_end), end it into a throw-away graph. Returns True if it did something."""
    try:
        if torch.cuda.is_current_stream_capturing():
            g = torch.cuda.CUDAGraph()
            try:
                g.capture_end()
            except Exception:
                pass
            del g
            return True
    except Exception:
        pass
    return False


def reset_generator_capture_state() -> bool:
    """Trivial successful capture on a fresh side stream -> runs the CUDA generator's capture epilogue (clears the 'capturing' flag). Numerics-free."""
    try:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            torch.cuda.synchronize()
            g.capture_begin()
            x = torch.zeros(1, device="cuda"); x.add_(1)
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); del g
        return True
    except Exception:  # noqa
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        return False


LAST_PROBE_ERRORS = {}


def rng_health_probe() -> bool:
    """Prove the default CUDA generator serves eager draws, WITHOUT consuming the stock RNG stream (states saved and restored around one draw)."""
    try:
        dev = torch.cuda.current_device()
        cpu_state = torch.random.get_rng_state(); cuda_state = torch.cuda.get_rng_state(dev)
        try:
            torch.rand(1, device="cuda"); torch.cuda.synchronize()
        finally:
            torch.cuda.set_rng_state(cuda_state, dev); torch.random.set_rng_state(cpu_state)
        return True
    except Exception as e:  # noqa
        LAST_PROBE_ERRORS["rng"] = f"{type(e).__name__}: {str(e)[:200]}"
        return False


def _private(name_candidates):
    for n in name_candidates:
        f = getattr(torch._C, n, None)
        if f is not None:
            return f
    return None


def release_failed_capture_pool(graph=None, pool=None, device=None, capture_began=None, capture_ended=False) -> dict:
    """Allocator bookkeeping that CUDAGraph.capture_end() would have done, for a capture that FAILED:
    capture_begin() calls beginAllocateToPool(dev, pool_id) (+ use_count++ on the pool) BEFORE cudaStreamBeginCapture; capture_end() calls
    endAllocateToPool only after a successful cudaStreamEndCapture, and the graph's destructor releases the pool only if a graph exists.  So after a
    failed capture (either pool style: private pool or an explicit graph_pool_handle shared across graphs) the device allocator still lists the
    capture as underway -> the next torch.cuda.empty_cache() trips TORCH_INTERNAL_ASSERT(captures_underway.empty()), and the pool's use_count is never
    decremented.  Fix: endAllocateToPool(dev, pool_id) if still underway, then releasePool(dev, pool_id) exactly once — but ONLY when the capture really
    did not complete (capture_ended=False), never for a graph whose capture_end succeeded (its destructor releases; a second release would free a
    shared pool under live graphs).
      graph : the failed torch.cuda.CUDAGraph (pool id read via graph.pool());  pool : explicit pool id (tuple) if you have no graph object;
      capture_began : True if capture_begin() returned (None = unknown);  capture_ended : True if capture_end() returned without raising."""
    rep = {"pool_id": None, "end_allocate_to_pool": None, "release_pool": None}
    if capture_ended:
        rep["skipped"] = "capture_end completed: graph owns the pool bookkeeping"; return rep
    pid = None
    if pool is not None:
        try:
            pid = tuple(pool)
        except TypeError:                     # torch returns an opaque _MemPoolHandle / tuple depending on version
            pid = pool
    elif graph is not None:
        try:
            pid = tuple(graph.pool())
        except Exception as e:  # noqa
            rep["pool_err"] = repr(e)[:120]
    if pid is None or (isinstance(pid, tuple) and len(pid) == 2 and pid[0] == 0 and pid[1] == 0):
        rep["skipped"] = "no pool id (pass pool= the handle given to capture_begin; graph.pool() is unavailable after a failed capture)"; return rep
    rep["pool_id"] = str(pid)
    dev = torch.cuda.current_device() if device is None else int(device)
    f_end = _private(("_cuda_endAllocateCurrentStreamToPool", "_cuda_endAllocateToPool"))
    f_rel = _private(("_cuda_releasePool",))
    was_underway = False
    if f_end is not None:
        try:
            f_end(dev, pid); was_underway = True; rep["end_allocate_to_pool"] = "done"
        except Exception as e:  # noqa  ("not currently recording to mempool_id": capture_end already ended it, or beginAllocateToPool never ran)
            rep["end_allocate_to_pool"] = f"not underway ({str(e)[:80]})"
    else:
        rep["end_allocate_to_pool"] = "unavailable in this torch"
    # release exactly once iff beginAllocateToPool ran (use_count was incremented) and no graph exists to release it later
    do_release = bool(capture_began) or was_underway
    if f_rel is not None and do_release:
        try:
            f_rel(dev, pid); rep["release_pool"] = "done"
        except Exception as e:  # noqa
            rep["release_pool"] = f"error ({str(e)[:80]})"
    elif f_rel is None:
        rep["release_pool"] = "unavailable in this torch"
    else:
        rep["release_pool"] = "skipped (capture never began allocating to the pool)"
    return rep


def allocator_health_probe() -> bool:
    """empty_cache() + a small alloc/free must work (this is what trips the INTERNAL ASSERT when a failed capture is left 'underway')."""
    try:
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        x = torch.empty(1024, device="cuda"); del x
        torch.cuda.empty_cache()
        return True
    except Exception as e:  # noqa
        LAST_PROBE_ERRORS["allocator"] = f"{type(e).__name__}: {str(e)[:200]}"
        return False


def recover_after_failed_capture(graph=None, pool=None, device=None, capture_began=None, capture_ended=False,
                                 raise_if_unhealthy: bool = True, log=sys.stderr) -> dict:
    rep = {"dangling_capture_ended": end_dangling_capture()}
    rep["cuda_error_state"] = clear_pending_cuda_error()
    rep["pool"] = release_failed_capture_pool(graph=graph, pool=pool, device=device, capture_began=capture_began, capture_ended=capture_ended)
    rep["generator_recovered"] = reset_generator_capture_state()
    rep["allocator_healthy"] = allocator_health_probe()
    rep["rng_healthy"] = rng_health_probe()
    if not (rep["rng_healthy"] and rep["allocator_healthy"]):
        rep["generator_recovered"] = reset_generator_capture_state() or rep["generator_recovered"]
        rep["allocator_healthy"] = allocator_health_probe(); rep["rng_healthy"] = rng_health_probe()
    rep["healthy"] = bool(rep["rng_healthy"] and rep["allocator_healthy"])
    if not rep["healthy"]:
        rep["probe_errors"] = dict(LAST_PROBE_ERRORS)
    if log is not None:
        print(f"[fpf recovery] after failed capture: {rep}", file=log, flush=True)
    if raise_if_unhealthy and not rep["healthy"]:
        raise RuntimeError(f"CUDA generator/allocator unusable after a failed graph capture ({rep}); refusing to continue — outputs would not be the stock function")
    return rep
