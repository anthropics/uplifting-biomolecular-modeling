"""[rf3_cudagraph] Runtime flags for the CUDA-graph diffusion sampler and the roll-out hoist.

Every change introduced by the rf3_cudagraph patch set is gated here and can be switched off at runtime:

  RF3_CUDAGRAPH        "0"      -> stock sampler and stock module code paths (byte-for-byte the original behaviour)
                       "replay" -> pre-drawn RNG + eager (un-captured) replay sampler (bitwise the same draws/arithmetic as stock)
                       "1"      -> pre-drawn RNG + one torch.cuda.CUDAGraph per sampler call (per (design, seed)), DEFAULT

Derived settings (module attributes, no variable): GRAPH_SAFE_OPS -> dense/slice rewrites of the eval-branch masked ops in the atom
attention encoder and the window compaction in the atom transformer (needed for capture; provably bitwise-equal), on iff
RF3_CUDAGRAPH=1; GRAPH_WARMUP = 3 eager warm-up iterations on the capture side stream before capture.

The flags are read at import time; set_mode() updates them at runtime (one process switching the sampler mode, e.g. per rank).
"""
import logging
import os

logger = logging.getLogger(__name__)


def _norm_mode(v: str) -> str:
    v = (v or "").strip().lower()
    if v in ("1", "true", "on", "graph", "cudagraph"):
        return "1"
    if v in ("replay", "eager", "predraw"):
        return "replay"
    return "0"


CUDAGRAPH_MODE = _norm_mode(os.environ.get("RF3_CUDAGRAPH", "1"))
GRAPH_SAFE_OPS = CUDAGRAPH_MODE == "1"   # on iff graph mode (capture needs the rewrites; they are bitwise-equal to the masked ops)
GRAPH_WARMUP = 3                         # eager warm-up iterations on the capture side stream before capture
# [rf3_cudagraph] GRAPH_WARM (lever `warm`, default off; no environment word — set at runtime by set_levers(), which the FPF adapter's
# arm lever step `@L1.warm` calls): the eager warm-up runs GRAPH_WARMUP steps before the FIRST capture of the process and
# GRAPH_WARMUP_REPEAT (1) before every later one. The warm-up reads and writes only the sampler's static clones (re-initialised after
# capture) and consumes no draw, so the roll-out's values cannot depend on the count (bitwise by construction); the one step kept fills
# the hoist cache before capture and runs every kernel's first-call work (autotune, heuristics) outside the capture.
GRAPH_WARM = False
GRAPH_WARMUP_REPEAT = 1
GRAPH_STATS = {"captures": 0, "warmup_steps": 0}   # captures of the process so far / eager warm-up steps spent (the exit tally reads them)
# [xattempt_hoist] RF3_HOIST "1" -> compute the step-invariant sub-graphs of DiffusionModule.forward ONCE per sampler roll-out
# (pair conditioning Z_II, single pre-conditioning, atom-encoder C_L/P_LL, every block's pair bias to_b(ln_0(Z))), reuse for all steps.
# Same modules, same inputs, same kernels -> the cached tensors are the tensors every stock step recomputes. "0" (default) -> stock code path.
HOIST = os.environ.get("RF3_HOIST", "0").strip().lower() in ("1", "true", "on")
HOIST_CACHE = None  # dict while a roll-out is in flight (set/cleared by the sampler), None otherwise -> modules take the stock path
HOIST_CALLS = None  # per-key call counter of the roll-out in flight (1-based denoiser call number for once-per-step keys)
HOIST_T = None      # number of denoiser calls of the roll-out in flight (told by the sampler), used to recheck the last call
HOIST_STATS = {"rollouts": 0, "entries_last": 0}


def hoist_begin(n_calls=None):
    """Called by the sampler at the start of a roll-out: opens a fresh per-roll-out cache iff RF3_HOIST=1."""
    global HOIST_CACHE, HOIST_CALLS, HOIST_T
    HOIST_CACHE = {} if HOIST else None
    HOIST_CALLS = {} if HOIST else None
    HOIST_T = n_calls


def hoist_end():
    """Called by the sampler at the end of a roll-out: drops the cache (after a device sync so no in-flight kernel reads freed blocks)."""
    global HOIST_CACHE, HOIST_CALLS, HOIST_T
    if HOIST_CACHE is not None:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        HOIST_STATS["rollouts"] += 1
        HOIST_STATS["entries_last"] = len(HOIST_CACHE)
    HOIST_CACHE = None
    HOIST_CALLS = None
    HOIST_T = None


def hoist_get(key, fn):
    """cache[key] if present else fn() (stored). With no roll-out in flight (HOIST_CACHE None) this is just fn()."""
    c = HOIST_CACHE
    if c is None:
        return fn()
    n = HOIST_CALLS.get(key, 0) + 1
    HOIST_CALLS[key] = n
    v = c.get(key)
    if v is None:
        v = fn()
        c[key] = v
    return v

def warmup_steps() -> int:
    """[rf3_cudagraph] Eager warm-up steps before the next capture: GRAPH_WARMUP, or GRAPH_WARMUP_REPEAT once this process has captured
    under GRAPH_WARM (a later capture of the process — any shape — has every kernel path already run eagerly at least once)."""
    if GRAPH_WARM and GRAPH_STATS["captures"] > 0:
        return int(GRAPH_WARMUP_REPEAT)
    return int(GRAPH_WARMUP)


def note_capture(n_warmup: int) -> None:
    """[rf3_cudagraph] Called by the sampler after each capture (the count warmup_steps() keys on)."""
    GRAPH_STATS["captures"] += 1
    GRAPH_STATS["warmup_steps"] += int(n_warmup)


def set_levers(warm=None) -> None:
    """[rf3_cudagraph] Switch the roll-out's fixed-cost levers at runtime (None = leave unchanged)."""
    global GRAPH_WARM
    if warm is not None:
        GRAPH_WARM = bool(warm)


# Telemetry written by the sampler (read by the drivers; never used by the model).
LAST_CAPTURE: dict = {}
STEP_CALLBACK = None  # optional callable(step_index) invoked before every sampler step (profiling hooks)


def set_mode(mode: str, safe_ops=None, warmup=None, hoist=None) -> None:
    """Switch the sampler mode at runtime. safe_ops=None: on iff graph mode."""
    global CUDAGRAPH_MODE, GRAPH_SAFE_OPS, GRAPH_WARMUP, HOIST
    if hoist is not None:
        HOIST = bool(hoist)
    CUDAGRAPH_MODE = _norm_mode(mode)
    GRAPH_SAFE_OPS = (CUDAGRAPH_MODE == "1") if safe_ops is None else bool(safe_ops)
    if warmup is not None:
        GRAPH_WARMUP = int(warmup)


def describe() -> dict:
    return dict(
        RF3_CUDAGRAPH=CUDAGRAPH_MODE,
        RF3_GRAPH_SAFE_OPS=GRAPH_SAFE_OPS,
        RF3_CUDAGRAPH_WARMUP=GRAPH_WARMUP,
        RF3_HOIST=HOIST,
    )


def levers_state() -> dict:
    """[rf3_cudagraph] The runtime levers (set after import by set_levers; describe() keeps the switches' line as it was)."""
    return dict(warm=GRAPH_WARM, warmup_repeat=int(GRAPH_WARMUP_REPEAT), **GRAPH_STATS)
