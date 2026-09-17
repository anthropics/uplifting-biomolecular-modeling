"""tp_probe — box-local DIAGNOSTICS for the row-sharded line (``--n_gpu P > 1`` only; nothing here is imported at ``n_gpu = 1``).

Two env-guarded instruments, both OFF unless their variable is set (the line exports neither):

  * ``ODDE_TP_PROF_STACK=<n_blocks>:<k>`` — wrap the k-th sharded pair-stack call with ``n_blocks`` blocks on this rank in
    ``torch.profiler`` (CUDA + CPU activities) and print, on rank 0, the kernel table by self CUDA time plus bucket sums
    (nccl / gemm / flash_triattn / triton_other / layer_norm / elementwise / memcpy / other) and the call's wall:
    ``[opendde-opt tp] PROF …`` lines on stderr. One call per process; every other call runs untouched.
  * ``ODDE_TP_MEMSTATS_RESET=1`` — ``tp._memstats`` resets the allocator's peak counters after each census mark, so every
    ``MEMSTATS`` line reads the peak SINCE the previous mark (``since=<previous stage>``) instead of the running maximum.

Instrumentation only: a failure inside the probe prints one line and the run continues un-profiled."""
from __future__ import annotations

import os
import re
import sys
import time

TAG = "opendde-opt tp"
_STATE = {"stack_calls": {}, "done": False}

_BUCKETS = (
    ("nccl", re.compile(r"nccl", re.I)),
    ("flash_triattn", re.compile(r"flash|triatt|_attn_fwd|k2b", re.I)),
    ("gemm", re.compile(r"gemm|cutlass|cublas|xmma|wgmma|matmul|bmm|Kernel3|sm90_|sm80_|ampere_|hopper_", re.I)),
    ("layer_norm", re.compile(r"layer_?norm|LayerNorm|fastln|ln_", re.I)),
    ("memcpy", re.compile(r"memcpy|Memcpy|Memset|memset|copy_", re.I)),
    ("triton_other", re.compile(r"triton|_kernel_\d|silu|swiglu|gate", re.I)),
    ("elementwise", re.compile(r"elementwise|vectorized|unrolled|reduce_kernel|softmax|sigmoid|CatArray|index|gather|scatter|masked", re.I)),
)


def want_stack(n_blocks: int) -> bool:
    """True when THIS call (the next call with ``n_blocks`` blocks) is the one ``ODDE_TP_PROF_STACK`` names."""
    spec = os.environ.get("ODDE_TP_PROF_STACK", "").strip()
    if not spec or _STATE["done"]:
        return False
    try:
        nb, k = (int(x) for x in spec.split(":"))
    except ValueError:
        print(f"[{TAG}] PROF refused: ODDE_TP_PROF_STACK={spec!r} (want <n_blocks>:<k>)", file=sys.stderr, flush=True)
        _STATE["done"] = True
        return False
    c = _STATE["stack_calls"].get(int(n_blocks), 0) + 1
    _STATE["stack_calls"][int(n_blocks)] = c
    return int(n_blocks) == nb and c == k


class StackProfile:
    """Context manager around one pair-stack call: torch.profiler + wall; ``report()`` prints the tables (rank 0 only prints the long table)."""

    def __init__(self, n_blocks: int, N: int, R: int, rank):
        self.n_blocks, self.N, self.R, self.rank = int(n_blocks), int(N), int(R), rank
        self.prof = None
        self.t0 = self.wall = None

    def __enter__(self):
        import torch
        try:
            from torch.profiler import profile, ProfilerActivity
            self.with_stack = os.environ.get("ODDE_TP_PROF_WITH_STACK", "") == "1"
            self.prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=self.with_stack, with_stack=self.with_stack)
            torch.cuda.synchronize()
            self.prof.__enter__()
        except Exception as e:  # noqa: BLE001
            print(f"[{TAG}] PROF unavailable: {type(e).__name__}: {str(e)[:160]}", file=sys.stderr, flush=True)
            self.prof = None
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        import torch
        torch.cuda.synchronize()
        self.wall = time.perf_counter() - self.t0
        _STATE["done"] = True
        if self.prof is not None:
            try:
                self.prof.__exit__(None, None, None)
                self.report()
            except Exception as e:  # noqa: BLE001
                print(f"[{TAG}] PROF report failed: {type(e).__name__}: {str(e)[:160]}", file=sys.stderr, flush=True)
        return False

    def report(self):
        ka = self.prof.key_averages()
        rows = []
        for ev in ka:
            cu = getattr(ev, "self_device_time_total", None)
            if cu is None:
                cu = getattr(ev, "self_cuda_time_total", 0)
            dtype = str(getattr(ev, "device_type", ""))
            if cu <= 0:
                continue
            # keep device-side kernels only (DeviceType.CUDA); CPU ops carry self_cuda 0 in recent torch, but guard by name too
            if "CUDA" not in dtype.upper() and not ev.key.startswith(("void ", "ncclDevKernel", "ncclKernel", "triton", "Memcpy", "Memset")) and "kernel" not in ev.key.lower():
                continue
            rows.append((ev.key, cu / 1e3, ev.count))          # ms
        rows.sort(key=lambda r: -r[1])
        total = sum(r[1] for r in rows)
        buckets = {}
        for key, ms, n in rows:
            b = "other"
            for name, rx in _BUCKETS:
                if rx.search(key):
                    b = name
                    break
            agg = buckets.setdefault(b, [0.0, 0])
            agg[0] += ms; agg[1] += n
        hdr = f"[{TAG}] PROF rank={self.rank} stack={self.n_blocks}x{self.N} rows={self.R} wall_s={self.wall:.3f} per_block_s={self.wall / max(1, self.n_blocks):.4f} kernels_ms_total={total:.1f}"
        print(hdr, file=sys.stderr, flush=True)
        print(f"[{TAG}] PROF buckets " + " ".join(f"{b}={v[0]:.1f}ms/{v[1]}calls({100.0 * v[0] / max(total, 1e-9):.1f}%)" for b, v in sorted(buckets.items(), key=lambda kv: -kv[1][0])), file=sys.stderr, flush=True)
        if self.rank in (0, None):
            for key, ms, n in rows[:45]:
                print(f"[{TAG}] PROF kernel ms={ms:9.1f} n={n:7d} {key[:150]}", file=sys.stderr, flush=True)
            if getattr(self, "with_stack", False):                     # attribute the device time of aten copy/contiguous/clone/to ops to python call sites
                try:
                    ks = self.prof.key_averages(group_by_stack_n=8)
                    sites = []
                    for ev in ks:
                        if not re.search(r"aten::(copy_|contiguous|clone|to|_to_copy|permute|transpose)|direct_copy", ev.key):
                            continue
                        cu = getattr(ev, "device_time_total", None)
                        if cu is None:
                            cu = getattr(ev, "cuda_time_total", 0)
                        if cu <= 0:
                            continue
                        frames = [f for f in (ev.stack or []) if (".py" in f and "torch/" not in f)][:4]
                        sites.append((cu / 1e3, ev.count, ev.key, " <- ".join(frames)))
                    agg = {}
                    for ms, n, key, fr in sites:
                        a = agg.setdefault(fr, [0.0, 0, key]); a[0] += ms; a[1] += n
                    for fr, (ms, n, key) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:25]:
                        print(f"[{TAG}] PROF copysite ms={ms:9.1f} n={n:7d} op={key[:28]} site={fr[:420]}", file=sys.stderr, flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[{TAG}] PROF copysite failed: {type(e).__name__}: {str(e)[:160]}", file=sys.stderr, flush=True)


def stack_profile(n_blocks: int, N: int, R: int, rank):
    """A ``StackProfile`` when this call is the named one, else None."""
    return StackProfile(n_blocks, N, R, rank) if want_stack(n_blocks) else None


def memstats_reset_wanted() -> bool:
    return os.environ.get("ODDE_TP_MEMSTATS_RESET", "").strip() == "1"
