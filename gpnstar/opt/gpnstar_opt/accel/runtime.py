"""CUDA-graph runner, timing / memory / identity utilities and fast-mode toggles."""

from __future__ import annotations

import contextlib
import statistics
from typing import Any, Callable

import torch

from .inputs import llr_scores

# --------------------------------------------------------------------------------------
# CUDA graphs
# --------------------------------------------------------------------------------------


class GraphRunner:
    """Capture `model(**batch).logits` once for a fixed batch shape and replay it.
    Inputs are copied into static buffers; the returned tensor is the static output buffer
    (clone it if you need to keep it across replays)."""

    def __init__(self, model, example_batch: dict[str, torch.Tensor], *, n_warmup: int = 3, autocast_dtype=None, pool=None, on_mismatch: str = "raise",
                 check_overflow: bool = True):
        """on_mismatch: what to do when a batch does not match the captured shapes/dtypes or carries a different
        target_species than the captured one -- "raise" (default) or "eager" (run the patched model eagerly).
        check_overflow (default True): after every replay of a static-capacity de-dup graph, read the plan's device overflow
        flag (one host sync) and, if a clade capacity was exceeded, RECOMPUTE that batch eagerly through the stock K/V
        projections (exact) -- so flagged logits can never be consumed by a caller that forgets to poll static_overflow().
        `self.n_overflow_recomputes` counts these events.  check_overflow=False restores the asynchronous behaviour (then the
        caller MUST poll static_overflow() itself)."""
        assert on_mismatch in ("raise", "eager")
        self.model = model
        self.autocast_dtype = autocast_dtype
        self.on_mismatch = on_mismatch
        self.check_overflow = bool(check_overflow)
        self.n_overflow_recomputes = 0
        self.static = {k: v.clone() for k, v in example_batch.items()}
        self.signature = {k: (tuple(v.shape), v.dtype) for k, v in self.static.items()}
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warmup):
                self.out = self._fwd()
            # exact-mode runtime validation (patches.patch_unified_kv) must have settled before capture
            st = getattr(getattr(model, "model", model), "_exact_state", None)
            extra = 0
            while st is not None and getattr(st, "kv_fallback_full", False) and extra < 8:
                self.out = self._fwd()
                extra += 1
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool):
            self.out = self._fwd()
        torch.cuda.synchronize()

    def _fwd(self):
        with torch.inference_mode():
            if self.autocast_dtype is not None:
                with torch.autocast("cuda", dtype=self.autocast_dtype):
                    return self.model(**self.static).logits
            return self.model(**self.static).logits

    def mismatch(self, batch: dict[str, torch.Tensor]) -> str | None:
        """None if `batch` can be replayed through the captured graph, else the reason."""
        if set(batch) != set(self.static):
            return f"keys {sorted(batch)} != captured {sorted(self.static)}"
        for k, v in batch.items():
            if (tuple(v.shape), v.dtype) != self.signature[k]:
                return f"{k}: shape/dtype {(tuple(v.shape), v.dtype)} != captured {self.signature[k]}"
        ts = batch.get("target_species")
        if ts is not None and not torch.equal(ts.to(self.static["target_species"].device), self.static["target_species"]):
            return "target_species differs from the captured batch (the graph baked in its species constants)"
        return None

    def __call__(self, batch: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        if batch is not None:
            why = self.mismatch(batch)
            if why is not None:
                if self.on_mismatch == "eager":
                    return self._eager(batch)
                raise ValueError(f"GraphRunner: batch cannot be replayed through the captured graph ({why}); "
                                 "capture a runner for this shape or construct with on_mismatch='eager'.")
            for k, v in batch.items():
                if v.data_ptr() != self.static[k].data_ptr():
                    self.static[k].copy_(v, non_blocking=True)
        self.graph.replay()
        if self.check_overflow:
            flag = self._overflow_flag()
            if flag is not None and bool(flag.item()):  # host sync; capacity exceeded -> these logits are NOT valid
                flag.zero_()
                self.n_overflow_recomputes += 1
                return self._eager(batch if batch is not None else self.static, force_stock=True)
        return self.out

    def _overflow_flag(self):
        if not hasattr(self, "_oflag"):
            from .patches import _overflow_flag_tensor
            B, L = self.signature["input_ids"][0][:2]
            try:
                self._oflag = _overflow_flag_tensor(self.model, B, L)
            except Exception:  # model without exact state (e.g. stock model captured for timing)
                self._oflag = None
        return self._oflag

    def _eager(self, batch, force_stock: bool = False):
        core = getattr(self.model, "model", self.model)
        st = getattr(core, "_exact_state", None)
        if force_stock and st is not None:
            st.force_stock_once = True  # one forward on the stock K/V projections (exact; no capacity involved)
        with torch.inference_mode():
            if self.autocast_dtype is not None:
                with torch.autocast("cuda", dtype=self.autocast_dtype):
                    return self.model(**batch).logits.clone()
            return self.model(**batch).logits.clone()


# --------------------------------------------------------------------------------------
# forward callables
# --------------------------------------------------------------------------------------


def stock_forward(model, *, inference_mode: bool = False) -> Callable[[dict], torch.Tensor]:
    """Stock forward: eval + no_grad (what transformers' Trainer.prediction_step does) or, if inference_mode=True,
    eval + torch.inference_mode.  Benchmarks must use the SAME context for stock and for eager kit variants."""
    model.eval()
    grad_ctx = torch.inference_mode if inference_mode else torch.no_grad

    def f(batch):
        with grad_ctx():
            return model(**batch).logits

    return f


def eager_forward(model, *, inference_mode: bool = True, autocast_dtype=None) -> Callable[[dict], torch.Tensor]:
    model.eval()
    grad_ctx = torch.inference_mode if inference_mode else torch.no_grad

    def f(batch):
        with grad_ctx():
            if autocast_dtype is not None:
                with torch.autocast("cuda", dtype=autocast_dtype):
                    return model(**batch).logits
            return model(**batch).logits

    return f


# --------------------------------------------------------------------------------------
# timing / memory
# --------------------------------------------------------------------------------------


def cuda_time(fn: Callable, batch, *, iters: int = 30, warmup: int = 10) -> dict[str, Any]:
    """Per-iteration latency with CUDA events (ms). Warm-up excluded."""
    for _ in range(warmup):
        fn(batch)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn(batch)
        ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    q = statistics.quantiles(times, n=4, method="inclusive")
    return {
        "median_ms": statistics.median(times),
        "p25_ms": q[0],
        "p75_ms": q[2],
        "min_ms": min(times),
        "max_ms": max(times),
        "mean_ms": statistics.fmean(times),
        "iters": iters,
        "warmup": warmup,
        "all_ms": times,
    }


def wall_time(fn: Callable, batch, *, iters: int = 30, warmup: int = 10) -> float:
    """End-to-end ms/iter including host overhead (sync at the end only)."""
    import time

    for _ in range(warmup):
        fn(batch)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(batch)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


def peak_memory(fn: Callable, batch) -> dict[str, float]:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn(batch)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del out
    return {"peak_alloc_gb": peak / 2**30, "baseline_alloc_gb": base / 2**30, "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}


# --------------------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------------------


def compare_logits(out: torch.Tensor, ref: torch.Tensor, center: int) -> dict[str, Any]:
    """Identity evidence: bitwise equality, max |dlogits|, argmax agreement, VEP-score agreement."""
    out32 = out.detach().float()
    ref32 = ref.detach().float()
    diff = (out32 - ref32).abs()
    llr_o = llr_scores(out32, center)
    llr_r = llr_scores(ref32, center)
    dllr = (llr_o - llr_r).abs()
    return {
        "bitwise_identical": bool(out.dtype == ref.dtype and torch.equal(out, ref)),
        "max_abs_dlogits": float(diff.max()),
        "mean_abs_dlogits": float(diff.mean()),
        "max_rel_dlogits": float((diff / ref32.abs().clamp_min(1e-6)).max()),
        "argmax_identical": bool(torch.equal(out32.argmax(-1), ref32.argmax(-1))),
        "argmax_mismatch_frac": float((out32.argmax(-1) != ref32.argmax(-1)).float().mean()),
        "llr_identical": bool(torch.equal(llr_o, llr_r)),
        "max_abs_dllr": float(dllr.max()),
        "out_dtype": str(out.dtype),
    }


# --------------------------------------------------------------------------------------
# fast-mode toggles (NOT exact unless proven): TF32, autocast, torch.compile
# --------------------------------------------------------------------------------------


def _safe(fn, default=None):
    """fn() or `default` when reading a backend flag raises (the legacy TF32 flag beside the precision interface)."""
    try:
        return fn()
    except RuntimeError:
        return default


def _cudnn_tf32_get() -> bool:
    """cuDNN TF32 state through the interface in use (legacy flag, else the conv precision setting)."""
    try:
        return bool(torch.backends.cudnn.allow_tf32)
    except RuntimeError:
        conv = getattr(torch.backends.cudnn, "conv", None)
        return str(getattr(conv, "fp32_precision", "ieee")) == "tf32"


def _cudnn_tf32_set(value: bool) -> None:
    try:
        torch.backends.cudnn.allow_tf32  # noqa: B018
        torch.backends.cudnn.allow_tf32 = bool(value)
    except RuntimeError:
        for part in ("conv", "rnn"):
            sub = getattr(torch.backends.cudnn, part, None)
            if sub is not None and hasattr(sub, "fp32_precision"):
                sub.fp32_precision = "tf32" if value else "ieee"


@contextlib.contextmanager
def tf32(enabled: bool):
    from .patches import _tf32_get, _tf32_set
    old_mm = _tf32_get()
    old_cudnn = _cudnn_tf32_get()
    _tf32_set(enabled)
    _cudnn_tf32_set(enabled)
    try:
        yield
    finally:
        _tf32_set(old_mm)
        _cudnn_tf32_set(old_cudnn)


def compile_model(model, mode: str):
    """torch.compile with inductor settings that do not alter numerics policy (no TF32 change,
    eager-equivalent RNG, no fast-math reassociation flags beyond inductor defaults)."""
    import torch._inductor.config as icfg

    icfg.fallback_random = True
    if hasattr(icfg, "emulate_precision_casts"):
        icfg.emulate_precision_casts = True
    # keep cuBLAS for GEMMs unless max-autotune explicitly searches Triton templates
    return torch.compile(model, mode=mode, fullgraph=False, dynamic=False)


def env_report() -> dict[str, Any]:
    import platform

    import transformers

    props = torch.cuda.get_device_properties(0)
    rep = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "transformers": transformers.__version__,
        "gpu_name": props.name,
        "gpu_total_mem_gb": props.total_memory / 2**30,
        "gpu_sm_count": props.multi_processor_count,
        "gpu_cc": f"{props.major}.{props.minor}",
        "matmul_allow_tf32_default": _safe(lambda: __import__("gpnstar_opt.accel.patches", fromlist=["_tf32_get"])._tf32_get()),
        "cudnn_allow_tf32_default": _safe(_cudnn_tf32_get),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "has_deprecated_sdp_kernel_ctx": hasattr(torch.backends.cuda, "sdp_kernel"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": __import__("os").environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    try:
        import gpn  # noqa: F401
        from importlib.metadata import version

        rep["gpn"] = version("gpn")
    except Exception as e:  # pragma: no cover
        rep["gpn"] = f"unavailable: {e}"
    try:
        import subprocess

        rep["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
    except Exception:
        pass
    return rep


H100_PEAK_TFLOPS = {
    # dense, no sparsity. SXM5 numbers; PCIe is lower (fp32 51, bf16 756).
    "fp32": 66.9,
    "tf32": 494.7,
    "bf16": 989.4,
    "fp16": 989.4,
}
