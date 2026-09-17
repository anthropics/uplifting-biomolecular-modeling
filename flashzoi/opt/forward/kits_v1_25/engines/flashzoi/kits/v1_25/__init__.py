"""The Flashzoi kit v1.25: ``KitRunner(model)`` attaches the exact composition (stage1, sites, crop, head, precast, graph, pinned, ln_fused,
decoder_fused, transformer_fused, relu_epilogue, skip_fused, decoder_fused2) to a stock Borzoi model; ``KitRunner(model).predict_tensor(x)``
/ ``.predict(x)`` are the calls, ``.remove()`` detaches it. The tables below (LEVERS, LEVER_CLASS, PINS, ARM) are the kit's settings;
``_wrap.build`` turns them into the namespace. The Triton kernels compile on their first launch in a process and persist in Triton's own
cache dir (``TRITON_CACHE_DIR``, else ``~/.triton/cache``): the first process on a machine compiles, later ones load; the apply line printed
at construction states every setting in effect, the cache dir and the compiles counted in the process (``jit:``)."""
from __future__ import annotations

import os
import threading
import time

_T_IMPORT0 = time.perf_counter()      # the module import wall (this module + the wrapper + the kernels' definitions), reported on the runner's apply record

# Triton compiles each kernel on its first launch in a process and keeps the artefacts in its cache dir (TRITON_CACHE_DIR, else ~/.triton/cache);
# the kit writes no cache of its own. The compile hook below counts the compiles of THIS process for the apply line's `jit:` clause.
JIT_COMPILE_LOG = []      # per-call triton compile() durations in THIS process (a cached-group load ≈ ms; a real compile ≥ s) — the apply line prints count + max


_HOOK_BOUND = []      # [(module, attr, original)] bound by _hook_triton_compile — restored by _unhook_triton_compile (remove)


def _unhook_triton_compile() -> bool:
    """Detach the compile hook: every bound attribute restored to the ORIGINAL function object."""
    ok = True
    while _HOOK_BOUND:
        mod, name, orig = _HOOK_BOUND.pop()
        setattr(mod, name, orig); ok = ok and getattr(mod, name) is orig
    return ok


def _hook_triton_compile() -> str:
    """Wrap Triton's compile() under every name the JIT binds it by (triton.compiler, triton.compiler.compiler, triton.runtime.jit) and
    log each call's duration (the apply line's `jit:` clause and jit_after_job() report the count); no driver use here."""
    try:
        import triton.compiler as _tcm, triton.compiler.compiler as _tcc
        import triton.runtime.jit as _tj
    except Exception as e:                                                  # noqa: BLE001 — stated on the line; the kit still runs without the hook
        return f"compile hook not attached ({type(e).__name__})"
    orig = _tcc.compile
    def _timed(*a, **k):
        _t = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            JIT_COMPILE_LOG.append(time.perf_counter() - _t)
    bound = []
    for mod, name in ((_tcc, "compile"), (_tcm, "compile"), (_tj, "compile")):
        if getattr(mod, name, None) is orig:
            setattr(mod, name, _timed); bound.append(mod.__name__); _HOOK_BOUND.append((mod, name, orig))   # kept for the detach
    return "compile hook on " + ", ".join(bound) if bound else "compile hook found no binding"


def triton_cache_dir() -> str:
    """The process's effective Triton cache dir: TRITON_CACHE_DIR when set (configs/*.env point it at a persistent directory), else Triton's
    default ~/.triton/cache. Read, never written, by the kit."""
    return os.environ.get("TRITON_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".triton", "cache")


JIT_CACHE_DIR = None        # the effective Triton cache dir, read at the first KitRunner (jit_start)


def jit_after_job() -> dict:
    """The compile counter AFTER THE JOB: every compile() this process ran, with its duration — a real compile is
    >= ~1 s, a cached-group load is ms. Read it after the LAST forward, never after the warm-up."""
    log = list(JIT_COMPILE_LOG)
    real = [d for d in log if d >= 1.0]
    return {"compile_calls": len(log), "max_ms": round(max(log) * 1000, 1) if log else 0.0, "real_compiles": len(real), "cache_dir": JIT_CACHE_DIR,
            "verdict": ("0 real compiles after the job (every call a cached-group load)" if not real else f"{len(real)} REAL compile(s) after the job (the first: {[round(d, 2) for d in real][:6]} s)")}


ARM = "v1.25"
IMPORT_WALL = {"import_s": 0.0, "jit_started_at": None, "jit_started_after_import_s": None}
_JIT = {"started": False, "lock": threading.Lock()}
JIT_CACHE_STATUS = "not started: the compile hook attaches at the FIRST KitRunner (after the model loads), never at import"


def jit_start(reason: str) -> bool:
    """At the FIRST KitRunner of the process (never at import): attach the compile hook and read the effective Triton cache dir for the apply
    line's `jit:` clause — `<dir> (<n> files at attach); <hook>`. The kernels themselves compile (or load from that dir) at their first
    launch, inside the first runner's warm-up forward. Idempotent: the first caller does the work (True); every later call is a no-op (False)."""
    global JIT_CACHE_DIR, JIT_CACHE_STATUS
    with _JIT["lock"]:
        if _JIT["started"]:
            return False
        _JIT["started"] = True
        IMPORT_WALL["jit_started_at"] = reason; IMPORT_WALL["jit_started_after_import_s"] = time.perf_counter() - _T_IMPORT0
        hook = _hook_triton_compile()
        JIT_CACHE_DIR = triton_cache_dir()
        try:
            n = sum(len(files) for _, _, files in os.walk(JIT_CACHE_DIR))
        except OSError:
            n = 0
        JIT_CACHE_STATUS = f"Triton cache {JIT_CACHE_DIR} ({n} files at attach; kernels compile on first launch and persist there); {hook}"
        return True


LEVERS = ("stage1", "sites", "crop", "head", "precast", "graph", "pinned", "ln_fused", "decoder_fused", "transformer_fused", "relu_epilogue", "skip_fused", "decoder_fused2")
ALL = frozenset(LEVERS)
PINS = {"device_names": ["NVIDIA H100 80GB HBM3", "NVIDIA H200"], "device_classes": [{"class": "h100", "sm": [9, 0], "memory_mib": 81559}, {"class": "h200", "sm": [9, 0], "memory_mib": 143771}], "act_dtype": "float16", "stage1": ["mma", [64, 128, 8]], "stage1_order": 0, "crop": "aligned", "bias_fold": True,
        "head": "baddbmm", "stack": "nchw", "tower": "nhwc", "precast": "fp16 (fz_exact.precast_bf16 casts to ACT_DTYPE)", "graph": True,
        "pinned_lease_pool": True, "fz_exact_sha256": None,
        "stack_kwargs": {"decoder_fused": True, "transformer_fused": True, "relu_epilogue": True, "skip_fused": True, "decoder_fused2": True}, "pre_stack": ["swap_layernorms"]}
LEVER_CLASS = {
    "stage1": "exact on the pinned stack (Triton stage-1 kernel; the class is per image/GPU)",
    "sites": "exact on the pinned stack (PyTorch's own invstd; hybrid NHWC tower on cuDNN 9.1 fp16) — the channels_last conv is NOT bitwise on cuDNN 9.10 (fz_exact.py:341-342, res_tower.0.conv_layer): the class holds per image",
    "crop": "exact on the pinned stack at batch 1 with the 8-aligned windows (the unaligned crop is NOT bitwise on cuDNN 9.1); the class holds per batch shape",
    "head": "exact on the pinned stack (cuBLAS fp32 GEMM == cuDNN fp32 k=1 conv under the same TF32 class)",
    "precast": "exact by construction (the one-time round-to-nearest fp16 cast autocast performs per call)",
    "graph": "exact by construction (captured eager kernels, static shapes)",
    "pinned": "exact (bytes unchanged; the host copy path only)",
    "ln_fused": "exact on the pinned stack (FusedLayerNorm reproduces ATen's vectorised LayerNorm bitwise, fp16 in/out)",
    "decoder_fused": "exact on the pinned stack (decoder BN->GELU sites as one Triton pass)",
    "transformer_fused": "exact on the pinned stack (residual add fused into the next LN)",
    "relu_epilogue": "exact on the pinned stack (fc1 bias+ReLU in the cuBLASLt epilogue)",
    "skip_fused": "exact on the pinned stack (bias add + NHWC->NCHW transpose in one Triton pass; pool-then-bias exact by monotone rounding)",
    "decoder_fused2": "exact on the pinned stack (k_up2_add upsample+add one pass fp16 out; horizontal-conv BN+GELU via k_bn_gelu; head bmm + fused bias+softplus)",
}



from . import _wrap                    # the wrapper: KitRunner (the apply), the helper route, the refusals, remove
globals().update(_wrap.build(__name__, ARM, PINS, LEVERS, LEVER_CLASS, stack_kwargs=PINS["stack_kwargs"], pre_stack=tuple(PINS["pre_stack"])))
remove = _wrap.remove          # THE DOCUMENTED REMOVE — kit.remove(runners) (the README line beside the apply line)
IMPORT_WALL["import_s"] = time.perf_counter() - _T_IMPORT0      # the whole package import (wrapper + kernels' definitions); the line prints it per process
