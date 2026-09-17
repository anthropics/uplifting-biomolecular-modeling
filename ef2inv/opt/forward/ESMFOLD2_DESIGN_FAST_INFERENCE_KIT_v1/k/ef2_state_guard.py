"""ef2_state_guard — joint rule D59: an in-process / resident runtime must not leak process-global numeric state across
chained steps (design step -> loss -> critic; request -> request).  The kit's levers change *scheduling* only, so the
contract is simple: the process-global state snapshot taken when the kit is enabled must be identical at every later
step boundary.  This module provides the snapshot, a diff, and a guard object that asserts (or restores + reports).

Snapshot covers: torch float32 matmul precision, TF32 flags (matmul, cudnn), cudnn benchmark/deterministic/enabled,
deterministic-algorithms mode, bf16/fp16 reduced-precision-reduction flags, default dtype, autocast state (gpu/cpu,
dtype, nesting), grad mode / inference mode, flash/mem-efficient/math SDPA toggles, and the env switches the kit reads.
RNG state is deliberately NOT part of equality (a design step consumes RNG by design); `rng_digest()` is provided
separately so callers can log it per step and compare against a stock run (the cookbook re-seeds per step).
"""
from __future__ import annotations
import os, hashlib
import torch

_KIT_ENV = ("EF2_FAST_KIT", "EF2_FAST_KIT_PPPL", "EF2_FAST_KIT_CKPT", "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG", "DET_SCATTER")


def snapshot() -> dict:
    b = torch.backends
    s = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda.matmul.allow_tf32": b.cuda.matmul.allow_tf32,
        "cudnn.allow_tf32": b.cudnn.allow_tf32,
        "cudnn.benchmark": b.cudnn.benchmark,
        "cudnn.deterministic": b.cudnn.deterministic,
        "cudnn.enabled": b.cudnn.enabled,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "default_dtype": str(torch.get_default_dtype()),
        "grad_enabled": torch.is_grad_enabled(),
        "inference_mode": torch.is_inference_mode_enabled(),
        "autocast_gpu_enabled": torch.is_autocast_enabled("cuda") if _accepts_device(torch.is_autocast_enabled) else torch.is_autocast_enabled(),
        "autocast_cpu_enabled": torch.is_autocast_enabled("cpu") if _accepts_device(torch.is_autocast_enabled) else torch.is_autocast_cpu_enabled(),
        "autocast_gpu_dtype": str(torch.get_autocast_dtype("cuda")) if hasattr(torch, "get_autocast_dtype") else str(torch.get_autocast_gpu_dtype()),
    }
    for name in ("allow_bf16_reduced_precision_reduction", "allow_fp16_reduced_precision_reduction"):
        if hasattr(b.cuda.matmul, name):
            s["cuda.matmul." + name] = getattr(b.cuda.matmul, name)
    for name, fn in (("sdp_flash", "flash_sdp_enabled"), ("sdp_mem_efficient", "mem_efficient_sdp_enabled"), ("sdp_math", "math_sdp_enabled"), ("sdp_cudnn", "cudnn_sdp_enabled")):
        if hasattr(b.cuda, fn):
            try: s[name] = bool(getattr(b.cuda, fn)())
            except Exception: pass
    for k in _KIT_ENV:
        s["env." + k] = os.environ.get(k)
    return s


def _accepts_device(fn) -> bool:
    try:
        fn("cuda"); return True
    except TypeError:
        return False
    except Exception:
        return True


def diff(a: dict, b: dict) -> dict:
    return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def rng_digest() -> str:
    """sha256 (first 16 hex) of CPU + current-device CUDA RNG states — for logging per step, not part of the equality contract."""
    h = hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        h.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


class StateGuard:
    """g = StateGuard(); ... ; g.check("after step 12")  -> raises AssertionError (mode='assert') or restores what can be
    restored and records the violation (mode='restore').  g.violations lists every diff seen; g.n_checks counts calls."""
    def __init__(self, mode: str = "assert", label: str = "kit-enable"):
        assert mode in ("assert", "restore", "log")
        self.mode, self.ref, self.ref_label = mode, snapshot(), label
        self.n_checks, self.violations = 0, []

    def check(self, where: str = "") -> dict:
        self.n_checks += 1
        d = diff(snapshot(), self.ref)
        if d:
            self.violations.append(dict(where=where, diff=d))
            if self.mode == "assert":
                raise AssertionError(f"D59 process-global state changed at '{where}' vs '{self.ref_label}': {d}")
            if self.mode == "restore":
                restore(self.ref)
        return d

    def summary(self) -> dict:
        return dict(ref_label=self.ref_label, n_checks=self.n_checks, n_violations=len(self.violations), violations=self.violations[:5], ref=self.ref)


def restore(ref: dict) -> None:
    b = torch.backends
    torch.set_float32_matmul_precision(ref["float32_matmul_precision"])
    b.cuda.matmul.allow_tf32 = ref["cuda.matmul.allow_tf32"]; b.cudnn.allow_tf32 = ref["cudnn.allow_tf32"]
    b.cudnn.benchmark = ref["cudnn.benchmark"]; b.cudnn.deterministic = ref["cudnn.deterministic"]; b.cudnn.enabled = ref["cudnn.enabled"]
    if torch.are_deterministic_algorithms_enabled() != ref["deterministic_algorithms"]:
        torch.use_deterministic_algorithms(ref["deterministic_algorithms"])
    torch.set_default_dtype(getattr(torch, ref["default_dtype"].split(".")[-1]))
    for name in ("allow_bf16_reduced_precision_reduction", "allow_fp16_reduced_precision_reduction"):
        k = "cuda.matmul." + name
        if k in ref and hasattr(b.cuda.matmul, name):
            setattr(b.cuda.matmul, name, ref[k])
