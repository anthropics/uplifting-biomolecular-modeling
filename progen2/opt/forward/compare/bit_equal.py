"""Bit-pattern equality of arrays — never float equality (torch.eq / == / np.array_equal admit -0.0 == +0.0 and refuse NaN == NaN).
The kits' table checks import ``bit_equal`` from here.

    bit_equal(a, b)          -> bool   shape AND dtype equal, and every element's bit pattern equal (an integer view of the
                                       dtype's width; NaN == NaN when the bits match; -0.0 != +0.0)
    bit_mismatch_count(a, b) -> int    elements whose bit patterns differ (shape/dtype mismatch = every element)
    signed_zero_count(a)     -> {neg_zero, pos_zero, nan, n}   the zero / NaN audit of an array
    sha256_bytes(a)          -> str    sha256 over the raw C-order bytes (the other admitted form of a bit-exact count)
"""
from __future__ import annotations

import hashlib

import numpy as np

_INT_VIEW = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64, 16: None}


def _as_numpy(a) -> np.ndarray:
    """numpy / torch (cpu copy; bf16 via a uint16 view) / python scalars -> a contiguous numpy array, dtype preserved."""
    try:
        import torch
        if isinstance(a, torch.Tensor):
            t = a.detach().cpu().contiguous()
            if t.dtype == torch.bfloat16:
                return t.view(torch.int16).numpy().view(np.uint16)
            return t.numpy()
    except ImportError:
        pass
    return np.ascontiguousarray(np.asarray(a))


def _bits(x: np.ndarray) -> np.ndarray:
    w = x.dtype.itemsize
    if x.dtype.kind in "biu":
        return x
    view = _INT_VIEW.get(w)
    if view is None:                                                   # complex128 / float128: compare the raw bytes
        return x.view(np.uint8)
    return x.view(view)


def bit_equal(a, b) -> bool:
    if _both_torch(a, b) and (a.is_cuda or b.is_cuda):             # CUDA tensors: the DEVICE form, no D2H
        return bit_compare(a, b, device="device")["mismatch_count"] == 0
    x, y = _as_numpy(a), _as_numpy(b)
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    return bool(np.array_equal(_bits(x), _bits(y)))                   # integer arrays: array_equal IS the bit compare


def bit_mismatch_count(a, b) -> int:
    if _both_torch(a, b) and (a.is_cuda or b.is_cuda):             # CUDA tensors: the DEVICE form, no D2H
        return bit_compare(a, b, device="device")["mismatch_count"]
    x, y = _as_numpy(a), _as_numpy(b)
    if x.shape != y.shape or x.dtype != y.dtype:
        return int(max(x.size, y.size))
    return int(np.count_nonzero(_bits(x) != _bits(y)))


def _both_torch(a, b) -> bool:
    try:
        import torch
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
    except ImportError:
        return False


_TORCH_INT_VIEW = {"float64": "int64", "float32": "int32", "float16": "int16", "bfloat16": "int16", "int64": "int64", "int32": "int32",
                   "int16": "int16", "int8": "int8", "uint8": "uint8", "bool": "uint8"}


def bit_compare(a, b, device: str = None) -> dict:
    """The SAME
    semantics as bit_equal / bit_mismatch_count — bit-pattern equality through an integer view of the dtype's width (-0.0 != +0.0,
    NaN == NaN by pattern) — computed on the tensor's OWN device when both inputs are torch tensors (device='device', or CUDA
    inputs with device=None), else on the host (the default). Returns {n_total, n_equal, mismatch_count, compare_form}
    with compare_form 'bit_equal/host' | 'bit_equal/device'; a shape / dtype mismatch counts every element as a mismatch."""
    try:
        import torch
        is_t = isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
    except ImportError:
        torch = None; is_t = False
    use_device = is_t and (device == "device" or (device is None and (a.is_cuda or b.is_cuda)))
    if use_device:
        if a.shape != b.shape or a.dtype != b.dtype:
            n = int(max(a.numel(), b.numel()))
            return {"n_total": n, "n_equal": 0, "mismatch_count": n, "compare_form": "bit_equal/device"}
        iv = _TORCH_INT_VIEW.get(str(a.dtype).replace("torch.", ""))
        if iv is None:
            raise TypeError(f"bit_compare: no integer view for dtype {a.dtype}")
        x = a.detach().contiguous().view(getattr(torch, iv)); y = b.detach().to(a.device).contiguous().view(getattr(torch, iv))
        mism = int(torch.ne(x, y).sum().item())
        n = int(x.numel())
        return {"n_total": n, "n_equal": n - mism, "mismatch_count": mism, "compare_form": "bit_equal/device"}
    mism = bit_mismatch_count(a, b); n = int(_as_numpy(a).size)
    return {"n_total": n, "n_equal": max(n - mism, 0) if mism <= n else 0, "mismatch_count": mism, "compare_form": "bit_equal/host"}


def _as_numpy_for_audit(a):
    """bf16 (torch) is UPCAST to float32 for the zero / NaN census — exact for ±0 and NaN (a uint16 bit
    view would count nothing there); every other dtype audits as itself."""
    try:
        import torch
        if isinstance(a, torch.Tensor) and a.dtype == torch.bfloat16:
            return a.detach().cpu().contiguous().float().numpy(), "torch.bfloat16->float32"
    except ImportError:
        pass
    x = _as_numpy(a)
    return x, str(x.dtype)


def signed_zero_count(a) -> dict:
    """The zero / NaN audit: how many -0.0, +0.0 and NaN an array holds (a float-equal count over an array with none of
    them is bit-exact by this audit; otherwise only bit_equal decides)."""
    x, dtype_name = _as_numpy_for_audit(a)
    out = {"n": int(x.size), "neg_zero": 0, "pos_zero": 0, "nan": 0, "dtype": dtype_name}
    if x.dtype.kind == "f":
        zero = x == 0
        out["neg_zero"] = int(np.count_nonzero(zero & np.signbit(x)))
        out["pos_zero"] = int(np.count_nonzero(zero & ~np.signbit(x)))
        out["nan"] = int(np.count_nonzero(np.isnan(x)))
    out["bit_exact_by_audit"] = out["neg_zero"] == 0 and out["nan"] == 0   # +0.0 alone cannot hide a sign or NaN difference
    return out


def sha256_bytes(a) -> str:
    return hashlib.sha256(_as_numpy(a).tobytes(order="C")).hexdigest()
