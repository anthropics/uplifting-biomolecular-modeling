"""is_oom(exc): True when ``exc`` is an out-of-memory failure — the kit's ONE spelling of that test.

Every served except-branch that would otherwise continue on another route (an eager path, the stock path, fewer levers) calls it
first and re-raises: an out-of-memory error is never served by a fallback; it propagates, or the served unit fails by name.

Recognised, on the exception itself and one level down its ``__cause__`` / ``__context__`` (an out-of-memory error re-raised as
another exception by a wrapper):
  * ``torch.OutOfMemoryError`` / ``torch.cuda.OutOfMemoryError`` — by isinstance when torch is already imported, by class name otherwise
    (this module imports nothing but the standard library);
  * a ``RuntimeError`` whose text says ``out of memory`` (CUDA's own error strings, e.g. ``CUDA error: out of memory``);
  * ``MemoryError`` (host memory);
  * jaxlib's ``XlaRuntimeError`` carrying ``RESOURCE_EXHAUSTED``.
"""
from __future__ import annotations

import sys

__all__ = ["is_oom"]

_OOM_CLASS_NAMES = frozenset({"OutOfMemoryError"})


def _torch_oom_types() -> tuple:
    torch = sys.modules.get("torch")
    if torch is None:
        return ()
    found = []
    for cls in (getattr(torch, "OutOfMemoryError", None), getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)):
        if isinstance(cls, type) and cls not in found:
            found.append(cls)
    return tuple(found)


def _is_oom_one(exc) -> bool:
    if not isinstance(exc, BaseException):
        return False
    if isinstance(exc, MemoryError):
        return True
    torch_types = _torch_oom_types()
    if torch_types and isinstance(exc, torch_types):
        return True
    name = type(exc).__name__
    if name in _OOM_CLASS_NAMES:
        return True
    if name == "XlaRuntimeError" and "RESOURCE_EXHAUSTED" in str(exc):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def is_oom(exc: BaseException) -> bool:
    """True when ``exc`` — or the exception it directly wraps — is an out-of-memory failure (see the module docstring for the forms)."""
    return _is_oom_one(exc) or _is_oom_one(getattr(exc, "__cause__", None)) or _is_oom_one(getattr(exc, "__context__", None))
