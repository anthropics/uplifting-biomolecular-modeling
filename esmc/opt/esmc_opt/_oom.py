"""The kit's one out-of-memory classifier.  Standard library only at import (no torch, no jax, no tensorflow).

    is_oom(exc) -> bool

True for: torch's ``OutOfMemoryError`` (``torch.OutOfMemoryError`` / ``torch.cuda.OutOfMemoryError`` — by ``isinstance`` when torch is already imported in
this process, by class name otherwise), any exception whose class (or a base) is named ``OutOfMemoryError``, an exception whose message contains
``CUDA out of memory`` or ``CUDA error: out of memory`` (older torch raises a plain ``RuntimeError``), jaxlib's ``XlaRuntimeError`` (by class name) whose
message contains ``RESOURCE_EXHAUSTED``, TensorFlow's ``ResourceExhaustedError`` (by class name), the host's ``MemoryError``; the same one level down
``__cause__`` / ``__context__`` (a wrapper that re-raised an out-of-memory as something else).  False otherwise.

Every SERVED-path broad ``except`` that reroutes around a lever or a kernel launch (a stock / eager / unfused / uncaptured / counted-fallback route)
starts with ``if is_oom(e): raise``: a fallback needs at least as much memory as the path it replaces, so an out-of-memory is the caller's to see —
never counted as a lever's refusal, never rerouted.

This kit pins no shared core, so it carries this classifier itself: ONE spelling, one home
(this module is the one home: the package and its kits import it); the kit's CPU tests assert the
copies are identical and exercises the classifier.
"""
import sys

__all__ = ["is_oom"]

_MESSAGES = ("cuda out of memory", "cuda error: out of memory")


def _is_oom_here(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    torch = sys.modules.get("torch")
    if torch is not None:
        kinds = tuple(k for k in (getattr(torch, "OutOfMemoryError", None), getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)) if isinstance(k, type))
        if kinds and isinstance(exc, kinds):
            return True
    names = [k.__name__ for k in type(exc).__mro__]
    if "OutOfMemoryError" in names or "ResourceExhaustedError" in names:      # torch's class / TensorFlow's class, by name (no framework import)
        return True
    msg = str(exc)
    low = msg.lower()
    if any(m in low for m in _MESSAGES):
        return True
    if "XlaRuntimeError" in names and "RESOURCE_EXHAUSTED" in msg:
        return True
    return False


def is_oom(exc: BaseException) -> bool:
    if _is_oom_here(exc):
        return True
    for inner in (getattr(exc, "__cause__", None), getattr(exc, "__context__", None)):
        if inner is not None and _is_oom_here(inner):
            return True
    return False
