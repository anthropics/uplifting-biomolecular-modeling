"""opt_core.oom — the one out-of-memory classifier of the core and of the kit that pins it.  Standard library only at import (no framework).

    is_oom(exc) -> bool

True for: torch's ``OutOfMemoryError`` (``torch.OutOfMemoryError`` / ``torch.cuda.OutOfMemoryError`` — by ``isinstance`` when torch is already imported in
this process, by class name otherwise), any exception whose class is named ``OutOfMemoryError``, a ``RuntimeError`` (or other exception) whose message
contains ``CUDA out of memory`` or ``CUDA error: out of memory`` (older torch), jaxlib's ``XlaRuntimeError`` / jax's ``JaxRuntimeError`` (by class name) whose message contains
``RESOURCE_EXHAUSTED``, TensorFlow's ``ResourceExhaustedError`` (by class name), the host's ``MemoryError``; the same one level down ``__cause__`` / ``__context__`` (a wrapper that re-raised an OOM as
something else).  False otherwise.

    is_oom_text(text) -> bool

The same vocabulary over TEXT (a child process's log line or stderr tail, where no exception object exists): a line naming ``OutOfMemoryError`` /
``ResourceExhaustedError``, containing ``CUDA out of memory`` / ``CUDA error: out of memory``, ``RESOURCE_EXHAUSTED``, or the host's ``MemoryError``.

Every SERVED-path broad ``except`` that reroutes around a lever or kernel (a stock / eager / engine-forward fallback) starts with
``if is_oom(e): raise``: a fallback needs at least as much memory as the fused path it replaces, so an out-of-memory is the caller's to see — never
counted as a kernel error, never rerouted.
"""
import sys

__all__ = ["is_oom", "is_oom_text"]

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
    if ("XlaRuntimeError" in names or "JaxRuntimeError" in names) and "RESOURCE_EXHAUSTED" in msg:   # jaxlib < 0.8: XlaRuntimeError; jax >= 0.8: JaxRuntimeError (XlaRuntimeError an alias name, absent from the MRO)
        return True
    return False


def is_oom(exc: BaseException) -> bool:
    if _is_oom_here(exc):
        return True
    for inner in (getattr(exc, "__cause__", None), getattr(exc, "__context__", None)):
        if inner is not None and _is_oom_here(inner):
            return True
    return False


_TEXT_NAMES = ("OutOfMemoryError", "ResourceExhaustedError", "RESOURCE_EXHAUSTED", "MemoryError")


def is_oom_text(text: str) -> bool:
    """True when ``text`` (a log line, a stderr tail) carries the out-of-memory vocabulary of :func:`is_oom`: the classifier for a CHILD process's
    output, where the parent holds text, not an exception."""
    s = str(text)
    if any(n in s for n in _TEXT_NAMES):
        return True
    low = s.lower()
    return any(m in low for m in _MESSAGES)
