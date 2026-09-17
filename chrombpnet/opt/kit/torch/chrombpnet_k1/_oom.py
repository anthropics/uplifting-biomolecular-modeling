"""is_oom(exc) -> bool — this kit's one out-of-memory classifier (stdlib only; no framework is imported here).

The rule it serves: a broad ``except`` that reroutes around a lever or a kernel launch (a fallback, the stock route, a cold path)
re-raises an out-of-memory error FIRST — ``if is_oom(e): raise`` is the handler's first statement — so an OOM propagates to the
caller and no fallback is applied; every other exception keeps the handler's existing named route.

True for:
  * ``torch.OutOfMemoryError`` / ``torch.cuda.OutOfMemoryError`` — by ``isinstance`` when torch is already imported in this
    process, else by class name (``OutOfMemoryError`` from a ``torch`` module);
  * a ``RuntimeError`` whose text carries ``CUDA out of memory`` or ``CUDA error: out of memory`` (the pre-class spelling);
  * ``MemoryError`` (host memory; numpy's ``_ArrayMemoryError`` is one);
  * ``tf.errors.ResourceExhaustedError`` — by ``isinstance`` when tensorflow is already imported, else by class name;
  * jaxlib's ``XlaRuntimeError`` / ``JaxRuntimeError`` whose text carries ``RESOURCE_EXHAUSTED``.
The exception's ``__cause__`` and ``__context__`` are examined one level down (a wrapper raised ``from`` an OOM is an OOM).
"""
import sys

__all__ = ["is_oom"]

_CUDA_OOM_TEXT = ("CUDA out of memory", "CUDA error: out of memory")


def _text(e):
    try:
        return str(e)
    except Exception:  # noqa: BLE001 — an exception whose __str__ raises carries no text
        return ""


def _framework_class(module_name, *attr_path):
    """The class at sys.modules[module_name].<attr_path> when that module is already imported, else None (nothing is imported here)."""
    obj = sys.modules.get(module_name)
    for a in attr_path:
        if obj is None:
            return None
        obj = getattr(obj, a, None)
    return obj if isinstance(obj, type) else None


def _is_oom_one(e):
    if e is None or not isinstance(e, BaseException):
        return False
    if isinstance(e, MemoryError):
        return True
    t = type(e)
    mod = getattr(t, "__module__", "") or ""
    name = t.__name__
    for cls in (_framework_class("torch", "OutOfMemoryError"), _framework_class("torch", "cuda", "OutOfMemoryError"),
                _framework_class("tensorflow", "errors", "ResourceExhaustedError")):
        if cls is not None and isinstance(e, cls):
            return True
    if name == "OutOfMemoryError" and (mod == "torch" or mod.startswith("torch.")):
        return True
    if name == "ResourceExhaustedError" and (mod == "tensorflow" or mod.startswith("tensorflow.")):
        return True
    msg = _text(e)
    if isinstance(e, RuntimeError) and any(s in msg for s in _CUDA_OOM_TEXT):
        return True
    if name in ("XlaRuntimeError", "JaxRuntimeError") and "RESOURCE_EXHAUSTED" in msg:
        return True
    return False


def is_oom(e):
    """True when ``e`` — or its direct ``__cause__`` / ``__context__`` — is an out-of-memory error (see the module text)."""
    return (_is_oom_one(e) or _is_oom_one(getattr(e, "__cause__", None)) or _is_oom_one(getattr(e, "__context__", None)))
