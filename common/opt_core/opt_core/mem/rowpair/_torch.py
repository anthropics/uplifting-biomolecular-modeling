"""Lazy handles on ``torch`` and ``torch.distributed`` for this package: ``from ._torch import torch, dist`` binds two proxies whose first
attribute access imports the module (a missing torch is a :class:`RowpairRefused` naming it). Importing any ``rowpair`` module therefore
imports nothing heavy; the row-sharded statement bodies keep their ``torch.`` / ``dist.`` spelling unchanged."""
from __future__ import annotations

import importlib
import threading

__all__ = ["torch", "dist", "real_torch"]


class _LazyModule(object):
    __slots__ = ("_name", "_mod", "_lock")

    def __init__(self, name: str):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_mod", None)
        object.__setattr__(self, "_lock", threading.Lock())

    def _load(self):
        mod = object.__getattribute__(self, "_mod")
        if mod is None:
            with object.__getattribute__(self, "_lock"):
                mod = object.__getattribute__(self, "_mod")
                if mod is None:
                    name = object.__getattribute__(self, "_name")
                    try:
                        mod = importlib.import_module(name)
                    except Exception as exc:  # noqa: BLE001
                        from . import RowpairRefused
                        raise RowpairRefused(f"{name} is not importable in this interpreter ({type(exc).__name__}: {exc})") from None
                    object.__setattr__(self, "_mod", mod)
        return mod

    def __getattr__(self, attr):
        return getattr(self._load(), attr)

    def __repr__(self):
        return f"<lazy module {object.__getattribute__(self, '_name')}>"


torch = _LazyModule("torch")
dist = _LazyModule("torch.distributed")


def real_torch():
    """The imported ``torch`` module itself (for ``isinstance`` checks against ``torch.Tensor`` and other places a proxy will not do)."""
    return torch._load()
