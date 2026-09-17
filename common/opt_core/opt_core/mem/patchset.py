"""PatchSet — the adapter's install record of attribute patches, applied and restored as one set. Framework-free (standard library
only): a framework adapter and a test double use the same record.
"""
from __future__ import annotations

import threading
from typing import Any, List, Tuple

from .primitives import MemLeverRefused

__all__ = ["PatchSet"]


class PatchSet:
    """The adapter's install record: ``replace(owner, name, new)`` rebinds ``owner.name`` (a class or module attribute) and keeps the
    original (returned, and served again by ``original(owner, name)`` for the stock call of a guard); ``restore()`` puts every original
    back in reverse order; ``names()`` lists ``Owner.name`` strings for the report. Replacing an attribute the owner does not have, or
    the same attribute twice, raises :class:`MemLeverRefused` (a lever wired against a module surface that moved is refused by name)."""

    def __init__(self, lever: str):
        self.lever = str(lever)
        self._lock = threading.Lock()
        self._orig: List[Tuple[Any, str, Any, bool]] = []

    def replace(self, owner: Any, name: str, new: Any) -> Any:
        with self._lock:
            if not hasattr(owner, name):
                raise MemLeverRefused(self.lever, f"{getattr(owner, '__name__', owner)!s} has no attribute {name!r} to patch")
            for o, nm, _, _own in self._orig:
                if o is owner and nm == name:
                    raise MemLeverRefused(self.lever, f"{getattr(owner, '__name__', owner)!s}.{name} is already patched by this lever")
            orig = getattr(owner, name)
            own = name in vars(owner)                     # inherited attribute: restore() deletes the override instead of pinning a copy
            self._orig.append((owner, name, orig, own))
            setattr(owner, name, new)
            return orig

    def original(self, owner: Any, name: str) -> Any:
        with self._lock:
            for o, nm, orig, _own in self._orig:
                if o is owner and nm == name:
                    return orig
        raise MemLeverRefused(self.lever, f"{getattr(owner, '__name__', owner)!s}.{name} was not patched by this lever")

    def restore(self) -> List[str]:
        with self._lock:
            out = []
            for owner, name, orig, own in reversed(self._orig):
                if own:
                    setattr(owner, name, orig)
                else:
                    delattr(owner, name)                  # the override goes; the inherited attribute is served again by the MRO
                out.append(f"{getattr(owner, '__name__', owner)!s}.{name}")
            self._orig.clear()
            return out

    def names(self) -> List[str]:
        with self._lock:
            return [f"{getattr(o, '__name__', o)!s}.{nm}" for o, nm, _, _own in self._orig]

    def __len__(self) -> int:
        return len(self._orig)
