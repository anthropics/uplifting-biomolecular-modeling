"""The late-activation instance counter: how many objects of a named class exist in this process, deterministically.

Contract. The kit's late-activation rule: activation may run any time after the upstream packages are imported but is refused
by name once a model instance exists (a model built before activation would run unconfigured). :func:`register_instance_counter`
wraps ``cls.__new__`` (a staticmethod calling the original ``__new__`` unchanged; ``__signature__`` taken from ``__init__`` so
``inspect.signature(cls)`` is unchanged) and keeps a total (``built``) plus a weak set of the live instances — ``__new__`` sees every
creation: a direct call, a factory, ``copy.deepcopy``, unpickling, a subclass. When the class's module is already imported the wrap
happens now and the instances that already exist are added once by a gc scan (``gc_seeded``); otherwise a self-removing meta-path
finder wraps the class right after its module body runs (the program's import order stands; nothing is imported by the core) and seeds
the count from a gc scan the same way, so an instance the module body itself built is counted.
:func:`instance_check` returns ``{"n", "method": "counted" | "gc" | "none", "built"}``: ``counted`` = the wrap; ``gc`` = a scan of the
garbage collector's objects (the class could not be wrapped, or its instances take no weak references); ``none`` = the module is not
imported (0 instances). The class is named by the kit (module + attribute); the core knows no model.
"""
from __future__ import annotations

import functools
import inspect
import sys
import weakref
from typing import Optional

_COUNTERS: dict = {}          # (module_name, class_name) -> {"state": ..., "finder": ...}


def _entry(module_name: str, class_name: str) -> dict:
    return _COUNTERS.setdefault((module_name, class_name), {"state": None, "finder": None})


def _class(module_name: str, class_name: str) -> Optional[type]:
    m = sys.modules.get(module_name)
    if m is None:
        return None
    cls = getattr(m, class_name, None)
    return cls if isinstance(cls, type) else None


def _is_instance_of(o, cls: type) -> bool:
    """``isinstance(o, cls)``, safe over ``gc.get_objects()``: a dangling ``weakref.proxy`` (its referent already collected --
    a frame, a guard, anything a tracing/compilation layer holds transiently) raises ``ReferenceError`` on the type check
    itself, not on attribute access; that is a dead object, never an instance of anything a kit's counter tracks."""
    try:
        return isinstance(o, cls)
    except ReferenceError:
        return False


def _count_instances_of(entry: dict, cls: type, seed_from_gc: bool = False) -> bool:
    """Wrap ``cls.__new__`` so every instance created from now on is counted. One wrap per class object; the original ``__new__`` is
    called unchanged and the class signature is kept. Returns False when the class cannot be wrapped (the counter stays on the gc scan)."""
    st = entry["state"]
    if st is not None and st["cls"] is cls:
        return st["wrapped"]
    live: "weakref.WeakSet" = weakref.WeakSet()
    state = {"cls": cls, "wrapped": False, "live": live, "built": 0, "weakref": True, "gc_seeded": None}
    entry["state"] = state
    orig_new = cls.__new__

    @functools.wraps(orig_new)
    def __new__(klass, *args, **kwargs):
        obj = orig_new(klass) if orig_new is object.__new__ else orig_new(klass, *args, **kwargs)
        state["built"] += 1
        try:
            live.add(obj)
        except TypeError:                                                  # no weak references (__slots__ without __weakref__): total only
            state["weakref"] = False
        return obj

    if "__new__" not in cls.__dict__:
        try:
            __new__.__signature__ = inspect.signature(cls.__init__)        # inspect.signature(cls) keeps reporting __init__'s parameters
        except (TypeError, ValueError):
            pass
    try:
        cls.__new__ = staticmethod(__new__)
    except (TypeError, AttributeError):
        state["live"] = None
        return False
    if seed_from_gc:
        import gc
        n = 0
        for o in gc.get_objects():
            if _is_instance_of(o, cls):
                n += 1
                try:
                    live.add(o)
                except TypeError:
                    state["weakref"] = False
        state["gc_seeded"] = n
    state["wrapped"] = True
    return True


class _ClassFinder:
    """Meta-path finder that wraps the named class's constructor right after its module body has run. Fires once and removes itself."""

    def __init__(self, module_name: str, class_name: str):
        self.module_name = module_name
        self.class_name = class_name
        self.armed = True

    def find_spec(self, fullname, path=None, target=None):
        if not self.armed or fullname != self.module_name:
            return None
        spec = None
        for finder in sys.meta_path:
            if finder is self:
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:  # noqa: BLE001
                spec = None
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            return None
        self.armed = False
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig):
            _orig(module)                                                  # the module body first, then the wrap on the class
            self.remove()
            cls = getattr(module, self.class_name, None)
            if isinstance(cls, type):
                _count_instances_of(_entry(self.module_name, self.class_name), cls)      # the finder route: no gc scan at import
        spec.loader.exec_module = exec_module
        return spec

    def remove(self) -> None:
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass


def register_instance_counter(module_name: str, class_name: str) -> dict:
    """Count instances of ``<module_name>.<class_name>`` from now on: on the class now when its module is imported (the instances that
    already exist are found once by a gc scan), otherwise at the module's import through a meta-path finder. Idempotent. Returns
    :func:`instance_check`."""
    entry = _entry(module_name, class_name)
    cls = _class(module_name, class_name)
    if cls is not None:
        _count_instances_of(entry, cls, seed_from_gc=True)
    elif entry["finder"] is None or entry["finder"] not in sys.meta_path:
        f = _ClassFinder(module_name, class_name)
        sys.meta_path.insert(0, f)
        entry["finder"] = f
    return instance_check(module_name, class_name)


def instance_check(module_name: str, class_name: str) -> dict:
    """``{"n", "method": "counted" | "gc" | "none", "built"}`` for the named class (see the module contract)."""
    cls = _class(module_name, class_name)
    if cls is None:
        return {"n": 0, "method": "none", "built": 0}
    st = _entry(module_name, class_name)["state"]
    same = st is not None and st["cls"] is cls
    if same and st["wrapped"] and st["weakref"]:
        return {"n": len(st["live"]), "method": "counted", "built": st["built"]}
    import gc
    return {"n": sum(1 for o in gc.get_objects() if _is_instance_of(o, cls)), "method": "gc", "built": st["built"] if same else 0}


def counter_note(module_name: str, class_name: str) -> Optional[str]:
    """The sentence a report carries when the count is not deterministic, or None."""
    st = _entry(module_name, class_name)["state"]
    if st is None:
        return None
    if not st["wrapped"]:
        return f"{class_name} instances are counted by a gc scan: the class could not be wrapped"
    if not st["weakref"]:
        return f"{class_name} instances are counted by a gc scan: its instances take no weak references"
    return None
