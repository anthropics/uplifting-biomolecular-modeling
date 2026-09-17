"""The Haiku recipe of the row-sharded pair stack (for a dm-haiku model library).

How a kit installs the plan on a Haiku model without touching stock bytes: it REBINDS the ``__call__`` (or another method) of the stock pair
sub-layer classes to bodies that run the stock code on the local row block with the collectives of :mod:`shard` / :mod:`trimul` / :mod:`triatt`
inserted, and wraps the per-layer block (the Evoformer / Pairformer iteration) in ONE :func:`region` (a ``shard_map`` whose body runs under a
Haiku frame). Four Haiku facts make this work and are encoded here once:

  1. A function assigned to ``Cls.__call__`` after class creation runs WITHOUT the module's name scope (Haiku's metaclass wraps methods at class
     creation): parameters would resolve under the caller's scope. :func:`wrap_method` re-applies Haiku's own wrapper; :func:`rebind` does the
     assignment through a :class:`opt_core.mem.patchset.PatchSet` (the memory family's framework-free install record) and hands back the STOCK body for the guard dispatch (a rebound body calls the stock body when :func:`in_region` is false: the same class
     serves sharded and replicated call sites).
  2. ``hk.scan`` inside a ``shard_map`` body (the stock row-chunk loops) threads Haiku's rng state through the loop and writes the carried tracer back
     into the outer frame — a tracer leak. The pair sub-layers draw no rng at inference, so the body runs under a temporary frame WITHOUT an rng
     (:func:`body_frame`; ``rng=None``); ``hk.scan`` then threads only empty state and keeps the stock module naming (replacing it with ``lax.scan``
     renames the chunk loop's modules). A region that DOES draw keys (a sampler) passes the explicit key as ``rng=``.
  3. Sub-modules created inside a non-``__call__`` method live under the scope ``~<method>`` — rebind that method by its own name (``rebind(…,
     "_embed_features", …)``), never inline it into ``__call__``.
  4. Ordered ``jax.debug.callback`` effects are refused on more than one device; phase markers inside a region are unordered callbacks on a data
     dependency, or host-side.

:func:`jit_apply` binds the transformed ``apply`` to the mesh (``jax.jit`` with replicated in/out shardings; the pair is sharded INSIDE by the
regions and constraints) and :func:`shard.put` places parameters / inputs. ``haiku._src`` names are private API: every use is behind a signature
probe with a refusal by name when the installed haiku lacks it (tested: dm-haiku 0.0.16). Standard library at import.
"""
from __future__ import annotations

import contextlib
import inspect
import threading
from typing import Any, Callable, Optional

from .. import MemLeverRefused
from ..patchset import PatchSet
from . import LEVER, _lazy
from . import shard as _shard

__all__ = ["PatchSet", "wrap_method", "rebind", "stock_body", "body_frame", "running_init", "in_region", "region_depth", "region", "jit_apply"]

_LOCAL = threading.local()


def region_depth() -> int:
    """How many :func:`body_frame` contexts this thread is inside (0 outside any sharded region)."""
    return int(getattr(_LOCAL, "depth", 0))


def in_region() -> bool:
    """True inside a :func:`body_frame` / :func:`region` body: a rebound sub-layer body runs its sharded form; False: it calls the stock body."""
    return region_depth() > 0


def _internal_state_cm(rng: Any, lever: str):
    hk = _lazy.haiku(lever)  # noqa: F841 — import proves haiku is present before touching haiku._src
    try:
        from haiku._src import stateful  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        raise MemLeverRefused(lever, f"haiku._src.stateful unavailable ({e!r})") from None
    if not hasattr(stateful, "temporary_internal_state") or not hasattr(stateful, "InternalState"):
        raise MemLeverRefused(lever, "haiku._src.stateful lacks temporary_internal_state/InternalState (certified on dm-haiku 0.0.16)")
    state = stateful.InternalState(params=None, state=None, rng=rng)
    params = inspect.signature(stateful.temporary_internal_state).parameters
    if "share_python_state" in params:
        return stateful.temporary_internal_state(state, share_python_state=True)
    return stateful.temporary_internal_state(state)


def running_init(lever: str = LEVER) -> bool:
    """``hk.running_init()`` — True while ``transform(f).init`` traces (parameter initialisers draw rng keys then)."""
    hk = _lazy.haiku(lever)
    fn = getattr(hk, "running_init", None)
    if fn is None:
        raise MemLeverRefused(lever, "haiku.running_init is absent (certified on dm-haiku 0.0.16)")
    return bool(fn())


@contextlib.contextmanager
def body_frame(rng: Any = None, lever: str = LEVER):
    """The Haiku frame a ``shard_map`` body runs under AT APPLY: params / state as they are, ``rng=None`` (no key is threaded through ``hk.scan`` —
    fact 2) or the explicit key of a sampling region. AT INIT (``hk.running_init()``) the frame is left as it is — initialisers need the key, and
    the kit loads trained parameters rather than initialising through regions. Increments the region depth for :func:`in_region`. Must be
    entered INSIDE the traced body (per call)."""
    cm = None if running_init(lever) else _internal_state_cm(rng, lever)
    _LOCAL.depth = region_depth() + 1
    try:
        if cm is None:
            yield
        else:
            with cm:
                yield
    finally:
        _LOCAL.depth = region_depth() - 1


def wrap_method(cls: Any, name: str, fn: Callable, lever: str = LEVER) -> Callable:
    """``fn`` wrapped the way Haiku's metaclass wraps ``cls.<name>`` (module name scope, interceptors, ``__wrapped__``) — fact 1. Compat: the 3-argument
    ``wrap_method(name, fn, cls_resolver)`` of dm-haiku 0.0.13+ and the 2-argument form before it."""
    _lazy.haiku(lever)
    try:
        from haiku._src import module as hk_module  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        raise MemLeverRefused(lever, f"haiku._src.module unavailable ({e!r})") from None
    wm = getattr(hk_module, "wrap_method", None)
    if wm is None:
        raise MemLeverRefused(lever, "haiku._src.module.wrap_method is absent (certified on dm-haiku 0.0.16)")
    params = list(inspect.signature(wm).parameters)
    if "cls_resolver" in params or len(params) >= 3:
        return wm(name, fn, lambda: cls)
    return wm(name, fn)


def stock_body(cls: Any, name: str = "__call__") -> Callable:
    """The UNWRAPPED stock function behind ``cls.<name>`` (``__wrapped__`` of Haiku's method wrapper, else the attribute itself) — what a rebound
    body calls for call sites outside a region."""
    f = getattr(cls, name)
    return getattr(f, "__wrapped__", f)


def rebind(patches: PatchSet, cls: Any, name: str, fn: Callable, lever: Optional[str] = None) -> Callable:
    """Install ``fn`` as ``cls.<name>`` WITH Haiku's method wrapper, recorded in ``patches`` (``patches.restore()`` puts the stock attribute back;
    a class without ``name`` or a double patch is refused by name). Returns the unwrapped STOCK body for the guard dispatch."""
    orig = stock_body(cls, name)
    patches.replace(cls, name, wrap_method(cls, name, fn, lever or patches.lever))
    return orig


def region(fn: Callable, rmesh: Any, in_specs: Any, out_specs: Any, rng: Any = None, lever: str = LEVER) -> Callable:
    """``fn`` as a sharded region: ``shard_map(body, mesh, in_specs, out_specs)`` where ``body`` enters :func:`body_frame` (rng-less unless ``rng``)
    and calls ``fn`` on the local blocks. The one form every per-layer block of the plan uses."""
    def body(*args, **kwargs):
        with body_frame(rng, lever):
            return fn(*args, **kwargs)
    return _shard.shard_map(body, rmesh, in_specs, out_specs)


def jit_apply(apply_fn: Callable, rmesh: Any, n_args: Optional[int] = None, **jit_kwargs) -> Callable:
    """``jax.jit(apply_fn)`` bound to the mesh with REPLICATED input and output shardings (``n_args``: give one replicated sharding per positional
    argument as a tuple; ``None``: one sharding broadcast to every argument leaf). The pair is sharded inside the program by the regions and the
    constraints; parameters and the batch enter replicated (:func:`shard.put`)."""
    jax = _lazy.jax()
    rep = _shard.named(rmesh, _shard.replicated_spec())
    ins = rep if n_args is None else tuple(rep for _ in range(int(n_args)))
    return jax.jit(apply_fn, in_shardings=ins, out_shardings=rep, **jit_kwargs)
