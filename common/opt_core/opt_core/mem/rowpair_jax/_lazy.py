"""The lazy imports of the sub-package: jax / jax.numpy / numpy / haiku are imported here, inside functions, and a failure is a
:class:`opt_core.mem.MemLeverRefused` naming the lever — never an ImportError at activation of an unrelated mode."""
from __future__ import annotations

from .. import MemLeverRefused
from . import LEVER


def jax(lever: str = LEVER):
    try:
        import jax as _jax  # noqa: PLC0415 — the one lazy import of the mechanism
    except Exception as e:  # noqa: BLE001
        raise MemLeverRefused(lever, f"jax import failed ({e!r})") from None
    return _jax


def jnp(lever: str = LEVER):
    jax(lever)
    import jax.numpy as _jnp  # noqa: PLC0415
    return _jnp


def np(lever: str = LEVER):
    try:
        import numpy as _np  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        raise MemLeverRefused(lever, f"numpy import failed ({e!r})") from None
    return _np


def haiku(lever: str = LEVER):
    try:
        import haiku as _hk  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        raise MemLeverRefused(lever, f"haiku import failed ({e!r})") from None
    return _hk


def sharding(lever: str = LEVER):
    """``(Mesh, NamedSharding, PartitionSpec)`` of the running jax."""
    _jax = jax(lever)
    try:
        from jax.sharding import Mesh, NamedSharding, PartitionSpec  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        raise MemLeverRefused(lever, f"jax.sharding unavailable in jax {getattr(_jax, '__version__', '?')} ({e!r})") from None
    return Mesh, NamedSharding, PartitionSpec


def shard_map_impl(lever: str = LEVER):
    """``(shard_map callable, flavour)``: ``jax.shard_map`` (``check_vma``) on a jax that has it, else ``jax.experimental.shard_map.shard_map``
    (``check_rep``); neither → refused by name (the jax is older than the mechanism)."""
    _jax = jax(lever)
    fn = getattr(_jax, "shard_map", None)
    if fn is not None:
        return fn, "jax.shard_map"
    try:
        from jax.experimental.shard_map import shard_map as fn2  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        raise MemLeverRefused(lever, f"shard_map unavailable in jax {getattr(_jax, '__version__', '?')} ({e!r}); jax >= 0.4.30 is required") from None
    return fn2, "jax.experimental.shard_map"
