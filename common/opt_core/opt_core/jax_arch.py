"""The two card readers of a jax device: its ``sm`` class (:func:`jax_sm`) and the memory its allocator may use
(:func:`jax_device_memory`) — the jax counterparts of :func:`opt_core.arch.current_sm` / :func:`opt_core.arch.device_memory`.
jax is imported inside the call that needs it, never at import; a missing jax or a device without the fact reads None with the
source named. Standard library at import.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from .arch import sm_of

__all__ = ["jax_sm", "jax_device_memory"]


def jax_sm(device=None, *, devices: Optional[Sequence] = None) -> Tuple[Optional[str], dict]:
    """(sm class or None, ``{"platform", "device_kind", "probe"}``) of a jax device: ``device`` as given, else the first of ``devices``,
    else ``jax.devices()[0]`` (imports jax inside the call). A non-GPU platform or a device without ``compute_capability`` reads None with
    the platform named."""
    if device is None:
        if devices is None:
            try:
                import jax                                # noqa: PLC0415
            except ImportError as e:
                return None, {"platform": None, "device_kind": None, "probe": "jax missing (%s)" % (e,)}
            devices = jax.devices()
        if not devices:
            return None, {"platform": None, "device_kind": None, "probe": "jax: no devices"}
        device = devices[0]
    platform = getattr(device, "platform", None)
    kind = getattr(device, "device_kind", None)
    cc = getattr(device, "compute_capability", None)
    sm = sm_of(cc) if (cc is not None and str(platform).lower() in ("gpu", "cuda")) else None
    return sm, {"platform": platform, "device_kind": kind, "probe": "jax" if sm else "jax: platform %s has no CUDA compute capability" % (platform,)}


def jax_device_memory(device=None) -> dict:
    """``{"total_bytes", "free_bytes", "source"}`` from a jax device's ``memory_stats()`` (``bytes_limit`` = what the XLA allocator may use,
    free = limit - ``bytes_in_use``); None fields with the source named when the backend gives no stats."""
    if device is None:
        try:
            import jax                                    # noqa: PLC0415
        except ImportError as e:
            return {"total_bytes": None, "free_bytes": None, "source": "jax missing (%s)" % (e,)}
        devs = jax.devices()
        if not devs:
            return {"total_bytes": None, "free_bytes": None, "source": "jax: no devices"}
        device = devs[0]
    stats = None
    try:
        stats = device.memory_stats()
    except Exception as e:  # noqa: BLE001 — a backend without stats names itself
        return {"total_bytes": None, "free_bytes": None, "source": "jax: memory_stats unavailable (%s)" % (type(e).__name__,)}
    if not stats or "bytes_limit" not in stats:
        return {"total_bytes": None, "free_bytes": None, "source": "jax: memory_stats without bytes_limit"}
    limit = int(stats["bytes_limit"])
    used = int(stats.get("bytes_in_use", 0))
    return {"total_bytes": limit, "free_bytes": max(0, limit - used), "source": "jax"}
