"""Host allocator environment: the glibc malloc tunables a host-heavy process applies to itself and exports to its children, with a plan
(what would apply and why), an apply step that refuses by name when it cannot take effect, a stamp for the run record and the
activation-evidence fields. The CUDA caching-allocator policy of a process (``PYTORCH_CUDA_ALLOC_CONF``) is ``opt_core.mem.torch_alloc``
(``export`` / ``facts``); this module holds the HOST side only.

Contract. ``plan_host_malloc`` / ``ensure_host_malloc`` / ``stamp_host_malloc``: three glibc tunables applied through ``mallopt``
(ctypes) AND exported so an exec'd child reads the same values at start (a forked child inherits the tuned arenas); ``<opt_out_var>=0``
keeps the stock allocator behaviour; a libc without ``mallopt`` (or one that rejects a tunable) is a named refusal
(``HostMallocRefused``), never a silent skip. ``line_fields(stamp)`` is the seq modules' evidence hook (str -> str). Standard library only.
"""
from __future__ import annotations

import os
from typing import Dict, Mapping, MutableMapping, Optional

__all__ = ["HOST_MALLOC_TUNABLES", "HostMallocRefused", "plan_host_malloc", "ensure_host_malloc", "stamp_host_malloc", "host_malloc_line",
           "line_fields"]

HOST_MALLOC_TUNABLES = {"MALLOC_MMAP_MAX_": "0", "MALLOC_TRIM_THRESHOLD_": "-1", "MALLOC_TOP_PAD_": "536870912"}      # 512 MB top pad
_MALLOPT = {"MALLOC_MMAP_MAX_": (-4, 0), "MALLOC_TRIM_THRESHOLD_": (-1, -1), "MALLOC_TOP_PAD_": (-2, 536870912)}      # glibc malloc.h: M_TRIM_THRESHOLD -1, M_TOP_PAD -2, M_MMAP_MAX -4


class HostMallocRefused(RuntimeError):
    """``mallopt`` did not accept a tunable on this libc: the host-malloc lever cannot apply (set ``<opt_out_var>=0`` for the stock allocator)."""


def plan_host_malloc(opt_out_var: str, environ: Optional[Mapping[str, str]] = None) -> Dict[str, object]:
    """``{active, present, source}``: inactive iff ``<opt_out_var>=0``; ``present`` = the three variables as found."""
    environ = os.environ if environ is None else environ
    present = {k: environ.get(k) for k in HOST_MALLOC_TUNABLES}
    if (environ.get(opt_out_var, "1") or "1") == "0":
        return {"active": False, "present": present, "source": f"{opt_out_var}=0 (opt-out: the stock's allocator behaviour)"}
    if all(present[k] == v for k, v in HOST_MALLOC_TUNABLES.items()):
        return {"active": True, "present": present, "source": "explicit env at process start (the user's own knob) + mallopt re-applied by the launcher"}
    return {"active": True, "present": present, "source": "launcher: mallopt in-process + the env exported for exec'd children"}


def ensure_host_malloc(opt_out_var: str, environ: Optional[MutableMapping[str, str]] = None) -> Dict[str, object]:
    """Apply the tunables through ``mallopt`` and export them into ``environ`` (default ``os.environ``); return the stamp
    ``{tunables, active, source, mallopt}``. Raises ``HostMallocRefused`` when a ``mallopt`` call is not accepted."""
    environ = os.environ if environ is None else environ
    p = plan_host_malloc(opt_out_var, environ)
    if not p["active"]:
        return {"tunables": p["present"], "active": False, "source": p["source"], "mallopt": None}
    import ctypes
    import ctypes.util
    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
    if not hasattr(libc, "mallopt"):
        raise HostMallocRefused(f"this libc has no mallopt — the host-malloc lever cannot apply (refused; set {opt_out_var}=0 to run with the stock allocator)")
    libc.mallopt.argtypes = (ctypes.c_int, ctypes.c_int)
    libc.mallopt.restype = ctypes.c_int
    rets = {}        # type: Dict[str, int]
    for k, (param, val) in _MALLOPT.items():
        rets[k] = int(libc.mallopt(param, val))
        if rets[k] != 1:
            raise HostMallocRefused(f"mallopt({param}, {val}) returned {rets[k]} — the host-malloc lever cannot apply on this libc (refused; set {opt_out_var}=0 to run with the stock allocator)")
    for k, v in HOST_MALLOC_TUNABLES.items():
        environ[k] = v
    return {"tunables": dict(HOST_MALLOC_TUNABLES), "active": True, "source": p["source"], "mallopt": rets}


def stamp_host_malloc(opt_out_var: str, environ: Optional[Mapping[str, str]] = None) -> Dict[str, object]:
    """``{tunables, active, source}`` as the environment shows them now (what a row records)."""
    p = plan_host_malloc(opt_out_var, environ)
    return {"tunables": p["present"], "active": p["active"], "source": p["source"]}


def line_fields(host_malloc: Mapping[str, object]) -> Dict[str, str]:
    """The activation-evidence fields (the seq modules' hook: str -> str) of a host-malloc stamp (``ensure_host_malloc`` /
    ``stamp_host_malloc``): ``host_malloc`` = ``on|off (<source>)``."""
    return {"host_malloc": f"{'on' if host_malloc.get('active') else 'off'} ({host_malloc.get('source')})"}


def host_malloc_line(stamp: Mapping[str, object]) -> str:
    """The activation-evidence clause: ``host_malloc=on|off (<source>)``."""
    return "host_malloc=" + line_fields(stamp)["host_malloc"]
