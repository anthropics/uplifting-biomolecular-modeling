"""ew1_i64 — ew1's Triton kernels with int64 offsets. kernels.py here is ew1/kernels.py with the program / row index widened to
int64 (`tl.program_id(0).to(tl.int64)` in _silu_mul_kernel, _silu_kernel and _gelu_kernel, the rope row ids of _rope_rows, the row ids of
_add_rmsnorm_kernel and _embed_add_kernel). Below 2^31 elements the arithmetic is ew1's (bitwise by construction: the same fp32 opmath, the
same roundings; only the index type widens); above it ew1's int32 offsets wrap silently and these kernels index correctly.

A component, never applied alone: the patch sites stay ew1's (ew1/patches.py binds the instance forwards and calls `_K()` = the
ew1.kernels MODULE). `install()` rebinds the six kernel entry points ON that module object (ENTRY_POINTS: clamp_rope_qkv / silu_mul /
silu_bf16 / gelu_bf16 / add_rmsnorm / embed_add) to this dir's functions — ew1's files untouched, its counters unchanged (they count at
the patch sites). `uninstall()` puts ew1's own functions back. The eager kit (kits/eager) composes it on every apply.
"""
from __future__ import annotations

import os

KIT = "ew1_i64"
_HERE = os.path.dirname(os.path.abspath(__file__))
ENTRY_POINTS = ("clamp_rope_qkv", "silu_mul", "silu_bf16", "gelu_bf16", "add_rmsnorm", "embed_add")
UPSTREAM_PATCH = {"sha256": "be5b1bed0a39f7e6dfd68d093475f2be8ebf7578a5a3e39c0f706a91fc73a57c", "lines": 6}
_S = {"installed": False, "orig": {}}


class ComponentRefused(RuntimeError):
    pass



def install() -> dict:
    """Rebind the kernel entry points on the ew1.kernels MODULE object (what ew1/patches.py `_K()` returns) to this dir's."""
    if _S["installed"]:
        return {"installed": True, "already": True}
    from ..ew1 import kernels as K_frozen
    from . import kernels as K_i64
    for name in ENTRY_POINTS:
        if not hasattr(K_frozen, name) or not hasattr(K_i64, name):
            raise ComponentRefused(f"ew1_i64: entry point {name} missing on {'ew1.kernels' if not hasattr(K_frozen, name) else 'ew1_i64.kernels'}")
    for name in ENTRY_POINTS:
        _S["orig"][name] = getattr(K_frozen, name)
        setattr(K_frozen, name, getattr(K_i64, name))
    _S["installed"] = True
    return {"installed": True, "entry_points": list(ENTRY_POINTS)}


def uninstall() -> None:
    if not _S["installed"]:
        return
    from ..ew1 import kernels as K_frozen
    for name, fn in _S["orig"].items():
        setattr(K_frozen, name, fn)
    _S.update({"installed": False, "orig": {}})


def in_force() -> bool:
    """True iff every entry point on ew1.kernels IS this dir's function (the composition's own in-force check)."""
    if not _S["installed"]:
        return False
    from ..ew1 import kernels as K_frozen
    from . import kernels as K_i64
    return all(getattr(K_frozen, n) is getattr(K_i64, n) for n in ENTRY_POINTS)
