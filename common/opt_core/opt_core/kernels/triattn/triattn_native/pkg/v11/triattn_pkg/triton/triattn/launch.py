"""Low-overhead repeat launches of @triton.jit / @gluon.jit kernels.

`launch(cache, fn, grid, args, meta, **opts)` == `fn[grid](*args, **meta, **opts)`, but after the first call with a given
(meta, opts, runtime specialisation of args) it reuses the CompiledKernel handle and launches it directly, skipping the JIT
dispatcher's Python argument binding -- the dominant eager-mode cost for kernels with dozens of arguments (~100 us per call).
The cache key mirrors Triton's own runtime specialisation (ints: == 1 and % 16; tensors: dtype and 16-byte alignment; tensor
descriptors: dtype, block shape, layout), so a cached handle is only reused for arguments it was compiled for.  `meta` holds the
constexpr arguments (they must be declared after all runtime arguments in fn's signature), `opts` the compile options
(num_warps, num_stages, maxnreg).
"""
from __future__ import annotations

import torch

_CONSTEXPR_NAMES = {}


def _constexpr_names(fn):
    names = _CONSTEXPR_NAMES.get(fn)
    if names is None:
        names = [p.name for p in fn.params if p.is_constexpr]
        _CONSTEXPR_NAMES[fn] = names
    return names


def _spec(a):
    if isinstance(a, bool):
        return ("b", a)
    if isinstance(a, int):
        return ("i", a == 1, a % 16 == 0)
    if isinstance(a, float):
        return "f"
    if isinstance(a, torch.Tensor):
        return ("t", a.dtype, a.data_ptr() % 16 == 0)
    if a is None:
        return None
    base = getattr(a, "base", None)          # gluon TensorDescriptor
    return ("d", getattr(base, "dtype", None), tuple(getattr(a, "block_shape", ()) or ()), repr(getattr(a, "layout", None)))


def call_key(*tensors, extra=()):
    """A cache key covering everything Triton specialises on for calls whose scalar arguments are all functions of these
    tensors' shapes/strides (+ `extra`): shapes, strides, dtypes, 16-byte pointer alignment."""
    return tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.data_ptr() % 16 == 0) if t is not None else None for t in tensors) + tuple(extra)


def launch(cache: dict, fn, grid, args: tuple, meta: dict, key=None, **opts):
    if key is None:
        key = tuple(_spec(a) for a in args)
    key = (key, tuple(sorted(meta.items())), tuple(sorted(opts.items())))
    ck = cache.get(key)
    if ck is None:
        ck = fn[grid](*args, **meta, **opts)
        cache[key] = ck
        return ck
    g3 = tuple(grid) + (1,) * (3 - len(grid))
    ck[g3](*args, *[meta[n] for n in _constexpr_names(fn)])
    return ck
