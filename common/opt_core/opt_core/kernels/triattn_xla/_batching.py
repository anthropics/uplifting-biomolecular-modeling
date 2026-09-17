"""vmap rules for the FFI launches.  `jax.ffi.ffi_call` has no batching rule of its own, and a launch spec is registered for ONE static shape, so
a trace under jax.vmap (hk.vmap, a vmapped confidence head inside a layer_stack scan, ...) needs a rule.  Two layers:
  (1) every ffi_call is built with vmap_method="sequential" where the running jax accepts it: a vmapped trace the rules below do not cover
      still runs, one launch per sample, each with the registered (unbatched) shapes;
  (2) `fold_into_batch(fixed, generic)` wraps a launch in jax.custom_batching.custom_vmap with the rule: move the vmapped axis into the
      kernel's own batch axis B (q/k/v [V*B, N, ...], bias [V*B, H, S, S], mask [V*B, N, S]; operands that are not vmapped are broadcast) and
      launch ONCE through `generic` (which re-checks the envelope for the folded shape); when the folded call is refused, the samples are
      launched one by one (unrolled) -- the rows are independent per batch element, so either way every sample gets the bytes an unbatched
      call gives.  `per_sample(fn)` is the unrolled rule alone (the differentiable row's backward, whose kernels share one bias per call).
The fold rule restates, for every row and inside the package, the wrapper the AF3 JAX kit wrote around this face for its vmapped confidence
head (same authorship and licence as this repository).  Where jax.custom_batching is absent the wrappers are the identity and (1) applies;
report()["vmap"] says which."""
from typing import Callable, Optional

_STATE = {"custom_vmap": None, "folded": 0, "unrolled": 0, "sequential_ffi": None}


def custom_vmap_available() -> Optional[str]:
    """None when jax.custom_batching.custom_vmap is importable, else the reason."""
    if _STATE["custom_vmap"] is None:
        try:
            from jax import custom_batching        # noqa: F401
            _ = custom_batching.custom_vmap
            _STATE["custom_vmap"] = ""
        except Exception as e:                      # noqa: BLE001
            _STATE["custom_vmap"] = "%s: %s" % (type(e).__name__, str(e)[:200])
    return _STATE["custom_vmap"] or None


def _bcast(x, batched: bool, V: int):
    import jax.numpy as jnp
    return x if batched else jnp.broadcast_to(x[None], (V,) + tuple(x.shape))


def fold_into_batch(fixed: Callable, generic: Callable, n_out: int = 1) -> Callable:
    """fixed(*arrays) for the traced shapes; generic(*arrays) for any leading batch size (raises Refused outside the envelope).  Arrays: q, k, v
    [B, N, ...], bias [B, H, S, S] and optionally mask [B, N, S] -- all with the kernel batch axis B leading.  Returns fixed wrapped with the fold rule."""
    if custom_vmap_available() is not None:
        return fixed
    import jax.numpy as jnp
    from jax import custom_batching
    from . import Refused

    @custom_batching.custom_vmap
    def call(*arrays):
        return fixed(*arrays)

    @call.def_vmap
    def rule(axis_size, in_batched, *arrays):
        V = int(axis_size)
        xs = [_bcast(x, bool(b), V) for x, b in zip(arrays, in_batched)]          # every operand [V, B, ...]
        B = int(xs[0].shape[1])
        folded = [x.reshape((V * B,) + tuple(x.shape[2:])) for x in xs]
        try:
            out = generic(*folded)
            outs = list(out) if n_out > 1 else [out]
            outs = [o.reshape((V, B) + tuple(o.shape[1:])) for o in outs]
            _STATE["folded"] += 1
        except Refused:                                                          # the folded envelope is not served (e.g. the bias-flag budget): per sample, same kernel as unbatched
            per = [generic(*[x[i] for x in xs]) for i in range(V)]
            outs = [jnp.stack([p[j] for p in per]) for j in range(n_out)] if n_out > 1 else [jnp.stack(per)]
            _STATE["unrolled"] += 1
        if n_out > 1:
            return tuple(outs), tuple(True for _ in outs)
        return outs[0], True

    return call


def per_sample(fn: Callable, n_out: int) -> Callable:
    """fn wrapped with an unrolled per-sample vmap rule (outputs: a tuple of n_out arrays)."""
    if custom_vmap_available() is not None:
        return fn
    import jax.numpy as jnp
    from jax import custom_batching

    @custom_batching.custom_vmap
    def call(*arrays):
        return fn(*arrays)

    @call.def_vmap
    def rule(axis_size, in_batched, *arrays):
        V = int(axis_size)
        xs = [_bcast(x, bool(b), V) for x, b in zip(arrays, in_batched)]
        per = [fn(*[x[i] for x in xs]) for i in range(V)]
        _STATE["unrolled"] += 1
        outs = tuple(jnp.stack([p[j] for p in per]) for j in range(n_out))
        return outs, tuple(True for _ in outs)

    return call


def status() -> dict:
    why = custom_vmap_available()
    return {"rule": "custom_vmap: fold the vmapped axis into B (one launch), unrolled per sample when the folded call is refused" if why is None
            else "none (%s) -- ffi_call vmap_method=%s" % (why, _STATE["sequential_ffi"]),
            "ffi_vmap_method": _STATE["sequential_ffi"], "folded": _STATE["folded"], "unrolled": _STATE["unrolled"]}
