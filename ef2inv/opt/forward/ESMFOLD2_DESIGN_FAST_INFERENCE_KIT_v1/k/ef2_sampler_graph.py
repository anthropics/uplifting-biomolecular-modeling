"""ef2_sampler_graph — CUDA-graph replay of the structure module's denoising network inside DiffusionStructureHead.sample().

`sample()` runs `diffusion_module(x_noisy=..., t_hat=..., <fold features>, inference_cache=cache)` once per denoising step under
no_grad: 1 step per design-step fold, 34 at the confidence phase (num_sampling_steps=50 after the sigma cap), 200 in a hero-critic fold.
From the second step on, every call of one sample() has identical shapes, flags and argument OBJECTS (the features and the step-invariant
tensors the fork fills ONCE into `inference_cache` and reuses: conditioned pair z, DiffusionConditioning.forward l.1385-1395; atom-encoder
statics c_base / 3D-RoPE / varlen indices / n_tokens, atom encoder l.849-886 — modeling_esmfold2_common.py at the pin; the loop itself is
DiffusionStructureHead.sample l.1840-1925); only `x_noisy` and `t_hat` change.  At design
sizes the call is launch-bound (hundreds of small kernels per step; host launch time exceeds GPU time), so this lever captures
the call once per sample() — after the cache-filling first step and one eager warm-up step — into a CUDA graph with static x_noisy / t_hat
buffers and replays it for the remaining steps; the sampler's own per-step host work between calls (centre/augment, the Karras .item()
reads, the Kabsch SVD) stays eager.  Capture runs on a side stream with capture_error_mode='thread_local' (allocator activity of other
threads' streams cannot abort it) and WITHOUT the gc.collect()/empty_cache() of torch.cuda.graph's context manager (no allocator flush per
confidence step) — except once, by name, when a capture runs out of device memory: the capture's private pool cannot draw on the
ordinary pool's cached-but-free segments, so that one capture is retried after empty_cache() ('capture_retries'; a second failure, or any
other, is 'capture_failed' and the sample() continues eager).  Numerics: EXACT (replay runs the captured kernels on the same arguments; bitwise to eager
under the det recipe — unit test + notes).  The graph, its private pool and the references it holds live for that one sample() call and
are dropped when it returns: no captured pointer can outlive the objects it points to, nothing pair-sized
survives the fold, and a call whose argument objects / shapes / cache keys differ from the captured ones runs eager by name
('mismatch'), as do calls under grad, without a warm cache, or asking for atom intermediates ('eager').  Outputs are returned as fresh
tensors (clone of the static output: atoms x 3 and tokens x 768 floats), never as views of graph memory.

Usage:
    import ef2_sampler_graph as sg
    sg.enable(model)             # wraps structure_head.sample and structure_head.diffusion_module.forward (instance-level)
    sg.describe(model) -> "sampler_graph captures=.. replays=.. eager=.. mismatch=.. capture_failed=.."
    sg.disable(model)
Composes with ef2_pairbias_attn (either order of enable/disable): with its 'hoist' the captured step contains no pair-bias kernels.
"""
from __future__ import annotations

import torch
from transformers.models.esmfold2 import modeling_esmfold2_common as C

TAG = "ef2_sampler_graph"
N_EAGER_WARM = 1            # eager steps with a warm cache before capture (step 0 fills the cache and is always eager)


# ---- chain-aware instance-method wrapping (shared convention with ef2_pairbias_attn: several levers may wrap sample()) ----
class _Wrapped:
    """`obj.name` replaced by this callable: calls fn(self, *a, **kw); fn delegates to self._ef2_prev (the callable it wrapped —
    the bound method or another lever's wrapper) and reads the module as self.obj."""

    def __init__(self, fn, prev, tag, obj):
        self.fn, self._ef2_prev, self._ef2_tag, self.obj = fn, prev, tag, obj

    def __call__(self, *a, **kw):
        return self.fn(self, *a, **kw)


def _wrap_method(obj, name: str, fn, tag: str) -> None:
    setattr(obj, name, _Wrapped(fn, getattr(obj, name), tag, obj))


def _unwrap_method(obj, name: str, tag: str) -> bool:
    cur, parent = getattr(obj, name, None), None
    while cur is not None and getattr(cur, "_ef2_tag", None) != tag:
        parent, cur = cur, getattr(cur, "_ef2_prev", None)
    if cur is None:
        return False
    if parent is not None:
        parent._ef2_prev = cur._ef2_prev
    elif getattr(cur._ef2_prev, "_ef2_tag", None) is None and name in vars(obj):
        delattr(obj, name)                       # this was the only wrapper: back to the class's method
    else:
        setattr(obj, name, cur._ef2_prev)
    return True


# ---- per-sample() scope holding the graph ----
class _Scope:
    __slots__ = ("graph", "sig", "held", "static_x", "static_t", "static_out", "n_warm", "failed")

    def __init__(self):
        self.graph = None
        self.sig = None
        self.held = None
        self.static_x = self.static_t = self.static_out = None
        self.n_warm = 0
        self.failed = None


OOM_ERRORS = tuple(t for t in (getattr(torch, "OutOfMemoryError", None), getattr(torch.cuda, "OutOfMemoryError", None)) if isinstance(t, type))   # torch's CUDA out-of-memory exception class(es) at this pin


class _State:
    def __init__(self):
        self.stats = dict(samples=0, captures=0, replays=0, eager=0, mismatch=0, capture_failed=0, capture_retries=0)
        self.last_failure = None


def _signature(kw: dict, cache: dict):
    """Everything a replay depends on besides the two staticised inputs: the identity of every other argument object, the shapes /
    dtypes / devices of x_noisy and t_hat, and the cache's key structure (held objects cannot be recycled while the scope lives)."""
    x, t = kw["x_noisy"], kw["t_hat"]
    idents = tuple((k, id(v)) for k, v in kw.items() if k not in ("x_noisy", "t_hat"))
    ckeys = (tuple(cache.keys()), tuple(cache["atomencoder"].keys()) if isinstance(cache.get("atomencoder"), dict) else None)
    return (idents, tuple(x.shape), x.dtype, x.device, tuple(t.shape), t.dtype, t.device, ckeys,
            torch.is_autocast_enabled(), torch.is_inference_mode_enabled())


def _dm_forward(w, *args, **kw):
    prev, dm = w._ef2_prev, w.obj
    st: _State = dm._sg_state
    sc: _Scope | None = dm._sg_scope
    cache = kw.get("inference_cache")
    graphable = (sc is not None and sc.failed is None and not args and not torch.is_grad_enabled()
                 and isinstance(cache, dict) and "z" in cache and cache.get("atomencoder")
                 and not kw.get("return_atom_repr", False)
                 and torch.is_tensor(kw.get("x_noisy")) and kw["x_noisy"].is_cuda
                 and torch.is_tensor(kw.get("t_hat")) and kw["t_hat"].is_cuda)
    if not graphable:
        st.stats["eager"] += 1
        return prev(*args, **kw)
    sig = _signature(kw, cache)
    if sc.graph is None:
        if sc.n_warm < N_EAGER_WARM:
            sc.n_warm += 1
            st.stats["eager"] += 1
            return prev(**kw)
        for attempt in (0, 1):                                   # a capture that runs out of device memory is retried ONCE after the caching allocator returns
            try:                                                 # its unused segments (a capture allocates from a private pool: cached-but-free blocks of the
                sc.static_x = kw["x_noisy"].clone()              # ordinary pool are out of its reach until empty_cache); any other failure, or a second one, is
                sc.static_t = kw["t_hat"].clone()                # named (describe(): capture_failed, last_failure) and this sample() continues eager
                sc.held = dict(kw)                               # keep every captured argument object alive for the scope
                g = torch.cuda.CUDAGraph()
                cur = torch.cuda.current_stream()
                side = torch.cuda.Stream()
                side.wait_stream(cur)
                cur.synchronize()
                with torch.cuda.stream(side):
                    g.capture_begin(capture_error_mode="thread_local")
                    try:
                        sc.static_out = prev(**{**kw, "x_noisy": sc.static_x, "t_hat": sc.static_t})
                    finally:
                        g.capture_end()
                cur.wait_stream(side)
                sc.graph, sc.sig = g, sig
                st.stats["captures"] += 1
                break
            except Exception as e:  # noqa: BLE001
                sc.graph = sc.static_out = sc.held = None
                g = None
                torch.cuda.synchronize()
                if attempt == 0 and isinstance(e, OOM_ERRORS):
                    st.stats["capture_retries"] += 1
                    torch.cuda.empty_cache()                     # the cached segments back to the driver: the retry's private pool can be served
                    continue
                sc.failed = e
                st.stats["capture_failed"] += 1
                st.last_failure = repr(e)
                return prev(**kw)
    elif sig != sc.sig:
        st.stats["mismatch"] += 1
        return prev(**kw)
    sc.static_x.copy_(kw["x_noisy"])
    sc.static_t.copy_(kw["t_hat"])
    sc.graph.replay()
    st.stats["replays"] += 1
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sc.static_out.items()}


def _scoped_sample(w, *args, **kw):
    dm = w.obj.diffusion_module
    st: _State = dm._sg_state
    outer = dm._sg_scope is None
    if outer:
        dm._sg_scope = _Scope()
        st.stats["samples"] += 1
    try:
        return w._ef2_prev(*args, **kw)
    finally:
        if outer:
            dm._sg_scope = None                                  # graph, pool, static buffers and held references dropped here


def enable(model: torch.nn.Module) -> _State:
    """Wrap sample() and diffusion_module.forward of every DiffusionStructureHead in `model` (instance-level; reversible)."""
    if hasattr(model, "_sg_state"):
        return model._sg_state
    heads = [m for m in model.modules() if isinstance(m, C.DiffusionStructureHead)]
    if not heads:
        raise RuntimeError("no DiffusionStructureHead in this model: nothing to graph")
    st = _State()
    for sh in heads:
        sh._sg_state = sh.diffusion_module._sg_state = st
        sh.diffusion_module._sg_scope = None
        _wrap_method(sh.diffusion_module, "forward", _dm_forward, TAG)
        _wrap_method(sh, "sample", _scoped_sample, TAG)
    model._sg_state = st
    return st


def disable(model: torch.nn.Module) -> None:
    for sh in [m for m in model.modules() if isinstance(m, C.DiffusionStructureHead)]:
        _unwrap_method(sh, "sample", TAG)
        _unwrap_method(sh.diffusion_module, "forward", TAG)
        for o, n in ((sh, "_sg_state"), (sh.diffusion_module, "_sg_state"), (sh.diffusion_module, "_sg_scope")):
            if hasattr(o, n):
                delattr(o, n)
    if hasattr(model, "_sg_state"):
        del model._sg_state


def describe(model) -> str:
    st = getattr(model, "_sg_state", None)
    if st is None:
        return "sampler_graph=off"
    s = st.stats
    extra = f" last_failure={st.last_failure}" if st.last_failure else ""
    return (f"sampler_graph samples={s['samples']} captures={s['captures']} replays={s['replays']} eager={s['eager']} "
            f"mismatch={s['mismatch']} capture_failed={s['capture_failed']} capture_retries={s['capture_retries']}{extra}")
