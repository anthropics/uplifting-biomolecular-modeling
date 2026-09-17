"""Cast cache: low-precision copies of long-lived parameters computed once, and in-place Linear pre-casting with a misuse guard.

Why. Under ``torch.autocast`` every ``F.linear`` / ``matmul`` casts its fp32 weight to the autocast dtype on EVERY call (autocast's own
cast cache lives only for one autocast region and only for leaf tensors that require grad); a 200-step sampler pays hundreds of cast kernels
and full fp32 weight reads per step. Serving a cached copy made by the same ``Tensor.to(dtype)`` gives the consumer operands that are
BIT-EXACT what autocast would have produced, so at a call site that stock already runs under autocast the substitution is exact by
construction (:func:`verify_identical` is the probe an exact mode runs once to attest it on its stack); at a call site stock runs in fp32
it changes the GEMM's operand precision and is tolerance-class — the kit's mode table says which, per engine.

Contract.
* :class:`CastCache` ``.get(p, dtype, transform=None)`` returns the cached ``transform(p.detach().to(dtype))`` for a LONG-LIVED tensor
  ``p`` (an ``nn.Parameter``, a registered buffer, or a tensor the adapter pins with ``pin=True``); the entry is keyed on ``id(p)`` AND holds
  ``p`` itself and its ``(data_ptr, _version, dtype, shape, device)``: a hit whose holder is not the same object, or whose storage /
  version / shape changed (weights reloaded or updated in place), is recast and counted as ``refreshes`` — a recycled ``id`` can never
  serve another tensor's copy. A per-call temporary (not a Parameter, not pinned) is cast WITHOUT caching and counted as ``uncached``:
  correct, just not faster, and visible in the census instead of silent.
* :func:`precast_linear_` converts the ``weight`` / ``bias`` of the selected ``nn.Linear`` modules to ``dtype`` IN PLACE (the fp32 masters
  are dropped: inference only) and, with ``guard=True``, installs a forward pre-hook that raises :class:`PrecastMisuse` when such a module
  is called with autocast disabled on a floating input of another dtype — the fp32-island case where an in-place precast would silently
  compute in low precision (or fail with a dtype error) instead of matching stock.
* :func:`census` / ``CastCache.census()`` -> the activation-evidence fields (``entries bytes hits misses refreshes uncached``); the adapter
  prints them once per process.
torch is imported inside the functions (``opt_core.precision.require_torch``); this module imports nothing heavy.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

from . import PrecisionError, require_torch

_DTYPE_WORDS = {"bf16": "bfloat16", "bfloat16": "bfloat16", "fp16": "float16", "float16": "float16", "half": "float16",
                "fp32": "float32", "float32": "float32"}


class PrecastMisuse(PrecisionError):
    """A pre-cast Linear was called outside autocast with an input of another floating dtype (an fp32 island)."""

    event = "precast_misuse"


def dtype_of(word, torch=None):
    """``"bf16"|"fp16"|"fp32"`` (or a torch dtype) -> torch dtype."""
    torch = require_torch(torch)
    if isinstance(word, str):
        if word not in _DTYPE_WORDS:
            raise ValueError("dtype word must be one of %s, not %r" % (sorted(_DTYPE_WORDS), word))
        return getattr(torch, _DTYPE_WORDS[word])
    return word


class CastCache:
    """dtype copies of long-lived tensors. One instance per adapter (``name`` tags its census)."""

    def __init__(self, name: str = "cast_cache", torch=None):
        self.name = name
        self._torch = torch
        self._entries: dict = {}
        self.hits = 0
        self.misses = 0
        self.refreshes = 0
        self.uncached = 0
        self.unsized = 0

    @staticmethod
    def _stamp(p):
        return (p.data_ptr(), p._version, p.dtype, tuple(p.shape), str(p.device))

    def get(self, p, dtype, transform: Optional[Callable] = None, pin: bool = False, key_extra=None):
        """The cached ``transform(p.detach().to(dtype))`` (``transform`` e.g. ``lambda t: t.t().contiguous()``; ``None`` = the cast alone).
        The transform's code object is part of the key (two call sites = two entries); ``key_extra`` distinguishes further.
        Non-Parameter, unpinned tensors are cast uncached (counted)."""
        torch = require_torch(self._torch)
        dtype = dtype_of(dtype, torch)
        cacheable = pin or isinstance(p, torch.nn.Parameter) or getattr(p, "_opt_core_pinned", False)
        if not cacheable:
            self.uncached += 1
            t = p.detach().to(dtype)
            return transform(t) if transform is not None else t
        key = (id(p), dtype, key_extra, getattr(transform, "__code__", transform))   # one entry per (tensor, dtype, transform site)
        ent = self._entries.get(key)
        stamp = self._stamp(p)
        if ent is not None:
            holder, old_stamp, out = ent
            if holder is p and old_stamp == stamp:
                self.hits += 1
                return out
            self.refreshes += 1
        else:
            self.misses += 1
        t = p.detach().to(dtype)
        out = transform(t) if transform is not None else t
        self._entries[key] = (p, stamp, out)
        return out

    def clear(self) -> int:
        n = len(self._entries)
        self._entries.clear()
        return n

    def census(self) -> dict:
        nbytes = 0
        unsized = 0
        for _, _, out in self._entries.values():
            try:
                nbytes += out.numel() * out.element_size()
            except (AttributeError, TypeError):          # a transform returned something that is not a tensor: counted, not hidden
                unsized += 1
        self.unsized = unsized
        return {"cache": self.name, "entries": len(self._entries), "bytes": nbytes, "hits": self.hits, "misses": self.misses,
                "refreshes": self.refreshes, "uncached": self.uncached, "unsized": unsized}


def verify_identical(p, dtype, cache: Optional[CastCache] = None, torch=None) -> bool:
    """The exactness probe: the cached copy equals a fresh ``p.detach().to(dtype)`` bit-exact (``torch.equal``). True by construction on one
    stack; an exact mode runs it once per process and refuses (not-active) on False rather than assuming."""
    torch = require_torch(torch)
    dtype = dtype_of(dtype, torch)
    cached = (cache or CastCache("probe", torch)).get(p, dtype, pin=True)
    return bool(torch.equal(cached, p.detach().to(dtype)))


def _guard_hook(dtype, torch):
    def hook(module, args):
        if not args:
            return None
        x = args[0]
        if not hasattr(x, "dtype") or not x.is_floating_point() or x.dtype == dtype:
            return None
        dev = x.device.type
        try:
            on = torch.is_autocast_enabled(dev)
        except TypeError:                                   # older torch: per-device functions
            on = torch.is_autocast_enabled() if dev == "cuda" else torch.is_autocast_cpu_enabled()
        if on:
            return None
        raise PrecastMisuse("pre-cast %s(%s) called outside autocast with a %s input: an fp32 island — leave this module un-cast "
                            "(predicate) or open the autocast region the stock code runs it under" % (type(module).__name__, dtype, x.dtype),
                            module=type(module).__name__, weight_dtype=str(dtype), input_dtype=str(x.dtype))
    return hook


def precast_linear_(root, dtype="bf16", predicate: Optional[Callable[[str, object], bool]] = None, types: Optional[Iterable] = None,
                    guard: bool = True, torch=None) -> dict:
    """Convert ``weight`` and ``bias`` of every selected module under ``root`` to ``dtype`` in place. Selected = instances of ``types``
    (default ``nn.Linear``) for which ``predicate(qualified_name, module)`` is true (default: all). ``guard`` installs the
    :class:`PrecastMisuse` pre-hook on each. Returns the activation record ``{"precast": dtype word, "modules": n, "params": n (distinct
    Parameters converted; a weight tied across modules is converted once and stays shared), "bytes_before": b, "bytes_after": b,
    "guard": on|off, "skipped": n (predicate), "skipped_parametrized": n (parametrized tensors are left alone, counted)}``.
    Inference only: the fp32 masters are not kept."""
    torch = require_torch(torch)
    dt = dtype_of(dtype, torch)
    kinds = tuple(types) if types is not None else (torch.nn.Linear,)
    try:
        from torch.nn.utils.parametrize import is_parametrized  # noqa: PLC0415
    except ImportError:                                                    # very old torch: no parametrizations to protect
        def is_parametrized(module, tensor_name=None):
            return False
    n_mod = n_par = skipped = skipped_parametrized = 0
    b_before = b_after = 0
    converted = {}                                                         # id(old Parameter) -> (old, new): tied weights stay tied
    for name, m in root.named_modules():
        if not isinstance(m, kinds):
            continue
        if predicate is not None and not predicate(name, m):
            skipped += 1
            continue
        touched = False
        for pname in ("weight", "bias"):
            p = getattr(m, pname, None)
            if p is None or not torch.is_floating_point(p) or p.dtype == dt:
                continue
            if is_parametrized(m, pname):
                skipped_parametrized += 1
                continue
            prev = converted.get(id(p))
            if prev is None or prev[0] is not p:
                b_before += p.numel() * p.element_size()
                with torch.no_grad():
                    q = torch.nn.Parameter(p.detach().to(dt), requires_grad=False)
                converted[id(p)] = (p, q)
                b_after += q.numel() * q.element_size()
                n_par += 1
            else:
                q = prev[1]
            setattr(m, pname, q)
            touched = True
        if touched:
            n_mod += 1
            if guard and not getattr(m, "_opt_core_precast_guard", False):
                m.register_forward_pre_hook(_guard_hook(dt, torch))
                m._opt_core_precast_guard = True
    return {"precast": str(dt).replace("torch.", ""), "modules": n_mod, "params": n_par, "bytes_before": b_before, "bytes_after": b_after,
            "guard": "on" if guard else "off", "skipped": skipped, "skipped_parametrized": skipped_parametrized}


def pin(t):
    """Mark a long-lived non-Parameter tensor (e.g. a registered buffer) as cacheable by :meth:`CastCache.get`. Returns ``t``."""
    t._opt_core_pinned = True
    return t
