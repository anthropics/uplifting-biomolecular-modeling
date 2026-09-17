"""The item-keyed hoist: the eager stack's denoiser wrapper keyed on the ITEM, its state released at every item boundary.

The carried ``chai1_eager.stack.HoistedDiffusionWrapper`` keys its hoisted precompute (and CUDA graph) on the crop, the DATA POINTERS
of two static inputs and the coordinate shape (its ``forward``: ``key = (crop_size, ...data_ptr(), ...data_ptr(), shape)``). In a
process that folds several items, a new item's static tensors can land at the addresses the previous item's freed ones had: the key
matches, the stale precompute is reused, and the item is denoised with the previous item's conditioning. ``ItemKeyedDenoiser`` keys
on the item instead — the static input tensor objects upstream's ``run_folding_on_context`` builds once per item (weak
references: a collected tensor never matches, whatever address its successor lands on) plus the crop and every tensor input's shape
and dtype — and on a new item drops the previous item's state (the precompute cache, the CUDA graph with its private pool, the
graph's static tensors) before the stack re-hoists through its own ``forward``. ``ItemBoundaryTrunk`` drops that state at the first
trunk call of every item, so the trunk of item k+1 runs with nothing of item k resident: a multi-item process holds one item's
denoiser state at a time (the memory lever; ``RELEASE`` names it on the activation line).

The state lives for the item's DIFFUSION PHASE only. Upstream's fold body runs the confidence head after the last denoiser step
(``chai1.py`` run_folding_on_context: ``del diffusion_module`` then ``confidence_head.forward`` per sample) and nothing reads the hoisted
tensors again, so ``adopt(parts, S, handle=h)`` also wraps the handle's loader ONCE (the ``pairtrack`` exactln precedent:
``chai1_eager.stack.make_loader`` serves ``confidence_head.pt`` through it): the part served for the confidence head releases every adopted
denoiser's item before its forward (``ConfidenceEntry``; ``release_all`` — a no-op from the second sample on). The item's hoist cache
(hoist2: the 16 hoisted pair biases [1,16,N,N] and the blocked atom-pair masks / indices over 23·N atoms, quadratic in N) and, under a graphed row, the step's
graph pool are back with the allocator before the confidence head's [N, N, 4·c] transients — the fold's peak on the memory line — are
drawn. Bookkeeping only: outputs bitwise, no re-hoist (the next item hoists its own; a second seed of the same input is a new fold with new
static tensors either way), counted on the same ``n_release`` (EXIT ``hoist_releases=``: one per completed item). The trunk-boundary
release stays for an item whose confidence head never ran (a fold that failed inside its sampler).

``adopt(parts, S)`` re-classes the installed parts in place — the stack's own instances keep every attribute (the add-on's policies,
the stack's ``hf`` / ``events``) — so the carried files stay unedited. The counters (``n_precompute``, ``n_release``) feed the exit
tally: ``hoist_precomputes=<n>/<items>`` — one hoist per item that reached its denoiser, never more (a per-call object would say so), never
fewer among the completed ones (a stale hit would say so: ``verdict``); folds that failed before the denoiser ran are counted apart
(``unreached`` -> ``hoist_unreached=<k>``); the driver route refuses at exit on a verdict (``HOIST MISMATCH``).
"""
from __future__ import annotations

import sys
import weakref
from typing import Optional, Tuple

ITEM_TENSORS = ("token_pair_trunk_repr", "atom_single_input_feats")   # the static inputs whose objects (compared with `is`) name the item (upstream builds one set per item)
KEYED = "item"
RELEASE = "trunk_boundary"
STATE = {"keyed": KEYED, "release": RELEASE}
HOIST_FIELDS = ("cache", "_graph", "_static_kw", "_static_out")      # HoistedForward's per-item state (chai1_eager/hoist.py precompute / step_graphed)
CONF_KEY = "confidence_head.pt"                                       # the component upstream's fold body runs after the item's last denoiser step (chai1.py: load_exported key)
LOADER_ATTR = "chai1_opt_hoist"                                       # marks the handle's loader once wrapped (adopt(..., handle=h); idempotent)

_LIVE: "weakref.WeakSet" = weakref.WeakSet()
_CLASSES = {}
_COUNTS = {"conf_entry": 0}                                           # releases that happened at a confidence head's entry (the rest: trunk boundary / restore)


class ConfidenceEntry:
    """The part served for ``confidence_head.pt`` with the item's hoisted denoiser state released before its forward: the diffusion phase is
    over when the confidence head runs (``release_all``; nothing resident from the second sample's call on = a no-op). Every other attribute
    is the served part's own (``flat`` / ``jit_module`` / ``inner`` pass through)."""
    chai1_opt_lever = "hoist"

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def forward(self, *args, **kw):
        if release_all():
            _COUNTS["conf_entry"] += 1
        return self._inner.forward(*args, **kw)

    def __call__(self, *args, **kw):
        if release_all():
            _COUNTS["conf_entry"] += 1
        return self._inner(*args, **kw)


def serve(handle, parts: Optional[dict] = None) -> bool:
    """Wrap ``handle.loader`` once so the confidence head it serves is a ``ConfidenceEntry`` (and keep ``handle.C1.load_exported`` on the
    wrapper when it held the loader, as ``pairtrack``'s exactln wrap does — the activation's loader-identity check reads both). False when
    there is nothing to wrap: no loader, already wrapped, or parts that serve no flat confidence head (``flat_rest`` false: the TorchScript
    head is upstream's own object; the trunk-boundary release covers that line)."""
    loader = getattr(handle, "loader", None)
    if not callable(loader) or getattr(loader, LOADER_ATTR, False):
        return False
    if parts is not None and not parts.get("flat_rest", True):
        return False

    def load_exported(comp_key, device, _inner=loader):
        part = _inner(comp_key, device)
        return ConfidenceEntry(part) if str(comp_key) == CONF_KEY else part
    setattr(load_exported, LOADER_ATTR, True)
    load_exported.__wrapped__ = loader
    load_exported.__name__ = getattr(loader, "__name__", "load_exported")     # the wrapped loader's own name (activation words that print it read the same)
    C1 = getattr(handle, "C1", None)
    if C1 is not None and getattr(C1, "load_exported", None) is loader:
        C1.load_exported = load_exported
    handle.loader = load_exported
    return True


def static_key(crop_size, kw: dict) -> tuple:
    """``(crop, (name, shape, dtype) of every tensor input in name order)``: the shape signature of an item."""
    return (crop_size,) + tuple((k, tuple(v.shape), str(v.dtype)) for k, v in sorted(kw.items()) if hasattr(v, "shape") and hasattr(v, "dtype"))


def same_item(refs: Tuple[weakref.ref, ...], kw: dict) -> bool:
    """True when every ITEM_TENSORS entry of ``kw`` is the very object the hoist was keyed on (a collected one never matches)."""
    return len(refs) == len(ITEM_TENSORS) and all(r() is kw[n] for r, n in zip(refs, ITEM_TENSORS))


def _empty_cache() -> None:
    torch = sys.modules.get("torch")                      # imported by the stack before any part exists; absent = no allocator to empty
    if torch is not None:
        torch.cuda.empty_cache()


def release_all() -> int:
    """Release every adopted denoiser's hoisted item; returns how many held one."""
    return sum(1 for d in list(_LIVE) if d.release())


def classes(S):
    """The two subclasses over the carried module ``S`` (``chai1_eager.stack``), built once per module object."""
    if id(S) in _CLASSES:
        return _CLASSES[id(S)]

    class ItemKeyedDenoiser(S.HoistedDiffusionWrapper):
        n_precompute = 0; n_release = 0; _sig = None; _item: tuple = ()      # class defaults: an adopted instance starts from them

        def forward(self, crop_size, *, return_on_cpu=False, move_to_device=None, **kw):
            kw = S._move(kw, move_to_device)
            sig = static_key(crop_size, kw)
            if sig != self._sig or not same_item(self._item, kw):
                self.release()                                              # the stack's own forward hoists on key None
                self.n_precompute += 1
                self._sig = sig; self._item = tuple(weakref.ref(kw[n]) for n in ITEM_TENSORS)
            return super().forward(crop_size, return_on_cpu=return_on_cpu, **kw)

        def release(self) -> bool:
            """Drop the hoisted item's state in every crop and return the graph pools to the device; True when something was resident."""
            if self.key is None and self._sig is None:
                return False
            for hf in self.hf.values():
                for f in HOIST_FIELDS:
                    setattr(hf, f, None)
            self.key = None; self._sig = None; self._item = ()
            self.n_release += 1
            _empty_cache()
            return True

    class ItemBoundaryTrunk(S.EagerTrunkWrapper):
        def forward(self, crop_size, **kw):
            release_all()                                                    # the trunk runs first in every item
            return super().forward(crop_size, **kw)

    _CLASSES[id(S)] = (ItemKeyedDenoiser, ItemBoundaryTrunk)
    return _CLASSES[id(S)]


def adopt(parts: dict, S, handle=None) -> dict:
    """Re-class ``parts['diffusion']`` / ``parts['trunk']`` (the stack's own instances) in place and, given the installed ``handle``
    (``chai1_eager.stack.StackHandle``: ``stack.apply_eager`` passes the one it keeps), wrap its loader so the confidence head releases the
    item's denoiser state at its entry (``serve``); returns ``STATE`` for the report."""
    D, T = classes(S)
    dw, tw = parts.get("diffusion"), parts.get("trunk")
    if type(dw) is not S.HoistedDiffusionWrapper or type(tw) is not S.EagerTrunkWrapper:
        raise TypeError(f"the installed parts are not the stack's own wrappers: diffusion={type(dw).__name__} trunk={type(tw).__name__}")
    dw.__class__ = D; tw.__class__ = T
    _LIVE.add(dw)
    if handle is not None:
        serve(handle, parts)
    return dict(STATE)


def conf_entry_releases() -> int:
    """How many releases happened at a confidence head's entry in this process (the rest were trunk-boundary releases)."""
    return int(_COUNTS["conf_entry"])


def stats(dw) -> Optional[dict]:
    """``{keyed, release, n_precompute, n_release}`` of an adopted denoiser; None for any other object."""
    if not isinstance(dw, tuple(c[0] for c in _CLASSES.values())):
        return None
    return dict(STATE, n_precompute=dw.n_precompute, n_release=dw.n_release)


def verdict(st: Optional[dict], items: Optional[int], ok: Optional[int] = None) -> Optional[str]:
    """None when every seed-fold that could reach its denoiser had exactly one hoist; else the named MISMATCH the exit tally and the driver
    route refuse on. ``items`` = fold calls attempted (report.items_folded: seed-folds × trunk samples — every run_folding_on_context call
    hoists its own trunk sample's tensors once; the seed-independent feature context and ESM embeddings are built once per input and reused
    across seeds and trunk samples), ``ok`` = fold calls of the completed seed-folds (report.items_ok). Fewer hoists
    than attempts is NOT a mismatch when every completed fold is covered (``n >= ok``): the missing ones are folds that ended — failed —
    before their denoiser ran (``unreached`` names that count); a completed fold without a hoist of its own (``n < ok``) means a stale
    precompute was served; more hoists than folds means an item was hoisted twice."""
    if st is None or items is None:
        return None
    n = st.get("n_precompute")
    if n == items:
        return None
    if n > items:
        return f"hoist_precomputes={n}>{items}: an item was hoisted more than once"
    covered = items if ok is None else ok
    if n >= covered:
        return None
    return f"hoist_precomputes={n}<{covered} completed seed-folds: a completed item was denoised without a hoist of its own (a stale precompute served)"


def unreached(st: Optional[dict], items: Optional[int], ok: Optional[int] = None) -> Optional[int]:
    """How many attempted seed-folds ended before their denoiser hoist (``items - n`` when that shortfall is fully explained by failed
    folds, ``n >= ok``); None when there is no shortfall or it is a MISMATCH (verdict)."""
    if st is None or items is None:
        return None
    n = st.get("n_precompute")
    if n is None or n >= items:
        return None
    covered = items if ok is None else ok
    return (items - n) if n >= covered else None
