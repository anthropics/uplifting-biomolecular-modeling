"""ef2_lazy_structure — deferred diffusion sampling for ESMFold2-Experimental(-Fast) (exact lever, package v3 style).

`ESMFold2ExperimentalModel.forward` always runs the diffusion structure sampler (`DiffusionStructureHead.sample`: pair/single
conditioning, atom encoder, the token diffusion transformer, atom decoder — once per sampling step) after the trunk, under
`torch.no_grad()` and the fork's own `_seed_context(seed)`, and stores the coordinates in `output["sample_atom_coords"]`. Inside
`forward` the coordinates feed only the confidence head. A caller that reads the distogram (a design step: `calculate_confidence=False`,
`num_sampling_steps=1`) pays for a structure it never reads: 1 sampling step = the full conditioning + encoder/transformer/decoder pass.

This lever makes the sample LAZY when — and only when — deferring it cannot change a single bit anywhere:
  * the call carries `seed=<int>`: the fork seeds every RNG to `seed` right before the sampler and restores every RNG state right after
    it (`_seed_context`), so the sampler's random stream is pinned by the seed and the ambient stream is untouched whether the sampler runs
    inside `forward`, later, or never;
  * the confidence head does not run in this call (`model.confidence_head is None` at the moment the sampler is reached — e.g. the model
    has none, or `ef2_autograd_kernels.enable_skip_unused_confidence` removed it for a `calculate_confidence=False` call): nothing inside
    `forward` reads the coordinates.
Then `forward` returns a `LazyOutput` (a `dict`) whose `"sample_atom_coords"` entry is computed on first access by running the stock
sampler with the identical arguments under the identical `no_grad` + `_seed_context(seed)` + autocast state, once; every other entry is
the stock tensor. A call with `seed=None`, or one whose confidence head runs, takes the stock path unchanged (counted as `eager`).
Never read, never computed: the design loop's 130 no-confidence steps skip the sampler entirely.

Exactness: the distogram and every other output, and all gradients, are the stock kernels' own results (the sampler runs after them and
feeds nothing back); materialized coordinates equal the eager ones bit for bit under a seed (unit test; the design-step proof in the
track notes). Assumes what stock assumes of a seeded call: the sampler's inputs are not modified in place between the call and the access.
Memory: the deferred entry holds the sampler's arguments until it is read or the output is dropped (in the cookbook: the end of the
design step, i.e. through the losses and the backward); the fp32 pair copy stock passes (`z.float()`) is held as the bf16 pair it was
upcast from when that round trip is exact (checked on the bit pattern, no fp32-sized transient), i.e. +L²·256·2 bytes,
beside the relative-position encoding and `s_inputs` the entry also references.
Stacking: this lever wraps `model.forward` like `ef2_autograd_kernels.enable_skip_unused_confidence`; either enable order works
(deferral tests `model.confidence_head is None` at the moment the sampler is reached); `disable()` must run in reverse order of
`enable()` and refuses by name otherwise.

Usage:
    import ef2_lazy_structure as els
    els.enable(model)      # instance-level, idempotent;  els.disable(model) restores stock;  els.stats(model) -> dict(deferred, materialized, eager)
"""
from __future__ import annotations

import types

import torch
from transformers.models.esmfold2.modeling_esmfold2_common import _seed_context

KEY = "sample_atom_coords"


class _Deferred:
    """The recorded sampler call: runs once, under the grad / seed / autocast state of the original call site."""
    __slots__ = ("fn", "args", "kwargs", "seed", "autocast", "z_bf16", "result")

    def __init__(self, fn, args, kwargs, seed, autocast):
        self.fn, self.args, self.kwargs, self.seed, self.autocast, self.result = fn, args, dict(kwargs), seed, autocast, None
        self.z_bf16 = None
        z = self.kwargs.get("z_trunk")
        if torch.is_tensor(z) and z.dtype == torch.float32 and z.is_contiguous():
            # hold the pair as bf16 when fp32 -> bf16 -> fp32 is the identity on it (stock passes z.float() of the bf16 trunk output):
            # true iff the low 16 bits of every fp32 word are zero — tested on an int16 view of those bits (no fp32-sized transient;
            # one device->host read, where stock's sampler has its own at the first denoising step)
            if not bool(z.view(torch.int16)[..., 0::2].any()):
                self.z_bf16 = z.to(torch.bfloat16); self.kwargs["z_trunk"] = None

    def materialize(self):
        if self.result is None:
            kw = self.kwargs
            if self.z_bf16 is not None:
                kw = dict(kw); kw["z_trunk"] = self.z_bf16.float()
            with torch.no_grad(), _seed_context(self.seed), torch.autocast("cuda", enabled=self.autocast[0], dtype=self.autocast[1]):
                self.result = self.fn(*self.args, **kw)
            self.fn = self.args = self.kwargs = self.z_bf16 = None   # drop the held arguments
        return self.result


class LazyOutput(dict):
    """The model's output dict with one deferred entry (`sample_atom_coords`), computed on first read. Every read path of a dict
    materializes it (item access, get, pop, setdefault, values, items, copy, dict(...) / {**...} / other.update(...), pickling, repr)."""
    __slots__ = ("_lazy", "_stats")

    def __init__(self, base: dict, deferred: _Deferred, stats: dict):
        super().__init__(base); self._lazy = deferred; self._stats = stats

    def _materialize(self):
        d = self._lazy
        if d is not None:
            out = d.materialize()
            dict.__setitem__(self, KEY, out[KEY])
            self._lazy = None; self._stats["materialized"] += 1

    def lazy_state(self) -> str:
        return "deferred" if self._lazy is not None else "materialized"

    def __getitem__(self, k):
        if k == KEY: self._materialize()
        return dict.__getitem__(self, k)

    def get(self, k, default=None):
        if k == KEY: self._materialize()
        return dict.get(self, k, default)

    def pop(self, k, *d):
        if k == KEY: self._materialize()
        return dict.pop(self, k, *d)

    def setdefault(self, k, default=None):
        if k == KEY: self._materialize()
        return dict.setdefault(self, k, default)

    def __iter__(self):                 # overridden so dict.update / dict(...) / {**...} take the mapping protocol (keys + __getitem__), never the raw storage
        return dict.__iter__(self)

    def keys(self):
        return dict.keys(self)

    def values(self):
        self._materialize(); return dict.values(self)

    def items(self):
        self._materialize(); return dict.items(self)

    def copy(self):
        self._materialize(); return dict(self)

    def __reduce__(self):
        self._materialize(); return (dict, (dict(self),))

    def __repr__(self):
        self._materialize(); return dict.__repr__(self)

    def __eq__(self, other):
        self._materialize(); return dict.__eq__(self, other)

    __hash__ = None


class _Placeholder:
    """Stands in for the sampler's output inside `forward` for one deferred call; never leaves the wrapper."""


def _forward(model, orig_forward, *args, **kw):
    seed = kw.get("seed")
    if seed is None:                                        # no seed: the sampler draws from the ambient stream at the call site — stock path, eager
        model._els_stats["eager"] += 1
        return orig_forward(*args, **kw)
    head = model.structure_head
    stock_sample = head.sample                              # the class's method as bound now (any other instance patch included)
    slot = {}

    def sample(*a, **k):
        if model.confidence_head is not None:               # the confidence head will read the coordinates inside forward: eager, stock
            model._els_stats["eager"] += 1
            return stock_sample(*a, **k)
        slot["d"] = _Deferred(stock_sample, a, k, seed, (torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda")))
        return {KEY: _Placeholder()}
    had = "sample" in vars(head)                            # an instance-level binding already present (another patch) is restored, not dropped
    head.sample = sample                                    # instance attribute for the duration of this call only
    try:
        out = orig_forward(*args, **kw)
    finally:
        if had:
            head.sample = stock_sample
        else:
            del head.sample                                 # back to the class's method
    d = slot.get("d")
    if d is None:
        return out
    if not isinstance(out, dict) or not isinstance(dict.get(out, KEY), _Placeholder):
        raise RuntimeError("ef2_lazy_structure: forward did not return the sampler's output under 'sample_atom_coords' — the fork's forward changed; disable this lever")
    model._els_stats["deferred"] += 1
    return LazyOutput(out, d, model._els_stats)


def enable(model) -> None:
    """Patch `model.forward` (instance-level, reversible, idempotent)."""
    if hasattr(model, "_els_orig_forward"):
        return
    if not hasattr(model, "structure_head") or not hasattr(model.structure_head, "sample"):
        raise RuntimeError("ef2_lazy_structure: model has no structure_head.sample")
    orig = model.forward
    model._els_orig_forward = orig
    model._els_stats = dict(deferred=0, materialized=0, eager=0)
    model._els_wrapper = types.MethodType(lambda self, *a, **kw: _forward(self, orig, *a, **kw), model)
    model.forward = model._els_wrapper


def disable(model) -> None:
    """Restore the forward this lever wrapped. Levers that wrap `model.forward` stack; unwrap them in reverse order of enable() — a
    `disable()` while another lever's wrapper sits on top is refused by name rather than silently dropping that wrapper."""
    if hasattr(model, "_els_orig_forward"):
        if vars(model).get("forward") is not model._els_wrapper:
            raise RuntimeError("ef2_lazy_structure.disable: model.forward was re-wrapped by another lever after enable(); disable that one first (LIFO)")
        model.forward = model._els_orig_forward
        del model._els_orig_forward, model._els_wrapper


def stats(model) -> dict:
    return dict(getattr(model, "_els_stats", {}))
