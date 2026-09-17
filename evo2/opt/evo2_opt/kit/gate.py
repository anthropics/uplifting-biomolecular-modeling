"""The call-surface gate: per model-forward call the kit path or, by name, the stock path (generation calls with inference params, padded forwards); the shape manager releases the shape-bound stores when the (batch, length) changes so one shape's buffers live at a time."""
from __future__ import annotations

import contextlib
import functools
import sys
import threading

PREFIX = "[evo2-kit]"


def decide_route(B: int, L: int, inference_params_dict, padding_mask) -> tuple[str, str | None]:
    """('kit', None) or ('stock', reason): every (batch, length) takes the kit path; the stock path serves, by name, the calls the scoring
    levers are not built for — cached generation (prefill / one-token decode steps carry inference params) and padded forwards."""
    if inference_params_dict is not None:
        return "stock", "inference_params|generate: cached-state prefill / decode steps"
    if padding_mask is not None:
        return "stock", "padding_mask|padded forward"
    return "kit", None


class Gate:
    """The process-wide mode the gated attributes read: 'kit', or 'stock' inside a passthrough call. Counts land in ``counter``."""

    def __init__(self, counter):
        self.counter = counter
        self._local = threading.local()
        self.announced: set = set()
        self.log = sys.stderr

    @property
    def mode(self) -> str:
        return getattr(self._local, "mode", "kit")

    @contextlib.contextmanager
    def passthrough(self, reason: str):
        tag = reason.split("|", 1)[0]
        self.counter["gate_passthrough"] += 1
        self.counter[f"passthrough:{tag}"] += 1
        if tag not in self.announced:                          # once per reason: these calls run the stock path (the stock's bytes), named
            self.announced.add(tag)
            print(f"{PREFIX} stock path for {reason.replace('|', ' calls (', 1)}) — named once; counted on the EXIT line", file=self.log, flush=True)
        prev = self.mode
        self._local.mode = "stock"
        try:
            yield
        finally:
            self._local.mode = prev

    @contextlib.contextmanager
    def kit(self):
        self.counter["gate_kit"] += 1
        prev = self.mode
        self._local.mode = "kit"
        try:
            yield
        finally:
            self._local.mode = prev


def gated(gate: Gate, kit_fn, stock_fn, name: str):
    """The wrapper installed on a patched class/module attribute: the kit function in kit mode, the stock function in passthrough."""
    @functools.wraps(kit_fn)
    def wrapper(*args, **kwargs):
        if gate.mode == "stock":
            return stock_fn(*args, **kwargs)
        return kit_fn(*args, **kwargs)
    wrapper.__kit_fn__ = kit_fn
    wrapper.__stock_fn__ = stock_fn
    wrapper.__gated_name__ = name
    wrapper.__wrapped_stock__ = stock_fn
    return wrapper


class ShapeManager:
    """The shape-bound stores of the composed kit, released together on a change of the gated shape (B, L).
    ``stores``: {name: release()} callables (dict.clear, cuFFT plan destruction, ...); ``model_stores``: the model-bound state (L-keyed
    filter caches, FP8 weight workspaces) released before a passthrough only; ``devices``: the CUDA devices whose caching allocator is
    emptied after a release so the freed blocks return before the new shape allocates."""

    def __init__(self, stores: dict, counter, devices=(), empty_cache=None, model_stores: dict | None = None):
        self.stores = dict(stores)
        self.model_stores = dict(model_stores or {})
        self.counter = counter
        self.devices = list(devices)
        self.empty_cache = empty_cache
        self.shape: tuple[int, int] | None = None
        self.live = False
        self.releases = 0
        self.full_releases = 0

    def ensure(self, B: int, L: int) -> bool:
        """True iff the stores were released (the shape changed from a previous one)."""
        new = (int(B), int(L))
        if self.shape == new:
            return False
        released = self.shape is not None
        if released:
            self.release()
        self.shape = new
        self.live = True
        return released

    def release(self, full: bool = False) -> None:
        for name, rel in self.stores.items():
            rel()
        self.releases += 1
        self.counter["shape_release"] += 1
        if full:
            for name, rel in self.model_stores.items():
                rel()
            self.full_releases += 1
            self.counter["full_release"] += 1
            self.live = False
        if self.devices:
            import torch
            empty = self.empty_cache or torch.cuda.empty_cache
            for d in self.devices:
                with (torch.cuda.device(d) if torch.cuda.is_available() else contextlib.nullcontext()):
                    empty()

    def reset(self) -> None:
        self.shape = None


def route_forward(orig_forward, gate: Gate, manager: ShapeManager, decide):
    """The model forward's wrapper (vortex StripedHyena.forward(self, x, inference_params_dict=None, padding_mask=None)): ``decide`` picks
    the route for this call; in kit mode the shape-bound stores follow the (B, L); a passthrough releases every store first (the stock path
    then has the device's memory; the next kit forward re-fills them exactly as the first forward of a process does)."""
    @functools.wraps(orig_forward)
    def forward(self, x, inference_params_dict=None, padding_mask=None, *args, **kwargs):
        B, L = int(x.shape[0]), int(x.shape[1])
        route, reason = decide(self, B, L, inference_params_dict, padding_mask)
        if route == "kit":
            manager.ensure(B, L)
            with gate.kit():
                return orig_forward(self, x, inference_params_dict, padding_mask, *args, **kwargs)
        if manager.shape is not None or manager.live:
            manager.release(full=True); manager.reset()
        with gate.passthrough(reason):
            return orig_forward(self, x, inference_params_dict, padding_mask, *args, **kwargs)
    forward.__wrapped_stock__ = orig_forward
    forward.__gated_name__ = "StripedHyena.forward"
    return forward


def forward_gate(orig_forward, gate: Gate, manager: ShapeManager):
    """The route decision without the hook rule (hooks.forward_gate adds it first)."""
    return route_forward(orig_forward, gate, manager, lambda model, B, L, ip, pm: decide_route(B, L, ip, pm))
