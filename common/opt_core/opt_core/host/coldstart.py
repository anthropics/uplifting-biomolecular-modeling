"""Cold start: build a model without the random initialisation a checkpoint load overwrites; read the checkpoint without a second copy.

Contract. Three forms (torch imported inside the call; each answers a missing or too-old torch with
:class:`~opt_core.host.events.HostRefused`), plus the proof that makes them exact:

:func:`no_init` — a context manager under which the ``torch.nn.init`` fillers (and the ``reset_parameters`` they are called from)
write nothing: modules construct with their parameters ALLOCATED but not filled. It counts the skipped calls. Constants a module
computes in ``__init__`` by arithmetic (``torch.arange``, ``torch.zeros(...)`` assignments, sinusoid tables) are NOT init calls and
run as stock — which is why the form is safe for non-persistent buffers.

:func:`meta_init` — a context manager constructing on the ``meta`` device (no memory, no fills at all; torch ≥ 2.0), followed by
:func:`materialize` ``(module, state_dict)`` = ``load_state_dict(..., assign=True)`` (torch ≥ 2.1). Anything the checkpoint does not
cover is still on ``meta`` afterwards — :func:`materialize` refuses by name listing it (``reason=left_on_meta``), because a meta
tensor has no bytes to run with; such a module uses :func:`no_init` instead.

:func:`mmap_state_dict` ``(path)`` — ``torch.load(mmap=True, weights_only=True, map_location="cpu")`` (torch ≥ 2.1; the zipfile
checkpoint format torch.save writes): tensors are paged in as ``load_state_dict`` copies them, no up-front read of the whole file into host memory.

:func:`load_total` ``(module, state_dict)`` — the totality proof: ``load_state_dict(strict=False)`` then a census of missing and
unexpected keys; ANY missing key is ``HostRefused reason=missing_keys`` (a parameter or persistent buffer would have kept its
constructed — here: unfilled — value), unexpected keys are reported, not fatal (``strict_unexpected=True`` makes them fatal).
:func:`state_digest` ``(module)`` is the sha256 over every parameter and buffer (name, dtype, shape, bytes) — the dev check that a
cold-started model equals a stock-constructed-then-loaded one, once per engine and checkpoint.

The two ends of the process, framework-free: :func:`lazy_import` ``(name)`` — the module object now, its import executed at first
attribute access (``importlib.util.LazyLoader``; a CLI that exits on a usage error never pays for the framework import), and
:func:`fast_exit` ``(code, synced=paths)`` — fsync the named output files, run the registered ``atexit`` hooks (the kits' EXIT tally
lines print; CPython's ``atexit._run_exitfuncs``), flush stdio, then ``os._exit``: the interpreter's teardown of a multi-GB model and the
device context is skipped. Only after every output is written and synced — nothing after :func:`fast_exit` runs.

Evidence: :class:`ColdstartReport` (form, skipped init calls, keys loaded, missing, unexpected, load seconds) → ``active_line(tag, name)``
``[<tag>] LEVER name=F6.weights_residency_init state=on impl=host.coldstart origin=core form=no_init|meta|stock skipped_inits=.. keys=.. missing=0 unexpected=.. load_s=..``;
a refusal's line is ``exc.event.lever_line(tag, "F6.weights_residency_init")`` → ``... state=skipped reason=<reason> ...``.
Numerics: none — every weight the model runs with comes from the checkpoint, and the refusal above is what guarantees "every".
JAX engines: the load forms are torch state-dict mechanics and answer ``REFUSED reason=missing:torch`` there (parameter trees are
plain arrays — nothing to skip); :func:`lazy_import` and :func:`fast_exit` apply as they are.
"""
from __future__ import annotations

import atexit
import contextlib
import hashlib
import importlib.util
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, List, Mapping, Optional

from .events import HostEvent, HostRefused, require

LEVER = "host.coldstart"

_INIT_NAMES = ("uniform_", "normal_", "trunc_normal_", "constant_", "ones_", "zeros_", "eye_", "dirac_", "xavier_uniform_",
               "xavier_normal_", "kaiming_uniform_", "kaiming_normal_", "orthogonal_", "sparse_")


@dataclass
class ColdstartReport:
    form: str                                    # no_init · meta · stock
    skipped_inits: int = 0
    keys: int = 0                                # state-dict entries applied to the module (unexpected ones excluded)
    missing: List[str] = field(default_factory=list)
    unexpected: List[str] = field(default_factory=list)
    load_s: float = 0.0

    def evidence(self) -> dict:
        return {"lever": LEVER, "form": self.form, "skipped_inits": self.skipped_inits, "keys": self.keys,
                "missing": len(self.missing), "unexpected": len(self.unexpected), "load_s": round(self.load_s, 2)}

    def active_line(self, tag: str, name: str = "F6.weights_residency_init", origin: str = "core") -> str:
        """The kit's ONE activation line for this lever: ``[<tag>] LEVER name=<name> state=on impl=host.coldstart origin=core ...`` (``name`` = strategy id)."""
        ev = self.evidence()
        return HostEvent(LEVER, "ACTIVE", "", {k: v for k, v in ev.items() if k != "lever"}).lever_line(tag, name, origin)


class _Counter:
    def __init__(self) -> None:
        self.n = 0


@contextlib.contextmanager
def no_init() -> Iterator[_Counter]:
    """``with no_init() as c: model = Model(cfg)`` — ``torch.nn.init.*`` fillers are no-ops inside; ``c.n`` = calls skipped. A process-global
    patch: construct one model at a time under it (not from concurrent threads). A filler a module bound by ``from torch.nn.init import
    f`` at its own import is not intercepted (it runs: correct, only slower) — ``c.n`` shows what was."""
    torch = require(LEVER, "torch")
    init = torch.nn.init
    counter = _Counter()
    saved = {}

    def make_noop(name):
        def noop(tensor, *args, **kwargs):
            counter.n += 1
            return tensor
        noop.__name__ = name
        return noop

    for name in _INIT_NAMES:
        if hasattr(init, name):
            saved[name] = getattr(init, name)
            setattr(init, name, make_noop(name))
    try:
        yield counter
    finally:
        for name, fn in saved.items():
            setattr(init, name, fn)


@contextlib.contextmanager
def meta_init() -> Iterator[None]:
    """``with meta_init(): model = Model(cfg)`` — parameters and buffers on the ``meta`` device (torch ≥ 2.0); follow with :func:`materialize`."""
    torch = require(LEVER, "torch", "2.0")
    with torch.device("meta"):
        yield


def _left_on_meta(module) -> List[str]:
    out = [n for n, p in module.named_parameters() if p.device.type == "meta"]
    out += [n for n, b in module.named_buffers() if b is not None and b.device.type == "meta"]
    return out


def materialize(module, state_dict: Mapping[str, Any], *, strict_unexpected: bool = False) -> ColdstartReport:
    """``load_state_dict(state_dict, assign=True)`` on a meta-constructed module (torch ≥ 2.1), then the totality proof: a key the
    checkpoint lacks, or any tensor still on ``meta``, is :class:`HostRefused`."""
    require(LEVER, "torch", "2.1")
    t0 = time.perf_counter()
    result = module.load_state_dict(state_dict, strict=False, assign=True)
    missing, unexpected = sorted(result.missing_keys), sorted(result.unexpected_keys)
    rep = ColdstartReport("meta", 0, len(state_dict) - len(unexpected), missing, unexpected, time.perf_counter() - t0)
    _refuse_if_partial(rep, strict_unexpected)
    left = _left_on_meta(module)
    if left:
        raise HostRefused(LEVER, "left_on_meta", n=len(left), first=left[0], hint="construct under no_init() instead: these tensors are not in the checkpoint")
    return rep


def load_total(module, state_dict: Mapping[str, Any], *, form: str = "no_init", skipped_inits: int = 0,
               strict_unexpected: bool = False) -> ColdstartReport:
    """``load_state_dict(strict=False)`` + the totality proof (missing keys refuse by name). ``form``/``skipped_inits`` label the report."""
    require(LEVER, "torch")
    t0 = time.perf_counter()
    result = module.load_state_dict(state_dict, strict=False)
    missing, unexpected = sorted(result.missing_keys), sorted(result.unexpected_keys)
    rep = ColdstartReport(form, skipped_inits, len(state_dict) - len(unexpected), missing, unexpected, time.perf_counter() - t0)
    _refuse_if_partial(rep, strict_unexpected)
    return rep


def _refuse_if_partial(rep: ColdstartReport, strict_unexpected: bool) -> None:
    if rep.missing:
        raise HostRefused(LEVER, "missing_keys", form=rep.form, n=len(rep.missing), first=rep.missing[0])
    if strict_unexpected and rep.unexpected:
        raise HostRefused(LEVER, "unexpected_keys", form=rep.form, n=len(rep.unexpected), first=rep.unexpected[0])


def mmap_state_dict(path: str, *, weights_only: bool = True) -> Mapping[str, Any]:
    """The checkpoint at ``path`` memory-mapped (torch ≥ 2.1, zipfile format): ``torch.load(mmap=True, map_location="cpu")``. A file
    in the older non-zip serialisation is :class:`HostRefused` ``reason=not_zipfile`` — re-save it once with ``torch.save`` or load it the stock way, by choice."""
    torch = require(LEVER, "torch", "2.1")
    import zipfile
    if not zipfile.is_zipfile(path):
        raise HostRefused(LEVER, "not_zipfile", path=path)
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=weights_only)
    except TypeError as exc:                              # a torch build without the mmap keyword
        raise HostRefused(LEVER, "too_old:torch", found=getattr(torch, "__version__", "0"), need="2.1", error=type(exc).__name__)


def state_digest(module, *, include_buffers: bool = True) -> str:
    """sha256 (hex) over ``(name, dtype, shape, bytes)`` of every parameter (and buffer): equal digests ⇔ equal weights, for the
    once-per-checkpoint dev check that a cold-started model equals the stock-constructed one."""
    torch = require(LEVER, "torch")
    h = hashlib.sha256()
    items = list(module.named_parameters())
    if include_buffers:
        items += [(n, b) for n, b in module.named_buffers() if b is not None]
    for name, t in sorted(items, key=lambda kv: kv[0]):
        ten = t.detach()
        if ten.device.type == "meta":
            raise HostRefused(LEVER, "left_on_meta", first=name)
        ten = ten.cpu().contiguous()
        h.update(name.encode())
        h.update((str(ten.dtype) + "|" + ",".join(map(str, ten.shape))).encode())
        raw = ten.reshape(-1).view(torch.uint8) if ten.numel() else ten.new_empty((0,), dtype=torch.uint8)
        h.update(raw.numpy().tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------ process ends


def lazy_import(name: str):
    """The module ``name`` whose import runs at first attribute access (stdlib ``LazyLoader``). An unknown name is ``ImportError`` now,
    not later: the spec is resolved eagerly, only the execution is deferred. A module already imported is returned as it is."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.find_spec(name)
    if spec is None or spec.loader is None:
        raise ImportError(f"no module named {name!r}")
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def fast_exit(code: int = 0, *, synced: Iterable[str] = ()) -> None:
    """fsync each path in ``synced`` (the run's outputs), flush stdout/stderr, run the ``atexit`` hooks (EXIT tally lines), then
    ``os._exit(code)`` — no interpreter teardown. A path that cannot be synced is an ``OSError`` here, BEFORE exit: an unsynced output is
    not an output."""
    for path in synced:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    try:
        atexit._run_exitfuncs()                           # the kits' registered EXIT lines; os._exit would skip them
    finally:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:                             # noqa: BLE001 — a closed stream has nothing to flush
                pass
    os._exit(int(code))
