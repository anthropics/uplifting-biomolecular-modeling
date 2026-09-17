"""Sampling-loop mechanisms of a generative roll-out (structure diffusion or sequence decoding): how a captured or batched denoising (or
decoding) step keeps the stock random stream, the refuse-to-patch gate a loop adapter passes first, and the host-sync census of a step —
engine-free, framework-lazy. Step-invariant memoisation (compute once per roll-out, reuse per step) is :mod:`opt_core.capture.hoist`.

    source_guard    the refuse-to-patch gate over a stock function an adapter replaces (digest or required text; a changed upstream is a
                    recorded refusal and the stock function stays)
    rng             RNG discipline around captured and batched steps: generator checkpoints over warm-up, generator registration with a
                    graph, the declared capture contract and its train-mode gate, Philox offset arithmetic for padded/batched designs
    sync_census     the host-sync census of one step (development aid and regression guard; changes no numerics)
    evidence        the fail-closed census gate over a mechanism's counters (the kit composes its ONE line itself with ``opt_core.report``)

Every mechanism keeps counters and returns them as ``evidence_fields()``; the kit prints them in its ONE activation line per arm per
process, composed with :mod:`opt_core.report` (``[<tag>] <VERB> key=value ...``) — this package prints nothing. Values are opaque to this package: tensors are the
caller's, equality is the caller's (a torch kit passes ``torch.equal``), torch is imported inside the functions that need it. A kit's
adapter names the stock function, its guard and the generators; this package holds the bookkeeping and the refusals.

This module imports no sub-module at import (PEP 562: each is bound on first access).
"""

from .. import lazy_getattr

__getattr__ = lazy_getattr(__name__)          # every sub-module by name, on first access
