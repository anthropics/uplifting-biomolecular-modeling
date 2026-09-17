"""opt_core.capture — graph / JIT capture and compile caches: the strategies of the capture family, one implementation each.

    graphs      CUDA-graph capture-or-replay of one callable site keyed by argument signature: warm-up, capture, bit-exact checking,
                RNG discipline, private or generation pools, memory budget, failed-capture recovery, the LEVER evidence line (torch, lazily)
    pool        the capture POLICIES (fixed K-th sighting, break-even, pinned table, job plan — pure Python) and GraphPool: several sites
                under one memory budget, one item/job boundary, optional shape bucketing, one census
    hoist       ConstMemo: step-invariant values computed once per item inside a step loop, bypassed inside captures, the LEVER evidence line;
                RowDedup: a row-independent statement on identical rows evaluated on one row and expanded back (F7.row_dedup)
    compile     torch.compile plumbing of a compile lever (F3.torch_compile): nodynamo boundary marker, the one installer over (owner, attr)
                targets with a recompile-budget floor, dynamo's census on a LEVER line, the fail-closed gate (torch, lazily)
    xla_cache   serialized XLA executables per shape bucket + the environment rule of JAX's persistent compilation cache (jax, lazily)

The compile-cache KEY of the running stack and the cache-directory rule are :mod:`opt_core.jit_cache`. This module imports no sub-module at import (PEP 562: each is bound on first access).
"""
from __future__ import annotations

from .. import lazy_getattr

__getattr__ = lazy_getattr(__name__)          # every sub-module by name, on first access
