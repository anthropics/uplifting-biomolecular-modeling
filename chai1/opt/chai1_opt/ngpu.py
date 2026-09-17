"""The ``--n_gpu P`` resource axis on Chai-1: P ∈ {1}. Every mode folds on one GPU per process; the shared core's ``opt_core.mem.ngpu`` is
the one producer of the axis's words (the ``n_gpu=P sharding=<scheme>`` tokens of the ACTIVE / EXIT lines and the refusal sentences) —
this module only applies its rules in the kit's order and adds Chai-1's own applicability sentence:

* ``--n_gpu`` absent or ``1`` under any mode: the engine's single-GPU path; ACTIVE / EXIT carry ``n_gpu=1 sharding=none``.
* ``--n_gpu P>1`` under ``exact`` / ``fast``: refused by name with the core's mode sentence (sharded reductions are never bitwise).
* ``--n_gpu P>1`` under ``big``: refused by name — not applicable on chai1 (:data:`REFUSE_NOT_APPLICABLE`, the one sentence the CLI,
  the README and CHANGES quote): the trunk's pair stack runs as structured eager modules (``chai1_eager/trunk.py``) and is shardable, but the
  pair representation is initialised, denoised and scored inside sealed TorchScript exports (``token_embedder`` / ``diffusion_module`` /
  ``confidence_head``, flat per-crop transpilations of ``chai1_eager/ts2eager.py`` with no shardable pair structure) that run whole per rank,
  and chai_lab 0.6.1 refuses inputs above 2048 tokens (``stock/src/chai_lab/chai1.py:243-245``, ``AVAILABLE_MODEL_SIZES`` max 2048), a crop
  ``big`` folds on one 80 GB card — row sharding buys no reach. Never shrunk to one GPU silently.
* a non-integer or ``P < 1`` is a usage error (``opt_core.mem.ngpu.check_n_gpu``).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

SUPPORTED: Tuple[int, ...] = (1,)                       # the P set this kit ships; multi-GPU forms (big_xP) come from it
SCHEME = "rowpair"                                     # the scheme name the tokens would carry for P > 1 (opt_core.mem.ngpu.SCHEMES); P = 1 prints sharding=none
CORE_MODULE = "opt_core.mem.ngpu"
REFUSE_NOT_APPLICABLE = ("refused: n_gpu>1 not applicable on chai1: the trunk's pair stack is shardable but pair init, the diffusion module and the "
                         "confidence head are sealed TorchScript exports that run whole per rank, and the model input limit of 2048 tokens fits one "
                         "card under --mode big (row sharding buys no reach)")


def core():
    """``opt_core.mem.ngpu`` — a hard import: the pinned core carries it (ImportError = the caller's ``core_missing`` refusal)."""
    from ._core import ensure_importable
    ensure_importable()
    import importlib
    return importlib.import_module(CORE_MODULE)


def resolve(n_gpu, mode: Optional[str]) -> int:
    """The accepted P for ``mode``: 1. ``None`` (flag absent) is 1. Raises ValueError (usage) for a malformed value and the core's
    ``NGpuRefused`` — ``str(exc.reason)`` is the sentence the NOT ACTIVE line prints — for P > 1: the core's mode sentence under
    ``exact`` / ``fast``, :data:`REFUSE_NOT_APPLICABLE` under ``big``."""
    M = core()
    p = M.check_n_gpu(1 if n_gpu is None else n_gpu)
    M.refuse_unless_big(p, mode)                      # exact / fast: 'refused: n_gpu>1 requires --mode big (…)'
    if p not in SUPPORTED:
        raise M.NGpuRefused(REFUSE_NOT_APPLICABLE)
    return p


def refusal(n_gpu, mode: Optional[str]) -> Optional[str]:
    """The refusal sentence for (n_gpu, mode), or None when the value is accepted; ValueError for a malformed value."""
    M = core()
    try:
        resolve(n_gpu, mode)
    except M.NGpuRefused as e:
        return str(e.reason)
    return None


def active_pairs(p: int) -> List[Tuple[str, object]]:
    """``[("n_gpu", P), ("sharding", "none"|<scheme>)]`` — the core's pairs, for the ACTIVE / EXIT lines."""
    return core().active_pairs(int(p), SCHEME)


def active_fields(p: int) -> str:
    """``n_gpu=1 sharding=none`` — the core's rendering (``opt_core.mem.ngpu.active_fields``)."""
    return core().active_fields(int(p), SCHEME)
