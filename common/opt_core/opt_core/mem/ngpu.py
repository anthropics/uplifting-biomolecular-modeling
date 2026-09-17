"""opt_core.mem.ngpu — the ``--n_gpu P`` resource axis of the memory mode, framework-free (standard library only): the ONE producer of
the ``n_gpu=P sharding=<scheme>`` token text the kit's ACTIVE / EXIT line carries and of the refusal sentences, for the
row-sharded pair stack (scheme ``rowpair``) and any other
scheme the kit drives. The kit and the pair-stack package import these names; nobody re-types the words.

Rules (the interface the kit ships):
  * ``--n_gpu`` is explicit, default 1, a positive integer (:func:`check_n_gpu`); never auto-detected, never shrunk to what is visible.
  * ``n_gpu == 1`` is the engine's single-GPU path under any mode: token ``n_gpu=1 sharding=none``.
  * ``n_gpu > 1`` is accepted only under ``--mode big`` (sharded reductions reorder sums, so the run is fast-class, never bit-exact):
    under ``exact`` / ``fast`` it is REFUSED BY NAME with :data:`REFUSE_MODE` and a non-zero exit.
  * fewer visible devices than ``n_gpu`` is REFUSED BY NAME with :data:`REFUSE_VISIBLE` (``refused: n_gpu=P visible=K``).
  * an accepted ``n_gpu = P > 1`` run's ACTIVE / EXIT lines carry ``n_gpu=P sharding=<scheme>`` (:func:`active_fields`).

API:
    check_n_gpu(n_gpu) -> int                         positive int or ValueError (a usage error)
    sharding_value(P, scheme) -> str                  ``scheme`` when P > 1 else ``none``; ``scheme`` must be in :data:`SCHEMES`
    active_pairs(P, scheme) -> [(k, v), (k, v)]       ``[("n_gpu", P), ("sharding", <scheme|none>)]`` — positional pairs for ``opt_core.report``
    active_fields(P, scheme) -> str                   ``n_gpu=P sharding=<scheme|none>`` (rendered by ``opt_core.report.kv``)
    refuse_unless_big(n_gpu, mode) -> int           the mode rule; raises :class:`NGpuRefused` (``reason`` = :data:`REFUSE_MODE`)
    refuse_unless_visible(n_gpu, visible) -> int      the resource rule for P > 1 (``visible`` = the device count the framework reports, an int);
                                                      raises :class:`NGpuRefused` (``reason`` = ``refused: n_gpu=P visible=K``)
    visible_refusal(P, K) -> str                      the rendered sentence
    NGpuRefused(reason, lever="n_gpu")                a :class:`opt_core.mem.MemLeverRefused`; ``str(exc.reason)`` is the exact sentence
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from . import MemLeverRefused

__all__ = ["SCHEMES", "SHARDING_NONE", "MEMORY_MODE", "REFUSE_MODE", "REFUSE_VISIBLE", "NGpuRefused", "TP_LEVER", "CERTIFIED_SM",
           "TP_ARCH_NOTE", "check_n_gpu", "sharding_value",
           "active_pairs", "active_fields", "refuse_unless_big", "refuse_unless_visible", "visible_refusal"]

SCHEMES = ("rowpair", "foldcp2d")            # rowpair: the pair representation row-sharded over P devices; foldcp2d: 2-D fold context parallelism
SHARDING_NONE = "none"
MEMORY_MODE = "big"                        # the one mode under which n_gpu > 1 is accepted
REFUSE_MODE = "refused: n_gpu>1 requires --mode big (sharded reductions are not bitwise)"
REFUSE_VISIBLE = "refused: n_gpu={P} visible={K}"


TP_LEVER = "F7.tensor_parallel"          # the row-sharded pair stack's strategy id (the LEVER line's strategy= word)
CERTIFIED_SM = ()                        # sm classes the row-sharded line supports (opt_core.arch); a class is added only once it has been tested
TP_ARCH_NOTE = "row-sharded pair stack: the framework's collectives + the engine's own statements; no arch-specific kernels, no sm floor"


def _declare_arch() -> None:
    """The ONE arch declaration of ``F7.tensor_parallel`` (:mod:`opt_core.arch`), shared by every stack that imports this
    module: no arch-specific kernel -> no floor among the registry's classes, no exclusion."""
    from .. import arch
    arch.declare(TP_LEVER, certified=CERTIFIED_SM, note=TP_ARCH_NOTE)


_declare_arch()

class NGpuRefused(MemLeverRefused):
    """An ``--n_gpu`` rule refused the run (``reason`` = the exact sentence). The kit prints it and exits non-zero; nothing converts it into
    a single-GPU run."""

    def __init__(self, reason: str, lever: str = "n_gpu"):
        super().__init__(lever, reason)


def check_n_gpu(n_gpu) -> int:
    """``n_gpu`` as a positive int; anything else is a usage error (ValueError)."""
    try:
        p = int(n_gpu)
    except (TypeError, ValueError):
        raise ValueError(f"n_gpu={n_gpu!r}: a positive integer is required") from None
    if p < 1 or (isinstance(n_gpu, float) and n_gpu != p):
        raise ValueError(f"n_gpu={n_gpu!r}: a positive integer is required")
    return p


def sharding_value(P, scheme: str = "rowpair") -> str:
    if scheme not in SCHEMES:
        raise ValueError(f"sharding scheme {scheme!r} not in {SCHEMES}")
    return scheme if check_n_gpu(P) > 1 else SHARDING_NONE


def active_pairs(P, scheme: str = "rowpair") -> List[Tuple[str, object]]:
    """``[("n_gpu", P), ("sharding", <scheme|none>)]`` in THIS order."""
    p = check_n_gpu(P)
    return [("n_gpu", p), ("sharding", sharding_value(p, scheme))]


def active_fields(P, scheme: str = "rowpair") -> str:
    """``n_gpu=P sharding=<scheme>`` for P > 1, ``n_gpu=1 sharding=none`` for P == 1."""
    from ..report import kv
    return kv(*active_pairs(P, scheme))


def refuse_unless_big(n_gpu, mode: Optional[str]) -> int:
    """``n_gpu == 1`` passes under any mode; ``n_gpu > 1`` only under ``big`` — else :class:`NGpuRefused` (:data:`REFUSE_MODE`)."""
    p = check_n_gpu(n_gpu)
    if p > 1 and (mode or "").strip().lower() != MEMORY_MODE:
        raise NGpuRefused(REFUSE_MODE)
    return p


def visible_refusal(P: int, K: int) -> str:
    return REFUSE_VISIBLE.format(P=int(P), K=int(K))


def refuse_unless_visible(n_gpu, visible: int) -> int:
    """P > 1 needs ``visible >= P`` devices — fewer is :class:`NGpuRefused` (``refused: n_gpu=P visible=K``). P == 1 is not checked here
    (the engine's single-GPU path owns its device check)."""
    p = check_n_gpu(n_gpu)
    if p == 1:
        return p
    k = int(visible)
    if k < p:
        raise NGpuRefused(visible_refusal(p, k))
    return p
