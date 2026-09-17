"""boltz2_opt.tp — the ``--n_gpu P`` resource axis of the memory mode (``big``), kit side. This module holds no mechanism and no words
of its own: the refusal sentences and the ACTIVE / EXIT tokens (``n_gpu=P sharding=rowpair``; ``n_gpu=1 sharding=none`` at P = 1) are
the core's (``opt_core.mem.ngpu``, the one producer — imported, never re-typed); the row-sharded pair stack that serves P > 1 is the
core's ``opt_core.mem.rowpair`` (family F8), installed by this kit's memory adapter. What is engine-specific lives here: which modes take
the axis (``TP_MODES``), which P this kit serves (``SUPPORTED_P``) and how the machine's visible devices are counted (``stack.visible_gpus``:
nvidia-smi in CUDA_VISIBLE_DEVICES order — no torch import in the launching process, no device name or memory size assumed).

    P = 1            every worker mode: the single-GPU line, byte for byte (no process group, no worker fan-out, no environment change)
    P > 1            ``--mode big`` only — refused BY NAME under every other mode (``exact``, ``fast``: sharded
                     reductions reorder sums, never an exact or fast line); fewer than P visible GPUs refused BY
                     NAME (never auto-sized); a P outside SUPPORTED_P refused BY NAME naming the set
    core absent      a core without ``opt_core.mem.ngpu`` is not the core this kit pins: ``reason=core_missing:opt_core.mem.ngpu``, the
                     NOT ACTIVE line, exit 3 — the same rule as a missing core (``_autoload``)

Contract (cli / worker / report import this): ``check(n_gpu, mode, visible)`` -> the accepted P or ``Refused(reason)``;
``fields(P)`` -> the core's positional evidence pairs for the ACTIVE / EXIT lines; ``fields_text(P)`` -> the rendered tokens.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

TP_MODES = ("big",)                 # the modes whose route takes n_gpu > 1 (the memory mode)
SUPPORTED_P: Tuple[int, ...] = (1, 2, 4, 8)   # the P values this kit's adapter (boltz2_opt.rowpair) serves; callers render big_xP arms for the P > 1 members
SCHEME = "rowpair"                    # the sharding scheme this kit's memory adapter drives at P > 1 (opt_core.mem.ngpu.SCHEMES)
WORDS_MODULE = "opt_core.mem.ngpu"    # the one producer of the n_gpu tokens and refusal sentences
COLLECTIVE_TIMEOUT_S = 1800.0           # the rank group's collective (NCCL) timeout of an n_gpu run, seconds — passed to the core launcher (run_rank_processes nccl_timeout_s); the line's own constant, read from no environment word


class Refused(RuntimeError):
    """An n_gpu request this kit refuses by name (the NOT ACTIVE path, exit 3): the core's refusal sentence, or this kit's P set."""


def words():
    """The core's n_gpu words module, or ``Refused(reason=core_missing:…)`` when the imported core does not carry it."""
    import importlib
    try:
        return importlib.import_module(WORDS_MODULE)
    except ImportError as e:
        raise Refused(f"reason=core_missing:{WORDS_MODULE} — the imported opt_core does not carry the n_gpu axis this kit pins "
                      f"({type(e).__name__}: {e}; pip install -e common/opt_core -e boltz2/opt at the pinned core)") from None


def parse(n_gpu) -> int:
    """``--n_gpu`` as an explicit positive integer by the core's rule (an absent flag, None, is 1); anything else is a usage error (ValueError)."""
    return words().check_n_gpu(1 if n_gpu is None else n_gpu)




def check(n_gpu, mode: str, visible: Optional[int] = None) -> int:
    """The accepted P for ``mode`` on this machine, or ``Refused`` with the reason BY NAME, in this order: P > 1 under a mode outside TP_MODES
    (the core's ``refuse_unless_big`` sentence — every kit mode but ``big`` is judged as a non-memory mode), P outside SUPPORTED_P (this
    kit's set), fewer than P visible GPUs (the core's ``refuse_unless_visible`` sentence; ``visible`` = the count ``stack.visible_gpus``
    found, required for P > 1). P = 1 is accepted for every mode without touching the machine."""
    w = words()
    p = parse(n_gpu)
    if p == 1:
        return 1
    try:
        w.refuse_unless_big(p, mode if mode in TP_MODES else None)   # a kit mode the core's rule does not name (exact, fast) is not the memory mode
    except w.NGpuRefused as e:
        raise Refused(str(e.reason)) from None
    if p not in SUPPORTED_P:
        raise Refused(f"refused: n_gpu={p} not served: this kit's row-sharded pair stack serves n_gpu in "
                      f"{{{','.join(str(x) for x in SUPPORTED_P)}}} (boltz2_opt.tp.SUPPORTED_P)")
    if visible is None:
        raise Refused(f"refused: n_gpu={p}: the visible GPU count was not taken (stack.visible_gpus)")
    try:
        w.refuse_unless_visible(p, int(visible))
    except w.NGpuRefused as e:
        raise Refused(str(e.reason)) from None
    return p


def family():
    """The core's row-sharding launcher module (``opt_core.mem.rowpair.launch``), or ``Refused(reason=core_missing:…)``."""
    import importlib
    try:
        return importlib.import_module("opt_core.mem.rowpair.launch")
    except ImportError as e:
        raise Refused(f"reason=core_missing:opt_core.mem.rowpair — the imported opt_core does not carry the row-sharded pair stack this kit pins ({type(e).__name__}: {e})") from None


def fields(n_gpu) -> List[Tuple[str, object]]:
    """``[("n_gpu", P), ("sharding", "rowpair"|"none")]`` — the core's positional pairs for the ACTIVE / EXIT lines (``active_pairs``)."""
    w = words()
    return w.active_pairs(w.check_n_gpu(n_gpu), SCHEME)


def fields_text(n_gpu) -> str:
    """``n_gpu=P sharding=<rowpair|none>`` rendered by the core (``active_fields``)."""
    w = words()
    return w.active_fields(w.check_n_gpu(n_gpu), SCHEME)
