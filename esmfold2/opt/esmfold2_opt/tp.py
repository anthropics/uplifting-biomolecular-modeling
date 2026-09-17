"""The ``--n_gpu P`` resource axis of the memory mode (``pred | check --mode big --n_gpu P``): request parsing, the refusals, the
``n_gpu=P sharding=<rowpair|none>`` token of the ACTIVE / DRY-RUN / EXIT lines, and the axis's LEVER line — every word of them from the
release tree's one producer, :mod:`opt_core.mem.ngpu` (nothing is re-typed here).

The interface: ``--n_gpu`` is explicit, default 1, never auto-detected and never shrunk
to the visible device count; ``n_gpu = 1`` is the engine's single-GPU path under any mode (token ``n_gpu=1 sharding=none``); ``n_gpu > 1``
is accepted only under ``--mode big`` — under ``exact`` / ``fast`` / ``off`` it is REFUSED BY NAME (``opt_core.mem.ngpu.REFUSE_MODE``,
exit 2). What this kit ships on the axis is :data:`N_GPU_SHIPPED`: ``{1, 2, 4, 8}`` — ``n_gpu > 1`` under ``big`` launches P rank processes
that run the row-sharded pair stack (``rowpair.py``, ``sharding=rowpair``); a P outside :data:`N_GPU_SHIPPED` is REFUSED BY NAME with
:data:`REFUSE_NOT_SHIPPED` (exit 3). There is no environment spelling of the axis (an
``ESMFOLD2_OPT_*`` name the package does not declare is refused at interpreter start, ``_autoload.py``).
"""
from __future__ import annotations

from typing import Optional, Tuple

from .report import EXIT_NOT_ACTIVE, EXIT_OK, EXIT_USAGE

MODE = "big"                                                        # the one mode under which n_gpu > 1 is accepted (== opt_core.mem.ngpu.MEMORY_MODE; test_tp locks the equality)
SHARDING = "rowpair"                                                  # the scheme token an n_gpu > 1 line of this engine carries (ngpu.SCHEMES)
N_GPU_SHIPPED: Tuple[int, ...] = (1, 2, 4, 8)                        # the P set this kit ships
REFUSE_NOT_SHIPPED = ("refused: n_gpu={P} not shipped: this kit ships n_gpu in {{1,2,4,8}} (README §Applicability: the row-sharded pair stack, "
                      "rowpair.py over opt_core.mem.rowpair, is measured at those P)")


class TpError(Exception):
    def __init__(self, msg: str, code: int = EXIT_NOT_ACTIVE):
        super().__init__(msg)
        self.code = code


def _ngpu():
    """The release tree's ONE producer of the axis' token text and refusal sentences (``opt_core.mem.ngpu``), imported at first use: with no
    core, or a core without the producer, the axis refuses BY NAME (``core_missing:opt_core.mem.ngpu``, the NOT ACTIVE exit code) — never a
    traceback, never a token spelled by the kit."""
    try:
        from opt_core.mem import ngpu
    except ImportError as e:
        raise TpError(f"core_missing:opt_core.mem.ngpu ({e}; the --n_gpu axis' producer is the release tree's opt_core: "
                      "pip install -e common/opt_core -e esmfold2/opt)", EXIT_NOT_ACTIVE) from None
    return ngpu


def __getattr__(name):                                                 # `tp.ngpu` is the producer module itself
    if name == "ngpu":
        return _ngpu()
    raise AttributeError(f"module 'esmfold2_opt.tp' has no attribute {name!r}")


def n_gpu_requested(arg) -> int:
    """``--n_gpu`` as a positive int: absent (None / blank) is 1, an explicit ``1`` is 1; anything that is not a positive integer is a
    usage error (``opt_core.mem.ngpu.check_n_gpu``'s words)."""
    if arg is None or (isinstance(arg, str) and arg.strip() == ""):
        return 1
    try:
        return _ngpu().check_n_gpu(arg.strip() if isinstance(arg, str) else arg)
    except ValueError as e:
        raise TpError(f"--n_gpu: {e}", EXIT_USAGE) from e


def not_for_route(p: int = 1) -> dict:
    """``{registry lever name: reason token}`` — the levers of a mode's single-GPU set the ``n_gpu = p`` route leaves OUT of a rank process
    (``esmfold2_opt.rowpair.NOT_FOR_ROUTE``; empty at ``p == 1``): the package subtracts them from configure()'s set (``modes``: levers_off) and
    names each on its LEVER line as ``state=not_for_route:n_gpu>1 reason=<token>`` — a declaration, so the all-or-refuse verdict counts the
    route's own set. Read without importing torch-heavy code paths beyond the adapter module itself."""
    if int(p) <= 1:
        return {}
    from .rowpair import not_for_route as _nfr
    return _nfr(int(p))


def for_route(p: int = 1) -> tuple:
    """The registry levers the ``n_gpu = p`` route ADDS to the set (``esmfold2_opt.rowpair.FOR_ROUTE``: the row-chunking levers installed by
    rowpair.install_rank after the pair stack is sharded; empty at ``p == 1``)."""
    if int(p) <= 1:
        return ()
    from .rowpair import for_route as _fr
    return tuple(_fr(int(p)))


def eager_under_sharding(p: int = 1) -> dict:
    """``{lever: note}`` — levers that stay installed and engaged WITHOUT their CUDA graph on a rank process (``rowpair.EAGER_UNDER_SHARDING``)."""
    if int(p) <= 1:
        return {}
    from .rowpair import EAGER_UNDER_SHARDING as _E
    return dict(_E)


def active_fields(p: int = 1) -> str:
    """``n_gpu=1 sharding=none`` (``n_gpu=P sharding=rowpair`` for P > 1) — the release tree's token text, rendered by opt_core."""
    return _ngpu().active_fields(int(p), SHARDING)


def refusal(mode: str, p: int) -> Tuple[Optional[str], int]:
    """(refusal sentence, exit code) for a (mode, n_gpu) request, or (None, 0): the mode rule first (n_gpu > 1 only under ``big``:
    ``REFUSE_MODE``, exit 2), then the shipped set (n_gpu > 1 under ``big``: :data:`REFUSE_NOT_SHIPPED`, exit 3)."""
    ngpu = _ngpu()
    try:
        p = ngpu.refuse_unless_big(p, mode)
    except ngpu.NGpuRefused as e:
        return str(e.reason), EXIT_USAGE
    except ValueError as e:
        return f"--n_gpu: {e}", EXIT_USAGE
    if p not in N_GPU_SHIPPED:
        return REFUSE_NOT_SHIPPED.format(P=p), EXIT_NOT_ACTIVE
    return None, EXIT_OK


def precheck(mode: str, n_gpu_arg) -> int:
    """``--n_gpu`` read and checked against the mode and the shipped set, before any pin or box gate (cli.py calls it first): returns P
    or raises :class:`TpError` carrying the refusal sentence and its exit code."""
    p = n_gpu_requested(n_gpu_arg)
    why, code = refusal(mode, p)
    if why:
        raise TpError(why, code)
    return p


def lever_line(p: int = 1) -> str:
    """The axis's LEVER line in the release tree's grammar (opt_core.report.lever_line; strategy = the row-sharded pair stack's canonical id,
    ``opt_core.mem.ngpu.TP_LEVER``): ``state=off reason=n_gpu=1:single_gpu_line`` at n_gpu = 1 (nothing sharded is installed), ``state=on
    impl=esmfold2_opt.rowpair tier=2`` at n_gpu > 1."""
    ngpu = _ngpu()
    from opt_core.report import lever_line as _lever_line
    from .report import TAG
    fields = dict(ngpu.active_pairs(int(p), SHARDING))
    if int(p) > 1:
        return _lever_line(TAG, "n_gpu", "on", impl="esmfold2_opt.rowpair", origin="kit", strategy=ngpu.TP_LEVER,
                           shipped=",".join(str(x) for x in N_GPU_SHIPPED), mode=MODE, tier="2", **fields)
    return _lever_line(TAG, "n_gpu", "off", reason="n_gpu=1:single_gpu_line", impl="esmfold2_opt.tp", origin="kit", strategy=ngpu.TP_LEVER,
                       shipped=",".join(str(x) for x in N_GPU_SHIPPED), mode=MODE, **fields)
