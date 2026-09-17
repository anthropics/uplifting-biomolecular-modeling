"""opt_core.precision — precision policy and deterministic-recipe primitives a kit's modes declare; no engine knowledge.

A kit states, per mode, WHAT numerics it runs at (a :class:`policy.Policy`), WHICH deterministic recipe a ``--det <level>`` means
(:mod:`recipe`, rendered into the core's ``opt_core.det.Recipe`` shape), and uses two primitives whose numerics contract is written
down once here instead of once in each adapter: a deterministic segment-reduce for atomic scatters (:mod:`det_scatter`) and a cast cache for
low-precision weight copies (:mod:`cast_cache`). Every primitive RETURNS the record the kit's adapter prints as its one
activation-evidence line (``opt_core.report.kv`` fields); a refused precondition raises a named :class:`PrecisionError` subclass or is
counted as a named fallback in the primitive's census — never a silent stock path.

Submodules, by mechanism (each imports the framework it needs INSIDE its functions; importing this package imports nothing heavy):

    policy        ``Policy`` (matmul precision, cuDNN TF32, autocast dtype, cuDNN timing run), ``apply`` / ``expect`` / ``autocast_context``,
                  the process ``numerics_signature`` and ``NumericsGuard`` (the cache-key rule for captured graphs and jit caches),
                  ``input_precision_for`` (the tl.dot input-precision word a kernel must use for fp32 inputs under a given matmul precision)
    recipe        deterministic recipes in the core's shape: ``torch_recipe`` (cuBLAS workspace + the kit's own switches + a det site),
                  ``apply_torch`` (the in-process statements of a level: deterministic algorithms, cuDNN deterministic / timing run off)
                  with the CUDA-initialisation order check
    xla           the XLA form of a deterministic recipe: ``xla_recipe`` (flag words prepended to the caller's ``XLA_FLAGS``)
    det_scatter   ``segment_reduce_rows`` — a fixed-order, fp32-accumulating replacement for atomic scatter-add / scatter-mean over an
                  index (atom -> token aggregation), with the two upstream call shapes ``scatter_mean_into`` and ``scatter_reduce`` and a
                  counted, named fallback for layouts it does not serve
    cast_cache    ``CastCache`` — dtype copies of long-lived parameters keyed on equality + storage + version (refreshes on in-place
                  update, never serves a recycled id), ``precast_linear_`` (in-place low-precision Linear weights with an fp32-island
                  misuse guard), and the bit-exact output probe an exact mode runs once
    probe         ``IdentityProbe``: the bit-equality probe of an alternative formulation — one cell per real shape run on the device and
                  compared bit-exact, a sticky verdict (all | per_cell), per-call serve() with unprobed / undecided counted, LEVER line, gate, record

Classes are the kit's: a policy or primitive is ``exact`` only where the engine's equality tests
prove it bit-exact; everything here that changes a rounding point says so in its module docstring.

This file imports nothing heavy (every sub-module by name on first access, PEP 562), like ``opt_core/__init__``.
"""
from __future__ import annotations

from .. import lazy_getattr

__all__ = ("PrecisionError", "FrameworkMissing", "require_torch")


class PrecisionError(RuntimeError):
    """Base of every named refusal in this package. ``.event`` is the one-word event name the adapter logs; ``.fields`` the kv detail."""

    event = "precision_refused"

    def __init__(self, message: str = "", **fields):
        super().__init__(message)
        self.fields = dict(fields)

    def record(self) -> dict:
        """``{"event": ..., "reason": ..., **fields}`` — what the adapter prints (``opt_core.report.kv(**err.record())``)."""
        out = {"event": self.event, "reason": str(self)}
        out.update(self.fields)
        return out


class FrameworkMissing(PrecisionError):
    """The framework a primitive needs is not importable in this interpreter (the named refusal of the lazy import)."""

    event = "framework_missing"


def require_torch(torch=None):
    """The lazy torch import every torch primitive goes through: the caller's module when given, else ``import torch``;
    :class:`FrameworkMissing` (never ImportError, never a silent no-op) when it is absent."""
    if torch is not None:
        return torch
    try:
        import torch as _torch  # noqa: PLC0415 — lazy by contract (the package imports nothing heavy at top level)
    except ImportError as exc:
        raise FrameworkMissing("torch is not importable: %s" % (exc,), framework="torch")
    return _torch


__getattr__ = lazy_getattr(__name__)          # every sub-module by name, on first access
