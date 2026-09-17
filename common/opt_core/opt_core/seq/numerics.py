"""The numerics class of a process, in the sequence kits' vocabulary: a FAÇADE over ``opt_core.precision.policy`` (the one reader and
the one setter of torch's process switches) plus the class words the kits put on their activation lines.

* :func:`readback` / :func:`readback_if_loaded` — ``opt_core.precision.policy.numerics_signature()`` projected onto the keys the kits'
  manifests record (:data:`READBACK_KEYS`): ``matmul_allow_tf32`` (``matmul_tf32``), ``cudnn_allow_tf32`` (``cudnn_tf32``),
  ``cudnn_benchmark``, ``cudnn_deterministic``, ``deterministic_algorithms``, ``deterministic_algorithms_warn_only`` (``warn_only``),
  ``float32_matmul_precision`` (``matmul``); plus two facts the signature does not carry: ``autocast_gpu_dtype`` — the dtype a bare
  ``torch.autocast("cuda")`` region would run at in this process (torch's default, whether or not a region is open) — and
  ``cublas_workspace_config`` from the environment. Read, never set; a manifest records these, never the requested values.
* :func:`classify` — ONE word for a readback, relative to torch's defaults: ``det`` (cudnn.deterministic or deterministic algorithms on)
  · ``tf32-matmul`` (matmul TF32 on, or float32 matmul precision below ``highest``) · ``prod+benchmark`` (cudnn.benchmark on) ·
  ``fp32-conv`` (cudnn TF32 off) · ``prod`` (torch's defaults: cudnn TF32 on, matmul TF32 off, timing run off, nothing deterministic),
  tested in that order. Which word a kit's composition is tested for is the kit's own data; :func:`line_fields` is the
  activation-evidence hook (``numerics=<word>`` → ``opt_core.report.kv``).
* :func:`scoped_tf32` — the TF32 switches NAMED (``None`` = untouched, never written) set for a ``with`` block through
  ``policy.apply`` and put back through ``policy.restore`` on every exit path (a kit's tier-2 TF32 arm runs its forward inside it and
  leaves no residue); :func:`assert_unchanged` is the residue rule as a check over two readbacks.
* :func:`tf32_override_env` / :func:`tf32_override_refusal` / :func:`tf32_override_gate` — the environment half (standard library only):
  the TF32 library overrides (``NVIDIA_TF32_OVERRIDE``, ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE``) move a process out of every tested
  class silently, so a kit refuses to run while one is PRESENT (any value), every mode, ``off`` included, exit ``EXIT_NOT_ACTIVE``.
  These are ``opt_core.precision.policy``'s functions of the same names where that module carries them (opt_core >= 0.3.1) and this
  module's own definitions otherwise — one behaviour, one sentence, either way.
* :func:`require_torch` / :class:`TorchUnavailable` — ``opt_core.precision.require_torch`` / ``opt_core.precision.FrameworkMissing``
  under the names the sequence kits import (the named refusal of the lazy torch import).

Importing this module imports nothing heavy (``opt_core.gates``, ``opt_core.precision``'s package file); torch is touched only inside
the torch-half functions.
"""
from __future__ import annotations

import contextlib
import os
import sys
from typing import Iterator, List, Mapping, Optional

from ..gates import Gate
from ..precision import FrameworkMissing as TorchUnavailable
from ..precision import require_torch

CLASSES = ("det", "tf32-matmul", "prod+benchmark", "fp32-conv", "prod")
READBACK_KEYS = ("matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_benchmark", "cudnn_deterministic", "deterministic_algorithms",
                 "deterministic_algorithms_warn_only", "float32_matmul_precision", "autocast_gpu_dtype", "cublas_workspace_config")
CUBLAS_ENV = "CUBLAS_WORKSPACE_CONFIG"
# signature field -> readback key (the projection)
SIGNATURE_TO_READBACK = {"matmul_tf32": "matmul_allow_tf32", "cudnn_tf32": "cudnn_allow_tf32", "cudnn_benchmark": "cudnn_benchmark",
                         "cudnn_deterministic": "cudnn_deterministic", "deterministic_algorithms": "deterministic_algorithms",
                         "warn_only": "deterministic_algorithms_warn_only", "matmul": "float32_matmul_precision"}


class NumericsResidue(AssertionError):
    """A switch reads back differently after a call than before it (:func:`assert_unchanged`)."""


# ------------------------------------------------------------------------------------------------------ environment half

def _policy():
    from ..precision import policy  # noqa: PLC0415 — sibling, light (stdlib + opt_core.gates); kept out of the import line
    return policy


try:  # opt_core.precision.policy carries the gate from opt_core 0.3.1
    from ..precision.policy import TF32_OVERRIDE_ENV, tf32_override_env, tf32_override_gate, tf32_override_refusal
except ImportError:
    TF32_OVERRIDE_ENV = ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")

    def tf32_override_env(environ: Optional[Mapping[str, str]] = None) -> List[str]:
        """The TF32 override variables PRESENT in ``environ`` (this process's by default), sorted — exact names, any value."""
        environ = os.environ if environ is None else environ
        return sorted(k for k in environ if k in TF32_OVERRIDE_ENV)

    def tf32_override_refusal(hits) -> str:
        """The refusal sentence for :func:`tf32_override_env` hits (a report carries it verbatim after ``NOT ACTIVE:``)."""
        return ("TF32 override variables set in the environment %s: refused — a row's numerics class is its mode's, never a library "
                "override's; unset them" % (list(hits),))

    def tf32_override_gate(environ: Optional[Mapping[str, str]] = None, *, name: str = "tf32_override_env") -> Gate:
        """The environment half as an ``opt_core.gates`` check: ok with no override present, else refused with :func:`tf32_override_refusal`."""
        hits = tf32_override_env(environ)
        if hits:
            return Gate(name=name, ok=False, reason=tf32_override_refusal(hits), details={"hits": hits, "checked": list(TF32_OVERRIDE_ENV)})
        return Gate(name=name, ok=True, details={"hits": [], "checked": list(TF32_OVERRIDE_ENV)})


# ------------------------------------------------------------------------------------------------------------ torch half

def _default_autocast_gpu_dtype(torch) -> Optional[str]:
    if hasattr(torch, "get_autocast_dtype"):
        return str(torch.get_autocast_dtype("cuda"))
    if hasattr(torch, "get_autocast_gpu_dtype"):
        return str(torch.get_autocast_gpu_dtype())
    return None


def _read(torch, environ: Optional[Mapping[str, str]] = None) -> dict:
    environ = os.environ if environ is None else environ
    sig = dict(_policy().numerics_signature(torch))
    out = {}
    for field, key in SIGNATURE_TO_READBACK.items():
        v = sig[field]
        out[key] = str(v) if key == "float32_matmul_precision" else bool(v)
    out["autocast_gpu_dtype"] = _default_autocast_gpu_dtype(torch)
    out["cublas_workspace_config"] = environ.get(CUBLAS_ENV)
    return {k: out[k] for k in READBACK_KEYS}


def readback(environ: Optional[Mapping[str, str]] = None, torch=None) -> dict:
    """The switches as torch reports them now (imports torch; :class:`TorchUnavailable` when it cannot) — the :data:`READBACK_KEYS`."""
    return _read(require_torch(torch), environ)


def readback_if_loaded(environ: Optional[Mapping[str, str]] = None) -> Optional[dict]:
    """:func:`readback` when torch is already imported in this process, else None — for a package that never imports torch itself."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    return _read(torch, environ)


def classify(rb: Mapping) -> str:
    """The class word of a readback (module contract: the five words, tested in order)."""
    if rb.get("cudnn_deterministic") or rb.get("deterministic_algorithms"):
        return "det"
    if rb.get("matmul_allow_tf32") or str(rb.get("float32_matmul_precision", "highest")) != "highest":
        return "tf32-matmul"
    if rb.get("cudnn_benchmark"):
        return "prod+benchmark"
    if not rb.get("cudnn_allow_tf32"):
        return "fp32-conv"
    return "prod"


def line_fields(rb: Optional[Mapping]) -> dict:
    """``{"numerics": <class word or "unread">}`` — the activation-line field, passed to ``opt_core.report.kv(**fields)``."""
    return {"numerics": classify(rb) if rb is not None else "unread"}


@contextlib.contextmanager
def scoped_tf32(matmul: Optional[bool] = None, cudnn: Optional[bool] = None, torch=None) -> Iterator[dict]:
    """The TF32 switches NAMED (``None`` = untouched, never written) set for the block — matmul through the process matmul precision
    (``"high"`` on / ``"highest"`` off, which drives ``cuda.matmul.allow_tf32``), cudnn through ``cudnn.allow_tf32`` — and restored to the
    values found on every exit path. Yields the values found for the switches it touched (``{"matmul_allow_tf32"?, "cudnn_allow_tf32"?}``;
    empty when both are None)."""
    policy = _policy()
    torch = require_torch(torch)
    found = {}
    keep = {}
    if matmul is not None or cudnn is not None:
        snap = policy.snapshot(torch)
        if matmul is not None:
            found["matmul_allow_tf32"] = bool(snap["matmul_tf32"])
            keep["matmul"] = snap["matmul"]
            keep["matmul_tf32"] = snap["matmul_tf32"]
        if cudnn is not None:
            found["cudnn_allow_tf32"] = bool(snap["cudnn_tf32"])
            keep["cudnn_tf32"] = snap["cudnn_tf32"]
        policy.apply(policy.Policy("scoped_tf32", matmul=(None if matmul is None else ("high" if matmul else "highest")),
                                   cudnn_tf32=(None if cudnn is None else bool(cudnn))), torch)
    try:
        yield dict(found)
    finally:
        if keep:
            policy.restore(keep, torch)


def assert_unchanged(before: Mapping, after: Mapping, what: str = "numerics switches") -> None:
    """The residue rule: every key of ``before`` reads back the same in ``after``; :class:`NumericsResidue` names the ones that moved."""
    moved = {k: (before[k], after.get(k)) for k in before if after.get(k) != before[k]}
    if moved:
        raise NumericsResidue("%s changed across the call {name: (before, after)} = %s" % (what, moved))


__all__ = ["CLASSES", "CUBLAS_ENV", "NumericsResidue", "READBACK_KEYS", "SIGNATURE_TO_READBACK", "TF32_OVERRIDE_ENV", "TorchUnavailable",
           "assert_unchanged", "classify", "line_fields", "readback", "readback_if_loaded", "require_torch", "scoped_tf32",
           "tf32_override_env", "tf32_override_gate", "tf32_override_refusal"]
