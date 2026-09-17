"""The numerics policy a mode declares, the process numerics signature, and the input-precision word for fp32 tensor-core kernels.

Contract. A :class:`Policy` is a frozen record of the four process-wide numerics switches torch exposes for inference — the fp32 matmul
precision (``"highest"`` = IEEE fp32, no TF32 · ``"high"`` = TF32 tensor cores · ``"medium"`` = bf16 tensor cores for fp32 matmuls),
cuDNN TF32, the autocast dtype the kit's forward runs under (``None`` = no autocast region of the kit's own), cuDNN timing run — plus a
note. ``None`` in a field means "the policy does not touch it". A kit keeps one Policy per mode in its mode table: ``off`` and ``exact``
name the stock engine's own policy (read from the stock source and cited in the kit's STOCK.md; :func:`expect` proves it is in force and
sets nothing), ``fast`` may name another (a TF32 or autocast variant) and :func:`apply` sets it. Changing any field relative to the
stock policy changes rounding: such a mode is tolerance-class (``fast``) unless the engine's equality tests prove otherwise.

:func:`numerics_signature` is the tuple every captured CUDA graph, compiled artefact or cached tensor is only valid under (matmul
precision, TF32 flags, cuDNN deterministic / timing run, deterministic algorithms, autocast state, plus the kit's own kernel-backend
words); :class:`NumericsGuard` compares it against the value at capture time and hands the adapter a named event when it changed — the
adapter drops its graphs and caches and logs the event; nothing is dropped or kept silently. :func:`snapshot` / :func:`restore` are the
observer form (record the switches before code the kit does not own runs, put them back after, report what moved). The framework is a
READER (:data:`READERS`: ``torch`` today; another framework's reader registers beside it) — signature, guard, snapshot take ``reader=``.

Multi-stage in-process pipelines whose stock form is one subprocess per stage: take :func:`snapshot` ONCE at import, BEFORE any
``torch.no_grad()`` / ``torch.inference_mode()`` / ``torch.autocast()`` region is entered and before any stage runs; call :func:`restore`
OUTSIDE such regions — between stages, never inside a stage's context managers — before each stage and after the last; record
``dict(torch_reader())`` at the END of each stage into that stage's activation line. A Policy is per (mode, stage) on such an engine, held by
the kit as ``{stage: Policy}``; :func:`expect` runs after the stage's own setter, never before. :func:`restore` touches the numerics
signature only; process-global non-numerics state (grad mode, :data:`PROCESS_STATE_FIELDS`) is restored only when requested by name
(``process_state=...``).

:func:`input_precision_for` is the one table a Triton / CUDA kernel taking fp32 inputs reads to choose its ``tl.dot`` input precision
so that a shared kernel never runs TF32 products on an engine whose stock policy is ``"highest"``.

:func:`tf32_override_env` / :func:`tf32_override_refusal` / :func:`tf32_override_gate` are the environment half of the policy (standard
library only, no torch): the TF32 library overrides — ``NVIDIA_TF32_OVERRIDE`` steers cuBLAS/cuDNN underneath any framework,
``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`` torch's cuBLAS flag — move a process out of every declared policy silently, so an exact-class arm
(``exact`` and ``off``: bitwise equality with stock is the claim) refuses to run while one is PRESENT in the environment (any value:
``=0`` forces TF32 off as much as ``=1`` forces it on) and a tolerance-class arm (``exact=False``) runs and names it on its line
(``tf32_override=<VAR>=<value>``); the refusal sentence is what the kit prints after ``NOT ACTIVE:`` and the gate form plugs into
``opt_core.gates.run_gates``. :func:`expect` follows the same tier rule for live switches that differ from the declared policy.

Activation evidence: :func:`apply` / :func:`expect` return ``{"policy", "matmul", "cudnn_tf32", "autocast", "cudnn_benchmark",
"changed"}`` — the adapter prints it once per process with its own prefix (``opt_core.report.kv``).
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

from ..gates import Gate
from . import PrecisionError, require_torch

MATMUL_PRECISIONS = ("highest", "high", "medium")          # torch.set_float32_matmul_precision vocabulary
AUTOCAST_DTYPES = (None, "bf16", "fp16")
_DTYPE_ATTR = {"bf16": "bfloat16", "fp16": "float16"}
# tl.dot(input_precision=...) words for fp32 operands, by the process matmul precision they must not exceed
_INPUT_PRECISION = {"highest": "ieee", "high": "tf32", "medium": "tf32"}


class PolicyMismatch(PrecisionError):
    """The live process flags differ from the policy a mode declares (raised by :func:`expect`; an exact mode's fail-closed gate)."""

    event = "policy_mismatch"


@dataclass(frozen=True)
class Policy:
    name: str
    matmul: Optional[str] = None              # "highest" | "high" | "medium" | None (untouched)
    cudnn_tf32: Optional[bool] = None
    autocast: Optional[str] = None            # None | "bf16" | "fp16" — the dtype of the kit's own autocast region, if it opens one
    cudnn_benchmark: Optional[bool] = None
    note: str = ""

    def __post_init__(self):
        if self.matmul is not None and self.matmul not in MATMUL_PRECISIONS:
            raise ValueError("Policy %r: matmul must be one of %s or None, not %r" % (self.name, MATMUL_PRECISIONS, self.matmul))
        if self.autocast not in AUTOCAST_DTYPES:
            raise ValueError("Policy %r: autocast must be one of %s, not %r" % (self.name, AUTOCAST_DTYPES, self.autocast))

    def describe(self) -> dict:
        """The kv fields of the policy itself (no torch): what a DRY-RUN line prints."""
        return {"policy": self.name, "matmul": _word(self.matmul), "cudnn_tf32": _word(self.cudnn_tf32),
                "autocast": _word(self.autocast), "cudnn_benchmark": _word(self.cudnn_benchmark)}


# Presets by numerics class (no engine names: a kit cites its stock source for the one it declares under off/exact).
FP32_STRICT = Policy("fp32_strict", matmul="highest", cudnn_tf32=False, note="IEEE fp32 matmuls, no TF32 anywhere")
FP32_TF32 = Policy("fp32_tf32", matmul="high", cudnn_tf32=True, note="TF32 tensor cores for fp32 matmuls and cuDNN")
BF16_AUTOCAST = Policy("bf16_autocast", autocast="bf16", note="bf16 autocast region over fp32 masters; matmul precision untouched")
BF16_AUTOCAST_STRICT = Policy("bf16_autocast_strict", matmul="highest", cudnn_tf32=False, autocast="bf16",
                              note="bf16 autocast region; fp32 islands stay IEEE")
UNTOUCHED = Policy("untouched", note="the interpreter's defaults, whatever they are (recorded, not set)")


def _word(v) -> str:
    if v is None:
        return "untouched"
    if isinstance(v, bool):
        return "on" if v else "off"
    return str(v)


def _fp32_state(torch) -> Tuple[str, bool, bool]:
    """(matmul precision word, matmul TF32 on, cuDNN TF32 on) from ``get_float32_matmul_precision`` / the ``allow_tf32`` switches, or —
    when torch refuses those getters because the per-backend ``fp32_precision`` setting was used in this process (torch >= 2.9 raises
    ``RuntimeError`` then; the setters still set) — from ``fp32_precision`` itself: ``"tf32"`` = on, matmul word ``"high"`` if on else
    ``"highest"`` (the mapping torch documents for the two APIs)."""
    try:
        return (str(torch.get_float32_matmul_precision()), bool(torch.backends.cuda.matmul.allow_tf32), bool(torch.backends.cudnn.allow_tf32))
    except RuntimeError:
        mm = str(getattr(torch.backends.cuda.matmul, "fp32_precision", "ieee")) == "tf32"
        cu = str(getattr(getattr(torch.backends.cudnn, "conv", None), "fp32_precision", "ieee")) == "tf32"
        return ("high" if mm else "highest", mm, cu)


def live(torch=None) -> dict:
    """The live values of the four switches (imports torch lazily)."""
    torch = require_torch(torch)
    matmul, matmul_tf32, cudnn_tf32 = _fp32_state(torch)
    return {"matmul": matmul, "cudnn_tf32": cudnn_tf32, "matmul_tf32": matmul_tf32, "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)}


def apply(policy: Policy, torch=None) -> dict:
    """Set the switches the policy names (module contract) and return the activation record:
    ``policy.describe()`` + ``changed=[field,...]`` (the fields whose live value this call changed) + ``live=<matmul>/<tf32 words>``.
    Autocast is not a process switch: the record names it, :func:`autocast_context` opens it."""
    torch = require_torch(torch)
    before = live(torch)
    if policy.matmul is not None:
        torch.set_float32_matmul_precision(policy.matmul)      # also drives torch.backends.cuda.matmul.allow_tf32 (high/medium -> True)
    if policy.cudnn_tf32 is not None:
        torch.backends.cudnn.allow_tf32 = bool(policy.cudnn_tf32)
    if policy.cudnn_benchmark is not None:
        torch.backends.cudnn.benchmark = bool(policy.cudnn_benchmark)
    after = live(torch)
    rec = policy.describe()
    rec["changed"] = [k for k in sorted(after) if after[k] != before[k]]
    rec["live"] = "%s/matmul_tf32=%s/cudnn_tf32=%s" % (after["matmul"], _word(after["matmul_tf32"]), _word(after["cudnn_tf32"]))
    return rec


def mismatches(policy: Policy, torch=None) -> dict:
    """``{field: (declared, live)}`` for every switch the policy names whose live value differs; empty when the policy is in force."""
    now = live(torch)
    out = {}
    if policy.matmul is not None and now["matmul"] != policy.matmul:
        out["matmul"] = (policy.matmul, now["matmul"])
    if policy.cudnn_tf32 is not None and now["cudnn_tf32"] != bool(policy.cudnn_tf32):
        out["cudnn_tf32"] = (bool(policy.cudnn_tf32), now["cudnn_tf32"])
    if policy.cudnn_benchmark is not None and now["cudnn_benchmark"] != bool(policy.cudnn_benchmark):
        out["cudnn_benchmark"] = (bool(policy.cudnn_benchmark), now["cudnn_benchmark"])
    return out


def expect(policy: Policy, torch=None, *, exact: bool = True) -> dict:
    """Prove the policy is in force WITHOUT setting anything (the exact / off modes' form: the stock engine sets its own switches; the
    kit only attests them). Returns the activation record with ``changed=[]`` and ``words``. A differing field under ``exact=True`` raises
    :class:`PolicyMismatch` naming every one — the adapter refuses the mode, it never 'fixes' an exact arm's numerics (another matmul
    precision is another rounding: bitwise equality is at stake); under ``exact=False`` (a tolerance-class arm attesting, not setting) the
    differences are the word ``numerics=drift(<field>:<live>!=<declared>,…)`` in ``rec["words"]`` and ``rec["drift"]``, and the arm proceeds."""
    from ..report import word as _named  # noqa: PLC0415
    torch = require_torch(torch)
    diff = mismatches(policy, torch)
    if diff and exact:
        detail = ",".join("%s:declared=%s,live=%s" % (k, _word(v[0]), _word(v[1])) for k, v in sorted(diff.items()))
        raise PolicyMismatch("live numerics differ from policy %r: %s" % (policy.name, detail), policy=policy.name, fields=detail)
    rec = policy.describe()
    rec["changed"] = []
    rec["drift"] = {k: {"declared": _word(v[0]), "live": _word(v[1])} for k, v in sorted(diff.items())}
    rec["words"] = [_named("numerics", "drift", *["%s:%s!=%s" % (k, _word(v[1]), _word(v[0])) for k, v in sorted(diff.items())])] if diff else []
    now = live(torch)
    rec["live"] = "%s/matmul_tf32=%s/cudnn_tf32=%s" % (now["matmul"], _word(now["matmul_tf32"]), _word(now["cudnn_tf32"]))
    return rec


def autocast_context(policy: Policy, torch=None, device_type: str = "cuda"):
    """``torch.autocast(device_type, dtype)`` for a policy with an autocast dtype, else a null context. The kit opens it around its own
    forward; it never wraps stock code it does not own."""
    if policy.autocast is None:
        return contextlib.nullcontext()
    torch = require_torch(torch)
    return torch.autocast(device_type=device_type, dtype=getattr(torch, _DTYPE_ATTR[policy.autocast]))


def input_precision_for(matmul_precision: str, tf32x3: bool = False) -> str:
    """The ``tl.dot`` input-precision word a kernel uses for fp32 operands under a process matmul precision: ``"highest"`` -> ``"ieee"``
    (or ``"tf32x3"`` when the kit's mode declares the 3-pass compensated product — tolerance-class, near-fp32 accuracy at tensor-core
    rate), ``"high"`` / ``"medium"`` -> ``"tf32"``. bf16 / fp16 operands do not read this table (their MMA is exact in the operand type)."""
    if matmul_precision not in MATMUL_PRECISIONS:
        raise ValueError("matmul precision must be one of %s, not %r" % (MATMUL_PRECISIONS, matmul_precision))
    if matmul_precision == "highest" and tf32x3:
        return "tf32x3"
    return _INPUT_PRECISION[matmul_precision]


# ---------------------------------------------------------- TF32 library overrides -------------------------------------------------------
TF32_OVERRIDE_ENV = ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")


def tf32_override_env(environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """The TF32 override variables PRESENT in ``environ`` (this process's by default), sorted — exact names, any value."""
    environ = os.environ if environ is None else environ
    return sorted(k for k in environ if k in TF32_OVERRIDE_ENV)


def tf32_override_refusal(hits) -> str:
    """The refusal sentence for :func:`tf32_override_env` hits (a report carries it verbatim after ``NOT ACTIVE:``)."""
    return ("TF32 override variables set in the environment %s: refused — a row's numerics class is its mode's, never a library "
            "override's; unset them" % (list(hits),))


def tf32_override_gate(environ: Optional[Mapping[str, str]] = None, *, name: str = "tf32_override_env", exact: bool = True) -> Gate:
    """The environment half as an ``opt_core.gates`` check: ok with no override present. An override present is, under ``exact=True`` (an
    exact / off arm: the library override changes rounding underneath the mode — bitwise equality is at stake), refused with
    :func:`tf32_override_refusal`; under ``exact=False`` (a tolerance-class arm) ok with the word ``tf32_override=<VAR>=<value>,…`` — the arm
    runs and its line names what the library was told. ``details["hits"]`` lists the variables either way."""
    from ..report import word as _named  # noqa: PLC0415
    environ = os.environ if environ is None else environ
    hits = tf32_override_env(environ)
    details = {"hits": hits, "checked": list(TF32_OVERRIDE_ENV)}
    if hits and exact:
        return Gate(name=name, ok=False, reason=tf32_override_refusal(hits), details=details)
    if hits:
        return Gate(name=name, ok=True, details=details, words=(_named("tf32_override", ",".join("%s=%s" % (k, environ.get(k)) for k in hits)),))
    return Gate(name=name, ok=True, details=details)


# ------------------------------------------------------------- numerics signature ---------------------------------------------------------
SIGNATURE_FIELDS = ("matmul", "matmul_tf32", "cudnn_tf32", "cudnn_deterministic", "cudnn_benchmark", "deterministic_algorithms",
                    "warn_only", "autocast_gpu", "autocast_gpu_dtype", "autocast_cpu")
RESTORABLE_FIELDS = ("matmul", "matmul_tf32", "cudnn_tf32", "cudnn_deterministic", "cudnn_benchmark", "deterministic_algorithms", "warn_only")
PROCESS_STATE_FIELDS = ("grad_enabled",)        # process-global NON-numerics state: never in the signature; snapshot/restore take it by name only


def process_state_reader(torch=None) -> Tuple[Tuple[str, object], ...]:
    """:data:`PROCESS_STATE_FIELDS` as pairs — ``(("grad_enabled", bool),)``. Not part of :func:`torch_reader` / the signature."""
    torch = require_torch(torch)
    return (("grad_enabled", bool(torch.is_grad_enabled())),)


def _process_state_names(names) -> tuple:
    names = tuple(names)
    bad = [n for n in names if n not in PROCESS_STATE_FIELDS]
    if bad:
        raise ValueError("process_state names must come from %s, not %r" % (PROCESS_STATE_FIELDS, bad))
    return names


def torch_reader(torch=None) -> Tuple[Tuple[str, object], ...]:
    """The torch READER: :data:`SIGNATURE_FIELDS` as ``((field, value), ...)``. Readers are the framework extension point — a reader is any
    callable ``(module=None) -> pairs``; another framework's reader (its default matmul precision and numerics flags) registers under its own name
    with the same shape and nothing else in this module changes."""
    torch = require_torch(torch)
    try:
        ac_on = bool(torch.is_autocast_enabled("cuda"))            # device-generic API (torch >= 2.4)
        ac_dtype = str(torch.get_autocast_dtype("cuda")) if ac_on else "none"
        ac_cpu = bool(torch.is_autocast_enabled("cpu"))
    except TypeError:                                               # pre-2.4 API: per-device functions
        ac_on = bool(torch.is_autocast_enabled())
        ac_dtype = str(torch.get_autocast_gpu_dtype()) if ac_on else "none"
        ac_cpu = bool(torch.is_autocast_cpu_enabled())
    matmul, matmul_tf32, cudnn_tf32 = _fp32_state(torch)
    return (("matmul", matmul),
            ("matmul_tf32", matmul_tf32),
            ("cudnn_tf32", cudnn_tf32),
            ("cudnn_deterministic", bool(torch.backends.cudnn.deterministic)),
            ("cudnn_benchmark", bool(torch.backends.cudnn.benchmark)),
            ("deterministic_algorithms", bool(torch.are_deterministic_algorithms_enabled())),
            ("warn_only", bool(torch.is_deterministic_algorithms_warn_only_enabled())),
            ("autocast_gpu", ac_on),
            ("autocast_gpu_dtype", ac_dtype),
            ("autocast_cpu", ac_cpu))


READERS = {"torch": torch_reader}                                   # framework name -> reader; frameworks register here (lazy imports inside)


def numerics_signature(torch=None, extra: Sequence[Tuple[str, object]] = (), reader: Optional[Callable] = None) -> Tuple[Tuple[str, object], ...]:
    """The process numerics mode as ``((field, value), ...)``: the ``reader``'s pairs (default :func:`torch_reader`; ``torch`` is handed to
    it as its module argument), then the kit's ``extra`` pairs (its kernel-backend words, e.g. ``("triangle_backend", "cueq")``). A captured
    graph / jit artefact / cached tensor produced under one signature is invalid under another (module contract)."""
    reader = torch_reader if reader is None else reader
    sig = list(reader(torch))
    sig.extend((str(k), v) for k, v in extra)
    return tuple(sig)


def snapshot(torch=None, reader: Optional[Callable] = None, process_state: Sequence[str] = ()) -> dict:
    """``dict(numerics_signature())`` — the value :func:`restore` takes (an observer records it before code it does not own runs). With
    ``process_state`` (names from :data:`PROCESS_STATE_FIELDS`) the record gains ONE extra key ``"process_state": {name: value}``; without
    it the record is exactly the signature fields."""
    snap = dict(numerics_signature(torch, reader=reader))
    names = _process_state_names(process_state)
    if names:
        state = dict(process_state_reader(torch))
        snap["process_state"] = {n: state[n] for n in names}
    return snap


def restore(snap, torch=None, process_state: Sequence[str] = (), clear_autocast_cache: bool = False) -> dict:
    """Put the torch process switches back to a :func:`snapshot` (:data:`RESTORABLE_FIELDS`; autocast is a region, not a switch, and is
    only reported). Returns ``{"restored": [fields that differed and were set back — a coupled pair such as matmul + matmul_tf32 names
    both], "unrestorable": [fields that differ but are not switches]}`` — the adapter logs it; a non-empty ``unrestorable`` is the
    caller's to act on, never dropped. Opt-in, appended last, absent = today's record byte for byte: ``process_state`` (names from
    :data:`PROCESS_STATE_FIELDS`, e.g. ``("grad_enabled",)``) restores that process-global non-numerics state from the snapshot's
    ``"process_state"`` key and reports it under ``"process_state": {"restored": [...]}``; ``clear_autocast_cache=True`` calls
    ``torch.clear_autocast_cache()`` after restoring and records ``"autocast_cache": "cleared"``."""
    torch = require_torch(torch)
    snap = dict(snap)
    names = _process_state_names(process_state)
    now = dict(torch_reader(torch))
    differs = {k for k in RESTORABLE_FIELDS if k in snap and now.get(k) != snap[k]}
    restored = []
    if "matmul" in differs:
        torch.set_float32_matmul_precision(snap["matmul"]); restored.append("matmul")
    if "matmul_tf32" in differs:                                       # coupled with matmul: set explicitly too, reported either way
        torch.backends.cuda.matmul.allow_tf32 = bool(snap["matmul_tf32"]); restored.append("matmul_tf32")
    if "cudnn_tf32" in differs:
        torch.backends.cudnn.allow_tf32 = bool(snap["cudnn_tf32"]); restored.append("cudnn_tf32")
    if "cudnn_deterministic" in differs:
        torch.backends.cudnn.deterministic = bool(snap["cudnn_deterministic"]); restored.append("cudnn_deterministic")
    if "cudnn_benchmark" in differs:
        torch.backends.cudnn.benchmark = bool(snap["cudnn_benchmark"]); restored.append("cudnn_benchmark")
    if "deterministic_algorithms" in differs or "warn_only" in differs:
        torch.use_deterministic_algorithms(bool(snap.get("deterministic_algorithms", now["deterministic_algorithms"])),
                                           warn_only=bool(snap.get("warn_only", now["warn_only"])))
        restored.extend([k for k in ("deterministic_algorithms", "warn_only") if k in differs])
    after = dict(torch_reader(torch))
    unrestorable = [k for k in snap if k not in RESTORABLE_FIELDS and k in after and after[k] != snap[k]]
    rec = {"restored": restored, "unrestorable": unrestorable}
    if names:
        have = snap.get("process_state")
        if not isinstance(have, dict) or any(n not in have for n in names):
            raise ValueError("restore(process_state=%r): the snapshot carries no such process_state — take snapshot(process_state=...)" % (names,))
        cur = dict(process_state_reader(torch))
        ps_restored = []
        if "grad_enabled" in names and cur["grad_enabled"] != bool(have["grad_enabled"]):
            torch.set_grad_enabled(bool(have["grad_enabled"])); ps_restored.append("grad_enabled")
        rec["process_state"] = {"restored": ps_restored}
    if clear_autocast_cache:
        torch.clear_autocast_cache()
        rec["autocast_cache"] = "cleared"
    return rec


def signature_diff(a: Sequence[Tuple[str, object]], b: Sequence[Tuple[str, object]]) -> list:
    """Names of the fields whose values differ between two signatures (fields present in only one count as differing)."""
    da, db = dict(a), dict(b)
    return [k for k in list(dict.fromkeys(list(da) + list(db))) if da.get(k, _MISSING) != db.get(k, _MISSING)]


_MISSING = object()


class NumericsGuard:
    """Holds the signature its owner's caches were built under. :meth:`check` returns ``None`` while it is unchanged, else the named event
    ``{"event": "numerics_signature_changed", "changed": [fields], "n": <count so far>}`` after calling ``on_change(event)`` (the owner's
    drop-graphs-and-caches function) and then adopting the new signature. The adapter logs the returned event; the guard prints nothing."""

    EVENT = "numerics_signature_changed"

    def __init__(self, on_change: Optional[Callable[[dict], None]] = None, torch=None, extra: Callable[[], Sequence] = tuple,
                 reader: Optional[Callable] = None):
        self._torch = torch
        self._extra = extra
        self._reader = reader
        self.on_change = on_change
        self.changes = 0
        self.signature = numerics_signature(torch, extra(), reader)

    def snapshot(self) -> dict:
        """The held signature as a dict (what :func:`restore` takes)."""
        return dict(self.signature)

    def check(self) -> Optional[dict]:
        now = numerics_signature(self._torch, self._extra(), self._reader)
        if now == self.signature:
            return None
        changed = signature_diff(self.signature, now)
        self.changes += 1
        event = {"event": self.EVENT, "changed": changed, "n": self.changes}
        if self.on_change is not None:
            self.on_change(event)                           # the owner drops its graphs/caches first ...
        self.signature = now                                # ... then the new signature is adopted
        return event
