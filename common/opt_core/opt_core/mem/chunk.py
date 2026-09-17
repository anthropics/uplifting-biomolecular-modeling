"""opt_core.mem.chunk — chunked / streamed forms of the pair-shaped operations of a cofolding trunk (the `big` levers of family ``chunk``).

Contract. Every function here re-states an engine's STOCK statement over blocks of one axis of a pair-shaped tensor ``[..., I, J, C]``
(rows = the I axis) or an MSA-shaped tensor ``[..., S, N, C]`` (rows = the S axis), so that the transients stock holds for all rows at
once (a LayerNorm'd copy of the pair, the hidden-dim intermediates of a transition, the q/k/v/gate projections of a triangle attention,
the operands of a triangle multiplication, the softmax of a confidence head's logits) exist for one block of rows at a time. Per
element the arithmetic is the engine's own: the same modules, the same dtype casts, the same reduction axes — a chunk that only
re-orders INDEPENDENT rows is declared ``exact="bitwise"``; a chunk that splits a reduction across blocks is declared ``exact="band"``.
The declaration is made per call site (a keyword every function takes or fixes), it is written to the record with its reason, and it is
a CLAIM BY CONSTRUCTION: whether a row-blocked launch of a GEMM / LayerNorm / attention kernel returns the same bits as the full-extent
launch is a property of the device / framework / BLAS stack, so the ``bitwise`` label of a chunked op is tested per stack by the
engine's equality row (the D55 rule); a measured mismatch is recorded as ``band`` with its max|Δ|. fp32 GEMMs are the known exception
(the pair-transition kit measured 9/12 fp32 sites non-bit-exact under row chunking on the H100 stack): an fp32 site — an input or an
operand that is float32 with no autocast active — is never chunked silently; :func:`pair_transition_chunked` and
:func:`triangle_multiplication_chunked` take an ``fp32`` decision, ``passthrough`` (the un-chunked statement runs; a recorded
``fallback`` event at the lever) | ``band`` (chunked under the ``band`` label) | ``refuse``. The core never decides an engine's
threshold: :class:`ChunkPolicy` carries the engine's own settings (``always``, an explicit token count, ``off``, a table of
(min_tokens, rows), or ``auto`` from a calibration the adapter passes) and every decision is recorded with its source. A row count is
a VALUE (the ``rows`` setting or a table row); an adapter that sizes it from a byte budget at run time does so with
:func:`opt_core.mem.budget.rows_within` (the package's one budget→rows function, refusing by name when nothing fits), and the device
total the ``auto`` threshold budgets against is read through :func:`opt_core.arch.device_memory` (the package's one device-memory reader).

Levers (registered by name with :func:`opt_core.mem.registry.register`, family ``chunk``; a kit's big line names them and its
adapter passes the hook points through ``Ctx.hooks[<lever>]`` — refused by name when absent — and the settings through
the kit's ``Ctx.settings[<lever>]``):

  ``chunk_pair_transition``   hooks ``cls`` (the transition module class), ``attr`` (its forward, default ``forward``), optional ``call``
                    (``(module, args, kwargs) -> (x, rebind)``: the argument adapter, default = first positional / ``x``),
                    ``is_site`` (``(module, x, args, kwargs) -> True | <skip reason>``, default: rank >= 3 and no ``chunk_size`` kwarg),
                    ``table`` / ``calibration`` (:class:`ChunkPolicy` inputs); settings ``rows`` (256), ``tok`` (``always``),
                    ``fp32`` (``passthrough`` | ``refuse``). Registered ``bitwise``.
  ``chunk_pair_transition_band``  the same binding with fp32 sites chunked under the band label (registered ``band``); a big line
                    whose transitions run fp32 without autocast names this twin — the line's tier reads off its lever names.
  ``chunk_triangle_attention``  hooks ``cls``, ``attr``, ``parts`` (``module -> TriAttnParts``), ``starting`` (``module -> bool``), optional
                    ``call`` (default: ``(x, mask)`` from the first two positionals / ``x`` / ``mask``), ``is_site``, ``table``,
                    ``calibration``; settings ``rows`` (256), ``tok`` (``always``).
  ``chunk_triangle_multiplication``  hooks ``cls``, ``attr``, ``parts`` (``module -> TriMulParts``), optional ``call``, ``is_site``, ``table``,
                    ``calibration``; settings ``rows`` (256; the row block of both modes), ``tok`` (``always``), ``mode`` (``rows`` |
                    ``channels``), ``hidden`` (the hidden channel count; needed by ``channels``), ``channels`` (32; the channel block
                    of ``channels`` mode), ``fp32`` (``refuse`` | ``passthrough``). Registered ``bitwise``.
  ``chunk_triangle_multiplication_band``  the same binding with fp32 products chunked under the band label (registered ``band``): the
                    lever of an engine whose product operands are cast to float32.
  ``chunk_confidence_head``    hooks ``cls``, ``attr``, ``logits`` (``(module, args, kwargs) -> logits``), ``statement`` (``module -> the
                    per-row statement``), ``finish`` (``(module, rows, args, kwargs) -> the head's result``: the across-row
                    finishing statements on the assembled per-row outputs), optional ``table`` / ``calibration``; settings
                    ``rows`` (256), ``tok`` (``always``).
  ``chunk_msa_rows``     hooks ``cls``, ``attr``, optional ``call`` (its ``rebind`` may take ``(t, i0, i1)``: a masked statement
                    re-binds its mask to the block's rows), ``is_site``, ``exact`` (``bitwise``, the declaration for the
                    statement), ``reason``, ``table`` / ``calibration``; settings ``rows`` (256), ``tok`` (``always``).

``apply(ctx)`` of each lever wraps ``cls.attr``: per call the argument adapter extracts the tensor, ``is_site`` names a skip, the
policy decides (recorded: a call below the threshold is a ``skip`` event with the decision), the chunked form runs (a ``mark`` event)
or the fp32 passthrough runs the stock statement (a ``fallback`` event, so the census and the exit gate see it); ``undo`` restores
the stock attribute. The :class:`Applied` names the sites patched, the settings in force and the policy; its label is FIXED at
apply from the registration and the ``fp32`` setting (the ``_band`` twin for fp32 sites under band) and is never widened in-call — a
call whose decision would exceed it is a named fallback or a refusal, never a relabel.

Records. Every call writes one entry through the record's per-call sink: ``record.call(lever, site, exact, reason, **details)`` —
``record`` is any object with such a ``call`` method or a bare callable of that signature (the adapter passes the process's applied
record; the lever wrappers pass ``ctx.record``); ``details`` carry ``dim``, ``chunk``, ``n_chunks``, ``rows``, ``shape``, ``dtype``,
``decision`` (``chunked`` | ``passthrough``), ``mode``/``node`` where relevant and ``wall_s``. With no record the standalone sink
:data:`LOG` (a :class:`CallLog`, for tests and standalone runs — never a process's record) takes the entry; :func:`summary` folds
entries into the counts an ACTIVE line prints (per lever and site, with passthroughs counted apart from chunked calls).

Refusals. Every precondition that is absent raises :class:`ChunkRefusal` — a :class:`opt_core.mem.registry.RefusalError` whose
``.name`` is the precondition (``not_a_tensor``, ``bad_axis``, ``bad_chunk``, ``fn_output_shape``, ``bad_exact_label``, ``fp32_site``,
``no_calibration``, ``no_device_total``, ``torch_absent`` …) and whose ``.refusal`` names the lever; inside ``apply`` it is recorded as the
lever's refusal by :func:`opt_core.mem.apply`. There is no silent no-op and no silent route to the un-chunked statement: the two
un-chunked routes this module takes — the fp32 passthroughs — are named decisions with their own record entries and lever events,
and a chunk that covers the whole axis in one block runs the statement once (recorded ``n_chunks=1``), which IS the un-chunked statement.

Hook shape of the functions (what an engine's adapter binds; the per-engine module classes and attribute names live in the engines'
adapters, never here):
  * :func:`chunk_rows`                     — ``fn`` = any row-local statement; ``dim`` = the row axis; ``chunk`` = rows per block.
  * :func:`pair_transition_chunked`        — ``forward`` = the transition module's un-chunked body (LayerNorm → GEMMs → activation →
    GEMM, per position); the fp32 decision.
  * :func:`triangle_attention_chunked`     — :class:`TriAttnParts`: ``layer_norm`` (the module's input LayerNorm), ``bias`` (the
    triangle-bias projection incl. the engine's dtype cast, applied to LayerNorm'd ROWS: ``[..., R, J, H]``), ``attention`` (the
    engine's core on LayerNorm'd query rows with the FULL bias and the row-sliced mask: q/k/v/gate projections, softmax over the
    key axis, gating, output projection). Two passes: pass 1 builds the full bias from row blocks and drops each LayerNorm'd
    block; pass 2 recomputes the LayerNorm per query-row block. ``starting=False`` runs on the transposed views.
  * :func:`triangle_multiplication_chunked` — :class:`TriMulParts`: ``layer_norm_in``, ``operand_a`` / ``operand_b`` (the gated,
    masked projections on LayerNorm'd rows, with the engine's casts — a ``.float()`` here makes the product an fp32 site),
    ``product`` (the engine's einsum: reduction over the shared axis), ``finish`` (output LayerNorm → output projection × gate, per
    position). ``mode="rows"``: the fixed operand is built from row blocks and laid out as the batched einsum consumes it, the free
    operand and the product per output-row block, the output LayerNorm is row-local. ``mode="channels"``: operands per channel block
    (sliced AFTER the full-channel projection); the FULL product is held because the output LayerNorm spans the hidden channels.
  * :func:`confidence_head_chunked`        — ``statement`` = the engine's per-row stock statement on a block of pair logits
    (softmax + expectation, contact probabilities, the per-row alignment-score term) returning named tensors; outputs are
    assembled along the row axis; reductions ACROSS rows (a max over frames, a mean over rows) stay in the engine's finishing
    statements on the assembled tensors.
  * :func:`msa_rows_chunked`               — ``fn`` = a row-local MSA statement (an MSA transition, a pair-weighted average whose
    weights come from the pair); an outer-product mean chunked over S splits its reduction (``band``); chunked over its OUTPUT rows it is
    ``bitwise`` by construction — the call site declares which.

Mechanisms, each written in this module's own form (no engine code is carried): the row-chunked pair transition with its fp32
guard; row-chunked confidence scoring; the four attach-time levers (transition rows, two-pass triangle attention, conditioning rows,
dead-tensor free); the LayerNorm-recompute triangle-attention prologue (``tri_ln``) and lazy relative-position features; the
channel-chunked triangle multiplication; the calibrated quadratic ``auto`` threshold (:class:`AutoCalibration` states the policy
shape only — an engine's calibration values are never copied here); the chunk-aware block path (row bits of bf16 K=256 GEMMs and of
the cuEquivariance attention kernel are launch-extent independent on the pinned H100 stack, which is what makes row chunking
``bitwise`` there); and the chunk-regime outer-product mean.

torch is imported lazily inside the functions (never at module level): :func:`opt_core.mem.registry.discover` imports this module in
every kit's process, JAX kits included, and the levers refuse by name (``torch_absent``) where torch is absent.
"""
from __future__ import annotations

import dataclasses
import inspect
import math
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .registry import Applied, EXACT_LABELS, LEVERS, Refusal, RefusalError, off_ref, refuse, register

if TYPE_CHECKING:                                     # annotations only — the runtime imports torch inside the functions
    import torch

__all__ = [
    "LEVERS_HERE", "DECLARED_LABELS", "FP32_POLICIES", "LOG", "CallLog", "ChunkRefusal", "summary", "active_fragment", "autocast_active",
    "chunk_rows", "pair_transition_chunked",
    "TriAttnParts", "triangle_attention_chunked",
    "TriMulParts", "triangle_multiplication_chunked",
    "confidence_head_chunked", "msa_rows_chunked",
    "AutoCalibration", "ChunkDecision", "ChunkPolicy", "parse_setting",
    "chunk_pair_transition", "chunk_pair_transition_band", "chunk_triangle_attention", "chunk_triangle_multiplication", "chunk_triangle_multiplication_band",
    "chunk_confidence_head", "chunk_msa_rows",
]

LEVERS_HERE = ("chunk_pair_transition", "chunk_pair_transition_band", "chunk_triangle_attention", "chunk_triangle_multiplication",
               "chunk_triangle_multiplication_band", "chunk_confidence_head", "chunk_msa_rows")
DECLARED_LABELS = ("bitwise", "band")                 # what a call site declares; "measured" (registry.EXACT_LABELS) is the label of a measurement record
FP32_POLICIES = ("passthrough", "band", "refuse")
Record = Optional[Any]                                # an object with .call(lever, site, exact, reason, **details) or a bare callable of that signature


def _torch():
    """The torch module, imported on first use; a process without torch refuses by name."""
    try:
        import torch
    except ImportError as e:                          # pragma: no cover - the refusal path of a torch-less process
        raise ChunkRefusal("chunk", "torch_absent", f"torch is not importable: {e}") from None
    return torch


class ChunkRefusal(RefusalError):
    """A precondition of a chunked op is absent: refused by name — ``.name`` (the precondition), ``.detail``, ``.refusal`` (the
    registry's :class:`Refusal` naming the lever); never a silent no-op or an un-recorded stock route."""

    def __init__(self, lever: str, name: str, detail: str, **details):
        super().__init__(refuse(lever, name, detail, **details))
        self.lever = lever
        self.name = name
        self.detail = detail


# ------------------------------------------------------------------------------------------------------------ the record sink


class CallLog:
    """The standalone per-call sink (tests, standalone runs): the same ``call`` duck type as a process's applied record."""

    def __init__(self):
        self.entries: List[dict] = []

    def call(self, lever: str, site: str, exact: str, reason: str, **details) -> dict:
        entry = {"lever": lever, "site": site, "exact": exact, "reason": reason}
        entry.update(details)
        self.entries.append(entry)
        return entry

    def clear(self) -> None:
        self.entries.clear()

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i):
        return self.entries[i]


LOG = CallLog()


def _emit(record: Record, lever: str, site: str, exact: str, reason: str, **details):
    """One per-call entry to the record's sink (``record.call``; a bare callable; :data:`LOG` when none): a sink that takes no
    details receives the four fields."""
    sink = LOG if record is None else record
    fn = getattr(sink, "call", None)
    if fn is None and callable(sink):
        fn = sink
    if fn is None and hasattr(sink, "note"):                     # a record without a per-call sink: its note line + the standalone log (never dropped)
        sink.note(f"{lever} {site} exact={exact} decision={details.get('decision', 'chunked')}: {reason}")
        return LOG.call(lever, site, exact, reason, **details)
    if not callable(fn):
        raise ChunkRefusal(lever, "bad_record", f"record is {type(record).__name__}: expected an object with .call(lever, site, exact, reason, **details) or a callable")
    try:
        return fn(lever, site, exact, reason, **details)
    except TypeError:
        return fn(lever, site, exact, reason)


def summary(entries: Optional[Sequence[Mapping]] = None) -> Dict[str, dict]:
    """Fold per-call entries (default :data:`LOG`) into ``{lever: {calls, chunked, passthrough, sites: {site: {exact: count}}}}``."""
    out: Dict[str, dict] = {}
    for e in (LOG.entries if entries is None else entries):
        lv = out.setdefault(e.get("lever", "?"), {"calls": 0, "passthrough": 0, "chunked": 0, "sites": {}})
        lv["calls"] += 1
        lv["passthrough" if e.get("decision") == "passthrough" else "chunked"] += 1
        site = lv["sites"].setdefault(e.get("site", "?"), {})
        site[e.get("exact", "?")] = site.get(e.get("exact", "?"), 0) + 1
    return out


def active_fragment(entries: Optional[Sequence[Mapping]] = None) -> str:
    """The ACTIVE-line fragment: ``chunk: <lever>=<calls>/<passthrough> …`` (both counts, always)."""
    s = summary(entries)
    return "chunk: " + " ".join(f"{lv}={c['calls']}/{c['passthrough']}" for lv, c in s.items()) if s else "chunk: none"


# ------------------------------------------------------------------------------------------------------------ preconditions


def _tensor(lever: str, what: str, x: Any):
    if not _torch().is_tensor(x):
        raise ChunkRefusal(lever, "not_a_tensor", f"{what} is {type(x).__name__}, expected a torch.Tensor")
    return x


def _axis(lever: str, x, dim: int, what: str = "dim") -> int:
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ChunkRefusal(lever, "bad_axis", f"{what}={dim!r} is not an int")
    if not -x.dim() <= dim < x.dim():
        raise ChunkRefusal(lever, "bad_axis", f"{what}={dim} is outside a rank-{x.dim()} tensor")
    return dim % x.dim()


def _chunk(lever: str, chunk: Any) -> int:
    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1:
        raise ChunkRefusal(lever, "bad_chunk", f"chunk={chunk!r} is not a positive int")
    return chunk


def _label(lever: str, exact: Any) -> str:
    if exact not in DECLARED_LABELS:
        raise ChunkRefusal(lever, "bad_exact_label", f"exact={exact!r}, expected one of {'|'.join(DECLARED_LABELS)} (declared per call site)")
    return exact


def _callable(lever: str, what: str, fn: Any) -> Callable:
    if not callable(fn):
        raise ChunkRefusal(lever, "not_callable", f"{what} is {type(fn).__name__}, expected a callable")
    return fn


def _fp32_policy(lever: str, fp32: Any) -> str:
    if fp32 not in FP32_POLICIES:
        raise ChunkRefusal(lever, "bad_fp32_policy", f"fp32={fp32!r}, expected {'|'.join(FP32_POLICIES)}")
    return fp32


def _dtype(x) -> str:
    return str(x.dtype).replace("torch.", "")


def autocast_active(device_type: str) -> bool:
    """Whether autocast is enabled for ``device_type`` in the calling thread (the dtype a Linear runs in is decided by it, not by the input)."""
    torch = _torch()
    try:
        return bool(torch.is_autocast_enabled(device_type))
    except TypeError:                                            # the older signature: cuda only, cpu through its own query
        if device_type == "cpu":
            return bool(torch.is_autocast_cpu_enabled())
        return bool(torch.is_autocast_enabled())


def _fp32_site(t) -> bool:
    """An fp32 site: the tensor is float32 and no autocast is active for its device (the GEMMs would run in fp32)."""
    return t.dtype == _torch().float32 and not autocast_active(t.device.type)


# ------------------------------------------------------------------------------------------------------------ chunk_rows


def chunk_rows(fn: Callable, x: "torch.Tensor", dim: int, chunk: int, *, exact: str, reason: str, record: Record = None,
               out: Optional["torch.Tensor"] = None, accumulate: bool = False, with_offsets: bool = False,
               lever: str = "chunk_rows", site: str = "chunk_rows", extra: Optional[Mapping[str, Any]] = None) -> "torch.Tensor":
    """Apply ``fn`` to consecutive blocks of ``chunk`` entries of ``x`` along axis ``dim`` and assemble the results along that axis.

    ``fn(block)`` (or ``fn(block, i0, i1)`` with ``with_offsets=True``, the offsets of the block on the axis) returns a tensor of the
    same rank as ``x`` whose extent on ``dim`` equals the block's; its dtype fixes the output's. The output is allocated on the
    first block (``torch.empty``) or is ``out`` (same rank, full extent on ``dim``); with ``accumulate=True`` each block is ADDED into
    ``out`` (the residual add of a trunk layer per block: per element the same add as the full statement, so no full output plane
    is ever held). ``chunk >= extent`` runs ``fn`` once on ``x`` itself (no copy: the un-chunked statement; ``n_chunks=1`` in the
    record). ``exact`` is the call site's declaration: ``bitwise`` = ``fn`` is row-local along ``dim`` (a re-ordering of independent
    rows); ``band`` = ``fn`` splits a reduction over ``dim`` across blocks — the reason is recorded verbatim. The record entry
    carries ``lever``/``site`` (the adapter's names), the axis, the block size, the block count, the extent and the shape/dtype.
    """
    torch = _torch()
    x = _tensor(lever, "x", x)
    fn = _callable(lever, "fn", fn)
    d = _axis(lever, x, dim)
    chunk = _chunk(lever, chunk)
    exact = _label(lever, exact)
    if not isinstance(reason, str) or not reason.strip():
        raise ChunkRefusal(lever, "no_reason", "the exactness declaration needs a non-empty reason")
    n = x.shape[d]
    t0 = time.perf_counter()
    if out is not None:
        out = _tensor(lever, "out", out)
        if out.dim() != x.dim() or out.shape[d] != n:
            raise ChunkRefusal(lever, "bad_out", f"out has shape {tuple(out.shape)}, expected rank {x.dim()} with extent {n} on axis {d}")
    if accumulate and out is None:
        raise ChunkRefusal(lever, "accumulate_without_out", "accumulate=True needs an out tensor to add into")
    n_chunks = 0
    i0 = 0
    while i0 < n:
        i1 = min(n, i0 + chunk)
        block = x if (i0 == 0 and i1 == n) else x.narrow(d, i0, i1 - i0)
        y = fn(block, i0, i1) if with_offsets else fn(block)
        if not torch.is_tensor(y):
            raise ChunkRefusal(lever, "fn_output_not_tensor", f"fn returned {type(y).__name__} for rows [{i0}, {i1})")
        if y.dim() != x.dim() or y.shape[d] != i1 - i0:
            raise ChunkRefusal(lever, "fn_output_shape", f"fn returned shape {tuple(y.shape)} for rows [{i0}, {i1}) of {tuple(x.shape)} on axis {d}")
        if out is None:
            if i0 == 0 and i1 == n:
                out = y                                          # one block: the statement's own output, no copy
                n_chunks = 1
                break
            shape = list(y.shape)
            shape[d] = n
            out = torch.empty(shape, dtype=y.dtype, device=y.device)
        elif y.dtype != out.dtype and not accumulate:
            raise ChunkRefusal(lever, "fn_output_dtype", f"fn returned {y.dtype} for rows [{i0}, {i1}), the output is {out.dtype}")
        dst = out.narrow(d, i0, i1 - i0)
        if accumulate:
            dst.add_(y)
        else:
            dst.copy_(y)
        del y
        n_chunks += 1
        i0 = i1
    details = {"dim": d, "chunk": chunk, "n_chunks": n_chunks, "rows": n, "shape": tuple(x.shape), "dtype": _dtype(x),
               "accumulate": bool(accumulate), "wall_s": round(time.perf_counter() - t0, 6)}
    if extra:
        details.update(dict(extra))
    details.setdefault("decision", "chunked")
    _emit(record, lever, site, exact, reason, **details)
    return out


def _passthrough_out(lever: str, y, x, out, accumulate: bool):
    """The fp32 passthrough's output handling: ``out``/``accumulate`` as the chunked path would apply them."""
    torch = _torch()
    if not torch.is_tensor(y) or y.shape != x.shape:
        raise ChunkRefusal(lever, "forward_output_shape", f"the statement returned {type(y).__name__ if not torch.is_tensor(y) else tuple(y.shape)} for {tuple(x.shape)}")
    if out is not None:
        _tensor(lever, "out", out)
        if out.shape != x.shape:
            raise ChunkRefusal(lever, "bad_out", f"out has shape {tuple(out.shape)}, expected {tuple(x.shape)}")
        (out.add_ if accumulate else out.copy_)(y)
        return out
    return y


# ------------------------------------------------------------------------------------------------------------ pair transition


def pair_transition_chunked(x: "torch.Tensor", forward: Callable, *, chunk: int, record: Record = None, fp32: str = "passthrough",
                            row_dim: int = -3, out: Optional["torch.Tensor"] = None, accumulate: bool = False,
                            lever: str = "chunk_pair_transition") -> "torch.Tensor":
    """Row-chunked pair transition: the module's un-chunked body ``forward`` (LayerNorm → up-projections → activation → down-projection,
    per position) on blocks of ``chunk`` rows of ``x [..., I, J, C]`` (``row_dim`` = the I axis). Stock holds the LayerNorm'd copy and up
    to three ``[..., I, J, 4C]`` intermediates for all rows; here they exist for one block.

    Exactness: ``bitwise`` by construction (row-local); the pair-transition kit's evidence is that bf16 / autocast sites hold bit-exact
    on the H100 stack while fp32 GEMM sites do NOT (their launch-extent-dependent reduction order) — so an fp32 site (``x`` is float32
    and no autocast is active for its device) is decided by ``fp32``: ``"passthrough"`` runs ``forward(x)`` un-chunked and records
    ``decision="passthrough"`` (the tested kit's guard; a ``fallback`` event at the lever); ``"band"`` chunks it under the ``band``
    label; ``"refuse"`` raises ``ChunkRefusal("fp32_site")``. ``out``/``accumulate`` as in :func:`chunk_rows` (a per-block residual add).
    """
    x = _tensor(lever, "x", x)
    fn = _callable(lever, "forward", forward)
    if x.dim() < 3:
        raise ChunkRefusal(lever, "not_pair_shaped", f"x has rank {x.dim()}, expected [..., I, J, C]")
    d = _axis(lever, x, row_dim, "row_dim")
    fp32 = _fp32_policy(lever, fp32)
    fp32_site = _fp32_site(x)
    if fp32_site and fp32 == "refuse":
        raise ChunkRefusal(lever, "fp32_site", "x is float32 without autocast: a chunked fp32 GEMM is not bitwise by construction (fp32='refuse')")
    if fp32_site and fp32 == "passthrough":
        t0 = time.perf_counter()
        y = _passthrough_out(lever, fn(x), x, out, accumulate)
        _emit(record, lever, "pair_transition", "bitwise", "fp32 site run un-chunked (the kit's guard): the un-levered statement",
              decision="passthrough", dim=d, chunk=chunk, n_chunks=1, rows=x.shape[d], shape=tuple(x.shape), dtype="float32",
              accumulate=bool(accumulate), wall_s=round(time.perf_counter() - t0, 6))
        return y
    exact = "band" if fp32_site else "bitwise"
    reason = ("fp32 site chunked under fp32='band': fp32 GEMM row bits depend on the launch extent" if fp32_site
              else "row-local statement (LayerNorm, GEMMs, activation per position); reduction axes untouched")
    return chunk_rows(fn, x, d, chunk, exact=exact, reason=reason, record=record, out=out, accumulate=accumulate,
                      lever=lever, site="pair_transition", extra={"decision": "chunked", "fp32_site": fp32_site})


# ------------------------------------------------------------------------------------------------------------ triangle attention


@dataclasses.dataclass(frozen=True)
class TriAttnParts:
    """The three callables of a triangle-attention module, bound by the adapter (see the module docstring's hook shape).

    ``layer_norm(x_rows)``: the input LayerNorm on a block of rows. ``bias(x_ln_rows)``: the triangle-bias projection (with the engine's
    cast, e.g. ``.float()``) on LayerNorm'd rows ``[..., R, J, C] -> [..., R, J, H]`` — the row axis stays where it is; the adapter's
    ``attention`` permutes the assembled full bias as its core expects. ``attention(x_ln_rows, bias_full, mask_rows)``: the engine's
    attention statement on LayerNorm'd QUERY rows with the FULL bias (every query row attends over the whole key axis with the bias of
    every (j, k) pair) and the mask sliced to those rows (``None`` when the call has none).
    """
    layer_norm: Callable
    bias: Callable
    attention: Callable


def triangle_attention_chunked(x: "torch.Tensor", mask: Optional["torch.Tensor"], parts: TriAttnParts, *, chunk: int, starting: bool = True,
                               record: Record = None, row_dim: int = -3, mask_row_dim: int = -2, out: Optional["torch.Tensor"] = None,
                               accumulate: bool = False, lever: str = "chunk_triangle_attention") -> "torch.Tensor":
    """Two-pass, query-row-chunked triangle attention on ``x [..., I, J, C]`` with ``mask [..., I, J]`` (or ``None``).

    Pass 1 (``bias``): for each block of rows, LayerNorm → bias projection; the full bias ``[..., I, J, H]`` is assembled and each
    LayerNorm'd block is dropped — stock computes the bias once from a LayerNorm'd copy of the WHOLE pair that it then keeps for
    the attention. Pass 2 (``attention``): for each block of query rows, LayerNorm recomputed → the engine's attention core with
    the full bias and the rows' mask. Held: the full bias and one block's LayerNorm'd rows / projections / logits — never the
    LayerNorm'd pair (measured on H100: the lever's peak is stock's row-chunked peak minus exactly the LayerNorm'd plane,
    ``N² · C · 2 B``, and minus the output plane too in the residual form). The bias is full because a query row ``(i, j)`` attends
    over keys ``(i, k)`` with the bias of pair ``(j, k)`` — every row needs every bias entry; the mask is per query row and is sliced
    with the block. ``starting=False`` (the ending node) runs the same two passes on the transposed views ``x.transpose(-2, -3)`` /
    ``mask.transpose(-1, -2)`` and returns the transposed view of the result (the engine's statement transposes back the same way; a
    LayerNorm of a strided view equals the LayerNorm of its contiguous copy). Exactness: ``bitwise`` by construction (pass 1 re-orders
    independent rows; pass 2 is the per-query-row statement stock's own chunked path runs — the softmax over the key axis is whole
    within a row); tested per stack by the equality row. ``out``/``accumulate`` (``[..., I, J, C]`` in the caller's orientation)
    as in :func:`chunk_rows`.
    """
    x = _tensor(lever, "x", x)
    if not isinstance(parts, TriAttnParts):
        raise ChunkRefusal(lever, "bad_parts", f"parts is {type(parts).__name__}, expected TriAttnParts")
    for name in ("layer_norm", "bias", "attention"):
        _callable(lever, f"parts.{name}", getattr(parts, name))
    if x.dim() < 3:
        raise ChunkRefusal(lever, "not_pair_shaped", f"x has rank {x.dim()}, expected [..., I, J, C]")
    if x.shape[-2] != x.shape[-3]:
        raise ChunkRefusal(lever, "not_square", f"x has pair extents {x.shape[-3]} x {x.shape[-2]}, a triangle attention needs I == J")
    d = _axis(lever, x, row_dim, "row_dim")
    if d != x.dim() - 3:
        raise ChunkRefusal(lever, "bad_axis", f"row_dim={row_dim} is not the I axis (-3) of [..., I, J, C]")
    if mask is not None:
        mask = _tensor(lever, "mask", mask)
        if mask.dim() != x.dim() - 1 or tuple(mask.shape[-2:]) != tuple(x.shape[-3:-1]):
            raise ChunkRefusal(lever, "bad_mask", f"mask has shape {tuple(mask.shape)}, expected {tuple(x.shape[:-1])}")
        _axis(lever, mask, mask_row_dim, "mask_row_dim")
    if out is not None:
        out = _tensor(lever, "out", out)
        if out.shape != x.shape:
            raise ChunkRefusal(lever, "bad_out", f"out has shape {tuple(out.shape)}, expected {tuple(x.shape)}")
    chunk = _chunk(lever, chunk)
    t0 = time.perf_counter()
    if not starting:
        x = x.transpose(-2, -3)
        mask = None if mask is None else mask.transpose(-1, -2)
        out = None if out is None else out.transpose(-2, -3)
    node = "starting" if starting else "ending"
    # pass 1: the full triangle bias from row blocks; no LayerNorm'd plane is kept
    bias_full = chunk_rows(lambda xr: parts.bias(parts.layer_norm(xr)), x, d, chunk, exact="bitwise",
                           reason=f"{node} node, pass 1: LayerNorm + bias projection per row block (row-local)", record=record,
                           lever=lever, site="triangle_attention_bias", extra={"node": node})
    md = None if mask is None else _axis(lever, mask, mask_row_dim, "mask_row_dim")

    def _rows(xr, i0, i1):
        mr = None if mask is None else mask.narrow(md, i0, i1 - i0)
        return parts.attention(parts.layer_norm(xr), bias_full, mr)

    y = chunk_rows(_rows, x, d, chunk, exact="bitwise",
                   reason=f"{node} node, pass 2: LayerNorm recomputed per query-row block; full bias; softmax whole over the key axis",
                   record=record, out=out, accumulate=accumulate, with_offsets=True, lever=lever, site="triangle_attention",
                   extra={"node": node, "wall_s_total": round(time.perf_counter() - t0, 6)})
    del bias_full
    return y.transpose(-2, -3) if not starting else y


# ------------------------------------------------------------------------------------------------------------ triangle multiplication


@dataclasses.dataclass(frozen=True)
class TriMulParts:
    """The callables of a triangle-multiplication module, bound by the adapter.

    ``layer_norm_in(x_rows)``; ``operand_a(x_ln, mask)`` / ``operand_b(x_ln, mask)``: the gated, masked hidden projections with the
    engine's casts (``mask`` is the matching slice of the pair mask or ``None``; an engine whose projection fuses ``a|b`` slices the
    halves AFTER the projection); ``product(a, b)``: the engine's einsum over the shared axis — outgoing ``"...ikc,...jkc->...ijc"``
    (a: rows i), incoming ``"...kic,...kjc->...ijc"`` (a: columns i) — called with the free operand restricted to one block of output
    rows and the fixed operand ``b`` whole; ``finish(product_rows, x_ln_rows)``: output LayerNorm → output projection × sigmoid gate
    (per position).
    """
    layer_norm_in: Callable
    operand_a: Callable
    operand_b: Callable
    product: Callable
    finish: Callable
    outgoing: bool = True


def _arrange_fixed(b, outgoing: bool):
    """The fixed operand in the storage order the batched einsum consumes it in — ``(..., C, K, J)`` for the right operand of both
    products (batch dims, the summed axis, the free axis) — as a strided view of that contiguous buffer with the logical shape unchanged.
    The einsum's own permute of a row-major ``b`` to that order is not contiguous, so its reshape to the batched matrices CLONES ``b``
    once per call (a full hidden plane per output-row block, measured on H100); with the arranged layout the permute is a
    view. The batched matrices the BLAS receives are the same bytes in the same order either way — the bits are the same.
    """
    nd = b.dim()
    perm = (*range(nd - 3), nd - 1, nd - 2, nd - 3) if outgoing else (*range(nd - 3), nd - 1, nd - 3, nd - 2)   # b: [.., J, K, C] | [.., K, J, C] -> (.., C, K, J)
    inv = [0] * nd
    for i, p in enumerate(perm):
        inv[p] = i
    return b.permute(*perm).contiguous().permute(*inv)


def triangle_multiplication_chunked(x: "torch.Tensor", mask: Optional["torch.Tensor"], parts: TriMulParts, *, chunk: int, mode: str = "rows",
                                    hidden: Optional[int] = None, row_block: Optional[int] = None, fp32: str = "band", arrange: bool = True, record: Record = None,
                                    out: Optional["torch.Tensor"] = None, accumulate: bool = False, lever: str = "chunk_triangle_multiplication") -> "torch.Tensor":
    """Chunked triangle multiplication on ``x [..., I, J, C]`` with ``mask [..., I, J]`` (or ``None``).

    ``mode="rows"``: the fixed operand ``b`` is built whole from row blocks (LayerNorm recomputed per block, never a LayerNorm'd
    plane) and, with ``arrange=True``, laid out as the batched einsum consumes it (:func:`_arrange_fixed`: one transient copy of
    ``b`` at arrangement, no clone per block); then per block of OUTPUT rows: the free operand ``a`` from the block's rows (outgoing)
    or columns (incoming — a column block of ``x`` is a strided view, the einsum is stock's with I = the block), the product over the
    whole shared axis (the reduction is not split), ``finish`` on the block (output LayerNorm per position, gate from the block's
    LayerNorm'd rows). Held: ``b`` + the output + one block of ``a`` / product / output-LayerNorm transients — stock holds the
    LayerNorm'd pair, ``a``, ``b``, the product and the output for all rows. Measured (H100, torch 2.13.0+cu130; bf16, C = C_hid = 128, 6000
    tokens, peak allocated above the resident baseline): rows 512 arranged 22.39 GB, rows 128 arranged 19.42 GB, vs 30.80 GB
    un-arranged (= b + the einsum's clone of b + output + block) and stock's 55.30; the arranged and un-arranged outputs are
    torch.equal. Without the arrangement the einsum's clone of ``b`` is held beside ``b`` for every block.

    ``mode="channels"`` (``hidden`` = the hidden channel count, ``chunk`` = channels per block, ``row_block`` = the row block of the
    operand and finish passes, default ``chunk``; recorded): ``a`` and ``b`` per channel block — each assembled from row blocks of the LayerNorm'd pair (LayerNorm recomputed per block) as the block's FULL-channel operand
    sliced to the channel range, so the projection GEMMs are stock's own launches (a channel-sliced projection weight changes the
    GEMM's extent and is NOT bit-exact on a CPU BLAS stack — measured, max|Δ| 2.4e-7 fp32 — hence the slice is taken after the
    projection, at the cost of repeating the projection once per channel block) — the product written channel block by channel
    block into ONE full product tensor, then ``finish`` per row block. The full product is held because the output LayerNorm
    normalises over ALL hidden channels of a position — a channel chunk bounds the operands, not the product; a running-statistics
    LayerNorm would split that reduction (``band``) and is not offered. Held: the full product + two channel-block planes + the
    output + a block. Measured (H100, torch 2.13.0+cu130; bf16, C = C_hid = 128, 6000 tokens): channels with 32-channel blocks 20.74 GB
    (``row_block`` 32) — smaller than the un-arranged rows mode (28.44–30.80) and between the arranged rows modes (19.42 at rows
    128, 22.39 at rows 512); with fp32 operands at 4000 tokens: rows 128 17.70, rows 512 21.64, channels 32 20.58 GB vs
    stock 49.15. Neither mode dominates: the engine's own measurements decide, never a rule from this docstring.

    Exactness: ``bitwise`` by construction in both modes (per channel the product is an independent GEMM of the batched einsum;
    the reduction over the shared axis is whole) for bf16 / autocast products; an fp32 PRODUCT — the operands are float32 with no
    autocast (an engine that casts ``a``/``b`` to float32 before its einsum, whatever ``x`` is) — is a chunked fp32 GEMM whose M
    extent (rows mode) or batch extent (channels mode) the chunk changes, the non-bit-exact class the pair-transition kit measured;
    decided by ``fp32``: ``"band"`` (default: chunked under the ``band`` label), ``"passthrough"`` (the un-chunked composition of
    the parts runs; ``decision="passthrough"``, a ``fallback`` event at the lever), ``"refuse"``. The operand dtype is probed on a
    one-row block before anything is held. The record names the mode and the decision. Measured (H100, torch 2.13.0+cu130): the chunked fp32 product (rows 512 / 128, channels 32; K = N = 2000 and 4000, C_hid = 128) was torch.equal to
    stock's in every cell — ``band`` is the declared label; an engine's equality row may narrow it to ``bitwise`` on its stack.
    """
    torch = _torch()
    x = _tensor(lever, "x", x)
    if not isinstance(parts, TriMulParts):
        raise ChunkRefusal(lever, "bad_parts", f"parts is {type(parts).__name__}, expected TriMulParts")
    for name in ("layer_norm_in", "operand_a", "operand_b", "product", "finish"):
        _callable(lever, f"parts.{name}", getattr(parts, name))
    if x.dim() < 3:
        raise ChunkRefusal(lever, "not_pair_shaped", f"x has rank {x.dim()}, expected [..., I, J, C]")
    if x.shape[-2] != x.shape[-3]:
        raise ChunkRefusal(lever, "not_square", f"x has pair extents {x.shape[-3]} x {x.shape[-2]}, a triangle multiplication needs I == J")
    if mask is not None:
        mask = _tensor(lever, "mask", mask)
        if tuple(mask.shape) != tuple(x.shape[:-1]):
            raise ChunkRefusal(lever, "bad_mask", f"mask has shape {tuple(mask.shape)}, expected {tuple(x.shape[:-1])}")
    if mode not in ("rows", "channels"):
        raise ChunkRefusal(lever, "bad_mode", f"mode={mode!r}, expected rows|channels")
    if out is not None:
        out = _tensor(lever, "out", out)
        if out.shape != x.shape:
            raise ChunkRefusal(lever, "bad_out", f"out has shape {tuple(out.shape)}, expected {tuple(x.shape)}")
    chunk = _chunk(lever, chunk)
    fp32 = _fp32_policy(lever, fp32)
    if mode == "channels" and (not isinstance(hidden, int) or isinstance(hidden, bool) or hidden < 1):
        raise ChunkRefusal(lever, "bad_hidden", f"hidden={hidden!r} is not a positive int (the hidden channel count)")
    d = x.dim() - 3                                              # the I axis
    dc = x.dim() - 2                                             # the J axis
    t0 = time.perf_counter()
    direction = "outgoing" if parts.outgoing else "incoming"
    # the operand dtype decides whether the product is an fp32 site: probed on one row, nothing held
    probe = parts.operand_a(parts.layer_norm_in(x.narrow(d, 0, 1)), None if mask is None else mask.narrow(d, 0, 1))
    if not torch.is_tensor(probe) or probe.dim() != x.dim():
        raise ChunkRefusal(lever, "operand_output_shape", f"operand_a returned {tuple(probe.shape) if torch.is_tensor(probe) else type(probe).__name__} for one row of {tuple(x.shape)}")
    if mode == "channels" and probe.shape[-1] != hidden:
        raise ChunkRefusal(lever, "operand_output_shape", f"operand_a returned {probe.shape[-1]} hidden channels, hidden={hidden}")
    fp32_site = _fp32_site(probe)
    del probe
    if fp32_site and fp32 == "refuse":
        raise ChunkRefusal(lever, "fp32_site", "the operands are float32 without autocast: a chunked fp32 product is not bitwise by construction (fp32='refuse')")
    if fp32_site and fp32 == "passthrough":
        x_ln = parts.layer_norm_in(x)
        p = parts.product(parts.operand_a(x_ln, mask), parts.operand_b(x_ln, mask))
        y = _passthrough_out(lever, parts.finish(p, x_ln), x, out, accumulate)
        del p, x_ln
        _emit(record, lever, "triangle_multiplication", "bitwise", f"{direction}: fp32 product run un-chunked (fp32='passthrough'): the un-levered composition of the parts",
              decision="passthrough", mode=mode, dim=d, chunk=chunk, n_chunks=1, rows=x.shape[d], shape=tuple(x.shape), dtype=_dtype(x),
              accumulate=bool(accumulate), fp32_site=True, wall_s=round(time.perf_counter() - t0, 6))
        return y
    exact = "band" if fp32_site else "bitwise"
    why = " [fp32 product chunked under fp32='band': the GEMM extent the chunk changes decides fp32 row bits]" if fp32_site else ""
    extra = {"mode": mode, "fp32_site": fp32_site}
    if mode == "rows":
        b_full = chunk_rows(lambda xr, i0, i1: parts.operand_b(parts.layer_norm_in(xr), None if mask is None else mask.narrow(d, i0, i1 - i0)),
                            x, d, chunk, exact="bitwise", reason=f"{direction}: fixed operand from row blocks (row-local)", record=record,
                            with_offsets=True, lever=lever, site="triangle_multiplication_operand", extra={**extra, "operand": "b"})
        if arrange:
            b_full = _arrange_fixed(b_full, parts.outgoing)

        def _rows(xr, i0, i1):
            x_ln_rows = parts.layer_norm_in(xr)
            if parts.outgoing:
                a_blk = parts.operand_a(x_ln_rows, None if mask is None else mask.narrow(d, i0, i1 - i0))
            else:
                xc = x.narrow(dc, i0, i1 - i0)                    # a column block (strided view): the incoming free operand
                a_blk = parts.operand_a(parts.layer_norm_in(xc), None if mask is None else mask.narrow(dc, i0, i1 - i0))
            p = parts.product(a_blk, b_full)
            del a_blk
            return parts.finish(p, x_ln_rows)

        y = chunk_rows(_rows, x, d, chunk, exact=exact,
                       reason=f"{direction}: free operand + product per output-row block; reduction over the shared axis whole; output LayerNorm per position{why}",
                       record=record, out=out, accumulate=accumulate, with_offsets=True, lever=lever, site="triangle_multiplication",
                       extra={**extra, "arranged": bool(arrange), "wall_s_total": round(time.perf_counter() - t0, 6)})
        del b_full
        return y
    # channels
    n = x.shape[d]
    row_block = chunk if row_block is None else _chunk(lever, row_block)          # the row block of the operand / finish passes
    extra["row_block"] = row_block
    product = None
    c0 = 0
    n_c = 0
    while c0 < hidden:
        c1 = min(hidden, c0 + chunk)

        def _operand(which, c0=c0, c1=c1):
            fn = parts.operand_a if which == "a" else parts.operand_b

            def _rows(xr, i0, i1):
                full = fn(parts.layer_norm_in(xr), None if mask is None else mask.narrow(d, i0, i1 - i0))
                if not torch.is_tensor(full) or full.shape[-1] != hidden:
                    raise ChunkRefusal(lever, "operand_output_shape", f"operand_{which} returned {tuple(full.shape) if torch.is_tensor(full) else type(full).__name__}, expected {hidden} hidden channels")
                return full[..., c0:c1].contiguous()

            return chunk_rows(_rows, x, d, row_block, exact="bitwise", reason=f"{direction}: operand {which}, channels [{c0}, {c1}) sliced from the full projection per row block",
                              record=record, with_offsets=True, lever=lever, site="triangle_multiplication_operand", extra={**extra, "operand": which})

        a_c = _operand("a")
        b_c = _operand("b")
        p_c = parts.product(a_c, b_c)
        del a_c, b_c
        if not torch.is_tensor(p_c) or p_c.shape[-1] != c1 - c0:
            raise ChunkRefusal(lever, "product_output_shape", f"product returned {tuple(p_c.shape) if torch.is_tensor(p_c) else type(p_c).__name__} for channels [{c0}, {c1})")
        if product is None:
            shape = list(p_c.shape)
            shape[-1] = hidden
            product = torch.empty(shape, dtype=p_c.dtype, device=p_c.device)
        product[..., c0:c1].copy_(p_c)
        del p_c
        n_c += 1
        c0 = c1
    _emit(record, lever, "triangle_multiplication_product", exact,
          f"{direction}: product per channel block into one full product (the output LayerNorm spans the hidden channels){why}",
          decision="chunked", dim=x.dim() - 1, chunk=chunk, n_chunks=n_c, rows=hidden, shape=tuple(x.shape), dtype=_dtype(x),
          wall_s=round(time.perf_counter() - t0, 6), **extra)
    y = chunk_rows(lambda xr, i0, i1: parts.finish(product.narrow(d, i0, i1 - i0), parts.layer_norm_in(xr)), x, d, row_block, exact="bitwise",
                   reason=f"{direction}: finish per row block (output LayerNorm per position, gate from the block's LayerNorm'd rows)",
                   record=record, out=out, accumulate=accumulate, with_offsets=True, lever=lever, site="triangle_multiplication",
                   extra={**extra, "wall_s_total": round(time.perf_counter() - t0, 6)})
    del product
    return y


# ------------------------------------------------------------------------------------------------------------ confidence head


def confidence_head_chunked(logits: "torch.Tensor", statement: Callable, *, chunk: int, record: Record = None, row_dim: int = -3,
                            lever: str = "chunk_confidence_head") -> Dict[str, "torch.Tensor"]:
    """Row-chunked confidence scoring: the engine's per-row stock statement on blocks of ``chunk`` rows of the pair logits
    ``logits [..., I, J, bins]`` (``row_dim`` = the I axis), its named outputs assembled along the row axis.

    ``statement(logit_rows) -> {name: tensor}``: every output carries the block's row count on the SAME axis index as the logits'
    row axis (``[..., R, J]`` for a per-pair expectation, ``[..., R]`` for a per-row term); a reduction inside a row (softmax over
    bins, the expectation, a mean over J) is whole. Reductions ACROSS rows (a max over alignment frames, a mean over rows, a
    per-chain aggregate) are the engine's finishing statements on the assembled tensors — outside this function, so the chunk never
    splits them. Stock holds the softmax of the whole ``[..., I, J, bins]`` plane (fp32) and its expectation products; here one
    block. ``bitwise`` by construction (the softmax and the expectation are per pair; an fp32 softmax over bins is not a GEMM).
    """
    torch = _torch()
    logits = _tensor(lever, "logits", logits)
    fn = _callable(lever, "statement", statement)
    if logits.dim() < 3:
        raise ChunkRefusal(lever, "not_pair_shaped", f"logits has rank {logits.dim()}, expected [..., I, J, bins]")
    d = _axis(lever, logits, row_dim, "row_dim")
    chunk = _chunk(lever, chunk)
    n = logits.shape[d]
    t0 = time.perf_counter()
    outs: Dict[str, Any] = {}
    keys: Optional[Tuple[str, ...]] = None
    i0 = 0
    n_chunks = 0
    while i0 < n:
        i1 = min(n, i0 + chunk)
        block = logits if (i0 == 0 and i1 == n) else logits.narrow(d, i0, i1 - i0)
        res = fn(block)
        if not isinstance(res, Mapping) or not res:
            raise ChunkRefusal(lever, "statement_output", f"statement returned {type(res).__name__} for rows [{i0}, {i1}), expected a non-empty mapping of tensors")
        if keys is None:
            keys = tuple(res.keys())
        elif tuple(res.keys()) != keys:
            raise ChunkRefusal(lever, "statement_keys", f"statement returned keys {tuple(res.keys())} for rows [{i0}, {i1}), expected {keys}")
        for k in keys:
            v = res[k]
            if not torch.is_tensor(v) or v.dim() <= d or v.shape[d] != i1 - i0:
                raise ChunkRefusal(lever, "statement_output_rows", f"output {k!r} is {tuple(v.shape) if torch.is_tensor(v) else type(v).__name__} for rows [{i0}, {i1}): axis {d} must carry the block's rows")
            if i0 == 0 and i1 == n:
                outs[k] = v
                continue
            if k not in outs:
                shape = list(v.shape)
                shape[d] = n
                outs[k] = torch.empty(shape, dtype=v.dtype, device=v.device)
            if v.dtype != outs[k].dtype:
                raise ChunkRefusal(lever, "statement_output_dtype", f"output {k!r} is {v.dtype} for rows [{i0}, {i1}), the assembled tensor is {outs[k].dtype}")
            outs[k].narrow(d, i0, i1 - i0).copy_(v)
        del res
        n_chunks += 1
        i0 = i1
    _emit(record, lever, "confidence_head", "bitwise",
          "per-row statement (softmax over bins, expectations, per-row terms) on row blocks; across-row reductions stay outside",
          decision="chunked", dim=d, chunk=chunk, n_chunks=n_chunks, rows=n, shape=tuple(logits.shape), dtype=_dtype(logits),
          outputs=tuple(keys or ()), wall_s=round(time.perf_counter() - t0, 6))
    return outs


# ------------------------------------------------------------------------------------------------------------ MSA rows


def msa_rows_chunked(fn: Callable, m: "torch.Tensor", *, chunk: int, exact: str, reason: str, record: Record = None, row_dim: int = -3,
                     out: Optional["torch.Tensor"] = None, accumulate: bool = False, with_offsets: bool = False, lever: str = "chunk_msa_rows") -> "torch.Tensor":
    """``fn`` on blocks of ``chunk`` MSA rows of ``m [..., S, N, C]`` (``row_dim`` = the S axis), assembled along S. With
    ``with_offsets=True`` ``fn(block, i0, i1)`` receives the block's row range — a masked MSA statement slices its mask (and any
    other per-row operand) to ``[i0, i1)`` itself.

    A row-local MSA statement (an MSA transition; a pair-weighted average, whose weights come from the pair and are shared by every
    row) is ``bitwise``; an outer-product mean whose sum over S is split across blocks is ``band`` — chunk it over its OUTPUT rows
    (:func:`chunk_rows` on the pair axis) to keep it ``bitwise``. The call site declares which with its reason.
    """
    m = _tensor(lever, "m", m)
    if m.dim() < 3:
        raise ChunkRefusal(lever, "not_msa_shaped", f"m has rank {m.dim()}, expected [..., S, N, C]")
    d = _axis(lever, m, row_dim, "row_dim")
    return chunk_rows(fn, m, d, chunk, exact=exact, reason=reason, record=record, out=out, accumulate=accumulate, with_offsets=with_offsets, lever=lever, site="msa_rows")


# ------------------------------------------------------------------------------------------------------------ policy


def parse_setting(value: Any, lever: str = "chunk") -> Any:
    """The XL-policy setting grammar: ``0``/``off``/``""``/``None`` → ``"off"``; ``always`` → ``"always"`` (engage at every size);
    ``auto`` → ``"auto"``; a positive int → that token count (engage above it)."""
    if value is None:
        return "off"
    s = str(value).strip().lower()
    if s in ("", "0", "off", "none", "false"):
        return "off"
    if s in ("auto", "always"):
        return s
    try:
        n = int(s)
    except ValueError:
        raise ChunkRefusal(lever, "bad_setting", f"setting {value!r}: expected off|always|auto|<tokens>") from None
    if n <= 0:
        return "off"
    return n


@dataclasses.dataclass(frozen=True)
class AutoCalibration:
    """The quadratic peak model of the XL policy: ``peak_gb(N) = base_gb + full_gb_at_ref * (N / ref_ntok)**2`` — the adapter passes
    its engine's audited values (the core carries none). ``threshold(total_gb)`` = the largest token count, a multiple of ``multiple``,
    at which the un-levered path stays under ``headroom * total_gb``, raised to ``floor`` and clamped to ``clamp`` when given (the
    policy's 1024 floor and its per-card cap)."""
    base_gb: float
    full_gb_at_ref: float
    ref_ntok: int
    headroom: float = 0.85
    multiple: int = 128
    clamp: Optional[int] = None
    floor: Optional[int] = None

    def __post_init__(self):
        if self.full_gb_at_ref <= 0 or self.ref_ntok <= 0 or not 0 < self.headroom <= 1 or self.multiple < 1:
            raise ChunkRefusal("chunk", "bad_calibration", f"{self}")
        if self.floor is not None and self.clamp is not None and self.floor > self.clamp:
            raise ChunkRefusal("chunk", "bad_calibration", f"floor {self.floor} > clamp {self.clamp}")

    def peak_gb(self, n_tokens: int) -> float:
        return self.base_gb + self.full_gb_at_ref * (n_tokens / self.ref_ntok) ** 2

    def threshold(self, total_gb: float) -> int:
        if not isinstance(total_gb, (int, float)) or isinstance(total_gb, bool) or total_gb <= 0:
            raise ChunkRefusal("chunk", "no_device_total", f"auto needs the device total in GB, got {total_gb!r}")
        budget = self.headroom * total_gb - self.base_gb
        n = self.ref_ntok * math.sqrt(budget / self.full_gb_at_ref) if budget > 0 else 0.0
        n = int(n // self.multiple) * self.multiple
        if self.floor is not None:
            n = max(n, int(self.floor))
        if self.clamp is not None:
            n = min(n, int(self.clamp))
        return n


@dataclasses.dataclass(frozen=True)
class ChunkDecision:
    """What a policy decided for one call: ``engaged`` (chunk or not), ``rows`` (the block size when engaged), ``threshold`` (the token
    count above which the lever engages, ``None`` for off), ``source`` (``explicit`` | ``off`` | ``always`` | ``auto`` | ``table``),
    ``n_tokens``, ``detail``."""
    engaged: bool
    rows: Optional[int]
    threshold: Optional[int]
    source: str
    n_tokens: int
    detail: str

    def as_record(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ChunkPolicy:
    """The XL-policy shape for a chunk lever: ``setting`` = ``off`` | ``always`` | ``auto`` | ``<tokens>`` (the lever engages ABOVE
    that token count; :func:`parse_setting`), ``rows`` = the block size, or ``table`` = ``((min_tokens, rows), ...)`` ascending — the
    last row whose ``min_tokens`` is below the call's token count applies (an engine's own stepped chunk sizes above its thresholds);
    ``calibration`` = the :class:`AutoCalibration` ``auto`` needs (refused by name when absent). ``decide(n_tokens, total_gb)`` returns
    a :class:`ChunkDecision`; the adapter records it (``as_record()``) beside the lever's flag — a threshold is never hidden."""
    lever: str
    setting: Any = "off"
    rows: Optional[int] = None
    table: Tuple[Tuple[int, int], ...] = ()
    calibration: Optional[AutoCalibration] = None

    def __post_init__(self):
        object.__setattr__(self, "setting", parse_setting(self.setting, self.lever))
        if self.rows is not None:
            _chunk(self.lever, self.rows)
        last = -1
        table = tuple(tuple(r) for r in self.table)
        for row in table:
            if not (len(row) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in row)) or row[0] < 0 or row[1] < 1 or row[0] <= last:
                raise ChunkRefusal(self.lever, "bad_table", f"table {self.table!r}: expected ascending (min_tokens, rows) pairs with rows >= 1")
            last = row[0]
        object.__setattr__(self, "table", table)
        if self.rows is None and not table:
            raise ChunkRefusal(self.lever, "no_rows", f"policy {self.lever!r}: neither rows nor a table")
        if self.calibration is not None and not isinstance(self.calibration, AutoCalibration):
            raise ChunkRefusal(self.lever, "bad_calibration", f"calibration is {type(self.calibration).__name__}, expected AutoCalibration")

    def threshold(self, total_gb: Optional[float] = None) -> Tuple[Optional[int], str]:
        if self.setting == "off":
            return None, "off"
        if self.setting == "always":
            return 0, "always"
        if self.setting == "auto":
            if self.calibration is None:
                raise ChunkRefusal(self.lever, "no_calibration", f"policy {self.lever!r}: setting auto without a calibration")
            return self.calibration.threshold(total_gb), "auto"
        return int(self.setting), "explicit"

    def rows_for(self, n_tokens: int) -> Optional[int]:
        rows = self.rows
        for min_tokens, r in self.table:
            if n_tokens > min_tokens:
                rows = r
        return rows

    def decide(self, n_tokens: int, total_gb: Optional[float] = None) -> ChunkDecision:
        if not isinstance(n_tokens, int) or isinstance(n_tokens, bool) or n_tokens < 0:
            raise ChunkRefusal(self.lever, "bad_tokens", f"n_tokens={n_tokens!r}")
        thr, source = self.threshold(total_gb)
        if thr is None:
            return ChunkDecision(False, None, None, "off", n_tokens, f"{self.lever}: off")
        rows = self.rows_for(n_tokens)
        if n_tokens <= thr:
            return ChunkDecision(False, None, thr, source, n_tokens, f"{self.lever}: {n_tokens} tokens <= threshold {thr} ({source})")
        if rows is None:
            return ChunkDecision(False, None, thr, source, n_tokens, f"{self.lever}: {n_tokens} tokens > threshold {thr} ({source}) but the table has no rows for this size")
        src = "table" if (self.table and rows != self.rows) else source
        return ChunkDecision(True, rows, thr, src, n_tokens, f"{self.lever}: {n_tokens} tokens > threshold {thr} ({source}); rows={rows}")

    def describe(self) -> dict:
        return {"lever": self.lever, "setting": self.setting, "rows": self.rows, "table": list(self.table),
                "calibration": None if self.calibration is None else dataclasses.asdict(self.calibration)}


# ------------------------------------------------------------------------------------------------------------ the levers


def _default_call(module, args, kwargs):
    """The default argument adapter: ``x`` = the first positional (else ``kwargs['x']``), ``mask`` = the second (else ``kwargs.get('mask')``);
    ``rebind(t)`` returns the call's ``(args, kwargs)`` with ``x`` replaced by ``t`` (the stock statement on a block)."""
    if args:
        x = args[0]

        def rebind(t):
            return (t, *args[1:]), dict(kwargs)
    elif "x" in kwargs:
        x = kwargs["x"]

        def rebind(t):
            k = dict(kwargs)
            k["x"] = t
            return args, k
    else:
        return None, None, None
    mask = args[1] if len(args) > 1 else kwargs.get("mask")
    return x, mask, rebind


def _takes_offsets(rebind) -> bool:
    """Whether a ``rebind`` accepts ``(t, i0, i1)``: three or more positional parameters, or a var-positional one."""
    try:
        params = list(inspect.signature(rebind).parameters.values())
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        return True
    positional = [p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    return len(positional) >= 3


def _default_site(module, x, args, kwargs):
    """The default site test: a tensor of rank >= 3 with no engine-side ``chunk_size`` set (stock's own chunked path is left alone, by name)."""
    torch = _torch()
    if not torch.is_tensor(x) or x.dim() < 3:
        return "input is not a rank>=3 tensor"
    if kwargs.get("chunk_size") is not None:
        return "stock chunk_size set on the call"
    return True


def _hook(ctx, lever: str, key: str, default=None, *, required: bool = False, is_callable: bool = False):
    if required:
        v = ctx.require(lever, key)[key]
    else:
        v = ctx.hook(lever, key, default)
    if is_callable and v is not None:
        _callable(lever, f"hooks.{key}", v)
    return v


def _policy(ctx, lever: str, rows: int) -> ChunkPolicy:
    tok = ctx.setting(lever, "tok", "always")
    if parse_setting(tok, lever) == "off":
        raise ChunkRefusal(lever, "setting.tok", f"tok={tok!r} is off: a lever of the line cannot be switched off by a setting (leave it out by name: {off_ref(lever)})")
    table = _hook(ctx, lever, "table", ())
    calibration = _hook(ctx, lever, "calibration")
    if parse_setting(tok, lever) == "auto" and calibration is None:
        raise ChunkRefusal(lever, "no_calibration", f"tok={tok!r} needs hooks.calibration (an AutoCalibration of the engine's audited values)")
    return ChunkPolicy(lever, tok, rows=rows, table=tuple(tuple(r) for r in table) if table else (), calibration=calibration)


def _bind(ctx, lever: str, required_hooks: Sequence[str], defaults: Mapping[str, Any]) -> dict:
    """The lever's preconditions, checked by name: the class and its attribute, the callables, the settings, the policy."""
    cls = _hook(ctx, lever, "cls", required=True)
    attr = _hook(ctx, lever, "attr", "forward")
    if not isinstance(attr, str) or not hasattr(cls, attr) or not callable(getattr(cls, attr)):
        raise ChunkRefusal(lever, "hooks.attr", f"{getattr(cls, '__name__', cls)!r} has no callable attribute {attr!r}")
    stock = getattr(cls, attr)
    if getattr(stock, "big_lever", None) is not None:
        raise ChunkRefusal(lever, "hooks.attr", f"{getattr(cls, '__name__', cls)!r}.{attr} is already patched by lever {stock.big_lever!r} (one lever per site; a subclass inherits its parent's patch)")
    hooks = {"cls": cls, "attr": attr, "stock": stock}
    for k in required_hooks:
        hooks[k] = _hook(ctx, lever, k, required=True, is_callable=True)
    hooks["call"] = _hook(ctx, lever, "call", _default_call, is_callable=True)
    hooks["is_site"] = _hook(ctx, lever, "is_site", _default_site, is_callable=True)
    rows = ctx.setting(lever, "rows", defaults.get("rows", 256), cast=int)
    _chunk(lever, rows)
    settings = {"rows": rows, "tok": ctx.setting(lever, "tok", "always")}
    if "fp32" in defaults:
        settings["fp32"] = _fp32_policy(lever, ctx.setting(lever, "fp32", defaults["fp32"]))
    if "mode" in defaults:
        mode = ctx.setting(lever, "mode", defaults["mode"])
        if mode not in ("rows", "channels"):
            raise ChunkRefusal(lever, "setting.mode", f"mode={mode!r}, expected rows|channels")
        hidden = ctx.setting(lever, "hidden", None, cast=int)
        channels = ctx.setting(lever, "channels", defaults.get("channels", 32), cast=int)
        if mode == "channels" and (not isinstance(hidden, int) or isinstance(hidden, bool) or hidden < 1):
            raise ChunkRefusal(lever, "setting.hidden", f"mode=channels needs hidden (the hidden channel count), got {hidden!r}")
        _chunk(lever, channels)
        settings.update(mode=mode, hidden=hidden, channels=channels)
    hooks["policy"] = _policy(ctx, lever, rows)
    hooks["settings"] = settings
    return hooks


def _event(rec, kind: str, lever: str, detail: str) -> None:
    """The lever's census event on the record (``mark`` / ``skip`` / ``fallback``); a record without the method gets a note; no record, no event."""
    if rec is None:
        return
    fn = getattr(rec, kind, None)
    if fn is None:
        note = getattr(rec, "note", None)
        if note is not None:
            note(f"{lever}: {kind}: {detail}")
        return
    if kind == "mark":
        fn(lever, detail=detail)
    else:
        fn(lever, detail)


def _device_total_gb(x) -> Optional[float]:
    """The total memory (GB) of the CUDA device ``x`` lives on, through :func:`opt_core.arch.device_memory` (the package's one reader of a
    card's total / free bytes); None for a tensor off the GPU or a total the reader cannot state (the policy then refuses ``no_device_total``)."""
    if x.device.type != "cuda":
        return None
    from .. import arch                                          # standard library at import; torch is reached through sys.modules only

    total = arch.device_memory(x.device.index if x.device.index is not None else 0)["total_bytes"]
    return None if total is None else total / 1e9


def _install(ctx, lever: str, bound: dict, exact: str, exact_reason: str, run: Callable) -> Applied:
    """Patch ``cls.attr`` with the lever's wrapper: per call the site test, the policy decision and the record's events (``skip`` /
    ``mark`` / ``fallback``); ``run(module, x, mask, rebind, rows, sink) -> result`` is the lever's chunked statement (its per-call entries name the decision).
    The Applied's label is the registration's (the ceiling of its calls; the ``_band`` twin of an fp32-capable lever carries band)."""
    cls, attr, stock, call, is_site, policy = (bound[k] for k in ("cls", "attr", "stock", "call", "is_site", "policy"))
    rec = ctx.record
    applied = Applied(lever=lever, settings=dict(bound["settings"]), sites=(f"{getattr(cls, '__module__', '?')}.{getattr(cls, '__qualname__', cls)}.{attr}",),
                      exact=exact, exact_reason=exact_reason, notes=[f"policy={policy.describe()}"])

    class _Sink:                                                  # the record's per-call sink (ONE place); keeps the last decision for the census event
        last_decision = "chunked"

        def call(self, lv, site, ex, reason, **details):
            self.last_decision = details.get("decision", "chunked")
            if ex == "band" and applied.exact == "bitwise":     # cannot happen through the levers (the fp32 setting is bound at attach); refused, never widened silently
                raise ChunkRefusal(lv, "label_ceiling", f"a call at {site} labelled band under the bitwise lever {lv}: bind the _band lever")
            return _emit(rec, lv, site, ex, reason, **details)

    sink = _Sink()

    def wrapper(self, *args, **kwargs):
        x, mask, rebind = call(self, args, kwargs)
        if x is None:
            _event(rec, "skip", lever, "no input tensor found by the argument adapter")
            return stock(self, *args, **kwargs)
        site = is_site(self, x, args, kwargs)
        if site is not True:
            _event(rec, "skip", lever, str(site))
            return stock(self, *args, **kwargs)
        n_tokens = int(x.shape[-2])
        decision = policy.decide(n_tokens, _device_total_gb(x) if policy.setting == "auto" else None)
        if not decision.engaged:
            _event(rec, "skip", lever, decision.detail)
            return stock(self, *args, **kwargs)
        sink.last_decision = "chunked"
        result = run(self, x, mask, rebind, decision.rows, sink)
        if not hasattr(rec, "call"):                             # a record with the per-call sink marks / falls back inside call(); without it the wrapper does
            _event(rec, "fallback" if sink.last_decision == "passthrough" else "mark", lever,
                   f"fp32 site run un-chunked (fp32=passthrough) at {n_tokens} tokens" if sink.last_decision == "passthrough" else decision.detail)
        return result

    wrapper.__name__ = getattr(stock, "__name__", attr)
    wrapper.__doc__ = getattr(stock, "__doc__", None)
    wrapper.__wrapped__ = stock
    wrapper.big_lever = lever
    own = attr in vars(cls)                                      # an inherited attribute is restored by removing the override, not by re-setting the parent's function
    setattr(cls, attr, wrapper)

    def undo():
        if own:
            setattr(cls, attr, stock)
        else:
            delattr(cls, attr)

    applied.undo = undo
    return applied


def _applies(bind: Callable) -> Callable:
    def applies(ctx) -> Optional[Refusal]:
        try:
            bind(ctx)
        except RefusalError as e:
            return e.refusal
        return None

    return applies


# the fp32-capable mechanisms come as a lever PAIR: ``<name>`` registered bit-exact (fp32 sites pass through as a recorded fallback or
# refuse — never chunked under band) and ``<name>_band`` registered band (fp32 sites chunked under the band label; bf16 / autocast
# calls still carry bit-exact per call). A registered label is the ceiling of the lever's calls and is narrowed only by an equality
# record (the registry's rule), so the tier of a big line reads off its lever names.

def _fp32_of(lever: str, settings: MutableMapping[str, Any], default: str) -> str:
    v = settings.get("fp32", default)
    if lever.endswith("_band"):
        if v not in (None, "band"):
            raise ChunkRefusal(lever, "setting.fp32", f"{lever} chunks fp32 sites under band by definition; fp32={v!r} is a setting of {lever[:-5]}")
        settings["fp32"] = "band"
        return "band"
    if v == "band":
        raise ChunkRefusal(lever, "setting.fp32", f"fp32=band is the lever {lever}_band (registered band); {lever} is registered bitwise: fp32=passthrough|refuse")
    return v


# chunk_pair_transition / chunk_pair_transition_band --------------------------------------------------------------------------------------------

def _bind_trans(ctx, lever: str = "chunk_pair_transition"):
    b = _bind(ctx, lever, (), {"rows": 256, "fp32": ("band" if lever.endswith("_band") else "passthrough")})
    _fp32_of(lever, b["settings"], "passthrough")
    return b


def _apply_trans(ctx, lever: str) -> Applied:
    b = _bind_trans(ctx, lever)
    stock, fp32 = b["stock"], b["settings"]["fp32"]

    def run(module, x, mask, rebind, rows, sink):
        def forward(t):
            a, k = rebind(t)
            return stock(module, *a, **k)
        return pair_transition_chunked(x, forward, chunk=rows, fp32=fp32, record=sink, lever=lever)

    lv = LEVERS[lever]
    return _install(ctx, lever, b, lv.exact, lv.exact_reason, run)


@register("chunk_pair_transition", family="chunk", exact="bitwise",
          exact_reason="row-local pair transition on row blocks (bf16 / autocast sites: independent rows re-ordered, no reduction split); an fp32 site is a named decision, passthrough (a recorded fallback) | refuse — never chunked under band (that is chunk_pair_transition_band)",
          applies=_applies(_bind_trans), description="row-chunked pair transition", preconditions=("hooks.cls", "hooks.attr", "setting.rows", "setting.tok", "setting.fp32"),
          settings=("rows", "tok", "fp32"))
def chunk_pair_transition(ctx) -> Applied:
    return _apply_trans(ctx, "chunk_pair_transition")


@register("chunk_pair_transition_band", family="chunk", exact="band",
          exact_reason="chunk_pair_transition with fp32 sites chunked under the band label (the pair-transition kit measured 9/12 fp32 sites non-bitwise under row chunking); bf16 / autocast calls carry bitwise per call",
          applies=_applies(lambda ctx: _bind_trans(ctx, "chunk_pair_transition_band")), description="row-chunked pair transition, fp32 sites under band",
          preconditions=("hooks.cls", "hooks.attr", "setting.rows", "setting.tok"), settings=("rows", "tok", "fp32"))
def chunk_pair_transition_band(ctx) -> Applied:
    return _apply_trans(ctx, "chunk_pair_transition_band")


# chunk_triangle_attention ---------------------------------------------------------------------------------------------------------------------

def _bind_triatt(ctx):
    return _bind(ctx, "chunk_triangle_attention", ("parts", "starting"), {"rows": 256})


@register("chunk_triangle_attention", family="chunk", exact="bitwise",
          exact_reason="two-pass triangle attention: the full bias from row blocks, then the per-query-row statement with the full bias (the softmax over the key axis is whole)",
          applies=_applies(_bind_triatt), description="two-pass query-row-chunked triangle attention", preconditions=("hooks.cls", "hooks.attr", "hooks.parts", "hooks.starting", "setting.rows", "setting.tok"),
          settings=("rows", "tok"))
def chunk_triangle_attention(ctx) -> Applied:
    b = _bind_triatt(ctx)
    parts, starting = b["parts"], b["starting"]

    def run(module, x, mask, rebind, rows, sink):
        return triangle_attention_chunked(x, mask, parts(module), chunk=rows, starting=bool(starting(module)), record=sink, lever="chunk_triangle_attention")

    return _install(ctx, "chunk_triangle_attention", b, chunk_triangle_attention.lever.exact, chunk_triangle_attention.lever.exact_reason, run)


# chunk_triangle_multiplication / chunk_triangle_multiplication_band ---------------------------------------------------------------------------------

def _bind_trimul(ctx, lever: str = "chunk_triangle_multiplication"):
    b = _bind(ctx, lever, ("parts",), {"rows": 256, "fp32": ("band" if lever.endswith("_band") else "refuse"), "mode": "rows", "channels": 32})
    _fp32_of(lever, b["settings"], "refuse")
    return b


def _apply_trimul(ctx, lever: str) -> Applied:
    b = _bind_trimul(ctx, lever)
    parts, s = b["parts"], b["settings"]

    def run(module, x, mask, rebind, rows, sink):
        if s["mode"] == "channels":
            return triangle_multiplication_chunked(x, mask, parts(module), chunk=s["channels"], mode="channels", hidden=s["hidden"], row_block=rows,
                                                   fp32=s["fp32"], record=sink, lever=lever)
        return triangle_multiplication_chunked(x, mask, parts(module), chunk=rows, mode="rows", fp32=s["fp32"], record=sink, lever=lever)

    lv = LEVERS[lever]
    return _install(ctx, lever, b, lv.exact, lv.exact_reason, run)


@register("chunk_triangle_multiplication", family="chunk", exact="bitwise",
          exact_reason="triangle multiplication per output-row block (rows) or per channel block (channels) with the reduction over the shared axis whole (bf16 / autocast products); an fp32 product is a named decision, refuse (default) | passthrough (a recorded fallback) — never chunked under band (that is chunk_triangle_multiplication_band)",
          applies=_applies(_bind_trimul), description="chunked triangle multiplication (rows | channels)", preconditions=("hooks.cls", "hooks.attr", "hooks.parts", "setting.rows", "setting.tok", "setting.mode", "setting.hidden", "setting.channels", "setting.fp32"),
          settings=("rows", "tok", "mode", "hidden", "channels", "fp32"))
def chunk_triangle_multiplication(ctx) -> Applied:
    return _apply_trimul(ctx, "chunk_triangle_multiplication")


@register("chunk_triangle_multiplication_band", family="chunk", exact="band",
          exact_reason="chunk_triangle_multiplication with fp32 products chunked under the band label (the chunk changes the fp32 GEMM's extent: the pair-transition kit's non-bitwise class; measured torch.equal on the H100 cu130 stack at 2000 / 4000 — an identity record narrows it); bf16 / autocast products carry bitwise per call",
          applies=_applies(lambda ctx: _bind_trimul(ctx, "chunk_triangle_multiplication_band")), description="chunked triangle multiplication, fp32 products under band",
          preconditions=("hooks.cls", "hooks.attr", "hooks.parts", "setting.rows", "setting.tok", "setting.mode", "setting.hidden", "setting.channels"), settings=("rows", "tok", "mode", "hidden", "channels", "fp32"))
def chunk_triangle_multiplication_band(ctx) -> Applied:
    return _apply_trimul(ctx, "chunk_triangle_multiplication_band")


# chunk_confidence_head -----------------------------------------------------------------------------------------------------------------------

def _bind_conf(ctx):
    return _bind(ctx, "chunk_confidence_head", ("logits", "statement", "finish"), {"rows": 256})


@register("chunk_confidence_head", family="chunk", exact="bitwise",
          exact_reason="the per-row confidence statement (softmax over bins, expectations, per-row terms) on row blocks of the pair logits; across-row reductions stay in the head's finishing statements",
          applies=_applies(_bind_conf), description="row-chunked confidence scoring", preconditions=("hooks.cls", "hooks.attr", "hooks.logits", "hooks.statement", "hooks.finish", "setting.rows", "setting.tok"),
          settings=("rows", "tok"))
def chunk_confidence_head(ctx) -> Applied:
    b = _bind_conf(ctx)
    logits_of, statement, finish = b["logits"], b["statement"], b["finish"]

    def call(module, args, kwargs):
        x = logits_of(module, args, kwargs)
        return x, None, (lambda t: (args, kwargs))

    b["call"] = call

    def run(module, logits, mask, rebind, rows, sink):
        return finish(module, confidence_head_chunked(logits, statement(module), chunk=rows, record=sink, lever="chunk_confidence_head"), *rebind(logits))

    return _install(ctx, "chunk_confidence_head", b, chunk_confidence_head.lever.exact, chunk_confidence_head.lever.exact_reason, run)


# chunk_msa_rows ------------------------------------------------------------------------------------------------------------------------

def _bind_msa(ctx):
    b = _bind(ctx, "chunk_msa_rows", (), {"rows": 256})
    b["exact"] = _label("chunk_msa_rows", _hook(ctx, "chunk_msa_rows", "exact", "bitwise"))
    b["reason"] = _hook(ctx, "chunk_msa_rows", "reason", "row-local MSA statement on blocks of MSA rows")
    if not isinstance(b["reason"], str) or not b["reason"].strip():
        raise ChunkRefusal("chunk_msa_rows", "hooks.reason", "the exactness declaration needs a non-empty reason")
    return b


@register("chunk_msa_rows", family="chunk", exact="bitwise",
          exact_reason="a row-local MSA statement on blocks of MSA rows (the adapter declares the label per statement; an OPM chunked over S would be band and is chunked over its output rows instead)",
          applies=_applies(_bind_msa), description="MSA-row-chunked statement", preconditions=("hooks.cls", "hooks.attr", "hooks.exact", "setting.rows", "setting.tok"),
          settings=("rows", "tok"))
def chunk_msa_rows(ctx) -> Applied:
    """The ``call`` hook's ``rebind`` may take ``(t, i0, i1)`` (the block and its row range: a masked MSA statement re-binds its mask to
    the same rows) or ``(t)``; the arity is read once at attach."""
    b = _bind_msa(ctx)
    stock, exact, reason = b["stock"], b["exact"], b["reason"]

    def run(module, m, mask, rebind, rows, sink):
        offsets = _takes_offsets(rebind)

        def fn(t, i0, i1):
            a, k = rebind(t, i0, i1) if offsets else rebind(t)
            return stock(module, *a, **k)
        return msa_rows_chunked(fn, m, chunk=rows, exact=exact, reason=reason, record=sink, with_offsets=True, lever="chunk_msa_rows")

    return _install(ctx, "chunk_msa_rows", b, chunk_msa_rows.lever.exact, chunk_msa_rows.lever.exact_reason, run)
