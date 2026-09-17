"""The evidence of the row-sharded pair stack: the ACTIVE/EXIT tokens, the ONE lever line, per-device peak memory, the per-site plan, the card declaration.

The ``n_gpu=P sharding=rowpair`` tokens the kit's ACTIVE / EXIT lines and the checking rows carry (``n_gpu=1 sharding=none`` at P=1) come
from :mod:`opt_core.mem.ngpu` — the ONE producer; :func:`active_fields` / :func:`active_text` here are its pairs / text forms
under this package's default scheme. :func:`line` renders the family's ``LEVER`` line through
:func:`opt_core.report.lever_line` (``impl=opt_core.mem.rowpair_jax@<core version> origin=core``). :func:`device_peaks` reads
``memory_stats()`` per device of the mesh AFTER the run (``peak_bytes_in_use`` — the per-card peak of the memory ladder; ``bytes_limit`` beside it;
a backend without stats reads ``unavailable``; scope :data:`PEAK_SCOPE` — the XLA allocator, not the device) and :func:`peak_fields` folds them into
blank-free tokens (``xla_peak_bytes=d0:<n>,d1:<n> xla_peak_gb_max=<x> peak_scope=xla_allocator``).
The XLA client-memory environment as found comes from :func:`mesh.xla_memory_env` — a peak without ``prealloc`` / ``mem_fraction`` beside it is not a
measurement. :data:`SITES` is the per-site plan as data (site → locality, collectives, numerics class) that the API document and the tests read;
:data:`CLASSES` names the numerics classes. :data:`ARCH` is the family's card declaration until the core's arch registry is the producer (the
mechanism is XLA collectives + jnp: any device kind the kit's jax build drives; per-shard fused kernels carry their own sm gates). Standard library
at import.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .. import ngpu as _ngpu
from . import LEVER, SHARDING, _lazy

IMPL = "opt_core.mem.rowpair_jax"
CLASS_MOVES_BYTES, CLASS_ROW_LOCAL, CLASS_COMPLETE, CLASS_REORDERED = "moves_bytes", "row_local", "complete_contraction", "reordered"
CLASSES = {
    CLASS_MOVES_BYTES: "collectives, constraints, re-layouts: values are moved between devices, never combined — exact",
    CLASS_ROW_LOCAL: "the same per-element arithmetic on a subset of rows; library code generation (GEMM blocking, LayerNorm fusion) depends on the row extent: a stated tolerance class — bitwise is REPORTED per stack, never asserted",
    CLASS_COMPLETE: "every output element's reduction has the dense length over the dense operands; the GEMM shape differs (M=N/P), so the accumulation order inside the library kernel may differ: stated tolerance (tier-2)",
    CLASS_REORDERED: "cross-device psum: P partial sums added in the collective's order: stated tolerance",
}
# site -> (locality, collectives per call, numerics class); the plan every adapter follows (names are generic sub-layer roles, not engine classes)
SITES: Dict[str, Tuple[str, str, str]] = {
    "trimul_outgoing": ("row block; projections/gates row-local", "all_gather(partner operand) [gather] | P x ppermute(block) [ring]", CLASS_COMPLETE),
    "trimul_incoming": ("row block", "all_gather(a) + all_to_all(b) [gather] | 2 x all_to_all + P x ppermute [ring]", CLASS_COMPLETE),
    "triatt_starting": ("row-local given the gathered bias", "all_gather(bias [N/P,N,H])", CLASS_ROW_LOCAL),
    "triatt_ending": ("row block of x^T", "all_to_all in + all_to_all out + all_gather(bias)", CLASS_ROW_LOCAL),
    "pair_transition": ("row-local", "none", CLASS_ROW_LOCAL),
    "outer_product_mean": ("output rows local; left operand sliced to my rows; MSA replicated", "none", CLASS_ROW_LOCAL),
    "msa_pair_bias": ("projection row-local", "all_gather(logits [N/P,N,H])", CLASS_MOVES_BYTES),
    "single_pair_logits": ("projection row-local", "all_gather(logits [N/P,N,H])", CLASS_MOVES_BYTES),
    "prologue_pair_embeddings": ("row-local under the row constraint", "none", CLASS_ROW_LOCAL),
    "template_pair_stack": ("the same sub-layers on [N,N,c_t]: as above", "as the trunk sub-layers", CLASS_COMPLETE),
    "recycle_carry": ("pair leaves constrained to rows at the loop boundary", "none", CLASS_MOVES_BYTES),
    "heads_symmetrize": ("left + left^T per row block", "all_to_all", CLASS_MOVES_BYTES),
    "heads_masked_mean": ("scalar over the whole map", "2 x psum", CLASS_REORDERED),
    "heads_full_maps": ("[N,N] outputs assembled for the writer", "all_gather", CLASS_MOVES_BYTES),
    "confidence_pairformer": ("the trunk's sub-layers on the local block", "as the trunk sub-layers", CLASS_COMPLETE),
    "diffusion_pair_conditioning": ("row block (query rows local; keys over all tokens)", "all_gather(attention output rows) per block", CLASS_ROW_LOCAL),
}
TP_LEVER_ID = "F7.tensor_parallel"          # the canonical strategy id of the row-sharded pair stack; declared ONCE in the core's arch
                                            # registry by opt_core.mem.ngpu's owner — this package READS the registry, never declares the shared id


SCHEMES = _ngpu.SCHEMES                       # the sharding schemes the ACTIVE / EXIT token names (opt_core.mem.ngpu is the one producer)


def card_fields(rmesh: Any, lever_id: str = TP_LEVER_ID, strict: bool = False) -> Dict[str, Any]:
    """The CARD column's words for the mesh's first device, read from the core's arch registry (:func:`opt_core.arch.lever_state` — ``undeclared`` /
    ``uncertified`` cards RUN, named ``card_support=<word>``; an excluded card reads ``card_state=off card_reason=unsupported_card:<sm>``; ``strict`` turns
    untested into off): ``{"card_state": …, "card_support": …[, "card_reason": …]}`` for the lever line. The sm itself is already on the line (``sm=``)."""
    from ... import arch  # noqa: PLC0415
    sm = rmesh.facts[0].get("sm") if getattr(rmesh, "facts", None) else None
    st = arch.lever_state(lever_id, sm, strict=strict)
    out: Dict[str, Any] = {"card_state": st["state"]}
    out.update({k: v for k, v in (st.get("evidence") or {}).items() if k != "sm"})
    if st.get("reason"):
        out["card_reason"] = st["reason"]
    return out

def active_fields(n_gpu: int, scheme: str = SHARDING) -> List[Tuple[str, Any]]:
    """``[("n_gpu", P), ("sharding", <scheme> | "none")]`` — :func:`opt_core.mem.ngpu.active_pairs` (the ONE producer of the token text the
    kit's lines carry; this is the pairs form the lever line composes). P=1 → ``sharding=none``."""
    return _ngpu.active_pairs(n_gpu, scheme)


def active_text(n_gpu: int, scheme: str = SHARDING) -> str:
    """``n_gpu=P sharding=<scheme>`` as text — :func:`opt_core.mem.ngpu.active_fields`."""
    return _ngpu.active_fields(n_gpu, scheme)


def impl_label() -> str:
    """``opt_core.mem.rowpair_jax@<core version>`` — the ``impl=`` value of the lever line."""
    from ... import __version__  # noqa: PLC0415
    return f"{IMPL}@{__version__}"


PEAK_SCOPE = "xla_allocator"                  # what the peak numbers cover: the XLA client's allocator pool in-use high-water — NOT NCCL communicator
                                              # buffers, the CUDA context, or any non-XLA allocation in the process (torch in a hybrid process); a memory
                                              # ladder's 'peak per card' is the device high-water read outside the process (nvidia-smi sampling), not this


def device_peaks(rmesh_or_devices: Any) -> List[Dict[str, Any]]:
    """Per device of the mesh (or of a device list): ``{id, kind, peak_bytes_in_use, bytes_in_use, bytes_limit, total_bytes}`` from ``memory_stats()`` NOW —
    call after the run. Scope :data:`PEAK_SCOPE`: the XLA allocator's in-use high-water since process start (preallocation does not count as in-use;
    NCCL / CUDA-context / non-XLA memory is outside it). A backend without stats gives ``None`` values (rendered ``unavailable``)."""
    from .mesh import device_facts  # noqa: PLC0415
    devs = getattr(rmesh_or_devices, "devices", rmesh_or_devices)
    return [{k: f[k] for k in ("id", "kind", "peak_bytes_in_use", "bytes_in_use", "bytes_limit", "total_bytes")} for f in device_facts(list(devs))]


def peak_fields(peaks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Blank-free lever-line tokens from :func:`device_peaks`, named by scope: ``xla_peak_bytes=d0:<n>,d1:<n>`` (``unavailable`` where the backend reports
    none), ``xla_peak_gb_max=<GB, 1e9, 3 decimals>``, ``peak_scope=xla_allocator``; when no device reports a positive peak the line NAMES why instead of a
    zero: ``xla_peak_unavailable=platform_allocator`` (``XLA_PYTHON_CLIENT_ALLOCATOR=platform`` keeps no XLA pool statistics), ``no_memory_stats`` (the
    backend returned none) or ``backend_reports_zero``."""
    import os  # noqa: PLC0415
    have = [p for p in peaks if p.get("peak_bytes_in_use") is not None]
    positive = [p for p in have if int(p["peak_bytes_in_use"]) > 0]
    out: Dict[str, Any] = {}
    out["xla_peak_bytes"] = ",".join(f"d{p['id']}:{int(p['peak_bytes_in_use'])}" for p in have) if have else "unavailable"
    out["xla_peak_gb_max"] = (f"{max(int(p['peak_bytes_in_use']) for p in positive) / 1e9:.3f}") if positive else "unavailable"
    if not positive:
        alloc = (os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR") or "").lower()
        out["xla_peak_unavailable"] = "platform_allocator" if alloc == "platform" else ("no_memory_stats" if not have else "backend_reports_zero")
    out["peak_scope"] = PEAK_SCOPE
    return out


ON_REQUIRED = ("schedule", "kernel", "kernel_reason", "sites")      # state=on is fail-closed: the trimul schedule, the per-shard kernel decision, the sharded sites


def line(tag: str, state: str, n_gpu: int, *pairs, lever: str = LEVER, reason: Optional[str] = None, strategy: Optional[str] = None,
         rmesh: Any = None, environ: Optional[Mapping[str, str]] = None, peaks: Optional[Sequence[Mapping[str, Any]]] = None, **evidence) -> str:
    """The family's ONE activation-evidence line: ``[<tag>] LEVER name=<lever> state=<on|off|skipped> [reason=…] impl=opt_core.mem.rowpair_jax@<v>
    origin=core [strategy=…] n_gpu=P sharding=rowpair|none [axis visible platform devices device_kind sm xla_pool_limit device_total_bytes xla_pool_fraction
    shard_map — when rmesh is given] prealloc=… mem_fraction=… client_mem_fraction=… allocator=… [xla_peak_bytes=… xla_peak_gb_max=… peak_scope=… — when peaks
    are given] [a2a_split=<k> — state=on and a re-layout all_to_all was issued in k pieces at the element limit (:func:`shard.all_to_all`)] schedule=…
    kernel=… kernel_reason=… sites=… <evidence k=v …>``. ``state=on`` REQUIRES ``rmesh`` and the keys :data:`ON_REQUIRED` in ``evidence``
    (ValueError otherwise — a sharded run whose schedule / per-shard kernel / site list is not on the record is not evidence); ``sites`` may be a sequence
    (joined with ``,``). ``state=off reason=n_gpu=1`` is the P=1 form."""
    from ...report import lever_line  # noqa: PLC0415
    from .mesh import xla_memory_env  # noqa: PLC0415
    from .shard import shard_map_flavour  # noqa: PLC0415
    if state == "on":
        missing = [k for k in ON_REQUIRED if k not in evidence]
        if missing or rmesh is None:
            raise ValueError(f"evidence.line state=on requires rmesh and {', '.join(ON_REQUIRED)} (missing: {', '.join(missing) or 'rmesh'})")
    if "sites" in evidence and not isinstance(evidence["sites"], str):
        evidence["sites"] = ",".join(str(x) for x in evidence["sites"])
    fields: List[Tuple[str, Any]] = list(active_fields(n_gpu))
    if rmesh is not None:
        d = rmesh.describe()
        fields += [(k, d[k]) for k in ("axis", "visible", "platform", "devices", "device_kind", "sm", "xla_pool_limit", "device_total_bytes", "xla_pool_fraction")]
        fields.append(("shard_map", shard_map_flavour()))
    fields += list(xla_memory_env(environ).items())
    if peaks is not None:
        pf = peak_fields(peaks)
        fields += [(k, pf[k]) for k in ("xla_peak_bytes", "xla_peak_gb_max", "xla_peak_unavailable", "peak_scope") if k in pf]
    if state == "on" and "a2a_split" not in evidence:                  # a2a_split=<k>: a re-layout all_to_all was issued in k pieces at the element limit (shard.all_to_all) —
        from .shard import a2a_fields  # noqa: PLC0415                    present only when a split was traced in this process (P>1 above the limit); absent, the line reads as before
        fields += list(a2a_fields().items())
    fields += list(pairs)
    ordered = [(k, evidence.pop(k)) for k in ON_REQUIRED if k in evidence]
    return lever_line(tag, lever, state, *(fields + ordered), reason=reason, impl=impl_label(), origin="core", strategy=strategy, **evidence)


def plan_rows() -> List[Tuple[str, str, str, str]]:
    """``[(site, locality, collectives, numerics_class), …]`` in :data:`SITES` order — the table the API document prints."""
    return [(k, v[0], v[1], v[2]) for k, v in SITES.items()]
