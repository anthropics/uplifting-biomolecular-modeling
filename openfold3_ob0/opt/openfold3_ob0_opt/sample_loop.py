"""The ``sample_loop`` lever (``opt_core.mem.sample_loop``) on OpenFold3: the diffusion roll-out and the confidence head run ``chunk`` of the
``S = --num-diffusion-samples`` structures per pass, every pass's outputs moved to the host, the per-sample outputs assembled along the sample
dimension (dim 1 of ``[1, S, ...]``) in stock order — so the device holds the roll-out activations, the confidence pair representation and its
logits for ``chunk`` samples, never ``S``.

Lines. ``big`` at ``--n_gpu P > 1`` (line tp, the row-sharded pair representation: opt/openfold3_ob0_opt/tp_rowpair): ON by default with ``chunk = 1``; every rank runs the
same loop on its rows and rewinds ITS OWN default CUDA generator per pass (the ranks' streams are identical by construction and
``diffusion.sync_replicated`` guards the sampled coordinates per pass). ``big`` at ``--n_gpu 1`` (line resident): the line's own per-sample
confidence rule stands (OpenFold3's ``per_sample_token_cutoff`` / ``per_sample_atom_cutoff``); this lever is not bound there.

Exactness: the draws are stock's per sample (``opt_core.mem.ckpt.StockOrderDraws``: the generator rewound per pass, every batch-shaped draw made
at the ``S`` shape and sliced), so chunked == batched bit for bit where the engine's kernels are batch-invariant; the tier-2 (tolerance) line
carries its tier word. Refused by name (``opt_core.mem.sample_loop.SampleLoopRefused``): ``k > S``,
``k < 1``.

Surface: :func:`plan_for` (the plan from ``S`` and the environment), :func:`run_rollout` (the loop around the adapter's two per-pass callables,
the sample-invariant confidence leaves kept once and checked equal across passes, the small per-atom leaves returned to the device, the
census line ``[sample_loop] samples= chunk= passes= draw_order= host= peak_gb=`` on rank 0's stderr and in the schedule census).
"""
from __future__ import annotations

import sys
from typing import Any, Callable, Dict, Mapping, Optional

LEVER = "sample_loop"
CHUNK = 1                                        # samples per pass on the tp line (one: the roll-out's device memory is independent of --num-diffusion-samples)
SAMPLE_DIM = 1                                   # [1, S, ...]: OpenFold3's predict batch of 1, then the sample dimension
INVARIANT_KEYS = ("contact_probs",)              # confidence leaves with no sample dimension (the distogram head reads the trunk pair representation)
DEVICE_KEYS = ("plddt", "iptm", "ptm", "disorder", "has_clash", "sample_ranking_score", "gpde")   # small per-sample leaves returned to the device after assembly
MIN_CORE = "0.4.4"
LAST: Dict[str, Any] = {}                        # the census fields of the last roll-out this lever drove (the kit's activation record reads it: stack._p_sample_loop)


def core():
    """``opt_core.mem.sample_loop``, refused by name when the installed opt_core lacks it."""
    try:
        from opt_core.mem import sample_loop as SL  # noqa: PLC0415
    except ImportError as e:
        try:
            import opt_core  # noqa: PLC0415
            v = getattr(opt_core, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            v = "unimportable"
        raise RuntimeError(f"refused: the {LEVER} lever needs opt_core.mem.sample_loop (opt_core >= {MIN_CORE}); installed opt_core {v} lacks it "
                           f"({type(e).__name__}: {e})") from e
    return SL


def plan_for(n_samples: int):
    """The tp line's plan: ``CHUNK`` samples per pass (source ``default``)."""
    return core().plan(int(n_samples), None, default_chunk=CHUNK)


def census_line(fields: Mapping[str, Any]) -> str:
    peaks = fields.get("sample_peaks_gb") or []
    ptxt = ",".join("na" if p is None else f"{p:g}" for p in peaks)
    return (f"[{LEVER}] samples={fields.get('samples')} chunk={fields.get('sample_chunk')} source={fields.get('sample_chunk_source')} "
            f"passes={fields.get('sample_passes')} draw_order={fields.get('draw_order')} host={fields.get('sample_host')} "
            f"peak_gb={fields.get('sample_peak_gb')} peaks_gb=[{ptxt}]")


def one_pass_fields(plan) -> Dict[str, Any]:
    """The census fields of the one batched pass (``k == S``: nothing chunked, no host round trip)."""
    return {"samples": plan.n_samples, "sample_chunk": plan.chunk, "sample_chunk_source": plan.source, "sample_passes": 1, "draw_order": plan.draw_order,
            "sample_host": "device", "sample_peak_gb": None, "sample_peaks_gb": []}


def emit(fields: Mapping[str, Any], log: Optional[Callable[[str], None]], rank: int = 0, record_schedule: Optional[Callable[..., None]] = None) -> str:
    """The lever's census line (the line's marker ``[sample_loop] samples=``): through ``log`` when given (the rank's logger, which writes the rank
    log / stderr with its rank prefix), else on rank 0's stderr; and the schedule census when ``record_schedule`` is given. One emission per roll-out."""
    line = census_line(fields)
    LAST.clear()
    LAST.update(dict(fields))
    if log is not None:
        log(line)
    elif rank == 0:
        print(line, file=sys.stderr, flush=True)
    if record_schedule is not None:
        record_schedule(samples=fields["samples"], sample_chunk=fields["sample_chunk"], sample_chunk_source=fields["sample_chunk_source"],
                        sample_passes=fields["sample_passes"], sample_draw_order=fields["draw_order"], sample_host=fields["sample_host"],
                        sample_peak_gb=fields["sample_peak_gb"])
    return line


def _split_invariant(conf: Dict[str, Any], keep: Dict[str, Any], chunk: slice, equal: Callable[[Any, Any], bool]) -> Dict[str, Any]:
    """Pop the sample-invariant leaves of a pass's confidence dict: kept from the first pass, checked equal (bitwise) on later passes."""
    out = dict(conf)
    for k in INVARIANT_KEYS:
        if k not in out:
            continue
        v = out.pop(k)
        if k in keep:
            if not equal(keep[k], v):
                raise core().SampleLoopRefused(f"sample-invariant confidence leaf {k!r} differs between passes (pass of samples {chunk.start}:{chunk.stop}) — "
                                               "it must not depend on the sampled coordinates")
        else:
            keep[k] = v
    return out


def run_rollout(plan, diffuse: Callable[[slice, Any], Any], confidence: Callable[[Any, slice], Dict[str, Any]], *, device, log: Callable[[str], None],
                rank: int = 0, record_schedule: Optional[Callable[..., None]] = None) -> Dict[str, Any]:
    """The loop of ``OpenFold3._rollout`` on the tp line: per pass ``x = diffuse(chunk, draws)`` (``[1, k, N_atom, 3]``, replicated) then
    ``aux = confidence(x, chunk)`` (the dict ``aux_heads_rows`` returns: ``plddt_logits`` / ``experimentally_resolved_logits`` ``[1, k, N_atom, *]``
    and ``_tp_confidence`` = the finished per-sample confidence dict); returns the assembled ``{'atom_positions_predicted', 'plddt_logits',
    'experimentally_resolved_logits', '_tp_confidence'}`` over all ``S`` samples (per-atom leaves back on ``device``; ``pae`` / ``pde`` and the
    scalars where ``aux_heads_rows`` put them, concatenated on the host)."""
    import torch  # noqa: PLC0415
    SL = core()
    keep: Dict[str, Any] = {}

    def _rollout(chunk, draws):
        return diffuse(chunk, draws)

    def _confidence(x, chunk):
        aux = dict(confidence(x, chunk))
        conf = _split_invariant(aux.pop("_tp_confidence"), keep, chunk, torch.equal)
        return {"atom_positions_predicted": x, **aux, "_tp_confidence": conf}          # every leaf [1, k, ...]; nested dicts concatenated leaf-wise

    gen_device = device if (device is not None and str(device).startswith("cuda")) else None
    out = SL.run(plan, _rollout, _confidence, device=gen_device, sample_dim=SAMPLE_DIM, host=True, log=log)
    f = out.fields()
    emit(f, log, rank, record_schedule)
    result = dict(out.value)
    conf = dict(result.pop("_tp_confidence"))
    for k in INVARIANT_KEYS:
        if k in keep:
            conf[k] = keep[k]
    result = {k: (t.to(device) if isinstance(t, torch.Tensor) else t) for k, t in result.items()}   # per-atom leaves back on the device (small)
    for k in DEVICE_KEYS:                                                                            # per-sample scalars / per-atom plddt back on the device;
        if isinstance(conf.get(k), torch.Tensor):                                                    # pae / pde / contact_probs stay on rank 0's host
            conf[k] = conf[k].to(device)
    result["_tp_confidence"] = conf
    result["_sample_loop"] = f
    return result
