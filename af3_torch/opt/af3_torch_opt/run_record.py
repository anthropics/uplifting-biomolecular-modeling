"""The record of one `pred` in this process — the activation report, the stock proof, the partial verdict, the three steps and every item's
account: what the printed lines are rendered from, kept in memory for callers driving the package from Python (cli.last_run()); nothing is written."""
from __future__ import annotations

from typing import List, Optional


def build(report: dict, steps: List[dict], items: List[dict], inputs: List[str], argv: List[str], ok: bool,
          proof: Optional[dict] = None, partial: Optional[dict] = None, reports: Optional[dict] = None) -> dict:
    return {
        "activation": report,
        "stock_proof": proof,
        "partial": partial,                      # None, or {events: [{kind: levers|fallback|stock, …}], kinds, allow_partial}: the run is not the mode it claims
        "census": report.get("census"),          # the kit's kernel census of the model process (None for off): per-lever served/fallback counters, dead levers, graphs, hoist
        "graph_resets": report.get("graph_resets"),     # captured whole-step graphs dropped at item start (forward.py _reset_item_state); None for off
        "graph_captures": report.get("graph_captures"), # whole-step graphs captured, one per fast item; None for off
        "fallbacks_expected": report.get("fallbacks_expected"),   # the census's fallbacks inside the kernels' declared coverage (registry.EXPECTED_FALLBACKS): named, not a gate
        "kernel_routes": report.get("kernel_routes"),   # the kernels served from the shared core's carried copies: the model process's route_check verdict, core copy, imported_from (registry.KERNEL_ROUTES)
        "big": report.get("big"),                   # the memory levers in force (big.selection: levers, disabled toggles, settings) — None outside mode big
        "big_record": report.get("big_record"),     # the model process's record of them (forward.json big: alloc facts, diff_free calls, items above the kit row)
        "inputs": list(inputs),
        "argv": list(argv),
        "steps": steps,                          # [{step, argv, rc, wall_s, log, ok, error?}] in run order: featurise · forward · postprocess
        "reports": dict(reports or {}),        # the three steps' own reports (featurise / forward / postprocess json), read back — the work dir holding them is removed after a clean pred
        "items": items,                          # per input: {name, input, seeds, n_tokens, bucket, forwards: [{seed, forward_s, peak_mem_gb, peak_reserved_gb, ok, error?}], files, ranking, top, ok, error?}
        "ok": ok,
    }
