"""Inputs the stock parser skips — one accounting, one line family, both routes.

Boltz 2.2.1's ``process_inputs`` parses each YAML inside a try/except (boltz/main.py ``process_input``): an input it cannot parse — e.g. a
bond or ligand naming a chain id the schema never registered (``KeyError: 'GLY1_'`` on a branched-glycan chain id) — is reported on its own
stdout as ``Failed to process <path>. Skipping. Error: <text>.`` (the traceback on stderr) and left OUT of ``processed/manifest.json``;
the process carries on with the inputs that parsed and exits 0. That record list is where a skip is read, on both routes:

* the kit worker (opt/forward/trunk_levers/src/bz_worker_lev*.py ``process_item``) parses one YAML per item in a fresh child of its zygote (prep.py),
  whose captured stdout / stderr carry those words; an item whose record is absent from the manifest just written is SKIPPED: the worker
  prints ``LINE`` once, records it under ``skipped`` in its log, never builds a DataLoader for it (an empty one is where ``next()`` raised
  StopIteration) and predicts every other item — ``next_ready`` is the pipelined schedule's advance over skipped pairs;
* the stock route (cli.cmd_pred_off, one YAML per process) reads the same manifest under upstream's results directory after the process.

Either way the unit is accounted failed with a named reason (worker.run / cli.cmd_pred_off: ``outputs.failed_units``, the EXIT tally's
``failed``, exit 1 — the kit's "outputs short" exit), never silently and never as something else: a process whose every input was skipped
ran no ``predict_step``, so its KERNELS census says ``verdict=NO-STEP`` — report.kernels_exit lets the SKIPPED lines speak for exactly that
case instead of the REQUIRE guard's exit 5.

The second upstream skip is a batch, not an input: ``Boltz2.predict_step`` catches a CUDA out-of-memory ``RuntimeError`` itself
(boltz/model/models/boltz2.py:1123-1128), prints ``| WARNING: ran out of memory, skipping batch`` on its stdout, and hands the writer
``{"exception": True}``, which writes nothing for that batch (boltz/data/write/writer.py:60-62); the process goes on and exits 0. On the
worker route the kit worker sees that return value per (input, seed) (its predict_step wrapper), records the unit under ``failed`` in its
log with ``oom_reason`` and predicts the other units; worker.run names the unit ``FAILED item=<name> seed=<s> reason=upstream skipped
batch: CUDA out of memory (…)`` from that entry (``failed_from_log`` → ``units``) and relays boltz's own WARNING line; the exit is 1. On
the stock route boltz's WARNING line is on the caller's stdout itself and the unit is FAILED outputs short.
Standard library only: the worker imports this module in the model process.
"""
from __future__ import annotations

import json
import os
import re
from typing import Callable, List, Optional

PREFIX = "[boltz2-opt]"
MARK = "SKIPPED item="
LINE_RE = r"\[boltz2-opt\] SKIPPED item=.*"                       # worker.RELAY_LINES: the worker's line, verbatim on the caller's stdout
FAILED_MARK = "FAILED item="
STOCK_SKIP_RE = re.compile(r"^Failed to process (?P<path>.+?)\. Skipping\. Error: (?P<error>.*)\.\s*$", re.M)   # boltz/main.py process_input's own words (stdout)
TRACEBACK_LAST_RE = re.compile(r"^(?P<exc>[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning)): (?P<msg>.*)$", re.M)   # traceback.print_exc()'s last line (stderr)
REASON = "stock parser skipped the input"
OOM_REASON = "upstream skipped batch: CUDA out of memory"           # Boltz2.predict_step's own catch (boltz2.py:1123-1128): the batch is skipped, nothing written, the process exits 0
STOCK_OOM_RE = r"\| WARNING: ran out of memory, skipping batch.*"    # boltz's own words on that path (stdout) — worker.RELAY_LINES carries them to the caller's transcript


def oom_reason(peak_alloc_gib=None) -> str:
    """``upstream skipped batch: CUDA out of memory (boltz Boltz2.predict_step caught torch's OOM and wrote no structure for this unit[; peak
    allocated <x> GiB]; --mode big, then --n_gpu, is the line for inputs this large)`` — the FAILED reason of an (input, seed) unit whose
    predict_step returned ``{"exception": True}``."""
    peak = f"; peak allocated {float(peak_alloc_gib):.1f} GiB" if peak_alloc_gib is not None else ""
    return f"{OOM_REASON} (boltz Boltz2.predict_step caught torch's OOM and wrote no structure for this unit{peak}; --mode big, then --n_gpu, is the line for inputs this large)"


def upstream_skipped(digs: Optional[dict]) -> Optional[str]:
    """The named reason when the worker's predict_step wrapper recorded boltz's own batch skip for the unit just predicted (``digs`` = its
    per-item facts: ``upstream_skipped`` True, ``peak_mem_GB``), else None."""
    d = digs or {}
    return oom_reason(d.get("peak_mem_GB")) if d.get("upstream_skipped") else None


def reason(error: Optional[str] = None) -> str:
    """``stock parser skipped the input (<error>)`` — the error text is boltz's own when its output carried one."""
    err = (error or "").strip()
    return f"{REASON} ({err})" if err else f"{REASON} (no record in processed/manifest.json; boltz's \"Failed to process … Skipping\" line above names the parser error)"


def line(name: str, why: str) -> str:
    """``[boltz2-opt] SKIPPED item=<name> reason=<why>`` — one per skipped input, the same family as ACTIVE / PHASE / KERNELS."""
    return f"{PREFIX} {MARK}{name} reason={why}"


def failed_line(name: str, seed, why: str) -> str:
    """``[boltz2-opt] FAILED item=<name> seed=<seed> reason=<why>`` — one per expected (input, seed) unit without its outputs."""
    return f"{PREFIX} {FAILED_MARK}{name} seed={seed} reason={why}"


def stock_error(stdout_text: str, stderr_text: str = "", stem: Optional[str] = None) -> Optional[str]:
    """The parser error boltz reported for the input whose file stem is `stem` (any input when None): the traceback's last line
    (``KeyError: 'GLY1_'``) when stderr carried one, else the ``Error: <text>`` of the stdout line; None when boltz reported no skip."""
    hit = None
    for m in STOCK_SKIP_RE.finditer(stdout_text or ""):
        p = m.group("path")
        if stem is None or os.path.splitext(os.path.basename(p))[0] == stem:
            hit = m; break
    if hit is None:
        return None
    tb = [m for m in TRACEBACK_LAST_RE.finditer(stderr_text or "")]
    return f"{tb[-1].group('exc')}: {tb[-1].group('msg')}".strip() if tb else (hit.group("error").strip() or "?")


def record_ids(manifest_path: str) -> Optional[List[str]]:
    """The record ids of a processed manifest (boltz's ``processed/manifest.json``: ``{"records": [{"id": <yaml stem>, …}]}``); None when
    the file is absent or unreadable (the process never reached the parser: not a parser skip)."""
    if not os.path.isfile(manifest_path):
        return None
    try:
        return [str(r.get("id")) for r in (json.load(open(manifest_path)).get("records") or [])]
    except (OSError, ValueError, AttributeError, TypeError):
        return None


def stock_results_manifest(out_dir: str, yaml_path: str) -> str:
    """Where upstream's CLI writes the processed manifest for one input file: ``<out_dir>/boltz_results_<stem>/processed/manifest.json``."""
    stem = os.path.splitext(os.path.basename(yaml_path))[0]
    return os.path.join(out_dir, f"boltz_results_{stem}", "processed", "manifest.json")


def next_ready(k: int, n: int, skipped_at: Callable[[int], bool]) -> int:
    """The pipelined schedule's advance: the first index i in [k, n) with ``not skipped_at(i)``, else n. ``skipped_at`` may process the
    input as a side effect (the worker parses lazily, in order); it is asked about each index at most once per call, in order."""
    while k < n and skipped_at(k):
        k += 1
    return k


def units(items: list, seeds: list, skipped: dict, missing: Callable[[str, int], Optional[str]], failed: Optional[dict] = None) -> dict:
    """The completeness account of a worker-route run: every expected (item, seed) unit is ok (``missing(name, seed)`` is None: every expected
    file is there — worker.expected_files: ``<name>_model_0.cif``, plus ``affinity_<name>.json`` for an input declaring the affinity property) or
    failed with a named reason — the worker's SKIPPED reason when the stock parser skipped the input, else the reason the worker recorded for
    that very unit (``failed`` = {(name, seed): reason} from its log's ``failed`` entries: boltz's own batch skip on CUDA out of memory,
    oom_reason), else outputs short naming the first missing file. Counts close: ``ok + failed == expected``. ``skipped`` = {name: reason}
    from the worker log's ``skipped`` entries."""
    ok = 0; failed_units = []; named = failed or {}
    for it in items:
        for s in seeds:
            lack = missing(it["name"], s)
            if lack is None:
                ok += 1
            else:
                why = skipped.get(it["name"]) or named.get((it["name"], int(s))) or f"outputs short (no {lack} under by_seed/{it['name']}/s{s}/)"
                failed_units.append({"name": it["name"], "seed": s, "reason": why})
    n = ok + len(failed_units)
    return {"expected": n, "ok": ok, "failed": len(failed_units), "status": "complete" if not failed_units else "incomplete",
            "failed_units": failed_units, "skipped": [{"name": k, "reason": v} for k, v in skipped.items()],
            "all_skipped": bool(items) and all(it["name"] in skipped for it in items)}


def skipped_from_log(wlog: Optional[dict]) -> dict:
    """{name: reason} from the worker log's ``skipped`` entries (bz_worker: one per input the stock parser skipped)."""
    out = {}
    for e in (wlog or {}).get("skipped") or []:
        if isinstance(e, dict) and e.get("name"):
            out[str(e["name"])] = str(e.get("reason") or reason())
    return out


def failed_from_log(wlog: Optional[dict]) -> dict:
    """{(name, seed): reason} from the worker log's ``failed`` entries (bz_worker: one per (input, seed) unit boltz's predict_step skipped on
    CUDA out of memory — the reason is oom_reason's)."""
    out = {}
    for e in (wlog or {}).get("failed") or []:
        if isinstance(e, dict) and e.get("name") and e.get("seed") is not None:
            try:
                out[(str(e["name"]), int(e["seed"]))] = str(e.get("reason") or oom_reason())
            except (TypeError, ValueError):
                continue
    return out
