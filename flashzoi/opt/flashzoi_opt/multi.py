"""`pred --jobs <file>` — ONE process serves every job of the file (`input<TAB>out` per line, each what --input / --out take) where
the plain call serves one job per process; for many small jobs the weights load and the kit apply are paid once instead of per job.
The process is the one-process route's process with the job boundary moved inside it: the recipe (`--det`), the activation,
the one weights load and the kit's apply line on each replicate happen ONCE, before the first job; then per job the SAME loop as the
one-process route (loop.run_items: the documented call per item, the item written verbatim, its row), the kit's evidence read back
(stack.settle) and the job's own `opt_manifest.json` — every job written to its own output directory with the same files, the same
names, the same bytes per item as the one-process route writes them. Per-job state (the item list, the arrays, the rows) is built and
released per job; resident across jobs are the four kit-attached replicates (the weights, the kit's kernels and graphs, its pinned
pool) — the state the one-process route already holds across the items of one job. Nothing of the kit is changed and no numerics
switch is touched between jobs (the kit reads the process's numerics class back after every call).

Lines (report.py): the ACTIVE line ends ` multi=<n>`; `[flashzoi-opt multi] ready mode=<m> load=<s>s jobs=<n>` after the one load;
ONE `[flashzoi-opt multi] item <i>/<n> <out> items=<k> ok=<k> failed=<f> wall=<s>s rc=<rc>` line per job (the job's wall inside the
resident process: its first item's load from disk to its manifest written; rc 0 = every item ok); `[flashzoi-opt multi] DONE jobs=<n>
ok=<k> failed=<f> load=<s>s wall=<s>s` at the end; the per-window `[flashzoi-opt] pred` lines and the EXIT tally (totals over every job)
as on the one-process route. A refusal is the package's `[flashzoi-opt] NOT ACTIVE: <reason>` line, exit 3.
"""
from __future__ import annotations

import time

from . import inputs, loop, report
from . import settings as _settings


def plan(jobs: list) -> list:
    """[(input, out, items), ...]: every job's items resolved by the one item reader (inputs.list_items) BEFORE any load, so a job the
    reader refuses (inputs.InputError) refuses the whole form by name before a weight is read."""
    return [(inp, out, inputs.list_items(inp)) for inp, out in jobs]


def run_jobs(models, planned: list, settings: _settings.Settings = _settings.DEFAULT, device: str = "cuda", after_job=None) -> dict:
    """Every job through loop.run_items in file order; `after_job(index, input, out, counts, wall_s)` is called after each job (the CLI
    writes the job's manifest there). Returns {jobs, ok, failed, items, per_job: [{index, input, out, items, ok, failed, wall_s}]}."""
    n = len(planned)
    per_job, ok_jobs, items_total = [], 0, 0
    for i, (inp, out, items) in enumerate(planned, 1):
        t0 = time.perf_counter()
        counts = loop.run_items(models, items, out, settings, device)
        if after_job is not None:
            after_job(i, inp, out, counts, time.perf_counter() - t0)       # the job's manifest is part of the job's outputs: inside its wall
        wall = time.perf_counter() - t0
        rec = {"index": i, "input": inp, "out": out, "items": counts["items"], "ok": counts["ok"], "failed": counts["failed"], "wall_s": round(float(wall), 4)}
        per_job.append(rec)
        items_total += counts["items"]
        ok_jobs += int(counts["failed"] == 0)
        report.emit(report.multi_item_line(i, n, out, counts, wall))
    return {"jobs": n, "ok": ok_jobs, "failed": n - ok_jobs, "items": items_total, "per_job": per_job}
