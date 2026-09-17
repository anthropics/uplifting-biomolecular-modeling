"""The sample route in-process — mode ``exact`` for `sample` — and the package's programmatic entry to it (``load`` / ``sample`` / ``close``;
api.py re-exports them beside score's).

    kit = kit_module()                                             # opt/serving/pipeline_v0_4/progen2_decode.py, its directory on sys.path
    h = kit.load(<stock `sample` module>, "progen2-<size>", workdir, fp16=, device=, rng_seed=42, rng_deterministic=, max_length=)
                                                                   # sample.py's own load order (set_env, set_seed, the weights read once, the tokenizer)
                                                                   # + the decode levers installed, sized in length for max_length, no batch size held yet
    h.hold([(num_samples, max_length), ...])                       # the work in front of the model: its batch sizes' static K/V slots allocated (after the fit check)
    h.unit(context, max_length, num_samples, rng_seed, t, p, rng_deterministic)   # ONE stock sample() call after set_seed: the block sample.py prints

sample.py's own settings are honoured as it honours them: ``--fp16`` (the load's dtype route), ``--device`` (the CUDA card; `cpu` never reaches
here — the activation refuses it by name), ``--rng-seed`` / ``--rng-deterministic`` (set_seed before every unit), ``--sanity`` (its witness runs
once per loaded model before the first unit, whatever the flag: the stock's own order). A job (``run_items``) is ONE load sized for the job's
longest max_length with the job's batch sizes held from the load, then one unit per item; a single call is a job of one item. The memory rule is
the kit's (progen2_decode.py: one fit rule, the named out-of-memory error, no fallback); every unit runs at the same speed from the first.

Lines (report.py): ``ACTIVE ... route=sample kit=opt/serving/pipeline_v0_4 on=<levers>`` once the levers are installed in this process (the
evidence: the handle's own lever list against the mode's composition — a lever missing is a refusal by name, nothing generated), ``load
variant=<v> t=<s>s slots=<batch sizes held|-> max_length=<slots length>`` when the model is loaded and sized for the work in front of it, ``ready
variant=<v> t=<s>s`` (t = the wall from the route's start), then per item ``item <id> <s>s`` (or ``... FAILED <the named error>``) and ``PEAK
item=<id> alloc_gib=<a> reserved_gib=<r> pid=<pid>`` (this process's allocator high-water marks).
"""
from __future__ import annotations

import importlib
import os
import sys
import time
from typing import List, Optional

from . import modes, report, stack
from .outputs import SAMPLE_DEFAULTS

OUT_OF_MEMORY = "OutOfMemory"   # the kit's named out-of-memory class, by name (progen2_decode.OutOfMemory)


def kit_module():
    """The generation kit's module (opt/serving/pipeline_v0_4/progen2_decode.py), its directory on sys.path."""
    d = stack.kit_dir("serving")
    if d not in sys.path:
        sys.path.insert(0, d)
    return importlib.import_module(modes.GENERATION_MODULE)


def stock_module(workdir: str):
    """The stock `sample` module imported from the run directory (its bytes are the pinned checkout's: the activation hashed them)."""
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    return importlib.import_module("sample")


def outside_defaults(fp16: bool, rng_deterministic: bool, device: str) -> list:
    """The stock settings given outside the tested defaults, as the stack line names them (the levers engage; the uncertainty is named)."""
    out = []
    if not fp16:
        out.append("fp16=false")
    if not rng_deterministic:
        out.append("rng_deterministic=false")
    if str(device) not in ("cuda:0", "cuda") and str(device).split(":")[0] != "cpu":
        out.append(f"device={device}")
    return out


def load(model: str, *, fp16: bool = True, device: str = "cuda:0", rng_seed: int = 42, rng_deterministic: bool = True, max_length: int = SAMPLE_DEFAULTS["max_length"],
         shapes=None, rep: Optional[dict] = None):
    """Load one size for generation and return its handle: `model` = the stock's --model name (progen2-small … progen2-xlarge) or its size word.
    The activation is resolved and gated first (the stock files' digests, the kit tree, the card: a refusal raises stack.ActivationError with the
    NOT ACTIVE line printed), the model loaded as sample.py loads it with the decode levers installed (sized in length for `max_length` or the longest
    of `shapes`), the ACTIVE line printed from the levers actually installed, the batch sizes of `shapes` ([(num_samples, max_length), ...] — the
    work in front of the model, when known) held, then the load and ready lines. `rep` = an activation report the caller already holds (the CLI's);
    None resolves one here."""
    variant = model if model in modes.VARIANTS else modes.variant_of_model(model)
    if rep is None:
        rep = stack.activate("exact", variant, strict=True, route="sample", device=device, outside_defaults=outside_defaults(fp16, rng_deterministic, device))
    t0 = time.monotonic()
    wd = stack.workdir(variant)
    up = stack.upstream_name(variant)
    shapes = [(int(shp[0]), int(shp[1])) for shp in (shapes or [])]
    L = max(shp[1] for shp in shapes) if shapes else int(max_length)   # sized in length for the work in front of the model (the kit buckets it: 256 / 512 / 1024 / 2048, capped at n_positions)
    kit = kit_module()
    h = kit.load(stock_module(wd), up, wd, fp16=bool(fp16), device=str(device), rng_seed=int(rng_seed), rng_deterministic=bool(rng_deterministic), max_length=L)
    planned, got = report.route_on(rep, "sample"), list(h.levers)
    if sorted(planned) != sorted(got):   # a mode is all of its levers: the levers this process installed ARE the mode's composition, or nothing is generated
        missing = [(name, "not installed by the load") for name in planned if name not in got] + [(name, "installed but not in the mode's composition") for name in got if name not in planned]
        line = report.refused_line(f"{report.cannot_run_words(missing)} — nothing generated", "exact", variant, "sample")
        print(line, file=sys.stderr, flush=True)
        rep.update(active=False, refused=line)
        h.close()
        raise stack.ActivationError(line)
    rep["applied"] = "configured"
    report.log_active(rep, got)
    h.variant, h.rep = variant, rep
    if shapes:
        h.hold(shapes)
    print(report.load_line(variant, h.load_s, h.held_batches(), h.kit.S.max_slots), file=sys.stderr, flush=True)
    print(report.ready_line(variant, time.monotonic() - t0), file=sys.stderr, flush=True)
    return h


class Sampled:
    """One unit's result: ``block`` = the text sample.py prints for the call (the context line, per sample a blank line + index + truncation,
    `done.`); ``ids`` = the (num_samples, length) int64 token tensor as sampled (host copy); ``record`` = the unit's facts (sampler.tokens_sha256,
    held_batches, max_slots, mem_policy, kit_stats, mem_stats, unit_s)."""

    def __init__(self, block: str, ids, record: dict):
        self.block, self.ids, self.record = block, ids, record


def sample(h, context: str = SAMPLE_DEFAULTS["context"], max_length: int = SAMPLE_DEFAULTS["max_length"], num_samples: int = SAMPLE_DEFAULTS["num_samples"],
           rng_seed: int = SAMPLE_DEFAULTS["rng_seed"], t: float = SAMPLE_DEFAULTS["t"], p: float = SAMPLE_DEFAULTS["p"], rng_deterministic: bool = True) -> Sampled:
    """ONE sample.py call's worth on the loaded handle (its flags, its defaults): set_seed(rng_seed, rng_deterministic), the stock sample() through the
    exact sampler and the decode levers, truncate — the same block bytes a fresh `sample.py` process prints. A batch size whose static K/V cannot fit
    raises the kit's OutOfMemory (the named error; nothing is retried or rerouted)."""
    r = h.unit(context=str(context), max_length=int(max_length), num_samples=int(num_samples), rng_seed=int(rng_seed), t=float(t), p=float(p), rng_deterministic=bool(rng_deterministic))
    return Sampled(r["block"], r["ids"], r["record"])


def close(h) -> dict:
    """Release the handle's slots and restore the stock module's functions."""
    return h.close()


def run_items(items: List[dict], variant: str, rep: dict, emit, settings: dict) -> dict:
    """The sample route of the command line: ONE load for the items (their (num_samples, max_length) shapes in front of the model: the slots sized
    and held from the load), then per item one unit — ``emit(item, block)`` receives each block (a job writes items/<id>/block.txt, a single call prints it) and the item and PEAK
    lines follow; an item whose batch size cannot fit is named (``item <id> ... FAILED OUT OF MEMORY at ...``) and the job goes on. ``settings`` =
    sample.py's job-level flags as it types them (fp16, device, rng_deterministic; sanity rides along: the witness runs once per load). Returns the summary."""
    kit = kit_module()
    shapes = [(int(it["num_samples"]), int(it["max_length"])) for it in items]
    h = load(variant, fp16=settings["fp16"], device=settings["device"], rng_deterministic=settings["rng_deterministic"], shapes=shapes, rep=rep)
    n_ok = 0
    for it in items:
        t0 = time.monotonic()
        try:
            r = h.unit(context=it["context"], max_length=it["max_length"], num_samples=it["num_samples"], rng_seed=it["rng_seed"], t=it["t"], p=it["p"], rng_deterministic=settings["rng_deterministic"])
        except kit.OutOfMemory as e:   # the named row (OUT_OF_MEMORY): this item fails by name, the next is attempted (OOM_POLICY: nothing retried, nothing rerouted)
            print(report.item_line(it["item_id"], time.monotonic() - t0) + f" FAILED {str(e)[:600]}", file=sys.stderr, flush=True)
            continue
        emit(it, r["block"])
        n_ok += 1
        print(report.item_line(it["item_id"], time.monotonic() - t0), file=sys.stderr, flush=True)
        print(report.peak_line(it["item_id"], *h.peak_gib(), os.getpid()), file=sys.stderr, flush=True)
    report.note_items(len(items))
    return {"n_items": len(items), "n_complete": n_ok, "rc": 0 if n_ok == len(items) else 1, "load_s": h.load_s, "held": h.held, "on": list(h.levers)}
