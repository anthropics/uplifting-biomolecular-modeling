"""The scoring route in-process — mode ``exact`` for `score`: the stock boot order through the stock module's own functions, then the
scoring kit on the loaded model and its ``Scorer`` for the forwards. likelihood.py's own settings are honoured as it honours them:
``--rng-seed`` / ``--rng-deterministic`` (its set_seed), ``--fp16`` (create_model's dtype AND the autocast of every forward), ``--device``
(the CUDA card; `cpu` never reaches here — the levers run on CUDA, the activation refuses it by name), ``--sanity`` (its section 5, below).

    likelihood.set_env(); likelihood.set_seed(seed, deterministic=rng_deterministic)   # the stock's preamble (likelihood.py L134-135)
    model = likelihood.create_model(ckpt="<run dir>/checkpoints/<name>", fp16=fp16).to(device)   # the stock's load (L152), the run directory by path
    tokenizer = likelihood.create_tokenizer_custom("<run dir>/tokenizer.json")                    # L156
    kit.apply(model, ew=True, size="<name>", resident_fp16=False)                     # the kit: v0_ew composed first, tables bound
    scorer = kit.Scorer(model, tokenizer, autocast=fp16, max_rows=1)                   # B = 1 per direction: the exact class on every size
    rows, meta = scorer.score_T(units)                                                 # units = [{"unit_key": item_id, "seq": context}, ...]

Per item the block is likelihood.py's result lines — ``ll_sum=<float>`` / ``ll_mean=<float>`` printed from the kit's row (the same reduction
the stock prints: likelihood.py L295-299, `.5 * (lr + rl)`), preceded under ``--sanity true`` by the stock's own sanity section (its `ce` / `ll`
closures and the `if args.sanity:` block, L172-273, compiled from the stock file and run on THIS model after set_seed, as every fresh stock
process runs them: the same prints, print_time's timer lines left out) — so the two arms compare as the bytes of block.txt
(outputs.block_of_stdout's rule for the stock's stdout). One ``score_T`` call per item: the kit's batched form is exact only at its own shapes,
so the port never batches. A sequence longer than the model's n_positions is FAILED by name for that item (the stock cannot run it either);
the job goes on.

Activation: the ACTIVE line is printed right after apply (report.log_active) with the mode's levers — all of them: what the kit found
untested and engaged on (a stack or gelu_new bytes off the tested ones, kernels JIT-compiled from PTX) rides on it as ``notes=``; what cannot
run (the elementwise kit's library missing / not loading / no image for the card) is the kit's KitRefused naming the levers — the CLI's refusal
by name, exit 3, nothing scored. The exit rule: the kit's own rotary-table check after apply (its resident tables equal the stock function's
output at every position, n/n) and, after the items, its counted-path assertion (`assert_counted_path` on this process's counters, two forwards
per unit) are the evidence every lever of the composition ran on every forward; a contradiction there means the kit did not compute what it
names — a PARTIAL activation of the run, recorded in the report (`partial`, `partial_detection`) and the summary, the outputs kept, exit 3.
"""
from __future__ import annotations

import importlib
import os
import sys
import time
from typing import List, Optional

from . import modes, report, stack

PARTIAL_DETECTION = "the scoring kit's rotary-table check (n/n) and its counted-path assertion (assert_counted_path on this process's counters: two forwards per unit on the counted path)"
KIT_REFUSED = "KitRefused"                # the kits' fail-loud class, by name (v0_score_r3_1.KitRefused / v0_ew.ext.KitRefused: two classes, one name)
SANITY_DEFS, SANITY_FLAG = ("ce", "ll"), "sanity"   # likelihood.py main(): the closures its sanity section calls (section 4) and the flag its `if` tests (section 5)
_STATE = {"model_index": 0}


def _forward_root_on_path() -> str:
    root = stack.kit_dir("forward")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def kit_module():
    _forward_root_on_path()
    return importlib.import_module(modes.SCORING_KIT_MODULE)


def apply_kit(model, variant: str) -> dict:
    """The kit on a loaded model: ``apply(model, ew=True, size=<upstream name>, resident_fp16=False)`` (a KitRefused names what cannot run).
    The record names what engaged (``optimizations``: the kit's rows + `ew:<lever>` per elementwise lever in force) and what the kit found
    untested and engaged on (``notes``, the elementwise kit's own words)."""
    kit = kit_module()
    up = stack.upstream_name(variant)
    rec = kit.apply(model, resident_fp16=False, ew=True, size=up)
    _STATE["model_index"] += 1
    ew = rec.get("ew") or {}
    return {"kit": kit.KIT, "ew": True, "resident_fp16": False, "size": up, "model_index": _STATE["model_index"],
            "optimizations": list(modes.SCORING_OPTIMIZATIONS) + [f"ew:{n}" for n in (ew.get("levers") or [])],
            "notes": list(ew.get("notes") or []), "apply_record": rec}


def stock_module(workdir: str):
    """The stock `likelihood` module imported from the run directory (its bytes are the pinned checkout's: the activation hashed them)."""
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    return importlib.import_module("likelihood")


def boot(variant: str, workdir: str, seed: int = 42, fp16: bool = True, device: str = "cuda:0", rng_deterministic: bool = True):
    """The stock load order through the stock module's own functions (likelihood.py L134-156, its own flags; the run directory's
    `checkpoints/<name>` and `tokenizer.json` by path, the caller's cwd untouched); returns (likelihood_module, model, tokenizer, device,
    t_load_s). A CUDA card other than the current one is made current first (the kit's tables and forwards live on it)."""
    lk = stock_module(workdir)
    import torch
    lk.set_env()
    lk.set_seed(seed, deterministic=rng_deterministic)
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is not None and dev.index != torch.cuda.current_device():
        torch.cuda.set_device(dev)
    t0 = time.monotonic()
    model = lk.create_model(ckpt=os.path.join(workdir, "checkpoints", stack.upstream_name(variant)), fp16=fp16).to(dev)
    tokenizer = lk.create_tokenizer_custom(file=os.path.join(workdir, "tokenizer.json"))
    return lk, model, tokenizer, dev, time.monotonic() - t0


def units_of(items: List[dict]) -> List[dict]:
    return [{"unit_key": it["item_id"], "seq": it["context"]} for it in items]


def block_of_row(row: dict) -> str:
    """likelihood.py L298-299 from the kit's row (floats printed as Python prints them)."""
    if not row.get("ok", True):
        return ""
    return f"ll_sum={row['ll_sum']}\nll_mean={row['ll_mean']}\n"


class _Untimed:
    """print_time (likelihood.py L22-31) without its two lines: the sanity section's timers are the stock CLI's progress output, not a result
    (outputs.block_of_stdout leaves them out of the stock's block the same way)."""

    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def stock_sanity_code(lk):
    """likelihood.py main()'s sanity statements, compiled from the stock file itself (never re-typed): the `ce` / `ll` closures (section 4) and
    the `if args.sanity:` block (section 5). main()'s locals they read (model, tokenizer, device, args) become the namespace they run in."""
    import ast
    tree = ast.parse(open(lk.__file__, "r", encoding="utf-8").read(), filename=lk.__file__)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    body = [n for n in main.body if (isinstance(n, ast.FunctionDef) and n.name in SANITY_DEFS)
            or (isinstance(n, ast.If) and isinstance(n.test, ast.Attribute) and n.test.attr == SANITY_FLAG)]
    if [type(n).__name__ for n in body] != ["FunctionDef", "FunctionDef", "If"]:
        raise RuntimeError(f"likelihood.py main(): the sanity section is not the pinned one ({[type(n).__name__ for n in body]})")
    return compile(ast.Module(body=body, type_ignores=[]), lk.__file__, "exec")


def run_stock_sanity(code, lk, model, tokenizer, device, upstream: str, fp16: bool) -> str:
    """ONE run of the stock's sanity section on this model, as a fresh likelihood.py process runs it before its section 7: its own prints
    (the cross-entropy triple, ll_0..2, the three sequences, ll_x_*) collected and returned; its asserts assert."""
    import argparse
    import contextlib
    import io
    ns = dict(vars(lk))                                                          # the stock module's own globals (torch, random, cross_entropy, log_likelihood*, ...)
    ns.update(model=model, tokenizer=tokenizer, device=device, args=argparse.Namespace(model=upstream, fp16=bool(fp16), sanity=True), print_time=_Untimed)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, ns)                                                           # noqa: S102 — the stock's own statements
    return buf.getvalue()


class Loaded:
    """One size loaded for scoring (``load``): the stock module, the model with the kit applied, the tokenizer, the Scorer and the run's report.
    ``score`` runs one item on it; ``close`` runs the route's own proof (the counted-path assertion) and drops the model."""

    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.n_ok = 0
        self.closed = False

    def close(self) -> dict:
        return finish(self)


class Scored:
    """One item's result: ``block`` = likelihood.py's result lines (`ll_sum=` / `ll_mean=`, after its sanity section's prints when asked; empty
    when the item failed), ``row`` = the kit's row (ok, ll_sum, ll_mean | error), ``wall_s``."""

    def __init__(self, block: str, row: dict, wall_s: float):
        self.block, self.row, self.wall_s = block, row, wall_s


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


def load(model: str, *, fp16: bool = True, device: str = "cuda:0", rng_seed: int = 42, rng_deterministic: bool = True, sanity: bool = False,
         rep: Optional[dict] = None) -> Loaded:
    """Load one size for scoring and return its handle: `model` = the stock's --model name or its size word. The activation is resolved and gated
    first unless the caller hands its report (`rep`; a refusal raises stack.ActivationError, its line printed), then the stock load, the kit's apply
    (the ACTIVE line from its record, the APPLIED and ready lines), the kit's rotary-table check (a mismatch is recorded as the run's partial
    activation) and the Scorer at B=1 per direction. ``sanity`` compiles likelihood.py's sanity section for ``score`` to run before each item."""
    variant = model if model in modes.VARIANTS else modes.variant_of_model(model)
    if rep is None:
        rep = stack.activate("exact", variant, strict=True, route="score", device=device, outside_defaults=outside_defaults(fp16, rng_deterministic, device))
    workdir = stack.workdir(variant)
    lk, mdl, tokenizer, dev, t_load = boot(variant, workdir, seed=rng_seed, fp16=fp16, device=device, rng_deterministic=rng_deterministic)
    rec = apply_kit(mdl, variant)
    rep["applied"], rep["apply_record"] = "configured", {k: v for k, v in rec.items() if k != "apply_record"}
    report.log_active(rep, rec["optimizations"], notes=rec["notes"])
    print(report.applied_line(rep, rec), file=sys.stderr, flush=True)
    print(report.ready_line(variant, t_load), file=sys.stderr, flush=True)
    kit = kit_module()
    rep["partial"], rep["partial_detection"] = [], PARTIAL_DETECTION
    rot = kit.rotary_table_check(mdl)                                   # the kit's own in-place check: resident tables == the stock function's output, n/n
    rep["rotary_check"] = rot
    if rot.get("n_mismatch"):
        rep["partial"].append(f"rotary tables: {rot.get('n_mismatch')}/{rot.get('n')} positions differ from the stock function (first {rot.get('mismatch_lengths')})")
    scorer = kit.Scorer(mdl, tokenizer, autocast=bool(fp16), max_rows=1)
    rep["scorer"] = {"autocast": bool(fp16), "max_rows": 1, "route": scorer.runner.record()["route"]}
    snap = kit.counters_snapshot()                                     # the counters before any scoring forward (the counted-path assert reads the delta; the sanity forwards go through the model, not the Scorer)
    return Loaded(lk=lk, model=mdl, tokenizer=tokenizer, device=dev, variant=variant, upstream_name=stack.upstream_name(variant), rep=rep, kit=kit, scorer=scorer,
                  snap=snap, seed=int(rng_seed), fp16=bool(fp16), rng_deterministic=bool(rng_deterministic), t_load=t_load, record=rec,
                  sanity_code=(stock_sanity_code(lk) if sanity else None))


def score(h: Loaded, context: str, item_id: str = "item0000") -> Scored:
    """ONE likelihood.py call's worth on the loaded handle: after the stock's sanity section when the handle was loaded with ``sanity`` (re-seeded
    first, as a fresh process is), one ``score_T`` call at B=1 per direction — the block = the lines likelihood.py prints. A sequence longer than
    the model's n_positions fails by name (row.ok False, the stock cannot score it either)."""
    t0 = time.monotonic()
    pre = ""
    n_tok = len(h.tokenizer.encode(context).ids)
    if n_tok > h.scorer.n_positions:                                   # the model's context: the stock's causal-mask buffer ends there too (it cannot run the sequence either)
        r = {"ok": False, "error": f"{n_tok} tokens exceed the model's n_positions {h.scorer.n_positions} (the stock cannot score it either)"}
    else:
        if h.sanity_code is not None:
            h.lk.set_seed(h.seed, deterministic=h.rng_deterministic)   # a fresh stock process seeds, then draws its x_random: so does every item here
            pre = run_stock_sanity(h.sanity_code, h.lk, h.model, h.tokenizer, h.device, h.upstream_name, h.fp16)
        rows, meta = h.scorer.score_T(units_of([{"item_id": item_id, "context": context}]))
        r = rows[0] if rows else {"ok": False, "error": "no row"}
    h.n_ok += int(bool(r.get("ok")))
    return Scored((pre + block_of_row(r)) if r.get("ok") else "", r, time.monotonic() - t0)


def finish(h: Loaded) -> dict:
    """The route's own proof after the items: the kit's counted-path assertion on this process's counters (two forwards per unit: lr + rl); its
    refusal is the run's PARTIAL activation (recorded on the report; the outputs stay, the CLI decides the exit). Prints the `score ok` line when
    complete; returns {units, forwards, counters, partial}. Then the model is dropped."""
    if h.closed:
        return dict(h.rep.get("counted_path") or {}, partial=list(h.rep.get("partial") or []))
    h.closed = True
    kit, rep = h.kit, h.rep
    rep["counters"] = kit.counters_snapshot()
    forwards = 2 * h.n_ok
    delta = kit.counters_delta(h.snap)
    try:
        kit.assert_counted_path(delta, n_forwards=forwards)
    except kit.KitRefused as e:
        rep["partial"].append(f"counted path: {e}")
    rep["counted_path"] = {"units": h.n_ok, "forwards": forwards, "counters": delta, "complete": not rep["partial"]}
    if not rep["partial"]:
        print(report.score_ok_line(h.n_ok, forwards, delta), file=sys.stderr, flush=True)     # partial: the CLI prints the partial-exit line (report.partial_line)
    h.model = h.scorer = None
    return dict(rep["counted_path"], partial=list(rep["partial"]))


def run_items(items: List[dict], variant: str, rep: dict, emit, seed: int = 42, fp16: bool = True, device: str = "cuda:0",
              rng_deterministic: bool = True, sanity: bool = False) -> dict:
    """The score route of the command line: ``load`` (the ACTIVE line from the record), then per item ``score`` — ``emit(item, block)`` receives
    each item's block (a job writes items/<id>/block.txt, one call prints it) and the item line follows — then ``finish`` (the exit rule).
    ``seed`` / ``fp16`` / ``device`` / ``rng_deterministic`` / ``sanity`` are likelihood.py's own flags as it types them. Returns the summary
    (`partial` = the exit rule's findings)."""
    h = load(variant, fp16=fp16, device=device, rng_seed=seed, rng_deterministic=rng_deterministic, sanity=sanity, rep=rep)
    for it in items:
        r = score(h, it["context"], it["item_id"])
        emit(it, r.block)
        print(report.item_line(it["item_id"], r.wall_s) + ("" if r.row.get("ok") else f" FAILED {r.row.get('error')}"), file=sys.stderr, flush=True)
    n_ok = h.n_ok
    report.note_items(len(items))
    fin = finish(h)
    return {"n_items": len(items), "n_complete": n_ok, "rc": 0 if n_ok == len(items) else 1, "t_load_s": h.t_load, "partial": fin["partial"],
            "on": list(h.record["optimizations"]), "notes": list(h.record["notes"])}
