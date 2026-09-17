"""Per-item PHASE timing line — the same boundaries on every single-process route (``pred --mode off``: the stock subprocess,
``stock_pred``; ``exact`` / ``fast`` / ``big`` (one GPU): the stock CLI in-process, ``cli.run_stock_cli``), installed once the stock
runner is imported and before its command runs. It changes no computed value: each site is the callable found on the class at install
time, called unchanged between ``torch.cuda.synchronize()`` + ``time.perf_counter()`` pairs (the synchronize is skipped when CUDA is not
initialised, so a CPU process pays nothing).

Sites (the pinned stock, ``stock/src``):
  trunk   = ``protenix.model.protenix.Protenix.get_pairformer_output`` — input embedding, relative-position / template / MSA modules and
            the pairformer stack over every recycle of the item (a kit mode's own wrapper of the site, e.g. big's seam, is inside the span);
  sampler = ``protenix.model.protenix.Protenix.sample_diffusion`` — the whole diffusion sampling of the item, every step of every sample
            (the diffusion-conditioning cache ``prepare_cache`` the loop builds before it is in ``total_s`` and in no phase);
  conf    = ``protenix.model.protenix.Protenix.run_confidence_head`` — the confidence heads (summed if the loop calls it more than once);
  fwd     = the model class's ``forward`` (MODEL_CLASS.FWD_SITE) — the call ``InferenceRunner.predict`` makes (``self.model(...)``:
            trunk recycles, the MSA / template modules, the diffusion sampler over every sample, the confidence heads) — the model
            forward alone: the feature dict's device copy-in before it and the CIF / JSON dump after it are outside the span;
  total   = ``runner.inference.InferenceRunner.predict`` — device copy-in + the model forward; the runner's own "Model forward time" line
            (predict + dump) is the cross-check above it;
  lm      = none: the protenix-v2 checkpoint has no protein-LM term (``esm.enable`` false) -> ``-``; a checkpoint with ``esm.enable``
            computes its ESM embeddings in the dataloader (``ESMFeaturizer``), outside the item's forward -> ``NA`` + one PHASE-NOTE.
Item = the runner's ``sample_name`` (= the plan item's name; ``runner/inference.py`` ``infer_predict``) with the seed as its own ``seed=`` token —
# the PHASE line grammar (item=<plan item name>, optional seed=<s>); the
seed is read off ``runner.inference.seed_everything`` (called once per seed pass before the items).

Line (stdout, once per successful predict call; nothing on an item that raised):
  ``PHASE item=<sample_name> seed=<seed> lm_s=<-|NA> trunk_s=<f> sampler_s=<f> conf_s=<f> fwd_s=<f> total_s=<f>``
Guard: trunk_s + sampler_s + conf_s <= fwd_s <= total_s by construction (nested spans); a violation prints ``PHASE-NOTE guard …`` — never hidden.
The multi-GPU line (``big --n_gpu P``, torchrun ranks) does not pass through these installers: its per-stage seconds are the unit's
``runmeta`` (``t_trunk_s`` / ``t_diffusion_s`` / ``t_confidence_s``, ``tp.py``). This module imports the standard library only (the stock
subprocess holds it); torch is read from ``sys.modules`` at call time.
"""
from __future__ import annotations

import functools
import importlib
import sys
import time
from typing import Optional

PREFIX = "PHASE"
MODEL_MODULE, MODEL_CLASS = "protenix.model.protenix", "Protenix"
RUNNER_MODULE, RUNNER_CLASS = "runner.inference", "InferenceRunner"
SITES = {"trunk": "get_pairformer_output", "sampler": "sample_diffusion", "conf": "run_confidence_head"}   # phase -> Protenix method
FWD_SITE = "forward"                                                                                      # the model class's method: the model forward alone
FWD = "fwd"                                                                                               # its accumulator / line token (fwd_s)
TOTAL_SITE = "predict"                                                                                    # InferenceRunner method
SEED_SITE = "seed_everything"                                                                             # runner.inference global
PHASES = tuple(SITES)
ACCUMULATORS = PHASES + (FWD,)                                                                            # every timed span of an item: the three phases + the model forward around them
_STATE = {"installed": None, "seed": None, "acc": {p: 0.0 for p in ACCUMULATORS}, "calls": {p: 0 for p in ACCUMULATORS}, "noted": set(), "lines": 0}


def _sync() -> None:
    cuda = getattr(sys.modules.get("torch"), "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_initialized", None)) and cuda.is_initialized():   # CUDA in use: drain the stream; a CPU process pays nothing
        cuda.synchronize()


def _note_once(key: str, text: str) -> None:
    if key not in _STATE["noted"]:
        _STATE["noted"].add(key)
        print(f"PHASE-NOTE {text}", flush=True)


def timed(fn, phase: str):
    """``fn`` between synchronize + perf_counter pairs, its seconds added to the item's ``phase`` accumulator (idempotent per phase)."""
    if getattr(fn, "_phase_timed", None) == phase:
        return fn

    @functools.wraps(fn)
    def wrapper(*a, **k):
        _sync(); t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            _sync()
            _STATE["acc"][phase] += time.perf_counter() - t0
            _STATE["calls"][phase] += 1
    wrapper._phase_timed = phase
    return wrapper


def item_name(data) -> str:
    """The item: the runner's ``sample_name`` (= the plan item's name)."""
    return data.get("sample_name", "unknown") if isinstance(data, dict) else "unknown"


def item_seed(seed=None):
    """The seed of this item's forward (the runner's current seed unless given)."""
    return _STATE["seed"] if seed is None else seed


def lm_field(runner) -> str:
    """``-`` (no protein-LM term in the model) or ``NA`` (esm.enable: the embedding is computed in the dataloader, outside the forward)."""
    emb = getattr(getattr(runner, "model", None), "input_embedder", None)
    if emb is not None and (getattr(emb, "esm_configs", None) or {}).get("enable"):
        _note_once("lm", "lm not separable from the item forward: esm.enable — the ESM embedding is computed by the dataloader "
                         "(ESMFeaturizer) before InferenceRunner.predict, outside total_s")
        return "NA"
    return "-"


def phase_line(item: str, seed, lm: str, acc: dict, total: float) -> str:
    """``PHASE item=<item> seed=<seed> lm_s=<lm> trunk_s=<f> sampler_s=<f> conf_s=<f> fwd_s=<f> total_s=<f>`` — the reducer's grammar (item =
    the plan item's bare name; seed its own token; every value one whitespace-free key=value token; ``fwd_s`` = ``acc["fwd"]``, the model
    forward, immediately before ``total_s``)."""
    fwd = acc.get(FWD, 0.0)
    line = (f"{PREFIX} item={item} seed={seed} lm_s={lm} " + " ".join(f"{p}_s={acc[p]:.3f}" for p in PHASES)
            + f" {FWD}_s={fwd:.3f} total_s={total:.3f}")
    s = sum(acc[p] for p in PHASES)
    if s > fwd * 1.001 + 1e-3:
        line += f"\nPHASE-NOTE guard violated on {item}: trunk_s+sampler_s+conf_s={s:.3f} > fwd_s={fwd:.3f}"
    if fwd > total * 1.001 + 1e-3:
        line += f"\nPHASE-NOTE guard violated on {item}: fwd_s={fwd:.3f} > total_s={total:.3f}"
    return line


def _cuda():
    """torch.cuda when torch is imported and a device is present, else None (read from sys.modules at call time: stock passes import torch themselves)."""
    cuda = getattr(sys.modules.get("torch"), "cuda", None)
    return cuda if cuda is not None and cuda.is_available() else None


def peak_reset() -> None:
    """The item's allocator-peak window opens: torch's peak counters reset (no-op without CUDA)."""
    cuda = _cuda()
    if cuda is not None:
        cuda.reset_peak_memory_stats()


def peak_lines(item: str, seed) -> list:
    """The torch peak grammar, per item: `PEAK item=<item> alloc_gib=<f> reserved_gib=<f>` (max_memory_allocated / max_memory_reserved
    since peak_reset, GiB = 2^30; exactly these tokens, item first) then `PEAK-NOTE item=<item> seed=<s|->`; without CUDA no PEAK line — one
    `PEAK-NOTE item=<item> seed=<s|-> cuda=absent`."""
    s = "-" if seed is None else seed
    cuda = _cuda()
    if cuda is None:
        return [f"PEAK-NOTE item={item} seed={s} cuda=absent"]
    gib = float(2 ** 30)
    return [f"PEAK item={item} alloc_gib={cuda.max_memory_allocated() / gib:.2f} reserved_gib={cuda.max_memory_reserved() / gib:.2f}", f"PEAK-NOTE item={item} seed={s}"]


def timed_total(fn):
    """``InferenceRunner.predict``: resets the item's accumulators (the three phases and the model forward) and the allocator-peak window,
    times the call, prints the PHASE line and the item's PEAK line(s) after a successful return."""
    if getattr(fn, "_phase_timed", None) == "total":
        return fn

    @functools.wraps(fn)
    def predict(self, data, *a, **k):
        for p in ACCUMULATORS:
            _STATE["acc"][p] = 0.0; _STATE["calls"][p] = 0
        peak_reset()
        _sync(); t0 = time.perf_counter()
        out = fn(self, data, *a, **k)
        _sync(); total = time.perf_counter() - t0
        print(phase_line(item_name(data), item_seed(), lm_field(self), _STATE["acc"], total), flush=True)
        for ln in peak_lines(item_name(data), item_seed()):                 # the item's allocator peak in the reader's grammar (every single-process route: stock and kit alike)
            print(ln, flush=True)
        _STATE["lines"] += 1
        return out
    predict._phase_timed = "total"
    return predict


def seed_recorder(fn):
    if getattr(fn, "_phase_timed", None) == "seed":
        return fn

    @functools.wraps(fn)
    def seed_everything(*a, **k):
        _STATE["seed"] = k.get("seed", a[0] if a else None)
        return fn(*a, **k)
    seed_everything._phase_timed = "seed"
    return seed_everything


def install() -> dict:
    """Wrap the sites in place (idempotent). Returns ``{"installed": bool, "sites": {phase: qualname}, "reason": str|None}``; a stock tree
    that is not importable leaves everything untouched and says so once (``PHASE-NOTE timing not installed: …``)."""
    try:
        model = importlib.import_module(MODEL_MODULE); runner = importlib.import_module(RUNNER_MODULE)
        cls, rcls = getattr(model, MODEL_CLASS), getattr(runner, RUNNER_CLASS)
        sites = {}
        for phase, attr in SITES.items():
            setattr(cls, attr, timed(getattr(cls, attr), phase)); sites[phase] = f"{MODEL_MODULE}.{MODEL_CLASS}.{attr}"
        setattr(cls, FWD_SITE, timed(getattr(cls, FWD_SITE), FWD)); sites[FWD] = f"{MODEL_MODULE}.{MODEL_CLASS}.{FWD_SITE}"   # nn.Module.__call__ resolves forward on the class: the runner's self.model(...) lands here
        setattr(rcls, TOTAL_SITE, timed_total(getattr(rcls, TOTAL_SITE))); sites["total"] = f"{RUNNER_MODULE}.{RUNNER_CLASS}.{TOTAL_SITE}"
        setattr(runner, SEED_SITE, seed_recorder(getattr(runner, SEED_SITE))); sites["seed"] = f"{RUNNER_MODULE}.{SEED_SITE}"
    except Exception as e:  # noqa: BLE001  (an absent stock tree: the route runs untimed, named)
        _STATE["installed"] = False
        _note_once("install", f"timing not installed: {type(e).__name__}: {e}")
        return {"installed": False, "sites": {}, "reason": f"{type(e).__name__}: {e}"}
    _STATE["installed"] = True
    return {"installed": True, "sites": sites, "reason": None}


def lines_printed() -> int:
    return int(_STATE["lines"])
