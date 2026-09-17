"""The package's programmatic entry points — the functions `run.sh sample|score --mode exact` is built on, for a caller that keeps a model
loaded across many calls in its own process (one size per process, as for the command line):

    from progen2_opt.api import load, sample, score, close
    g = load("progen2-xlarge")                                   # generation: sample.py's load order + the decode levers; fp16=True, device="cuda:0"
    r = sample(g, context="1", max_length=512, num_samples=64, rng_seed=7, t=0.2, p=0.95)   # one sample.py call's worth: r.block (its stdout block), r.ids, r.record
    close(g)
    s = load("progen2-base", route="score")                      # scoring: likelihood.py's load order + the scoring kit applied
    score(s, "1MKV...GSGS2").block                               # `ll_sum=...` / `ll_mean=...` — one likelihood.py call's worth
    close(s)                                                     # the scoring route's counted-path proof, then the model is dropped

Every call prints the package's lines on stderr as the command line does (stack / ACTIVE / load / ready ...); a refusal by name raises
``progen2_opt.stack.ActivationError`` with its NOT ACTIVE line printed. The static K/V slots of a `num_samples` value are allocated on its first
call (after the kit's fit check; a batch size that cannot fit raises the kit's OutOfMemory) and kept in the handle; every call runs at the same
speed from the first.
"""
from __future__ import annotations

from . import generate as _generate
from . import score as _score
from .generate import Sampled, sample          # noqa: F401 — re-exported
from .score import Scored, score               # noqa: F401 — re-exported

__all__ = ["load", "sample", "score", "close", "Sampled", "Scored"]


def load(model: str, *, route: str = "sample", fp16: bool = True, device: str = "cuda:0", rng_seed: int = 42, rng_deterministic: bool = True, **kw):
    """Load one size (`model` = the stock's --model name) for ``route`` "sample" (generate.load: + max_length=, shapes=) or "score" (score.load: +
    sanity=); the stock's own --fp16 / --device / --rng-seed / --rng-deterministic as it types them. Returns the route's handle."""
    if route == "sample":
        return _generate.load(model, fp16=fp16, device=device, rng_seed=rng_seed, rng_deterministic=rng_deterministic, **kw)
    if route == "score":
        return _score.load(model, fp16=fp16, device=device, rng_seed=rng_seed, rng_deterministic=rng_deterministic, **kw)
    raise ValueError(f"route {route!r}: sample | score")


def close(handle) -> dict:
    """Release a handle from ``load`` (generation: the slots, the stock module restored; scoring: the counted-path proof, the model dropped)."""
    return handle.close()
