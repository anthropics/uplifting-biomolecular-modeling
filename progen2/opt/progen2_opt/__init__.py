"""progen2_opt — the ProGen2 inference optimizations behind the stock's own command line.

    python -m progen2_opt sample [--mode exact|off] <sample.py's own flags>       # == `progen2-opt sample ...` == `run.sh sample ...`
    python -m progen2_opt score  [--mode exact|off] <likelihood.py's own flags>
    from progen2_opt.api import load, sample, score, close                       # the same routes for a long-lived caller (api.py)

ProGen2 IS a command line upstream (`sample.py`, `likelihood.py`), and both kits run in the calling process: the generation kit
(opt/serving/pipeline_v0_4: the weights read once, the decode component's resident rotary tables + static K/V slots, the exact sampler)
behind `sample` (generate.py), the scoring kit (opt/forward/…: `apply(model, ...)` + `Scorer`) behind `score` (score.py). The command layer
(cli.py) resolves and gates the mode (the stock files' digests, the kit tree) and prints the activation lines; the optimizations are applied
by the kits' own code at load. Nothing is imported at interpreter start beyond this module; torch and the stock load only inside a command.

Modes (`modes.KIT_MODES`): "exact" = the kits (the package default), "off" = stock (the upstream CLIs in a clean process, their own
flags untouched).
Sizes: the stock's `--model` names (`modes.MODEL_NAMES`) — one checkpoint per process. No environment variable selects a mode or a size.

Engagement (stack.activate, report.py): a lever engages wherever its mechanism applies, and a mode is ALL of its levers: what is untested
about the box (a stack off the pinned versions, a card outside the known classes, a stock setting outside the tested defaults, kernels
JIT-compiled for a newer card, a replaced function's bytes off the tested ones) is NAMED on the activation lines and engaged on — never a
reason to disengage; what CANNOT run (no CUDA device / `--device cpu`, the fused-kernel library missing or not loading or without an image for
the card, a lever the load did not install) makes the mode refuse BY NAME (exit 3, the lever and the reason) — it never runs under its
name with a subset and never falls back to the stock route. The one refusal every mode shares: the stock files at the stock dir are not the
pinned commit's bytes (that changes what stock means).
"""
__version__ = "0.1.0"
