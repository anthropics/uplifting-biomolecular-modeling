"""Precision words — the tree's own numerics levers on upstream's inference precision policy (`runner/inference.py:update_inference_configs`).

One word is carried: ``sampler_amp`` (fast / big, tier 2). Upstream runs the diffusion sampler in fp32 with autocast disabled below 3,840
tokens (`configs.skip_amp.sample_diffusion = True`, `runner/inference.py:1489-1516`; the model applies it per call through
`autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)`, `opendde/model/opendde.py:1332`) and exposes
`OPENDDE_FORCE_SAMPLE_DIFFUSION_AMP=1` to lift that (`:1511-1512`). Its fp32 sampler GEMMs already run on TF32 tensor cores (`enable_tf32=True`,
`opendde/model/opendde.py:2086-2097`) — the word is not about TF32. With the word bound, an item whose token count (upstream's own `n_token`, the
policy function's argument) is at most ``MODEL_OPT_SAMPLER_AMP_MAX_TOKENS`` runs its sampler under the runner's ambient bf16 autocast
(`--dtype bf16`, the stock base of this tree): `skip_amp.sample_diffusion = False` — exactly the configs value upstream's variable sets, set at
upstream's own policy site, per item, after upstream's size policy ran. The variable itself stays absent on every route (modes.UPSTREAM_EXPOSED_UNUSED;
the stock caller strips and proves it absent). Above the word's gate upstream's policy stands untouched (counted `above_gate`); under a stated
`--dtype fp32` there is no autocast region to inherit and the word is inert by construction (counted `no_amp_dtype`). The confidence head's flag
(`skip_amp.confidence_head`, `OPENDDE_FORCE_CONFIDENCE_AMP`) is never touched (its effect on the sampler phase is small; CHANGES.md "Levers").

The knob. ``MODEL_OPT_SAMPLER_AMP_MAX_TOKENS`` (default = upstream's own 3,840: engaged on every item below upstream's AMP regime) is the
per-card retune point (CHANGES.md "Levers"). The word's gain shrinks with size and depends on the sampler step graph (``stepgraph``) being on
the line: without the graph the per-block re-casts of the O(N²) pair-bias tensors are launch-bound host time, so a card or line without the
graph lowers the knob.

Installed from `stack._apply` for the lines that carry the word through the core's per-site patch (`opt_core.autoload.patch_attr_at_import`):
`runner.inference.update_inference_configs` is re-bound at once when the module is imported, else right after its body executes
(`_prepare_prediction_batch` resolves the name at call time, `runner/inference.py:1545`). Idempotent. STATS counts every policy call and what the
word did with it; `kit_stats()` feeds the LEVER row (report.lever_evidence: calls / engaged / above_gate / gate). Composes with the sampler step graph (stepgraph): the autocast casts are
kernels inside the captured step, no host sync, no per-step shape.
"""
from __future__ import annotations

import os
import sys

from opt_core import autoload

TARGET = "runner.inference"
FUNC = "update_inference_configs"
TAG = "opendde-opt"
LEVER = "sampler_amp"
WORDS = (LEVER,)                                      # the precision words this module carries (registry rows of kit `house`)
GATE_ENV = "MODEL_OPT_SAMPLER_AMP_MAX_TOKENS"
UPSTREAM_AMP_ABOVE = 3840                             # runner/inference.py:1501: above it upstream runs the sampler under autocast itself (the word changes nothing there)
GATE_MAX = UPSTREAM_AMP_ABOVE                         # tokens: engaged at <= the gate. As shipped = upstream's own AMP size, i.e. no kit gate below upstream's own AMP regime;
                                                      # the knob (GATE_ENV) is the per-card retune point (CHANGES.md "Levers")
ABOVE_GATE = "above_gate"
STATS = {"installed": False, "armed": False, "calls": 0, "engaged": 0, "above_gate": 0, "upstream_amp": 0, "no_amp_dtype": 0,
         "gate": None, "last_n_token": None}


class ActivationError(RuntimeError):
    """`runner.inference` has no `update_inference_configs` to wrap: the kit's activation fails by name."""


def gate() -> int:
    """The word's upper size gate in tokens (``MODEL_OPT_SAMPLER_AMP_MAX_TOKENS``, default GATE_MAX; 0 = never engaged)."""
    raw = (os.environ.get(GATE_ENV) or "").strip()
    if not raw:
        return GATE_MAX
    try:
        return max(0, int(raw))
    except ValueError:
        raise ValueError(f"{GATE_ENV}={raw!r}: expected a non-negative integer (tokens)") from None


def _dtype_word(configs) -> str:
    d = getattr(configs, "dtype", None)
    return str(d).lower() if d is not None else "bf16"


def decide(n_token: int, configs=None) -> str:
    """The word's per-item decision at upstream's policy site: 'engaged' | 'above_gate' | 'upstream_amp' | 'no_amp_dtype'."""
    if configs is not None and _dtype_word(configs) not in ("bf16", "bfloat16", "torch.bfloat16"):
        return "no_amp_dtype"                                                   # a stated --dtype fp32 (or fp16): no bf16 autocast region for the sampler to inherit
    if int(n_token) > UPSTREAM_AMP_ABOVE:
        return "upstream_amp"                                                   # upstream's own policy already runs the sampler under autocast there
    if int(n_token) > gate():
        return ABOVE_GATE
    return "engaged"


def make_wrapper(orig):
    """The function installed as `runner.inference.update_inference_configs`: upstream's policy, then the word's rule on the result."""
    def update_inference_configs(configs, n_token, *args, **kwargs):
        configs = orig(configs, n_token, *args, **kwargs)
        STATS["calls"] += 1
        STATS["gate"] = gate()
        STATS["last_n_token"] = int(n_token)
        d = decide(n_token, configs)
        STATS[d] += 1
        if d == "engaged":
            configs.skip_amp.sample_diffusion = False                               # = what OPENDDE_FORCE_SAMPLE_DIFFUSION_AMP=1 sets at runner/inference.py:1511-1512
        return configs
    update_inference_configs._sampler_amp = True
    update_inference_configs._orig = orig
    return update_inference_configs


def _sync() -> None:
    p = STATS.get("patch")
    if p is not None:
        STATS["installed"], STATS["armed"] = p.state == "installed", p.state == "armed"


def install() -> None:
    """Patch now when `runner.inference` is imported, else at its import (the core's per-site patch). Idempotent."""
    _sync()
    if STATS["installed"] or STATS["armed"]:
        return
    try:
        STATS["patch"] = autoload.patch_attr_at_import(TARGET, FUNC, make_wrapper, tag=TAG, name=LEVER)
    except autoload.PatchError as e:
        raise ActivationError(str(e)) from None
    _sync()


def fallbacks(planned) -> list:
    """The word's named events at exit: the patch armed in this process and the runner module imported, but its policy function not wrapped
    (the armed patch did not fire). A process that never imported the runner has no event (inert: no prediction could reach the site), nor has
    one where the word was planned but never installed (the install accounting names that)."""
    if LEVER not in planned:
        return []
    _sync()
    out = []
    if STATS.get("patch") is not None and TARGET in sys.modules and not STATS["installed"]:
        out.append(f"{LEVER}: {TARGET} is imported but {FUNC} is not wrapped (the patch armed in this process did not fire)")
    return out


def kit_stats() -> dict:
    _sync()
    d = {k: v for k, v in STATS.items() if k != "patch"}
    d["gate"] = STATS["gate"] if STATS["gate"] is not None else gate()
    return d


def _reset() -> None:
    """Test support: forget the plan and the counters (the patch object is kept: one per site per process)."""
    for k in ("calls", "engaged", "above_gate", "upstream_amp", "no_amp_dtype"):
        STATS[k] = 0
    STATS["gate"] = None
    STATS["last_n_token"] = None
