"""The package's own hooks on the stock runner (`runner.inference`): the big lever `guard_lift` and the per-item failure record
every kit mode carries.

guard_lift (PTX_GUARD_LIFT=1): the stock runner refuses `n_token > 2560` for protenix-v2 by an assertion whose text names memory as
the reason (`runner/inference.py` `update_inference_configs`: "It might cause OOM"). Lifted, the item runs with the precision
settings protenix-v2 runs at every size stock accepts: the confidence head under the runner's autocast (`skip_amp.confidence_head`
False — the runner's > 2560 setting, equal to protenix-v2's own) and the diffusion sampler in fp32 (`skip_amp.sample_diffusion` True).
The runner's generic policy for `n_token > 3840` — written for the other checkpoints' memory — would run the 200-step sampler under
bf16 autocast (`skip_amp.sample_diffusion` False: the atom encoder reads the noisy positions through a bf16 Linear and the denoised
coordinates are rounded to bf16 every step); the lift sets that one field back to True after the runner's call (counted
`sampler_fp32_items`), so no arithmetic changes at any size. An out-of-memory above the guard is a NAMED failure (item_failures below), never a
silent skip and never a retry at lower precision. big only: the stock run keeps the guard.

The seam: `runner.inference.update_inference_configs(configs, n_token)` — the runner calls it per item after its own N_token
adjustments and hands the result to the model (`InferenceRunner.update_model_configs`). The guard reads `configs.model_name` and nothing
else above it, so the lift calls the stock function with the name marked `+guard-lifted` for that call and restores it. Installed by
`stack._apply` when the switch is set: immediately if `runner.inference` is imported, else by a post-import finder.

item_failures: the stock runner catches an item's exception, logs `[Rank r] <name> failed: <e>` (`runner/inference.py` `infer_predict`)
and goes on; the process exits 0 with fewer outputs. A logging handler on the runner's logger keeps every such line as a record
{item, reason_class (OOM | refused | other), text}; the outputs census names them beside the missing files (the kit's exit rule already turns the
short census into EXIT_FAIL).
"""
from __future__ import annotations

import os
import sys

import logging
import re

ENV_GUARD = "PTX_GUARD_LIFT"
TARGET = "runner.inference"
FUNC = "update_inference_configs"
GUARD_TOKENS = 2560                                    # runner/inference.py update_inference_configs: the protenix-v2 assertion's bound
GUARD_MARK = "+guard-lifted"
_STATE = {"guard_lift": False, "patched": False, "guard_lifted_items": 0, "sampler_fp32_items": 0, "max_n_token": 0, "finder": None,
          "failures": [], "handler": None, "listeners": []}


def add_item_listener(fn) -> None:
    """`fn(n_token)` runs at every update_inference_configs call (once per item and seed): the big census's unit delimiter."""
    ls = _STATE.setdefault("listeners", [])
    if fn not in ls:
        ls.append(fn)


def guard_lift_from_env(environ=None) -> bool:
    """PTX_GUARD_LIFT=1 lifts the runner's n_token guard (big only); unset/0 keeps it; any other value is refused by name."""
    v = (environ if environ is not None else os.environ).get(ENV_GUARD)
    if v in (None, "", "0"):
        return False
    if v != "1":
        raise ValueError(f"{ENV_GUARD}={v!r}: 1 (lift the stock n_token guard) or unset")
    return True


def _patch(module) -> None:
    orig = getattr(module, FUNC)
    if getattr(orig, "_ptx_runner_hooks", False):
        return

    def update_inference_configs(configs, n_token, _orig=orig):
        _STATE["max_n_token"] = max(_STATE["max_n_token"], int(n_token))
        for fn in list(_STATE.get("listeners") or ()):
            fn(int(n_token))
        lifted = _STATE["guard_lift"] and int(n_token) > GUARD_TOKENS and getattr(configs, "model_name", None) == "protenix-v2"
        if lifted:
            name = configs.model_name
            configs.model_name = name + GUARD_MARK            # the assertion reads the name (and nothing else above it); restored below
            try:
                cfg = _orig(configs, n_token)
            finally:
                configs.model_name = name
            _STATE["guard_lifted_items"] += 1
            if not cfg.skip_amp.sample_diffusion:                # the runner's large-N policy (bf16 sampler) is not protenix-v2's: the sampler stays fp32 as at every size stock accepts
                cfg.skip_amp.sample_diffusion = True
                _STATE["sampler_fp32_items"] += 1
        else:
            cfg = _orig(configs, n_token)
        return cfg
    update_inference_configs._ptx_runner_hooks = True
    update_inference_configs.__wrapped__ = orig
    setattr(module, FUNC, update_inference_configs)
    _STATE["patched"] = True


class _FailureHandler(logging.Handler):
    """Keeps the runner's per-item failure lines (`[Rank r] <name> failed: <e>`) as records; OOM named by class."""
    LINE = re.compile(r"\[Rank \d+\]\s+(?P<item>\S+)\s+failed:\s*(?P<text>.*)", re.S)

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        m = self.LINE.search(msg)
        if not m:
            return
        text = m.group("text").strip()
        cls = ("OOM" if re.search(r"out of memory|OutOfMemoryError|CUDA error: out of memory", text, re.I)
               else "refused" if text.startswith("refused:") else "other")          # `refused: …` = a kit guard's named refusal (templates.TemplateSlotsAllDummy)
        _STATE["failures"].append({"item": m.group("item"), "reason_class": cls, "text": text[:400]})


def install_failure_record() -> None:
    """Every kit mode: the runner's item-failure lines become records — ONE handler on the runner's logger, however many times it is
    installed in a process (the handler reads the module's current state at emit time)."""
    log = logging.getLogger(TARGET)
    have = [h for h in log.handlers if isinstance(h, _FailureHandler)]
    if have:
        _STATE["handler"] = have[0]
        return
    _STATE["handler"] = _FailureHandler(level=logging.ERROR)
    log.addHandler(_STATE["handler"])


def failures() -> list:
    return list(_STATE["failures"])


class PostImportFinder:
    """Run ``on_import(module)`` on ``target`` right after its first execution (the kit's `_autoload.Finder` form: exec_module wrapped,
    never a find_spec side effect; the finder removes itself). The one post-import patch form of the package: the runner hooks arm
    it on `runner.inference`; big.py arms it on the model modules its levers patch."""

    def __init__(self, target: str = TARGET, on_import=None):
        self.target = target
        self.on_import = on_import if on_import is not None else _patch

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.target:
            return None
        spec = None
        for finder in sys.meta_path:
            if finder is self:
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:
                spec = None
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            return None
        orig = spec.loader.exec_module
        me = self

        def exec_module(module, _orig=orig):
            _orig(module)
            try:
                me.on_import(module)
            finally:
                try:
                    sys.meta_path.remove(me)
                except ValueError:
                    pass
        spec.loader.exec_module = exec_module
        return spec


_PostImportFinder = PostImportFinder                      # the runner's own instance (target runner.inference, on_import=_patch)


def install(guard_lift: bool = False) -> list:
    """Arm the guard lift; returns its applied marker (`GUARD_LIFT:1`, `(patched)` now or `(armed)` for the runner's import)."""
    _STATE["guard_lift"] = _STATE["guard_lift"] or guard_lift
    mod = sys.modules.get(TARGET)
    if mod is not None:
        _patch(mod); how = "patched"
    else:
        if _STATE["finder"] is None:
            _STATE["finder"] = _PostImportFinder()
            sys.meta_path.insert(0, _STATE["finder"])
        how = "armed"
    return [f"GUARD_LIFT:1({how})"] if guard_lift else []


def state() -> dict:
    """The lever's own end-of-run record: the lift, whether the runner was patched, the items it applied to (and, of those, the items
    whose sampler setting it put back to fp32), the item failures the runner logged."""
    return {"guard_lift": _STATE["guard_lift"], "patched": _STATE["patched"], "guard_lifted_items": _STATE["guard_lifted_items"],
            "sampler_fp32_items": _STATE["sampler_fp32_items"], "max_n_token": _STATE["max_n_token"], "failures": failures()}
