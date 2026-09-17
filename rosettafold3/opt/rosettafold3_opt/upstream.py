"""The upstream-outcome census: what rf3 itself did with each input of this process — folded, early-stopped, or nothing to predict.

Upstream decides two things a kit lever never sees at its own seam: ``skip_existing`` can leave ZERO structures to predict
("Found 0 structures to predict!"), and ``early_stopping_plddt_threshold`` (CLI default 0.5) stops an input after the recycle-1
probe ("Early stopping triggered … No structure will be written") — no diffusion roll-out, no confidence pass. In both cases a
lever whose seam is the roll-out (the sampler graph, the DiT kernels, the FPF kernels of a trunk that never ran) reaches NO call
although the mode engaged correctly: the exit verdict must name that (``no_call_reached:zero_items`` / ``no_call_reached:early_stop``)
and the run exits with stock's code, instead of failing closed as if a lever had silently not engaged (report.fpf_tally / tally,
fold.big_failures read :func:`no_rollout_reason`). A process where at least one input DID roll out keeps the fail-closed rule.

The census is byte-neutral and always on (every kit row): the two trainer classes' ``validation_step`` (``rf3.trainers.rf3``, the
same classes hostlean.py patches) are wrapped once at class level to COUNT — items entered (outermost call), items whose
``network_output['early_stopped']`` is true — and return upstream's result untouched. Stdlib only; composes with hostlean's wrapper in
either order (distinct ``__name__``); idempotent.
"""
from __future__ import annotations

import sys
from typing import Callable, Dict, Optional

TRAINERS_MODULE = "rf3.trainers.rf3"
TRAINER_CLASSES = ("RF3Trainer", "RF3TrainerWithConfidence")
WRAP_NAME = "validation_step_upstream_census"

STATE: Dict[str, object] = {"armed": False, "installed": False, "reason": None, "items": 0, "early_stopped": 0, "classes": []}
_DEPTH = {"n": 0}


def _early_stopped(out) -> bool:
    """Upstream's contract (rf3.inference_engines.rf3): ``validation_step(...)['network_output'].get('early_stopped', False)``."""
    try:
        net = out.get("network_output") if isinstance(out, dict) else None
        return bool(net.get("early_stopped", False)) if isinstance(net, dict) else False
    except Exception:                                        # noqa: BLE001 — a census never raises into upstream
        return False


def _wrap(orig):
    def validation_step(self, *args, **kwargs):
        outer = _DEPTH["n"] == 0                             # a subclass step calling its base: the outer call is the item
        _DEPTH["n"] += 1
        if outer:
            STATE["items"] = int(STATE["items"]) + 1
        try:
            out = orig(self, *args, **kwargs)
        finally:
            _DEPTH["n"] -= 1
        if outer and _early_stopped(out):
            STATE["early_stopped"] = int(STATE["early_stopped"]) + 1
        return out
    validation_step.__wrapped__ = orig
    validation_step.__name__ = WRAP_NAME
    validation_step.__qualname__ = WRAP_NAME
    validation_step.__doc__ = getattr(orig, "__doc__", None)
    return validation_step


def _already_wrapped(fn) -> bool:
    while fn is not None:
        if getattr(fn, "__name__", "") == WRAP_NAME:
            return True
        fn = getattr(fn, "__wrapped__", None)
    return False


def enable(trainers=None) -> dict:
    """Wrap the trainer classes' own ``validation_step`` (class level, once each). A class upstream no longer has is recorded, not raised."""
    if trainers is None:
        import importlib
        trainers = importlib.import_module(TRAINERS_MODULE)
    done, missing = [], []
    for c in TRAINER_CLASSES:
        cls = getattr(trainers, c, None)
        vs = cls.__dict__.get("validation_step") if cls is not None else None
        if vs is None:
            missing.append(c)
            continue
        if not _already_wrapped(vs):
            setattr(cls, "validation_step", _wrap(vs))
        done.append(c)
    STATE.update(installed=bool(done), classes=done, reason=(None if done else f"no validation_step on {','.join(missing) or TRAINER_CLASSES}"))
    return describe()


def arm(rep: dict, install_watch: Callable) -> dict:
    """Activation-time hook on EVERY kit row: wrap the trainer classes when ``rf3.trainers.rf3`` executes (the kit's one import-watch
    installer, chained with hostlean's on the same module); a module imported already is wrapped at once. Never raises out of the hook."""
    if STATE.get("armed"):
        return rep
    STATE["armed"] = True

    def on_trainers(module):
        try:
            rep["upstream"] = enable(trainers=module)
        except Exception as e:                               # noqa: BLE001 — a census that cannot install leaves the fail-closed verdict as it was
            STATE.update(installed=False, reason=f"{type(e).__name__}: {e}")
            rep["upstream"] = describe()

    if TRAINERS_MODULE in sys.modules:
        on_trainers(sys.modules[TRAINERS_MODULE])
    else:
        install_watch(TRAINERS_MODULE, on_trainers, rep)
    return rep


def describe() -> dict:
    items, early = int(STATE["items"]), int(STATE["early_stopped"])
    return {"installed": bool(STATE["installed"]), "items": items, "early_stopped": early, "rolled_out": max(0, items - early),
            "reason": STATE.get("reason")}


def no_rollout_reason(desc: Optional[dict] = None) -> Optional[str]:
    """``zero_items`` (upstream predicted nothing: skip_existing / an empty selection), ``early_stop`` (every input of this process
    early-stopped before its roll-out), else None — also None when the census is not installed (the verdict stays fail-closed)."""
    d = desc or describe()
    if not d.get("installed"):
        return None
    if int(d.get("items") or 0) == 0:
        return "zero_items"
    if int(d.get("rolled_out") or 0) == 0:
        return "early_stop"
    return None


def reset() -> None:
    """Tests: zero the counters (the wrapper stays)."""
    STATE.update(items=0, early_stopped=0)
