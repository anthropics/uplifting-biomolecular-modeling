"""Lever ``hostlean``: host-side work inside the item's forward clock that no output of the fold consumes.

Upstream's ``validation_step`` (``rf3/trainers/rf3.py``: ``RF3Trainer`` and ``RF3TrainerWithConfidence``) runs, after the model forward and
before the metrics, the two symmetry resolutions of the loss module (``rf3/loss/af3_losses.py``: ``SubunitSymmetryResolution``,
``ResidueSymmetryResolution``). They permute the NATIVE coordinates against the prediction — their only writes are
``metrics_extra_info["X_gt_L"]`` and ``["crd_mask_L"]`` (the prediction is read, never written) — with Python loops over entities,
chains, automorphic residues and diffusion samples: thousands of small kernels, boolean-index host syncs and device-to-host copies per
item, a visible share of an inference forward. The permuted natives reach only metrics that read ground truth
(``extra_info.X_gt_L`` / ``crd_mask_L``, ``ground_truth_atom_array_stack``: lddt / distogram / chirality metrics); the inference
engine's metric set (ptm, iptm, count_clashing_chains) reads none of them, so there the resolutions are dead computations.

The rule, decided per ``validation_step`` call from the trainer's OWN metric table and logged: no configured metric declares a
ground-truth input (``Metric.kwargs_to_compute_args``) → both resolutions return their input unchanged (counted ``skipped``); any
metric that reads ground truth, or whose inputs cannot be read (no declaration table, its own ``compute_from_kwargs``) → the
resolutions run as stock (counted ``kept:<metric>``) — a named step-aside, never silent. Output-identical by construction (dead-code
elimination): every value a consumer reads is computed by the statements stock runs. The training step's resolutions (the loss path)
are untouched: the skip is armed only inside ``validation_step``.

``enable()`` patches the two trainer classes' ``validation_step`` and the two resolution classes' ``forward`` (class level, so it
may run before or after the engine builds its trainer); ``describe()`` / ``lever_evidence()`` feed the exit tally.
"""
import sys
from collections import Counter
from typing import Callable, Dict, List, Optional

NAME = "hostlean"
TRAINERS_MODULE = "rf3.trainers.rf3"                          # the trigger: both trainer classes live here (it imports rf3.loss.af3_losses itself)

GT_STACK_ARGS = ("ground_truth_atom_array_stack",)          # MetricManager kwargs that ARE ground truth
GT_EXTRA_KEYS = ("X_gt_L", "crd_mask_L")                    # the extra_info keys the resolutions write
TRAINER_CLASSES = ("RF3Trainer", "RF3TrainerWithConfidence")
RESOLUTION_CLASSES = ("SubunitSymmetryResolution", "ResidueSymmetryResolution")

STATE: Dict[str, object] = {"on": False, "installed": False, "reason": None, "armed": False}
CENSUS: Counter = Counter()
_ACTIVE = {"skip": False, "depth": 0}
_ORIG: Dict[str, object] = {}
_CONSUMERS: Dict[int, List[str]] = {}                        # id(MetricManager) -> the metrics that read ground truth


class HostleanRefused(RuntimeError):
    """The lever cannot be installed (upstream classes absent or reshaped)."""


def _path(key) -> tuple:
    if isinstance(key, str):
        return (key,)
    try:
        return tuple(key)
    except TypeError:
        return (repr(key),)


def reads_ground_truth(metric) -> Optional[str]:
    """Why this metric reads ground truth (a word), or None when its declared inputs provably do not."""
    try:
        from foundry.metrics.metric import Metric
    except Exception:                                        # pragma: no cover - the CPU tests pass a base explicitly
        Metric = None
    if Metric is not None and type(metric).compute_from_kwargs is not Metric.compute_from_kwargs:
        return "own compute_from_kwargs"
    try:
        table = metric.kwargs_to_compute_args
    except Exception as e:
        return f"inputs unreadable ({type(e).__name__})"
    if not table:
        return "no input table (receives every argument)"
    for key in table.values():
        p = _path(key)
        if p[0] in GT_STACK_ARGS:
            return f"reads {p[0]}"
        if p[0] == "extra_info" and (len(p) == 1 or p[1] in GT_EXTRA_KEYS):
            return "reads extra_info" + (f".{p[1]}" if len(p) > 1 else " (whole)")
    return None


def gt_consumers(metric_manager) -> List[str]:
    """Names (with the reason) of the manager's metrics that read ground truth; [] when none does; a manager without a readable
    ``metrics`` table counts as one unknown consumer (conservative)."""
    if metric_manager is None:
        return []
    metrics = getattr(metric_manager, "metrics", None)
    if not isinstance(metrics, dict):
        return [f"<manager {type(metric_manager).__name__}: no metrics table>"]
    out = []
    for name, m in metrics.items():
        why = reads_ground_truth(m)
        if why:
            out.append(f"{name}({why})")
    return out


def consumers_of(trainer) -> List[str]:
    mm = getattr(trainer, "metrics", None)
    key = id(mm)
    if key not in _CONSUMERS:
        _CONSUMERS[key] = gt_consumers(mm)
    return _CONSUMERS[key]


def _wrap_validation_step(orig):
    def validation_step(self, *args, **kwargs):
        if _ACTIVE["depth"] > 0:                            # a nested call (a subclass step calling its base): the outer call decided and counted
            return orig(self, *args, **kwargs)
        consumers = consumers_of(self) if STATE["on"] else ["<lever off>"]
        skip = not consumers
        CENSUS["steps:skip" if skip else "steps:kept:" + ",".join(consumers)] += 1
        _ACTIVE["skip"] = skip
        _ACTIVE["depth"] += 1
        try:
            return orig(self, *args, **kwargs)
        finally:
            _ACTIVE["depth"] -= 1
            _ACTIVE["skip"] = False
    validation_step.__wrapped__ = orig
    validation_step.__name__ = "validation_step_hostlean"
    return validation_step


def _wrap_resolution(orig, name):
    def forward(self, network_output, loss_input, *rest, **kwargs):
        if _ACTIVE["skip"]:
            CENSUS[f"skipped:{name}"] += 1
            return loss_input                               # what stock returns: the same dict (stock mutates and returns it)
        CENSUS[f"resolved:{name}"] += 1
        return orig(self, network_output, loss_input, *rest, **kwargs)
    forward.__wrapped__ = orig
    forward.__name__ = f"{name}_forward_hostlean"
    return forward


def enable(trainers=None, losses=None) -> dict:
    """Install the lever on upstream's classes (``rf3.trainers.rf3`` / ``rf3.loss.af3_losses`` unless given). Idempotent."""
    if trainers is None:
        import rf3.trainers.rf3 as trainers
    if losses is None:
        import rf3.loss.af3_losses as losses
    missing = [c for c in TRAINER_CLASSES if not hasattr(getattr(trainers, c, None), "validation_step")] + \
              [c for c in RESOLUTION_CLASSES if not hasattr(getattr(losses, c, None), "forward")]
    if missing:
        STATE.update({"on": False, "installed": False, "reason": f"upstream classes absent: {','.join(missing)}"})
        raise HostleanRefused(STATE["reason"])
    for c in TRAINER_CLASSES:
        cls = getattr(trainers, c)
        vs = cls.__dict__.get("validation_step")            # each class defines its own; wrap each once
        if vs is not None and getattr(vs, "__name__", "") != "validation_step_hostlean":
            _ORIG[f"{c}.validation_step"] = vs
            cls.validation_step = _wrap_validation_step(vs)
    for c in RESOLUTION_CLASSES:
        cls = getattr(losses, c)
        fw = cls.__dict__.get("forward")
        if fw is not None and not getattr(fw, "__name__", "").endswith("_forward_hostlean"):
            _ORIG[f"{c}.forward"] = fw
            cls.forward = _wrap_resolution(fw, c)
    STATE.update({"on": True, "installed": True, "reason": None})
    return describe()


def arm(rep: dict, install_watch: Callable) -> dict:
    """Activation-time hook: when the row names ``hostlean``, patch the trainer / resolution classes as soon as ``rf3.trainers.rf3``
    executes (``install_watch(trigger, callback, rep)``: the kit's one import-watch installer, chained with any other lever's watch on the
    same module); a module imported already is patched at once. Records ``rep["hostlean"]`` (:func:`describe`); a reshaped upstream is a
    named refusal on the record (``on: False, reason``), never a raise out of the hook."""
    if NAME not in (rep.get("levers") or []) or STATE.get("armed"):
        return rep
    STATE["armed"] = True

    def on_trainers(module):
        try:
            rep[NAME] = enable(trainers=module)
        except HostleanRefused as e:
            rep[NAME] = {"on": False, "installed": False, "reason": str(e), "census": None}
        if rep[NAME].get("on"):
            rep["levers_applied"] = list(dict.fromkeys(list(rep.get("levers_applied") or []) + [NAME]))

    if TRAINERS_MODULE in sys.modules:
        on_trainers(sys.modules[TRAINERS_MODULE])
    else:
        install_watch(TRAINERS_MODULE, on_trainers, rep)
    return rep


def lever_tokens(desc: Optional[dict] = None) -> List[tuple]:
    """(key, value) tokens for the LEVER line: skipped / resolved / steps_kept (+ kept=<reasons> when a metric keeps the resolutions)."""
    d = desc or describe()
    c = d.get("census") or census()
    out = [("skipped", c["skipped"]), ("resolved", c["resolved"]), ("steps_kept", c["steps_kept"])]
    if c["kept_reasons"]:
        out.append(("kept", "|".join(c["kept_reasons"])))
    return out


def disable(trainers=None, losses=None) -> None:
    """Restore upstream's functions (tests; an engine never calls it)."""
    if trainers is None:
        import rf3.trainers.rf3 as trainers
    if losses is None:
        import rf3.loss.af3_losses as losses
    for k, fn in list(_ORIG.items()):
        cname, attr = k.split(".")
        mod = trainers if cname in TRAINER_CLASSES else losses
        setattr(getattr(mod, cname), attr, fn)
        del _ORIG[k]
    STATE.update({"on": False, "installed": False})
    _CONSUMERS.clear()


def census() -> dict:
    c = dict(CENSUS)
    return {"skipped": sum(v for k, v in c.items() if k.startswith("skipped:")), "resolved": sum(v for k, v in c.items() if k.startswith("resolved:")),
            "steps_skip": c.get("steps:skip", 0), "steps_kept": sum(v for k, v in c.items() if k.startswith("steps:kept:")),
            "kept_reasons": sorted(k[len("steps:kept:"):] for k in c if k.startswith("steps:kept:")), "by_key": c}


def describe() -> dict:
    return {"on": STATE["on"], "installed": STATE["installed"], "reason": STATE["reason"],
            "rule": "symmetry resolution skipped when no configured metric reads ground truth (else kept, by name)",
            "census": census() if STATE["installed"] else None}


def lever_evidence() -> list:
    c = census()
    return [("skipped", c["skipped"]), ("resolved", c["resolved"]), ("steps_kept", c["steps_kept"])] + ([("kept", "|".join(c["kept_reasons"]))] if c["kept_reasons"] else [])
