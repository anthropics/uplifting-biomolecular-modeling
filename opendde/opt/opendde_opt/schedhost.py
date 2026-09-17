"""sched_host — the tree's own lever on the diffusion sampler's two per-step host round-trips (exact class: the same values reach the same
device arithmetic; only WHERE two host decisions are taken changes).

The stock sampler loop (`opendde/model/generator.py`, `sample_diffusion`) stalls the host on the GPU twice per denoiser step:
  * `generator.py:204`  `gamma = float(gamma0) if c_tau > gamma_min else 0` — `c_tau` is a 0-d element of the CUDA `noise_schedule`, so the
    `if` reads a CUDA comparison back to the host (a device synchronisation per step);
  * `model/utils.py:69`  `uniform_random_rotation(...).to(device)` inside `centre_random_augmentation` — the per-step random rotation is drawn
    on the host (scipy, the stock `numpy_rng`) and copied to the GPU with a BLOCKING pageable copy (torch waits for the stream).
Both make the host wait for the previous step's GPU work before it can prepare the next one, so no step is ever enqueued ahead; with the
denoiser replayed from a graph (`stepgraph`) they are the remaining host round-trips of a step.

What the lever does (two patches, both at import, `opt_core.autoload`):
  1. `InferenceNoiseScheduler.__call__` returns the stock schedule tensor viewed as `HostSchedule` — a `torch.Tensor` subclass that carries a
     host copy of the schedule taken ONCE per sampler call (one device-to-host read instead of one per step). Indexing / slicing / iteration
     keep the host copy alongside; the ONE operation answered from the host copy is `element > <python number>` on a 0-d element (the loop's
     `c_tau > gamma_min`), computed on the CPU copy in the schedule's own dtype — torch's own comparison semantics, the same boolean. Every
     other operation (the arithmetic on `c_tau_last`, `noise_schedule[0] * randn(...)`, `c_tau - t_hat`, ...) runs on the device tensor
     exactly as stock and returns plain tensors: the device arithmetic, its operands and its order are unchanged — bitwise by construction.
  2. `generator.centre_random_augmentation` (the name the sampler loop resolves): while ONE call runs on CUDA coordinates, the utils-module
     name `uniform_random_rotation` it calls is served by a wrapper that draws the rotation exactly as stock (same scipy call, same
     `numpy_rng`, same float32 host tensor) and moves it to the coordinates' device with a NON-BLOCKING copy (the driver stages a pageable
     source synchronously, so the values are those of the host tensor; the stream is not waited for). The stock `.to(device)` that follows is
     then a no-op. Same values, same dtype, same device op order.
A schedule that is not a 1-D CUDA tensor, or coordinates not on CUDA, pass through the stock code (counted as `aside`, never a failure).

Evidence: `STATS` — `sched_calls` (schedules wrapped), `host_cmp` (per-step comparisons answered on the host), `rot_copies` (per-step
rotations moved without a stream wait), `aside` words; the kit's LEVER row prints them (`report.lever_evidence`), `ran.COUNTERS["sched_host"]`
= host_cmp + rot_copies. Left out by name with `MODEL_OPT_LEVERS_OFF=sched_host` (modes.LEVER_SWITCHES).
"""
from __future__ import annotations

import json
import sys
import threading
from typing import Optional

LEVER = "sched_host"
TAG = "opendde-opt"
SCHED_MODULE, SCHED_ATTR = "opendde.model.generator", "InferenceNoiseScheduler.__call__"
AUG_MODULE, AUG_ATTR = "opendde.model.generator", "centre_random_augmentation"      # the name the sampler loop resolves (imported from model.utils)
UTILS_MODULE, ROT_ATTR = "opendde.model.utils", "uniform_random_rotation"          # the name centre_random_augmentation's body resolves

_STATS0 = {"installed": False, "patches": 0, "sched_calls": 0, "host_cmp": 0, "rot_copies": 0, "aug_calls": 0, "aside": {}, "failures": 0,
           "failure": None, "wrap_cpu": False}
STATS = json.loads(json.dumps(_STATS0))
_PATCH = {"sched": None, "aug": None}
_TLS = threading.local()


class ActivationError(RuntimeError):
    pass


def _aside(word: str) -> None:
    STATS["aside"][word] = STATS["aside"].get(word, 0) + 1


# ---------------------------------------------------------------------------------------------------------------- the schedule view
def _schedule_class():
    """The `torch.Tensor` subclass, built on first use (torch is imported lazily by this module)."""
    cls = _CLS.get("cls")
    if cls is not None:
        return cls
    import torch

    class HostSchedule(torch.Tensor):
        """The stock noise schedule (same storage, same device, same dtype) with a host copy for the loop's per-step `> gamma_min` decision."""

        @staticmethod
        def __new__(cls, data, shadow):
            t = torch.Tensor._make_subclass(cls, data, bool(data.requires_grad))
            t._host = shadow
            return t

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func is torch.Tensor.__getitem__ and len(args) == 2 and isinstance(args[0], cls) and _host_indexable(args[1]):
                with torch._C.DisableTorchFunctionSubclass():
                    out = func(*args, **kwargs)
                try:
                    return cls(out, args[0]._host[args[1]])
                except Exception:  # noqa: BLE001
                    return out
            if func in (torch.Tensor.unbind, torch.unbind) and args and isinstance(args[0], cls):
                dim = kwargs.get("dim", args[1] if len(args) > 1 else 0)
                with torch._C.DisableTorchFunctionSubclass():
                    outs = func(*args, **kwargs)
                if dim in (0, -args[0].dim()) or args[0].dim() == 1:
                    hs = args[0]._host.unbind(0)
                    if len(hs) == len(outs):
                        return tuple(cls(o, h) for o, h in zip(outs, hs))
                return outs
            if func in (torch.Tensor.__gt__, torch.Tensor.gt, torch.gt) and len(args) == 2 and isinstance(args[0], cls) \
                    and args[0].dim() == 0 and isinstance(args[1], (int, float)) and not isinstance(args[1], bool) and not kwargs:
                STATS["host_cmp"] += 1                                            # the loop's `c_tau > gamma_min`: the host copy, the schedule's dtype, torch's semantics
                return bool(args[0]._host > args[1])
            with torch._C.DisableTorchFunctionSubclass():                         # everything else: the stock op on the device tensor, a plain result
                return func(*args, **kwargs)

    _CLS["cls"] = HostSchedule
    return HostSchedule


_CLS: dict = {}


def _host_indexable(idx) -> bool:
    return isinstance(idx, (int, slice)) and not isinstance(idx, bool)


def host_schedule(sched):
    """`sched` viewed as HostSchedule (its host copy read here, once), or `sched` itself when it is not a 1-D floating tensor on CUDA
    (`wrap_cpu` admits CPU tensors: the unit tests)."""
    import torch
    if not torch.is_tensor(sched) or sched.dim() != 1 or not sched.is_floating_point():
        _aside("schedule_not_1d_float")
        return sched
    if not sched.is_cuda and not STATS["wrap_cpu"]:
        _aside("schedule_not_cuda")
        return sched
    cls = _schedule_class()
    if isinstance(sched, cls):
        return sched
    shadow = sched.detach().to("cpu")                                             # ONE device-to-host read per sampler call
    out = cls(sched.detach(), shadow)
    STATS["sched_calls"] += 1
    return out


def make_sched_wrapper(orig):
    def scheduler_call(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        try:
            return host_schedule(out)
        except Exception as e:  # noqa: BLE001                                     # a torch without the subclass hooks: the stock tensor, named
            STATS["failures"] += 1
            STATS["failure"] = f"host_schedule_failed:{type(e).__name__}"
            return out
    scheduler_call._orig = orig
    scheduler_call._sched_host = True
    return scheduler_call


# ---------------------------------------------------------------------------------------------------------------- the rotation copy
def make_aug_wrapper(orig):
    def centre_random_augmentation(*args, **kwargs):
        import torch
        x = kwargs.get("x_input_coords", args[0] if args else None)
        if not (torch.is_tensor(x) and x.is_cuda):
            _aside("coords_not_cuda")
            return orig(*args, **kwargs)
        utils = sys.modules.get(UTILS_MODULE)
        stock_rot = getattr(utils, ROT_ATTR, None) if utils is not None else None
        if stock_rot is None or getattr(_TLS, "depth", 0) > 0:
            return orig(*args, **kwargs)
        dev = x.device

        def uniform_random_rotation(N_sample=1, numpy_rng=None):
            r = stock_rot(N_sample=N_sample, numpy_rng=numpy_rng)               # the stock host draw (scipy Rotation.random, the stock numpy_rng)
            if torch.is_tensor(r) and not r.is_cuda:
                STATS["rot_copies"] += 1
                return r.to(dev, non_blocking=True)                               # pageable source: staged synchronously by the driver, no stream wait
            return r

        STATS["aug_calls"] += 1
        _TLS.depth = getattr(_TLS, "depth", 0) + 1
        setattr(utils, ROT_ATTR, uniform_random_rotation)
        try:
            return orig(*args, **kwargs)
        finally:
            setattr(utils, ROT_ATTR, stock_rot)
            _TLS.depth -= 1
    centre_random_augmentation._orig = orig
    centre_random_augmentation._sched_host = True
    return centre_random_augmentation


# ---------------------------------------------------------------------------------------------------------------- install / census
def _sync() -> None:
    ps = [p for p in _PATCH.values() if p is not None]
    STATS["patches"] = sum(1 for p in ps if getattr(p, "state", None) == "installed")
    STATS["installed"] = bool(ps) and STATS["patches"] == len(ps)


def install() -> None:
    """Patch the scheduler and the sampler's augmentation name (now, or at the generator module's import). Idempotent."""
    from opt_core import autoload
    if _PATCH["sched"] is None:
        try:
            _PATCH["sched"] = autoload.patch_attr_at_import(SCHED_MODULE, SCHED_ATTR, make_sched_wrapper, tag=TAG, name=LEVER)
            _PATCH["aug"] = autoload.patch_attr_at_import(AUG_MODULE, AUG_ATTR, make_aug_wrapper, tag=TAG, name=LEVER)
        except autoload.PatchError as e:
            raise ActivationError(str(e)) from None
    _sync()


def armed() -> bool:
    return any(p is not None for p in _PATCH.values())


def _reset() -> None:
    """Test support: counters back to import time (the patches stay: one AttrPatch per site per process)."""
    STATS.clear(); STATS.update(json.loads(json.dumps(_STATS0)))


def served() -> int:
    return int(STATS["host_cmp"]) + int(STATS["rot_copies"])


def fallbacks(planned) -> list:
    """Failures by name (a schedule the view could not be built for); step-asides (CPU schedule / CPU coordinates) are not failures."""
    if LEVER not in planned or not STATS["failure"]:
        return []
    return [f"{LEVER}:{STATS['failure']}"]


def kit_stats() -> dict:
    _sync()
    return json.loads(json.dumps(STATS, default=str))
