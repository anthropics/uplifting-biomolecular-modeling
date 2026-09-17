"""The sampler hook point of a diffusion roll-out (AF3 Alg. 18): an externally supplied sampler object may see / replace the initial
noise, the per-step noise and the coordinates after every denoising update. The roll-out LOOP is the engine's (its adapter calls
:func:`call_init_noise` / :func:`call_project` at the two statements); this module fixes the protocol so one hook client works in every
engine and at every ``--n_gpu`` (coordinates and noise are REPLICATED under row sharding: the hook is called identically on every rank,
must be deterministic, and needs no communication). Default (no spec): no hook object, the engine's loop untouched.

Protocol v1:
    factory(info) -> hook        called once per roll-out on every rank; ``info`` = :func:`build_info` (n_token, n_atom, atom_to_token
                                 [N_atom] long, asym_id / entity_id / sym_id [N_token], num_steps, num_samples, seed, + informational extras
                                 such as mode / atom_mask / token_mask)
    hook.init_noise(noise, ctx)  optional; called on the INITIAL sample ``x = c_0 * randn(...)`` (``ctx['step'] == -1``) and, when
                                 ``getattr(hook, 'per_step_noise', False)``, on every per-step noise tensor before ``x_noisy = x + noise``
    hook.project(x, ctx)         optional; called after EACH update ``x = x_noisy + step_scale * (c_tau - t_hat) * delta``; the last call
                                 (``ctx['step'] == num_steps - 1``) returns the final coordinates
    ctx                          ``{'step': int, 't_hat': float, 'sigma_next': float, 'gamma': float, 'aug_rot': Tensor|None,
                                 'aug_trans': Tensor|None, 'aug_center': Tensor|None}`` — the engine's loop variables of this step (the random
                                 augmentation applied at the top of the step, so a hook can track a fixed frame)
    returns                      a tensor of identical shape / dtype / device (:func:`checked`); padded atoms left alone

API:
    ENV                                   ``ROWPAIR_SAMPLER_HOOK`` — the default spec variable (``'<module>:<factory>'``); an adapter may read its own
    spec_from_env(env) / configured(env)  the spec string ('' when unset) / bool
    make(spec, info)                      import the factory and build the hook (None for an empty spec; a malformed spec is a ValueError by name)
    build_info(...)                       the info dict from explicit tensors / ints (engine-free; extras pass through)
    call_init_noise(hook, noise, ctx, per_step) / call_project(hook, x, ctx)
                                          the two call sites: absent hook or absent method -> the input unchanged; present -> :func:`checked` result
    Equality / RecenterExample            reference hooks (equality with a call log; re-centre each sample on its real atoms' centroid)
"""
from __future__ import annotations

import importlib
import os
from typing import Any, Callable, Dict, Optional

__all__ = ["ENV", "PROTOCOL_VERSION", "spec_from_env", "configured", "make", "build_info", "checked", "call_init_noise", "call_project",
           "Identity", "RecenterExample"]

ENV = "ROWPAIR_SAMPLER_HOOK"
PROTOCOL_VERSION = 1
CTX_KEYS = ("step", "t_hat", "sigma_next", "gamma", "aug_rot", "aug_trans", "aug_center")


def spec_from_env(env: str = ENV) -> str:
    return os.environ.get(env, "").strip()


def configured(env: str = ENV) -> bool:
    return bool(spec_from_env(env))


def make(spec: Optional[str], info: Dict[str, Any]):
    """``factory(info)`` for ``spec = '<module>:<factory>'`` (module importable on every rank); None when ``spec`` is empty."""
    spec = (spec or "").strip()
    if not spec:
        return None
    mod, sep, name = spec.partition(":")
    if not sep or not mod or not name:
        raise ValueError(f"sampler hook spec must be '<module>:<factory>', got {spec!r}")
    factory: Callable = getattr(importlib.import_module(mod), name)
    return factory(info)


def build_info(*, n_token: int, n_atom: int, atom_to_token, asym_id, entity_id, sym_id, num_steps: int, num_samples: int, seed: int,
               **extras) -> Dict[str, Any]:
    """The factory's ``info``: the fixed keys of protocol v1 plus informational ``extras`` (mode, atom_mask, token_mask, ...)."""
    info = dict(n_token=int(n_token), n_atom=int(n_atom), atom_to_token=atom_to_token, asym_id=asym_id, entity_id=entity_id, sym_id=sym_id,
                num_steps=int(num_steps), num_samples=int(num_samples), seed=int(seed), protocol=PROTOCOL_VERSION)
    info.update(extras)
    return info


def checked(y, x, what: str):
    """``y`` if it is a tensor shaped / typed / placed like ``x``, else TypeError naming the hook method."""
    ok = hasattr(y, "shape") and hasattr(y, "dtype") and hasattr(y, "device")
    if not ok or tuple(y.shape) != tuple(x.shape) or y.dtype != x.dtype or y.device != x.device:
        got = (tuple(y.shape), str(y.dtype), str(y.device)) if ok else type(y).__name__
        raise TypeError(f"sampler hook {what}() must return a tensor like its input {tuple(x.shape)} {x.dtype} {x.device}; got {got}")
    return y


def call_init_noise(hook, noise, ctx: Dict[str, Any], *, per_step: bool = False):
    """The noise statement's hook call: the initial noise always (``per_step=False``), a per-step noise only when the hook asks
    (``hook.per_step_noise``). No hook / no method -> ``noise``."""
    if hook is None:
        return noise
    if per_step and not getattr(hook, "per_step_noise", False):
        return noise
    fn = getattr(hook, "init_noise", None)
    return noise if fn is None else checked(fn(noise, ctx), noise, "init_noise")


def call_project(hook, x, ctx: Dict[str, Any]):
    """The update statement's hook call. No hook / no method -> ``x``."""
    fn = getattr(hook, "project", None) if hook is not None else None
    return x if fn is None else checked(fn(x, ctx), x, "project")


# ----------------------------------------------------------------------------------------------------------------- reference hooks
class Identity(object):
    """Equality hook with a call log (``calls``: [(method, step), ...]); ``per_step_noise = True`` so the per-step call site is exercised."""
    per_step_noise = True

    def __init__(self, info):
        self.info = info
        self.calls = []

    def init_noise(self, noise, ctx):
        self.calls.append(("init_noise", int(ctx["step"])))
        return noise

    def project(self, x, ctx):
        self.calls.append(("project", int(ctx["step"])))
        return x


class RecenterExample(Identity):
    """Re-centres each sample on the centroid of its real atoms (``info['atom_mask']``) after every update — deterministic, rank-identical,
    non-trivial (a test that the hook is invoked identically everywhere)."""
    per_step_noise = False

    def project(self, x, ctx):
        self.calls.append(("project", int(ctx["step"])))
        m = self.info["atom_mask"].to(x)
        m = m.reshape((1,) * (x.dim() - 2) + (-1, 1))
        centroid = (x * m).sum(dim=-2, keepdim=True) / m.sum(dim=-2, keepdim=True).clamp(min=1)
        return x - centroid * (m > 0).to(x)
