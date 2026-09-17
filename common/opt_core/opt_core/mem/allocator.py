"""The device allocator under ``big``: its policy as levers (checked by read-back), its settings recorded, the torch peak counters
for the adapter — never numerics.

Contract (torch). Lever ``expandable_segments`` (family ``allocator``, ``bitwise``: the segment policy moves where bytes live, not
their values) puts ``expandable_segments:True`` into ``PYTORCH_CUDA_ALLOC_CONF`` for the kit process — through the environment when
torch has not initialised CUDA yet (the allocator reads the variable at its first use), through ``torch.cuda.memory._set_allocator_settings``
when it has (recorded as ``applied_via=runtime_api``); every other key of a pre-set value is kept. The export is CHECKED, not
trusted: the Applied carries :func:`expandable_in_force` as its effectiveness check — the record runs it at every unit boundary
until CUDA is up (pending before that) and reads the setting back from ``torch.cuda.memory._snapshot()["allocator_settings"]``; a
setting not in force, or one never readable by the exit gate, is a named fallback on the ``process`` unit (partial at the gate).
Measured on NVIDIA H100 80GB HBM3, torch 2.7.1+cu128: read-back True on both the env and the runtime-API path, outputs
sha256-equal to stock on every unit, a lower reserved peak. Preconditions, each a refusal by
name: ``conf`` — a pre-set ``expandable_segments:False`` is the kit's pin and is never overridden; ``graphs`` — CUDA-graph capture
in the composed line (``ctx.graphs``): this composer keeps the policy off graphed lines (torch >= 2.1 keeps graph private pools on their own
non-expandable segments — the policy then serves the non-graph heap only and a pool release does not shrink it; a kit that wants both
declares the composition through ``mem.torch_alloc.export(graphs_on=True, allow_with_graphs=True)`` in its mode table), so the line drops
its graph levers (``compose_big(drop=…)``) or the flag turns this lever off; ``torch`` — torch importable; ``cuda_state`` — CUDA
uninitialised or the runtime API present; ``environ`` — a writable ``ctx.environ`` when the environment path is the one
taken (the precondition is re-checked at apply time: the state acted on is the state read then). Lever ``cache_release`` (``allocator``, ``bitwise``) records the ``policy`` of
``torch.cuda.empty_cache`` calls — ``per_unit`` (default), ``per_stage`` or ``never`` — and the kit calls :func:`release` at its
points (``unit`` / ``stage``); each call is an event on the census. CUDA-graph private pools have no switch of their own: the policy
is recorded as ``cuda_graphs`` on the lever (False under the rule above). :func:`reset_peak` / :func:`counters` are the per-pass peak
counters (``max_memory_allocated`` / ``max_memory_reserved`` and the current values, in GiB) for the adapter that wants torch peaks
per pass: reset at the pass start, read after it. The peak instrument (``peak.py``, the launcher's sampled high-water mark)
is read-only and separate: it neither reads nor resets these counters.

Contract (JAX). :func:`jax_settings` records ``XLA_PYTHON_CLIENT_PREALLOCATE`` / ``_MEM_FRACTION`` / ``_ALLOCATOR`` (and ``XLA_FLAGS``)
as found; :func:`jax_export` is the one writer a JAX lever uses: an unset variable is exported and recorded, a variable set to the
wanted value is recorded as kept, a variable set to another value is a refusal naming it — a kit's pinned XLA variables are never
overridden; a read-only ``ctx.environ`` is a refusal naming ``environ``; an XLA backend already initialised
(``jax._src.xla_bridge.backends_are_initialized()``) is a refusal naming ``backend_initialized`` — the variables are read at backend
start, an export after it is a no-op. :func:`snapshot` is both records together, the ``allocator`` block of the applied record.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Mapping, MutableMapping, Optional

from . import torch_alloc as _talloc
from .registry import Applied, Ctx, Pending, Refusal, RefusalError, off_ref, refuse, register

TORCH_CONF = _talloc.ENV                                   # "PYTORCH_CUDA_ALLOC_CONF" — owned by opt_core.mem.torch_alloc
EXPANDABLE = "expandable_segments"
JAX_VARS = ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_ALLOCATOR", "XLA_FLAGS")
RELEASE_POLICIES = ("per_unit", "per_stage", "never")
GIB = float(1 << 30)


class _Nowhere(dict):
    """A write target that keeps nothing: the lever's runtime-API path under a read-only ``ctx.environ``."""

    def __setitem__(self, k, v):
        pass


_NOWHERE = _Nowhere()


# ------------------------------------------------------------------------------------------------------------ the conf string


parse_conf = _talloc.parse_conf                            # the ONE parser / formatter / state reader / read-back live in torch_alloc; served here too
format_conf = _talloc.format_conf
torch_state = _talloc.torch_state
allocator_settings = _talloc.allocator_settings


def expandable_in_force():
    """The effectiveness check of ``expandable_segments``: None when the allocator reports ``expandable_segments`` True; a
    :class:`Pending` while there is nothing to read yet (torch not imported, CUDA not initialised — the record re-checks at every
    unit boundary and finalises at the exit gate); else the reason (torch.cuda unavailable, no readable settings, the setting False —
    the export never reached the allocator)."""
    t = sys.modules.get("torch")
    if t is None:
        return Pending("torch not imported yet: the setting was never exercised")
    if not t.cuda.is_available():
        return "torch.cuda is unavailable: the setting was never exercised"
    if not t.cuda.is_initialized():
        return Pending("CUDA not initialised yet: the setting was never exercised")
    s = allocator_settings()
    if s is None or EXPANDABLE not in s:
        return "torch.cuda.memory._snapshot() reports no allocator_settings.expandable_segments (torch too old to read back)"
    return None if bool(s[EXPANDABLE]) else "allocator_settings.expandable_segments is False: the export did not reach the allocator"


def torch_settings(environ: Optional[Mapping[str, str]] = None) -> dict:
    """The torch allocator settings in force: the conf variable (raw and parsed) and the torch state."""
    environ = os.environ if environ is None else environ
    raw = environ.get(TORCH_CONF)
    try:
        parsed = parse_conf(raw)
    except ValueError as e:
        parsed = {"<malformed>": str(e)}
    return {"conf": raw, "parsed": parsed, "torch": torch_state()}


def jax_state() -> dict:
    """``{"imported", "version", "backends_initialized"}`` without importing jax (``sys.modules`` only); ``backends_initialized`` reads
    ``jax._src.xla_bridge.backends_are_initialized()`` (None when jax is absent or the function is)."""
    j = sys.modules.get("jax")
    if j is None:
        return {"imported": False, "version": None, "backends_initialized": None}
    init = None
    xb = sys.modules.get("jax._src.xla_bridge")
    fn = getattr(xb, "backends_are_initialized", None)
    if callable(fn):
        try:
            init = bool(fn())
        except Exception:  # noqa: BLE001
            init = None
    elif xb is not None:
        init = bool(getattr(xb, "_backends", None))                                     # older releases: the backend table itself
    return {"imported": True, "version": getattr(j, "__version__", None), "backends_initialized": init}


def jax_settings(environ: Optional[Mapping[str, str]] = None) -> dict:
    """The XLA allocator variables as found (None = unset) and the jax state; never written here."""
    environ = os.environ if environ is None else environ
    out = {v: environ.get(v) for v in JAX_VARS}
    out["jax"] = jax_state()
    return out


def snapshot(environ: Optional[Mapping[str, str]] = None, *, framework: str = "torch") -> dict:
    """The allocator settings as found: ``{"framework", "torch": torch_settings, "jax": jax_settings}`` — the ``found`` part of the
    record's ``allocator`` block (``{"found", "writes", "settings"}``: what was found, every variable written, the levers' values)."""
    return {"framework": framework, "torch": torch_settings(environ), "jax": jax_settings(environ)}


def _record_write(ctx: Ctx, lever: str, name: str, before, after, via: str) -> None:
    rec = ctx.record
    if rec is not None and isinstance(getattr(rec, "allocator", None), dict):
        rec.allocator.setdefault("writes", []).append({"lever": lever, "name": name, "before": before, "after": after, "via": via})


# ------------------------------------------------------------------------------------------------------------ expandable segments


def _expandable_applies(ctx: Ctx) -> Optional[Refusal]:
    lever = EXPANDABLE
    raw = ctx.environ.get(TORCH_CONF)
    try:
        conf = parse_conf(raw)
    except ValueError as e:
        return refuse(lever, "conf", f"{TORCH_CONF}={raw!r}: {e}")
    pre = conf.get(EXPANDABLE)
    if pre is not None and pre.strip().lower() != "true":
        return refuse(lever, "conf", f"{TORCH_CONF} pins {EXPANDABLE}:{pre} — the kit's own allocator setting is never overridden "
                                     f"(leave this lever out by name, {off_ref(EXPANDABLE)}, to keep it)",
                      conf=dict(conf))
    if ctx.graphs:
        return refuse(lever, "graphs", "CUDA-graph capture is in the composed line: this composer keeps the allocator policy off graphed lines "
                                       "(torch >= 2.1 keeps graph private pools on their own, non-expandable segments, so the policy serves the "
                                       "non-graph heap only and a pool release does not shrink it) — a kit that wants both declares the composition "
                                       "through mem.torch_alloc.export(graphs_on=True, allow_with_graphs=True) in its mode table; here: drop the "
                                       f"graph levers (compose_big(drop=…)) or set {off_ref(EXPANDABLE)}")
    st = torch_state()
    if not st["imported"] and importlib.util.find_spec("torch") is None:
        return refuse(lever, "torch", "torch is not importable in this process: the conf variable would reach no allocator")
    if st["imported"] and st["cuda_initialized"]:
        if not st["runtime_api"]:
            return refuse(lever, "cuda_state", "torch has initialised CUDA and has no torch.cuda.memory._set_allocator_settings: the "
                                               "conf variable cannot reach the allocator now — export it before the first CUDA call",
                          torch=st)
    elif not isinstance(ctx.environ, MutableMapping):
        return refuse(lever, "environ", "ctx.environ is read-only and CUDA is not initialised: the conf variable cannot be exported "
                                        "for the allocator to read (pass a writable mapping — os.environ)")
    return None


@register(EXPANDABLE, family="allocator", exact="bitwise",
          exact_reason="the allocator's segment policy changes where bytes live, never their values",
          applies=_expandable_applies, description="PYTORCH_CUDA_ALLOC_CONF expandable_segments:True for the kit process",
          preconditions=("conf", "graphs", "torch", "cuda_state", "environ"), settings=(), scope="process")
def expandable_segments(ctx: Ctx) -> Applied:
    """Apply (module contract): the environment path when CUDA is uninitialised, the runtime API otherwise; both recorded."""
    before = ctx.environ.get(TORCH_CONF)
    conf = parse_conf(before)
    conf[EXPANDABLE] = "True"
    after = format_conf(conf)
    st = torch_state()                                                                   # the state acted on, read here (not in applies)
    via = "runtime_api" if (st["imported"] and st["cuda_initialized"]) else "env"
    environ = ctx.environ
    if via == "runtime_api" and not st["runtime_api"]:
        raise RefusalError(refuse(EXPANDABLE, "cuda_state", "CUDA initialised between the precondition check and apply, and torch has "
                                                            "no torch.cuda.memory._set_allocator_settings", torch=st))
    if via == "env" and not isinstance(environ, MutableMapping):
        raise RefusalError(refuse(EXPANDABLE, "environ", "ctx.environ is read-only: the conf variable cannot be exported"))
    notes = []
    _talloc.write_conf(after, environ if isinstance(environ, MutableMapping) else _NOWHERE, via=via)   # the ONE writer (env, + runtime API when initialised)
    if via == "runtime_api":
        notes.append("CUDA was initialised before big applied: the setting reached the allocator through the runtime API")
    _record_write(ctx, EXPANDABLE, TORCH_CONF, before, after, via)

    def undo():
        if isinstance(environ, MutableMapping):
            if before is None:
                environ.pop(TORCH_CONF, None)
            else:
                environ[TORCH_CONF] = before
        if via == "runtime_api":
            _talloc.write_conf(before if before is not None else format_conf({EXPANDABLE: "False"}), _NOWHERE, via="runtime_api")

    return Applied(lever=EXPANDABLE, settings={EXPANDABLE: True, "conf": after, "applied_via": via, "cuda_graphs": bool(ctx.graphs)},
                   sites=(TORCH_CONF,), notes=notes, undo=undo, verify=expandable_in_force)


# ------------------------------------------------------------------------------------------------------------ cache release


def _cache_release_applies(ctx: Ctx) -> Optional[Refusal]:
    policy = ctx.setting("cache_release", "policy", "per_unit")
    if policy not in RELEASE_POLICIES:
        return refuse("cache_release", "policy", f"policy {policy!r} is not one of {RELEASE_POLICIES}")
    return None


@register("cache_release", family="allocator", exact="bitwise",
          exact_reason="releasing cached blocks to the driver changes no tensor",
          applies=_cache_release_applies, description="torch.cuda.empty_cache at the kit's points (policy per_unit|per_stage|never)",
          preconditions=("policy",), settings=("policy",))
def cache_release(ctx: Ctx) -> Applied:
    policy = ctx.setting("cache_release", "policy", "per_unit")
    return Applied(lever="cache_release", settings={"policy": policy}, sites=("torch.cuda.empty_cache",))


def release(ctx: Ctx, point: str, *, unit: Optional[str] = None) -> bool:
    """The kit's call at a release point (``"unit"`` at a unit's end, ``"stage"`` after a stage): ``torch.cuda.empty_cache()`` when
    the policy says so; the call (or the policy's skip) is an event on the census for lever ``cache_release``. Returns True when
    the cache was released."""
    rec = ctx.record
    policy = None
    if rec is not None:
        for a in rec.applied:
            if a.lever == "cache_release":
                policy = a.settings.get("policy")
    if policy is None:
        return False                                                      # the lever is not applied: nothing to record
    want = (policy == "per_unit" and point == "unit") or (policy == "per_stage" and point in ("unit", "stage"))
    if not want:
        rec.skip("cache_release", f"policy {policy}: no release at {point}", unit=unit)
        return False
    t = sys.modules.get("torch")
    if t is None or not t.cuda.is_available():
        rec.fallback("cache_release", "torch.cuda unavailable: nothing released", unit=unit)
        return False
    t.cuda.empty_cache()
    rec.mark("cache_release", unit=unit, detail=point)
    return True


# ------------------------------------------------------------------------------------------------------------ peak counters


def reset_peak(device=None) -> bool:
    """``torch.cuda.reset_peak_memory_stats`` for the device (the per-pass reset); False when torch / CUDA is unavailable."""
    t = sys.modules.get("torch")
    if t is None or not t.cuda.is_available():
        return False
    t.cuda.reset_peak_memory_stats(device)
    return True


def counters(device=None) -> dict:
    """``{"max_allocated_gib", "max_reserved_gib", "allocated_gib", "reserved_gib", "device"}`` from ``torch.cuda`` (None when
    unavailable — recorded as such, never as zero)."""
    t = sys.modules.get("torch")
    if t is None or not t.cuda.is_available():
        return {"max_allocated_gib": None, "max_reserved_gib": None, "allocated_gib": None, "reserved_gib": None, "device": None}
    dev = t.cuda.current_device() if device is None else device
    return {"max_allocated_gib": t.cuda.max_memory_allocated(dev) / GIB, "max_reserved_gib": t.cuda.max_memory_reserved(dev) / GIB,
            "allocated_gib": t.cuda.memory_allocated(dev) / GIB, "reserved_gib": t.cuda.memory_reserved(dev) / GIB, "device": str(dev)}


# ------------------------------------------------------------------------------------------------------------ JAX


def jax_export(ctx: Ctx, lever: str, **wanted) -> dict:
    """A JAX lever's one writer of XLA variables: for each ``NAME=value`` wanted — unset: exported (recorded as a write); set to the
    wanted value: kept (recorded); set to another value: :class:`RefusalError` naming ``xla.<NAME>`` (the kit's pin is never
    overridden). Returns ``{NAME: {"value", "state"}}`` for the Applied's settings."""
    out: dict = {}
    environ = ctx.environ
    st = jax_state()
    if st["backends_initialized"]:
        raise RefusalError(refuse(lever, "backend_initialized", "the XLA backend is initialised: an exported variable reaches no "
                                                                "allocator now — export before the first jax device call", jax=st))
    for name, value in wanted.items():
        value = str(value)
        pre = environ.get(name)
        if pre is None:
            if not isinstance(environ, MutableMapping):
                raise RefusalError(refuse(lever, "environ", f"ctx.environ is read-only: {name} cannot be exported"))
            environ[name] = value
            _record_write(ctx, lever, name, None, value, "env")
            out[name] = {"value": value, "state": "exported"}
        elif str(pre).strip().lower() == value.strip().lower():
            out[name] = {"value": pre, "state": "kept"}
        else:
            raise RefusalError(refuse(lever, f"xla.{name}", f"{name}={pre!r} is pinned by the kit and lever {lever} wants {value!r}: "
                                                            f"never overridden — change the kit's pin or turn the lever off by name",
                                      pinned=pre, wanted=value))
    return out
