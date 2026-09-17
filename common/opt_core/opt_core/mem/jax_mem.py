"""Contract. Memory levers of a JAX/XLA folding engine as config and environment transforms — engine-free levers of the ``big``
registry (family ``jax``, ``frameworks=("jax",)``), declared with ``@register`` at import and applied by name through
:func:`opt_core.mem.apply`: the sub-batch knob, the compile bucket policy, the XLA environment (recorded, never overridden), the
client memory fraction, and the flash triangle-attention feasibility verdict. Hooks come from ``ctx.hooks[<lever>]``
(:data:`HOOKS`), values through ``ctx.setting`` (the kit's ``ctx.settings[<lever>]``); a precondition
that is absent is a ``Refusal`` naming it; a configuration that would be stock's own (a sub-batch value equal to stock's, a bucket
policy without a probe bucket) refuses by name — never a phantom change, never a narrowed label; every ``Applied`` carries its declared
label and the sites wired (a narrower label is the engine's equality row's to write, with its record id as ``narrowed_by``).
Scopes: ``subbatch`` is unit scope (the adapter marks it on every item whose sub-batched op ran, inside the item's unit);
``bucket_policy``, ``xla_env`` and ``mem_fraction`` act once per process (``scope="process"``). The one writer of XLA variables is
:func:`opt_core.mem.allocator.jax_export` (every write on the record's ``allocator`` block; a pinned different value refuses by
name ``xla.<NAME>``); nothing here writes the environment otherwise. No JAX is imported anywhere: the transforms are pure Python
over the engine's config object and environment.

Levers (name · declared exact · settings · hooks):

* ``subbatch`` · measured · ``value`` (an int row count, or a per-shape spec as JSON ``[[max_tokens|null, value], ...]``) · hooks
  ``config``, ``path`` — the sub-batch knob at the engine's own config path (a per-chunk row count of a sub-batched loop, or a
  per-shape spec such as a pair-transition shard table ``((max_tokens, value), ..., (None, value))``): :func:`set_config_path`
  sets the value and records stock's; the path must exist (:func:`get_config_path`: a knob the config does not carry is never
  invented) and be settable (:func:`settable`: a frozen dataclass or a locked mapping refuses by name); :func:`check_shape_spec`
  validates every row of a spec. :func:`shape_spec_value` resolves a spec at a token count for the record (informational: the
  engine resolves its own spec). A value equal to stock's refuses by name (nothing to apply). A sub-batch splits a batched op into
  chunks whose results are concatenated; the per-chunk kernels may differ from the full-batch ones under XLA: ``measured`` — the
  engine's equality row narrows with its id. :func:`mark_subbatch` is the adapter's per-item mark.
* ``bucket_policy`` · measured · ``probe_bucket`` (required) · hook ``buckets`` — the compile-bucket PROBE: :func:`resolve_bucket`
  maps a token count onto the engine's shipped table (the smallest bucket ``>= tokens``) and names the case — ``shipped`` (in the
  table), ``probe`` (the named ``probe_bucket`` beyond the table: a probe setting, never a silent extension), ``exact``
  (beyond the table with no probe bucket: the engine compiles the exact size). The lever is the probe form: without a probe bucket
  it refuses by name (the shipped table is the engine's own choice — nothing to apply, a lever changes nothing silently); with one,
  every item the adapter resolves through :func:`bucket_for` marks the record and a ``probe`` case is a named note. A probe bucket
  pads where stock compiles the exact size; the reductions the masked tokens sit in change shape: ``measured``.
* ``xla_env`` · bit-exact · no settings · hooks ``pinned``, ``requested`` (optional) — the XLA / JAX environment RECORDED, never
  overridden: :func:`xla_env_record` snapshots every ``XLA_*``, ``JAX_*``, ``TF_*``, ``NCCL_*`` and device variable of the
  process (the allocator block records the four allocator variables; this is the wider snapshot); :func:`xla_env_check` compares
  the live environment and a request against the engine's pins (its pinned environment): a requested value that differs
  from a pin refuses by name ``xla.<NAME>``; a live value that differs from a pin is a named note on the record (the engine's own
  pins gate decides whether the run proceeds).
* ``mem_fraction`` · bit-exact · ``fraction``, ``preallocate`` (0|1, optional), ``unified_memory`` (0|1) · hook ``pinned`` —
  the client memory fraction (and ``XLA_PYTHON_CLIENT_PREALLOCATE`` when ``preallocate`` is given) exported through
  ``allocator.jax_export``: unset → exported and recorded, equal → kept, a live different value → the primitive's refusal
  ``xla.<NAME>``; a pinned different value refuses first (``pins.<NAME>``); a fraction above 1 needs ``unified_memory=1`` (the
  adapter set the engine's unified-memory switch itself). The fraction has TWO variable names the client reads in order
  (:data:`opt_core.mem.peak.MEM_FRACTION_PRECEDENCE`: ``XLA_CLIENT_MEM_FRACTION``, then ``XLA_PYTHON_CLIENT_MEM_FRACTION``): the
  lever compares and writes under the name already present in the environment, else the name the kit pins, else
  ``XLA_PYTHON_CLIENT_MEM_FRACTION`` (:func:`mem_fraction_name`) — so it never plants the second name beside the first; both names
  already present refuses by name ``xla.MEM_FRACTION`` (the CUDA plugin does not initialise with both). An allocator-size lever: ``bitwise``.
* ``flash_triattn_jax`` · measured · no settings · hooks ``attention_family``, ``flash_implementation`` — feasibility only in this
  no flash (memory-transient) triangle-attention kernel for a JAX engine is in the core, so the lever refuses by design
  (``applies`` always returns a ``Refusal`` naming ``kernel``) and no engine can switch it on before a kernel exists.
  :func:`flash_triattn_feasibility` records what the adapter states about its engine — ``attention_family`` (a free label) and
  ``flash_implementation`` (the engine's own attention setting; ``xla`` = the un-fused form, anything else = a fused implementation
  the stock already runs) — as the refusal's details.

The knob names, shard specs, bucket tables and pinned variables are the engine's own, read from
its ``STOCK.md`` and stock configs and passed in by the adapter; the flash verdict is the feasibility statement of
:data:`FLASH_VERDICT`.

This module imports only the standard library and the registry at module level.
"""
from __future__ import annotations

import json
import os
from typing import Any, Mapping, Optional, Sequence

from . import allocator, peak
from .ckpt import parse_bool
from .registry import Applied, Ctx, Refusal, RefusalError, off_ref, refuse, register, setting_ref

# The XLA client table is ONE: opt_core.mem.peak's (the instrument that must also run standalone carries it; every other module reads it
# from there) — the named client settings, the XLA_/JAX_ record prefixes, the two MEM_FRACTION names in the order the client library
# reads them. This module adds only the device / communication prefixes its wider record carries beside them.
RECORD_EXTRA_PREFIXES = ("TF_", "NCCL_", "CUDA_VISIBLE_DEVICES", "TPU_")
XLA_PREFIXES = tuple(peak.XLA_PREFIXES) + RECORD_EXTRA_PREFIXES
MEM_FRACTION_VAR = "XLA_PYTHON_CLIENT_MEM_FRACTION"      # the name this lever writes when neither MEM_FRACTION name is present or pinned
PREALLOCATE_VAR = "XLA_PYTHON_CLIENT_PREALLOCATE"
for _name in (MEM_FRACTION_VAR, PREALLOCATE_VAR):
    if _name not in peak.XLA_NAMES:                          # the table moved without this module: refuse at import, by name
        raise ImportError(f"opt_core.mem.jax_mem: {_name} is not in opt_core.mem.peak.XLA_NAMES {peak.XLA_NAMES}")
if MEM_FRACTION_VAR not in peak.MEM_FRACTION_PRECEDENCE:
    raise ImportError(f"opt_core.mem.jax_mem: {MEM_FRACTION_VAR} is not in opt_core.mem.peak.MEM_FRACTION_PRECEDENCE {peak.MEM_FRACTION_PRECEDENCE}")
JAX = ("jax",)

HOOKS: dict = {
    "subbatch": ("config", "path"),
    "bucket_policy": ("buckets",),
    "xla_env": ("pinned", "requested"),
    "mem_fraction": ("pinned",),
    "flash_triattn_jax": ("attention_family", "flash_implementation"),
}


class JaxMemError(RuntimeError):
    """A config or environment transform refused at run time (the message names the knob or the variable)."""


# ------------------------------------------------------------------------------------------------------------ config transforms


def _get_child(obj: Any, key: str):
    if isinstance(obj, Mapping):
        if key in obj:
            return obj[key]
        raise KeyError(key)
    if hasattr(obj, key):
        return getattr(obj, key)
    try:
        return obj[key]
    except (TypeError, KeyError, IndexError):
        raise KeyError(key) from None


def _set_child(obj: Any, key: str, value: Any) -> None:
    """Set ``key`` on ``obj``: item assignment on a dict, attribute assignment where the attribute exists, item assignment otherwise;
    a refusal of the object (a frozen dataclass, a locked mapping, a read-only attribute) is :class:`JaxMemError` naming the knob."""
    if isinstance(obj, dict):
        try:
            obj[key] = value
        except Exception as e:  # noqa: BLE001
            raise JaxMemError(f"config knob {key!r} on {type(obj).__name__} is not settable: {e!r}") from None
        return
    if hasattr(obj, key):
        try:
            setattr(obj, key, value)
        except Exception as e:  # noqa: BLE001
            raise JaxMemError(f"config knob {key!r} on {type(obj).__name__} is not settable: {e!r}") from None
        return
    try:
        obj[key] = value
    except Exception as e:  # noqa: BLE001
        raise JaxMemError(f"config knob {key!r} on {type(obj).__name__} is not settable: {e!r}") from None


def get_config_path(config: Any, path: str):
    """The value at a dotted ``path`` of an attribute- or mapping-style config; :class:`JaxMemError` names the first absent segment."""
    parts = [p for p in str(path).split(".") if p]
    if not parts:
        raise JaxMemError("empty config path")
    obj = config
    for i, key in enumerate(parts):
        try:
            obj = _get_child(obj, key)
        except KeyError:
            raise JaxMemError(f"config path {'.'.join(parts[:i + 1])!r} is absent (the engine's config carries no such knob)") from None
    return obj


def set_config_path(config: Any, path: str, value: Any) -> dict:
    """Set ``value`` at an EXISTING dotted ``path`` (a path whose parent or leaf is absent raises by name — a knob is never invented).
    Returns ``{"path", "stock_value", "value"}`` for the record."""
    parts = [p for p in str(path).split(".") if p]
    stock = get_config_path(config, path)
    parent = get_config_path(config, ".".join(parts[:-1])) if len(parts) > 1 else config
    _set_child(parent, parts[-1], value)
    return {"path": ".".join(parts), "stock_value": stock, "value": value}


def settable(config: Any, path: str) -> Optional[str]:
    """Whether the knob at ``path`` can be set: the stock value is written back onto itself (no change); ``None`` when that succeeds,
    else the reason (a frozen dataclass, a locked mapping, a read-only attribute) — the precondition of the ``subbatch`` lever."""
    try:
        stock = get_config_path(config, path)
        set_config_path(config, path, stock)
    except JaxMemError as e:
        return str(e)
    return None


def check_shape_spec(spec: Sequence) -> list:
    """Validate every row of a per-shape spec ``((max_tokens | None, value), ...)``: two fields per row, ``max_tokens`` None or an
    int, thresholds strictly increasing, a ``None`` row only last. Returns the rows as a list; :class:`JaxMemError` names the first bad row."""
    rows = list(spec)
    if not rows:
        raise JaxMemError("shape spec is empty")
    last = None
    for i, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise JaxMemError(f"shape spec row {i} {row!r} is not (max_tokens, value)")
        max_tokens = row[0]
        if max_tokens is None:
            if i != len(rows) - 1:
                raise JaxMemError(f"shape spec row {i} has max_tokens None before the last row")
            continue
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise JaxMemError(f"shape spec row {i} max_tokens {max_tokens!r} is not a positive int or None")
        if last is not None and max_tokens <= last:
            raise JaxMemError(f"shape spec row {i} max_tokens {max_tokens} is not above the previous row's {last}")
        last = max_tokens
    return rows


def shape_spec_value(spec: Sequence, tokens: int):
    """Resolve a per-shape spec at ``tokens``: the first row with ``max_tokens`` None or ``>= tokens`` (the whole spec is validated
    first); :class:`JaxMemError` when no row covers ``tokens`` (a spec without a ``None`` row is bounded). Informational — the
    engine resolves its own spec at run time."""
    for max_tokens, value in check_shape_spec(spec):
        if max_tokens is None or int(tokens) <= int(max_tokens):
            return value
    raise JaxMemError(f"no row of the shape spec {tuple(spec)!r} covers {tokens} tokens")


def parse_value(raw: str):
    """A ``value`` flag: an int (``"4"``), a JSON per-shape spec (``"[[2048, null], [null, 1024]]"``), or JSON ``null``."""
    s = str(raw).strip()
    if s.lstrip("-").isdigit():
        return int(s)
    v = json.loads(s)
    if isinstance(v, list):
        return tuple(tuple(r) if isinstance(r, list) else r for r in v)
    return v


def _value_kind(value: Any) -> str:
    """``count`` (an int >= 1), ``shape_spec`` (a validated per-shape spec) or :class:`JaxMemError`."""
    if isinstance(value, bool):
        raise JaxMemError(f"value {value!r} is neither a row count nor a per-shape spec")
    if isinstance(value, int):
        if value < 1:
            raise JaxMemError(f"row count must be >= 1 (got {value})")
        return "count"
    if isinstance(value, (list, tuple)):
        check_shape_spec(value)
        return "shape_spec"
    raise JaxMemError(f"value {value!r} is neither a row count nor a per-shape spec")


def _subbatch_applies(ctx: Ctx) -> Optional[Refusal]:
    lever = "subbatch"
    h = ctx.require(lever, "config", "path")
    if not isinstance(h["path"], str) or not h["path"].strip("."):
        return refuse(lever, "hooks.path", f"path must be a dotted config path (got {h['path']!r})")
    value = ctx.setting(lever, "value", None, cast=parse_value)
    if value is None:
        return refuse(lever, "value", f"no value: set {setting_ref(lever, 'value')} = <row count | JSON per-shape spec>")
    try:
        _value_kind(value)
    except JaxMemError as e:
        return refuse(lever, "value", f"malformed value: {e}")
    why = settable(h["config"], h["path"])
    if why is not None:
        return refuse(lever, "hooks.path", why)
    if get_config_path(h["config"], h["path"]) == value:
        return refuse(lever, "value", f"the value {value!r} is stock's own at {h['path']!r}: nothing to apply — turn the lever off by name "
                                      f"({off_ref(lever)}) or set a different value")
    return None


def mark_subbatch(ctx: Ctx, detail: Optional[str] = None) -> None:
    """The adapter's per-item mark (inside the item's unit): the sub-batched op ran on the item with the applied value. The lever not
    applied = nothing recorded."""
    rec = ctx.record
    if rec is None or "subbatch" not in rec.levers:
        return
    rec.mark("subbatch", detail=detail)


@register("subbatch", family="jax", exact="measured",
          exact_reason="a batched op split into chunks whose results are concatenated; the per-chunk kernels may differ from the full-batch "
                       "ones under XLA — the identity row states bitwise or band",
          applies=_subbatch_applies, description="the engine's sub-batch knob (a row count or a per-shape shard spec) set at its config path; stock's value recorded",
          preconditions=("hooks.config", "hooks.path", "value"), settings=("value",), frameworks=JAX)
def subbatch(ctx: Ctx) -> Applied:
    h = ctx.require("subbatch", "config", "path")
    value = ctx.setting("subbatch", "value", None, cast=parse_value)
    rec = set_config_path(h["config"], h["path"], value)
    rec["kind"] = _value_kind(value)
    rec["stock"] = False
    config, path, stock = h["config"], h["path"], rec["stock_value"]

    def undo():
        set_config_path(config, path, stock)

    return Applied(lever="subbatch", settings=rec, sites=(h["path"],), undo=undo)


# ------------------------------------------------------------------------------------------------------------ bucket policy


def resolve_bucket(tokens: int, buckets: Sequence[int], probe_bucket: Optional[int] = None) -> dict:
    """``{"tokens", "bucket", "padding", "case"}`` (module contract); :class:`JaxMemError` for a non-positive size, an unordered
    table, or a probe bucket inside the table or below the size."""
    tokens = int(tokens)
    if tokens < 1:
        raise JaxMemError(f"tokens must be >= 1 (got {tokens})")
    table = [int(b) for b in buckets]
    if table != sorted(table) or len(set(table)) != len(table) or any(b < 1 for b in table):
        raise JaxMemError(f"the bucket table must be strictly increasing positive sizes (got {table})")
    for b in table:
        if tokens <= b:
            return {"tokens": tokens, "bucket": b, "padding": b - tokens, "case": "shipped"}
    if probe_bucket is not None:
        pb = int(probe_bucket)
        if table and pb <= table[-1]:
            raise JaxMemError(f"probe bucket {pb} is not beyond the shipped table (last shipped {table[-1]})")
        if pb < tokens:
            raise JaxMemError(f"probe bucket {pb} is below the size {tokens}")
        return {"tokens": tokens, "bucket": pb, "padding": pb - tokens, "case": "probe"}
    return {"tokens": tokens, "bucket": tokens, "padding": 0, "case": "exact"}


def bucket_for(ctx: Ctx, tokens: int) -> dict:
    """The adapter's call per item: the bucket of the applied lever's table (and probe bucket) at ``tokens``; the case marks the
    record (a ``probe`` case is a named event on the Applied's notes). The lever not applied is :class:`JaxMemError` (the adapter
    resolves its shipped table itself when the lever is off)."""
    a = None
    for x in (ctx.record.applied if ctx.record is not None else ()):
        if x.lever == "bucket_policy":
            a = x
    if a is None:
        raise JaxMemError("bucket_policy is not applied on this record")
    r = resolve_bucket(tokens, a.settings["table"], a.settings.get("probe_bucket"))
    if r["case"] == "probe":
        a.notes.append(f"probe bucket {r['bucket']} at {tokens} tokens (padding {r['padding']}): a probe setting of record, not a release bucket")
    ctx.record.mark("bucket_policy", detail=f"tokens={tokens} bucket={r['bucket']} case={r['case']}")
    return r


def _bucket_policy_applies(ctx: Ctx) -> Optional[Refusal]:
    lever = "bucket_policy"
    h = ctx.require(lever, "buckets")
    table = h["buckets"]
    if isinstance(table, (str, bytes)) or not isinstance(table, (list, tuple)) or not table:
        return refuse(lever, "hooks.buckets", f"buckets must be the engine's shipped compile table (a non-empty sequence of sizes; got {table!r})")
    probe = ctx.setting(lever, "probe_bucket", None, cast=int)
    if probe is None:
        return refuse(lever, "probe_bucket", f"no probe bucket: nothing to apply — the shipped table is the engine's own choice; set "
                                             f"{setting_ref(lever, 'probe_bucket')} = <size beyond the table> or leave the lever out of the line")
    try:
        resolve_bucket(1, table, None)
        resolve_bucket(int(table[-1]) + 1, table, probe)
    except (JaxMemError, TypeError, ValueError) as e:
        return refuse(lever, "probe_bucket", str(e) or repr(e))
    return None


@register("bucket_policy", family="jax", exact="measured",
          exact_reason="a probe bucket beyond the shipped table pads where stock compiles the exact size; the reductions the masked tokens sit in "
                       "change shape under XLA — the identity row states bitwise or band",
          applies=_bucket_policy_applies, description="the compile-bucket probe: a named bucket beyond the shipped table, resolved per item (shipped | probe | exact)",
          preconditions=("hooks.buckets", "probe_bucket"), settings=("probe_bucket",), frameworks=JAX, scope="process")
def bucket_policy(ctx: Ctx) -> Applied:
    table = [int(b) for b in ctx.hook("bucket_policy", "buckets")]
    probe = int(ctx.setting("bucket_policy", "probe_bucket", None, cast=int))
    return Applied(lever="bucket_policy", settings={"table": table, "probe_bucket": probe, "stock": False}, sites=("hooks.buckets",),
                   notes=[f"probe bucket {probe} named beyond the shipped table (last shipped {table[-1]})"])


# ------------------------------------------------------------------------------------------------------------ XLA environment


def xla_env_record(environ: Optional[Mapping[str, str]] = None) -> dict:
    """The XLA client environment as :func:`opt_core.mem.peak.xla_env` records it (the named settings + every XLA_* / JAX_* variable) plus
    every variable whose name starts with one of :data:`RECORD_EXTRA_PREFIXES`, as found (sorted)."""
    env = os.environ if environ is None else environ
    rec = dict(peak.xla_env(env))
    rec.update({k: env[k] for k in env if k.startswith(RECORD_EXTRA_PREFIXES)})
    return {k: rec[k] for k in sorted(rec)}


def mem_fraction_name(environ: Mapping[str, str], pinned: Optional[Mapping[str, str]] = None) -> str:
    """The MEM_FRACTION variable this process's fraction lives under: the one name PRESENT in ``environ`` (in the client's read order,
    :data:`opt_core.mem.peak.MEM_FRACTION_PRECEDENCE`), else the one the kit PINS, else :data:`MEM_FRACTION_VAR`. Both names present is
    not a name: :func:`mem_fraction_conflict` states it and the lever refuses (the CUDA plugin does not initialise with both set)."""
    for source in (environ, pinned or {}):
        for name in peak.MEM_FRACTION_PRECEDENCE:
            if name in source:
                return name
    return MEM_FRACTION_VAR


def mem_fraction_conflict(environ: Mapping[str, str]) -> Optional[dict]:
    """``{name: value}`` of BOTH MEM_FRACTION names when both are present in ``environ`` (the conflict :func:`opt_core.mem.peak.jax_effective`
    reports as ``conflict: true``); None otherwise."""
    both = {n: environ[n] for n in peak.MEM_FRACTION_PRECEDENCE if n in environ}
    return both if len(both) == len(peak.MEM_FRACTION_PRECEDENCE) else None


def xla_env_check(pinned: Mapping[str, str], environ: Optional[Mapping[str, str]] = None,
                  requested: Optional[Mapping[str, str]] = None) -> dict:
    """``{"ok", "collisions", "live_differs", "unset_pins", "record"}``: a requested value that differs from a pinned one is a collision
    (the lever refuses; nothing is overridden); a pinned variable whose live value differs is ``live_differs`` (a named note on the
    record); a pinned variable unset in ``environ`` is listed (the adapter decides whether to set the pin). ``ok`` = no collision.
    Values compare stripped and case-insensitively (the ``jax_export`` rule)."""
    env = os.environ if environ is None else environ
    requested = dict(requested or {})
    collisions, live, unset = {}, {}, []
    for k, v in pinned.items():
        if k in requested and not _same(requested[k], v):
            collisions[k] = {"pinned": v, "requested": requested[k]}
        if k not in env:
            unset.append(k)
        elif not _same(env[k], v):
            live[k] = {"pinned": v, "live": env[k]}
    return {"ok": not collisions, "collisions": collisions, "live_differs": live, "unset_pins": unset, "record": xla_env_record(env)}


def _same(a: Any, b: Any) -> bool:
    return str(a).strip().lower() == str(b).strip().lower()


def _drift_notes(chk: Mapping) -> list:
    return [f"pinned variable differs live: {k}={v['live']} (pinned {v['pinned']})" for k, v in chk["live_differs"].items()]


def _xla_env_applies(ctx: Ctx) -> Optional[Refusal]:
    lever = "xla_env"
    h = ctx.require(lever, "pinned")
    if not isinstance(h["pinned"], Mapping):
        return refuse(lever, "hooks.pinned", f"pinned must be a mapping of the engine's pinned variables (got {type(h['pinned']).__name__})")
    requested = ctx.hook(lever, "requested")
    if requested is not None and not isinstance(requested, Mapping):
        return refuse(lever, "hooks.requested", "requested must be a mapping of variable -> value or absent")
    chk = xla_env_check(h["pinned"], ctx.environ, requested)
    for k, v in chk["collisions"].items():
        return refuse(lever, f"xla.{k}", f"{k}={v['requested']!r} is requested but the kit pins {v['pinned']!r}: never overridden — change the "
                                         "kit's pin or drop the request", pinned=v["pinned"], requested=v["requested"])
    return None


@register("xla_env", family="jax", exact="bitwise", exact_reason="the environment is recorded as found; nothing is set or overridden",
          applies=_xla_env_applies, description="the XLA / JAX environment recorded; a requested value that collides with a pin refuses; live drift noted",
          preconditions=("hooks.pinned", "hooks.requested", "xla"), settings=(), frameworks=JAX, scope="process")
def xla_env(ctx: Ctx) -> Applied:
    chk = xla_env_check(ctx.hook("xla_env", "pinned"), ctx.environ, ctx.hook("xla_env", "requested"))
    return Applied(lever="xla_env",
                   settings={"record": chk["record"], "pinned": dict(ctx.hook("xla_env", "pinned")), "unset_pins": chk["unset_pins"],
                             "live_differs": chk["live_differs"], "stock": True},
                   sites=("environ",), notes=_drift_notes(chk))


# ------------------------------------------------------------------------------------------------------------ memory fraction


def _fraction_str(f: float) -> str:
    s = f"{float(f):.4f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def _mem_fraction_wanted(ctx: Ctx) -> dict:
    f = ctx.setting("mem_fraction", "fraction", None, cast=float)
    pre = ctx.setting("mem_fraction", "preallocate", None, cast=parse_bool)
    pinned = ctx.hook("mem_fraction", "pinned", None)
    wanted = {mem_fraction_name(ctx.environ, pinned if isinstance(pinned, Mapping) else None): _fraction_str(f)}
    if pre is not None:
        wanted[PREALLOCATE_VAR] = "true" if pre else "false"
    return wanted


def _mem_fraction_applies(ctx: Ctx) -> Optional[Refusal]:
    lever = "mem_fraction"
    h = ctx.require(lever, "pinned")
    if not isinstance(h["pinned"], Mapping):
        return refuse(lever, "hooks.pinned", f"pinned must be a mapping of the engine's pinned variables (got {type(h['pinned']).__name__})")
    f = ctx.setting(lever, "fraction", None, cast=float)
    if f is None:
        return refuse(lever, "fraction", f"no fraction: set {setting_ref(lever, 'fraction')} = <0 < f>")
    if isinstance(f, bool) or not isinstance(f, (int, float)) or not float(f) > 0:
        return refuse(lever, "fraction", f"fraction must be numeric and > 0 (got {f!r})")
    unified = ctx.setting(lever, "unified_memory", False, cast=parse_bool)
    if float(f) > 1 and not unified:
        return refuse(lever, "unified_memory", f"fraction {f} above 1 needs unified memory: set unified_memory=1 after setting the engine's "
                                               "unified-memory switch")
    ctx.setting(lever, "preallocate", None, cast=parse_bool)                # a malformed flag refuses here by name
    both = mem_fraction_conflict(ctx.environ)
    if both is not None:
        return refuse(lever, "xla.MEM_FRACTION", "both " + " and ".join(peak.MEM_FRACTION_PRECEDENCE) + " are set: the CUDA plugin does not "
                      "initialise with both (jax falls back to the CPU backend) — unset one in the kit's environment; nothing is exported over it",
                      present=both)
    wanted = _mem_fraction_wanted(ctx)
    for k, v in wanted.items():
        if k in h["pinned"] and not _same(h["pinned"][k], v):
            return refuse(lever, f"pins.{k}", f"the kit pins {k}={h['pinned'][k]!r} and the lever wants {v!r}: never overridden — change the "
                                              "kit's pin or turn the lever off by name", pinned=h["pinned"][k], wanted=v)
    return None


@register("mem_fraction", family="jax", exact="bitwise", exact_reason="the client allocator's reservation size; no numerics move",
          applies=_mem_fraction_applies, description="the client memory fraction (+ preallocate) exported through allocator.jax_export where unset or equal, under the "
                      "MEM_FRACTION name already present or pinned (XLA_CLIENT_MEM_FRACTION | XLA_PYTHON_CLIENT_MEM_FRACTION); both present refuses",
          preconditions=("hooks.pinned", "fraction", "unified_memory", "preallocate", "pins", "xla"), settings=("fraction", "preallocate", "unified_memory"),
          frameworks=JAX, scope="process")
def mem_fraction(ctx: Ctx) -> Applied:
    wanted = _mem_fraction_wanted(ctx)
    exported = allocator.jax_export(ctx, "mem_fraction", **wanted)          # unset: exported + recorded; equal: kept; different: RefusalError xla.<NAME>
    chk = xla_env_check(ctx.hook("mem_fraction", "pinned"), ctx.environ)
    return Applied(lever="mem_fraction",
                   settings={"fraction": float(ctx.setting("mem_fraction", "fraction", None, cast=float)), "wanted": wanted, "exported": exported,
                             "stock": all(v["state"] == "kept" for v in exported.values())},
                   sites=tuple(wanted), notes=_drift_notes(chk))


# ------------------------------------------------------------------------------------------------------------ flash triangle attention


FLASH_VERDICT = ("no flash triangle-attention kernel for a JAX engine is in the core: the one existing Pallas kernel is a "
                 "forward+backward kernel over a different pairformer module and does not port; an engine whose attention setting is already a "
                 "fused implementation has the memory-transient class in stock; a forward-only kernel for another module family is a new "
                 "kernel, not a port")


def flash_triattn_feasibility(attention_family: Optional[str], flash_implementation: Optional[str] = None) -> dict:
    """``{"applies": False, "family", "verdict", "flash_implementation", "stock_fused"}``: what the adapter states about its engine's
    attention (``attention_family`` a free label; ``flash_implementation`` its own setting, ``xla`` = un-fused) beside the verdict."""
    fam = (attention_family or "").strip().lower() or "unknown"
    impl = (flash_implementation or "").strip().lower() or None
    stock_fused = None if impl is None else impl != "xla"
    return {"applies": False, "family": fam, "verdict": FLASH_VERDICT, "flash_implementation": impl, "stock_fused": stock_fused}


def _flash_applies(ctx: Ctx) -> Optional[Refusal]:
    v = flash_triattn_feasibility(ctx.hook("flash_triattn_jax", "attention_family"), ctx.hook("flash_triattn_jax", "flash_implementation"))
    return refuse("flash_triattn_jax", "kernel", "feasibility only: " + v["verdict"], **v)


@register("flash_triattn_jax", family="jax", exact="measured",
          exact_reason="no kernel in the core: the label would be the kernel's identity row",
          applies=_flash_applies, description="flash triangle attention for the JAX engines: feasibility verdict only; refuses by design",
          preconditions=("kernel",), settings=(), frameworks=JAX)
def flash_triattn_jax(ctx: Ctx) -> Applied:
    raise RefusalError(_flash_applies(ctx))                                 # unreachable through apply(): applies() always refuses
