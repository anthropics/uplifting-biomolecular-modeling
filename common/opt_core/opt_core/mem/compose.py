"""Mode composition: ``big`` as one more mode of a kit's ``opt_core.modes.ModeTable``, on top of a base mode.

Contract. :func:`compose_big` extends a kit's table with the mode ``big`` (the one spelling: the table, the ACTIVE line, the
manifest and the run script all use it) and returns the :class:`BigLine` the kit applies: the ``base`` mode whose levers big
composes on, the memory ``levers`` in order (the kit's default big line — each one the kit's switches can leave out by name), ``drop``, the
base's levers big leaves out by name (a graph-capture lever the allocator policy excludes, for instance), and the kit's named
alternative ``lines`` (``{name: levers}`` — a size ladder, a bucket policy) with an optional ``auto`` policy. The base rule is
:func:`base_for`: ``fast`` where the table has it, else ``exact``; a kit that names another base records it as an override — the line
says so, the record says so, never silently. The table is extended, never rebuilt: the kit's names, default and unknown-mode text stay.

Declared lines (B4 / N3). Every line — ``default`` (the ``levers`` argument, with ``tier`` / ``cost_note``) and each entry of
``lines={name: {"levers": [...], "tier": <bitwise|band|measured>, "cost_note": <str>}}`` — carries its DECLARED tier (the kit's
claim for the line's lever set, checked against the applied levers' composed label) and a ``cost_note`` (the speed cost as a ledger
citation — an expN row id — or ``"unmeasured"``): the declaration rides the record's ``line_policy["declared"]`` and the manifest
block; the MEASURED speed cost is a ledger fact cited by row id, never typed here. A bare lever sequence declares
``tier=None`` / ``cost_note="unstated"`` — recorded as such, never filled in silently.

The line selector. The kit passes ``line=<name>`` (``opt_core.mem.apply``) to pick a declared line (``default`` is the ``levers``
argument); ``line=auto`` runs the kit's ``auto`` policy — a callable ``policy(ctx) -> (name, reason)`` or ``(name, reason, details)`` over
whatever the kit put in ``ctx`` (its token count in ``ctx.extra``, say) — and the choice with its reason is recorded (``line_policy``);
``auto`` without a policy, or a policy naming an undeclared line, is a refusal naming ``line`` (:func:`select_line`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

from .. import modes
from .registry import BIG, EXACT_LABELS, NAME_RE, Ctx, Refusal, refuse

_NAME = NAME_RE                                                                   # the registry's one name grammar
BASE_RULE = "fast-else-exact"
DEFAULT_LINE = "default"
AUTO = "auto"


@dataclass(frozen=True)
class BigLine:
    """The kit's big line: ``base`` (the mode composed on), ``levers`` (in order — the ``default`` line), ``drop`` (base levers left
    out, by name), ``base_rule`` (``fast-else-exact`` when the base is the rule's, else ``override``), ``name`` (the mode's spelling),
    ``lines`` (every declared line by name, ``default`` included), ``auto`` / ``auto_default`` (the selector policy, module contract)."""

    base: str
    levers: tuple
    drop: tuple = ()
    base_rule: str = BASE_RULE
    name: str = BIG
    lines: dict = field(default_factory=dict)                        # name -> {"levers", "tier", "cost_note"} (``default`` = ``levers``)
    auto: Optional[Callable] = None                                  # policy(ctx) -> (name, reason[, details]) for the auto selector
    auto_default: bool = False                                       # the policy runs when the selector is unset

    def as_dict(self) -> dict:
        return {"name": self.name, "base": self.base, "levers": list(self.levers), "drop": list(self.drop), "base_rule": self.base_rule,
                "lines": {k: {"levers": list(v["levers"]), "tier": v["tier"], "cost_note": v["cost_note"]} for k, v in self.lines.items()},
                "auto": self.auto is not None, "auto_default": self.auto_default}

    def line_levers(self, name: str) -> tuple:
        return tuple(self.lines[name]["levers"])

    def describe(self) -> str:
        """``big = <base> + [a,b,c] - [x]``."""
        s = f"{self.name} = {self.base} + [{','.join(self.levers)}]"
        if self.drop:
            s += f" - [{','.join(self.drop)}]"
        if self.base_rule != BASE_RULE:
            s += f" (base {self.base_rule})"
        return s


def base_for(table: modes.ModeTable) -> str:
    """The rule: ``fast`` where the table has it, else ``exact``; a table with neither has nothing to compose on (raises)."""
    if modes.FAST in table.modes:
        return modes.FAST
    if modes.EXACT in table.modes:
        return modes.EXACT
    raise modes.ModeError(f"the mode table {table.modes} has neither {modes.FAST!r} nor {modes.EXACT!r}: nothing for {BIG} to compose on")


def compose_big(table: modes.ModeTable, base: Optional[str] = None, levers: Sequence[str] = (), *, name: str = BIG,
                  drop: Sequence[str] = (), lines: Optional[Mapping[str, object]] = None, auto: Optional[Callable] = None,
                  auto_default: bool = False, tier: Optional[str] = None, cost_note: Optional[str] = None) -> tuple:
    """``(table_with_big, BigLine)`` (module contract). ``base`` None = :func:`base_for`; a named base must be one of the table's
    lever modes (never ``off``); the mode name must be new to the table; lever names are ``[a-z][a-z0-9_]*`` and unique."""
    if not isinstance(table, modes.ModeTable):
        raise TypeError(f"compose_big wants an opt_core.modes.ModeTable, got {type(table).__name__}")
    if not _NAME.match(name):
        raise ValueError(f"mode name {name!r} is not [a-z][a-z0-9_]*")
    if name in table.modes:
        raise modes.ModeError(f"the mode table {table.modes} already has {name!r}: compose it once")
    try:
        rule = base_for(table)
    except modes.ModeError:
        if base is None:
            raise
        rule = None                                                                       # an explicit base on a table without fast / exact
    if base is None:
        base = rule
    base = str(base).strip().lower()
    if base not in table.kit_modes:
        raise modes.ModeError(f"base {base!r} is not a lever mode of the table (expected one of {'|'.join(table.kit_modes)})")
    levers = tuple(str(x) for x in levers)
    for lv in levers:
        if not _NAME.match(lv):
            raise ValueError(f"lever name {lv!r} is not [a-z][a-z0-9_]*")
    if len(levers) != len(set(levers)):
        raise ValueError(f"the big line repeats a lever: {levers}")
    drop = tuple(str(x) for x in drop)
    declared = {DEFAULT_LINE: _declare_line(DEFAULT_LINE, {"levers": levers, "tier": tier, "cost_note": cost_note})}
    for ln, spec in (lines or {}).items():
        if not _NAME.match(str(ln)) or ln == AUTO:
            raise ValueError(f"line name {ln!r} is not [a-z][a-z0-9_]* (and not {AUTO!r})")
        d = _declare_line(str(ln), spec)
        if ln == DEFAULT_LINE and d["levers"] != levers:
            raise ValueError(f"line {DEFAULT_LINE!r} is the levers argument ({levers}); lines[{DEFAULT_LINE!r}] says {d['levers']}")
        declared[str(ln)] = d
    if auto is not None and not callable(auto):
        raise TypeError("auto must be a callable policy(ctx) -> (name, reason[, details])")
    if auto_default and auto is None:
        raise ValueError("auto_default=True needs an auto policy")
    new = modes.ModeTable(modes=table.modes + (name,), default=table.default, unknown_message=table.unknown_message)
    line = BigLine(base=base, levers=levers, drop=drop, base_rule=BASE_RULE if base == rule else "override", name=name,
                     lines=declared, auto=auto, auto_default=auto_default)
    return new, line


def _declare_line(name: str, spec) -> dict:
    """``{"levers": tuple, "tier": label|None, "cost_note": str}`` from a line declaration — a mapping with those keys, or a bare lever
    sequence (``tier=None``, ``cost_note="unstated"``)."""
    if isinstance(spec, Mapping):
        unknown = set(spec) - {"levers", "tier", "cost_note"}
        if unknown:
            raise ValueError(f"line {name}: unknown keys {sorted(unknown)} (levers, tier, cost_note)")
        levers, tier, cost_note = spec.get("levers", ()), spec.get("tier"), spec.get("cost_note")
    else:
        levers, tier, cost_note = spec, None, None
    levers = tuple(str(x) for x in levers)
    for x in levers:
        if not _NAME.match(x):
            raise ValueError(f"line {name}: lever name {x!r} is not [a-z][a-z0-9_]*")
    if len(levers) != len(set(levers)):
        raise ValueError(f"line {name} repeats a lever: {levers}")
    if tier is not None and tier not in EXACT_LABELS:
        raise ValueError(f"line {name}: tier {tier!r} is not one of {EXACT_LABELS}")
    return {"levers": levers, "tier": tier, "cost_note": str(cost_note) if cost_note is not None else "unstated"}


def select_line(line: BigLine, ctx: Ctx, selector: Optional[str] = None) -> tuple:
    """Resolve the kit's line selector (module contract): ``selector`` is the line NAME the kit chose (from its own flag), ``"auto"``
    (the kit's declared policy decides), or None (``auto`` when the kit declared ``auto_default``, else the default line). Returns
    ``(levers, policy, refusal)`` — ``policy`` is the record's ``line_policy`` ``{"selector", "chosen", "reason", "details", "declared"}``
    (``declared`` = every line's ``{levers, tier, cost_note}``), ``refusal`` a :class:`Refusal` naming ``line`` or None. A refused
    selection falls to no line: the levers are ``()`` and the refusal gates the exit. Nothing is read from the environment."""
    raw = None if selector is None or str(selector).strip() == "" else str(selector).strip()
    selector = raw if raw is not None else (AUTO if line.auto_default else DEFAULT_LINE)
    policy = {"selector": selector, "chosen": None, "reason": None, "details": {},
              "declared": {k: {"levers": list(v["levers"]), "tier": v["tier"], "cost_note": v["cost_note"]} for k, v in line.lines.items()}}
    if selector == AUTO:
        if line.auto is None:
            return (), policy, refuse(BIG, "line", f"line selector auto but the kit declares no auto policy (declared lines: "
                                                     f"{', '.join(sorted(line.lines))})")
        out = line.auto(ctx)
        if not isinstance(out, (tuple, list)) or len(out) not in (2, 3) or not isinstance(out[0], str):
            raise TypeError(f"the auto policy returned {out!r}, expected (name, reason[, details])")
        chosen, reason = out[0], str(out[1])
        details = dict(out[2]) if len(out) == 3 and isinstance(out[2], Mapping) else {}
        policy.update({"chosen": chosen, "reason": reason, "details": details})
        if chosen not in line.lines:
            return (), policy, refuse(BIG, "line", f"the auto policy chose line {chosen!r}, which the kit does not declare "
                                                     f"(declared: {', '.join(sorted(line.lines))}); reason given: {reason}")
        return line.line_levers(chosen), policy, None
    if selector not in line.lines:
        return (), policy, refuse(BIG, "line", f"line selector {selector!r} names no declared line (declared: "
                                                 f"{', '.join(sorted(line.lines))}; or auto)")
    policy.update({"chosen": selector, "reason": "selected by name" if raw is not None else "the default line"})
    return line.line_levers(selector), policy, None
