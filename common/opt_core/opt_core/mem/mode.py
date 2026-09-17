"""opt_core.mem.mode — the ``big`` mode's application contract: :func:`apply` (the kit's line -> an :class:`AppliedRecord`, every
step recorded, fail-closed), :func:`census`, :func:`undo`, :class:`Refused`. The package binds these names on first access
(``opt_core.mem.apply`` …); this module imports the registry, the record and the composition; :func:`apply` imports the allocator snapshot it applies.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

from . import registry as _registry
from .compose import BigLine, select_line
from .record import AppliedRecord
from .registry import LEVER_MODULES, LEVERS, Applied, Ctx, Refusal, RefusalError, refuse

__all__ = ["Refused", "apply", "census", "undo"]

class Refused(Exception):
    """Raised by :func:`apply` under ``strict`` when a line lever refused: ``record`` holds every refusal (each names its
    precondition); the message is the record's NOT ACTIVE detail. The kit prints its own NOT ACTIVE line from the record and raises
    its ActivationError."""

    def __init__(self, record: AppliedRecord):
        super().__init__("; ".join(str(r) for r in record.refused) or "no lever applied")
        self.record = record


def apply(levers: Union[BigLine, Sequence[str]], ctx: Ctx, *, base: Optional[str] = None, strict: bool = True,
          modules: Sequence[str] = LEVER_MODULES, line: Optional[str] = None, switches: Optional[Mapping[str, Any]] = None,
          allow_partial: bool = False) -> AppliedRecord:
    """Apply the kit's big line (package contract). Steps, every one recorded: (0) for a :class:`BigLine` with declared lines,
    :func:`compose.select_line` resolves the kit's ``line`` selector (a declared name, or ``auto`` through the kit's policy — recorded as
    ``line_policy``; a refusal names ``line`` and no lever is applied); (1) :func:`registry.discover` imports the lever
    modules of this package (``modules``; an absent module and a module whose dependency the process lacks are named); (2) :func:`registry.selection` resolves the line against the
    kit's ``switches`` (a switch naming no lever, a value not on/off, an on for an unregistered lever refuse by name) and records
    ``allow_partial`` (the kit's parsed opt-out); (3) the allocator settings in force are snapshotted;
    (4) each selected lever, in order: framework match, ``applies(ctx)`` (its precondition check — a refusal is recorded with the
    precondition's name and the lever is not applied), ``apply(ctx)`` (the Applied joins the record with the lever's scope — a
    process-scope lever is marked on the ``process`` unit here; a RefusalError raised inside is a recorded refusal; any other exception
    propagates with its traceback — a broken lever is a defect, not a fallback); (5) under
    ``strict`` any refusal raises :class:`Refused` carrying the record (fail-closed); otherwise the record returns with its refusals
    and the exit gate turns them into ``EXIT_NOT_ACTIVE`` unless ``allow_partial``. ``levers`` is the :class:`BigLine` from
    :func:`compose_big` (its ``base`` is recorded) or a plain sequence of names (then ``base`` names the mode composed on)."""
    drop, base_rule, policy, line_refusal = (), None, {}, None
    if isinstance(levers, BigLine):
        base = levers.base if base is None else base
        drop, base_rule = levers.drop, levers.base_rule
        line_levers, policy, line_refusal = select_line(levers, ctx, line)
    else:
        if line is not None:
            raise TypeError("line= selects among a BigLine's declared lines; a plain lever sequence has none")
        line_levers = tuple(str(x) for x in levers)
    disc = _registry.discover(modules)
    sel = _registry.selection(line_levers, switches=switches, allow_partial=allow_partial)
    rec = AppliedRecord.from_selection(sel, prefix=ctx.prefix, base=base or "none", discovery=disc, drop=drop, base_rule=base_rule,
                                       opt_out=ctx.opt_out)
    rec.line_policy = dict(policy)
    if line_refusal is not None:
        rec.refuse(line_refusal)
    rec.extra = dict(ctx.extra)
    ctx.record = rec
    from . import allocator as _allocator
    rec.allocator = {"found": _allocator.snapshot(ctx.environ, framework=ctx.framework), "writes": [], "settings": {}}
    refused_by_selection = {r.lever for r in sel.refusals}
    for name in sel.levers:
        lv = LEVERS.get(name)
        if lv is None:
            continue                                                                    # refused by the selection, recorded there
        if name in refused_by_selection:
            continue
        if ctx.framework not in lv.frameworks:
            rec.refuse(refuse(name, "framework", f"lever {name} serves {', '.join(lv.frameworks)}; the kit's framework is {ctx.framework}"))
            rec.check(name, lv.preconditions, failed="framework", reason=rec.refused[-1].reason)
            continue
        try:
            r = lv.applies(ctx)
        except RefusalError as e:
            r = e.refusal
        if r is not None:
            if not isinstance(r, Refusal):
                raise TypeError(f"lever {name}: applies() returned {type(r).__name__}, expected None or a Refusal")
            rec.refuse(r)
            rec.check(name, lv.preconditions, failed=r.precondition, reason=r.reason)
            continue
        rec.check(name, lv.preconditions)
        try:
            a = lv.apply(ctx)
        except RefusalError as e:
            rec.refuse(e.refusal)
            rec.preconditions[-1].update({"ok": False, "failed": e.refusal.precondition, "reason": e.refusal.reason})
            continue
        if not isinstance(a, Applied):
            raise TypeError(f"lever {name}: apply() returned {type(a).__name__}, expected an Applied")
        if a.lever != name:
            raise ValueError(f"lever {name}: apply() returned an Applied named {a.lever!r}")
        rec.add(a, lv.exact, lv.exact_reason, scope=lv.scope)
        if lv.family == "allocator":
            rec.allocator["settings"].update(a.settings)
    tc = rec.tier_check()
    if tc is not None and not tc["ok"]:
        rec.note(f"line {rec.line_policy.get('chosen')}: declared tier {tc['declared']} but the applied levers compose to {tc['composed']}")
    if strict and rec.refused:
        raise Refused(rec)
    return rec


def census(record: AppliedRecord) -> dict:
    """``record.census()`` — expected vs observed lever set per unit (see :mod:`opt_core.mem.record`)."""
    return record.census()


def undo(record: AppliedRecord) -> list:
    """Restore the un-levered paths of every applied lever that can be undone, in reverse order; returns the names undone (a lever
    without ``undo`` is named in the record's notes)."""
    done = []
    for a in reversed(record.applied):
        if a.undo is None:
            record.note(f"{a.lever}: no undo (the lever cannot be undone in-process)")
            continue
        a.undo()
        done.append(a.lever)
    return done
