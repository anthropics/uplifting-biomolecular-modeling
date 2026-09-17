"""The applied record of a ``big`` process: what was applied, refused and checked; the mode line; the manifest block; the census.

Contract. One :class:`AppliedRecord` per process holds, in order, the levers applied (:class:`opt_core.mem.registry.Applied`), the
levers refused (:class:`opt_core.mem.registry.Refusal`, each naming its precondition), every precondition checked, every setting read
(value and source), the flags read, the allocator settings in force, and the lever modules the install holds. The mode line is
``big:<lever,lever,…>`` — the applied levers in order, :meth:`AppliedRecord.mode_line`; the kit prints it once as its ACTIVE line
(:meth:`AppliedRecord.active_line` composes one from the ``opt_core.report`` primitives; a kit whose ACTIVE grammar the launcher holds
composes its own from the same fields); the per-lever activation evidence is :meth:`AppliedRecord.lever_lines` — one
``[tag] LEVER name=<lever> state=on|skipped|off … strategy=<STRATEGIES.json id>`` line per lever in :func:`opt_core.report.lever_line`'s grammar,
the ONE producer of that line for the memory levers (strategy ids: :data:`STRATEGY_OF`, or the Lever's own) and writes the block :meth:`AppliedRecord.manifest_block` to ``opt_manifest.json`` under
``kit.big`` (:meth:`AppliedRecord.attach`). The exactness of the mode is composed from the applied levers' labels — ``bitwise`` only
when every lever is bit-exact, ``band`` when any is band and none measured, ``measured`` when any is measured — and the block names the
label per lever beside the declared one. A lever never widens its declared label (refused at :meth:`AppliedRecord.add`); it narrows it
only with the equality record id that proves the narrower label (``Applied.narrowed_by``), a named event.

The fallback census (total accounting per unit). A unit is one item / pass the kit delimits with :meth:`AppliedRecord.unit_begin` /
:meth:`AppliedRecord.unit_end`; a lever of scope ``process`` (an allocator policy, an exported variable) is marked once on the implicit
unit ``process`` when it is applied, and a kit that delimits no unit lands every event there. A lever's levered path marks itself when it
runs (:meth:`AppliedRecord.mark`); a degraded path is a named event (:meth:`AppliedRecord.fallback` — the reason is the row-level mode
field); a site that had nothing to do names why (:meth:`AppliedRecord.skip`). :meth:`AppliedRecord.census` compares, per unit, the
expected lever set — the applied levers of the unit's scope (the line after the flags, minus the refused, or the unit's own narrower
set when the kit named one) — with the observed one: a lever that fell back or never marked is ``partial`` on that unit; a skipped lever
is accounted with its reason and is not partial; a lever that marked without being expected is ``unexpected`` (partial when it is one
the flags turned off or the record refused — a patch that stayed in force). Marks refuse misuse by name: a unit opened over an open one,
a mark for a unit-scope lever after the last unit closed. :meth:`AppliedRecord.exit_gate` is the fail-closed gate: any refusal or any
partial unit turns a successful exit into ``opt_core.report.EXIT_NOT_ACTIVE`` through ``opt_core.report.verdict`` unless the kit's
opt-out is given — its parsed ``--allow-partial`` passed as ``allow_partial=`` at selection or to the gate; the opt-out is recorded in the
block; nothing here is silent and nothing is read from the environment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Iterable, Mapping, Optional, Sequence

from .. import report
from .registry import BIG, EXACT_LABELS, OPT_OUT, SCOPES, Applied, Pending, Refusal, Selection

PROCESS_UNIT = "process"                                    # the implicit unit when the kit delimits none
NO_UNIT = "<no unit observed>"                              # the census entry of a run that ran nothing
PARTIAL_REFUSED = "{prefix} NOT ACTIVE: big partial — {detail}; exit " + str(report.EXIT_NOT_ACTIVE) + " ({opt_out} records and proceeds)"
PARTIAL_ALLOWED = "{prefix} PARTIAL allowed: {detail} ({opt_out}, recorded)"


STRATEGY_OF = {                       # registered lever of this package -> canonical strategy id (opt_core/STRATEGIES.json); a Lever's own .strategy wins
    "expandable_segments": "F7.expandable_segments", "cache_release": "F7.cache_release",
    "host_park": "F7.pair_offload",
    "chunk_pair_transition": "F7.chunked_eval", "chunk_pair_transition_band": "F7.chunked_eval", "chunk_triangle_attention": "F7.chunked_eval",
    "chunk_triangle_multiplication": "F7.chunked_eval", "chunk_triangle_multiplication_band": "F7.chunked_eval",
    "chunk_confidence_head": "F7.chunked_eval", "chunk_msa_rows": "F7.chunked_eval",
    "samples_per_pass": "F7.ckpt_resume_samples", "sample_loop": "F7.sample_loop", "recycle_carry": "F7.recycle_carry", "seed_batch": "F7.seed_batch",
    "graph_capture": "F7.graph_pool_budget", "hoist_off": "F7.hoist_off",
    "subbatch": "F7.jax_subbatch", "bucket_policy": "F3.shape_buckets", "xla_env": "F7.jax_memory_flags", "mem_fraction": "F7.jax_memory_flags",
    "flash_triattn_jax": "F1.flash_triatt",
}


def strategy_of(lever: str) -> Optional[str]:
    """The strategy id lever ``lever``'s LEVER line carries: the registered Lever's own ``strategy`` when declared, else :data:`STRATEGY_OF`,
    else None (a kit-local lever registered without one: its line carries no ``strategy=``)."""
    from .registry import LEVERS
    lv = LEVERS.get(str(lever))
    if lv is not None and lv.strategy:
        return lv.strategy
    return STRATEGY_OF.get(str(lever))


def _token(v) -> str:
    """One blank-free token for a LEVER line value (report.kv renders lists with ','; blanks inside a value become '_')."""
    text = report.kv(("v", v)).split("=", 1)[1]
    return text.replace(" ", "_")


def compose_exact(labels: Iterable[str]) -> str:
    """The mode's label from its levers': ``measured`` if any, else ``band`` if any, else ``bitwise`` (an empty set changes nothing)."""
    labels = [str(x) for x in labels]
    for lb in labels:
        if lb not in EXACT_LABELS:
            raise ValueError(f"exact label {lb!r} is not one of {EXACT_LABELS}")
    if "measured" in labels:
        return "measured"
    if "band" in labels:
        return "band"
    return "bitwise"


@dataclass
class UnitRecord:
    """One unit's events: ``ran`` (levers whose levered path executed), ``fallback`` (lever -> reason), ``skipped`` (lever -> reason),
    ``events`` in order, ``expected`` (the unit's own expectation when the kit narrowed it, else None = the record's)."""

    unit: str
    ran: list = field(default_factory=list)
    fallback: dict = field(default_factory=dict)
    skipped: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    expected: Optional[tuple] = None
    closed: bool = False

    def as_dict(self) -> dict:
        return {"unit": self.unit, "ran": list(self.ran), "fallback": dict(self.fallback), "skipped": dict(self.skipped),
                "events": list(self.events), "expected": list(self.expected) if self.expected is not None else None,
                "closed": self.closed}


@dataclass
class AppliedRecord:
    """The record (module contract). ``base`` is the mode big composes on; ``line`` the kit's big line; ``expected`` the levers
    after the flags (the selection); ``applied`` / ``refused`` in order; ``preconditions`` one entry per lever checked; ``settings``
    one entry per value read; ``flags`` the kit's lever switches as given; ``allocator`` the settings in force; ``discovery`` the lever
    modules loaded / absent; ``units`` the census input; ``notes`` named events; ``extra`` the kit's own fields."""

    prefix: str
    base: str
    line: tuple = ()
    mode: str = BIG
    drop: tuple = ()
    base_rule: Optional[str] = None
    line_policy: dict = field(default_factory=dict)         # the line selector's record: selector / chosen / reason / details / declared
    expected: tuple = ()
    off_by_flag: tuple = ()
    on_by_flag: tuple = ()
    applied: list = field(default_factory=list)
    refused: list = field(default_factory=list)
    preconditions: list = field(default_factory=list)
    settings: list = field(default_factory=list)
    flags: dict = field(default_factory=dict)
    allocator: dict = field(default_factory=dict)
    discovery: dict = field(default_factory=dict)
    units: dict = field(default_factory=dict)
    allow_partial: bool = False
    opt_out: str = OPT_OUT
    notes: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    exit: Optional[dict] = None
    _open_unit: Optional[str] = None
    _labels: dict = field(default_factory=dict)             # lever -> the label fixed at add(); never widened after (B20)

    # ------------------------------------------------------------------------------------------------ what was applied
    @classmethod
    def from_selection(cls, sel: Selection, *, prefix: str, base: str, discovery: Optional[Mapping] = None, drop: Iterable[str] = (),
                       base_rule: Optional[str] = None, opt_out: str = OPT_OUT) -> "AppliedRecord":
        rec = cls(prefix=prefix, base=base, line=tuple(sel.line), expected=tuple(sel.levers), off_by_flag=tuple(sel.off_by_flag),
                  on_by_flag=tuple(sel.on_by_flag), flags=dict(sel.flags), allow_partial=bool(sel.allow_partial), opt_out=str(opt_out),
                  discovery=dict(discovery or {}), drop=tuple(drop), base_rule=base_rule)
        for r in sel.refusals:
            rec.refuse(r)
        return rec

    @property
    def levers(self) -> tuple:
        """The applied lever names in order — the mode line's list."""
        return tuple(a.lever for a in self.applied)

    @property
    def refused_names(self) -> tuple:
        return tuple(r.lever for r in self.refused)

    def add(self, applied: Applied, lever_exact: str, lever_reason: str, scope: str = "unit") -> Applied:
        """Record an applied lever. The label rule: the declared label fills a label the Applied left unset; a WIDER label than the
        declared one is refused outright (a lever that cannot hold its declared label declares the weaker one at registration); a
        NARROWER label needs ``Applied.narrowed_by`` — the equality record id that proves it — and is recorded as a named event. A lever
        added twice, or added after it was refused, is an error. A ``process``-scope lever without a deferred check is marked on the
        ``process`` unit here; one with a deferred effectiveness check is marked when the record runs that check."""
        if applied.lever in self.levers:
            raise ValueError(f"lever {applied.lever} is applied twice")
        if applied.lever in self.refused_names:
            raise ValueError(f"lever {applied.lever} was refused and cannot be applied")
        if scope not in SCOPES:
            raise ValueError(f"lever {applied.lever}: scope {scope!r} is not one of {SCOPES}")
        applied.declared_exact = lever_exact
        applied.scope = scope
        if applied.exact is None:
            applied.exact = lever_exact
        if applied.exact not in EXACT_LABELS:
            raise ValueError(f"lever {applied.lever}: exact {applied.exact!r} is not one of {EXACT_LABELS}")
        if EXACT_LABELS.index(applied.exact) > EXACT_LABELS.index(lever_exact):
            raise ValueError(f"lever {applied.lever}: exact {applied.exact!r} is wider than the declared {lever_exact!r}: a lever that "
                             f"cannot hold its declared label declares the weaker one at registration")
        if applied.exact != lever_exact:
            if not applied.narrowed_by:
                raise ValueError(f"lever {applied.lever}: exact {applied.exact!r} narrows the declared {lever_exact!r} without the identity "
                                 f"record id that proves it (Applied.narrowed_by)")
            if applied.exact_reason is None:
                applied.exact_reason = f"narrowed from {lever_exact} by identity record {applied.narrowed_by}"
            self.note(f"{applied.lever}: label_narrowed {lever_exact} -> {applied.exact} by identity record {applied.narrowed_by}")
            applied.notes.append(f"label_narrowed: {lever_exact} -> {applied.exact} ({applied.narrowed_by})")
        elif applied.exact_reason is None:
            applied.exact_reason = lever_reason
        self._labels[applied.lever] = applied.exact                                        # the label, fixed at add() (B20)
        self.applied.append(applied)
        if scope == "process" and applied.verify is None:
            self._unit(PROCESS_UNIT).ran.append(applied.lever)
            self._unit(PROCESS_UNIT).events.append({"lever": applied.lever, "kind": "ran", "detail": "applied (process scope)"})
        return applied

    def verify(self, final: bool = False) -> dict:
        """Run the deferred effectiveness checks the Applied entries carry — the read-back of a process-scope lever's effect. A check that
        returns None is settled in force (the lever is marked ``ran`` on the ``process`` unit); a string settles it not in force (a
        named ``fallback`` there, partial at the gate); :class:`opt_core.mem.registry.Pending` leaves it pending (nothing to read yet —
        CUDA not initialised) and it is re-run at every unit boundary; ``final=True`` (the exit gate) settles a still-pending check as
        the fallback its pending reason names. Returns ``{lever: outcome}`` for the checks run now."""
        out = {}
        for a in self.applied:
            if a.verify is None or (a.verified is not None and a.verified.get("state") != "pending"):
                continue
            got = a.verify()
            if isinstance(got, Pending) and not final:
                a.verified = {"ok": None, "state": "pending", "reason": got.reason}
            elif isinstance(got, Pending):
                a.verified = {"ok": False, "state": "final", "reason": f"never verifiable: {got.reason}"}
                self.fallback(a.lever, f"not in force: {a.verified['reason']}", unit=PROCESS_UNIT)
            elif got is None:
                a.verified = {"ok": True, "state": "final", "reason": None}
                self.mark(a.lever, unit=PROCESS_UNIT, detail="verified")
            else:
                a.verified = {"ok": False, "state": "final", "reason": str(got)}
                self.fallback(a.lever, f"not in force: {got}", unit=PROCESS_UNIT)
            out[a.lever] = a.verified
        return out

    @property
    def scopes(self) -> dict:
        return {a.lever: a.scope for a in self.applied}

    def get(self, lever: str) -> Applied:
        for a in self.applied:
            if a.lever == lever:
                return a
        raise KeyError(f"lever {lever!r} is not applied (applied: {', '.join(self.levers) or 'none'})")

    def detail(self, lever: str, detail: Mapping) -> dict:
        """Store a mechanism's own record under the applied lever (``Applied.detail``; JSON-ready values) — a park's ``record()``, a
        chunk summary, a carry's state. Returns the detail in force."""
        a = self.get(lever)
        a.detail.update(dict(detail))
        return a.detail

    def call(self, lever: str, site: str, exact: Optional[str] = None, reason: Optional[str] = None, *, decision: Optional[str] = None,
             unit: Optional[str] = None, **fields) -> dict:
        """The per-call exactness event of an applied lever (module contract): ``site`` is the op / hook point, ``exact`` the label of
        THIS call (a chunk covering the axis is bit-exact, a chunked reduction band …) with its ``reason``; ``decision`` names a call
        that did not take the levered path — ``passthrough`` (the un-chunked statement ran by policy) and ``fallback`` are
        :meth:`fallback` events with the reason (partial at the gate unless the opt-out), anything else marks the lever ``ran``. A
        lever's label is FIXED at :meth:`add` from its settings (B20): a call labelled wider than it is refused — a decision that would
        exceed the label is a named fallback (``decision=``), never a relabel — and the comparison is against the label captured at
        add(), so an in-place change of the Applied cannot bypass it; the per-call labels are counted per site under the lever's
        ``detail["calls"]``. Returns the event."""
        a = self.get(lever)
        fixed = self._labels[lever]                                                        # the label fixed at add(), not the live object
        if a.exact != fixed:
            raise ValueError(f"lever {lever}: the Applied's label was changed in place to {a.exact!r} after add() fixed {fixed!r}: a "
                             f"label is never widened or rewritten post-hoc (B13.3 / B20)")
        if exact is not None and exact not in EXACT_LABELS:
            raise ValueError(f"lever {lever}: call exact {exact!r} is not one of {EXACT_LABELS}")
        if exact is not None and EXACT_LABELS.index(exact) > EXACT_LABELS.index(fixed):
            raise ValueError(f"lever {lever}: a call at {site} labelled {exact!r} is wider than the lever's {fixed!r}: a call whose "
                             f"decision exceeds the declared label is a named fallback (decision=), never a relabel")
        if unit is None and self.scopes.get(lever) == "process":
            unit = PROCESS_UNIT
        if decision in ("passthrough", "fallback"):
            self.fallback(lever, f"{decision}: {reason or site}", unit=unit)
        else:
            self.mark(lever, unit=unit, detail=site)
        e = {"lever": lever, "kind": "call", "site": site, "exact": exact, "reason": reason, "decision": decision, "detail": dict(fields)}
        self._unit(unit, lever).events.append(e)
        if exact is not None and decision not in ("passthrough", "fallback"):
            calls = a.detail.setdefault("calls", {}).setdefault(site, {})
            calls[exact] = calls.get(exact, 0) + 1
        return e

    def sink(self, lever: str):
        """The bound per-call record sink of an applied lever for a mechanism that reports each call as a plain dict (the ``record=``
        callable of the chunked ops): the entry's ``op`` is the site, ``exact`` / ``reason`` the call's label, ``decision`` as in
        :meth:`call`; every other field rides in the event."""
        self.get(lever)                                                                  # applied, by name

        def _sink(entry: Mapping) -> None:
            e = dict(entry)
            e.pop("lever", None)
            self.call(lever, str(e.pop("op", None) or e.pop("site", None) or "?"), e.pop("exact", None), e.pop("reason", None),
                      decision=e.pop("decision", None), **e)
        return _sink

    # ------------------------------------------------------------------------------------------------ refusals / checks
    def refuse(self, refusal: Refusal) -> Refusal:
        self.refused.append(refusal)
        return refusal

    def check(self, lever: str, preconditions: Sequence[str], failed: Optional[str] = None, reason: Optional[str] = None) -> dict:
        """Record one lever's precondition check: every declared name, the one that failed (or None)."""
        entry = {"lever": lever, "checked": list(preconditions), "ok": failed is None, "failed": failed, "reason": reason}
        self.preconditions.append(entry)
        return entry

    def note_setting(self, lever: str, key: str, value, source: str) -> None:
        self.settings.append({"lever": lever, "key": key, "value": value, "source": source})

    def note(self, text: str) -> None:
        self.notes.append(str(text))

    @property
    def allocator_settings(self) -> dict:
        """The allocator values the levers set (the ``settings`` part of the ``allocator`` block)."""
        return dict((self.allocator or {}).get("settings") or {})

    @property
    def exact(self) -> str:
        """The mode's label, composed from the labels fixed at add() (an in-place change of an Applied never reaches it)."""
        return compose_exact(self._labels[a.lever] for a in self.applied)

    @property
    def exact_per_lever(self) -> dict:
        return {a.lever: self._labels[a.lever] for a in self.applied}

    # ------------------------------------------------------------------------------------------------ the lines
    def mode_line(self) -> str:
        """``big:<lever,lever,…>`` — the applied levers in order (``big:`` with none)."""
        return f"{self.mode}:" + ",".join(self.levers)

    def active_line(self, tag: str, **extra) -> str:
        """``[tag] ACTIVE mode=big:<levers> [line=<name>[(auto: <reason>)]] base=<base>[(override)] drop=<names|none> exact=<label>
        refused=<names|none> off=<names|none> on=<names|none> allocator=<k:v,…|none> [extra…]`` — the offered grammar (``line`` only
        when the kit declares more than the default line, or the selector is auto); a kit holding its own ACTIVE bytes composes from
        the fields."""
        base = self.base if self.base_rule in (None, "fast-else-exact") else f"{self.base}({self.base_rule})"
        p = self.line_policy
        line = None
        if p and (len(p.get("declared") or {}) > 1 or p.get("selector") == "auto"):
            line = p.get("chosen") or "none"
            if p.get("selector") == "auto":
                line += f"(auto: {p.get('reason')})"
        pairs = [("mode", self.mode_line())] + ([("line", line)] if line is not None else []) + [
            ("base", base), ("drop", list(self.drop)), ("exact", self.exact), ("refused", list(self.refused_names)),
            ("off", list(self.off_by_flag)), ("on", list(self.on_by_flag)), ("allocator", self.allocator_settings or None)]
        return f"{report.prefix(tag)} ACTIVE " + report.kv(*pairs, **extra)

    def lever_lines(self, tag: str) -> List[str]:
        """The per-lever activation-evidence lines of this process in the core's ONE grammar (:func:`opt_core.report.lever_line`), one per
        lever the process names, in this order: every APPLIED lever ``state=on`` (``impl`` = the lever's module, ``origin`` = ``core`` for a
        lever of this package else ``kit``, ``strategy`` = :func:`strategy_of`, evidence ``exact=<label> scope=<unit|process>`` + its settings
        as ``k=v`` + ``sites=`` + ``narrowed_by=`` when set + the deferred check's outcome pair once one ran); every REFUSED lever
        ``state=skipped reason=<precondition>`` (the sentence is in the NOT ACTIVE line and the manifest block; a line value is one token);
        every lever the flags turned OFF or the line DROPPED ``state=off reason=flag|drop``. The mode summary stays :meth:`active_line`."""
        from .registry import LEVERS
        out: List[str] = []

        def impl_origin(name: str):
            lv = LEVERS.get(name)
            mod = (lv.module if lv is not None else "") or "unregistered"
            return mod, ("core" if mod.startswith("opt_core.") else "kit")

        for a in self.applied:
            impl, origin = impl_origin(a.lever)
            ev = [("exact", self._labels.get(a.lever, a.exact)), ("scope", a.scope)]
            ev += [(str(k), _token(v)) for k, v in sorted(a.settings.items())]
            if a.sites:
                ev.append(("sites", _token(list(a.sites))))
            if a.narrowed_by:
                ev.append(("narrowed_by", _token(a.narrowed_by)))
            if a.verified is not None:
                ev.append(("verified", "ok" if a.verified.get("ok") else _token(a.verified.get("reason") or "not_in_force")))
            out.append(report.lever_line(tag, a.lever, "on", *ev, impl=impl, origin=origin, strategy=strategy_of(a.lever)))
        for r in self.refused:
            impl, origin = impl_origin(r.lever)
            out.append(report.lever_line(tag, r.lever, "skipped", reason=_token(r.precondition), impl=impl, origin=origin,
                                         strategy=strategy_of(r.lever)))
        named = {a.lever for a in self.applied} | {r.lever for r in self.refused}
        for name, why in [(n, "flag") for n in self.off_by_flag] + [(n, "drop") for n in self.drop]:
            if name in named:
                continue
            named.add(name)
            impl, origin = impl_origin(name)
            out.append(report.lever_line(tag, name, "off", reason=why, impl=impl, origin=origin, strategy=strategy_of(name)))
        return out

    def not_active_line(self, tag: str) -> str:
        """``[tag] NOT ACTIVE: big refused — <lever>: <precondition>: <reason>[; …] (mode=big base=<base>)``."""
        detail = "; ".join(str(r) for r in self.refused) or "no lever applied"
        return report.not_active_line(tag, f"{self.mode} refused — {detail}", f"mode={self.mode} base={self.base}")

    # ------------------------------------------------------------------------------------------------ the census
    def unit_begin(self, unit: str, expected: Optional[Iterable[str]] = None) -> UnitRecord:
        """Open a unit (an item / a pass). ``expected`` narrows the unit's lever set (recorded); a unit opened twice is an error."""
        unit = str(unit)
        if unit == PROCESS_UNIT:
            raise ValueError(f"unit name {PROCESS_UNIT!r} is the implicit process unit")
        if unit in self.units:
            raise ValueError(f"unit {unit!r} opened twice")
        if self._open_unit is not None:
            raise ValueError(f"unit {self._open_unit!r} is still open")
        self.verify()
        u = UnitRecord(unit=unit, expected=tuple(expected) if expected is not None else None)
        self.units[unit] = u
        self._open_unit = unit
        return u

    def unit_end(self, unit: Optional[str] = None) -> UnitRecord:
        unit = self._open_unit if unit is None else str(unit)
        if unit is None or unit not in self.units:
            raise ValueError(f"unit {unit!r} is not open")
        u = self.units[unit]
        u.closed = True
        if self._open_unit == unit:
            self._open_unit = None
        self.verify()
        return u

    def _unit(self, unit: Optional[str], lever: Optional[str] = None) -> UnitRecord:
        """The unit an event lands on: the named one; else the open one; else ``process`` for a process-scope lever or a kit that
        delimits no unit — a unit-scope event after the last unit closed is an error by name."""
        name = self._open_unit if unit is None else str(unit)
        if name is None:
            if lever is not None and self.scopes.get(lever, "unit") == "unit" and any(k != PROCESS_UNIT for k in self.units):
                raise ValueError(f"lever {lever}: no unit is open (the last one closed) — a unit-scope event needs unit_begin")
            name = PROCESS_UNIT
        if name not in self.units:
            self.units[name] = UnitRecord(unit=name)
        return self.units[name]

    def mark(self, lever: str, unit: Optional[str] = None, detail: Optional[str] = None) -> None:
        """The lever's levered path ran on the unit (the open one by default; the ``process`` unit for a process-scope lever)."""
        if unit is None and self.scopes.get(lever) == "process":
            unit = PROCESS_UNIT
        u = self._unit(unit, lever)
        if lever not in u.ran:
            u.ran.append(lever)
        u.events.append({"lever": lever, "kind": "ran", "detail": detail})

    def fallback(self, lever: str, reason: str, unit: Optional[str] = None) -> None:
        """A degraded path ran in place of the lever on the unit — the named row-level event; the unit is partial."""
        if unit is None and self.scopes.get(lever) == "process":
            unit = PROCESS_UNIT
        u = self._unit(unit, lever)
        u.fallback[lever] = str(reason)
        u.events.append({"lever": lever, "kind": "fallback", "detail": str(reason)})

    def skip(self, lever: str, reason: str, unit: Optional[str] = None) -> None:
        """The lever's site had nothing to do on the unit (named); accounted, not partial."""
        if unit is None and self.scopes.get(lever) == "process":
            unit = PROCESS_UNIT
        u = self._unit(unit, lever)
        u.skipped[lever] = str(reason)
        u.events.append({"lever": lever, "kind": "skip", "detail": str(reason)})

    def census(self) -> dict:
        """Expected vs observed per unit (module contract). ``units``: per unit ``{expected, ran, skipped, fallback, absent, unexpected,
        partial}``; ``partial_units``; ``partial`` (the union of partial levers); ``per_lever`` counts; ``n_units`` / ``n_ok`` /
        ``n_partial``; ``ok`` (no partial unit). A unit's expectation is the applied levers of its scope (``process`` levers on the
        ``process`` unit, ``unit`` levers on the kit's units), or the unit's own set when the kit narrowed it."""
        scopes = self.scopes
        applied = [a.lever for a in self.applied]
        units_out: dict = {}
        partial_units: list = []
        union: list = []
        per_lever: dict = {lv: {"ran": 0, "skipped": 0, "fallback": 0, "absent": 0, "unexpected": 0} for lv in applied}
        leaked = set(self.off_by_flag) | set(self.refused_names)
        for name, u in self.units.items():
            if u.expected is not None:
                expected = tuple(u.expected)
            elif name == PROCESS_UNIT and any(k != PROCESS_UNIT for k in self.units):
                expected = tuple(lv for lv in applied if scopes.get(lv) == "process")
            elif name == PROCESS_UNIT:
                expected = tuple(applied)                                            # a kit that delimits no unit
            else:
                expected = tuple(lv for lv in applied if scopes.get(lv, "unit") == "unit")
            absent = [lv for lv in expected if lv not in u.ran and lv not in u.skipped and lv not in u.fallback]
            unexpected = [lv for lv in u.ran if lv not in expected]
            partial = [lv for lv in expected if lv in u.fallback or lv in absent] + [lv for lv in unexpected if lv in leaked]
            for lv in expected:
                c = per_lever.setdefault(lv, {"ran": 0, "skipped": 0, "fallback": 0, "absent": 0, "unexpected": 0})
                c["fallback" if lv in u.fallback else "ran" if lv in u.ran else "skipped" if lv in u.skipped else "absent"] += 1
            for lv in unexpected:
                per_lever.setdefault(lv, {"ran": 0, "skipped": 0, "fallback": 0, "absent": 0, "unexpected": 0})["unexpected"] += 1
            units_out[name] = {"expected": list(expected), "ran": list(u.ran), "skipped": dict(u.skipped), "fallback": dict(u.fallback),
                               "absent": absent, "unexpected": unexpected, "partial": partial, "closed": u.closed}
            if partial:
                partial_units.append(name)
                union += [lv for lv in partial if lv not in union]
        n = len(units_out)
        return {"units": units_out, "partial_units": partial_units, "partial": union, "per_lever": per_lever, "n_units": n,
                "n_ok": n - len(partial_units), "n_partial": len(partial_units), "ok": not partial_units}

    def declared_tier(self) -> Optional[str]:
        """The tier the kit declared for the chosen line (``line_policy.declared[chosen].tier``), or None."""
        p = self.line_policy
        d = (p.get("declared") or {}).get(p.get("chosen") or "") if p else None
        return d.get("tier") if isinstance(d, Mapping) else None

    def tier_check(self) -> Optional[dict]:
        """The declared tier of the chosen line against the applied levers' composed label: ``{"declared", "composed", "ok"}`` — a
        composed label wider than the declared tier is a named note (the claim did not hold); None when no tier was declared."""
        declared = self.declared_tier()
        if declared is None:
            return None
        composed = self.exact
        ok = EXACT_LABELS.index(composed) <= EXACT_LABELS.index(declared)
        return {"declared": declared, "composed": composed, "ok": ok}

    def exit_gate(self, rc: int, *, expect_units: bool = True, incomplete: Optional[str] = None,
                  allow_partial: Optional[bool] = None) -> dict:
        """The fail-closed gate (module contract): ``opt_core.report.verdict`` over ``partial`` = every refusal (by the name it
        carries, deduplicated) plus the census's partial levers (plus every unit-scope lever when ``expect_units`` and nothing was
        observed at all — no kit unit and no event on the implicit ``process`` unit, which stands in for a kit that delimits none),
        with the opt-out = ``allow_partial`` when the kit passes its parsed flag here, else the value it passed at selection
        (``opt_core.mem.apply(..., allow_partial=)``; the record keeps the value that decided). Returns the verdict with ``census``,
        ``refused``, ``opt_out``, ``allow_partial_source`` and ``reasons`` added; stored on the record (``exit``)."""
        self.verify(final=True)
        c = self.census()
        partial = list(dict.fromkeys(self.refused_names))
        reasons = {}
        for r in self.refused:
            reasons.setdefault(r.lever, f"refused: {r.precondition}: {r.reason}")
        unit_levers = [a.lever for a in self.applied if a.scope == "unit"]
        observed = {lv for u in self.units.values() for lv in unit_levers if lv in u.ran or lv in u.fallback or lv in u.skipped}
        if expect_units and unit_levers and not any(k != PROCESS_UNIT for k in self.units) and not observed:
            partial += [lv for lv in unit_levers if lv not in partial]                        # nothing ran anywhere: no unit observed
            for lv in unit_levers:
                reasons.setdefault(lv, NO_UNIT)
        for lv in c["partial"]:
            if lv not in partial:
                partial.append(lv)
            units = [u for u, d in c["units"].items() if lv in d["partial"]]
            why = [f"{u}: {c['units'][u]['fallback'].get(lv) or ('ran while off / refused' if lv in c['units'][u]['unexpected'] else 'never marked')}"
                   for u in units]
            reasons.setdefault(lv, "; ".join(why))
        if allow_partial is not None:
            self.allow_partial = bool(allow_partial)
        allow = self.allow_partial
        v = report.verdict(rc, {"partial": partial, "gated": []}, allow, incomplete)
        v.update({"census": c, "refused": [r.as_dict() for r in self.refused], "opt_out": self.opt_out,
                  "allow_partial_source": "kit" if allow else None, "reasons": reasons})
        self.exit = v
        return v

    def exit_line(self, tag: str, verdict: Optional[Mapping] = None) -> Optional[str]:
        """The one line of a partial verdict — refused: :data:`PARTIAL_REFUSED`; allowed: :data:`PARTIAL_ALLOWED` (the detail is
        ``opt_core.report.partial_detail``; the opt-out named is the kit's ``opt_out`` word, ``--allow-partial``) — None when nothing is partial."""
        v = self.exit if verdict is None else verdict
        if v is None:
            raise ValueError("exit_gate has not run")
        if not v["partial"]:
            return None
        detail = report.partial_detail(v, v.get("reasons"))
        if v["allow_partial"]:
            return PARTIAL_ALLOWED.format(prefix=report.prefix(tag), detail=detail, opt_out=self.opt_out)
        if v["exit_code"] == report.EXIT_NOT_ACTIVE:
            return PARTIAL_REFUSED.format(prefix=report.prefix(tag), detail=detail, opt_out=self.opt_out)
        return None

    # ------------------------------------------------------------------------------------------------ the manifest block
    def manifest_block(self) -> dict:
        """The ``kit.big`` block of ``opt_manifest.json``: every field of the record, JSON-ready."""
        return {
            "mode": self.mode,
            "mode_line": self.mode_line(),
            "base": self.base,
            "base_rule": self.base_rule,
            "drop": list(self.drop),
            "line_policy": dict(self.line_policy),
            "tier_check": self.tier_check(),
            "line": list(self.line),
            "expected": list(self.expected),
            "levers": [dict(a.as_dict(), exact=self._labels[a.lever]) for a in self.applied],
            "exact": self.exact,
            "exact_per_lever": self.exact_per_lever,
            "refused": [r.as_dict() for r in self.refused],
            "preconditions": list(self.preconditions),
            "settings": list(self.settings),
            "flags": dict(self.flags),
            "off_by_flag": list(self.off_by_flag),
            "on_by_flag": list(self.on_by_flag),
            "allow_partial": {"opt_out": self.opt_out, "value": self.allow_partial},
            "allocator": dict(self.allocator),
            "discovery": dict(self.discovery),
            "census": self.census(),
            "exit": dict(self.exit) if self.exit is not None else None,
            "notes": list(self.notes),
            "extra": dict(self.extra),
        }

    def attach(self, kit: dict, key: str = BIG) -> dict:
        """``kit[key] = manifest_block()`` — the kit passes ``kit`` to ``opt_core.manifest.build(kit=…)``. Returns ``kit``."""
        kit[key] = self.manifest_block()
        return kit

    def as_dict(self) -> dict:
        return self.manifest_block()
