"""The lever registry of the ``big`` memory mode: what a lever is, how one is declared, how a kit's line selects them.

Contract. A :class:`Lever` is one memory mechanism with a name (``[a-z][a-z0-9_]*``), a ``family`` (one of :data:`FAMILIES`), an
``exact`` label (one of :data:`EXACT_LABELS`: ``bitwise`` — the levered path returns the same bytes as the un-levered path;
``band`` — within the kit's documented tolerance, the reason names the re-ordered reduction; ``measured`` — the equality tests decide
per run) with its ``exact_reason``, the ``preconditions`` it checks by name, the ``settings`` it reads (each a name the kit passes in
``ctx.settings[<lever>]``), and two callables: ``applies(ctx)`` returns ``None`` when every precondition holds or a
:class:`Refusal` naming the one that does not (never a silent no-op, never a silent stock fallback); ``apply(ctx)`` installs the lever
and returns an :class:`Applied` naming the values in force and the sites patched. :data:`LEVERS` is the registry by name; a lever
module declares its levers with the :func:`register` decorator at import; :func:`discover` imports the lever modules of this package
(:data:`LEVER_MODULES`) so the registry is complete before a line is applied — a module absent from the install is recorded by name,
never skipped silently. :class:`Ctx` is what a kit passes in: its prefix (the kit's variable stem, recorded), its tag, the framework, the hook points
per lever (module references, tensor names, chunk dimensions — the values a lever's docstring names), its setting values per lever,
the process environment its allocator levers act on. :func:`selection` resolves a kit's big line against the kit's explicit lever
SWITCHES (``switches={name: on/off}`` as the kit parsed them from its own flags: off leaves a line lever out, on adds a registered
lever; a name that is neither registered nor on the line, or a value that is not on/off, is a :class:`Refusal` by name) and carries
the kit's ``allow_partial`` opt-out. The core reads NO environment variable for the selection, a lever's settings or the opt-out:
the kit reads its own documented words and passes values (``switches=``, ``ctx.settings``, ``allow_partial=``). The core holds no lever value and
no engine name: every hook and every setting is the kit's. Lever modules import only the standard library at module level — a
framework (torch, jax) is imported inside the functions that need it — so :func:`discover` runs on any machine; a module that breaks the
rule is recorded ``broken`` by name on a machine without its framework and its levers refuse by name.

Families (one per lever; the vocabulary the record and the tables use):

    offload     host residency of named tensors / modules with streamed transfers (pinned RAM, row / column blocks)
    chunk       a chunked or streamed form of an operation (pair transition, triangle ops, heads, MSA rows, samples per pass)
    ckpt        recomputation in place of storage: block checkpointing, cycle / recycle checkpointing, seed batching caps
    allocator   the device allocator's policy (expandable segments, graph pools, cache release, peak counters) — never numerics
    jax         the JAX-side levers (sub-batching, bucket policy, a flash kernel where one exists)
    kernel      a memory-lean kernel routed in place of a stock op (the carried kernels of ``opt_core.kernels`` are cited by name)
    setting     a named value of the engine's own configuration that lowers peak (a cap, a dropped unread feature, a guard)
"""
from __future__ import annotations

import importlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

BIG = "big"
FAMILIES = ("offload", "chunk", "ckpt", "allocator", "jax", "kernel", "setting")
SCOPES = ("unit", "process")                                                      # where a lever's levered path marks itself
EXACT_LABELS = ("bitwise", "band", "measured")
FRAMEWORKS = ("torch", "jax")
LEVER_MODULES = ("allocator", "offload", "chunk", "ckpt", "sample_loop", "jax_mem")             # the lever modules of this package, import order
                                                                                  # (peak.py is the instrument, not a lever module)
OPT_OUT = "--allow-partial"                                                       # the kit's census opt-out flag, as its PARTIAL lines name it (Ctx.opt_out)
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")                                       # lever / line / mode names (the one definition)
_NAME = NAME_RE


class RegistryError(ValueError):
    """A lever declaration outside the contract (the message names the field)."""


@dataclass(frozen=True)
class Pending:
    """What a deferred effectiveness check (the Applied's check callable) returns while there is nothing to read yet (CUDA not initialised):
    the record re-runs the check at every unit boundary and finalises it at the exit gate, where a still-pending check is the named
    fallback ``reason``."""

    reason: str


# ------------------------------------------------------------------------------------------------------------ refusal / applied


@dataclass(frozen=True)
class Refusal:
    """Why a lever is not applied: ``lever``, the ``precondition`` that failed (a name from the lever's declaration or the selection
    grammar), the ``reason`` sentence a line can carry verbatim, ``details`` for the record."""

    lever: str
    precondition: str
    reason: str
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"lever": self.lever, "precondition": self.precondition, "reason": self.reason, "details": dict(self.details)}

    def __str__(self) -> str:
        return f"{self.lever}: {self.precondition}: {self.reason}"


class RefusalError(Exception):
    """Raised inside a lever's ``applies`` / ``apply`` to refuse by name; :func:`opt_core.mem.apply` records it as the lever's
    :class:`Refusal`."""

    def __init__(self, refusal: Refusal):
        super().__init__(str(refusal))
        self.refusal = refusal


def refuse(lever: str, precondition: str, reason: str, **details) -> Refusal:
    """The one constructor of a refusal: ``refuse("pair_offload", "hooks.pair_module", "no pair module named in ctx.hooks")``."""
    return Refusal(lever=lever, precondition=precondition, reason=reason, details=dict(details))


def refusal_from(exc: BaseException, lever: str) -> Refusal:
    """A :class:`Refusal` from a mechanism module's named refusal exception (duck-typed: its ``name`` or ``kind`` attribute is the
    precondition, its ``details`` mapping the details, ``str(exc)`` the reason) — the bridge an adapter uses in ``applies`` / ``apply``:
    ``except SomeRefusal as e: raise RefusalError(refusal_from(e, lever))``."""
    name = getattr(exc, "name", None) or getattr(exc, "kind", None) or type(exc).__name__
    details = getattr(exc, "details", None)
    return Refusal(lever=lever, precondition=str(name), reason=str(exc), details=dict(details) if isinstance(details, Mapping) else {})


@dataclass
class Applied:
    """What a lever did: ``settings`` (the values in force, e.g. ``{"chunk": 256}``), ``sites`` (the hook points patched, as the
    strings the kit named them by), ``exact`` / ``exact_reason`` (the lever's label — a lever may narrow its label at apply time, e.g.
    ``bitwise`` when a chunk equals the full dimension, with ``narrowed_by`` = the equality record id that proves it; it never widens
    it — a widening is refused), ``notes`` (named
    events), ``undo`` (restores the un-levered path; ``None`` when the lever cannot be undone in-process), the check field (a deferred
    effectiveness check for a lever whose effect can only be read later — an allocator setting read back once CUDA is up: the record
    runs it at every unit boundary until it settles and finalises it at the exit gate; ``None`` = in force, a string = the reason it
    is not (a named fallback), :class:`Pending` = nothing to read yet)."""

    lever: str
    settings: dict = field(default_factory=dict)
    sites: tuple = ()
    exact: Optional[str] = None
    exact_reason: Optional[str] = None
    notes: list = field(default_factory=list)
    undo: Optional[Callable[[], None]] = None
    declared_exact: Optional[str] = None                    # the lever's declared label (set by the record; shows a narrowing / widening)
    scope: str = "unit"                                     # the lever's scope (set by the record from the Lever)
    detail: dict = field(default_factory=dict)              # the mechanism's own record (a park's record(), a chunk summary), JSON-ready
    verify: Optional[Callable[[], object]] = None           # deferred effectiveness check: None = in force | str = not in force | Pending
    verified: Optional[dict] = None                         # the check's outcome once run (set by the record)
    narrowed_by: Optional[str] = None                       # the equality record id that proves a label narrower than the declared one

    def as_dict(self) -> dict:
        return {"lever": self.lever, "settings": dict(self.settings), "sites": list(self.sites), "exact": self.exact,
                "exact_reason": self.exact_reason, "declared_exact": self.declared_exact, "scope": self.scope,
                "notes": list(self.notes), "undoable": self.undo is not None, "detail": dict(self.detail),
                "verified": dict(self.verified) if self.verified is not None else None, "narrowed_by": self.narrowed_by}


# ------------------------------------------------------------------------------------------------------------ the lever


@dataclass(frozen=True)
class Lever:
    """One memory lever (module contract above). ``scope``: ``unit`` — the levered path marks itself on every unit the kit delimits
    (the census expects it there); ``process`` — the lever acts once for the process (an allocator policy, an exported variable) and
    is marked on the implicit ``process`` unit when applied. ``describe()`` is the one-line description the tables print."""

    name: str
    family: str
    exact: str
    exact_reason: str
    applies: Callable[["Ctx"], Optional[Refusal]]
    apply: Callable[["Ctx"], Applied]
    description: str = ""
    preconditions: tuple = ()
    settings: tuple = ()
    frameworks: tuple = ("torch",)
    module: str = ""
    scope: str = "unit"
    strategy: Optional[str] = None                     # the canonical strategy id (opt_core/STRATEGIES.json) or LOCAL.<kit>.<name> the LEVER line carries; None = record.STRATEGY_OF

    def __post_init__(self):
        if not isinstance(self.name, str) or not _NAME.match(self.name):
            raise RegistryError(f"lever name {self.name!r} is not [a-z][a-z0-9_]*")
        if self.family not in FAMILIES:
            raise RegistryError(f"lever {self.name}: family {self.family!r} is not one of {FAMILIES}")
        if self.exact not in EXACT_LABELS:
            raise RegistryError(f"lever {self.name}: exact {self.exact!r} is not one of {EXACT_LABELS}")
        if not self.exact_reason or not str(self.exact_reason).strip():
            raise RegistryError(f"lever {self.name}: exact_reason is empty (the label carries its reason)")
        if not callable(self.applies) or not callable(self.apply):
            raise RegistryError(f"lever {self.name}: applies and apply must be callables")
        for s in self.settings:
            if not _NAME.match(str(s)):
                raise RegistryError(f"lever {self.name}: setting name {s!r} is not [a-z][a-z0-9_]*")
        for fw in self.frameworks:
            if fw not in FRAMEWORKS:
                raise RegistryError(f"lever {self.name}: framework {fw!r} is not one of {FRAMEWORKS}")
        if self.scope not in SCOPES:
            raise RegistryError(f"lever {self.name}: scope {self.scope!r} is not one of {SCOPES}")
        if self.strategy is not None and (not str(self.strategy).strip() or any(c.isspace() for c in str(self.strategy))):
            raise RegistryError(f"lever {self.name}: strategy {self.strategy!r} is not one blank-free id")
        object.__setattr__(self, "preconditions", tuple(str(p) for p in self.preconditions))
        object.__setattr__(self, "settings", tuple(str(s) for s in self.settings))
        object.__setattr__(self, "frameworks", tuple(self.frameworks))

    def describe(self) -> str:
        return f"{self.name} [{self.family}; {self.exact}] {self.description or self.exact_reason}"

    def as_dict(self) -> dict:
        return {"name": self.name, "family": self.family, "exact": self.exact, "exact_reason": self.exact_reason,
                "description": self.description, "preconditions": list(self.preconditions), "settings": list(self.settings),
                "frameworks": list(self.frameworks), "module": self.module, "scope": self.scope}


LEVERS: dict = {}                                           # name -> Lever, in registration order


def register(name: str, *, family: str, exact: str, exact_reason: str, applies: Callable, description: str = "",
             preconditions: Iterable[str] = (), settings: Iterable[str] = (), frameworks: Iterable[str] = ("torch",),
             scope: str = "unit", strategy: Optional[str] = None, replace: bool = False) -> Callable:
    """Decorator on the lever's ``apply(ctx) -> Applied`` function::

        @register("pair_offload", family="offload", exact="bit-exact", exact_reason="...", applies=_pair_offload_applies,
                  preconditions=("hooks.pair_module", "torch.cuda"), settings=("rows",))
        def pair_offload(ctx) -> Applied: ...

    ``scope="process"`` for a lever that acts once per process (see :class:`Lever`); ``strategy`` = the canonical strategy id the lever's
    LEVER line carries (a kit-local lever passes ``LOCAL.<kit>.<name>``; the package's own levers are mapped in :data:`opt_core.mem.record.STRATEGY_OF`). A name registered twice is an error unless
    ``replace=True`` (a test's fixture). Returns the function unchanged, with ``function.lever`` set to the :class:`Lever`."""

    def deco(fn: Callable) -> Callable:
        lever = Lever(name=name, family=family, exact=exact, exact_reason=exact_reason, applies=applies, apply=fn,
                      description=description, preconditions=tuple(preconditions), settings=tuple(settings),
                      frameworks=tuple(frameworks), module=getattr(fn, "__module__", ""), scope=scope, strategy=strategy)
        if name in LEVERS and not replace:
            raise RegistryError(f"lever {name!r} is registered twice ({LEVERS[name].module} and {lever.module})")
        LEVERS[name] = lever
        fn.lever = lever  # type: ignore[attr-defined]
        return fn

    return deco


def unregister(name: str) -> Optional[Lever]:
    """Remove a lever (tests); returns it or None."""
    return LEVERS.pop(name, None)


def get(name: str) -> Lever:
    """The lever by name; an unknown name raises :class:`KeyError` naming the registry's contents."""
    if name not in LEVERS:
        raise KeyError(f"unknown lever {name!r} (registered: {', '.join(LEVERS) or 'none'})")
    return LEVERS[name]


def names(family: Optional[str] = None) -> tuple:
    """Registered lever names in registration order, optionally one family's."""
    return tuple(n for n, lv in LEVERS.items() if family is None or lv.family == family)


def table() -> list:
    """Every registered lever as a dict (the row form of the README's lever table)."""
    return [lv.as_dict() for lv in LEVERS.values()]


_DISCOVERED: dict = {}


def discover(modules: Iterable[str] = LEVER_MODULES, package: str = __package__) -> dict:
    """Import the lever modules of this package so their levers are registered: ``{"loaded": [...], "absent": [...], "broken": {...}}``.
    ``absent`` — the module itself is not in the install (its levers refuse by name at selection); ``broken`` — the module raised a
    ``ModuleNotFoundError`` for another name (a framework the process lacks — a torch-only module on a JAX-only machine — or a misspelt import
    inside the module), recorded with the error text: its levers refuse by name at selection and the record names the module, so the
    defect is loud without crashing a kit that does not need it. Every other import error (SyntaxError, ImportError of a name, a
    module-level exception) propagates: a broken lever module is a defect, not an absence. Verdicts are cached per process."""
    loaded, absent, broken = [], [], {}
    for m in modules:
        full = f"{package}.{m}"
        if full in _DISCOVERED:
            v = _DISCOVERED[full]
            if v is True:
                loaded.append(m)
            elif v is False:
                absent.append(m)
            else:
                broken[m] = v
            continue
        try:
            importlib.import_module(full)
        except ModuleNotFoundError as e:
            if e.name == full:
                _DISCOVERED[full] = False
                absent.append(m)
            else:
                _DISCOVERED[full] = f"{type(e).__name__}: {e}"
                broken[m] = _DISCOVERED[full]
        else:
            _DISCOVERED[full] = True
            loaded.append(m)
    return {"loaded": loaded, "absent": absent, "broken": broken}


# ------------------------------------------------------------------------------------------------------------ the context


class HookMissing(RefusalError):
    """A lever asked ``ctx.require`` for a hook the kit did not name."""


def framework(name: str, lever: str):
    """The framework module (``torch`` / ``jax``) imported on first use — the one lazy import of the lever modules (module level stays
    stdlib-only): a process without it is a :class:`RefusalError` naming the precondition ``<name>`` for ``lever``."""
    if name not in FRAMEWORKS:
        raise RegistryError(f"framework {name!r} is not one of {FRAMEWORKS}")
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise RefusalError(refuse(lever, name, f"{name} is not importable in this process ({e}): the lever needs it", error=str(e))) from None


@dataclass
class Ctx:
    """What a kit passes to the levers. ``prefix``: the kit's variable stem (``[A-Z][A-Z0-9_]*``; recorded in the block);
    ``tag``: the ``[tag]`` of its lines; ``framework``: ``torch`` or ``jax``; ``hooks``: per lever name, the engine's hook points —
    module references, tensor / attribute names, chunk dimensions, whatever that lever's docstring names; ``settings``: per lever name,
    the kit's values for the lever's declared settings (``{lever: {setting: value}}`` — the kit reads its own documented variables
    and passes what it read; a string value is cast by the lever); ``environ``: the process environment the allocator / XLA levers
    read and — ``opt_core.mem.allocator`` only, every write recorded — write (never read for the selection, a setting or the
    opt-out); ``graphs``: whether CUDA-graph capture is part of the line the mode composes on (the allocator hazard check reads it);
    ``opt_out``: the kit's census opt-out flag as its PARTIAL lines name it; ``extra``: the kit's own fields, recorded verbatim."""

    prefix: str
    tag: str
    framework: str = "torch"
    hooks: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    environ: Mapping = field(default_factory=lambda: os.environ)
    graphs: bool = False
    opt_out: str = OPT_OUT
    extra: dict = field(default_factory=dict)
    record: object = None                                  # the AppliedRecord being written (set by opt_core.mem.apply)

    def __post_init__(self):
        if not isinstance(self.prefix, str) or not re.match(r"^[A-Z][A-Z0-9_]*$", self.prefix):
            raise ValueError(f"ctx.prefix {self.prefix!r} is not [A-Z][A-Z0-9_]* (the kit's variable stem)")
        if self.framework not in FRAMEWORKS:
            raise ValueError(f"ctx.framework {self.framework!r} is not one of {FRAMEWORKS}")

    def hook(self, lever: str, key: str, default=None):
        """The kit's hook value for ``lever``/``key`` or ``default``."""
        return (self.hooks.get(lever) or {}).get(key, default)

    def require(self, lever: str, *keys: str) -> dict:
        """The named hook values, all present, or :class:`HookMissing` (a refusal naming ``hooks.<key>``) — the precondition check
        a lever runs first."""
        got = self.hooks.get(lever) or {}
        for k in keys:
            if k not in got or got[k] is None:
                raise HookMissing(refuse(lever, f"hooks.{k}", f"the kit named no {k!r} hook for lever {lever!r} in ctx.hooks",
                                         hooks_given=sorted(got)))
        return {k: got[k] for k in keys}

    def setting(self, lever: str, key: str, default=None, *, cast: Optional[Callable] = None):
        """A lever's setting value: the kit's ``ctx.settings[lever][key]`` when given — ``cast`` converts a STRING value (the kit
        passes what it read from its own variable; a failing cast is a refusal naming ``settings.<lever>.<key>``) — else
        ``default``. The value and its source (``ctx.settings`` / ``default``) are recorded on ``ctx.record`` when one is attached.
        Nothing is read from the environment."""
        lv = LEVERS.get(lever)
        if lv is not None and key not in lv.settings:
            raise RegistryError(f"lever {lever} declares no setting {key!r} (declared: {', '.join(lv.settings) or 'none'})")
        given = self.settings.get(lever) or {}
        if key in given:
            value, source = given[key], "ctx.settings"
            if cast is not None and isinstance(value, str):
                raw = value
                try:
                    value = cast(raw.strip())
                except Exception as e:  # noqa: BLE001
                    raise RefusalError(refuse(lever, setting_word(lever, key), f"{setting_ref(lever, key)}={raw!r} is not a valid {key} ({e})")) from None
        else:
            value, source = default, "default"
        rec = self.record
        if rec is not None and hasattr(rec, "note_setting"):
            rec.note_setting(lever, key, value, source)
        return value


# ------------------------------------------------------------------------------------------------------------ the selection


def setting_ref(lever: str, key: str) -> str:
    """How a refusal names a lever setting the kit passes: ``ctx.settings['<lever>']['<key>']``."""
    return f"ctx.settings[{lever!r}][{key!r}]"


def setting_word(lever: str, key: str) -> str:
    """The precondition word of a refused setting value: ``settings.<lever>.<key>``."""
    return f"settings.{lever}.{key}"


def off_ref(lever: str) -> str:
    """How a refusal names leaving a lever out: ``switches={'<lever>': False}`` (the kit's switch for it)."""
    return f"switches={{{lever!r}: False}}"


@dataclass
class Selection:
    """The levers a run applies, resolved from the kit's line and its switches: ``levers`` in order (the line's order, then the levers a
    switch turned on), ``off_by_flag`` / ``on_by_flag`` (named: left out / added by the kit's switches), ``flags`` (the switches as
    given: name -> on/off), ``refusals`` (a switch naming no lever, a value that is not on/off, an on for an unregistered lever),
    ``allow_partial`` (the kit's census opt-out)."""

    line: tuple
    levers: tuple
    off_by_flag: tuple = ()
    on_by_flag: tuple = ()
    flags: dict = field(default_factory=dict)
    refusals: list = field(default_factory=list)
    allow_partial: bool = False

    def as_dict(self) -> dict:
        return {"line": list(self.line), "levers": list(self.levers), "off_by_flag": list(self.off_by_flag),
                "on_by_flag": list(self.on_by_flag), "flags": dict(self.flags), "refusals": [r.as_dict() for r in self.refusals],
                "allow_partial": self.allow_partial}


_ON, _OFF = (True, 1, "1", "on"), (False, 0, "0", "off")


def switch_value(raw) -> Optional[bool]:
    """A switch value as on/off: ``True``/``1``/``"1"``/``"on"`` -> True, ``False``/``0``/``"0"``/``"off"`` -> False, anything else -> None."""
    v = raw.strip().lower() if isinstance(raw, str) else raw
    if any(v is x or (v == x and type(v) is type(x)) for x in _ON):
        return True
    if any(v is x or (v == x and type(v) is type(x)) for x in _OFF):
        return False
    return None


def selection(line: Sequence[str], *, switches: Optional[Mapping[str, Any]] = None, allow_partial: bool = False) -> Selection:
    """Resolve the kit's big line against its explicit switches (module contract). ``switches`` maps lever names to on/off as the kit
    parsed them from its own flags or variables (:func:`switch_value`): off for a line lever leaves it out (``off_by_flag``, named); on
    for a registered lever outside the line adds it (``on_by_flag``, named); on for a line lever and off for a lever outside the line change
    nothing; on for a lever the registry does not hold, a name that is neither registered nor on the line, and a value that is not on/off
    refuse by name. ``allow_partial`` is the kit's census opt-out (its ``--allow-partial``), recorded. A line lever the registry does not
    hold refuses by name (the lever module is absent or the name is misspelt) — never dropped silently. Nothing is read from the
    environment."""
    line = tuple(str(x) for x in line)
    if len(line) != len(set(line)):
        raise RegistryError(f"the big line repeats a lever: {line}")
    flags: dict = {}
    refusals: list = []
    off, on = [], []
    for name, raw in (switches or {}).items():
        lever = str(name)
        value = switch_value(raw)
        flags[lever] = raw if value is None else value
        known = lever in LEVERS or lever in line
        if value is None:
            refusals.append(refuse(lever if known else "big", f"switch.{lever}", f"switch {lever}={raw!r}: expected on (True/1) or off (False/0)"))
            continue
        if known:
            if not value:
                if lever in line:
                    off.append(lever)
            elif lever not in LEVERS:
                refusals.append(refuse(lever, "registry", f"switch {lever}=on names lever {lever!r}, which the registry does not hold "
                                                          f"(registered: {', '.join(LEVERS) or 'none'})"))
            elif lever not in line:
                on.append(lever)
            continue
        refusals.append(refuse("big", f"switch.{lever}", f"switch {lever!r} names no registered lever and no line lever "
                                                          f"(levers: {', '.join(LEVERS) or 'none'})"))
    for lever in line:
        if lever not in LEVERS and lever not in off:
            refusals.append(refuse(lever, "registry", f"line lever {lever!r} is not registered (its module is absent from the install "
                                                      f"or the name is misspelt; registered: {', '.join(LEVERS) or 'none'})"))
    levers = tuple(x for x in line if x not in off) + tuple(on)
    return Selection(line=line, levers=levers, off_by_flag=tuple(off), on_by_flag=tuple(on), flags=flags, refusals=refusals,
                     allow_partial=bool(allow_partial))
