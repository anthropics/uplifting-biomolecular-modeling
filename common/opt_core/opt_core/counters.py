"""Kernel-call counters: the per-process ledger of one lever and the arithmetic over call censuses.

Contract. A :class:`Ledger` is the ONE counting class of a lever in a process (thread-safe): every call the lever sees is counted once as
``served:<impl>`` (the kernel ran), ``fallback:<reason>`` (a declared guard sent the call to the engine's own code — the reason is the
kit's word) or ``error:<Type>`` (the kernel raised and the call went to the engine's code for THIS call); served shapes are censused up to
``max_shapes`` distinct keys; free-form facts (``count`` / ``peak`` / ``set``: bytes moved, peak rows, a threshold read back) ride along;
:meth:`Ledger.word` records a fact that is a named UNCERTAINTY the lever engaged under (``settings=safe:<reason>``,
``card_support=uncertified:<sm>`` — :mod:`opt_core.report`'s ``words``): it prints on the line like any fact, :meth:`Ledger.words` lists
it for the kit's activation line and exit record, and it never refuses the gate (an ACTIVE lever, exit 0).
The ledger renders the lever's activation-evidence line in the core's LEVER grammar (:func:`opt_core.report.lever_line`; the key order is
fixed there) and a fail-closed :meth:`Ledger.gate`: refused on any error, on any fallback reason outside the kit's ``expected`` list, and
when calls arrived and none was served. A census is plain data — ``{lever: {"<kind>:<detail>": n, ...}}`` — and the module functions are
its arithmetic: :func:`events_of` keeps the event keys, :func:`events_delta` is one item's increase, :func:`events_totals` sums items,
:func:`split_expected` splits events against the kit's expected table. No lever name, reason word or threshold lives here: they are the
kit's data.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SERVED, FALLBACK, ERROR = "served", "fallback", "error"          # the three kinds of a counted call: "<kind>:<detail>"
STOCK_BY_CELL_ASIDE = "stock_row_by_cell"                           # the named aside of a served-floor gate whose every call was ANSWERED with the cell's stock row
STOCK_BY_CELL_HEADS = ("stock", "statement", "stock_statement", "stock_row", "stock_row_by_cell", "stock_op", "library_op", "cueq", "torch_math")
                                                                    # the head token (before ':' / '(' / whitespace) of a fallback reason that says: the provider named
                                                                    # the STOCK row / the library op / the caller's own statement for this call BY NAME (the cell's decision)


def is_stock_by_cell(reason: Any) -> bool:
    """True when a fallback ``reason`` says the provider ANSWERED the call by naming the cell's stock row / the library op / the caller's own
    statement (``stock``, ``stock:<row>``, ``stock_statement``, ``statement``, ``stock_row...``, ``cueq:...``, ``torch_math:...``) -- the
    cell's decision, not a launch fallback: ``error:<Exc>`` reroutes, ``row_error``, ``below_min_tokens``, refusals by shape / dtype /
    build are NOT."""
    r = str(reason or "").strip().lower()
    if not r:
        return False
    head = r
    for sep in (":", "(", " ", "="):
        head = head.split(sep, 1)[0]
    return head in STOCK_BY_CELL_HEADS


def stock_by_cell_aside(fallbacks: Mapping[str, int], served: int, errors: Mapping[str, int]) -> Optional[str]:
    """``STOCK_BY_CELL_ASIDE`` when calls arrived, NONE was served, NO error was counted and EVERY fallback reason is a stock-by-cell answer
    (:func:`is_stock_by_cell`) -- the lever routed its provider and the provider answered every call with the cell's stock row by name: a
    named aside, not a served-floor violation.  None otherwise (a launch fallback, an error, a size gate, a refusal by shape keep the
    violation)."""
    if served or errors or not fallbacks or sum(int(n) for n in fallbacks.values()) <= 0:
        return None
    return STOCK_BY_CELL_ASIDE if all(is_stock_by_cell(r) for r, n in fallbacks.items() if int(n) > 0) else None
STATES = ("on", "off", "skipped")                                 # the state word of the LEVER line (report.lever_line)
EVENT_PREFIXES = (FALLBACK + ":", ERROR + ":")                    # the keys of a census that are events (a served call is not an event)


def _key(kind: str, detail: Any) -> str:
    return f"{kind}:{detail}"


class Ledger:
    """The per-process counters of ONE lever. ``name`` is the lever id printed as ``name=`` (``F<k>.<strategy>`` or the kit's local id),
    ``impl`` the kernel or module (``<module>@<version>``), ``origin`` ``core`` | ``kit`` (where the implementation lives), ``min_tokens`` the
    kit's size gate (printed when given), ``expected`` the fallback reasons the kit's mode DECLARES (anything else refuses :meth:`gate`),
    ``max_shapes`` the cap on distinct shape keys censused (further shapes are served, not listed)."""

    def __init__(self, name: str, *, impl: Optional[str] = None, origin: Optional[str] = None, min_tokens: Optional[int] = None,
                 expected: Iterable[str] = (), max_shapes: int = 8):
        self.name = str(name)
        self.impl = impl
        self.origin = origin
        self.min_tokens = min_tokens
        self.expected = tuple(expected)
        self.max_shapes = int(max_shapes)
        self._lock = threading.Lock()
        self._counts: Dict[str, int] = {}
        self._shapes: Dict[str, int] = {}
        self._facts: Dict[str, Any] = {}
        self._words: List[str] = []                                 # the fact keys that are words (named uncertainty), in recording order
        self.first: Optional[dict] = None                           # the first served call's facts, when the caller records them

    # ------------------------------------------------------------------------------------------------------------- events
    def _add(self, key: str, n: int) -> int:
        with self._lock:
            v = self._counts.get(key, 0) + int(n)
            self._counts[key] = v
            return v

    def serve(self, shape: Optional[str] = None, *, impl: Optional[str] = None, n: int = 1, first: Optional[Mapping] = None) -> int:
        """One served call (``served:<impl>``; the ledger's ``impl`` unless given). ``shape`` is censused; ``first`` is kept for the first
        served call only. Returns the served count of that impl."""
        v = self._add(_key(SERVED, impl if impl is not None else (self.impl if self.impl is not None else "kernel")), n)
        if shape is not None or (first is not None and self.first is None):
            with self._lock:
                if shape is not None and (shape in self._shapes or len(self._shapes) < self.max_shapes):
                    self._shapes[shape] = self._shapes.get(shape, 0) + int(n)
                if first is not None and self.first is None:
                    self.first = dict(first)
        return v

    def fallback(self, reason: str, n: int = 1) -> int:
        """One call sent to the engine's own code by a guard; ``reason`` is the kit's word (``below_min_tokens``, ``dtype``, ``mode_stock`` ...)."""
        return self._add(_key(FALLBACK, reason), n)

    def error(self, exc: Any, n: int = 1) -> int:
        """One call whose kernel raised (``error:<Type>``); ``exc`` is the exception or its type name."""
        name = exc if isinstance(exc, str) else (exc.__name__ if isinstance(exc, type) else type(exc).__name__)
        return self._add(_key(ERROR, name), n)

    # ------------------------------------------------------------------------------------------------------------- free-form facts
    def count(self, key: str, n: int = 1) -> int:
        with self._lock:
            v = int(self._facts.get(key, 0) or 0) + int(n)
            self._facts[key] = v
            return v

    def peak(self, key: str, value: Any) -> Any:
        with self._lock:
            cur = self._facts.get(key)
            if cur is None or value > cur:
                self._facts[key] = value
            return self._facts[key]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._facts[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._facts.get(key, default)

    def word(self, key: str, value: Any) -> str:
        """Record the fact ``key=value`` AS A WORD (module contract): a named uncertainty the lever engaged under. Returns the word."""
        from .report import word as _word
        with self._lock:
            self._facts[key] = value
            if key not in self._words:
                self._words.append(key)
        return _word(key, value)

    def words(self) -> List[str]:
        """The recorded words, ``["<key>=<value>", ...]`` in recording order (:func:`opt_core.report.word`'s grammar)."""
        from .report import word as _word
        with self._lock:
            return [_word(k, self._facts[k]) for k in self._words if k in self._facts]

    # ------------------------------------------------------------------------------------------------------------- census
    def counts(self) -> Dict[str, int]:
        """``{"<kind>:<detail>": n}`` — the census form the module functions read."""
        with self._lock:
            return dict(sorted(self._counts.items()))

    def facts(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._facts)

    def _by_kind(self, kind: str) -> Dict[str, int]:
        p = kind + ":"
        with self._lock:
            return dict(sorted((k[len(p):], v) for k, v in self._counts.items() if k.startswith(p)))

    @property
    def served(self) -> int:
        return sum(self._by_kind(SERVED).values())

    @property
    def fallbacks(self) -> Dict[str, int]:
        return self._by_kind(FALLBACK)

    @property
    def errors(self) -> Dict[str, int]:
        return self._by_kind(ERROR)

    @property
    def shapes(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._shapes)

    @property
    def calls(self) -> int:
        with self._lock:
            return sum(self._counts.values())

    @property
    def partial(self) -> bool:
        """Calls arrived and none was served: the lever is not active in this process (the kit's exit rule prices it)."""
        c = self.counts()
        return bool(c) and not any(k.startswith(SERVED + ":") for k in c)

    @property
    def state(self) -> str:
        """``on`` = served at least once in this process; ``skipped`` = never served (no calls, or every call fell back or raised)."""
        return "on" if self.served > 0 else "skipped"

    def skip_reason(self) -> Optional[str]:
        """The ``reason=`` of a lever that is not on: ``no_calls``, ``all_fallback:<most frequent reason>`` or ``all_error:<most frequent type>``."""
        if self.served > 0:
            return None
        fb, err = self.fallbacks, self.errors
        if not fb and not err:
            return "no_calls"
        kind, table = ("all_fallback", fb) if sum(fb.values()) >= sum(err.values()) else ("all_error", err)
        top = sorted(table.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        return f"{kind}:{top}"

    def unexpected(self) -> Dict[str, int]:
        """The fallback reasons counted that the kit did not declare in ``expected``."""
        return {r: n for r, n in self.fallbacks.items() if r not in self.expected}

    def fields(self) -> Dict[str, Any]:
        """Plain data for a manifest: the line's keys plus ``errors``, ``partial``, ``calls``, ``first``, the free-form facts under ``facts``
        and the ``words`` among them."""
        fb = self.fallbacks
        return {"name": self.name, "impl": self.impl, "origin": self.origin, "state": self.state, "served": self.served,
                "fallback": sum(fb.values()), "fallback_by": fb, "errors": self.errors, "min_tokens": self.min_tokens,
                "shapes": self.shapes, "partial": self.partial, "calls": self.calls, "first": self.first, "facts": self.facts(),
                "words": self.words()}

    def clear(self) -> None:
        with self._lock:
            self._counts.clear()
            self._shapes.clear()
            self._facts.clear()
            self._words.clear()
            self.first = None

    # ------------------------------------------------------------------------------------------------------------- verdicts
    def gate(self, name: Optional[str] = None, *, require_served: bool = True):
        """Fail-closed (:class:`opt_core.gates.Gate`) on the ACCOUNTING of this lever: refused on any ``error:*``, on any fallback reason
        outside ``expected`` (an undeclared reason is a census that does not close), and — unless ``require_served=False`` (a mode that
        routes no kernel) — when calls arrived and none was served (a promised lever that never served). The recorded :meth:`words` ride
        on the gate, passing or refused, and never refuse it. ``details`` = :meth:`fields`."""
        from .gates import Gate
        from .report import kv
        f = self.fields()
        gname = name or self.name
        words = tuple(f["words"])
        if f["errors"]:
            return Gate(name=gname, ok=False, reason="kernel error: " + kv(errors=f["errors"]), details=f, words=words)
        unexp = self.unexpected()
        if unexp:
            return Gate(name=gname, ok=False, reason="unexpected fallback: " + kv(fallback_by=dict(sorted(unexp.items()))), details=f, words=words)
        if require_served and f["calls"] > 0 and f["served"] == 0:
            aside = stock_by_cell_aside(f["fallback_by"], f["served"], f["errors"])
            if aside:                                                # every call ANSWERED with the cell's stock row by name, no launch fallback / error: a named aside, the gate passes
                return Gate(name=gname, ok=True, details=dict(f, aside=aside), words=words + (("aside:" + aside,) if ("aside:" + aside) not in words else ()))
            return Gate(name=gname, ok=False, reason=f"{f['calls']} calls, 0 served ({self.skip_reason()})", details=f, words=words)
        return Gate(name=gname, ok=True, details=f, words=words)

    # ------------------------------------------------------------------------------------------------------------- the line
    def pairs(self, state: Optional[str] = None, reason: Optional[str] = None) -> List[Tuple[str, Any]]:
        """The evidence pairs after ``name``/``state``/``reason`` in the LEVER grammar's order: impl, origin, served, fallback, fallback_by,
        [errors], [min_tokens], shapes, then the free-form facts sorted by key."""
        f = self.fields()
        out: List[Tuple[str, Any]] = [("impl", f["impl"]), ("origin", f["origin"]), ("served", f["served"]), ("fallback", f["fallback"]),
                                      ("fallback_by", f["fallback_by"])]
        if f["errors"]:
            out.append(("errors", f["errors"]))
        if f["min_tokens"] is not None:
            out.append(("min_tokens", f["min_tokens"]))
        out.append(("shapes", f["shapes"]))
        out += sorted(f["facts"].items())
        return out

    def line(self, tag: str, state: Optional[str] = None, reason: Optional[str] = None, **evidence) -> str:
        """This lever's ONE activation-evidence line in this process (:func:`opt_core.report.lever_line`). ``state`` defaults to :attr:`state`,
        ``reason`` to :meth:`skip_reason` when the state is not ``on``; a kit whose mode does not select the lever prints ``state="off"`` with
        its reason (the same line, zero counters). ``evidence`` keys follow the ledger's pairs."""
        from .report import lever_line
        st = state if state is not None else self.state
        why = reason if reason is not None else (self.skip_reason() if st != "on" else None)
        return lever_line(tag, self.name, st, *self.pairs(), reason=why, **evidence)


# ----------------------------------------------------------------------------------------------------------------- census arithmetic
def events_of(census: Optional[Mapping], *, prefixes: Sequence[str] = EVENT_PREFIXES, keys: Sequence[str] = (),
              skip: Iterable[str] = ()) -> Dict[str, Dict[str, int]]:
    """``{lever: {key: n}}`` keeping, per lever, the counters whose key starts with one of ``prefixes`` or is one of ``keys``; levers named in
    ``skip`` (census entries that are not levers: the kit's words) and non-dict entries are left out; levers with no event are left out."""
    skip = set(skip)
    out: Dict[str, Dict[str, int]] = {}
    for lever, counts in (census or {}).items():
        if lever in skip or not isinstance(counts, Mapping):
            continue
        ev = {k: v for k, v in counts.items() if k in keys or any(k.startswith(p) for p in prefixes)}
        if ev:
            out[str(lever)] = dict(ev)
    return out


def events_delta(before: Optional[Mapping], after: Optional[Mapping], *, dead_key: Optional[str] = "dead",
                 **events_kw) -> Tuple[Dict[str, Dict[str, int]], List[str]]:
    """``(delta, dead)``: the events of one item — the increase of every event counter from ``before`` to ``after`` (positive entries only;
    ``events_kw`` goes to :func:`events_of`) — and the sorted names under ``after[dead_key]`` (the levers the kit marked dead; [] without one)."""
    b, a = events_of(before, **events_kw), events_of(after, **events_kw)
    delta: Dict[str, Dict[str, int]] = {}
    for lever, counts in a.items():
        d = {k: v - b.get(lever, {}).get(k, 0) for k, v in counts.items() if v - b.get(lever, {}).get(k, 0) > 0}
        if d:
            delta[lever] = d
    dead = sorted((after or {}).get(dead_key) or {}) if dead_key else []
    return delta, [str(x) for x in dead]


def events_totals(deltas: Iterable[Mapping]) -> Dict[str, Dict[str, int]]:
    """The per-lever, per-key sum of a sequence of ``{lever: {key: n}}`` (the pass total of the items' deltas)."""
    out: Dict[str, Dict[str, int]] = {}
    for d in deltas:
        for lever, counts in (d or {}).items():
            t = out.setdefault(str(lever), {})
            for k, v in counts.items():
                t[k] = t.get(k, 0) + int(v)
    return out


def split_expected(events: Optional[Mapping], expected: Mapping[str, Iterable[str]]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    """``(expected, unexpected)``: ``events`` (``{lever: {key: n}}``) split by whether the kit's table lists the key for the lever."""
    exp: Dict[str, Dict[str, int]] = {}
    unexp: Dict[str, Dict[str, int]] = {}
    for lever, counts in (events or {}).items():
        allowed = set(expected.get(lever, ()))
        for k, v in counts.items():
            (exp if k in allowed else unexp).setdefault(str(lever), {})[k] = v
    return exp, unexp
