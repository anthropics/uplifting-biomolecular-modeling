"""A served-shape gate with total accounting — the FAIL-LOUD primitive under every size-gated lever.

Contract. A lever that serves a kernel only inside a tested size range (a token count, a row count, a key length) decides
per call through ONE ``SizeGate`` and records what happened to EVERY call: ``served``, ``gated`` (outside the range: the stock
path runs BY DESIGN, named with the bound that excluded it) or ``fallback`` (inside the range but the kernel refused or was
unavailable: the stock path runs AS A DEGRADATION, named with the reason). Nothing is dropped: ``census()["calls"]`` equals
served + gated + fallback, and ``fields()`` renders the census as one ``key=value`` fragment the kit appends to its
activation-evidence line or its exit-tally line (``opt_core.report.register_exit_tally``) — so a run record proves both that
the lever was wired AND how often it was active. ``problems()`` is the fail-closed gate: the kit calls it at exit (or after
warm-up) with what the mode promised (``expect_served_min``, ``allow_fallback``) and turns a non-empty list into its
partial-activation exit — a fallback is never silent and never merely a log line.

    from opt_core.attn.size_gate import SizeGate, from_env
    GATE = from_env("dit_attn", "MYKIT_DIT_ATTN_MIN_TOKENS", default_min=400)      # a malformed value raises by name
    ...
    d = GATE.decide(n_tokens)              # Decision(served=True|False, event='served'|'gated:lt_min'|'gated:gt_max'); counted; d.word names the bound ('gated:gt_max(unmeasured_above_<N>)')
    if d.served:
        try:
            out = fused_path(...)
        except SomeRefusal as e:           # e.g. opt_core.attn.sdpa_bias.Refused
            GATE.fallback(e.event)         # re-books this call: served -> fallback:<event>
            out = stock_path(...)
    else:
        out = stock_path(...)
    ...
    line = report.prefix(TAG) + " " + GATE.fields()    # gate.dit_attn=min400 calls=48 served=40 gated=8 fallback=0  (report.kv bytes)
    # or, as the evidence of the per-lever line: report.kv(("name", "dit_attn"), ("state", "on"), *GATE.evidence())
    bad = GATE.problems(expect_served_min=1, allow_fallback=False)   # [] or sentences for the partial-activation exit

The bounds are inclusive (``min_tokens <= n <= max_tokens``; ``None`` = unbounded). The gate holds no framework object and
imports nothing beyond the standard library; it is thread-safe (one lock around the counters) and cheap enough to call per
attention call. ``reset()`` starts a new census (a kit that reports per item resets after each item's line).
"""
from __future__ import annotations

import os
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

SERVED = "served"
GATED_LT_MIN = "gated:lt_min"
GATED_GT_MAX = "gated:gt_max"
FALLBACK_PREFIX = "fallback:"


@dataclass(frozen=True)
class Decision:
    """One gate decision: ``served``, the event name it was counted under, and ``word`` — the event qualified by what the crossed bound IS:
    ``gated:gt_max(unmeasured_above_<N>)`` for a largest-tested ceiling (nothing was measured above N: the gate names the uncertainty, R8),
    ``gated:gt_max(measured_<N>)`` for a measured crossover; ``gated:lt_min(measured_<N>)`` / ``gated:lt_min(unmeasured_below_<N>)`` likewise."""

    served: bool
    event: str
    n: int
    bound: Optional[int] = None
    measured: bool = True

    @property
    def word(self) -> str:
        if self.event == SERVED or self.bound is None:
            return self.event
        if self.measured:
            return f"{self.event}(measured_{self.bound})"
        return f"{self.event}(unmeasured_{'above' if self.event == GATED_GT_MAX else 'below'}_{self.bound})"


@dataclass
class SizeGate:
    """A named inclusive size range with a census of every decision. ``name`` appears in the rendered fields
    (``gate.<name>=…``); keep it short and free of spaces and commas."""

    name: str
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    source: str = "default"                                   # where the bounds came from: 'default' | 'env:<VAR>' | the kit's word
    min_measured: bool = True                                 # the floor is a measured crossover (below it the fast path is measured slower / different class)
    max_measured: bool = False                                # the ceiling is a measured limit; False = a largest-tested size (nothing measured above it: named 'unmeasured')
    _counts: Counter = field(default_factory=Counter, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def __post_init__(self) -> None:
        for k, v in (("min_tokens", self.min_tokens), ("max_tokens", self.max_tokens)):
            if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < 0):
                raise ValueError(f"SizeGate {self.name!r}: {k} must be a non-negative int or None, got {v!r}")
        if self.min_tokens is not None and self.max_tokens is not None and self.min_tokens > self.max_tokens:
            raise ValueError(f"SizeGate {self.name!r}: min_tokens {self.min_tokens} > max_tokens {self.max_tokens}")
        if not self.name or any(c in self.name for c in " ,="):
            raise ValueError(f"SizeGate name must be non-empty without spaces, commas or '=': {self.name!r}")

    # ---------------------------------------------------------------- deciding
    def admits(self, n: int) -> bool:
        """The pure predicate (counts nothing): ``min_tokens <= n <= max_tokens`` with None = unbounded."""
        if self.min_tokens is not None and n < self.min_tokens:
            return False
        if self.max_tokens is not None and n > self.max_tokens:
            return False
        return True

    def decide(self, n: int) -> Decision:
        """Decide for a call of size ``n`` and COUNT the decision ('served' | 'gated:lt_min' | 'gated:gt_max')."""
        bound, measured = None, True
        if self.min_tokens is not None and n < self.min_tokens:
            ev, bound, measured = GATED_LT_MIN, self.min_tokens, self.min_measured
        elif self.max_tokens is not None and n > self.max_tokens:
            ev, bound, measured = GATED_GT_MAX, self.max_tokens, self.max_measured
        else:
            ev = SERVED
        with self._lock:
            self._counts[ev] += 1
        return Decision(served=(ev == SERVED), event=ev, n=int(n), bound=bound, measured=measured)

    def fallback(self, reason: str, rebook: bool = True) -> str:
        """Record a degradation: a call the gate served but the fast path refused (``reason`` = a short event name, e.g. the
        ``.event`` of a refusal). With ``rebook`` (the default) the call's earlier 'served' count is moved to the fallback
        bucket so the census still sums to the call count; pass ``rebook=False`` when no ``decide`` preceded it."""
        reason = str(reason or "unnamed").replace(" ", "_").replace(",", ";").replace("=", ":")
        ev = FALLBACK_PREFIX + reason
        with self._lock:
            if rebook:
                if self._counts[SERVED] <= 0:
                    raise RuntimeError(f"SizeGate {self.name!r}: fallback({reason!r}) re-books a served call but none is counted")
                self._counts[SERVED] -= 1
            self._counts[ev] += 1
        return ev

    def reset(self) -> Dict[str, int]:
        """Start a new census; returns the counts it discarded."""
        with self._lock:
            old = dict(self._counts)
            self._counts.clear()
        return old

    # ---------------------------------------------------------------- reporting
    def census(self) -> dict:
        """``{name, min, max, source, calls, served, gated, gated_by, fallback, fallback_by}`` — counts close:
        calls == served + gated + fallback."""
        with self._lock:
            c = dict(self._counts)
        gated_by = {k[len("gated:"):]: v for k, v in c.items() if k.startswith("gated:") and v}
        fallback_by = {k[len(FALLBACK_PREFIX):]: v for k, v in c.items() if k.startswith(FALLBACK_PREFIX) and v}
        served = c.get(SERVED, 0)
        gated = sum(gated_by.values())
        fallback = sum(fallback_by.values())
        return {"name": self.name, "min": self.min_tokens, "max": self.max_tokens, "source": self.source,
                "calls": served + gated + fallback, "served": served, "gated": gated, "gated_by": gated_by,
                "fallback": fallback, "fallback_by": fallback_by}

    def bounds_word(self) -> str:
        """``min400`` · ``max2048`` · ``min400:max2048`` · ``any``."""
        parts = []
        if self.min_tokens is not None:
            parts.append(f"min{self.min_tokens}")
        if self.max_tokens is not None:
            parts.append(f"max{self.max_tokens}")
        return ":".join(parts) or "any"

    def evidence(self, key_prefix: str = "gate.") -> List[tuple]:
        """The census as ordered ``(key, value)`` pairs for the line formatter (``opt_core.report.kv(*pairs)`` or the keyword
        evidence of a per-lever line): ``gate.<name>``=<bounds word>, ``calls``, ``served``, ``gated``, ``fallback``, and
        ``fallback_by`` = the {reason: n} dict (listed in full — that is the point) only when a fallback happened."""
        c = self.census()
        pairs = [(key_prefix + self.name, self.bounds_word()), ("calls", c["calls"]), ("served", c["served"]),
                 ("gated", c["gated"]), ("fallback", c["fallback"])]
        if c["fallback_by"]:
            pairs.append(("fallback_by", dict(sorted(c["fallback_by"].items()))))
        return pairs

    def lever_evidence(self) -> List[tuple]:
        """The census in the per-lever evidence grammar (``[<tag>] LEVER name=… state=… impl=… origin=… served=<n> fallback=<n>
        [fallback_by=<reason:n,…>] [min_tokens=<n>] [max_tokens=<n>] gated=<n> calls=<n>``): the ordered pairs a kit adapter
        appends after its own name/state/impl/origin pairs —
        ``report.prefix(TAG) + " LEVER " + report.kv(("name", …), ("state", "on"), …, *GATE.lever_evidence())``."""
        c = self.census()
        pairs = [("served", c["served"]), ("fallback", c["fallback"])]
        if c["fallback_by"]:
            pairs.append(("fallback_by", dict(sorted(c["fallback_by"].items()))))
        if self.min_tokens is not None:
            pairs.append(("min_tokens", self.min_tokens))
            if not self.min_measured:
                pairs.append(("min_kind", "unmeasured"))             # an unmeasured floor: kept, labelled
        if self.max_tokens is not None:
            pairs.append(("max_tokens", self.max_tokens))
            if not self.max_measured:
                pairs.append(("max_kind", "unmeasured"))             # a largest-tested ceiling: calls above it are gated BY NAME as unmeasured, not as measured-slower
        pairs += [("gated", c["gated"]), ("calls", c["calls"])]
        return pairs

    def fields(self, key_prefix: str = "gate.") -> str:
        """The census as ONE line fragment, formatted by ``opt_core.report.kv``:
        ``gate.<name>=min400 calls=48 served=40 gated=8 fallback=0`` (+ ``fallback_by=<reason>:<n>,…`` when non-zero)."""
        from opt_core import report
        return report.kv(*self.evidence(key_prefix))

    def problems(self, expect_served_min: Optional[int] = None, allow_fallback: bool = False,
                 expect_calls: Optional[int] = None) -> List[str]:
        """The fail-closed gate, as sentences (empty = clean): any fallback when ``allow_fallback`` is False; fewer served
        calls than the mode promised; a call count different from the expected one (a wiring that never reached the gate is
        caught here, not assumed)."""
        c = self.census()
        out = []
        if c["fallback"] and not allow_fallback:
            out.append(f"gate {self.name}: {c['fallback']} fallback call(s) {sorted(c['fallback_by'].items())} — the fast path "
                       f"refused inside its certified range")
        if expect_served_min is not None and c["served"] < expect_served_min:
            out.append(f"gate {self.name}: served {c['served']} call(s), the mode expects at least {expect_served_min}")
        if expect_calls is not None and c["calls"] != expect_calls:
            out.append(f"gate {self.name}: {c['calls']} call(s) reached the gate, expected {expect_calls}")
        return out


def parse_bound(text: Optional[str], var: str) -> Optional[int]:
    """``'400'`` -> 400 · ``''``/``None``/``'none'``/``'off'`` -> None (unbounded). Anything else raises ValueError naming
    ``var`` — a malformed switch is a usage error, never a silent default."""
    if text is None:
        return None
    s = str(text).strip().lower()
    if s in ("", "none", "off", "unbounded"):
        return None
    try:
        v = int(s, 10)
    except ValueError:
        raise ValueError(f"{var}={text!r}: expected a non-negative integer or none/off") from None
    if v < 0:
        raise ValueError(f"{var}={text!r}: expected a non-negative integer")
    return v


def from_env(name: str, min_var: Optional[str] = None, default_min: Optional[int] = None,
             max_var: Optional[str] = None, default_max: Optional[int] = None,
             environ: Optional[Mapping[str, str]] = None) -> SizeGate:
    """Build a gate whose bounds a kit switch may override: ``min_var`` / ``max_var`` name environment variables (unset =
    the defaults). ``source`` records which variables were set, so the activation line can say where the bound came from."""
    env = os.environ if environ is None else environ
    lo, hi, src = default_min, default_max, []
    if min_var and env.get(min_var) is not None:
        lo = parse_bound(env.get(min_var), min_var)
        src.append(min_var)
    if max_var and env.get(max_var) is not None:
        hi = parse_bound(env.get(max_var), max_var)
        src.append(max_var)
    return SizeGate(name=name, min_tokens=lo, max_tokens=hi, source=("env:" + "+".join(src)) if src else "default")


def fields_of(gates, key_prefix: str = "gate.") -> str:
    """Several gates on one line, in the given order, space-separated."""
    return " ".join(g.fields(key_prefix) for g in gates)
