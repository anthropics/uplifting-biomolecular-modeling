"""Named events of the host-side primitives: the one form every refusal, bypass, fallback and mismatch takes.

Contract. A :class:`HostEvent` is ``(lever, event, reason, fields)`` with ``EVENT`` one of :data:`EVENTS`. Two line forms, both in
the :mod:`opt_core.report` grammar (``prefix`` + ``kv``): the kit's per-lever ACTIVATION line is the core's LEVER form —
:meth:`HostEvent.lever_line` ``(tag, name)`` → ``[<tag>] LEVER name=<F6.strategy> state=on impl=<lever> origin=core k=v ...`` for ``ACTIVE``
and ``... state=skipped reason=<reason> impl=<lever> origin=core ...`` for ``REFUSED`` (the strategy id is the adapter's); the
census and event RECORDS (``TALLY`` · ``BYPASS`` · ``FALLBACK`` · ``MISMATCH``) are :meth:`HostEvent.line` ``(tag)`` →
``[<tag>] HOST <lever> <EVENT> reason=<reason> k=v ...``, so a launcher greps one shape for every host-side record. :class:`HostRefused` carries the event of a precondition the primitive will not work around (no CUDA
for a pinned pool, a torch too old for ``mmap``, a checkpoint that does not cover the model, an input the memo cannot digest): the
adapter catches it, logs ``exc.event.line(tag)``, and either stops (the kit's NOT ACTIVE exit) or takes its stock path AS A RECORDED
CHOICE (``--allow-partial`` class) — the record is the point. :func:`framework` / :func:`require` are the lazy-import helpers the
framework-touching functions use: the module if importable (and new enough), else a ``REFUSED reason=missing:<name>`` /
``reason=too_old:<name><found><<need>`` event raised as :class:`HostRefused`.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .. import report

EVENTS = ("ACTIVE", "TALLY", "REFUSED", "BYPASS", "FALLBACK", "MISMATCH")


@dataclass
class HostEvent:
    lever: str                                   # host.memo · host.workers · host.outputs · host.coldstart
    event: str                                   # one of EVENTS
    reason: str = ""                             # a short machine word: no_cuda · missing:torch · undigestible:<type> · missing_keys ...
    fields: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event not in EVENTS:
            raise ValueError(f"unknown host event {self.event!r} (known: {', '.join(EVENTS)})")

    def pairs(self) -> List[Tuple[str, Any]]:
        """``(key, value)`` pairs in print order: reason first (when set), then the fields."""
        out: List[Tuple[str, Any]] = []
        if self.reason:
            out.append(("reason", self.reason))
        out.extend(self.fields.items())
        return out

    def line(self, tag: str) -> str:
        """``[<tag>] HOST <lever> <EVENT> reason=... k=v ...`` — the record form (census, bypass, fallback, mismatch)."""
        body = report.kv(*self.pairs())
        return f"{report.prefix(tag)} HOST {self.lever} {self.event}" + (" " + body if body else "")

    def lever_line(self, tag: str, name: str, origin: str = "core") -> str:
        """The per-lever activation line in the core's LEVER form: ``[<tag>] LEVER name=<name> state=on impl=<lever> origin=<origin> k=v ...``
        (``ACTIVE``) or ``... state=skipped reason=<reason> impl=<lever> origin=<origin> k=v ...`` (``REFUSED``). ``name`` is the strategy id
        (``F6.feature_cache``, ...); ``origin`` is ``core`` for these primitives (a kit re-using this form for a kit-local lever passes ``kit``)."""
        if origin not in ("core", "kit"):
            raise ValueError("origin is 'core' or 'kit'")
        if self.event == "ACTIVE":
            head = [("name", name), ("state", "on")]
        elif self.event == "REFUSED":
            head = [("name", name), ("state", "skipped"), ("reason", self.reason or "refused")]
        else:
            raise ValueError(f"{self.event} is a record (use .line), not an activation state")
        fields = [(k, v) for k, v in self.fields.items()]
        return f"{report.prefix(tag)} LEVER " + report.kv(*(head + [("impl", self.lever), ("origin", origin)] + fields))

    def as_dict(self) -> dict:
        return {"lever": self.lever, "event": self.event, "reason": self.reason, "fields": dict(self.fields)}


class HostRefused(RuntimeError):
    """A named precondition failure. ``.event`` is the :class:`HostEvent` (``REFUSED``); ``str(exc)`` is its tag-less line body."""

    def __init__(self, lever: str, reason: str, **fields: Any) -> None:
        self.event = HostEvent(lever, "REFUSED", reason, dict(fields))
        super().__init__(f"HOST {lever} REFUSED " + report.kv(*self.event.pairs()))


def _version_tuple(text: str) -> Tuple[int, ...]:
    out = []
    for part in str(text).split("+")[0].split("."):
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def framework(name: str) -> Optional[Any]:
    """The module ``name`` if it imports, else ``None`` — the probe form (no refusal)."""
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def require(lever: str, name: str, min_version: Optional[str] = None) -> Any:
    """The module ``name`` (imported now, inside the caller), at least ``min_version`` — else :class:`HostRefused`
    ``reason=missing:<name>`` / ``reason=too_old:<name>`` with ``found`` and ``need`` fields."""
    mod = framework(name)
    if mod is None:
        raise HostRefused(lever, f"missing:{name}")
    if min_version is not None:
        found = getattr(mod, "__version__", "0")
        if _version_tuple(found) < _version_tuple(min_version):
            raise HostRefused(lever, f"too_old:{name}", found=found, need=min_version)
    return mod
