"""The token-gated capture decision of a CUDA-graph / static-arena lever (standard library only).

Mechanism. A captured CUDA graph (or a static activation arena sized for replay) keeps a private memory pool the size of the captured
region's activations for as long as the graph is cached; at large token counts that pool, not the model, is what exhausts the card,
while at small token counts capture is where the speed of a fast line comes from. The kit therefore gates capture by token count: it captures
at or below a cap, runs eager above it, and says which on its activation line. This module is that decision and its line fragment,
so the gate is worded one way (``graph=capture`` | ``graph=eager:n_tok>cap`` | ``graph=eager:bytes>budget`` |
``graph=off``) and its cap knob is parsed one way. The cap VALUE (and any byte estimate) is the kit's, measured for its engine; a capture
layer CONSUMES the :class:`GateDecision` (prints ``gate=<reason>`` once on its LEVER line) and never re-derives it.

API:
    parse_cap(value, default)            env/CLI text -> cap: None/'' -> default; 'off'|'never'|'0' -> 0 (never capture);
                                         'always'|'inf'|'none' -> None (no cap); a positive integer -> that cap; else ValueError
    decide(n_tokens, cap, name='graph', enabled=True, need_bytes=None, budget_bytes=None)
                                         -> GateDecision(mode, capture, n_tokens, cap, reason, fragment, facts())
    GateDecision.fragment                the ``<name>=...`` token for the evidence line; ``.facts()`` adds ``gate=<reason>``
"""
from __future__ import annotations

from typing import Dict, Optional, Union

__all__ = ["parse_cap", "decide", "GateDecision"]


class GateDecision(object):
    """The outcome of :func:`decide` — the ONE wording of a capture gate, consumed as-is by a capture layer (it prints ``gate=<reason>``
    on its LEVER line and never re-derives the decision): ``mode`` (``capture`` | ``eager`` | ``off``), ``capture`` (bool), ``n_tokens``,
    ``cap`` (None = no cap, 0 = never), ``reason`` (``within_cap`` | ``no_cap`` | ``n_tok>cap`` | ``bytes>budget`` | ``off``), ``fragment``
    (``<name>=capture`` | ``<name>=eager:<reason>`` | ``<name>=off``) and ``facts()``."""

    __slots__ = ("name", "capture", "n_tokens", "cap", "reason", "need_bytes", "budget_bytes")

    def __init__(self, name: str, capture: bool, n_tokens: int, cap: Optional[int], reason: str,
                 need_bytes: Optional[int] = None, budget_bytes: Optional[int] = None):
        self.name, self.capture, self.n_tokens, self.cap, self.reason = str(name), bool(capture), int(n_tokens), cap, str(reason)
        self.need_bytes, self.budget_bytes = need_bytes, budget_bytes

    @property
    def mode(self) -> str:
        if self.capture:
            return "capture"
        return "off" if self.reason == "off" else "eager"

    @property
    def fragment(self) -> str:
        if self.capture:
            return f"{self.name}=capture"
        if self.reason == "off":
            return f"{self.name}=off"
        return f"{self.name}=eager:{self.reason if self.reason != 'n_tok>cap' else 'n_tok>' + str(self.cap)}"

    def facts(self) -> Dict[str, object]:
        out = {self.name: self.fragment.split("=", 1)[1], f"{self.name}_cap": ("none" if self.cap is None else self.cap),
               "n_tok": self.n_tokens, "gate": self.reason}
        if self.need_bytes is not None:
            out[f"{self.name}_need_gib"] = round(int(self.need_bytes) / 2 ** 30, 2)
            out[f"{self.name}_budget_gib"] = None if self.budget_bytes is None else round(int(self.budget_bytes) / 2 ** 30, 2)
        return out

    def __repr__(self) -> str:
        return f"GateDecision({self.fragment!r}, mode={self.mode!r}, n_tokens={self.n_tokens}, cap={self.cap!r}, reason={self.reason!r})"


def parse_cap(value: Union[str, int, None], default: Optional[int]) -> Optional[int]:
    """A cap from env/CLI text. ``None``/``''`` -> ``default``; ``off``/``never``/``0`` -> 0; ``always``/``inf``/``none``/``nocap`` -> None;
    a positive integer -> itself; anything else (negative, non-integer) -> ValueError naming the value."""
    if value is None:
        return default
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:
            raise ValueError(f"graph cap {value!r}: must be >= 0")
        return int(value)
    s = str(value).strip().lower()
    if s == "":
        return default
    if s in ("off", "never", "0", "eager"):
        return 0
    if s in ("always", "inf", "none", "nocap", "unlimited"):
        return None
    try:
        v = int(s)
    except ValueError:
        raise ValueError(f"graph cap {value!r}: expected a non-negative integer, 'off', or 'always'") from None
    if v < 0:
        raise ValueError(f"graph cap {value!r}: must be >= 0")
    return v


def decide(n_tokens: int, cap: Optional[int], name: str = "graph", enabled: bool = True, need_bytes: Optional[int] = None,
           budget_bytes: Optional[int] = None) -> GateDecision:
    """The capture decision of a graph / static-arena lever. Token gate: capture iff ``enabled`` and (``cap`` is None or
    ``n_tokens <= cap``); ``cap == 0`` or ``enabled=False`` is ``off``. Optional byte gate on top: when ``need_bytes`` (the kit's estimate of
    the captured region's pool) and ``budget_bytes`` (e.g. ``opt_core.mem.budget.budget_bytes(frac)``) are both given and
    ``need_bytes > budget_bytes``, the decision is eager with reason ``bytes>budget``. The decision is made once, here; a capture layer
    consumes the returned :class:`GateDecision`."""
    n = int(n_tokens)
    if not enabled or cap == 0:
        return GateDecision(name, False, n, 0 if cap == 0 else cap, "off", need_bytes, budget_bytes)
    c = None if cap is None else int(cap)
    if c is not None and n > c:
        return GateDecision(name, False, n, c, "n_tok>cap", need_bytes, budget_bytes)
    if need_bytes is not None and budget_bytes is not None and int(need_bytes) > int(budget_bytes):
        return GateDecision(name, False, n, c, "bytes>budget", need_bytes, budget_bytes)
    return GateDecision(name, True, n, c, "no_cap" if c is None else "within_cap", need_bytes, budget_bytes)
