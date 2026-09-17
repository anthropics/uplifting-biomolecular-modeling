"""The attention sub-batch (row-chunk) decision for one traced executable — engine-free.

A kit whose engine chunks row-batched attention by a configuration value fixed at trace time (an integer = that many rows per chunk,
``None`` = unchunked; the kit cites its engine's reading site) decides that value here. Chunking bounds the per-call logits transient; it
costs launches and, under a differentiated executable, re-associated accumulation (so 'unchunked where it fits' is a fast-class lever
wherever it changes the traced program, and the equality of the engine's stock rule where it does not). This module holds the DECISION,
not the patch: the kit adapter supplies the values (token count, the device's byte limit as jax reports it, its own measured peak points,
its engine's stock chunk), receives a :class:`SubbatchDecision`, writes ``decision.value`` into the config object it owns before the
executable is traced, and renders ``decision.fields()`` (a dict) into its ONE activation-evidence line with ``opt_core.report.kv``
(grammar ``[<tag>] <VERB> key=value ...`` — the sentence bytes stay the kit's).

    from opt_core import report
    from opt_core.jax_design.subbatch_policy import choose, parse_request, QuadraticPeak
    PEAK = QuadraticPeak(a=..., b=..., c=..., margin=0.10)             # the ADAPTER's measured fit (bytes = (a t^2 + b t + c) * (1 + margin))
    d = choose(tokens=L, requested=parse_request(os.environ.get(KIT_VAR), "auto"), device_bytes=total_or_None,
               peak_estimator=PEAK, stock_value=4, when_device_unknown="stock")   # stock_value: the ENGINE's chunk, cited by the kit
    cfg.<the engine's subbatch field> = d.value                        # before the executable is traced
    line_fragment = report.kv(**d.fields())                            # e.g. "subbatch=none subbatch_source=auto:fits subbatch_est_gb=22.9"

Every outcome is a NAMED source — there is no silent branch:

    requested          the caller asked for a value (an int, or None = unchunked); the estimate is not consulted
    auto:fits          estimated peak <= device bytes → unchunked (None)
    auto:exceeds       estimated peak >  device bytes → ``stock_value`` (the memory-saving chunk = stock behaviour)
    auto:no_device     no device size available → ``stock_value`` when ``when_device_unknown='stock'`` (the conservative default: the
                       measured fit is not consulted without a device size — off by name, not measured there) or None when ``'unchunked'``
                       (a kit that has always behaved so passes it explicitly); ``decision.words()`` = ``[subbatch=no_device(<side>)]``
                       is the named uncertainty for the kit's activation line
    kit                :func:`fixed`: the adapter's shipped constant decision (an int, or None = unchunked); no estimate, no device size consulted
    stock              ``requested='stock'``: the engine's own size rule, supplied as ``stock_rule(tokens)`` — the forward-only
                       executable of a design kit keeps this so predict-only numerics do not move

Standard library only; nothing here imports jax.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple, Union

Request = Union[None, int, str]          # None (unchunked) · int (that chunk) · 'auto' · 'stock'
SOURCES = ("requested", "auto:fits", "auto:exceeds", "auto:no_device", "stock", "kit")
WHEN_DEVICE_UNKNOWN = ("stock", "unchunked")
_NONE_WORDS = ("none", "null", "0", "", "unchunked")


class PolicyError(ValueError):
    """A malformed request or an inconsistent argument set — a usage error of the adapter, raised, never absorbed."""


def parse_request(text: Optional[str], default: Request = "auto") -> Request:
    """The kit variable's text → a request. ``None``/absent → ``default``; 'none'|'null'|'0'|''|'unchunked' → None (unchunked);
    'auto' → 'auto'; 'stock' → 'stock'; a positive integer → that chunk. Anything else is a :class:`PolicyError` naming the text."""
    if text is None:
        return default
    v = str(text).strip().lower()
    if v in _NONE_WORDS:
        return None
    if v in ("auto", "stock"):
        return v
    try:
        n = int(v)
    except ValueError:
        raise PolicyError(f"sub-batch request {text!r} is not one of: auto, stock, none, <positive integer>") from None
    if n <= 0:
        raise PolicyError(f"sub-batch request {text!r}: a chunk is a positive integer (0 and 'none' mean unchunked)")
    return n


@dataclass(frozen=True)
class QuadraticPeak:
    """Estimated peak bytes of the executable at ``tokens``: ``(a*t*t + b*t + c) * unit * (1 + margin)``. The coefficients are the
    ADAPTER's fit through its own measured points (state them beside the measured values in the kit's docs); ``unit`` scales the polynomial
    (1e9 when a, b, c are written in GB); ``margin`` is the safety fraction added on top."""

    a: float
    b: float
    c: float
    unit: float = 1e9
    margin: float = 0.10

    def __call__(self, tokens: int) -> int:
        t = float(tokens)
        return int((self.a * t * t + self.b * t + self.c) * self.unit * (1.0 + self.margin))

    def describe(self) -> str:
        return f"({self.a:g}*t^2+{self.b:g}*t+{self.c:g})*{self.unit:g}*(1+{self.margin:g})"


def fit_quadratic(points: Sequence[Tuple[float, float]], *, unit: float = 1.0, margin: float = 0.10) -> QuadraticPeak:
    """Least-squares quadratic through >= 3 measured ``(tokens, peak)`` points (peak in ``unit``s; e.g. GB with unit=1e9). Pure Python
    normal equations (3x3); exact through three points. The adapter measures the points on its pinned stack and cites them."""
    pts = [(float(t), float(p)) for t, p in points]
    if len(pts) < 3:
        raise PolicyError(f"fit_quadratic needs >= 3 (tokens, peak) points, got {len(pts)}")
    # normal equations for [a, b, c] over basis [t^2, t, 1]
    s = [[0.0] * 3 for _ in range(3)]
    r = [0.0, 0.0, 0.0]
    for t, p in pts:
        basis = (t * t, t, 1.0)
        for i in range(3):
            r[i] += basis[i] * p
            for j in range(3):
                s[i][j] += basis[i] * basis[j]
    a, b, c = _solve3(s, r)
    return QuadraticPeak(a=a, b=b, c=c, unit=unit, margin=margin)


def _solve3(m, r):
    """Gaussian elimination with partial pivoting on a 3x3 system (raises PolicyError when singular, e.g. repeated token counts)."""
    a = [row[:] + [r[i]] for i, row in enumerate(m)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda i: abs(a[i][col]))
        if abs(a[piv][col]) < 1e-300:
            raise PolicyError("fit_quadratic: singular system (need three distinct token counts)")
        a[col], a[piv] = a[piv], a[col]
        for i in range(3):
            if i != col:
                f = a[i][col] / a[col][col]
                for j in range(col, 4):
                    a[i][j] -= f * a[col][j]
    return tuple(a[i][3] / a[i][i] for i in range(3))


@dataclass
class SubbatchDecision:
    """One decision. ``value`` is what the adapter writes into the config (None = unchunked, int = chunk rows); ``source`` is one of
    :data:`SOURCES`; the values are those the decision used (None when not consulted). ``fields()`` is the census dict for the kit's line
    (the kit renders it with ``opt_core.report.kv``); ``as_dict()`` goes into the kit's manifest."""

    value: Optional[int]
    source: str
    tokens: int
    stock_value: Optional[int]
    estimated_bytes: Optional[int] = None
    device_bytes: Optional[int] = None
    requested: Request = "auto"
    details: dict = field(default_factory=dict)

    def fields(self, prefix: str = "subbatch") -> dict:
        """``{<prefix>: 'none'|<n>, <prefix>_source: <source>[, <prefix>_est_gb: <x.y>][, <prefix>_dev_gb: <n>]}`` — insertion-ordered,
        values rounded for the line (GB); the exact byte counts are in ``as_dict()``."""
        out = {prefix: "none" if self.value is None else self.value, f"{prefix}_source": self.source}
        if self.estimated_bytes is not None:
            out[f"{prefix}_est_gb"] = round(self.estimated_bytes / 1e9, 1)
        if self.device_bytes is not None:
            out[f"{prefix}_dev_gb"] = int(round(self.device_bytes / 1e9))
        return out

    def words(self, prefix: str = "subbatch") -> list:
        """The named uncertainty of this decision (:mod:`opt_core.report` ``words``): ``[<prefix>=no_device(<stock|unchunked>)]`` when
        ``auto`` could not read the device's memory and took the ``when_device_unknown`` side by name — the measured fit was not
        consulted, the run proceeds, exit 0; ``[]`` for every other source (their ``<prefix>_source`` field is the whole story)."""
        from ..report import word as _word  # noqa: PLC0415
        if self.source == "auto:no_device":
            return [_word(prefix, "no_device", self.details.get("when_device_unknown", "stock"))]
        return []

    def reason(self) -> str:
        """A sentence for the kit's log (the line carries ``fields()``; this is the human gloss)."""
        v = "unchunked" if self.value is None else f"chunks of {self.value}"
        if self.source == "requested":
            return f"sub-batch {v}: requested"
        if self.source == "stock":
            return f"sub-batch {v}: the stock size rule at {self.tokens} tokens"
        if self.source == "kit":
            return f"sub-batch {v}: the kit's shipped decision"
        est = f"{(self.estimated_bytes or 0) / 1e9:.1f} GB"
        if self.source == "auto:fits":
            return f"sub-batch {v}: estimated forward+backward peak {est} fits {self.device_bytes / 1e9:.0f} GB"
        if self.source == "auto:exceeds":
            return f"sub-batch {v}: estimated forward+backward peak {est} exceeds {self.device_bytes / 1e9:.0f} GB — keeping the stock chunk"
        return f"sub-batch {v}: no device size available (policy when_device_unknown)"

    def as_dict(self) -> dict:
        return {"value": self.value, "source": self.source, "tokens": self.tokens, "stock_value": self.stock_value,
                "estimated_bytes": self.estimated_bytes, "device_bytes": self.device_bytes,
                "requested": self.requested if not isinstance(self.requested, str) else str(self.requested), "details": dict(self.details)}


def fixed(*, tokens: int, value: Optional[int], stock_value: Optional[int]) -> SubbatchDecision:
    """The adapter's shipped constant decision for ONE executable traced at ``tokens``: ``value`` (an int chunk, or None = unchunked) with
    source ``kit`` — no estimate and no device size are consulted (a kit whose measured peaks show chunking saves nothing ships this)."""
    if int(tokens) <= 0:
        raise PolicyError(f"tokens must be a positive count, got {tokens!r}")
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
        raise PolicyError(f"value must be a positive int (a chunk) or None (unchunked), got {value!r}")
    return SubbatchDecision(value=value, source="kit", tokens=int(tokens), stock_value=stock_value, requested=value)


def choose(*, tokens: int, stock_value: Optional[int], requested: Request = "auto", device_bytes: Optional[int] = None,
           peak_estimator: Optional[Callable[[int], int]] = None,
           stock_rule: Optional[Callable[[int], Optional[int]]] = None, when_device_unknown: str = "stock") -> SubbatchDecision:
    """Decide the sub-batch for ONE executable traced at ``tokens``.

    stock_value          REQUIRED: the engine's memory-saving chunk as the kit cites it (what 'auto:exceeds' and 'auto:no_device' with
                         ``when_device_unknown='stock'`` resolve to); the core holds no engine's constant. None only for an engine whose
                         stock is unchunked.
    requested            None | int → that value, source 'requested'. 'stock' → ``stock_rule(tokens)`` (required then), source 'stock'.
                         'auto' → the estimate against the device (``peak_estimator`` required).
    device_bytes         the device's byte limit as the adapter read it (jax memory_stats 'bytes_limit'), or None when unknown.
    peak_estimator       callable tokens → estimated peak bytes of THIS executable (a :class:`QuadraticPeak` or the adapter's own).
    when_device_unknown  'stock' (default) or 'unchunked' — what 'auto' does without a device size; either way the source says so.
    """
    if int(tokens) <= 0:
        raise PolicyError(f"tokens must be a positive count, got {tokens!r}")
    tokens = int(tokens)
    if stock_value is not None and (not isinstance(stock_value, int) or isinstance(stock_value, bool) or stock_value <= 0):
        raise PolicyError(f"stock_value must be a positive int (the engine's chunk) or None, got {stock_value!r}")
    if when_device_unknown not in WHEN_DEVICE_UNKNOWN:
        raise PolicyError(f"when_device_unknown={when_device_unknown!r} is not one of {WHEN_DEVICE_UNKNOWN}")
    if requested is None or (isinstance(requested, int) and not isinstance(requested, bool)):
        if isinstance(requested, int) and requested <= 0:
            raise PolicyError(f"requested chunk must be positive, got {requested}")
        return SubbatchDecision(value=requested, source="requested", tokens=tokens, stock_value=stock_value,
                                device_bytes=device_bytes, requested=requested)
    if requested == "stock":
        if stock_rule is None:
            raise PolicyError("requested='stock' needs stock_rule (the engine's own size rule, tokens -> chunk|None)")
        return SubbatchDecision(value=stock_rule(tokens), source="stock", tokens=tokens, stock_value=stock_value,
                                device_bytes=device_bytes, requested="stock")
    if requested != "auto":
        raise PolicyError(f"requested={requested!r} is not None, a positive int, 'auto' or 'stock'")
    if peak_estimator is None:
        raise PolicyError("requested='auto' needs peak_estimator (the adapter's measured fit)")
    est = int(peak_estimator(tokens))
    if device_bytes is None or int(device_bytes) <= 0:
        value = stock_value if when_device_unknown == "stock" else None
        return SubbatchDecision(value=value, source="auto:no_device", tokens=tokens, stock_value=stock_value,
                                estimated_bytes=est, device_bytes=None, requested="auto",
                                details={"when_device_unknown": when_device_unknown})
    device_bytes = int(device_bytes)
    if est > device_bytes:
        return SubbatchDecision(value=stock_value, source="auto:exceeds", tokens=tokens, stock_value=stock_value,
                                estimated_bytes=est, device_bytes=device_bytes, requested="auto")
    return SubbatchDecision(value=None, source="auto:fits", tokens=tokens, stock_value=stock_value,
                            estimated_bytes=est, device_bytes=device_bytes, requested="auto")


def threshold_rule(threshold_tokens: int, chunk: int) -> Callable[[int], Optional[int]]:
    """A stock size rule of the common form 'chunk above N tokens, unchunked at or below' as a callable for ``stock_rule=``."""
    def rule(tokens: int) -> Optional[int]:
        return chunk if int(tokens) > int(threshold_tokens) else None
    rule.__name__ = f"threshold_rule({threshold_tokens},{chunk})"
    return rule
