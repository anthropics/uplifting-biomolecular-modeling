"""Batch ceilings and their dispatch: the int32-index bound of a kernel, the split of a request into calls, the tested-shape regime
plan, and the one activation-evidence clause a kit prints for them.

Contract. A kit whose kernels index a batch in 32-bit arithmetic has a hard ceiling: ``max_units(elems_per_unit)`` = the largest
number of units whose last element index stays below ``2**index_bits``. A request above a ceiling is never one call and never a silent
fault: ``plan_chunks`` splits it (full pieces of ``max_b`` and a remainder — the ``torch.split`` order), ``RegimePlan.plan`` composes
it from the routes a kit qualifies (tested batch sizes, largest first, plus a named tail form for the remainder), and
``BoundRefused`` is the named refusal for a kit whose only safe action above its bound is to refuse before any launch. Every plan has
one ``line(plan)`` clause for the kit's activation line and a ``counts(plan)`` dict for its stamp. The row split itself is
``opt_core.mem.row_blocks``. Pure standard library; no framework import; every number the functions use is an argument (the kit owns
its values and reads them from its own pins).
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from ..mem import row_blocks

__all__ = ["BoundRefused", "Chunk", "Plan", "Ceiling", "ceiling", "max_units", "check_bound", "plan_chunks", "RegimePlan", "line_fields", "line", "counts", "TAILS", "BOUND_KINDS"]

TAILS = ("pad", "pieces", "stock")        # the remainder forms a RegimePlan admits
BOUND_KINDS = ("int32", "certified")      # which ceiling binds: the kernels' index arithmetic, or the kit's tested plateau passed as cap


class BoundRefused(ValueError):
    """A request past a bound the kit cannot serve in pieces — raised before any launch, with the bound in the message.
    ``bound`` is the largest admissible value (``requested > bound`` was refused)."""

    def __init__(self, what: str, requested: int, bound: int, unit: str = "units", advice: str = ""):
        self.requested, self.bound, self.unit = int(requested), int(bound), unit
        msg = f"{what}: {requested} {unit} exceeds the bound ({unit} <= {bound})" + (f"; {advice}" if advice else "")
        super().__init__(msg)


class Chunk(NamedTuple):
    """One call of a plan: ``kind`` = the route name ('chunk' for plan_chunks; a tested size's route name or 'stock' for a regime),
    ``start`` = the first unit's index in the request, ``n`` = real units in the call, ``padded_to`` = the call's batch shape when the
    real units are padded up to it (else None)."""
    kind: str
    start: int
    n: int
    padded_to: Optional[int] = None


class Plan(NamedTuple):
    requested: int
    chunks: Tuple[Chunk, ...]
    reason: str

    @property
    def n_calls(self) -> int:
        return len(self.chunks)

    @property
    def padded_units(self) -> int:
        return sum((c.padded_to or c.n) - c.n for c in self.chunks)

    def sizes(self) -> List[int]:
        """The per-call batch shapes in order (a padded call counts its padded shape)."""
        return [c.padded_to or c.n for c in self.chunks]


class Ceiling(NamedTuple):
    """A batch ceiling and what binds it: ``kind`` = ``'int32'`` (the kernels' index arithmetic) or ``'certified'`` (the kit's tested
    plateau, passed as ``cap``); ``basis`` = the arithmetic sentence for the evidence line."""
    n: int
    kind: str
    basis: str


def ceiling(elems_per_unit: int, index_bits: int = 31, cap: Optional[int] = None) -> Ceiling:
    """The largest N with ``N * elems_per_unit <= 2**index_bits``, i.e. ``2**index_bits // elems_per_unit`` (at least 1); when the kit's
    tested ceiling ``cap`` is lower, ``cap`` binds and ``kind`` is ``'certified'``. ``elems_per_unit`` = the element count ONE unit
    contributes to the largest int32-indexed tensor of any launch (e.g. ``(L // 2) * 512`` for a store of L/2 positions x 512 channels)."""
    if int(elems_per_unit) <= 0:
        raise ValueError(f"elems_per_unit must be positive, got {elems_per_unit}")
    n32 = max(1, (1 << int(index_bits)) // int(elems_per_unit))
    basis32 = f"N x {int(elems_per_unit)} <= 2^{int(index_bits)}"
    if cap is not None and int(cap) < n32:
        return Ceiling(int(cap), "certified", f"certified ceiling {int(cap)} < the int32 bound {n32} ({basis32})")
    return Ceiling(n32, "int32", basis32 + (f"; certified ceiling {int(cap)} not binding" if cap is not None else ""))


def max_units(elems_per_unit: int, index_bits: int = 31, cap: Optional[int] = None) -> int:
    """``ceiling(...).n`` — the number alone."""
    return ceiling(elems_per_unit, index_bits, cap).n


def check_bound(what: str, requested: int, at_most: Optional[int] = None, below: Optional[int] = None, unit: str = "units", advice: str = "") -> None:
    """The refuse-before-launch form: raise ``BoundRefused`` when ``requested > at_most`` (``at_most`` = the largest admissible count,
    the ``max_units`` form) or when ``requested >= below`` (``below`` = the smallest count that overflows, the ``ceil(2**31 / width)``
    form). Exactly one of the two is given."""
    if (at_most is None) == (below is None):
        raise ValueError("check_bound: give exactly one of at_most= / below=")
    limit = int(at_most) if at_most is not None else int(below) - 1
    if int(requested) > limit:
        raise BoundRefused(what, requested, limit, unit, advice)


def plan_chunks(B: int, max_b: int, kind: str = "chunk") -> Plan:
    """Split ``B`` units into consecutive calls of at most ``max_b`` (full pieces first, the remainder last — the order
    ``torch.split(x, max_b)`` yields). ``B <= max_b`` is one call and the reason says so."""
    B, max_b = int(B), int(max_b)
    if B < 0 or max_b <= 0:
        raise ValueError(f"plan_chunks(B={B}, max_b={max_b}): B >= 0 and max_b >= 1 required")
    chunks = [Chunk(kind, r0, r1 - r0, None) for r0, r1 in row_blocks(B, max_b)]      # the core's one row-block arithmetic
    if B <= max_b:
        reason = f"B={B} <= bound {max_b}: one call"
    else:
        q, r = divmod(B, max_b)
        reason = f"B={B} > bound {max_b}: {q} call(s) of {max_b}" + (f" + 1 of {r}" if r else "") + " (chunked dispatch, same route per piece)"
    return Plan(B, tuple(chunks), reason)


class RegimePlan(object):
    """Compose a request of ``n`` units from a kit's tested batch sizes.

    ``certified``: the batch sizes whose route the kit qualifies — a mapping size -> route name (e.g. ``{8: "uf", 1: "b1"}``) or a plain
    sequence of sizes (each route then named ``b<size>``). ``max_batch``: an optional ceiling on any single call (an index bound);
    tested sizes above it are dropped by name (``.dropped``). ``tail``: what serves a remainder r that no whole tested call covers —
    ``'pad'``: one call of the smallest tested size >= r, padded (``padded_to`` set; the kit drops the pad rows); ``'pieces'``: r
    size-1 calls when 1 is tested (with ``piece_cost``: only when ``r * cost[1] <= cost[s]`` for the smallest covering size s, else one
    padded call of s — the cheaper composition), one padded call of s when 1 is not tested; ``'stock'``: one call of kind ``'stock'``
    for the remainder (the kit's stock route at that batch). ``pad_allowed=False`` forbids every padded call: a remainder that needs one
    raises ``BoundRefused``. ``piece_cost``: per-size cost in any unit (e.g. ms per call), read only by the ``'pieces'`` tail rule.
    """

    def __init__(self, certified, max_batch: Optional[int] = None, tail: str = "pieces", piece_cost: Optional[Dict[int, float]] = None,
                 pad_allowed: bool = True):
        if tail not in TAILS:
            raise ValueError(f"tail={tail!r} not in {TAILS}")
        if isinstance(certified, dict):
            routes = {int(k): str(v) for k, v in certified.items()}
        else:
            routes = {int(k): f"b{int(k)}" for k in certified}
        if not routes or any(k <= 0 for k in routes):
            raise ValueError(f"certified sizes must be positive and non-empty, got {sorted(routes)}")
        self.dropped = sorted(k for k in routes if max_batch and k > int(max_batch))
        self.routes = {k: v for k, v in routes.items() if k not in self.dropped}
        if not self.routes:
            raise ValueError(f"every certified size {sorted(routes)} is above max_batch={max_batch}")
        self.sizes = sorted(self.routes, reverse=True)          # largest first
        self.max_batch = int(max_batch) if max_batch else None
        self.tail = tail
        self.cost = {int(k): float(v) for k, v in (piece_cost or {}).items()}
        self.pad_allowed = bool(pad_allowed)

    def _covering(self, r: int) -> int:
        """The padded call's shape for a remainder r: the smallest tested size >= r among sizes > 1 (a padded call of size 1 is
        never a form); the largest tested size when none covers r."""
        fits = [s for s in self.sizes if s >= r and s > 1] or [s for s in self.sizes if s >= r]
        return min(fits) if fits else self.sizes[0]

    def plan(self, n: int) -> Plan:
        n = int(n)
        if n < 0:
            raise ValueError(f"plan(n={n}): n >= 0 required")
        chunks = []          # type: List[Chunk]
        s = 0
        rem = n
        for size in self.sizes:                                 # greedy: the largest tested size while it fits whole
            if size == 1 and self.tail == "pieces" and self.cost and len(self.sizes) > 1:
                break                                           # with costs known and a larger size tested, size-1 pieces are the tail decision below
            while rem >= size:
                chunks.append(Chunk(self.routes[size], s, size, None))
                s += size
                rem -= size
        notes = []           # type: List[str]
        if rem:
            cov = self._covering(rem)
            if self.tail == "stock":
                chunks.append(Chunk("stock", s, rem, None))
                notes.append(f"remainder {rem} on the stock route")
            elif self.tail == "pad":
                if not self.pad_allowed:
                    raise BoundRefused("RegimePlan(tail='pad')", rem, 0, "remainder units", "pad_allowed=False and no certified size divides the request")
                chunks.append(Chunk(self.routes[cov], s, rem, cov))
                notes.append(f"remainder {rem} as one {self.routes[cov]} call padded to {cov} ({cov - rem} pad rows dropped)")
            else:                                               # 'pieces'
                one = 1 in self.routes
                costed = bool(self.cost) and 1 in self.cost and cov in self.cost
                by_pieces = one and (not costed or not self.pad_allowed or rem * self.cost[1] <= self.cost[cov])
                if by_pieces:
                    for i in range(rem):
                        chunks.append(Chunk(self.routes[1], s + i, 1, None))
                    if costed:
                        notes.append(f"remainder {rem} as {rem} x {self.routes[1]} ({rem} x {self.cost[1]:g} = {rem * self.cost[1]:g} <= {self.cost[cov]:g} for one padded {self.routes[cov]})")
                    else:
                        notes.append(f"remainder {rem} as {rem} x {self.routes[1]}")
                elif self.pad_allowed:
                    chunks.append(Chunk(self.routes[cov], s, rem, cov))
                    why = f"{rem} x {self.cost[1]:g} = {rem * self.cost[1]:g} > {self.cost[cov]:g}" if (one and costed) else "no size-1 route certified"
                    notes.append(f"remainder {rem} as one {self.routes[cov]} call padded to {cov} ({why}; {cov - rem} pad rows dropped)")
                else:
                    raise BoundRefused("RegimePlan(tail='pieces')", rem, 0, "remainder units", "no size-1 route and pad_allowed=False")
        body = ", ".join(f"{v} x {k}" for k, v in _tally(chunks).items()) or "no calls"
        reason = f"B={n}: {body}" + (f"; {'; '.join(notes)}" if notes else "") + (f"; certified sizes above max_batch {self.max_batch} dropped: {self.dropped}" if self.dropped else "")
        return Plan(n, tuple(chunks), reason)


def _tally(chunks: Sequence[Chunk]) -> Dict[str, int]:
    out = {}             # type: Dict[str, int]
    for c in chunks:
        key = c.kind + (f"(pad {c.padded_to})" if c.padded_to else "")
        out[key] = out.get(key, 0) + 1
    return out


def counts(plan: Plan) -> Dict[str, int]:
    """The stamp fields of a plan: requested units, calls, padded rows, calls per kind (``calls_<kind>``)."""
    out = {"requested": plan.requested, "calls": plan.n_calls, "padded_rows": plan.padded_units}          # type: Dict[str, int]
    for k, v in _tally(plan.chunks).items():
        key = "calls_" + k.split("(")[0]
        out[key] = out.get(key, 0) + v
    return out


def line_fields(plan: Optional[Plan] = None, bound: Optional[Ceiling] = None) -> Dict[str, str]:
    """The activation-evidence fields (the seq modules' hook: str -> str): ``batch_plan`` = the plan's reason (when a plan is given);
    with a ``Ceiling``: ``bound`` = its number, ``bound_kind`` = ``certified|int32``, ``bound_basis`` = its arithmetic. One clause per
    process per distinct request shape is the convention (the kit decides when to print and joins the fields into its own line)."""
    out = {}             # type: Dict[str, str]
    if plan is not None:
        out["batch_plan"] = plan.reason
    if bound is not None:
        if bound.kind not in BOUND_KINDS:
            raise ValueError(f"bound.kind={bound.kind!r} not in {BOUND_KINDS}")
        out["bound"] = str(int(bound.n))
        out["bound_kind"] = bound.kind
        out["bound_basis"] = bound.basis
    return out


def line(plan: Optional[Plan] = None, bound: Optional[Ceiling] = None) -> str:
    """``batch plan: <reason>[ | bound=<N> <kind> (<basis>)]`` — ``line_fields`` joined as one clause."""
    f = line_fields(plan, bound)
    head = f"batch plan: {f['batch_plan']}" if "batch_plan" in f else "batch plan: none"
    return head + (f" | bound={f['bound']} {f['bound_kind']} ({f['bound_basis']})" if "bound" in f else "")
