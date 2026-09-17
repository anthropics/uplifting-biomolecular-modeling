"""Fail-loud memory and shape gates: a call the kit cannot serve within the device's bytes, or at a shape it holds no test record for,
takes the stock path BY NAME (decided before the launch, counted, announced once per reason) instead of ending in an out-of-memory error.

Contract. ``ShapeTable(certified).admit(shape, device_name, blockers)`` -> ``("kit", None)`` or ``("stock", "<tag>|<detail>")``: a
blocker present (a call form the kit is not tested for), a device class absent from the table, or a shape absent from its class is
the stock route with a stable reason tag (a counter key). ``LinearNeed`` scales a measured base (the kit's resident store bytes at a base
shape, the stock's activation peak delta at that shape) to another shape: stores linear in L, activations linear in B*L.
``DeviceBudget(margin_bytes=, reserve_frac=).fits(key, need_bytes, stock_need_bytes, free_bytes=None)`` -> ``Decision``: ok iff
``free >= need + stock_need + margin`` (``free`` measured through torch — synchronised, cache emptied — when not given), cached per key,
every decision kept as a line; the margin is the kit's tested value (no default). ``line_fields(...)`` is the activation-evidence hook.
``ReasonCounter`` counts every routed call per tag (in an ``opt_core.mem.Ledger``) and prints the reason once. The free-byte figure is
``opt_core.mem.budget.device_free_bytes`` (cudaMemGetInfo free + the caching allocator's reusable slack); a capture / static-arena
admission is ``opt_core.mem.graph_gate``'s decision, not this module's. Standard library at import; torch is reached only inside
``free_device_bytes``, through ``opt_core.seq.numerics.require_torch`` (the seq package's one torch refusal, ``TorchUnavailable``); a torch
without a CUDA device is ``CudaUnavailable``.
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Set, Tuple

from ..mem import Ledger

__all__ = ["ShapeTable", "LinearNeed", "Decision", "DeviceBudget", "ReasonCounter", "free_device_bytes", "CudaUnavailable", "GIB", "line_fields"]

GIB = float(1 << 30)


class CudaUnavailable(RuntimeError):
    """free_device_bytes() needs a CUDA device (torch imported fine, ``torch.cuda.is_available()`` is False); the caller passes
    ``free_bytes=`` explicitly when it measures memory itself."""


# --------------------------------------------------------------------------------------------------------------------- shape table
class ShapeTable(object):
    """The executable form of "a test record is per (device class, call shape)".

    ``certified``: ``{class_fragment: {shape tuple, ...}}`` in the kit's order — a class fragment is matched as a substring of the device
    name (e.g. ``"H100"`` in ``"NVIDIA H100 80GB HBM3"``); when several fragments occur in the name the LONGEST wins, ties by table order
    (so ``"GH200"`` listed beside ``"H200"`` classifies ``"NVIDIA GH200 480GB"`` as GH200 whatever the order). The decision never depends
    on set or hash iteration. ``ShapeTable.compose(base, added)`` = a frozen base table plus later test records, base classes first in base
    order, then classes only in ``added`` in their order."""

    def __init__(self, certified: Mapping[str, Iterable[Sequence[int]]]):
        self.certified = OrderedDict((str(c), {tuple(int(x) for x in s) for s in shapes}) for c, shapes in certified.items())   # type: Dict[str, Set[Tuple[int, ...]]]

    @classmethod
    def compose(cls, base: Mapping[str, Iterable[Sequence[int]]], added: Mapping[str, Iterable[Sequence[int]]]) -> "ShapeTable":
        keys = list(base) + [k for k in added if k not in base]
        return cls(OrderedDict((k, set(map(tuple, base.get(k, ()))) | set(map(tuple, added.get(k, ())))) for k in keys))

    def device_class(self, device_name: Optional[str]) -> Optional[str]:
        name = device_name or ""
        hits = [(len(c), -i, c) for i, c in enumerate(self.certified) if c in name]
        return max(hits)[2] if hits else None

    def admit(self, shape: Sequence[int], device_name: Optional[str], blockers: Optional[Sequence[Tuple[str, object, str]]] = None) -> Tuple[str, Optional[str]]:
        """``("kit", None)`` or ``("stock", reason)``. ``blockers``: ordered ``(tag, value, detail)`` triples — the first whose value is
        not None routes to stock with ``"<tag>|<detail>"`` (call forms the kit's kernels are not tested for)."""
        for tag, value, detail in (blockers or ()):
            if value is not None:
                return "stock", f"{tag}|{detail}"
        cls_ = self.device_class(device_name)
        if cls_ is None:
            return "stock", f"device_class|{device_name!r} has no certified shapes"
        shp = tuple(int(x) for x in shape)
        if shp not in self.certified[cls_]:
            return "stock", f"shape|{_fmt_shape(shp)} not certified on {cls_} (certified: {sorted(self.certified[cls_])})"
        return "kit", None

    def line(self, device_name: Optional[str]) -> str:
        """The activation-evidence clause: ``shapes=<class>:<sorted shapes>|none`` for the running device."""
        cls_ = self.device_class(device_name)
        if cls_ is None:
            return f"shapes=none ({device_name!r} not in {sorted(self.certified)})"
        return f"shapes={cls_}:" + ";".join(_fmt_shape(s) for s in sorted(self.certified[cls_]))


def _fmt_shape(shape: Sequence[int]) -> str:
    if len(shape) == 2:
        return f"(B={shape[0]}, L={shape[1]})"
    return "(" + ", ".join(str(x) for x in shape) + ")"


# -------------------------------------------------------------------------------------------------------------------- byte budgets
class LinearNeed(object):
    """Scale measured bytes from a base shape: kit stores are sized at the kit's max batch and linear in L (``store_bytes * L/L0``);
    the stock's activation peak delta is linear in B*L (``stock_peak_delta_bytes * (B*L)/(B0*L0)``)."""

    def __init__(self, base_B: int, base_L: int, store_bytes: int, stock_peak_delta_bytes: int = 0):
        self.B0, self.L0 = int(base_B), int(base_L)
        self.store_bytes, self.stock_peak_delta_bytes = int(store_bytes), int(stock_peak_delta_bytes)

    def at(self, B: int, L: int) -> Tuple[int, int, str]:
        need = int(self.store_bytes * (float(L) / self.L0)) if self.L0 else int(self.store_bytes)
        stock = int(self.stock_peak_delta_bytes * (float(B) * L) / max(self.B0 * self.L0, 1))
        basis = (f"the resident stores at L={self.L0} ({round(self.store_bytes / GIB, 2)} GiB) scaled by L/{self.L0}; "
                 f"the stock's peak delta at (B={self.B0}, L={self.L0}) ({round(self.stock_peak_delta_bytes / GIB, 2)} GiB) scaled by B·L")
        return need, stock, basis


class Decision(NamedTuple):
    key: Tuple
    ok: bool
    need_gib: float
    stock_need_gib: float
    free_gib: float
    margin_gib: float
    reason: str
    need_bytes: int = 0
    stock_need_bytes: int = 0


def free_device_bytes(device=None) -> Tuple[int, int]:
    """``(free, total)``: ``free`` = ``opt_core.mem.budget.device_free_bytes(device)`` (cudaMemGetInfo free + the caching allocator's
    reusable slack, reserved - allocated), ``total`` = the device's capacity. ``opt_core.seq.numerics.TorchUnavailable`` when torch cannot
    be imported, ``CudaUnavailable`` when it has no CUDA device — the caller then measures itself and passes ``free_bytes=``."""
    from .numerics import require_torch                                # the seq package's single torch refusal (lazy: stdlib at import)
    from ..mem.budget import device_free_bytes                         # the core's one free-byte figure
    torch = require_torch()
    free = device_free_bytes(device)
    if free is None:
        raise CudaUnavailable("free_device_bytes needs a CUDA device (torch.cuda.is_available() is False)")
    _free_raw, total = torch.cuda.mem_get_info(device)
    return int(free), int(total)


class DeviceBudget(object):
    """``fits(key, need_bytes, stock_need_bytes=0, free_bytes=None)``: ok iff ``free >= need + stock_need + margin`` where ``margin =
    margin_bytes + reserve_frac * total`` — both required keyword arguments (the kit's tested values; 0 states "no margin" by choice).
    Decisions are cached per ``key`` (``decided``), kept in order (``decisions``); ``forget(key)``
    drops one (e.g. after the kit released other shapes' stores and wants a second decision); ``lines()`` are the sentences for the run
    record. ``free_bytes``/``total_bytes`` given -> no torch call (the caller measured); else ``free_device_bytes()``."""

    def __init__(self, *, margin_bytes: int, reserve_frac: float, total_bytes: Optional[int] = None, tag: str = "membudget"):
        self.margin_bytes, self.reserve_frac, self.total_bytes, self.tag = int(margin_bytes), float(reserve_frac), total_bytes, tag
        self.decided = {}        # type: Dict[Tuple, Decision]
        self.decisions = []      # type: List[Decision]

    def margin(self, total: Optional[int]) -> int:
        return self.margin_bytes + int(self.reserve_frac * (total or self.total_bytes or 0))

    def fits(self, key, need_bytes: int, stock_need_bytes: int = 0, free_bytes: Optional[int] = None, total_bytes: Optional[int] = None,
             basis: str = "", note: str = "") -> Decision:
        """The decision for ``key``. A key already decided with the SAME (need_bytes, stock_need_bytes) returns that decision (free memory
        is not re-measured: one decision per shape per process is the contract); a different need for a decided key is decided afresh,
        replaces the cached one and is kept as a new line marked ``re-decided`` — never the stale verdict."""
        key = tuple(key) if isinstance(key, (tuple, list)) else (key,)
        prev = self.decided.get(key)
        if prev is not None:
            if (prev.need_bytes, prev.stock_need_bytes) == (int(need_bytes), int(stock_need_bytes)):
                return prev
            note = (note + " " if note else "") + f"(re-decided: need {round(prev.need_bytes / GIB, 2)}+{round(prev.stock_need_bytes / GIB, 2)} GiB -> {round(int(need_bytes) / GIB, 2)}+{round(int(stock_need_bytes) / GIB, 2)} GiB)"
        if free_bytes is None:
            free_bytes, total = free_device_bytes()
        else:
            total = total_bytes if total_bytes is not None else self.total_bytes
        m = self.margin(total)
        ok = int(free_bytes) >= int(need_bytes) + int(stock_need_bytes) + m
        verdict = "budget ok" if ok else "budget FAILS"
        reason = (f"shape {_fmt_shape(key)}: {verdict}{(' ' + note) if note else ''} (kit stores ≈ {round(need_bytes / GIB, 2)} GiB + the stock ≈ {round(stock_need_bytes / GIB, 2)} GiB + margin {round(m / GIB, 2)} "
                  f"{'≤' if ok else '>'} free {round(free_bytes / GIB, 2)} GiB{'; ' + basis if basis else ''}){'' if ok else ' → the stock path for this shape; counted'}")
        d = Decision(key, ok, round(need_bytes / GIB, 2), round(stock_need_bytes / GIB, 2), round(free_bytes / GIB, 2), round(m / GIB, 2), reason, int(need_bytes), int(stock_need_bytes))
        self.decided[key] = d
        self.decisions.append(d)
        return d

    def forget(self, key) -> None:
        key = tuple(key) if isinstance(key, (tuple, list)) else (key,)
        self.decided.pop(key, None)

    def lines(self) -> List[str]:
        return [d.reason for d in self.decisions]


# ------------------------------------------------------------------------------------------------------------ counted announcements
class ReasonCounter(object):
    """Count every routed call under its reason tag (the text before ``|``) and print the full reason ONCE per tag:
    ``[<tag>] <verb> <reason>; counted`` on ``stream`` (stderr). Counts live in an ``opt_core.mem.Ledger`` (``ledger=`` shares the kit's
    own; keys ``<prefix><tag>``); ``counts`` is the census for the kit's exit line / manifest."""

    def __init__(self, tag: str, verb: str = "STOCK PATH", stream=None, ledger: Optional[Ledger] = None, prefix: str = "routed_"):
        self.tag, self.verb, self.stream, self.prefix = tag, verb, stream, prefix
        self.ledger = ledger if ledger is not None else Ledger()
        self.announced = set()   # type: Set[str]

    def record(self, reason: str) -> int:
        key = reason.split("|", 1)[0]
        n = self.ledger.count(self.prefix + key, 1)
        if key not in self.announced:
            self.announced.add(key)
            print(f"[{self.tag}] {self.verb} {reason}; counted", file=self.stream or sys.stderr, flush=True)
        return n

    @property
    def counts(self) -> Dict[str, int]:
        """``{reason tag: n}`` — the ledger's ``<prefix><tag>`` counters with the prefix removed."""
        return {k[len(self.prefix):]: int(v) for k, v in self.ledger.facts().items() if k.startswith(self.prefix)}

    def line(self) -> str:
        """The exit clause: ``routed=<tag>:<n>,...|none``."""
        c = self.counts
        if not c:
            return "routed=none"
        return "routed=" + ",".join(f"{k}:{v}" for k, v in sorted(c.items()))


# ------------------------------------------------------------------------------------------------------------- evidence hook
def line_fields(table: Optional[ShapeTable] = None, device_name: Optional[str] = None, budget: Optional[DeviceBudget] = None,
                counter: Optional[ReasonCounter] = None) -> Dict[str, str]:
    """The activation-evidence fields (the seq modules' hook: str -> str): ``shapes`` (the table's tested shapes for the running
    device, when a table is given), ``membudget`` (the budget's latest decision sentence or ``none decided``), ``routed`` (the counter's
    census ``<tag>:<n>,...`` or ``none``)."""
    out = {}             # type: Dict[str, str]
    if table is not None:
        out["shapes"] = table.line(device_name).split("=", 1)[1]
    if budget is not None:
        out["membudget"] = budget.decisions[-1].reason if budget.decisions else "none decided"
    if counter is not None:
        out["routed"] = counter.line().split("=", 1)[1]
    return out
