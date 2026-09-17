"""The POLICY half of graph capture (when a shape is worth capturing) and the process-level POOL (several capture sites under one memory
budget, one item/job boundary and one evidence census).

Contract — policy. A :class:`CapturePolicy` is pure Python (standard library, Python 3.8): ``decide(key, sightings)`` answers
``("capture" | "eager", reason, final)`` for a key :class:`opt_core.capture.graphs.GraphCache` has not captured (``final`` True files the
key as eager for the rest of the generation without asking again), and ``observe(key, capture_s=, eager_s=, replay_s=)`` receives the
measured cost of every capture so a break-even rule can price the next key. The policies: :class:`FixedK` (capture on the
K-th sighting; K=1 captures at first sight, K=2 is the sighting rule — a shape seen once never pays a capture), :class:`BreakEven`
(K = ceil(capture_s / saving_s_per_call) from pinned numbers, per key or global, or ``auto``: learned from the site's first capture),
:class:`Table` (K per key from a pinned table; keys without a rule run eager BY NAME), :class:`Planned` (the job declares how many calls of
each key it will make; a key is captured at first sight iff its planned count reaches its break-even K, else it runs eager by name —
the job-histogram form). The policy KEY is the adapter's (``policy_key(args, kwargs)``: a batch class, a token bucket), never tensor
values; by default it is the argument signature. Sequence/genomics kits import these classes as they are; nothing here knows an engine.

Contract — pool. :class:`GraphPool` owns one :class:`opt_core.capture.graphs.GraphByteBudget` (ONE memory budget across all its sites), the
process's item and job boundaries (:meth:`GraphPool.new_item` fans out to every site; :meth:`GraphPool.generation` keeps at most N VALUE-keyed
generations of a site family resident, dropping the oldest whole — its counters and any partial reason are kept in the family's tombstone, and
the family prints ONE aggregated ``LEVER`` line under its stable name; :meth:`GraphPool.new_job` resets every site by name —
the job-scoped reset of a persistent server), an optional shape-bucketing hook per site (``bucket_fn(args, kwargs)`` returns ``None`` or
``(padded_args, padded_kwargs, unpad)`` — padding numerics are the adapter's and travel with the engine's class), and the census:
:meth:`GraphPool.evidence_lines` is one ``LEVER`` line per site (each lever prints ONE line per arm per process), :meth:`GraphPool.partial`
the first site whose activation is partial. Sites are created with the pool's defaults (``strict``, ``rng``, ``verify``, ``warmup``,
``policy``…) overridden per site.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from . import graphs as _graphs

Decision = Tuple[str, str, bool]          # (action: "capture" | "eager", reason, final)


class CapturePolicy:
    """Base policy: capture at first sight. Subclasses override :meth:`k` (the sighting on which a key is captured; None = never) and may
    use :meth:`observe`."""

    name = "policy"

    def k(self, key) -> Optional[int]:
        return 1

    def decide(self, key, sightings: int) -> Decision:
        k = self.k(key)
        if k is None:
            return "eager", f"{self.name}: no capture rule for key {self._short(key)} (off: not measured on this key)", True
        if sightings >= k:
            return "capture", f"{self.name}: sighting {sightings} >= K={k}", False
        return "eager", f"{self.name}: sighting {sightings} of K={k}", False

    def observe(self, key, *, capture_s: float, eager_s: Optional[float], replay_s: Optional[float]) -> None:  # noqa: D401
        """Measured cost of a capture of ``key`` (seconds): the capture itself, one eager call, one replay."""

    def describe(self) -> str:
        return self.name

    @staticmethod
    def _short(key) -> str:
        s = repr(key)
        return s if len(s) <= 80 else s[:77] + "..."


class FixedK(CapturePolicy):
    """Capture on the K-th sighting of a key (K=1: first call; K=2: the sighting rule)."""

    name = "fixed_k"

    def __init__(self, k: int = 2):
        self._k = max(1, int(k))

    def k(self, key) -> Optional[int]:
        return self._k

    def describe(self) -> str:
        return f"fixed_k(K={self._k})"


def break_even_k(capture_s: float, saving_s_per_call: float, minimum: int = 1, maximum: Optional[int] = None) -> Optional[int]:
    """K = ceil(capture_s / saving_s_per_call): the number of calls of one key after which a capture has paid for itself. A non-positive
    saving (replay no faster than eager) gives None (never capture) unless ``maximum`` caps it."""
    if saving_s_per_call is None or saving_s_per_call <= 0:
        return maximum
    k = max(int(minimum), int(math.ceil(float(capture_s) / float(saving_s_per_call))))
    return min(k, maximum) if maximum is not None else k


class BreakEven(CapturePolicy):
    """K from the break-even arithmetic. ``capture_s`` / ``saving_s`` are pinned numbers (a kit's PINS record) — global, or per key through
    ``per_key={key: (capture_s, saving_s)}``; ``auto=True`` learns them from the FIRST capture of the site (that key is captured on sighting
    ``probe_k``) and prices every later key with them. ``maximum`` caps K (None: a key whose replay saves nothing is never captured)."""

    name = "break_even"

    def __init__(self, capture_s: Optional[float] = None, saving_s: Optional[float] = None, *, per_key: Optional[Mapping] = None,
                 auto: bool = False, probe_k: int = 2, minimum: int = 1, maximum: Optional[int] = None):
        self.capture_s = capture_s
        self.saving_s = saving_s
        self.per_key = dict(per_key or {})
        self.auto = bool(auto)
        self.probe_k = max(1, int(probe_k))
        self.minimum = max(1, int(minimum))
        self.maximum = maximum
        self.observed: Dict[Any, dict] = {}

    def k(self, key) -> Optional[int]:
        if key in self.per_key:
            c, s = self.per_key[key]
            return break_even_k(c, s, self.minimum, self.maximum)
        if self.capture_s is not None and self.saving_s is not None:
            return break_even_k(self.capture_s, self.saving_s, self.minimum, self.maximum)
        if self.auto:
            return self.probe_k                       # nothing measured yet: probe
        return None

    def observe(self, key, *, capture_s, eager_s, replay_s):
        saving = (eager_s - replay_s) if (eager_s is not None and replay_s is not None) else None
        self.observed[key] = {"capture_s": capture_s, "eager_s": eager_s, "replay_s": replay_s, "saving_s": saving,
                              "k": break_even_k(capture_s, saving, self.minimum, self.maximum) if saving is not None else None}
        if self.auto:
            self.per_key[key] = (capture_s, saving if saving is not None else 0.0)
            if self.capture_s is None and saving is not None:
                self.capture_s, self.saving_s = capture_s, saving

    def describe(self) -> str:
        if self.capture_s is not None and self.saving_s is not None:
            return f"break_even(K={break_even_k(self.capture_s, self.saving_s, self.minimum, self.maximum)} from capture_s={self.capture_s:.4f} saving_s={self.saving_s:.6f})"
        return f"break_even({'auto, unmeasured' if self.auto else 'per-key table of %d' % len(self.per_key)})"


class Table(CapturePolicy):
    """K per key from a pinned table; a key without a row is captured on ``default_k`` or, when that is None, runs eager BY NAME — the
    decision reads ``table: no capture rule for key <key> (off: not measured on this key)``: a key the pinned measurements do not cover
    keeps the composition the measurements certify (no capture), and the line says why."""

    name = "table"

    def __init__(self, k_by_key: Mapping, default_k: Optional[int] = None):
        self.k_by_key = dict(k_by_key)
        self.default_k = default_k

    def k(self, key) -> Optional[int]:
        v = self.k_by_key.get(key, self.default_k)
        return None if v is None else max(1, int(v))

    def describe(self) -> str:
        return f"table(rows={len(self.k_by_key)} default_k={self.default_k})"


class Planned(CapturePolicy):
    """The job-histogram form: the job declares ``counts_by_key`` (how many calls of each key it will make); a key is captured AT FIRST
    SIGHT iff its planned count reaches the break-even K of ``pricing`` (any policy's :meth:`k`; a :class:`FixedK` gives a flat threshold),
    else it runs eager by name for the whole job. Keys the plan does not mention follow ``unplanned`` (``"eager"`` or ``"pricing"``)."""

    name = "planned"

    def __init__(self, counts_by_key: Mapping, pricing: Optional[CapturePolicy] = None, unplanned: str = "eager"):
        if unplanned not in ("eager", "pricing"):
            raise ValueError("unplanned must be 'eager' or 'pricing'")
        self.counts = dict(counts_by_key)
        self.pricing = pricing if pricing is not None else FixedK(2)
        self.unplanned = unplanned

    def k(self, key) -> Optional[int]:
        return self.pricing.k(key)

    def decide(self, key, sightings: int) -> Decision:
        if key not in self.counts:
            if self.unplanned == "eager":
                return "eager", f"planned: key {self._short(key)} not in the job's plan (off: not measured on this key)", True
            return self.pricing.decide(key, sightings)
        k = self.pricing.k(key)
        n = int(self.counts[key])
        if k is None or n < k:
            return "eager", f"planned: n={n} < break-even K={k} for key {self._short(key)}", True
        return "capture", f"planned: n={n} >= K={k} at sighting {sightings}", False

    def observe(self, key, **kw):
        self.pricing.observe(key, **kw)

    def describe(self) -> str:
        return f"planned(keys={len(self.counts)} distinct, items={sum(int(v) for v in self.counts.values())}, pricing={self.pricing.describe()})"


# ----------------------------------------------------------------------------------------------------------------- the pool
class Site:
    """One capture site of a :class:`GraphPool`: a :class:`opt_core.capture.graphs.GraphCache` plus the site's optional bucketing hook."""

    def __init__(self, cache: _graphs.GraphCache, bucket_fn: Optional[Callable] = None):
        self.cache = cache
        self.bucket_fn = bucket_fn
        self.bucketed_calls = 0

    @property
    def name(self) -> str:
        return self.cache.name

    def run(self, fn: Callable, *args, **kwargs):
        """``cache.run`` — through the bucketing hook when the site has one and it answers for these arguments."""
        if self.bucket_fn is not None:
            b = self.bucket_fn(args, kwargs)
            if b is not None:
                padded_args, padded_kwargs, unpad = b
                self.bucketed_calls += 1
                return unpad(self.cache.run(fn, *padded_args, **padded_kwargs))
        return self.cache.run(fn, *args, **kwargs)

    __call__ = run

    def wrap(self, fn: Callable) -> Callable:
        """``fn`` routed through this site (for patching a bound method in the adapter)."""
        def _wrapped(*args, **kwargs):
            return self.run(fn, *args, **kwargs)
        _wrapped.__wrapped__ = fn                # type: ignore[attr-defined]
        _wrapped.graph_site = self               # type: ignore[attr-defined]
        return _wrapped


class GraphPool:
    """Several capture sites under ONE memory budget, one item/job boundary and one census (module contract). ``defaults`` are
    :class:`opt_core.capture.graphs.GraphCache` keyword arguments applied to every site (``strict``, ``rng``, ``verify``, ``warmup``,
    ``max_entries``, ``on_full``, ``pool``, ``policy``, ``gate``, ``mode_fn``, ``log``, ``verbose`` …); :meth:`site` overrides them per site."""

    def __init__(self, tag: str, *, mem_budget_bytes=None, **defaults):
        self.tag = str(tag)
        self.ledger = _graphs.GraphByteBudget(mem_budget_bytes)
        self.defaults = dict(defaults)
        self.sites: Dict[str, Site] = {}
        self._site_overrides: Dict[str, dict] = {}
        self._generations: Dict[str, "OrderedDict[Any, Site]"] = {}
        self._gen_serial: Dict[str, list] = {}
        self._tombs: Dict[str, dict] = {}                              # family -> summed counters of evicted generations + their partial reasons
        self.generation_evictions = 0
        self.jobs = 0

    def site(self, name: str, *, bucket_fn: Optional[Callable] = None, **overrides) -> Site:
        """The site called ``name`` (created on first request with the pool defaults + ``overrides``; the same object afterwards)."""
        if name in self.sites:
            prev = self._site_overrides.get(name)
            if overrides and prev is not None and prev != overrides:
                raise ValueError(f"site {name!r} exists with different settings {prev} (asked {overrides}) — one site, one configuration")
            return self.sites[name]
        kw = dict(self.defaults)
        kw.update(overrides)
        kw.pop("mem_budget_bytes", None)
        cache = _graphs.GraphCache(name, ledger=self.ledger, **kw)
        s = Site(cache, bucket_fn=bucket_fn)
        self.sites[name] = s
        self._site_overrides[name] = dict(overrides)
        return s

    def generation(self, name: str, gen_key, *, max_generations: int = 2, **overrides) -> Site:
        """The site of generation ``gen_key`` within the family ``name``: a ``pool="generation"`` cache of its own (its graphs share one pool,
        replay in capture order, and are only ever dropped together). At most ``max_generations`` generations of a family are resident; asking
        for a new one drops the OLDEST whole (synchronize, graphs + buffers, gc, empty_cache) — never a single graph. VALUE-keyed families
        (trajectory index set, motif mask, cyclic flag) pass those values as ``gen_key``."""
        fam = self._generations.setdefault(name, OrderedDict())
        if gen_key in fam:
            fam.move_to_end(gen_key)
            return fam[gen_key]
        while len(fam) >= max(1, int(max_generations)):
            old_key, old_site = fam.popitem(last=False)
            self._entomb(name, old_site)                            # its counters and any PARTIAL reason survive in the family's tombstone
            old_site.cache.reset("generation_evicted")
            self.sites.pop(old_site.name, None)
            self.generation_evictions += 1
        kw = dict(self.defaults)
        kw.update(overrides)
        kw.pop("mem_budget_bytes", None)
        kw.setdefault("pool", "generation")
        kw.setdefault("on_full", "reset")
        site_name = f"{name}[{len(self._gen_serial.setdefault(name, [])) }]"
        self._gen_serial[name].append(gen_key)
        cache = _graphs.GraphCache(site_name, ledger=self.ledger, **kw)
        s = Site(cache)
        fam[gen_key] = s
        self.sites[site_name] = s
        return s

    _SUMMED = ("calls", "captures", "replays", "failures", "evictions", "items", "verify_pass", "verified_replays") + tuple(
        "eager_" + k for k in _graphs.EAGER_KINDS) + tuple("refused_" + k for k in _graphs.REFUSED_KINDS)

    def _entomb(self, family: str, site: "Site") -> None:
        tomb = self._tombs.setdefault(family, {"generations": 0, "partials": [], "capture_s": 0.0})
        st = site.cache.stats_
        for k in self._SUMMED:
            tomb[k] = tomb.get(k, 0) + int(st.get(k, 0))
        tomb["capture_s"] += float(st.get("capture_s", 0.0))
        tomb["generations"] += 1
        p = site.cache.partial()
        if p:
            tomb["partials"].append(f"{site.name}: {p}")

    def family_fields(self, family: str) -> List[tuple]:
        """The aggregated evidence of one generation family: resident generations + the tombstone of evicted ones (total accounting)."""
        fam = self._generations.get(family, OrderedDict())
        tomb = self._tombs.get(family, {})
        tot: Dict[str, Any] = {k: int(tomb.get(k, 0)) for k in self._SUMMED}
        capture_s = float(tomb.get("capture_s", 0.0))
        states, disabled = [], []
        for site in fam.values():
            st = site.cache.stats_
            for k in self._SUMMED:
                tot[k] += int(st.get(k, 0))
            capture_s += float(st.get("capture_s", 0.0))
            states.append(site.cache.state())
            if site.cache._disabled is not None:
                disabled.append(f"{site.name}: {site.cache._disabled}")
        partials = list(tomb.get("partials", [])) + [f"{s_.name}: {s_.cache.partial()}" for s_ in fam.values() if s_.cache.partial()]
        graphs_state = "active" if "active" in states else ("captured" if "captured" in states else ("eager" if "eager" in states else ("disabled" if disabled else "idle")))
        head = [("name", family), ("state", "skipped" if (disabled and len(disabled) == len(fam)) else "on")]
        if partials:
            head.append(("reason", partials[0]))
        return head + [("impl", "cuda_graphs"), ("origin", "core"), ("graphs", graphs_state), ("generations_resident", len(fam)),
                       ("generations_evicted", int(tomb.get("generations", 0))), ("captures", tot["captures"]), ("replays", tot["replays"]),
                       ("eager", {k[6:]: tot[k] for k in tot if k.startswith("eager_") and tot[k]}), ("refused", {k[8:]: tot[k] for k in tot if k.startswith("refused_") and tot[k]}),
                       ("failures", tot["failures"]), ("items", tot["items"]), ("partials", len(partials)), ("capture_s", round(capture_s, 3))]

    def family_line(self, family: str, tag: Optional[str] = None, name: Optional[str] = None) -> str:
        """ONE ``[<tag>] LEVER name=<family> …`` line for a generation family (stable name; resident + evicted generations summed)."""
        fields = self.family_fields(family)
        if name:
            fields[0] = ("name", name)
        return f"{_graphs.report.prefix(tag or self.tag)} LEVER {_graphs.report.kv(*fields)}"

    def new_item(self) -> None:
        for s in self.sites.values():
            s.cache.new_item()

    def new_job(self) -> None:
        """A persistent server's job boundary: every site drops its generation by name (graphs never outlive the job that captured them)."""
        self.jobs += 1
        for s in self.sites.values():
            if s.cache.stats_["captures"] or s.cache._entries or s.cache._eager_keys or s.cache._sightings:
                s.cache.reset("job")                                # reset() also forgets eager-filed keys and sightings: a new job starts clean

    def reset(self, reason: str) -> None:
        for s in self.sites.values():
            s.cache.reset(reason)

    def disable(self, reason: str) -> None:
        for s in self.sites.values():
            s.cache.disable(reason)

    def stats(self) -> dict:
        return {"tag": self.tag, "jobs": self.jobs, "generation_evictions": self.generation_evictions, "families": {f: dict(self.family_fields(f)) for f in sorted(set(self._generations) | set(self._tombs))},
                "ledger_bytes": self.ledger.bytes_total,
                "budget_bytes": self.ledger.budget() if self.ledger.measuring else None,
                "sites": {n: dict(s.cache.stats(), bucketed_calls=s.bucketed_calls,
                                   policy=(s.cache.policy.describe() if getattr(s.cache, "policy", None) is not None else None)) for n, s in self.sites.items()}}

    def evidence_lines(self, tag: Optional[str] = None, names: Optional[Mapping[str, str]] = None) -> List[str]:
        """One ``[<tag>] LEVER name=<site> state=… impl=cuda_graphs …`` line per site (``names={site: lever_name}`` renames)."""
        gen_site_names = {s_.name for fam in self._generations.values() for s_ in fam.values()}
        lines = [s.cache.evidence_line(tag or self.tag, (names or {}).get(n)) + (f" bucketed={s.bucketed_calls}" if s.bucket_fn is not None else "")
                 for n, s in self.sites.items() if n not in gen_site_names]
        lines += [self.family_line(f, tag, (names or {}).get(f)) for f in sorted(set(self._generations) | set(self._tombs))]
        return lines

    def partial(self) -> Optional[str]:
        """The first PARTIAL reason across plain sites, resident generations and the tombstones of evicted generations (nothing is lost to eviction)."""
        for n, s in self.sites.items():
            p = s.cache.partial()
            if p:
                return f"{n}: {p}"
        for fam, tomb in self._tombs.items():
            if tomb.get("partials"):
                return f"{fam} (evicted generation) {tomb['partials'][0]}"
        return None
