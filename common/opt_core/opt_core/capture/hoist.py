"""Step-invariant hoisting: compute once per item what a step loop would recompute every step (the 'const memo').

Contract. A diffusion sampler (or a recycle loop) calls the same sub-computations every step on inputs that do not change across steps
— pair-bias projections of the trunk pair representation, attention masks of a fixed layout, conditioning terms of a fixed noise
schedule. :class:`ConstMemo` returns the value computed on the FIRST call and the same tensor object on every later call whose keyed
inputs are the SAME objects, unmutated: an entry is keyed by the equality of its source tensors (held by weak reference — a freed source
evicts its entries, so an address re-used by a new tensor is never a hit) plus their in-place version counters (a source mutated in place
since the entry was made is a miss, by name) plus the adapter's ``extra`` (flags, the id of the module whose weights the computation reads).
:meth:`ConstMemo.value` is the explicit-key form (``value(name, key, compute)``: recompute when the adapter's invalidation key
changes) and :meth:`ConstMemo.rollout` the roll-out scope (entering with a new key drops everything: 'compute once per roll-out, reuse per
step'). Exact by construction when ``compute`` is a pure function of the keyed sources and of ``extra``: the memo hands back the very tensor the
first call produced. What the equality form cannot see — writes that do not bump the version counter (a graph replay into a static buffer, ``.data``/``out=``
writes), so sources must never be a graph's static buffers, and inference-mode tensors, which have no counter at all and are refused by
name (``refused_inference``: computed inline) — it holds by measurement instead: ``verify_every=k`` (recommended > 0 on tested lines) recomputes every k-th hit and demands
bit-exact equality (:class:`MemoStale` under ``strict``; the memo disables itself by name otherwise — ``stale`` in the evidence line).
Inside a CUDA-graph capture or its warm-up (:func:`opt_core.capture.graphs` sets the flag; ``torch.cuda.is_current_stream_capturing()``
is checked too) the memo is BYPASSED — the value is computed inline and nothing is stored — so a captured graph never reads memo memory
and the memo never holds graph-pool memory. :meth:`ConstMemo.new_item` drops everything (values never outlive the item that made them);
``max_entries`` bounds the table (LRU). Evidence: :meth:`ConstMemo.evidence_fields` / :meth:`ConstMemo.evidence_line` — ``LEVER name=<name> state=on impl=const_memo
memo=<active|computed|idle> hits= misses= bypassed= stale= entries= mib=``; "active" reads ``hits>0``.

:class:`RowDedup` is the row-axis companion (strategy ``F7.row_dedup``): a statement a sampler evaluates on ``n`` rows that are COPIES of one row
(a conditioning path on a trunk tensor repeated ``B`` times, a noise-level embedding shared by every sample of a step) is evaluated on ONE row and
expanded back — after an exact on-device uniformity read of the inputs; rows that are not uniform run the caller's statement on every row and
are COUNTED by name (``fallback_by=nonuniform:n``), never served silently. It changes the row count (the GEMM ``M``) of the statement, nothing per
element: bit-exact to the ``n``-row statement exactly when the kernels involved are row-invariant at that shape — a per-engine fact the kit's
equality leg proves (``verify_every`` recomputes the full statement and demands equality), so a kit's mode table names the class (``fast`` unless proven).

Standard library at import; torch is touched only through the tensors the caller hands in (Python 3.8 floor).
"""
from __future__ import annotations

import sys
import threading
import weakref
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Sequence, Tuple

from .. import report

_CAPTURING = threading.local()


def set_capturing(flag: bool) -> None:
    """Mark the current thread as inside a graph warm-up/capture (memo bypass). :class:`opt_core.capture.graphs.GraphCache` calls it."""
    _CAPTURING.depth = max(0, getattr(_CAPTURING, "depth", 0) + (1 if flag else -1))


def capturing() -> bool:
    """True inside a graph warm-up/capture on this thread, or when torch reports the current stream capturing."""
    if getattr(_CAPTURING, "depth", 0) > 0:
        return True
    torch = sys.modules.get("torch")
    try:
        return bool(torch is not None and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return False


class MemoStale(RuntimeError):
    """A checked hit recomputed to a different value: ``compute`` reads state the key does not cover (the adapter must key it or not hoist it)."""


def _is_tensor(x) -> bool:
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(x, torch.Tensor)


class ConstMemo:
    """The const memo of ONE hoisted computation site (module contract). ``get(sources, compute, extra=None)``: ``sources`` is a tensor or a
    sequence of tensors the value depends on; ``compute()`` produces the value (any object; tensors or trees of tensors typically); ``extra``
    is hashable adapter state folded into the key."""

    def __init__(self, name: str, *, max_entries: int = 64, verify_every: int = 0, strict: bool = True,
                 log: Optional[Callable[[str], None]] = None, verbose: bool = False):
        self.name = str(name)
        self.max_entries = max(1, int(max_entries))
        self.verify_every = max(0, int(verify_every))
        self.strict = bool(strict)
        self._log = log
        self.verbose = bool(verbose)
        self._lock = threading.RLock()
        self._table: "OrderedDict[Any, tuple]" = OrderedDict()      # key -> (weakrefs, versions, value, nbytes)
        self._values: dict = {}                                       # name -> (explicit key, value, nbytes)
        self._rollout_key = None
        self._disabled: Optional[str] = None
        self.stats_ = {"hits": 0, "misses": 0, "bypassed": 0, "stale": 0, "verified": 0, "evictions": 0, "items": 0, "rollouts": 0, "refused_inference": 0, "disabled_calls": 0}

    def _say(self, line: str) -> None:
        text = f"[opt_core.hoist:{self.name}] {line}"
        if self._log is not None:
            self._log(text)
        else:
            print(text, file=sys.stderr, flush=True)

    @staticmethod
    def _sources(sources) -> Tuple:
        if _is_tensor(sources):
            return (sources,)
        return tuple(sources)

    @staticmethod
    def _key(srcs: Sequence, extra) -> tuple:
        return (tuple((id(t), tuple(t.shape), str(t.dtype), str(t.device)) for t in srcs), extra)

    @staticmethod
    def _nbytes(value) -> int:
        if _is_tensor(value):
            return int(value.numel() * value.element_size())
        if isinstance(value, (list, tuple)):
            return sum(ConstMemo._nbytes(v) for v in value)
        if isinstance(value, dict):
            return sum(ConstMemo._nbytes(v) for v in value.values())
        return 0

    def disable(self, reason: str) -> None:
        with self._lock:
            if self._disabled is None:
                self._disabled = str(reason)
                self._say(f"DISABLED: {reason} -> every later call computes inline")
                self._table.clear()
                self._values.clear()

    def new_item(self) -> None:
        """Item boundary: every entry dropped (hoisted values never outlive the item that made them)."""
        with self._lock:
            self.stats_["items"] += 1
            self._table.clear()
            self._values.clear()
            self._rollout_key = None

    reset = new_item

    def rollout(self, key):
        """Context manager of one roll-out (one ``sample()`` / one trajectory): entering with a ``key`` different from the current roll-out's drops
        every entry first (explicit invalidation); leaving keeps the entries until the next different key or :meth:`new_item`."""
        memo = self

        class _Scope:
            def __enter__(self_inner):
                with memo._lock:
                    if memo._rollout_key is None or memo._rollout_key != key:
                        memo._table.clear()
                        memo._values.clear()
                        memo.stats_["rollouts"] += 1
                    memo._rollout_key = key
                return memo

            def __exit__(self_inner, *exc):
                return False
        return _Scope()

    def value(self, name: str, key, compute: Callable[[], Any]):
        """Explicit-key form: the value called ``name`` for invalidation key ``key`` (anything hashable the adapter derives from the roll-out:
        a step-independent conditioning id, ``(id(z_trunk), z_trunk._version)``, a chunk index). A different key for the same name recomputes;
        inside a capture the value is computed inline and not stored (same bypass rule as :meth:`get`)."""
        with self._lock:
            if self._disabled is not None:
                self.stats_["disabled_calls"] += 1
                return compute()
            if capturing():
                self.stats_["bypassed"] += 1
                return compute()
            hit = self._values.get(name)
            if hit is not None and hit[0] == key:
                self.stats_["hits"] += 1
                if self.verify_every and self.stats_["hits"] % self.verify_every == 0:
                    self._verify((name, key), hit[1], compute)
                return hit[1]
            v = compute()
            self.stats_["misses"] += 1
            self._values[name] = (key, v, self._nbytes(v))
            return v

    def get(self, sources, compute: Callable[[], Any], extra=None):
        """The hoisted value for ``(sources, extra)``: computed now (miss, bypass) or the first call's object (hit)."""
        with self._lock:
            if self._disabled is not None:
                self.stats_["disabled_calls"] += 1
                return compute()
            if capturing():
                self.stats_["bypassed"] += 1
                return compute()
            srcs = self._sources(sources)
            if not srcs or not all(_is_tensor(t) for t in srcs):
                raise TypeError(f"{self.name}: sources must be one or more tensors (the objects the value depends on)")
            if any(getattr(t, "is_inference", lambda: False)() for t in srcs):
                self.stats_["refused_inference"] += 1                # inference-mode tensors carry no version counter: in-place mutation would be invisible
                if self.stats_["refused_inference"] == 1:
                    self._say("source is an inference-mode tensor (no version counter): value computed inline, never memoised — use value(name, key, ...) with an explicit key")
                return compute()
            key = self._key(srcs, extra)
            versions = tuple(int(t._version) for t in srcs)
            hit = self._table.get(key)
            if hit is not None:
                refs, vers, value, _ = hit
                alive = all(r() is t for r, t in zip(refs, srcs))
                if alive and vers == versions:
                    self._table.move_to_end(key)
                    self.stats_["hits"] += 1
                    if self.verify_every and self.stats_["hits"] % self.verify_every == 0:
                        self._verify(key, value, compute)
                    return value
                del self._table[key]                                  # same address, different or mutated tensor: a miss by name
                if self.verbose:
                    self._say(f"miss (source {'mutated in place' if alive else 'replaced'}) key={str(key)[:120]}")
            value = compute()
            self.stats_["misses"] += 1
            refs = tuple(weakref.ref(t) for t in srcs)
            self._table[key] = (refs, versions, value, self._nbytes(value))
            while len(self._table) > self.max_entries:
                self._table.popitem(last=False)
                self.stats_["evictions"] += 1
            return value

    def _verify(self, key, value, compute) -> None:
        from .graphs import bitwise_equal, max_abs_diff  # noqa: PLC0415
        fresh = compute()
        self.stats_["verified"] += 1
        if bitwise_equal(value, fresh):
            return
        self.stats_["stale"] += 1
        msg = f"hoisted value differs from a fresh computation (max_abs_diff={max_abs_diff(value, fresh)}) key={str(key)[:120]} — compute() reads state outside the key"
        if self.strict:
            raise MemoStale(f"{self.name}: {msg}")
        self.disable("stale: " + msg)

    # ------------------------------------------------------------------------------------------------------------- census
    def stats(self) -> dict:
        with self._lock:
            d = dict(self.stats_)
            d.update(name=self.name, entries=len(self._table) + len(self._values), bytes=sum(e[3] for e in self._table.values()) + sum(e[2] for e in self._values.values()),
                     disabled=self._disabled, state=self.state())
            return d

    def state(self) -> str:
        if self._disabled is not None:
            return "disabled:" + self._disabled.replace(" ", "_")[:80]
        if self.stats_["hits"]:
            return "active"
        if self.stats_["misses"] or self.stats_["bypassed"]:
            return "computed"
        return "idle"

    def evidence_fields(self) -> List[Tuple[str, Any]]:
        s = self.stats_
        return [("lever", self.name), ("state", self.state()), ("hits", s["hits"]), ("misses", s["misses"]), ("bypassed", s["bypassed"]),
                ("stale", s["stale"]), ("verified", s["verified"]), ("refused_inference", s["refused_inference"]), ("entries", len(self._table) + len(self._values)), ("items", s["items"]), ("rollouts", s["rollouts"]),
                ("mib", int((sum(e[3] for e in self._table.values()) + sum(e[2] for e in self._values.values())) / 2 ** 20)), ("strict", int(self.strict))]

    def evidence_line(self, tag: str, name: Optional[str] = None) -> str:
        """``[<tag>] LEVER name=<name> state=<on|skipped> impl=const_memo memo=<active|computed|idle> hits=… misses=… bypassed=… stale=… …``."""
        fields = self.evidence_fields()
        detail = [(k, v) for k, v in fields if k not in ("lever", "state")]
        if self._disabled is not None:
            head = [("name", name or self.name), ("state", "skipped"), ("reason", self._disabled), ("impl", "const_memo"), ("origin", "core"), ("memo", "disabled")]
        else:
            head = [("name", name or self.name), ("state", "on"), ("impl", "const_memo"), ("origin", "core"), ("memo", self.state())]
        return f"{report.prefix(tag)} LEVER {report.kv(*(head + detail))}"

    def partial(self) -> Optional[str]:
        if self._disabled is not None:
            return f"disabled: {self._disabled}"
        if self.stats_["stale"]:
            return f"stale={self.stats_['stale']}"
        return None


class RowsNotUniform(RuntimeError):
    """``RowDedup(strict_uniform=True)``: the rows the adapter declared identical are not (the adapter's construction claim is false)."""


class DedupMismatch(RuntimeError):
    """A checked served call differs from the statement on every row: ``fn`` is not row-independent along ``dim``, or its kernels are not
    row-invariant at this shape (the GEMM ``M`` changed). The site cannot claim the class its mode table names; under ``strict`` it raises."""


class RowDedup:
    """Row de-duplication of ONE statement site (strategy ``F7.row_dedup``). ``apply(fn, x, dim)`` replaces ``fn(x)`` where ``x`` (a tensor, or a
    tuple/list of tensors sharing the row axis ``dim``) holds ``n`` rows the adapter expects to be identical and ``fn`` is row-independent along
    ``dim`` (row ``r`` of every output leaf depends only on row ``r`` of the inputs). ``klass`` is the numerics class the kit's mode table
    declares for the site: ``'exact'`` (the served value is CLAIMED bit-exact to the ``n``-row statement; ``verify_every=k`` proves it on every k-th
    served call) or ``'tolerance'`` (row-count-dependent kernels, e.g. a GEMM whose reduction order follows ``M``: tested by the kit's tier-2 leg;
    ``verify_every`` stays 0 by declaration and the line says ``class=tolerance verified=0``). Per call exactly ONE counted outcome:

      ``single``    n == 1: ``fn(x)`` as is (not a fallback).
      ``bypassed``  inside a CUDA-graph capture / warm-up (:func:`capturing`) WITHOUT ``trusted``: ``fn(x)`` inline — the uniformity read is a host
                    sync, which a capture forbids; de-duplicate BEFORE the capture (hoist) or pass ``trusted=True`` (no read; narrow / fn / expand
                    are capturable, so a trusted site is served inside a capture; ``verify_every`` never fires inside a capture).
      ``served``    the rows are uniform (an exact on-device read: every row ``== row 0``, one comparison kernel and one host sync per input tensor) or the
                    adapter passed ``trusted=True`` (it built ``x`` by ``repeat`` / ``expand`` in the same statement; counted ``trusted`` too):
                    ``y1 = fn(<x narrowed to row 0, dim kept>)``, returned EXPANDED to ``n`` rows at ``out_dim`` (default ``dim``) on every tensor
                    leaf (tensor, tuple/list/dict of tensors) — a view of zero new bytes; an in-place write into it raises in torch (loud, never a
                    silent aliasing); ``materialize=True`` returns contiguous copies instead. ``rows_saved += n - 1``.
      ``fallback``  the rows are NOT uniform: ``fn(x)`` on every row (the stock result), ``fallback_by['nonuniform'] += 1`` — named, printed on the
                    line, gateable; ``strict_uniform=True`` raises :class:`RowsNotUniform` instead.
      ``disabled``  after a non-strict mismatch (below): ``fn(x)``, counted ``disabled_calls``.

    ``verify_every=k`` (> 0 on qualification jobs): every k-th served call also evaluates ``fn(x)`` on every row and demands bit-exact equality with the
    served (expanded) value — ``verified += 1``; a difference is ``mismatched += 1`` and raises :class:`DedupMismatch` under ``strict=True`` (the
    default, the setting of a tested line), else disables the site BY NAME (``state=skipped reason=mismatch:…``) and returns the full-row value.
    No state outlives a call (nothing is cached): :meth:`new_item` only counts. Evidence: :meth:`evidence_fields` / :meth:`evidence_line` —
    ``LEVER name=<name> state=on impl=row_dedup origin=core dedup=<active|idle|fallback> served= trusted= single= fallback= fallback_by=nonuniform:<n>
    verified= mismatched= bypassed= rows_saved=``; :meth:`gate` the fail-closed sentences a kit folds into its partial-activation exit."""

    IMPL = "row_dedup"
    CLASSES = ("exact", "tolerance")     # the site's declared numerics class: verify_every is for 'exact' claims; a 'tolerance' site keeps it 0 by declaration

    def __init__(self, name: str, *, klass: str = "exact", verify_every: int = 0, strict: bool = True, strict_uniform: bool = False,
                 materialize: bool = False, log: Optional[Callable[[str], None]] = None, verbose: bool = False):
        self.name = str(name)
        if klass not in self.CLASSES:
            raise ValueError(f"klass must be one of {self.CLASSES} (the numerics class the kit's mode table declares for this site)")
        self.klass = klass
        self.verify_every = max(0, int(verify_every))
        if self.klass == "tolerance" and self.verify_every:
            raise ValueError(f"{name}: verify_every demands bitwise equality with the full-row statement — meaningful only for a site declared "
                             "'exact'; a 'tolerance' site (row-count-dependent kernels) is certified by the kit's tier-2 leg and keeps verify_every=0")
        self.strict = bool(strict)
        self.strict_uniform = bool(strict_uniform)
        self.materialize = bool(materialize)
        self._log = log
        self.verbose = bool(verbose)
        self._lock = threading.RLock()
        self._disabled: Optional[str] = None
        self.stats_ = {"calls": 0, "served": 0, "trusted": 0, "single": 0, "fallback": 0, "verified": 0, "mismatched": 0, "bypassed": 0,
                       "disabled_calls": 0, "rows_saved": 0, "items": 0}
        self.fallback_by: dict = {}

    # ------------------------------------------------------------------------------------------------------------- helpers
    def _say(self, line: str) -> None:
        if self._log is not None:
            self._log(f"[row_dedup:{self.name}] {line}")
        elif self.verbose:
            sys.stderr.write(f"[row_dedup:{self.name}] {line}\n")

    @staticmethod
    def _inputs(x) -> Tuple[tuple, bool]:
        many = isinstance(x, (tuple, list))
        xs = tuple(x) if many else (x,)
        if not xs or not all(_is_tensor(t) for t in xs):
            raise TypeError("RowDedup.apply: x must be a tensor or a non-empty tuple/list of tensors")
        return xs, many

    @staticmethod
    def _axis(t, dim: int) -> int:
        nd = int(t.dim())
        d = int(dim) + nd if int(dim) < 0 else int(dim)
        if not 0 <= d < nd:
            raise ValueError(f"RowDedup.apply: dim {dim} out of range for a tensor of rank {nd}")
        return d

    @staticmethod
    def _uniform(t, d: int) -> bool:
        """Every row of ``t`` along axis ``d`` equals row 0 (exact ``==`` per element; ONE comparison kernel, one host sync). A NaN row is never
        uniform (``NaN != NaN``): such inputs take the full-row statement."""
        n = int(t.shape[d])
        if n <= 1:
            return True
        first = t.narrow(d, 0, 1)
        return bool((t == first).all().item())

    def _expand(self, y, n: int, out_dim: int):
        if _is_tensor(y):
            d = self._axis(y, out_dim)
            if int(y.shape[d]) != 1:
                raise DedupMismatch(f"{self.name}: fn's output has {int(y.shape[d])} rows at out_dim={out_dim} for a 1-row input (expected 1): fn is not row-independent along dim")
            shape = list(y.shape)
            shape[d] = int(n)
            v = y.expand(*shape)
            return v.contiguous() if self.materialize else v
        if isinstance(y, tuple):
            return tuple(self._expand(v, n, out_dim) for v in y)
        if isinstance(y, list):
            return [self._expand(v, n, out_dim) for v in y]
        if isinstance(y, dict):
            return {k: self._expand(v, n, out_dim) for k, v in y.items()}
        return y                                                       # non-tensor leaves (None, ints, flags) pass through

    def disable(self, reason: str) -> None:
        with self._lock:
            if self._disabled is None:
                self._disabled = str(reason)
                self._say(f"DISABLED: {reason} -> every later call runs the statement on every row")

    def new_item(self) -> None:
        """Item boundary (accepted for symmetry with :class:`ConstMemo`; nothing is cached, so only counted)."""
        with self._lock:
            self.stats_["items"] += 1

    reset = new_item

    # ------------------------------------------------------------------------------------------------------------- the call
    def apply(self, fn: Callable[[Any], Any], x, dim: int, *, trusted: bool = False, out_dim: Optional[int] = None):
        """``fn(x)`` de-duplicated along ``dim`` (module contract). ``x``: a tensor or a tuple/list of tensors sharing ``n = x.shape[dim]``; ``fn``
        receives the same structure narrowed to row 0 (``dim`` kept, size 1) on the served path and ``x`` itself on every other path."""
        xs, many = self._inputs(x)
        axes = [self._axis(t, dim) for t in xs]
        n = int(xs[0].shape[axes[0]])
        if any(int(t.shape[a]) != n for t, a in zip(xs, axes)):
            raise ValueError(f"RowDedup.apply: every input must have {n} rows along dim={dim} (got {[int(t.shape[a]) for t, a in zip(xs, axes)]})")
        with self._lock:
            self.stats_["calls"] += 1
            if self._disabled is not None:
                self.stats_["disabled_calls"] += 1
                return fn(x)
            if n == 1:
                self.stats_["single"] += 1
                return fn(x)
            if trusted:                                             # no host read: narrow / fn / expand are capturable, so a trusted site serves inside a capture too
                self.stats_["trusted"] += 1
            elif capturing():                                       # the uniformity read is a host sync a capture forbids: the statement runs inline on every row
                self.stats_["bypassed"] += 1
                return fn(x)
            elif not all(self._uniform(t, a) for t, a in zip(xs, axes)):
                if self.strict_uniform:
                    raise RowsNotUniform(f"{self.name}: the {n} rows along dim={dim} are not identical (strict_uniform)")
                self.stats_["fallback"] += 1
                self.fallback_by["nonuniform"] = self.fallback_by.get("nonuniform", 0) + 1
                if self.fallback_by["nonuniform"] == 1:
                    self._say(f"rows not uniform along dim={dim} (n={n}) -> the statement runs on every row (counted fallback_by=nonuniform)")
                return fn(x)
            narrowed = tuple(t.narrow(a, 0, 1) for t, a in zip(xs, axes))
            y1 = fn(narrowed if many else narrowed[0])
            y = self._expand(y1, n, dim if out_dim is None else out_dim)
            self.stats_["served"] += 1
            self.stats_["rows_saved"] += n - 1
            if self.verify_every and self.stats_["served"] % self.verify_every == 0 and not capturing():   # a check is a host read: never inside a capture
                return self._verify(fn, x, y, n, dim)
            return y

    def _verify(self, fn, x, y, n: int, dim: int):
        from .graphs import bitwise_equal, max_abs_diff  # noqa: PLC0415
        full = fn(x)
        self.stats_["verified"] += 1
        if bitwise_equal(full, y):
            return y
        self.stats_["mismatched"] += 1
        msg = (f"mismatch: the 1-row statement expanded to {n} rows differs from the {n}-row statement (max_abs_diff={max_abs_diff(full, y)}, dim={dim}) "
               "- fn is not row-independent, or its kernels are not row-invariant at this shape")
        if self.strict:
            raise DedupMismatch(f"{self.name}: {msg}")
        self.disable(msg)
        return full

    # ------------------------------------------------------------------------------------------------------------- census
    def stats(self) -> dict:
        with self._lock:
            d = dict(self.stats_)
            d.update(name=self.name, klass=self.klass, fallback_by=dict(self.fallback_by), disabled=self._disabled, state=self.state())
            return d

    def state(self) -> str:
        if self._disabled is not None:
            return "disabled:" + self._disabled.replace(" ", "_")[:80]
        if self.stats_["served"]:
            return "active"
        if self.stats_["fallback"]:
            return "fallback"
        return "idle"

    def evidence_fields(self) -> List[Tuple[str, Any]]:
        s = self.stats_
        fb = ",".join(f"{k}:{v}" for k, v in sorted(self.fallback_by.items())) or "none"
        return [("lever", self.name), ("state", self.state()), ("class", self.klass), ("served", s["served"]), ("trusted", s["trusted"]), ("single", s["single"]),
                ("fallback", s["fallback"]), ("fallback_by", fb), ("verified", s["verified"]), ("mismatched", s["mismatched"]), ("bypassed", s["bypassed"]),
                ("disabled_calls", s["disabled_calls"]), ("rows_saved", s["rows_saved"]), ("verify_every", self.verify_every), ("strict", int(self.strict))]

    def evidence_line(self, tag: str, name: Optional[str] = None, strategy: Optional[str] = "F7.row_dedup") -> str:
        """``[<tag>] LEVER name=<name> state=<on|skipped> [reason=…] impl=row_dedup origin=core strategy=F7.row_dedup dedup=<active|idle|fallback> class=<exact|tolerance> served=… …``
        (:func:`opt_core.report.lever_line`; ``strategy=None`` omits the word for a kit that prints its own)."""
        detail = [(k, v) for k, v in self.evidence_fields() if k not in ("lever", "state")]
        if self._disabled is not None:
            return report.lever_line(tag, name or self.name, "skipped", ("dedup", "disabled"), *detail, reason=self._disabled.replace(" ", "_")[:120],
                                     impl=self.IMPL, origin="core", strategy=strategy)
        return report.lever_line(tag, name or self.name, "on", ("dedup", self.state()), *detail, impl=self.IMPL, origin="core", strategy=strategy)

    def gate(self, *, expect_fallback: Sequence[str] = (), max_fallback: Optional[int] = 0, require_served: bool = False) -> List[str]:
        """The fail-closed sentences (empty = pass): a fallback reason not in ``expect_fallback``; declared fallbacks above ``max_fallback`` (``None`` =
        any count); any ``mismatched``; a disabled site; ``require_served`` with nothing served. The kit folds them into its partial-activation exit."""
        s, out = self.stats_, []
        for reason, k in sorted(self.fallback_by.items()):
            if reason not in tuple(expect_fallback):
                out.append(f"{self.name}: undeclared fallback {reason}={k}")
            elif max_fallback is not None and k > int(max_fallback):
                out.append(f"{self.name}: fallback {reason}={k} above the expected {int(max_fallback)}")
        if s["mismatched"]:
            out.append(f"{self.name}: mismatched={s['mismatched']} (served value differs from the full-row statement)")
        if self._disabled is not None:
            out.append(f"{self.name}: disabled: {self._disabled}")
        if require_served and not s["served"]:
            out.append(f"{self.name}: requested but never served (calls={s['calls']} single={s['single']} bypassed={s['bypassed']} fallback={s['fallback']})")
        return out

    def partial(self) -> Optional[str]:
        problems = self.gate(expect_fallback=("nonuniform",), max_fallback=None)
        return "; ".join(problems) if problems else None
