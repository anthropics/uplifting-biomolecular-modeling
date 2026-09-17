"""opt_core.mem.primitives — the memory PRIMITIVES of the package (re-exported by :mod:`opt_core.mem`): engine-free mechanisms a kit's engine adapter composes into its own
modules. The core holds primitives only; the INSTALL of a memory line (which module attribute is rebound, at which token count, in
which mode) is the kit's adapter (``<pkg>.big`` by convention) and the kit's mode table.

Layout, by mechanism (the framework submodules import torch inside their functions, never at import; this module is standard
library only and imports nothing heavy):

    mem                 the refusal type (:class:`MemLeverRefused`), row-block plans (:func:`row_blocks`), the per-process counter
                        ledger (:class:`Ledger`), the activation-evidence line of a memory lever (:func:`evidence_line`), env parsing
    mem.torch_rowchunk  row-chunked evaluation of row-independent pair-tensor sub-modules into ONE preallocated output (transition,
                        conditioner prologues, per-layer pair biases), the pair-shape guard, attribute patch sets, dead-tensor release
    mem.torch_alloc     the CUDA caching-allocator policy of a process (``PYTORCH_CUDA_ALLOC_CONF`` expandable segments / gc threshold):
                        export-at-activation with named refusals, the effective-state read-back, the env row for a child process
    mem.torch_hostpair  a pinned-host mirror of a pair tensor served to the device in row blocks, a pinned buffer pool, the row-split
                        guard for layer_norm inputs at or above 2**31 elements
    mem.graph_gate      the token-gated capture decision of a CUDA-graph / static-arena lever and its line fragment

Contract (every submodule):
  * FAIL-LOUD. A primitive either acts or raises :class:`MemLeverRefused` naming the lever and the reason (library missing, shape or
    dtype outside the primitive's domain, allocator already initialised, ...). There is no record-and-continue path in the core: an
    adapter that wants the stock path for a DECLARED guard (training mode, autograd, below the token threshold, a call form the lever
    does not cover) tests the guard itself, counts the stock call in its ledger, and calls the stock attribute — a guard is data in the
    kit's mode table and its census is in the evidence line; an exception inside an acting lever is never converted into a stock call.
    CUDA out-of-memory raised inside a primitive propagates unchanged (``is_oom`` in ``torch_rowchunk`` classifies it).
  * NUMERICS. Row chunking changes WHEN memory is allocated and the ROW COUNT of the kernel launches of row-independent operations;
    per-element arithmetic, dtypes and the order of reductions inside every element are the caller's. Whether a row-chunked launch
    returns the same bits as the full launch is a property of the stack (cuBLAS / cuDNN heuristics may pick a kernel by M), so the
    class of a memory line (bit-exact vs the kit's base, or inside the engine's band) is tested per stack by the kit's
    equality record — never asserted here. Allocator policy and tensor release change placement and lifetime only.
  * EVIDENCE. Every primitive returns (or records in a :class:`Ledger`) the facts an adapter needs to print ONE activation-evidence
    line per lever per process with :func:`evidence_line` (the core's per-lever grammar ``[<tag>] LEVER name=… state=… k=v``, composed
    from ``opt_core.report.prefix`` / ``kv``).
  * No engine name, no engine import, no lever value: thresholds, row counts, lever names and env variable names are the kit's.
"""
from __future__ import annotations

import threading
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = ["MemLeverRefused", "row_blocks", "Ledger", "evidence_line", "env_int", "env_flag", "parse_levers"]


class MemLeverRefused(RuntimeError):
    """A memory primitive refused to act: ``lever`` names the lever (the adapter's registry name), ``reason`` the precondition that
    failed. The adapter lets it propagate (the kit's exit rule turns it into its NOT ACTIVE / failed line) — never a silent stock run."""

    def __init__(self, lever: str, reason: str):
        self.lever = str(lever)
        self.reason = str(reason)
        super().__init__(f"memory lever {self.lever!r} refused: {self.reason}")


def row_blocks(n: int, rows: int) -> List[Tuple[int, int]]:
    """``[(r0, r1), ...]`` covering ``range(n)`` in blocks of ``rows`` (the last block may be short); ``rows >= n`` gives one block.
    ``rows < 1`` or ``n < 0`` is a usage error (ValueError), not a refusal."""
    n, rows = int(n), int(rows)
    if rows < 1:
        raise ValueError(f"row_blocks: rows must be >= 1 (got {rows})")
    if n < 0:
        raise ValueError(f"row_blocks: n must be >= 0 (got {n})")
    return [(r0, min(r0 + rows, n)) for r0 in range(0, n, rows)]


class Ledger:
    """Per-process counters of a memory line (thread-safe): ``count(key, n)`` adds, ``peak(key, value)`` keeps a maximum, ``set(key, v)``
    records a value; ``facts()`` is a plain-dict copy for the evidence line and the kit's manifest. Keys are the adapter's
    (``<lever>_rowchunked_calls``, ``<lever>_stock_calls``, ``free_events`` ...)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._c: Dict[str, object] = {}

    def count(self, key: str, n: int = 1) -> int:
        with self._lock:
            v = int(self._c.get(key, 0) or 0) + int(n)
            self._c[key] = v
            return v

    def peak(self, key: str, value) -> object:
        with self._lock:
            cur = self._c.get(key)
            if cur is None or value > cur:
                self._c[key] = value
            return self._c[key]

    def set(self, key: str, value) -> None:
        with self._lock:
            self._c[key] = value

    def get(self, key: str, default=None):
        with self._lock:
            return self._c.get(key, default)

    def facts(self) -> Dict[str, object]:
        with self._lock:
            return dict(self._c)

    def clear(self) -> None:
        with self._lock:
            self._c.clear()


def evidence_line(tag: str, lever: str, state: str = "on", reason: Optional[str] = None, **facts) -> str:
    """The ONE activation-evidence line of a memory lever per process, in the core's per-lever grammar:
    ``[<tag>] LEVER name=<lever> state=<on|off|skipped> [reason=<why>] k=v ...`` (``opt_core.report.prefix`` / ``kv``: lists print
    ``a,b``, None prints ``none``, dicts ``k:v``). ``state``: ``on`` = applied and live in this process; ``off`` = not in this mode's lever
    set; ``skipped`` = selected but not applied — ``reason`` is then mandatory (it feeds the kit's partial census). The adapter prints it on
    stderr once per lever (``name`` may be a composed ``a,b,c`` for one memory line); the launcher matches its bytes."""
    from ..report import kv, prefix
    if state not in ("on", "off", "skipped"):
        raise ValueError(f"evidence_line: state must be on|off|skipped (got {state!r})")
    if state == "skipped" and not reason:
        raise ValueError("evidence_line: state=skipped requires reason=")
    pairs = [("name", lever), ("state", state)] + ([("reason", reason)] if reason else []) + list(facts.items())
    return f"{prefix(tag)} LEVER {kv(*pairs)}"


def env_int(environ: Optional[Mapping[str, str]], name: str, default: int) -> int:
    """An integer env knob: unset / empty -> ``default``; a non-integer value is a usage error (ValueError naming the variable) —
    a memory row never runs on a silently defaulted knob."""
    v = (environ or {}).get(name, "")
    if v is None or str(v).strip() == "":
        return int(default)
    try:
        return int(str(v).strip())
    except ValueError:
        raise ValueError(f"{name}={v!r}: an integer is required") from None


def env_flag(environ: Optional[Mapping[str, str]], name: str, default: bool = False) -> bool:
    """A boolean env knob: ``1/true/on/yes`` -> True, ``0/false/off/no`` or empty -> False when set; unset -> ``default``;
    anything else is a usage error (ValueError)."""
    if environ is None or name not in environ:
        return bool(default)
    v = str(environ.get(name, "")).strip().lower()
    if v in ("1", "true", "on", "yes"):
        return True
    if v in ("", "0", "false", "off", "no"):
        return False
    raise ValueError(f"{name}={environ.get(name)!r}: expected one of 1/true/on/yes/0/false/off/no")


def parse_levers(spec: Optional[str], known: Sequence[str], default: Iterable[str] = ()) -> Tuple[str, ...]:
    """A lever list from a spec string: ``None``/``""``/``0``/``off``/``none`` -> ``()``; ``1``/``all``/``on``/``default`` -> ``default``
    (or every known lever when no default is given); else a ``,``/``+`` separated list, each name held to ``known`` (an unknown name
    raises :class:`MemLeverRefused` naming it — the row is refused, never trimmed), duplicates dropped, order kept."""
    s = (spec or "").strip()
    if s.lower() in ("", "0", "off", "none", "stock", "false"):
        return tuple()
    if s.lower() in ("1", "all", "on", "default", "true"):
        d = tuple(default) or tuple(known)
        return tuple(d)
    out: List[str] = []
    for tok in s.replace("+", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in known:
            raise MemLeverRefused(tok, f"unknown lever name; known: {','.join(known)}")
        if tok not in out:
            out.append(tok)
    return tuple(out)
