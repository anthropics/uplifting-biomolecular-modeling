"""The `tuner_guard` lever (exact class): one process serving items of any mix of token counts through the engine's chunk-size tuner.

The engine's `ChunkSizeTuner` (one per `PairFormerStack` instance) caches the argument record of its last tuning — the shapes of the
(s, z) pair it tuned on — and, on the next chunked call, compares that record with the new call's to decide whether to re-tune
(`_compare_arg_caches`: `zip(strict=True)` over the records, recursing into `torch.Size`, `assert type(a1) is type(a2)`). Two call forms of
one stack whose tensors differ in RANK make that comparison raise instead of answering "changed": the confidence head calls its 4-block stack
BATCHED (samples folded into the leading dim: z rank 4) at or below its per-sample token cutoff (750) and PER SAMPLE (batch dims kept: z rank 5)
above it, so a process that predicts a query of <= 750 tokens and then a larger one fails every larger query with
`ValueError: zip() argument 2 is longer than argument 1` — on a runner configuration whose alternative-kernel flags are off (the engine keeps the
batched call's chunk size, and tunes it, only then; with a kernel flag on the batched call runs unchunked and nothing is cached, which is why
the stock configuration never meets it).

This lever wraps `ChunkSizeTuner._compare_arg_caches`: a comparison that RAISES (ValueError from the strict zip, AssertionError from the type
check) is a structurally different record — the tuner's own "args have changed shape/value, we need to re-tune" case — and is answered
`False` (inconsistent), counted by exception class on the census; the tuner then re-tunes for the new shapes and caches the new record, as
it does for any other change. Comparable records go through the engine's comparison untouched. Chunk size never changes the arithmetic of
a chunked layer (the engine tunes it per process already): exact.

Switch (the kit adapter's, `configure(ENV=…, M_CHUNK=…)`): <KIT>_TUNER_GUARD=1. Exit line
`<PREFIX> LEVER name=tuner_guard state=on resets=<Exc:n,…|none>` (resets: comparisons answered 'changed' because they raised, by exception class).
"""
from __future__ import annotations

import atexit
import os
import sys
from typing import Any, Dict, Optional

PREFIX = "[opt_core/of3_trunk.tuner_guard]"
ENV: Optional[str] = None                                # <KIT>_TUNER_GUARD=1
VALUES = ("1",)
M_CHUNK: Optional[str] = None                            # ….core.utils.chunk_utils (class ChunkSizeTuner)
CONFIGURABLE = ("PREFIX", "ENV", "M_CHUNK")

STATE: Dict[str, Any] = {"installed": False, "state": "off", "reason": "", "resets": {}}


def configure(**kw) -> None:
    """The kit adapter's binding: reassigns this module's engine words before install; unknown names raise."""
    for k, v in kw.items():
        if k not in CONFIGURABLE:
            raise KeyError(f"{__name__}.configure: unknown setting {k!r} (known: {', '.join(CONFIGURABLE)})")
        globals()[k] = v


def _log(msg: str) -> None:
    sys.stderr.write(f"{PREFIX} {msg}\n")


def _count(d: dict, k, n: int = 1) -> None:
    d[k] = d.get(k, 0) + n


def requested(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    if not ENV:
        return False
    v = (environ.get(ENV) or "").strip()
    if not v:
        return False
    if v not in VALUES:
        raise ValueError(f"{ENV}={v!r} is not one of {'|'.join(VALUES)}")
    return True


def serving() -> bool:
    return STATE["state"] == "on"


def census_line() -> str:
    resets = ",".join("%s:%d" % kv for kv in sorted(STATE["resets"].items())) or "none"
    return (f"{PREFIX} LEVER name=tuner_guard state={STATE['state']}" + (f" reason={STATE['reason']}" if STATE["reason"] else "") +
            f" resets={resets}")


def _make_compare(orig):
    def _compare_arg_caches(self, ac1, ac2):
        try:
            return orig(self, ac1, ac2)
        except (ValueError, AssertionError) as e:                          # records of different structure (tensor rank / entry type): changed args —
            _count(STATE["resets"], type(e).__name__)                     # the tuner re-tunes for the new call and caches its record
            if STATE["resets"][type(e).__name__] == 1:
                _log(f"argument records of different structure ({type(e).__name__}: {str(e)[:120]}) answered 'changed': the tuner re-tunes "
                     "for this call form (counted `resets` on the census from here on)")
            return False
    _compare_arg_caches.__wrapped__ = orig; _compare_arg_caches._of3opt_tuner_guard = True
    return _compare_arg_caches


def install(environ=None) -> dict:
    """Rebind `ChunkSizeTuner._compare_arg_caches` on the engine's chunk_utils module. Idempotent; an engine without the class / method is
    refused by name (state=refused, nothing patched)."""
    environ = os.environ if environ is None else environ
    if STATE["installed"] or not requested(environ):
        return STATE
    if not M_CHUNK:
        raise RuntimeError(f"{PREFIX} not bound to an engine: the kit adapter must configure(M_CHUNK=) before install")
    import importlib
    try:
        C = importlib.import_module(M_CHUNK)
        cls = C.ChunkSizeTuner
        orig = cls._compare_arg_caches
    except Exception as e:  # noqa: BLE001
        from ..oom import is_oom
        if is_oom(e):
            raise
        STATE.update(installed=True, state="refused", reason=f"no_tuner:{type(e).__name__}")
        _log(f"REFUSED: {M_CHUNK}.ChunkSizeTuner._compare_arg_caches not found ({type(e).__name__}: {e}) — the engine's tuner runs as it is")
        atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
        return STATE
    if not getattr(orig, "_of3opt_tuner_guard", False):
        cls._compare_arg_caches = _make_compare(orig)
    STATE.update(installed=True, state="on")
    _log("installed: ChunkSizeTuner._compare_arg_caches answers 'changed' for argument records of different structure (the tuner re-tunes) instead of raising")
    atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
    return STATE
