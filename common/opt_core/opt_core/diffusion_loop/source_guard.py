"""The refuse-to-patch rule over a stock function an adapter replaces, engine-free.

Contract. An adapter that re-executes, wraps or monkey-patches an upstream function was written against one text of that function.
:func:`source_guard` hashes the de-indented source of the function as the running installation has it and returns an
:class:`opt_core.gates.Gate`: ok when the sha256 is one the adapter names (and, when given, every required substring is present), refused
with a sentence naming the function and the observed digest otherwise — so a changed upstream is a recorded refusal, the adapter installs
nothing, and the stock function stays in place. With neither a digest nor a required text the gate refuses (an unguarded patch is not a
guard). :func:`source_sha256` is the digest an adapter records in its table. Standard library only.
"""
from __future__ import annotations

import hashlib
import inspect
import textwrap
from typing import Iterable, Optional, Sequence

def source_sha256(fn) -> str:
    """sha256 of the de-indented source text of ``fn`` (function, method or class), as ``inspect.getsource`` reads it."""
    text = textwrap.dedent(inspect.getsource(fn))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_guard(fn, expected_sha256: Iterable[str] = (), contains: Sequence[str] = (), name: Optional[str] = None):
    """The refuse-to-patch gate over a stock function. ``expected_sha256``: the digests (``source_sha256``) of every upstream text the
    adapter was written against — at least one must match when any are given; ``contains``: substrings that must all be present (the
    weaker form, for an adapter that tolerates unrelated edits). With neither given the gate refuses (an unguarded patch is not a
    guard). Returns :class:`opt_core.gates.Gate` named ``name`` (default ``source_guard:<qualname>``) whose details carry the observed
    digest, so a refusal on a new upstream prints the value the adapter's table needs."""
    from opt_core.gates import Gate
    qual = getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn)))
    gate_name = name or f"source_guard:{qual}"
    expected = [s.lower() for s in expected_sha256]
    if not expected and not contains:
        return Gate(name=gate_name, ok=False, reason=f"{qual}: source guard has neither a digest nor a required text — refusing to patch")
    try:
        text = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as e:
        return Gate(name=gate_name, ok=False, reason=f"{qual}: source not readable ({e.__class__.__name__}) — refusing to patch")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    details = {"sha256": digest, "expected": expected, "contains": list(contains)}
    if expected and digest not in expected:
        return Gate(name=gate_name, ok=False, details=details,
                    reason=f"{qual}: upstream source changed (sha256 {digest[:12]}… matches none of {len(expected)} known) — refusing to patch")
    missing = [s for s in contains if s not in text]
    if missing:
        return Gate(name=gate_name, ok=False, details=dict(details, missing=missing),
                    reason=f"{qual}: upstream source lacks {len(missing)} required text(s) ({missing[0]!r}…) — refusing to patch")
    return Gate(name=gate_name, ok=True, details=details)
