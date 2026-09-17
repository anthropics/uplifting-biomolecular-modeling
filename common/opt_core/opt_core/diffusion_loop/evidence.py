"""The fail-closed census check of a sampling-loop mechanism, engine-free.

Contract. A kit prints ONE activation line per arm per process, composed with :func:`opt_core.report.prefix` / :func:`opt_core.report.kv`
from the ``evidence_fields()`` every mechanism returns (this package formats nothing). :func:`census_gate` is the count check that goes with
it: every key of the kit's ``expected`` mapping whose observed value EXCEEDS it (fallback counts: refusals, syncs, restores short of
checkpoints …) is a refusal naming key, observed and expected; a key of ``expected`` absent from ``observed`` is a refusal too (an unobserved
count is not a zero). It returns an :class:`opt_core.gates.Gate` so the kit records it beside its other gates.
"""
from __future__ import annotations

from typing import Mapping


def census_gate(observed: Mapping, expected: Mapping, name: str = "loop_census"):
    """``observed``: a mechanism's ``evidence_fields()``; ``expected``: the kit's ceiling per key. Over or unobserved = refused Gate."""
    from opt_core.gates import Gate
    missing = sorted(k for k in expected if k not in observed)
    over = {k: (observed[k], v) for k, v in expected.items() if k in observed and _num(observed[k]) > _num(v)}
    if missing or over:
        parts = [f"{k}: observed {o} > expected {e}" for k, (o, e) in sorted(over.items())]
        parts += [f"{k}: not observed" for k in missing]
        return Gate(name=name, ok=False, reason="loop census over expectation — " + ", ".join(parts),
                    details={"over": over, "missing": missing})
    return Gate(name=name, ok=True, details={"checked": sorted(expected)})


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        raise TypeError(f"census value {v!r} is not a count") from None
