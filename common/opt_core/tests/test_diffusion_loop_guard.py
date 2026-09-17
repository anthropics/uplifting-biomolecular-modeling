"""The source guard (known digest accepted, changed text / missing text / unguarded / unreadable refused) and the fail-closed census gate — CPU, no torch, no engine."""
import pytest

from opt_core.diffusion_loop import evidence
from opt_core.diffusion_loop import source_guard as sg


def _stock_fn(x):
    y = x + 1  # marker-text
    return y


def test_source_guard_accepts_known_digest_and_refuses_changes():
    digest = sg.source_sha256(_stock_fn)
    ok = sg.source_guard(_stock_fn, expected_sha256=[digest.upper()])
    assert ok.ok and ok.details["sha256"] == digest and ok.name == "source_guard:_stock_fn"
    changed = sg.source_guard(_stock_fn, expected_sha256=["0" * 64])
    assert not changed.ok and "refusing to patch" in changed.reason and changed.details["sha256"] == digest
    weak = sg.source_guard(_stock_fn, contains=["marker-text"])
    assert weak.ok
    weak_missing = sg.source_guard(_stock_fn, contains=["marker-text", "not there"], name="g")
    assert not weak_missing.ok and weak_missing.name == "g" and weak_missing.details["missing"] == ["not there"]
    assert not sg.source_guard(_stock_fn).ok                       # unguarded = refused
    assert not sg.source_guard(len, contains=["x"]).ok            # builtins have no source: refused, not raised


def test_census_gate_is_fail_closed():
    g = evidence.census_gate({"passthrough": 0, "mismatches": 0, "rekeyed": 2}, {"passthrough": 0, "mismatches": 0, "rekeyed": 4})
    assert g.ok and g.details["checked"] == ["mismatches", "passthrough", "rekeyed"]
    g = evidence.census_gate({"passthrough": 3, "mismatches": 0}, {"passthrough": 0, "mismatches": 0, "rekeyed": 0})
    assert not g.ok and "passthrough: observed 3 > expected 0" in g.reason and "rekeyed: not observed" in g.reason
    with pytest.raises(TypeError):
        evidence.census_gate({"passthrough": "many"}, {"passthrough": 0})
