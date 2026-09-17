"""CPU tests of GraphCache.handle() / ReplayHandle (the bare-replay view of an armed entry): the uncaptured handle on a machine without CUDA
(named reason, eager value, HandleUnavailable), and the captured handle's replay / copy_in / alive / stale semantics over a duck-typed entry
(a fake graph counting replays) — the real capture path is GraphCache.run's, exercised on GPU by the adopting kits' equality legs."""
from __future__ import annotations

import pytest

from opt_core.capture import graphs
from opt_core.capture.graphs import GraphCache, HandleStale, HandleUnavailable, ReplayHandle


def test_handle_without_cuda_is_uncaptured_named_and_serves_the_eager_value():
    g = GraphCache("dev", strict=False)
    h = g.handle(lambda x: x + 1, 1)
    assert isinstance(h, ReplayHandle) and not h.captured and not h.alive and h.out == 2 and h.first_out == 2
    assert h.reason.startswith("disabled:") and h.replays == 0 and h.static_args is None
    with pytest.raises(HandleUnavailable):
        h.replay()
    assert g.handle_stats == {"handles": 1, "captured": 0, "eager": 1, "replays": 0, "verified": 0, "stale": 0}
    assert [k for k, _ in g.handle_fields()] == ["handles", "handle_captured", "handle_eager", "handle_replays", "handle_verified", "handle_stale"]
    with pytest.raises(graphs.CaptureUnavailable):
        GraphCache("strict").handle(lambda x: x, 1)


class _FakeGraph:
    def __init__(self):
        self.n = 0

    def replay(self):
        self.n += 1


def _armed_entry(cache, key=("k",), out=None):
    ent = graphs._Entry(key)
    ent.graph, ent.armed, ent.static_args, ent.static_kwargs, ent.static_out = _FakeGraph(), True, (), {}, out if out is not None else {"y": 7}
    cache._entries[key] = ent
    return ent


def test_captured_handle_replays_bare_counts_and_goes_stale_by_name():
    g = GraphCache("hot", strict=False)
    seen = []
    g.on_replay = seen.append
    ent = _armed_entry(g)
    h = ReplayHandle(g, ent, ent.static_out, None)
    assert h.captured and h.alive and h.key == ("k",) and h.out is ent.static_out
    for _ in range(5):
        assert h.replay() is ent.static_out
    assert ent.graph.n == 5 and h.replays == 5 and g.stats_["replays"] == 5 and g.handle_stats["replays"] == 5 and seen == [("k",)] * 5
    assert h.copy_in() == 0                                          # no tensor leaves in this fake tree: nothing to copy, no error
    g._entries.clear()                                               # the cache dropped the entry (reset / eviction / failed checking)
    assert not h.alive and h.first_out == {"y": 7}                   # the value the handle() call returned stays readable by name
    with pytest.raises(HandleStale):
        h.out                                                        # a captured-then-stale handle serves nothing silently
    with pytest.raises(HandleStale):
        h.replay()
    assert g.handle_stats["stale"] == 2


def test_key_for_is_the_one_key_rule_of_run_and_handle():
    g = GraphCache("k", strict=False, extra_key=lambda: "mode-a")
    k1 = g._key_for((1, "x"), {})
    assert k1[0] == "mode-a" and k1[2] is None and k1 == g._key_for((1, "x"), {})


def test_run_records_the_key_it_decided_under_for_handle():
    g = GraphCache("k2", strict=False)
    box = []
    g._run(lambda x: x, (1,), {}, box)                               # no CUDA here: answered before a key is formed -> box stays empty
    h = g.handle(lambda x: x, 1)
    assert box == [] and not h.captured and (h.reason.startswith("disabled:") or h.reason.startswith("eager:"))
