"""opt_core.seq.batch_plan: the int32 bound, chunked dispatch, the tested-shape regime plan, the line and the counts."""
import ast
import os

import pytest

from opt_core.seq import batch_plan as bp


def test_max_units_int32_forms():
    # a store of L/2 positions x 512 channels per unit at L = 524,288 -> 16 units (2**31 // (262144 * 512))
    assert bp.max_units((524288 // 2) * 512) == 16
    # n*C*L < 2**31 at C=512, L=2114 -> 1984 units; a kit's own lower plateau caps it
    assert bp.max_units(512 * 2114) == 1984
    assert bp.max_units(512 * 2114, cap=1024) == 1024
    c = bp.ceiling(512 * 2114, cap=1024)
    assert (c.n, c.kind) == (1024, "certified") and "certified ceiling 1024 < the int32 bound 1984" in c.basis
    c2 = bp.ceiling(512 * 2114, cap=4096)
    assert (c2.n, c2.kind) == (1984, "int32") and "not binding" in c2.basis
    assert bp.ceiling((524288 // 2) * 512) == bp.Ceiling(16, "int32", "N x 134217728 <= 2^31")
    assert bp.max_units(1 << 40) == 1                      # never 0
    assert bp.max_units(3, index_bits=63) == (1 << 63) // 3
    with pytest.raises(ValueError):
        bp.max_units(0)


def test_check_bound_two_forms():
    bp.check_bound("t", 9, below=10)
    bp.check_bound("t", 1984, at_most=1984)
    with pytest.raises(bp.BoundRefused) as ei:
        bp.check_bound("bare route", 10, below=10, unit="tokens", advice="score fewer rows per call")
    assert ei.value.requested == 10 and ei.value.bound == 9 and "tokens <= 9" in str(ei.value) and "score fewer" in str(ei.value)
    assert isinstance(ei.value, ValueError)
    with pytest.raises(bp.BoundRefused) as e2:
        bp.check_bound("k1", 1985, at_most=1984)
    assert e2.value.bound == 1984
    with pytest.raises(ValueError):
        bp.check_bound("both", 1, at_most=2, below=3)
    with pytest.raises(ValueError):
        bp.check_bound("neither", 1)


def test_plan_chunks_is_torch_split_order():
    p = bp.plan_chunks(37, 16)
    assert [(c.start, c.n) for c in p.chunks] == [(0, 16), (16, 16), (32, 5)]
    assert p.n_calls == 3 and p.padded_units == 0 and p.sizes() == [16, 16, 5]
    assert "B=37 > bound 16: 2 call(s) of 16 + 1 of 5" in p.reason
    one = bp.plan_chunks(16, 16)
    assert one.n_calls == 1 and "one call" in one.reason
    assert bp.plan_chunks(0, 4).n_calls == 0
    with pytest.raises(ValueError):
        bp.plan_chunks(3, 0)
    assert bp.counts(p) == {"requested": 37, "calls": 3, "padded_rows": 0, "calls_chunk": 3}


def test_regime_pieces_with_costs_matches_the_chunks_of_8_plus_b1_rule():
    # tested routes {8: uf, 1: b1}; remainder r as b1 pieces iff r*b1 <= uf, else one padded uf call
    rp = bp.RegimePlan({8: "uf", 1: "b1"}, tail="pieces", piece_cost={1: 30.0, 8: 100.0})
    p = rp.plan(19)                                        # 2 x uf + r=3: 3*30 = 90 <= 100 -> 3 b1 pieces
    assert [c.kind for c in p.chunks] == ["uf", "uf", "b1", "b1", "b1"] and p.padded_units == 0
    assert [(c.start, c.n) for c in p.chunks] == [(0, 8), (8, 8), (16, 1), (17, 1), (18, 1)]
    p2 = rp.plan(21)                                       # r=5: 150 > 100 -> one uf padded to 8
    assert [c.kind for c in p2.chunks] == ["uf", "uf", "uf"] and p2.chunks[-1] == bp.Chunk("uf", 16, 5, 8) and p2.padded_units == 3
    assert "padded to 8" in p2.reason and "5 x 30 = 150 > 100" in p2.reason
    p3 = bp.RegimePlan({8: "uf", 1: "b1"}, tail="pieces", piece_cost={1: 30.0, 8: 100.0}, pad_allowed=False).plan(21)
    assert [c.kind for c in p3.chunks].count("b1") == 5 and p3.padded_units == 0       # padded form refused -> pieces regardless of cost
    assert bp.RegimePlan({8: "uf", 1: "b1"}).plan(8).chunks == (bp.Chunk("uf", 0, 8, None),)
    only1 = bp.RegimePlan({1: "b1"}, piece_cost={1: 5.0}).plan(3)          # a single tested size: no tail decision, never a pad
    assert [c.kind for c in only1.chunks] == ["b1"] * 3 and only1.padded_units == 0
    assert bp.RegimePlan({8: "uf", 1: "b1"}).plan(1).chunks == (bp.Chunk("b1", 0, 1, None),)


def test_regime_pieces_without_costs_and_without_size_one():
    p = bp.RegimePlan([8, 1]).plan(10)
    assert [c.kind for c in p.chunks] == ["b8", "b1", "b1"]
    q = bp.RegimePlan([8, 4], tail="pieces").plan(10)      # no size-1 route: 8 + one b4 call holding 2 real rows padded to 4
    assert q.chunks == (bp.Chunk("b8", 0, 8, None), bp.Chunk("b4", 8, 2, 4)) and q.padded_units == 2
    with pytest.raises(bp.BoundRefused):
        bp.RegimePlan([8, 4], tail="pieces", pad_allowed=False).plan(10)


def test_regime_pad_and_stock_tails_and_max_batch():
    p = bp.RegimePlan([64], tail="pad").plan(130)
    assert p.sizes() == [64, 64, 64] and p.chunks[-1] == bp.Chunk("b64", 128, 2, 64) and p.padded_units == 62
    s = bp.RegimePlan({8: "kit"}, tail="stock").plan(11)
    assert s.chunks == (bp.Chunk("kit", 0, 8, None), bp.Chunk("stock", 8, 3, None)) and "stock route" in s.reason
    m = bp.RegimePlan([2048, 1024, 64], max_batch=1984, tail="pad")
    assert m.dropped == [2048] and m.sizes == [1024, 64]
    pm = m.plan(1100)
    assert pm.sizes() == [1024, 64, 64] and pm.chunks[-1].n == 12 and "dropped: [2048]" in pm.reason
    with pytest.raises(ValueError):
        bp.RegimePlan([4096], max_batch=1984)
    with pytest.raises(ValueError):
        bp.RegimePlan([8], tail="nope")
    assert bp.counts(pm) == {"requested": 1100, "calls": 3, "padded_rows": 52, "calls_b1024": 1, "calls_b64": 2}


def test_line_fields_and_line_format():
    p = bp.plan_chunks(37, 16)
    c = bp.ceiling((524288 // 2) * 512)
    assert bp.line_fields(p) == {"batch_plan": p.reason}
    assert bp.line_fields(p, bound=c) == {"batch_plan": p.reason, "bound": "16", "bound_kind": "int32", "bound_basis": c.basis}
    assert bp.line_fields(bound=bp.ceiling(512 * 2114, cap=1024))["bound_kind"] == "certified"
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in bp.line_fields(p, c).items())
    assert bp.line(p) == "batch plan: " + p.reason
    assert bp.line(p, bound=c) == "batch plan: " + p.reason + " | bound=16 int32 (N x 134217728 <= 2^31)"
    assert bp.line() == "batch plan: none"
    with pytest.raises(ValueError):
        bp.line_fields(bound=bp.Ceiling(3, "guess", ""))


def test_every_unit_lands_exactly_once():
    for n in range(0, 70):
        for rp in (bp.RegimePlan({8: "uf", 1: "b1"}, piece_cost={1: 3.0, 8: 10.0}), bp.RegimePlan([16, 4], tail="pad"), bp.RegimePlan([8], tail="stock")):
            p = rp.plan(n)
            covered = [i for c in p.chunks for i in range(c.start, c.start + c.n)]
            assert covered == list(range(n)), (n, p)
        pc = bp.plan_chunks(n, 16)
        assert [i for c in pc.chunks for i in range(c.start, c.start + c.n)] == list(range(n))


def test_source_parses_at_python_3_8_and_imports_stdlib_only():
    src = open(bp.__file__, encoding="utf-8").read()
    ast.parse(src, feature_version=(3, 8))
    tree = ast.parse(src)
    tops = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {(n.module if isinstance(n, ast.ImportFrom) else n.names[0].name) for n in tops}
    assert names <= {"__future__", "typing", "mem"}, names          # opt_core.mem (row_blocks) is standard library only
