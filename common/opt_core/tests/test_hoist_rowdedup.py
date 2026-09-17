"""CPU tests of opt_core.capture.hoist.RowDedup (strategy F7.row_dedup): the per-call outcomes (single / served / trusted / fallback:nonuniform /
bypassed / disabled), the expand-back of tensor trees, verify_every + DedupMismatch / disable-by-name, the LEVER line grammar and the gate.
Needs torch (CPU suffices): the module skips by name without it."""
from __future__ import annotations

import pytest

from opt_core import report
from opt_core.capture import hoist
from opt_core.capture.hoist import DedupMismatch, RowDedup, RowsNotUniform

torch = pytest.importorskip("torch", reason="RowDedup tensor tests need torch (CPU)")


def _rep(n=4, N=8, c=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    row = torch.randn(1, N, c, generator=g)
    return row.repeat(n, 1, 1)                       # n identical rows along dim 0


def _lin(c=16, o=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(o, c, generator=g)
    return lambda x: x @ W.t()                       # row-independent along dim 0


def test_import_surface_is_stdlib_only_and_names_exist():
    assert RowDedup.IMPL == "row_dedup" and RowDedup.CLASSES == ("exact", "tolerance")


def test_klass_declares_why_verified_is_zero():
    T = RowDedup("sdedup", klass="tolerance")
    T.apply(_lin(), _rep(n=4), 0)
    assert T.stats()["klass"] == "tolerance" and " class=tolerance served=1 " in T.evidence_line("kit-opt") and " verified=0 " in T.evidence_line("kit-opt")
    with pytest.raises(ValueError):
        RowDedup("sdedup", klass="tolerance", verify_every=1)         # check demands bit-exact equality: exact-class sites only
    with pytest.raises(ValueError):
        RowDedup("x", klass="fastish")
    assert issubclass(RowsNotUniform, RuntimeError) and issubclass(DedupMismatch, RuntimeError)


def test_served_uniform_rows_equal_the_full_statement_and_count():
    D, fn, x = RowDedup("cond"), _lin(), _rep(n=4)
    y = D.apply(fn, x, 0)
    assert y.shape == (4, 8, 8) and torch.equal(y, fn(x))
    s = D.stats()
    assert (s["calls"], s["served"], s["fallback"], s["single"], s["rows_saved"], s["state"]) == (1, 1, 0, 0, 3, "active")
    assert y[1].data_ptr() == y[0].data_ptr()                       # an expanded view: every row aliases row 0 (zero new bytes)
    with pytest.raises(RuntimeError):
        y.add_(1.0)                                                 # expanded view: torch refuses the aliasing write (loud)


def test_materialize_returns_a_writable_copy():
    D, fn, x = RowDedup("cond", materialize=True), _lin(), _rep(n=3)
    y = D.apply(fn, x, 0)
    y.add_(0.0)
    assert torch.equal(y, fn(x)) and y.is_contiguous()


def test_nonuniform_rows_run_the_full_statement_and_are_counted_by_name():
    D, fn, x = RowDedup("cond"), _lin(), _rep(n=4)
    x[2, 0, 0] += 1.0
    y = D.apply(fn, x, 0)
    assert torch.equal(y, fn(x))
    s = D.stats()
    assert s["fallback"] == 1 and s["fallback_by"] == {"nonuniform": 1} and s["served"] == 0 and s["state"] == "fallback"
    assert D.gate(expect_fallback=("nonuniform",), max_fallback=None) == []
    assert D.gate() == ["cond: undeclared fallback nonuniform=1"]
    assert D.gate(expect_fallback=("nonuniform",), max_fallback=0) == ["cond: fallback nonuniform=1 above the expected 0"]


def test_strict_uniform_raises_instead_of_falling_back():
    D, fn, x = RowDedup("cond", strict_uniform=True), _lin(), _rep(n=2)
    x[1] *= 2
    with pytest.raises(RowsNotUniform):
        D.apply(fn, x, 0)


def test_nan_rows_are_never_uniform():
    D, fn, x = RowDedup("cond"), _lin(), _rep(n=2)
    x[:, 0, 0] = float("nan")
    D.apply(fn, x, 0)
    assert D.stats()["fallback_by"] == {"nonuniform": 1}


def test_single_row_and_trusted_paths():
    D, fn = RowDedup("cond"), _lin()
    D.apply(fn, _rep(n=1), 0)
    x = _rep(n=5)
    y = D.apply(fn, x, 0, trusted=True)
    s = D.stats()
    assert s["single"] == 1 and s["served"] == 1 and s["trusted"] == 1 and torch.equal(y, fn(x))


def test_negative_dim_tuple_inputs_and_tree_outputs():
    D = RowDedup("pair")
    a, b = _rep(n=3, N=4, c=6), _rep(n=3, N=4, c=2, seed=7)
    fn = lambda xs: {"s": xs[0] * 2, "t": (xs[1] + 1, xs[1].sum(-1, keepdim=True)), "flag": 3}
    y = D.apply(fn, (a, b), -3)
    ref = fn((a, b))
    assert torch.equal(y["s"], ref["s"]) and torch.equal(y["t"][0], ref["t"][0]) and torch.equal(y["t"][1], ref["t"][1]) and y["flag"] == 3
    with pytest.raises(ValueError):
        D.apply(fn, (a, b[:2]), 0)                                  # row counts differ: adapter bug, raised
    with pytest.raises(TypeError):
        D.apply(fn, (a, "notatensor"), 0)


def test_out_dim_when_fn_moves_the_row_axis():
    D, x = RowDedup("perm"), _rep(n=4, N=3, c=5)
    fn = lambda t: t.permute(1, 0, 2)                               # rows move from dim 0 to dim 1
    y = D.apply(fn, x, 0, out_dim=1)
    assert torch.equal(y, fn(x))


def test_fn_not_row_independent_is_a_mismatch_by_shape():
    D, x = RowDedup("bad"), _rep(n=4)
    with pytest.raises(DedupMismatch):
        D.apply(lambda t: t.sum(0, keepdim=False).unsqueeze(0).repeat(2, 1, 1), x, 0)   # 2 rows out for 1 row in


def test_verify_every_passes_on_a_row_independent_statement():
    D, fn, x = RowDedup("cond", verify_every=1), _lin(), _rep(n=4)
    for _ in range(3):
        D.apply(fn, x, 0)
    s = D.stats()
    assert (s["served"], s["verified"], s["mismatched"]) == (3, 3, 0)


def test_verify_mismatch_raises_under_strict_and_disables_by_name_otherwise():
    x = _rep(n=4)
    with pytest.raises(DedupMismatch):                              # a statement whose value depends on the row COUNT: uniform rows hide it, check catches it
        RowDedup("x", verify_every=1, strict=True).apply(lambda t: t * 0 + t.shape[0], x, 0)
    D = RowDedup("x", verify_every=1, strict=False)
    y = D.apply(lambda t: t * 0 + t.shape[0], x, 0)
    assert torch.equal(y, x * 0 + 4)                                # the full-row value is returned on a non-strict mismatch
    s = D.stats()
    assert s["mismatched"] == 1 and s["state"].startswith("disabled:mismatch")
    D.apply(lambda t: t * 0 + t.shape[0], x, 0)
    assert D.stats()["disabled_calls"] == 1
    line = D.evidence_line("kit-opt")
    assert line.startswith("[kit-opt] LEVER name=x state=skipped reason=mismatch:") and " impl=row_dedup origin=core strategy=F7.row_dedup " in line
    assert D.gate(expect_fallback=("nonuniform",)) and D.partial()


def test_bypassed_inside_a_capture_unless_trusted():
    D, fn, x = RowDedup("cond", verify_every=1), _lin(), _rep(n=4)
    hoist.set_capturing(True)
    try:
        y = D.apply(fn, x, 0)                                       # untrusted: the uniformity read is a host sync -> bypassed, statement inline
        assert torch.equal(y, fn(x)) and D.stats()["bypassed"] == 1 and D.stats()["served"] == 0
        z = D.apply(fn, x, 0, trusted=True)                         # trusted: no read -> SERVED inside the capture (narrow / fn / expand are capturable)
        s = D.stats()
        assert torch.equal(z, fn(x)) and (s["served"], s["trusted"], s["bypassed"], s["verified"]) == (1, 1, 1, 0)   # check never fires inside a capture
    finally:
        hoist.set_capturing(False)
    D.apply(fn, x, 0, trusted=True)
    assert D.stats()["verified"] == 1                               # outside the capture the same site checks again


def test_evidence_line_grammar_and_strategy_word():
    D = RowDedup("cond_dedupe")
    D.apply(_lin(), _rep(n=4), 0)
    line = D.evidence_line("sampler-opt")
    assert line.startswith("[sampler-opt] LEVER name=cond_dedupe state=on impl=row_dedup origin=core strategy=F7.row_dedup dedup=active class=exact served=1 trusted=0 single=0 fallback=0 fallback_by=none ")
    assert " rows_saved=3 " in line + " "
    from opt_core import strategies
    assert strategies.check("F7.row_dedup") == "F7.row_dedup"
    named = D.evidence_line("kit", name="F7.row_dedup", strategy=None)
    assert named.startswith("[kit] LEVER name=F7.row_dedup state=on impl=row_dedup origin=core dedup=active class=exact ")
    assert D.gate(require_served=True) == [] and D.partial() is None
    E = RowDedup("idle")
    assert E.gate(require_served=True) == ["idle: requested but never served (calls=0 single=0 bypassed=0 fallback=0)"]
