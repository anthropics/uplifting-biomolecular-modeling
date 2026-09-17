"""The `sync_hoist` cell: the MSA row-index memo reproduces the engine's `_subsample_all_msa` draw for draw (same generator consumption, same
selected rows) on every branch (enough valid rows / too few with invalid rows to draw from / none invalid), the served statements are the
engine's (source guard), and the part list parses by name. CPU; the engine comparisons are skipped where the engine is not importable."""
import pytest

from openfold3_opt.cells import sync_hoist


def test_parts_and_switch_values_are_named():
    assert sync_hoist.parts({}) == sync_hoist.PARTS_ALL == ("msa_depth", "msa_rows", "templ")
    assert sync_hoist.parts({sync_hoist.ENV_PARTS: "all"}) == sync_hoist.PARTS_ALL
    assert sync_hoist.parts({sync_hoist.ENV_PARTS: "none"}) == ()
    assert sync_hoist.parts({sync_hoist.ENV_PARTS: "templ, msa_rows"}) == ("msa_rows", "templ")          # canonical order
    with pytest.raises(ValueError):
        sync_hoist.parts({sync_hoist.ENV_PARTS: "msa_dept"})
    assert sync_hoist.requested({}) is False and sync_hoist.requested({sync_hoist.ENV: "1"}) is True
    with pytest.raises(ValueError):
        sync_hoist.requested({sync_hoist.ENV: "yes"})
    with pytest.raises(ValueError):
        sync_hoist.requested({sync_hoist.ENV: "1", sync_hoist.ENV_PARTS: "bogus"})                       # a mistyped part list is refused with the switch
    line = sync_hoist.census_line()
    assert line.startswith(sync_hoist.PREFIX + " LEVER name=sync_hoist state=") and " fallback=" in line


def test_row_memo_reproduces_the_engines_subsample_draw_for_draw():
    torch = pytest.importorskip("torch")
    IE = pytest.importorskip("openfold3.core.model.feature_embedders.input_embedders")
    stock = IE.MSAModuleEmbedder._subsample_all_msa
    g = torch.Generator().manual_seed(0)
    n_msa, n_tok = 300, 20
    feat = torch.randn(n_msa, n_tok, 34, generator=g)
    mask = (torch.rand(n_msa, n_tok, generator=g) > 0.3).float()
    mask[250:] = 0.0                                                                                     # 50 padded (invalid) rows
    for n_sub in (100, 280, 300):                                                                        # valid >= n; valid < n with invalid rows; every row
        torch.manual_seed(7)
        ref = [stock(feat, mask, n_sub) for _ in range(4)]                                                # four recycle passes, the engine's statements
        torch.manual_seed(7)
        memo, got, events = {}, [], []
        for _ in range(4):
            f, m, ev = sync_hoist.subsample_rows(feat, mask, n_sub, memo)
            got.append((f, m)); events.append(ev)
        assert events == ["fill", "hit", "hit", "hit"], (n_sub, events)
        assert all(torch.equal(r[0], o[0]) and torch.equal(r[1], o[1]) for r, o in zip(ref, got)), n_sub
        f, m, ev = sync_hoist.subsample_rows(feat, mask, n_sub, None)                                   # no memo: the engine's statements, named
        assert ev == "nomemo" and f.shape == ref[0][0].shape
    other = mask.clone()                                                                                 # a different mask tensor never hits another's memo
    torch.manual_seed(7); memo = {}
    sync_hoist.subsample_rows(feat, mask, 100, memo)
    assert sync_hoist.subsample_rows(feat, other, 100, memo)[2] == "fill"


def test_the_served_statements_are_the_engines():
    pytest.importorskip("torch")
    IE = pytest.importorskip("openfold3.core.model.feature_embedders.input_embedders")
    assert sync_hoist._source_missing(IE.MSAModuleEmbedder.forward, sync_hoist.SOURCE_FORWARD) == []
    assert sync_hoist._source_missing(IE.MSAModuleEmbedder._subsample_all_msa, sync_hoist.SOURCE_SUBSAMPLE) == []


def test_template_verdict_is_read_once_from_the_addons_census_and_served_after():
    """The instance-level template-stack forward: the first call of a trunk scope runs the class forward (the add-on's wrap decides by value and
    counts a hit or a miss); identical -> the later calls hand template 0 to the class forward and expand its output xT (the add-on's own
    statements); distinct -> the later calls run the class forward unexamined, counted `fallback:templ_distinct`; no add-on census -> every
    call is the class forward's, counted `fallback:addon_stats_absent`."""
    torch = pytest.importorskip("torch")
    import sys
    import types
    stats = {"templ_distinct_hits": 0, "templ_distinct_miss": 0}
    addon = types.ModuleType(sync_hoist.ADDON_MODULE); addon.STATS = stats
    calls = []

    class Stack:                                                                                         # stands for TemplatePairStack with the add-on's class-level wrap
        def forward(self, t, mask=None):
            calls.append(tuple(t.shape))
            if t.shape[-4] > 1:                                                                          # the add-on's by-value verdict
                if bool((t == t.narrow(-4, 0, 1)).all()):
                    stats["templ_distinct_hits"] += 1
                    return (t.narrow(-4, 0, 1) * 2.0).expand_as(t)
                stats["templ_distinct_miss"] += 1
            return t * 2.0

    prev_mod, prev_state, prev_parts = sys.modules.get(sync_hoist.ADDON_MODULE), dict(sync_hoist.STATE), sync_hoist.STATE["parts"]
    sys.modules[sync_hoist.ADDON_MODULE] = addon
    try:
        sync_hoist.STATE.update(state="on", parts=sync_hoist.PARTS_ALL)
        for identical in (True, False):
            inst = Stack(); inst.forward = sync_hoist._make_tps_forward(inst)
            t = torch.ones(1, 4, 8, 8, 3) if identical else torch.arange(4.0).view(1, 4, 1, 1, 1).expand(1, 4, 8, 8, 3).contiguous()
            sync_hoist._SCOPE.update(gen=1, templ={}); sync_hoist.STATE["fallback"].clear(); calls.clear()
            outs = [inst.forward(t, None) for _ in range(4)]                                             # four recycle passes
            assert all(torch.equal(o, t * 2.0) for o in outs)
            if identical:
                assert calls == [(1, 4, 8, 8, 3), (1, 1, 8, 8, 3), (1, 1, 8, 8, 3), (1, 1, 8, 8, 3)] and sync_hoist.STATE["fallback"] == {}
            else:
                assert calls == [(1, 4, 8, 8, 3)] * 4 and sync_hoist.STATE["fallback"] == {"templ_distinct": 3}
        sync_hoist._SCOPE.update(gen=None, templ={})                                                    # outside a trunk scope: the class forward, untouched
        calls.clear(); inst.forward(torch.ones(1, 4, 2, 2, 3), None); assert calls == [(1, 4, 2, 2, 3)]
        del sys.modules[sync_hoist.ADDON_MODULE]                                                        # no add-on census: never served, named
        sync_hoist._SCOPE.update(gen=2, templ={}); sync_hoist.STATE["fallback"].clear()
        inst.forward(torch.ones(1, 4, 2, 2, 3), None); assert sync_hoist.STATE["fallback"] == {"addon_stats_absent": 1}
    finally:
        sync_hoist._SCOPE.update(gen=None, templ={})
        sync_hoist.STATE.clear(); sync_hoist.STATE.update(prev_state); sync_hoist.STATE["parts"] = prev_parts
        if prev_mod is not None:
            sys.modules[sync_hoist.ADDON_MODULE] = prev_mod
        else:
            sys.modules.pop(sync_hoist.ADDON_MODULE, None)
