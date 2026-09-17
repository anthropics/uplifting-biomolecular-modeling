"""The pair TriMul lever (c_z = c_hidden = 128) binds the shared core's TriMul provider by TIER word only (kit 0.2.38): `exact` under --mode exact, `fast` under
--mode fast, `big` under --mode big (levers_ptx1.TRIMUL_TIER / trimul_tier); the row the provider selects per key is served through its one face
(triangle_multiplication(selection=...)); a key it refuses BY NAME keeps upstream's own statement, counted `stock:trimul:<refusal>` (accounted `aside`, never
partial).  CLASS contracts only: which row a card / stack / size gets is the core's table, never asserted here by name."""
import os
import sys

import pytest

from protenix_v1_opt import kit as K, report as R


def _levers():
    pytest.importorskip("torch", reason="levers_ptx1 imports torch")
    sys.path.insert(0, os.path.dirname(K.levers_file(None)))
    try:
        import levers_ptx1 as L
    except Exception as e:
        pytest.skip(f"levers_ptx1 not importable: {e!r}")
    finally:
        sys.path.pop(0)
    return L


def test_the_tier_words_per_mode(monkeypatch):
    L = _levers()
    assert L.TRIMUL_TIER == {"exact": "exact", "fast": "fast"}
    for word, mode, tier in (("exact", "exact", "exact"), ("fast", "fast", "fast"), ("fast", "big", "big"), ("fast", None, "fast"), ("exact", "big", "exact")):
        monkeypatch.setitem(L.CFG, "trimul", word)
        monkeypatch.setattr(L, "kit_mode", lambda m=mode: m)
        assert L.trimul_tier() == tier, (word, mode)
    monkeypatch.setitem(L.CFG, "trimul", "stock")
    assert L.trimul_tier() is None
    src = open(K.levers_file(None)).read()
    for gone in ("TRIMUL_GATE_TOKENS", "TRIMUL_KIT_WORD", "TRIMUL_FAST_BIND", "PTX_TRIMUL_WORD", "fpf_trimul_v4", "fpf_trimul.kernels", "_seed_v4_cell"):
        assert gone not in src, gone                                                  # no row word, no size floor, no kit cell table, no kit-side kernel statement


def test_the_tier_words_resolve_or_refuse_by_name_on_both_cards():
    """CLASS contract of the provider (pure stdlib select): every tier word at the trunk's keys names SOME row of the provider (an exact-class row or the library
    row under `exact`) or refuses BY NAME — never an unknown word; the row itself is the core's table."""
    T = pytest.importorskip("opt_core.kernels.trimul", reason="the shared core's TriMul provider")
    exact_ok = set(getattr(T, "EXACT_ROWS", ())) | {"cueq", "torch_math", "tmk3_exact"}
    for cc in ("9.0", "8.0"):
        for tier in ("exact", "fast", "big"):
            for dt, kw in (("bf16", {}), ("fp32", {"tf32": True}), ("fp32", {})):
                for N in (61, 400, 800, 1200, 2048):
                    for d in ("outgoing", "incoming"):
                        try:
                            sel = T.select(cc, dt, 128, 128, N, d, word=tier, has_cueq=True, **kw)
                        except T.Refusal as r:
                            assert r.kind, (cc, tier, dt, N, d)                            # refused BY NAME: upstream's statement serves, accounted
                            continue
                        assert sel.row in T.ROW_NAMES or sel.row in ("cueq", "torch_math"), sel.row
                        if tier == "exact":
                            assert sel.row in exact_ok, (cc, dt, N, d, sel.row)            # the exact tier never names a tolerance-class kernel row


def _acct(word, core=None, sel=(), trimul_counts=None, bind="tier:fast"):
    counts = {"trimul": dict(trimul_counts if trimul_counts is not None else {word: 1000})}
    if core is not None:
        counts["core:trimul_" + word] = dict(core)
    if sel:
        counts["coresel:trimul_" + word] = {s: 1 for s in sel}
    return {"cfg": {"trimul": word}, "gates": {}, "counts": counts, "trimul_" + word: {"bind": bind, "stack": "H100:x/y/z"}}


def test_the_lever_line_names_the_rows_served_and_the_tier_binding():
    ev = R.kit_evidence(_acct("fast", {"native": 1000}, sel=("fast:bf16:C128:N800:out=>native@c,x2",)), "fast", [], [{"name": "x", "N_token": 800}], 300)["fast"]
    assert ev["served"] == 1000 and ev["row"] == "native" and "served_by" not in ev and ev["facts"] == ["bind=tier:fast"]
    line = [l for l in R.lever_lines({"fast": ev}, {"mode": "fast", "mode_trimul": "fast", "mode_levers": []}) if l.endswith(" lever=fast")][0]
    assert " state=on " in line and " row=native " in line and " bind=tier:fast " in line and "selections=fast:bf16:C128:N800:out=>native@c,x2" in line
    # two rows over the run's keys (bf16 trunk / tf32 confidence head): the majority row leads, served_by names both
    ev = R.kit_evidence(_acct("exact", {"native_exact": 900, "cueq": 100}, bind="tier:exact"), "exact", [], [{"name": "x", "N_token": 800}], 300)["exact"]
    assert ev["row"] == "native_exact" and ev["served_by"] == {"cueq": 100, "native_exact": 900} and ev["facts"] == ["bind=tier:exact"]
    ev = R.kit_evidence(_acct("fast", {"v4": 1000}, bind="tier:big"), "fast", [], [{"name": "x", "N_token": 800}], 300)["fast"]
    assert ev["row"] == "v4" and ev["facts"] == ["bind=tier:big"]


def test_a_key_refused_by_name_keeps_the_stock_statement_accounted_never_partial():
    # every call refused by name at the tier word (a card / dtype / width without a row): upstream's statement by design -> aside, exit 0
    acct = _acct("fast", core={}, trimul_counts={"stock:trimul:cc_unsupported": 1000})
    ev = R.kit_evidence(acct, "fast", [], [{"name": "x", "N_token": 800}], 300)["fast"]
    assert ev["served"] == 0 and ev["gated"] == {"stock:trimul:cc_unsupported": 1000} and not ev["fallback"] and ev["aside"] == {"state": "aside", "word": "stock:trimul:cc_unsupported"}
    line = [l for l in R.lever_lines({"fast": ev}, {"mode": "fast", "mode_trimul": "fast", "mode_levers": []}) if l.endswith(" lever=fast")][0]
    assert " state=skipped reason=aside " in line, line
    assert R.partial_of({"fast": ev}) == ([], None)
    # some calls served, some refused by name (e.g. one dtype): served, the refusals ride gated
    ev = R.kit_evidence(_acct("fast", core={"v4": 700}, trimul_counts={"fast": 700, "stock:trimul:dtype": 300}), "fast", [], [{"name": "x", "N_token": 800}], 300)["fast"]
    assert ev["served"] == 700 and ev["gated"] == {"stock:trimul:dtype": 300} and "aside" not in ev
    # a provider row's exception: upstream's statement answered that call -> read as partial (fallback)
    ev = R.kit_evidence(_acct("exact", core={"native_exact": 990}, trimul_counts={"exact": 990, "error:RuntimeError": 10}, bind="tier:exact"), "exact", [], [{"name": "x", "N_token": 800}], 300)["exact"]
    assert ev["fallback"] == {"error:RuntimeError": 10} and R.partial_of({"exact": ev})[0] == ["exact"]
