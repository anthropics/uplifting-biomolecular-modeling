"""CPU tests for the tri-attention token CEILING (report.ceiling_state; levers_ptx1.TRIATTN_CEILING_TOKENS = 2048): a run whose every item is
above the ceiling served no fused triangle-attention call BY DESIGN (upstream row-chunks the attention there: `stock:chunk`) — LEVER … state=skipped
reason=above_ceiling ceiling=2048 n=<smallest item>, exit 0, never a partial activation; a run with an item at or below the ceiling and no served
call stays partial."""
from protenix_v1_opt import kit as K, modes as M, report as R


def _acct(res, tri_counts, extra_counts=None):
    counts = {"trimul": {res.trimul: 12}, "triattn": dict(tri_counts), "transition": {("ttr" if "ttr" in res.levers else "xtr") + ":C=128": 40},
              "dit": {"apb:fp16": 48}, "atom": {"apb:tf32rn": 12}}
    counts.update(extra_counts or {})
    return {"cfg": {"trimul": res.trimul, **{lv: True for lv in res.levers}}, "counts": counts,
            "gates": {"trimul_tokens": 100, "transition_rows": 4096, "triattn_ceiling_tokens": 2048},
            "sampler": {"graphs": False, "prep": {"on": False, "aside": "no_sampler_graph"}, "hoist_installed": False}}


def test_ceiling_constant_is_the_kit_files():
    src = open(K.kit_home() + "/ptxfpf/levers_ptx1.py").read()
    assert "TRIATTN_CEILING_TOKENS = 2048" in src and "N > TRIATTN_CEILING_TOKENS" in src and '"triattn_ceiling_tokens": TRIATTN_CEILING_TOKENS' in src


def test_ceiling_state_words():
    c = {"word": "n>2048", "tokens": 2048}
    assert R.ceiling_state(0, c, [{"N_token": 2560}])["state"] == "ceiling-off"
    assert R.ceiling_state(0, c, [{"N_token": 2560}, {"N_token": 3000}]) == {"word": "n>2048", "tokens": 2048, "items_above": 2, "items_atmost": 0, "items_unsized": 0, "n_min": 2560, "state": "ceiling-off"}
    assert R.ceiling_state(0, c, [{"N_token": 2560}, {"N_token": 1200}])["state"] == "partial"        # an item below the ceiling with nothing served: not by design
    assert R.ceiling_state(0, c, [{"N_token": 2560}, {}])["state"] == "partial"                     # an unsized item: not by design
    assert R.ceiling_state(7, c, [{"N_token": 2560}])["state"] == "served"
    assert R.ceiling_state(0, None, [{"N_token": 2560}]) is None and R._kit_gates({"gates": {"triattn_ceiling_tokens": 2048}})["triattn_ceiling"] == c


def test_every_item_above_the_ceiling_is_accounted_exit_0():
    for mode, lv in (("fast", "gflash"), ("exact", "gblock")):
        res = M.resolve(mode, environ={})
        acct = _acct(res, {"stock:chunk": 1240})
        ev = R.kit_evidence(acct, res.trimul, (lv,), [{"N_token": 2560}], 300 if mode == "fast" else None)
        assert ev[lv]["served"] == 0 and ev[lv]["ceiling"]["state"] == "ceiling-off"
        partial, why = R.partial_of({lv: ev[lv]})
        assert partial == [] and why is None
        line = [l for l in R.lever_lines({lv: ev[lv]}, {"mode": mode}) if l.endswith(" lever=" + lv)][0]
        assert " state=skipped reason=above_ceiling " in line and " ceiling=2048 n=2560 " in line + " ", line


def test_an_item_below_the_ceiling_with_nothing_served_stays_partial():
    res = M.resolve("fast", environ={})
    acct = _acct(res, {"stock:chunk": 1240, "stock:gate": 3})
    ev = R.kit_evidence(acct, res.trimul, ("gflash",), [{"N_token": 2560}, {"N_token": 1200}], 300)
    assert ev["gflash"]["ceiling"]["state"] == "partial" and R.partial_of({"gflash": ev["gflash"]})[0] == ["gflash"]
