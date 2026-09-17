"""CPU tests for the sampler-graph token cap (modes.graph_cap / GRAPH_CAP_LEVERS): exact and fast resolve without `sg` and `hoist` BY NAME
when the run's largest input is above the mode's device-keyed cap; the LEVER lines and the ARMED/ACTIVE token say so; a caller's
PTX_SAMPLER_GRAPH_MAXTOK wins; big has nothing to cap. No GPU, no torch."""
import pytest

from protenix_v1_opt import modes as M, report as R

SIZE = "PROTENIX_V1_BIG_SIZE_N_TOKEN"          # big.size_of_run's caller word: the run's largest item, tokens (else the --input estimate)


def test_cap_constants_and_device_keying(monkeypatch):
    assert M.GRAPH_CAP_LEVERS == ("sg", "hoist", "sampler_prep") and M.CAP_REASON == "above_cap" and M.SAMPLER_GRAPH_MAXTOK_ENV == "PTX_SAMPLER_GRAPH_MAXTOK"
    assert M.SAMPLER_GRAPH_MAXTOK == {"exact": 1536, "fast": 1999} and M.SAMPLER_GRAPH_MAXTOK_SMALL == {"exact": 768, "fast": 999} and M.SAMPLER_GRAPH_MEM_MIB == 64 * 1024
    for mode in ("exact", "fast"):
        assert M.sampler_graph_maxtok(mode, None) == M.SAMPLER_GRAPH_MAXTOK[mode] == M.sampler_graph_maxtok(mode, 81559) == M.sampler_graph_maxtok(mode, 64 * 1024)
        assert M.sampler_graph_maxtok(mode, 40536) == M.SAMPLER_GRAPH_MAXTOK_SMALL[mode] == M.sampler_graph_maxtok(mode, 64 * 1024 - 1)
        assert 1340 <= M.SAMPLER_GRAPH_MAXTOK[mode] <= 2000 and M.SAMPLER_GRAPH_MAXTOK_SMALL[mode] * 2 <= M.SAMPLER_GRAPH_MAXTOK[mode]   # inside the measured-complete range; the small class at most half
    assert M.sampler_graph_maxtok("big", 81559) == 0
    monkeypatch.setattr(M, "device_memory_mib", lambda: 40536)
    assert M.graph_cap("fast", {}) == {"cap": 999, "source": "device", "memory_mib": 40536}
    assert M.graph_cap("exact", {M.SAMPLER_GRAPH_MAXTOK_ENV: "0"}) == {"cap": 0, "source": "environment", "memory_mib": None}
    with pytest.raises(RuntimeError, match="PTX_SAMPLER_GRAPH_MAXTOK='many'"):
        M.graph_cap("fast", {M.SAMPLER_GRAPH_MAXTOK_ENV: "many"})


@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_below_at_and_above_the_cap(monkeypatch, mode):
    monkeypatch.setattr(M, "device_memory_mib", lambda: 81559)
    shipped = M.KIT_MODES[mode].arm; cap = M.SAMPLER_GRAPH_MAXTOK[mode]
    below = M.resolve(mode, environ={SIZE: str(cap)})                                      # at the cap: the arm as shipped, the cap facts recorded
    assert below.arm == shipped and below.capped == () and below.graph_cap == {"cap": cap, "source": "device", "memory_mib": 81559, "n_token": cap, "n_token_source": "environment"}
    unsized = M.resolve(mode, environ={})                                                  # no input to size (check): no cap applied, the device cap still named
    assert unsized.arm == shipped and unsized.capped == () and unsized.graph_cap["n_token"] is None
    above = M.resolve(mode, environ={SIZE: str(cap + 1)})
    want = ("sg", "hoist", "sampler_prep")
    assert above.capped == want and ("dit_attn_exact" in above.levers) == (mode == "exact")   # dit_attn_exact serves the eager sampler too: never capped
    assert all(w not in above.arm.split("+") for w in want) and above.mode_arm == shipped and above.ablated == ()
    assert [w for w in shipped.split("+") if w not in want] == above.arm.split("+")           # the other words keep their order
    assert above.graph_cap["cap"] == cap and above.graph_cap["n_token"] == cap + 1
    small = M.SAMPLER_GRAPH_MAXTOK_SMALL[mode]
    monkeypatch.setattr(M, "device_memory_mib", lambda: 40536)                             # the 40 GB class: the small cap
    assert M.resolve(mode, environ={SIZE: str(small + 1)}).capped == want and M.resolve(mode, environ={SIZE: str(small)}).capped == ()
    assert M.resolve(mode, environ={SIZE: str(small + 1), M.SAMPLER_GRAPH_MAXTOK_ENV: "0"}).capped == ()      # the caller's 0 = no cap
    assert M.resolve(mode, environ={SIZE: "5000", M.SAMPLER_GRAPH_MAXTOK_ENV: "6000"}).capped == ()


def test_big_has_nothing_to_cap(monkeypatch):
    monkeypatch.setattr(M, "device_memory_mib", lambda: 81559)
    res = M.resolve("big", environ={SIZE: "3900"})
    assert res.capped == () and res.graph_cap is None and "sg" not in res.levers and "hoist" not in res.levers


def test_ablation_and_cap_compose(monkeypatch):
    monkeypatch.setattr(M, "device_memory_mib", lambda: 81559)
    res = M.resolve("fast", environ={SIZE: "3000", "MODEL_OPT_LEVERS_OFF": "sg"})          # sg withheld by the switch, hoist by the cap: each named once, by its own reason
    assert res.ablated == ("sg",) and res.capped == ("hoist", "sampler_prep") and "sg" not in res.levers and "hoist" not in res.levers
    res = M.resolve("exact", environ={SIZE: "3000", "MODEL_OPT_LEVERS_OFF": "sg,hoist"})   # both graph levers ablated: nothing left to cap
    assert res.capped == ("sampler_prep",) and "dit_attn_exact" in res.levers      # sampler_prep rides sg: still on the arm, withheld by the cap


def test_lines(monkeypatch):
    rep = {"mode": "exact", "arm": "exact+gblock+xtr+keep_pool+summary_hostidx+dit_attn_exact+lazy_init+template_dedupe+sampler_prep+atom_attn_exact+tmpl_triatt+tmpl_xtr", "levers_capped": ["sg", "hoist"],
           "graph_cap": {"cap": 1536, "source": "device", "memory_mib": 81559, "n_token": 3000, "n_token_source": "estimate"}}
    lines = [l for l in R.lever_lines({}, rep) if " reason=above_cap " in l]
    assert lines == [
        "[protenix-v1-opt] LEVER name=F3.cuda_graph_sampler state=skipped reason=above_cap impl=lib/kit112_src/infopt_graphs origin=kit strategy=F3.cuda_graph_sampler served=0 gated_by=graph_cap:1536 n=3000 lever=sg",
        "[protenix-v1-opt] LEVER name=LOCAL.step_invariant_hoist state=skipped reason=above_cap impl=lib/kit112_src/dit_hoist.py origin=kit strategy=LOCAL.step_invariant_hoist served=0 gated_by=graph_cap:1536 n=3000 lever=hoist",
    ]
    assert R.capped_token(rep) == " capped=sg,hoist@1536" and R.capped_token({"mode": "fast"}) == ""
    assert R.activation_line(dict(rep, cfg={"trimul": "exact", "gblock": True}, levers={})).endswith(" capped=sg,hoist@1536")
    assert R.armed_line(rep).count(" capped=sg,hoist@1536 ") == 1
