"""The memory row's sampler group under a per-card token ceiling (CHANGES 0.3.13: where fast's whole pass fits the card the memory row is
fast's composition — the roll-out, the DiT hoist and their rider ride big up to modes.BIG_SAMPLER_CEILINGS[card] tokens, chosen by number from
fast's measured pass peaks; above it they step aside BY NAME per prediction and the stock eager loop with the fused step serves, 0.3.8's composition).
Class contracts only: the ceiling is a number per card-memory floor, stated on the row for the probed card, absent = as composed, below every floor =
the group off by name; the sampler modules read the word and count the calls above it by name; the xP line is unchanged."""
import os
import re

from .. import modes, registry, sampler, report


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT_OPT = os.path.dirname(HERE)


def _src(rel):
    with open(os.path.join(KIT_OPT, rel)) as fh:
        return fh.read()


def test_ceiling_table_is_by_number_per_card_memory_floor():
    floors = dict(modes.BIG_SAMPLER_CEILINGS)
    assert set(floors) == {79_000, 39_000} and all(isinstance(t, int) and t >= 400 for t in floors.values()) and floors[79_000] >= floors[39_000], \
        "an 80 GB and a 40 GB ceiling, tokens by number, the larger card's at least the smaller's"
    assert modes.sampler_ceiling(None) is None and modes.sampler_ceiling(81559) == floors[79_000] == modes.sampler_ceiling(81920) == modes.sampler_ceiling(183359) \
        and modes.sampler_ceiling(40960) == floors[39_000] and modes.sampler_ceiling(24564) == 0, "H100 80GB / A100-80GB / a larger card -> the 80 GB ceiling; A100-40GB -> its own; a 24 GB card -> none"


def test_the_memory_row_never_carries_the_sampler_graph_group_and_the_ceiling_table_is_inert():
    """0.3.17 (QoL rule 3 as written: no CUDA graphs in the memory mode): the roll-out's whole-loop graph leaves big BY NAME at every size on every
    card (`off=rollout:rule:big_no_graphs`), its Kabsch rider and the DiT hoist with it (`needs:rollout`); their words leave the row; the fused
    bf16 step keeps its eager host (`weights=sample`); the per-card ceiling table stays in place (the number is still stated for the probed card)
    but binds nothing; a card below every floor names the group off by its memory floor first (card_off), as before; fast carries no ceiling."""
    try:
        for cc, mem in ((None, None), ("9.0", 81559), ("8.0", 81920), ("8.0", 40960), ("8.9", 24564)):
            modes.set_card(cc, mem) if cc else modes.set_card(None)
            r = modes.resolve("big")
            assert not (set(modes.BIG_SAMPLER_GROUP) & set(r["levers"])) and "dit_fused" in r["levers"] and r["env"]["BOLTZ_SAMPLER_DIT"].endswith("weights=sample"), (cc, mem)
            assert not any(k in r["env"] for k in ("BOLTZ_SAMPLER_ROLLOUT", "BOLTZ_DIT_HOIST", "BOLTZ_SAMPLER_ALIGN", modes.SAMPLER_MAX_TOKENS_ENV)), (cc, mem)
            why = {l: (r.get("off") or {}).get(l) or (r.get("card_off") or {}).get(l) for l in modes.BIG_SAMPLER_GROUP}
            if mem == 24564:
                assert all(w == "card_memory_below:39000MiB" for w in why.values()) and r["sampler_max_tokens"] == 0, why      # below every floor: the card names them off first
            else:
                assert why == {"rollout": "rule:big_no_graphs", "dit_hoist": "needs:rollout", "align_jacobi64": "needs:rollout"}, why
                assert r["sampler_max_tokens"] == (modes.sampler_ceiling(mem) if mem else None), (cc, mem)                     # the table still states the number; no word carries it
        modes.set_card("9.0", 81559)
        assert {"rollout", "dit_hoist"} <= set(modes.resolve("fast")["levers"]) and "sampler_max_tokens" not in modes.resolve("fast") and modes.SAMPLER_MAX_TOKENS_ENV not in modes.resolve("fast")["env"], "fast keeps the roll-out and carries no ceiling"
    finally:
        modes.set_card(None)


def test_the_xP_line_is_unchanged_by_name():
    try:
        modes.set_card("9.0", 81559); modes.set_n_gpu(2)
        r2 = modes.resolve("big")
        assert not (set(modes.BIG_SAMPLER_GROUP) & set(r2["levers"])) and all(r2["tp_off"][l].startswith("xP_line_unchanged:") for l in ("rollout", "dit_hoist")) and r2["tp_off"]["align_jacobi64"] \
            and not any(k in r2["env"] for k in ("BOLTZ_SAMPLER_ROLLOUT", "BOLTZ_DIT_HOIST", "BOLTZ_SAMPLER_ALIGN", "BOLTZ_SAMPLER_DIT", modes.SAMPLER_MAX_TOKENS_ENV)) and "sampler_max_tokens" not in r2
    finally:
        modes.set_n_gpu(1); modes.set_card(None)


def test_registry_word_and_adapter_vocabulary():
    assert registry.LEVERS["rollout"]["words"] == (modes.SAMPLER_MAX_TOKENS_ENV,) == (sampler.SWITCH_MAX_TOKENS,) and "above_max_tokens" in sampler.EXPECTED_SCOPE
    assert sampler.words({"BOLTZ_SAMPLER_MAX_TOKENS": "1400"}) == {"BOLTZ_SAMPLER_MAX_TOKENS": "1400"} and not sampler.requested({"BOLTZ_SAMPLER_MAX_TOKENS": "1400"}) \
        and sampler.requested({"BOLTZ_SAMPLER_ROLLOUT": "graph", "BOLTZ_SAMPLER_MAX_TOKENS": "card"}), "the ceiling word alone requests nothing"


def test_the_sampler_modules_read_the_word_by_source_contract():
    bz = _src("forward/sampler/src/bz_sampler.py"); dh = _src("forward/dit_hoist/src/boltz_dit_hoist.py"); mv = _src("forward/dit_hoist/src/make_worker_variant.py")
    i_scope, i_prep, i_inner = bz.index('why = "above_max_tokens"'), bz.index("dit.prepare("), bz.index("return inner(")
    assert 'def install(rollout="graph", kabsch="torch", dit=None, aligncap_module=None, max_tokens=None)' in bz and i_scope < i_prep < i_inner and 'STATS["token_gated"] += 1' in bz, \
        "the roll-out names a call above the ceiling out of scope BEFORE the fused step is prepared, so the stock eager loop it hands the call to runs the fused step"
    assert 'MAX_TOKENS_ENV = "BOLTZ_SAMPLER_MAX_TOKENS"' in dh and re.search(r"if mt and n_tokens > mt:\s*#[^\n]*\n\s*STATS\[\"token_gated\"\] \+= 1", dh) and dh.index("if mt and n_tokens > mt:") < dh.index("gate = headroom_gate(self, network_condition_kwargs, multiplicity)"), \
        "the hoist opens no cache generation for a prediction above the ceiling (decided before the headroom gate, resident state of an earlier prediction freed first)"
    assert '"token_gated", "max_tokens")' in mv, "the worker variant copies the per-item token-gate census (hoist_token_gated / hoist_max_tokens) the evidence reads"


def test_lever_lines_name_the_ceiling_and_the_step_aside():
    sr = {"applied": ["rollout", "align_jacobi64", "dit_fused"], "gate": {"ok": True},
          "levers": {"rollout": {"state": "on", "word": "graph", "max_tokens": 1400}, "align_jacobi64": {"state": "on", "word": "jacobi64"}, "dit_fused": {"state": "on", "word": "bf16"}},
          "module": {"stats": {"calls": 5, "samples": 0, "replays": 0, "captures": 0, "capture_s": [], "scope": {"above_max_tokens": 5}, "token_gated": 5, "kabsch": "device"},
                     "dit": {"stats": {"served": 3000, "calls": 3000, "weights": "sample"}}}}
    ev = {"sampler_report": sr, "hoist_level": "2", "n_items": 5, "token_gated": 5, "headroom_gated": 0, "sampler_max_tokens": 1400}
    assert report.lever_state("rollout", ev, {"mode": "big"})[:2] == ("skipped", "above_big_tokens:1400") == report.lever_state("dit_hoist", ev, {"mode": "big"})[:2] == report.lever_state("align_jacobi64", ev, {"mode": "big"})[:2]
    st, why, pairs = report.lever_state("dit_fused", ev, {"mode": "big"})
    assert (st, why) == ("on", None) and dict(pairs)["host"] == "eager", "every call above the ceiling: the fused step served from the eager loop"
    sr["module"]["stats"].update(samples=3, captures=3, replays=597, scope={"above_max_tokens": 2}, token_gated=2); ev.update(token_gated=2)
    st, why, pairs = report.lever_state("rollout", ev, {"mode": "big"})
    assert (st, why) == ("on", None) and dict(pairs)["token_gated"] == 2 and dict(pairs)["max_tokens"] == 1400 and dict(report.lever_state("dit_fused", ev, {"mode": "big"})[2])["host"] == "rollout+eager"
