"""MODEL_OPT_LEVERS_OFF (opendde_opt/ablate.py): levers left out of the resolved line by name through modes.line_without; refusals by name."""
import os

import pytest

from opendde_opt import ablate, modes, report, stack
from opendde_opt.tests.test_activation import fresh  # noqa: F401


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.delenv(ablate.ENV, raising=False)
    ablate._reset()
    yield
    ablate._reset()


def test_unset_changes_nothing(clean):
    s1 = modes.LINES["S1"]
    assert ablate.requested({}) == () and ablate.compose(s1, {}) is s1 and ablate.word() is None and ablate.gated_off(s1) == {}


def test_named_levers_are_composed_out_by_the_kits_own_rule(clean):
    s1 = modes.LINES["S1"]
    out = ablate.compose(s1, {ablate.ENV: "lnstream, fpf_trimul_exact,lnstream"})
    assert out is not s1 and out.name == "S1" and "lnstream" not in out.levers and "fpf_trimul_exact" not in out.levers
    assert not (set(modes.LEVER_SWITCHES["fpf_trimul_exact"]) & set(out.exports)) and set(modes.LEVER_SWITCHES["fpf_trimul_exact"]) <= set(out.unset)   # the lever's switches out (line_without's rule)
    assert [x for x in s1.levers if x not in ("lnstream", "fpf_trimul_exact", "stepgraph", "trimul_exact")] == list(out.levers)     # everything else as shipped, in order (the step graph goes with lnstream, the TriMul provider cell with the FPF adapter: modes.LEVER_SWITCHES)
    assert set(ablate.dropped()) == {"lnstream", "fpf_trimul_exact", "stepgraph", "trimul_exact"} and set(ablate.word().split(",")) == set(ablate.dropped())
    assert ablate.gated_off(out) == {lv: "ablated" for lv in ("lnstream", "fpf_trimul_exact", "stepgraph", "trimul_exact")}
    fast = ablate.compose(modes.LINES["LSTAR2A"], {ablate.ENV: "dit_attn_apb"})                         # 0.2.40: the SAMPLER attention kernel; its dependents (fp16 word, fused token stack + its lowp word) leave with it
    assert not {"dit_attn_apb", "dit_attn_fp16", "dit_fused", "dit_lowp"} & set(fast.levers) and "ODDE_DIT_ATTN" not in fast.exports and "arm_u" in fast.levers and "atom_attn_apb" in fast.levers


def test_refusals_by_name(clean):
    s1 = modes.LINES["S1"]
    with pytest.raises(modes.OpenModeError, match="unknown lever"):
        ablate.compose(s1, {ablate.ENV: "nonesuch"})
    with pytest.raises(modes.OpenModeError, match="does not carry arm_u"):
        ablate.compose(s1, {ablate.ENV: "arm_u"})                               # the exact line has no ARM U to leave out
    tp = modes.LINES["BIG_TP"]
    with pytest.raises(modes.OpenModeError, match="rowpair_tp cannot be left out of line BIG_TP alone: it is the --n_gpu P>1 line itself"):
        ablate.compose(tp, {ablate.ENV: "rowpair_tp"})                          # no individual rule: the reason is named
    with pytest.raises(modes.OpenModeError, match="mode off"):
        ablate.compose(None, {ablate.ENV: "lnstream"})
    with pytest.raises(modes.OpenModeError, match="is not a lever name"):
        ablate.requested({ablate.ENV: "lnstream;rm"})


def test_a_lever_a_size_gate_already_left_out_is_accepted(clean):
    below = modes.line_without(modes.LINES["S1"], ("arm_z",))                   # what the small-input floor hands over below 300 tokens
    out = ablate.compose(below, {ablate.ENV: "arm_z,lnstream"})
    assert "arm_z" not in out.levers and "lnstream" not in out.levers and ablate.dropped() == ("arm_z", "lnstream", "stepgraph")


def test_resolution_and_the_census_words(fresh, clean, monkeypatch):  # noqa: F811
    monkeypatch.setenv(ablate.ENV, "lnstream,chunk_lift")
    dry = stack.activate("fast", dry_run=True)                                   # the fast line carries both (chunk_lift is tolerance class: not on exact)
    assert "lnstream" not in dry["levers_planned"] and "chunk_lift" not in dry["levers_planned"] and dry["ablated"] == ["lnstream", "chunk_lift", "stepgraph"]
    assert dry["levers_gated_off"]["lnstream"] == "ablated"
    line = report.activation_line(dry)
    assert " ablated=lnstream,chunk_lift,stepgraph " in line and "levers=" in line, line
    rows = {ln.split("name=")[1].split()[0]: ln for ln in report.lever_lines(dry, stats={})}
    assert rows["lnstream"].startswith("[opendde-opt] LEVER name=lnstream state=off reason=ablated "), rows["lnstream"]
    monkeypatch.setenv(ablate.ENV, "bogus")
    monkeypatch.setattr(stack, "_REPORT", None)
    ref = stack.activate("exact", dry_run=True, strict=False)
    assert not ref.get("active") and not ref.get("dry_run_ok", False) and "unknown lever(s) bogus" in (ref.get("reason") or ""), ref.get("reason")


def test_every_lever_has_a_rule_and_the_rules_compose(clean):
    s1, fast, f, tp = (modes.LINES[n] for n in ("S1", "LSTAR2A", "BIG_F", "BIG_TP"))
    out = ablate.compose(s1, {ablate.ENV: "dit_hoist"})                          # a value switch: the token out of ODDE_ADDON_LEVERS, dit_align goes with it (LEVER_DEPENDENTS)
    assert "dit_hoist" not in out.levers and "dit_align" not in out.levers and "ODDE_ADDON_LEVERS" not in out.exports
    assert ablate.dropped() == ("dit_hoist", "dit_align")
    ablate._reset()
    out = ablate.compose(s1, {ablate.ENV: "dit_align"})
    assert out.exports["ODDE_ADDON_LEVERS"] == "dit_hoist" and "dit_hoist" in out.levers and "dit_align" not in out.levers
    ablate._reset()
    out = ablate.compose(fast, {ablate.ENV: "served_levers_hook"})                # the installer: every lever it installs goes with it
    assert not {"served_levers_hook", "dit_hoist", "dit_align", "arm_u", "arm_u23", "dit_attn_bf16"} & set(out.levers)
    assert not {"ODDE_SERVED_LEVERS", "ODDE_ADDON_LEVERS", "ODDE_ARM_U", "ODDE_DIT_ATTN"} & set(out.exports) and "lnstream" in out.levers
    ablate._reset()
    for name in ("drop_bond_mask", "alloc_auto", "cueq_tuned_cache", "lnstream", "stepgraph", "fpf_trimul_exact", "arm_z"):
        o = ablate.compose(s1, {ablate.ENV: name}); ablate._reset()
        gone = 4 if name == "arm_z" else 2 if name in ("lnstream", "fpf_trimul_exact") else 1   # lnstream takes the step graph along, arm_z its provider bindings triattn_exact + triattn_conf + transition_exact, fpf_trimul_exact trimul_exact (modes.LEVER_DEPENDENTS)
        assert name not in o.levers and len(o.levers) == len(s1.levers) - gone, name
    o = ablate.compose(fast, {ablate.ENV: "chunk_lift"}); ablate._reset()             # the chunk lever rides fast, not exact (tolerance class)
    assert "chunk_lift" not in o.levers and len(o.levers) == len(fast.levers) - 1 and "chunk_lift" not in s1.levers
    assert "CUEQ_TRITON_CACHE_DIR" not in ablate.compose(s1, {ablate.ENV: "cueq_tuned_cache"}).exports; ablate._reset()
    o = ablate.compose(tp, {ablate.ENV: "struct_pair_bf16,tp_triatt"})           # BIG_TP levers turned off by value
    assert o.exports["ODDE_TP_STRUCT_PAIR_DTYPE"] == "fp32" and o.exports["ROWPAIR_TRIATT_CORE"] == "torch" and "rowpair_tp" in o.levers
    ablate._reset()
    o = ablate.compose(f, {ablate.ENV: "sample_chunk"})                          # a big memory lever: the line rebuilt without it, the unit still on
    assert "sample_chunk" not in o.levers and o.exports.get("ODDE_OFFLOAD") == "all" and "pair_offload_struct" in o.levers
    ablate._reset()
    o = ablate.compose(f, {ablate.ENV: "no_dit_hoist"})                          # the hoist's removal left out: the hoist rides again
    assert "dit_hoist" in o.levers and "no_dit_hoist" not in o.levers
    ablate._reset()
    o = ablate.compose(f, {ablate.ENV: "pair_offload_conf"})                     # one offload stage out: the unit keeps the others
    assert o.exports["ODDE_OFFLOAD"] == "trunk,struct" and "pair_offload_conf" not in o.levers and "pair_offload_struct" in o.levers
    ablate._reset()
    o = ablate.compose(f, {ablate.ENV: "pair_offload_trunk,pair_offload_struct,pair_offload_conf"})   # every stage out: the unit off the line (fast's resident set)
    assert "ODDE_OFFLOAD" not in o.exports and not [x for x in o.levers if x.startswith("pair_offload")] and "stepgraph" in o.levers and "chunk_lift" in o.levers   # (the unit off = the resident set: chunk_lift, keep_pool and the sampler step graph ride it — 0.2.65; 0.2.46-0.2.64 dropped stepgraph from every big line by name)
    ablate._reset()
    o = ablate.compose(f, {ablate.ENV: "diffz"})
    assert o.exports["ODDE_OFFLOAD_DIFFZ"] == "0" and "diffz" not in o.levers and "bigln_guard" not in o.levers
