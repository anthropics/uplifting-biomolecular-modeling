"""The small-input floor (opendde_opt/smalln.py): the composition rule (below / above / mixed / no query / off), the refusal of a malformed
variable, the composed line (switches absent and unset, rows off, the FPF enable and the XL directory gone), the LEVER / ACTIVE words
(`state=off reason=below_gate:<tokens>/<gate>`, `floor=below_gate:<tokens>/<gate>`), and the lines the floor leaves alone (big)."""
import json
import os

import pytest

from opendde_opt import chunklift, big, modes, registry, report, smalln, stack
from opendde_opt.tests.test_activation import fresh  # noqa: F401  (stub upstream first on sys.path, clean switches, fresh activation state)


def _jobs(*sizes, ligand=False):
    """Upstream query items of the given residue counts (one protein chain each; `ligand` adds a CCD ligand entity to every item)."""
    out = []
    for i, n in enumerate(sizes):
        seqs = [{"proteinChain": {"sequence": "A" * n, "count": 1}}]
        if ligand:
            seqs.append({"ligand": {"ligand": "CCD_ATP", "count": 1}})
        out.append({"name": f"item{i}", "sequences": seqs})
    return out


def _tokens(*sizes, **kw):
    return big.count_tokens(_jobs(*sizes, **kw))


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.delenv(smalln.GATE, raising=False)
    smalln._reset(); chunklift._reset()
    yield
    smalln._reset(); chunklift._reset()


# ---------------------------------------------------------------------------------------------------------------- the rule
def test_default_floor_is_arms_design_gate_and_documented(clean):
    assert smalln.FLOOR_DEFAULT == "300" == registry.KNOBS[smalln.GATE][0] == registry.KNOBS["ODDE_ARM_T_MIN_TOKENS"][0]
    assert smalln.gate_value({}) == 300 and smalln.gate_value({smalln.GATE: "0"}) == 0 and smalln.gate_value({smalln.GATE: " 250 "}) == 250
    for lv in smalln.FLOOR_LEVERS:                                               # every floor lever is a registry lever with a composition rule, carried by a floor line
        assert lv in registry.LEVERS and lv in modes.LEVER_SWITCHES
        assert any(lv in modes.LINES[n].levers for n in modes.SMALL_LINES), lv
    assert set(modes.SMALL_LINES) == {modes.MODE_LINES["exact"], modes.MODE_LINES["fast"], modes.MODE_LINES["big"]}   # + the single-card big line (below its offload gate = fast's resident set)


def test_policy_below_above_mixed_unknown_off(clean):
    below = smalln.policy(_tokens(200, 200))
    assert below["below"] is True and below["case"] == "below" and below["max_tokens"] == 200 and below["gate"] == 300 and below["n_items"] == 2
    at = smalln.policy(_tokens(300))                                             # the floor value itself is at-or-above: bound (nothing changes at >= the gate)
    assert at["below"] is False and at["case"] == "above"
    above = smalln.policy(_tokens(400, 1000))
    assert above["below"] is False and above["case"] == "above" and above["max_tokens"] == 1000
    mixed = smalln.policy(_tokens(200, 400))                                     # one item above: today's composition, every lever bound (ARM's per-design gate serves the small item)
    assert mixed["below"] is False and mixed["case"] == "mixed" and "1 of 2 items below" in mixed["reason"]
    unknown = smalln.policy(None)                                                # no query read (`check`, the env route): the static line
    assert unknown["below"] is False and unknown["case"] == "unknown" and unknown["max_tokens"] is None
    off = smalln.policy(_tokens(9), {smalln.GATE: "0"})                          # 0 = the floor off
    assert off["below"] is False and off["case"] == "off" and off["gate"] == 0
    card = smalln.policy(_tokens(200), {smalln.GATE: "150"})                     # the card's own value decides
    assert card["below"] is False and card["gate"] == 150
    lig = smalln.policy(_tokens(120, ligand=True))                               # residue tokens are the unit: a ligand entity is named, not counted
    assert lig["below"] is True and "1 ligand/ion entity uncounted" in lig["reason"]


def test_malformed_variable_refused_by_name(clean):
    for bad in ("2.5k", "-1", "three hundred", "300.0"):
        with pytest.raises(smalln.SmallFloorRefusal) as ei:
            smalln.policy(_tokens(200), {smalln.GATE: bad})
        assert smalln.GATE in str(ei.value) and "non-negative integer" in str(ei.value)
    assert issubclass(smalln.SmallFloorRefusal, modes.OpenModeError)            # the activation's named-refusal class: NOT ACTIVE reason=..., never a traceback


# ---------------------------------------------------------------------------------------------------------------- the composed line
def test_line_without_composes_levers_out():
    s1 = modes.LINES["S1"]
    assert modes.line_without(s1, ()) is s1 and modes.line_without(s1, ("arm_u",)) is s1          # nothing of this line's to drop: the static row itself
    out = modes.line_without(s1, smalln.FLOOR_LEVERS)                          # exact below the floor: the trunk levers out, the sampler levers (DITFAST) kept
    assert out.name == "S1" and out.tier == "exact" and out.hook == s1.hook and out.allocator is None
    assert out.levers == ("served_levers_hook", "cueq_tuned_cache", "dit_hoist", "dit_align", "drop_bond_mask", "lnstream", "tmpl_dedup", "keep_pool", "sched_host", "structok_sync", "json_oneshot", "prefetch", "zprep_hoist", "alloc_auto", "stepgraph", "dit_attn_exact")   # + the SAMPLER exact kernel (bound at every size)
    for k in ("ODDE_ARM_Z", "FPF_ENGINE", "FPF_OPS", "FPF_IMPL", "ODDE_TRIMUL", "ODDE_XL", "ODDE_XL_STRICT"):
        assert k not in out.exports and k in out.unset, k
    assert out.exports == {"ODDE_SERVED_LEVERS": "1", "CUEQ_TRITON_CACHE_DIR": s1.exports["CUEQ_TRITON_CACHE_DIR"], "ODDE_ADDON_LEVERS": "dit_hoist,dit_align", "ODDE_SERVED_LEVERS_STRICT": "1", "ODDE_DIT_ATTN": "exact"}   # + the SAMPLER exact word (0.2.40 / 0.2.57; bound below the floor)
    assert out.fpf is None and modes.XL not in out.path_order and out.path_order == modes.KIT_PATH_ORDER
    fast = modes.line_without(modes.LINES["LSTAR2A"], smalln.FLOOR_LEVERS)       # fast below the floor: the ARM U arm and the provider rows out; DITFAST, the step graph, the sampler-autocast word and (0.2.40) the SAMPLER stack kept
    assert fast.levers == ("served_levers_hook", "cueq_tuned_cache", "dit_hoist", "dit_align", "drop_bond_mask", "lnstream", "tmpl_dedup", "keep_pool", "sched_host", "structok_sync", "json_oneshot", "prefetch", "zprep_hoist", "alloc_auto", "chunk_lift", "stepgraph") + modes._SP_FAST_LEVERS and fast.tier == "tier2"   # fast keeps its chunk lever (tolerance class); exact carries none
    assert out.levers == tuple(x for x in fast.levers if x not in ("chunk_lift",) + modes._SP_FAST_LEVERS) + modes._SP_EXACT_LEVERS   # fast keeps its SAMPLER stack below the floor; exact its bit-exact kernel
    for k in ("ODDE_ARM_U", "ODDE_ARM_U2_TRIMUL", "ODDE_ARM_T_SCOPE", "ODDE_ARM_Z"):
        assert k not in fast.exports and k in fast.unset, k
    sp = modes.SAMPLER_SWITCHES + ("ODDE_DIT_ATTN",)                          # the unit's switches + the DiT-attention word it shares with the ACCEL pool
    assert {k: v for k, v in fast.exports.items() if k not in sp} == {k: v for k, v in out.exports.items() if k not in sp} and fast.path_order == out.path_order and fast.fpf is None   # below the floor exact and fast are one trunk composition; they differ by fast's chunk lever, sampler-autocast word and their SAMPLER levers (0.2.40)
    assert {k: fast.exports[k] for k in sp if k in fast.exports} == modes._SP_FAST_EXPORTS and out.exports["ODDE_DIT_ATTN"] == "exact"
    assert set(smalln.FLOOR_LEVERS) | {"chunk_lift", "lnstream", "stepgraph"} <= set(modes.LEVER_SWITCHES)   # one composition rule per floor lever + the house levers' rules (+ the ablation rules of every other lever)
    from opendde_opt import registry
    norule = [n for n in registry.LEVERS if n not in modes.LEVER_SWITCHES and n not in modes.LEVER_NOT_SHEDDABLE and any(n in ln.levers for ln in modes.LINES.values())]
    assert norule == [], norule                                                     # every lever a line carries: a rule, or a named reason it cannot be shed alone   # one composition rule per floor lever + the chunk lever's ceiling rule + the LayerNorm-stream lever's ablation rule, no other
    with pytest.raises(ValueError, match="rowpair_tp cannot be left out alone"):
        modes.line_without(modes.LINES["BIG_TP"], ("rowpair_tp",))              # a lever without an individual rule is refused with its reason, never silently kept


def test_resolve_composes_at_the_plan(clean):
    tree = os.environ.get("MODEL_OPT") or os.getcwd()
    static = modes.resolve("exact", tree, {})
    assert static.line == modes.line_without(modes.LINES["S1"], ("chunk_lift",))   # no plan: the static row but for the chunk lever (its ceiling composes it out without a query)
    res = modes.Resolution(mode="exact", line=modes.LINES["S1"], tree=tree)
    pol = smalln.plan(res, None)                                                 # a floor line, no query: unknown -> static
    assert pol["case"] == "unknown" and modes.resolve("exact", tree, {}).line == modes.line_without(modes.LINES["S1"], ("chunk_lift",))   # the chunk lever: no ceiling plan in this process -> composed out (chunklift.compose)
    smalln._ST["plan"] = {**smalln.policy(_tokens(200, 180)), "tokens": None}
    below = modes.resolve("exact", tree, {})
    assert "arm_z" not in below.levers and "fpf_trimul_exact" not in below.levers and "ODDE_ARM_Z" not in below.exports and below.line.fpf is None
    assert smalln.word() == "below_gate:200/300" and smalln.gated_off(below.line) == {lv: "below_gate:200/300" for lv in ("arm_z", "fpf_trimul_exact", "triattn_exact", "triattn_conf", "trimul_exact", "transition_exact")}
    assert "dit_hoist" in below.levers and below.exports["ODDE_ADDON_LEVERS"] == "dit_hoist,dit_align"
    fbelow = modes.resolve("fast", tree, {})
    assert set(smalln.gated_off(fbelow.line)) == {"arm_u", "arm_u23", "triattn_core", "triattn_conf", "trimul_core", "ln_core", "transition_core"} and "ODDE_ARM_U" not in fbelow.exports and "ODDE_TRIATTN_CONF" not in fbelow.exports and fbelow.exports["ODDE_DIT_ATTN"] == modes.SAMPLER_FAST_WORD and "ODDE_TRIATTN" not in fbelow.exports and "ODDE_TRIMUL" not in fbelow.exports   # 0.2.40: the SAMPLER stack stays bound below the floor
    smalln._ST["plan"] = {**smalln.policy(_tokens(200, 400)), "tokens": None}   # mixed: the static row
    assert modes.resolve("exact", tree, {}).line == modes.line_without(modes.LINES["S1"], ("chunk_lift",)) and smalln.word() is None   # the chunk lever: no ceiling plan in this process -> composed out (chunklift.compose)
    with pytest.raises(smalln.SmallFloorRefusal):
        modes.resolve("exact", tree, {smalln.GATE: "abc"})                       # a malformed variable refuses on every resolution, plan or none


def test_big_lines_keep_their_own_gates(clean, monkeypatch):
    tree = os.environ.get("MODEL_OPT") or os.getcwd()
    for k in list(os.environ):
        if k.startswith(("OPENDDE_BIG_", "MODEL_OPT_BIG_")):
            monkeypatch.delenv(k)
    big._reset()
    res = modes.Resolution(mode="big", line=modes.LINES[modes.BIG_TP_LINE], tree=tree)
    assert smalln.plan(res, None) == {} and smalln.planned() is None             # the row-sharded line is not a floor line: no decision, nothing composed
    smalln._ST["plan"] = {**smalln.policy(_tokens(200)), "tokens": None}        # even with a below plan in the process, BIG_TP is the adapter's composition
    assert smalln.compose(modes.LINES[modes.BIG_TP_LINE]) is modes.LINES[modes.BIG_TP_LINE] and smalln.gated_off(modes.LINES[modes.BIG_TP_LINE]) == {}
    below = smalln.compose(modes.LINES["BIG_F"])                               # the single-card big line IS a floor line (0.2.35): below the floor its trunk levers leave like fast's
    assert not set(below.levers) & set(smalln.FLOOR_LEVERS) and set(modes.LINES["BIG_F"].levers) & set(smalln.FLOOR_LEVERS)
    assert set(smalln.gated_off(below)) == set(modes.LINES["BIG_F"].levers) & set(smalln.FLOOR_LEVERS)
    res_static = modes.resolve("big", tree)                                     # no big plan in this process: the static row carries the offload unit's levers, and a line that runs
    assert set(res_static.line.levers) & set(modes._OFFLOAD_LEVERS)              # the unit keeps the adapter's own gates (modes.resolve floors BIG_F only below its offload gate)
    assert set(res_static.line.levers) & set(smalln.FLOOR_LEVERS)
    off = modes.Resolution(mode="off", line=None, tree=tree)
    assert smalln.plan(off, None) == {} and smalln.gated_off(None) == {}


# ---------------------------------------------------------------------------------------------------------------- the printed words
def test_lever_and_active_words_below_the_floor(fresh, clean, tmp_path):
    q = tmp_path / "q.json"
    q.write_text(json.dumps(_jobs(200, 200)))
    res = modes.Resolution(mode="exact", line=modes.LINES["S1"], tree=os.environ["MODEL_OPT"])
    pol = smalln.plan(res, str(q))
    assert pol["below"] and pol["tokens"] == {"item0": {"residue_tokens": 200, "ligands_uncounted": 0}, "item1": {"residue_tokens": 200, "ligands_uncounted": 0}}
    dry = stack.activate("exact", dry_run=True)
    assert dry["levers_planned"] == ["served_levers_hook", "cueq_tuned_cache", "dit_hoist", "dit_align", "drop_bond_mask", "lnstream", "tmpl_dedup", "keep_pool", "sched_host", "structok_sync", "json_oneshot", "prefetch", "zprep_hoist", "alloc_auto", "stepgraph", "dit_attn_exact"]   # chunk_lift: no ceiling plan in this process -> composed out; + the SAMPLER exact kernel
    assert dry["levers_gated_off"] == {lv: "below_gate:200/300" for lv in ("arm_z", "fpf_trimul_exact", "triattn_exact", "triattn_conf", "trimul_exact", "transition_exact")}   # the exact line carries no chunk lever (tolerance class: fast / big)
    assert dry["small_input_floor"]["case"] == "below" and "ODDE_ARM_Z" not in dry["exports"] and "ODDE_ARM_Z" in dry["unset"]
    line = report.activation_line(dry)
    assert line.startswith("[opendde-opt] DRY RUN mode=exact line=S1(hook=ACCEL; ODDE_SERVED_LEVERS=1 CUEQ_TRITON_CACHE_DIR=$MODEL_OPT/opt/forward/fast_inference/levers/KIT/cueq_cache_shipped "
                           "ODDE_ADDON_LEVERS=dit_hoist,dit_align ODDE_SERVED_LEVERS_STRICT=1 ODDE_DIT_ATTN=exact) levers=served_levers_hook,cueq_tuned_cache,dit_hoist,dit_align,drop_bond_mask,lnstream,tmpl_dedup,keep_pool,sched_host,structok_sync,json_oneshot,prefetch,zprep_hoist,alloc_auto,stepgraph,dit_attn_exact floor=below_gate:200/300 n_gpu=1 sharding=none "), line
    rows = {ln.split("name=")[1].split()[0]: ln for ln in report.lever_lines(dry, stats={})}
    for lv in ("arm_z", "fpf_trimul_exact"):
        assert rows[lv].startswith(f"[opendde-opt] LEVER name={lv} state=off reason=below_gate:200/300 impl="), rows[lv]
    assert " state=off " in rows["arm_u"] and "reason=" not in rows["arm_u"]                        # a lever off this mode's line for no floor reason: off, no reason (as before)
    stack._APPLIED.clear(); stack._REPORT = None
    rep = stack.activate("exact")                                                # the real activation on the stub upstream: the composed line armed, the floor's switches absent
    assert rep["active"], rep.get("reason")
    assert os.environ.get("ODDE_ARM_Z") is None and os.environ.get("FPF_ENGINE") is None and os.environ["ODDE_SERVED_LEVERS"] == "1"
    assert " floor=below_gate:200/300 " in report.activation_line(rep) and report.floor_word(rep) == "below_gate:200/300"


def test_no_query_and_above_print_the_static_words(fresh, clean, tmp_path):
    dry = stack.activate("exact", dry_run=True)                                  # `check`: no query -> the line as shipped, no floor token
    assert "arm_z" in dry["levers_planned"] and dry["levers_gated_off"] == {} and dry["small_input_floor"] is None   # no query on the exact line: nothing composes out (no chunk lever there)
    assert " floor=" not in report.activation_line(dry)
    q = tmp_path / "q.json"
    q.write_text(json.dumps(_jobs(200, 700)))                                    # mixed sizes: bound
    smalln.plan(modes.Resolution(mode="fast", line=modes.LINES["LSTAR2A"], tree=os.environ["MODEL_OPT"]), str(q))
    stack._APPLIED.clear(); stack._REPORT = None
    dry = stack.activate("fast", dry_run=True)
    assert "arm_u" in dry["levers_planned"] and dry["levers_gated_off"] == {"chunk_lift": "above_gate:unknown/1024"} and dry["small_input_floor"]["case"] == "mixed"   # the chunk lever: no ceiling plan in this process -> composed out (chunklift.compose)
    assert " floor=" not in report.activation_line(dry) and all("below_gate" not in ln for ln in report.lever_lines(dry, stats={}))


def test_malformed_variable_is_not_active_by_name(fresh, clean, monkeypatch):
    monkeypatch.setenv(smalln.GATE, "small")
    dry = stack.activate("exact", dry_run=True)                                  # `check`: resolved to a named reason, no line (the verb exits 3 on it)
    assert not dry.get("active") and dry["reason"].startswith(f"{smalln.GATE}='small'") and "line" not in dry and dry["levers_planned"] == []
    monkeypatch.setattr(stack, "_REPORT", None)
    rep = stack.activate("fast")                                                 # the applying route: NOT ACTIVE by name, nothing exported
    assert not rep.get("active") and os.environ.get("ODDE_SERVED_LEVERS") is None
    assert report.activation_line(rep).startswith("[opendde-opt] NOT ACTIVE mode=fast reason=MODEL_OPT_SMALL_INPUT_FLOOR_TOKENS='small': a non-negative integer")
    monkeypatch.setattr(stack, "_REPORT", None)
    with pytest.raises(stack.ActivationError):
        stack.activate("fast", strict=True)
    monkeypatch.setattr(stack, "_REPORT", None)
    off = stack.activate("off", dry_run=True)                                    # the stock route reads no floor
    assert smalln.GATE not in (off.get("reason") or "")
