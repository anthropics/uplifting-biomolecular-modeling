"""The big adapter over opt_core.mem (opendde_opt/big.py): the mode's line and its evidence lines composed on `fast`, a caller's
OPENDDE_BIG_* variable refused by name, the size gates (the offload unit's MODEL_OPT_BIG_OFFLOAD_MIN_TOKENS with sample_chunk
following it, no_dit_hoist's MODEL_OPT_BIG_NO_DIT_HOIST_MIN_TOKENS: grammar, below / at / above, the pre-activation plan, the LEVER
rows' named `below_gate` state), the strict apply through the core's registry, the census unit per prediction (a lever that did not
run is a fallback by name: PARTIAL), sample_chunk's act on the stock config, the record folded into the activation report, the
LEVER-line counters."""
import itertools
import json
import os
import sys
import types

import pytest

import opt_core.mem
from opendde_opt import alloc, big, modes, registry, report as _report
from opendde_opt.tests import _stubs

pytestmark = pytest.mark.skipif(not hasattr(opt_core.mem, "register"),        # named: the adapter needs the core's memory-lever registry (opt_core >= 0.4.0);
                                reason="core_missing: opt_core.mem carries no lever registry (opt_core < 0.4.0) — every big line refuses by name on this core")

F = "BIG_F"
DROPPED = ("alloc_auto", "zprep_hoist", "dit_fused", "dit_lowp")   # fast's levers off every big line at every size (modes.BIG_DROP; zprep_hoist 0.2.47, dit_fused / dit_lowp 0.2.61: measured peak costs); the
                                                                     # sampler step graph `stepgraph` left the tuple at 0.2.65: it rides the resident set below the offload gate and leaves with the offload unit


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("ODDE_", "PYTORCH_CUDA", "OPENDDE_BIG", "MODEL_OPT_BIG")):
            monkeypatch.delenv(k, raising=False)
    big._reset()
    yield
    big._reset()


# ----------------------------------------------------------------------------------------------------------------- composition
def test_big_is_one_mode_on_one_line():
    assert modes.MODE_LINES["big"] == F and modes.line_of("big") is modes.LINES[F]
    assert modes.BIG_LINES == (F, "BIG_TP") and modes.BIG_BASE == modes.MODE_LINES["fast"] == "LSTAR2A"   # + the `--n_gpu P>1` line (tests/test_ngpu.py)
    assert modes.BIG_MEM_LEVERS == {F: ("pair_offload", "no_dit_hoist", "sample_chunk"), "BIG_TP": ("no_dit_hoist", "sample_chunk")}
    for n in modes.BIG_LINES:
        ln = modes.LINES[n]
        assert (ln.tier, ln.allocator, ln.hook) == ("tier2", modes.EXPANDABLE, modes.HOOK)
        assert not set(DROPPED) & set(ln.levers) and DROPPED == modes.BIG_DROP
        assert "ODDE_ADDON_LEVERS" not in ln.exports and "ODDE_ADDON_LEVERS" in ln.unset            # no_dit_hoist on every static row: the addon switch absent AND unset
        assert not {"dit_hoist", "dit_align"} & set(ln.levers) and "no_dit_hoist" in ln.levers and "drop_bond_mask" in ln.levers
        assert not set(ln.exports) & set(ln.unset) and all(x in registry.LEVERS for x in ln.levers)


def test_the_lines_are_fast_minus_alloc_auto_plus_their_memory_levers():
    fast = modes.LINES["LSTAR2A"]                                                       # the fast mode's line (modes.BIG_BASE)
    base = [x for x in fast.levers if x not in DROPPED + ("dit_hoist", "dit_align", "chunk_lift", "stepgraph", "keep_pool", "sampler_amp", "tmpl_dedup")]   # the rows no_dit_hoist / pair_offload turn off (modes.BIG_KIT_ROWS_OFF)
    f = modes.LINES[F]
    assert f.path_order == modes.OFFLOAD_PATH_ORDER and f.path_order[0] == modes.HOOK
    assert list(f.levers) == [x for x in base if not x.startswith("xl_")] + list(modes._OFFLOAD_LEVERS) + ["no_dit_hoist", "sample_chunk"]
    assert {k: v for k, v in f.exports.items() if k.startswith("ODDE_OFFLOAD")} == modes._OFFLOAD_EXPORTS
    assert f.exports["ODDE_ARM_U"] == "1" and f.exports["ODDE_DIT_ATTN"] == modes.SAMPLER_BIG_WORD and f.exports["ODDE_ATOM_ATTN"] == modes.SAMPLER_BIG_WORD and "ODDE_DIT_FUSED" not in f.exports and "ODDE_DIT_LOWP" not in f.exports and {"ODDE_DIT_FUSED", "ODDE_DIT_LOWP"} <= set(f.unset) and "ODDE_XL" not in f.exports      # fast's arm and (0.2.40) fast's SAMPLER stack on the offload path; the XL unit off it


def test_no_flag_composes_the_static_rows():
    for n in modes.BIG_LINES:
        assert big.compose(n, {}) is modes.LINES[n]
        assert modes.resolve(n, _stubs.TREE, {}).line is modes.LINES[n]
    assert modes.resolve("big", _stubs.TREE, {}).line is modes.LINES[F]
    with pytest.raises(ValueError, match="not a big line"):
        big.compose("BIG", {})


def _plan(n, line=F, offload_gate=1400, hoist_gate=2565):
    """A plan as big.plan records it for a query whose largest item counts `n` residue tokens (the shipped gates unless given)."""
    def pol(lever, gate):
        on = n >= gate
        return {"gate_var": big.SIZE_GATES[lever][0], "gate": gate, "max_tokens": n, "on": on, "source": "policy",
                "reason": f"largest item {n} residue tokens {'>=' if on else '<'} gate {gate}: the lever {'on' if on else 'off'}"}
    gates = {"pair_offload": offload_gate, "no_dit_hoist": hoist_gate}
    return {"line": line, "tokens": {"item0": {"residue_tokens": n, "ligands_uncounted": 0}},
            "policies": {lv: pol(lv, gates[lv]) for lv in modes.BIG_MEM_LEVERS[line] if lv in big.SIZE_GATES}}


def test_the_size_gates_are_the_packages_own_switches(monkeypatch):
    """Between the gates (plan decided the offload unit is in, the hoist stays in) the line is composed without no_dit_hoist through the
    core's selection: the DITFAST hoist rows and their switch come back, the NOTE names it; at / above every gate (or no plan) the static
    row stands; below the offload gate the unit AND sample_chunk leave the line with the hoist's removal: fast's resident lever set."""
    monkeypatch.setitem(big._ST, "plan", _plan(1500))
    assert big.switches(F) == {"no_dit_hoist": False} and set(big.gated_off(F)) == {"no_dit_hoist"}
    ln = big.compose(F, {})
    assert {"dit_hoist", "dit_align"} <= set(ln.levers) and ln.exports.get("ODDE_ADDON_LEVERS") == "dit_hoist,dit_align" and "no_dit_hoist" not in ln.levers
    assert ln.exports["ODDE_OFFLOAD"] == "all" and "pair_offload_struct" in ln.levers and "sample_chunk" in ln.levers and ln.path_order == modes.OFFLOAD_PATH_ORDER
    res = modes.resolve("big", _stubs.TREE, {})
    assert any(f"{F} composed on fast (LSTAR2A) at its flags: levers=['pair_offload', 'sample_chunk'] off_by_flag=['no_dit_hoist'] on_by_flag=[]" in n for n in res.notes)
    monkeypatch.setitem(big._ST, "plan", _plan(2956))
    assert big.switches(F) == {} and big.compose(F, {}) is modes.LINES[F]
    monkeypatch.setitem(big._ST, "plan", _plan(1400))                              # AT the offload gate: the unit is installed (>=), today's line byte for byte
    assert big.switches(F) == {"no_dit_hoist": False} and big.compose(F, {}).exports["ODDE_OFFLOAD"] == "all"
    monkeypatch.setitem(big._ST, "plan", None)
    assert big.compose(F, {}) is modes.LINES[F]                                    # no plan (the env route, a dry run): the static row, every lever on
    # below the offload gate: the offload unit is not installed, sample_chunk follows it, the hoist stays — fast's resident path on the fixed allocator
    monkeypatch.setitem(big._ST, "plan", _plan(1012))
    assert big.switches(F) == {"pair_offload": False, "sample_chunk": False, "no_dit_hoist": False}
    assert big.gated_off(F)["sample_chunk"] is big.gated_off(F)["pair_offload"]   # the follower carries the policy it follows
    ln = big.compose(F, {})
    fast = modes.LINES[modes.BIG_BASE]
    assert not any(k.startswith("ODDE_OFFLOAD") for k in ln.exports) and "ODDE_OFFLOAD" in ln.unset
    assert ln.path_order == modes.KIT_PATH_ORDER == fast.path_order and ln.allocator == modes.EXPANDABLE and ln.name == F
    assert list(ln.levers) == [x for x in fast.levers if x not in DROPPED] and not set(ln.levers) & (set(modes._OFFLOAD_LEVERS) | {"no_dit_hoist", "sample_chunk"})
    assert "chunk_lift" in ln.levers and "chunk_lift" not in modes.LINES[F].levers                  # fast's chunk lever rides the resident set only: the offload unit keeps upstream's clamp
    assert "keep_pool" in ln.levers and "keep_pool" not in modes.LINES[F].levers                    # the keep-pool release policy follows the same gate: bound on the resident set, off with the offload unit
    assert "stepgraph" in ln.levers and "stepgraph" not in modes.LINES[F].levers                    # the sampler step graph too (0.2.65): the resident line replays its own fp32 unfused step from the graph; the offload unit streams the pair tensors the step reads
    assert ln.levers.index("stepgraph") == [x for x in fast.levers if x not in DROPPED].index("stepgraph")   # at fast's own position: the resident set IS fast's lever order minus BIG_DROP
    dropped_sw = {sw for x in DROPPED for sw in modes.LEVER_SWITCHES.get(x, ())}                    # the dropped levers' own switches (ODDE_DIT_FUSED / ODDE_DIT_LOWP, 0.2.61) leave with them and are unset
    assert {k: v for k, v in ln.exports.items() if k != modes.ALLOCATOR} == {k: v for k, v in {**fast.exports, **modes.BIG_WORD_SWITCHES}.items() if k not in dropped_sw} and set(ln.unset) >= set(fast.unset) | dropped_sw   # fast's switches exactly (+ the line's fixed allocator; the TriMul / pair-bias-attention providers bound by their big tier word, 0.2.56 / 0.2.57) less the dropped levers'
    res = modes.resolve("big", _stubs.TREE, {})
    assert res.exports[modes.ALLOCATOR] == modes.EXPANDABLE and modes.kit_dir(_stubs.TREE, modes.OFFLOAD) not in res.sys_path
    assert any("off_by_flag=['pair_offload', 'no_dit_hoist', 'sample_chunk']" in n for n in res.notes)
    # the row-sharded line carries no offload unit: nothing follows an offload gate there (big --n_gpu P unaffected), only the hoist's gate
    monkeypatch.setitem(big._ST, "plan", _plan(1012, line=modes.BIG_TP_LINE))
    assert set(_plan(1012, line=modes.BIG_TP_LINE)["policies"]) == {"no_dit_hoist"}
    assert big.switches(modes.BIG_TP_LINE) == {"no_dit_hoist": False} and "sample_chunk" in big.compose(modes.BIG_TP_LINE, {}).levers
    assert "chunk_lift" not in big.compose(modes.BIG_TP_LINE, {}).levers and "chunk_lift" not in modes.LINES[modes.BIG_TP_LINE].levers   # the ranks run the core's pair stack (modes.BIG_TP_DROP)
    assert "keep_pool" not in big.compose(modes.BIG_TP_LINE, {}).levers and "keep_pool" not in modes.LINES[modes.BIG_TP_LINE].levers      # and off under --n_gpu P>1 (modes.BIG_TP_DROP)


def test_below_a_gate_the_lever_rows_read_off_by_the_gate(monkeypatch):
    """The LEVER row of a registry row whose memory lever the plan left out reads `state=off reason=below_gate:<tokens>/<gate>` — a named,
    non-partial state in the core's offload vocabulary; every other off row carries no gate reason."""
    monkeypatch.setitem(big._ST, "plan", _plan(1012))
    assert big.BELOW_GATE == "below_gate"
    for row in modes._OFFLOAD_LEVERS + ("sample_chunk",):
        assert big.gated_off_reason(row) == "below_gate:1012/1400", row
    assert big.gated_off_reason("no_dit_hoist") == "below_gate:1012/2565"
    assert big.gated_off_reason("dit_hoist") is None and big.gated_off_reason("arm_u") is None and big.gated_off_reason("alloc_auto") is None
    res = modes.resolve("big", _stubs.TREE, {})
    rep = {"active": True, "mode": "big", "line": modes.describe_line(res), "levers_planned": list(res.levers), "levers_applied": list(res.levers)}
    assert _report.lever_state("pair_offload_struct", rep) == ("off", "below_gate:1012/1400")
    assert _report.lever_state("sample_chunk", rep) == ("off", "below_gate:1012/1400") and _report.lever_state("no_dit_hoist", rep) == ("off", "below_gate:1012/2565")
    assert _report.lever_state("dit_hoist", rep) == ("on", None) and _report.lever_state("alloc_auto", rep) == ("off", None)
    fast_rep = dict(rep, mode="fast", line=modes.describe_line(modes.resolve("fast", _stubs.TREE, {})))
    assert _report.lever_state("pair_offload_struct", fast_rep) == ("off", None) and big.gated_off_reason("sample_chunk", fast_rep) is None   # another line's report: no gate word
    lines = {ln.split("name=")[1].split(" ")[0]: ln for ln in _report.lever_lines(rep, {})}
    assert " state=off reason=below_gate:1012/1400 " in lines["pair_offload_trunk"] and " state=off reason=below_gate:1012/1400 " in lines["sample_chunk"]
    assert " state=off reason=below_gate:1012/2565 " in lines["no_dit_hoist"] and " state=on " in lines["dit_hoist"] and " PARTIAL" not in " ".join(lines.values())
    monkeypatch.setitem(big._ST, "plan", _plan(1500))                              # between the gates: the offload rows are on the line, only the hoist's row is off by its gate
    assert big.gated_off_reason("pair_offload_struct") is None and big.gated_off_reason("sample_chunk") is None and big.gated_off_reason("no_dit_hoist") == "below_gate:1500/2565"
    monkeypatch.setitem(big._ST, "plan", None)
    assert big.gated_off_reason("pair_offload_struct") is None and big.gated_off_reason("no_dit_hoist") is None


@pytest.mark.parametrize("word", ["OPENDDE_BIG_NO_DIT_HOIST", "OPENDDE_BIG_PAIR_OFFLOAD", "OPENDDE_BIG_SAMPLE_CHUNK", "OPENDDE_BIG_SAMPLE_CHUNK_SAMPLES",
                                  "OPENDDE_BIG_PAIR_OFFLOAD_PIN", "OPENDDE_BIG_ALLOW_PARTIAL", "OPENDDE_BIG_LINE", "OPENDDE_BIG_FROB"])
def test_a_callers_big_variable_refuses_by_name(word):
    """The mode reads no OPENDDE_BIG_* variable: any of them — a former switch, setting, opt-out or an unknown name — refuses by name."""
    with pytest.raises(big.BigRefusal, match=word):
        big.compose(F, {word: "0"})
    with pytest.raises(modes.OpenModeError, match=word):                            # BigRefusal is an OpenModeError: cli/stack turn it into NOT ACTIVE, exit 3
        modes.resolve("big", _stubs.TREE, {word: "1"})
    for other in ("fast", "exact"):                                                  # the other modes never read that stem either; they do not look at it
        assert modes.resolve(other, _stubs.TREE, {word: "1"}).mode == other


def _query(tmp_path, *lengths, ligand=False):
    items = [{"name": f"item{i}", "sequences": [{"proteinChain": {"sequence": "M" * n, "count": 2}}] + ([{"ligand": {"ligand": "CCD_ATP"}}] if ligand else [])}
             for i, n in enumerate(lengths)]
    p = tmp_path / "q.json"
    p.write_text(json.dumps(items))
    return str(p)


def test_count_tokens_is_residues_times_count():
    jobs = [{"name": "a", "sequences": [{"proteinChain": {"sequence": "MKT", "count": 2}}, {"dnaSequence": {"sequence": "ACGT"}}, {"ligand": {"ligand": "CCD_ATP"}}]}]
    assert big.count_tokens(jobs) == {"a": {"residue_tokens": 10, "ligands_uncounted": 1}}


def test_the_size_gate_policies():
    t = lambda n: {"x": {"residue_tokens": n, "ligands_uncounted": 0}}     # noqa: E731
    assert big.SIZE_GATES == {"pair_offload": (big.OFFLOAD_GATE, "1400"), "no_dit_hoist": (big.NO_HOIST_GATE, "2565")}
    assert big.OFFLOAD_GATE == "MODEL_OPT_BIG_OFFLOAD_MIN_TOKENS" and big.NO_HOIST_GATE == "MODEL_OPT_BIG_NO_DIT_HOIST_MIN_TOKENS"
    assert big.GATE_FOLLOWERS == {"pair_offload": ("sample_chunk",)}
    for lever, (var, default) in big.SIZE_GATES.items():
        assert registry.KNOBS[var][0] == default                                                             # the registry's knob row = the adapter's default = configs/<gpu>.env
        gate = int(default)
        p = big.size_gate_policy(lever, t(1012), {})
        assert (p["gate_var"], p["gate"], p["on"], p["source"]) == (var, gate, False, "policy") and f"1012 residue tokens < gate {gate}" in p["reason"]
        assert big.size_gate_policy(lever, t(gate), {})["on"] is True and big.size_gate_policy(lever, t(gate - 1), {})["on"] is False   # on AT the gate
        assert big.size_gate_policy(lever, t(2956), {})["on"] is True
        assert big.size_gate_policy(lever, None, {})["on"] is True                                        # unknown size: the memory-safe side (the lever on)
        assert big.size_gate_policy(lever, t(1012), {var: "1000"})["on"] is True                          # the card's parameter
        assert big.size_gate_policy(lever, t(9999), {var: "0"})["on"] is True                             # 0 = on at every size
        p = big.size_gate_policy(lever, t(99999), {f"OPENDDE_BIG_{lever.upper()}": "0"})                # a caller's variable does not decide the gate
        assert (p["on"], p["source"]) == (True, "policy")
        for bad in ("2.5k", "-1", "off", " "):
            with pytest.raises(big.BigRefusal, match=var + r"=.*a non-negative integer \(residue tokens; 0 = the lever on at every size\) is required"):
                big.size_gate_policy(lever, t(1), {var: bad})
    with pytest.raises(KeyError):
        big.size_gate_policy("sample_chunk", t(1), {})                                                   # a follower has no gate of its own


def test_the_size_gates_default_by_the_cards_memory(monkeypatch):
    """Unset, a gate is the package's default for the RUNNING card (big.gate_default over device 0's total memory, probed once per process):
    under 64 GiB — the 40 GB A100 — 1160 / 1856, the values configs/a100.env exported on that card until they moved here; on a card of 64 GiB
    and over (H100, A100 80 GB), with no GPU visible, or when the probe fails, 1400 / 2565 (configs/h100.env restates them). A value set in the
    environment wins on every card, as before; empty = unset; a malformed value is refused by name as before."""
    t = lambda n: {"x": {"residue_tokens": n, "ligands_uncounted": 0}}     # noqa: E731
    assert big.SMALL_CARD_MIB == 65536 and big.SMALL_CARD_GATES == {"pair_offload": "1160", "no_dit_hoist": "1856"}
    assert set(big.SMALL_CARD_GATES) == set(big.SIZE_GATES) and all(int(big.SMALL_CARD_GATES[lv]) < int(d) for lv, (_v, d) in big.SIZE_GATES.items())
    large = {lv: int(d) for lv, (_v, d) in big.SIZE_GATES.items()}        # 1400 / 2565
    small = {lv: int(v) for lv, v in big.SMALL_CARD_GATES.items()}       # 1160 / 1856
    for mib, want in ((40960, small), (65535, small), (65536, large), (81559, large), (81920, large), (None, large), (0, large)):   # A100 40 GB | the boundary (< 64 GiB | 64 GiB) | H100 | A100 80 GB | no GPU visible / a failed probe
        monkeypatch.setitem(big._CARD, "memory_mib", mib)
        for lever, (var, _default) in big.SIZE_GATES.items():
            assert big.gate_default(lever) == str(want[lever]), (mib, lever)
            p = big.size_gate_policy(lever, t(1300), {})
            assert (p["gate_var"], p["gate"], p["on"], p["source"]) == (var, want[lever], 1300 >= want[lever], "policy"), (mib, lever)
            assert set(p) == {"gate_var", "gate", "max_tokens", "on", "source", "reason"}                        # the policy record's shape is unchanged (kit.big.<lever>_policy)
            assert big.size_gate_policy(lever, t(1300), {var: "1000"})["gate"] == 1000                         # a value set in the environment wins on every card (configs/h100.env, the caller)
            assert big.size_gate_policy(lever, t(1300), {var: ""})["gate"] == want[lever]                      # empty = unset, as before
            with pytest.raises(big.BigRefusal, match=var):
                big.size_gate_policy(lever, t(1), {var: "2.5k"})
    monkeypatch.setitem(big._CARD, "memory_mib", 40960)                                                       # the 40 GB card: a 1,300-token input is AT/ABOVE its offload gate (1160) and below the hoist's (1856)
    assert big.size_gate_policy("pair_offload", t(1300), {})["on"] is True and big.size_gate_policy("no_dit_hoist", t(1300), {})["on"] is False
    monkeypatch.setitem(big._CARD, "memory_mib", 81559)                                                       # the 80 GB card (H100 reads 81559 MiB): the same input is below both — fast's resident lever set
    assert big.size_gate_policy("pair_offload", t(1300), {})["on"] is False and big.size_gate_policy("no_dit_hoist", t(1300), {})["on"] is False


def test_the_card_is_probed_once_through_the_packages_gpu_reader(monkeypatch, tmp_path):
    """The probe is stack.gpu_info's memory.total (nvidia-smi: no torch import and no CUDA context, so the line's allocator policy still binds at
    activation), read once per process and only when a gate's variable is unset; resolve, the pre-activation plan and the composed line follow it."""
    from opendde_opt import stack
    calls = []

    def a100_40gb():
        calls.append(1)
        return {"name": "NVIDIA A100-SXM4-40GB", "sm": "80", "memory_mib": 40960, "cuda_visible": None}

    def no_reader():
        calls.append(1)
        raise RuntimeError("nvidia-smi not found")

    monkeypatch.setattr(stack, "gpu_info", a100_40gb)
    monkeypatch.setitem(big._CARD, "probed", False)                                                          # a fresh process: nothing probed yet
    res = modes.resolve("big", _stubs.TREE, {big.OFFLOAD_GATE: "1400", big.NO_HOIST_GATE: "2565"})        # both variables set (configs/h100.env): the card is never asked
    assert res.line is modes.LINES[F] and calls == [] and big._CARD["probed"] is False
    res = modes.resolve("big", _stubs.TREE, {})                                                               # unset: the gates are the 40 GB card's — one probe for both, memoised
    assert res.line is modes.LINES[F] and calls == [1] and big.card_memory_mib() == 40960 and calls == [1]
    pl = big.plan(res, _query(tmp_path, 650), {})                                                             # 1,300 residue tokens on the 40 GB card: the offload unit in (>= 1160), the hoist stays (< 1856)
    assert {lv: (p["gate"], p["on"]) for lv, p in pl["policies"].items()} == {"pair_offload": (1160, True), "no_dit_hoist": (1856, False)} and calls == [1]
    ln = big.compose(F, {})
    assert ln.exports["ODDE_OFFLOAD"] == "all" and "sample_chunk" in ln.levers and {"dit_hoist", "dit_align"} <= set(ln.levers) and "no_dit_hoist" not in ln.levers
    assert big.gated_off_reason("no_dit_hoist") == "below_gate:1300/1856" and big.gated_off_reason("pair_offload_struct") is None
    assert {lv: p["gate"] for lv, p in big.plan(res, _query(tmp_path, 650), {big.OFFLOAD_GATE: "1400"})["policies"].items()} == {"pair_offload": 1400, "no_dit_hoist": 1856}   # a set variable wins per gate
    monkeypatch.setattr(stack, "gpu_info", no_reader)                                                           # a fresh process whose probe fails: the 80 GB card's defaults, probed once
    monkeypatch.setitem(big._CARD, "probed", False)
    assert big.card_memory_mib() is None and (big.gate_default("pair_offload"), big.gate_default("no_dit_hoist")) == ("1400", "2565") and calls == [1, 1]


@pytest.mark.parametrize("var", ["MODEL_OPT_BIG_OFFLOAD_MIN_TOKENS", "MODEL_OPT_BIG_NO_DIT_HOIST_MIN_TOKENS"])
def test_a_malformed_gate_refuses_by_name_on_every_route(var):
    """A malformed size gate is a BigRefusal naming the variable at composition — `check`, `warm`, `pred` and the env route all resolve
    the line first — never a traceback and never a silent default; the other modes do not read it."""
    with pytest.raises(big.BigRefusal, match=var):
        big.compose(F, {var: "lots"})
    with pytest.raises(modes.OpenModeError, match=var):                             # BigRefusal is an OpenModeError: cli/stack turn it into NOT ACTIVE, exit 3
        modes.resolve("big", _stubs.TREE, {var: "1.4e3"})
    assert modes.resolve("big", _stubs.TREE, {var: "0"}).line is modes.LINES[F]   # well-formed: the static row (no plan)
    for other in ("fast", "exact"):
        assert modes.resolve(other, _stubs.TREE, {var: "lots"}).mode == other
    if var == big.OFFLOAD_GATE:                                                    # the row-sharded line carries no offload unit: its gate is not read there
        assert big.compose(modes.BIG_TP_LINE, {var: "lots"}) is modes.LINES[modes.BIG_TP_LINE]


def test_plan_decides_the_switches_before_activation(tmp_path):
    res = modes.resolve("big", _stubs.TREE, {})
    pl = big.plan(res, _query(tmp_path, 300, 506), {})                                                  # 1012 tokens: below both gates
    assert pl["line"] == F and set(pl["policies"]) == {"pair_offload", "no_dit_hoist"} and pl["tokens"]["item1"]["residue_tokens"] == 1012
    assert (pl["policies"]["pair_offload"]["on"], pl["policies"]["no_dit_hoist"]["on"]) == (False, False) and pl["policies"]["pair_offload"]["max_tokens"] == 1012
    assert big._ST["plan"] is pl and big.switches(F) == {"pair_offload": False, "sample_chunk": False, "no_dit_hoist": False}
    pl = big.plan(res, _query(tmp_path, 750), {})                                                       # 1500 tokens: the offload unit on, the hoist stays in
    assert (pl["policies"]["pair_offload"]["on"], pl["policies"]["no_dit_hoist"]["on"]) == (True, False) and big.switches(F) == {"no_dit_hoist": False}
    pl = big.plan(res, _query(tmp_path, 1478), {})                                                      # 2956 tokens: every lever on (the static row)
    assert all(p["on"] for p in pl["policies"].values()) and big.switches(F) == {} and all(p["source"] == "policy" for p in pl["policies"].values())
    pl = big.plan(res, None, {})                                                                         # no query (the env route): unknown size, every lever on
    assert all(p["on"] and p["max_tokens"] is None for p in pl["policies"].values()) and big.switches(F) == {}
    pl = big.plan(res, _query(tmp_path, 300, 506), {big.OFFLOAD_GATE: "1000"})                        # the card's parameter moves the gate
    assert pl["policies"]["pair_offload"]["on"] is True and pl["policies"]["pair_offload"]["gate"] == 1000
    for other in ("fast", "exact"):                                                                      # every other line: the hook is inert
        assert big.plan(modes.resolve(other, _stubs.TREE, {}), _query(tmp_path, 9999), {}) == {}
    # the decision composes the line at activation: between the gates the hoist rows are back and the unit is on
    big.plan(res, _query(tmp_path, 750), {})
    r = modes.resolve("big", _stubs.TREE, {})
    assert {"dit_hoist", "dit_align", "pair_offload_struct", "sample_chunk"} <= set(r.levers) and r.exports["ODDE_OFFLOAD"] == "all"
    # below the offload gate: fast's resident lever set, no offload switch, the unit's directory off the path
    big.plan(res, _query(tmp_path, 300, 506), {})
    r = modes.resolve("big", _stubs.TREE, {})
    assert not (set(r.levers) & set(modes._OFFLOAD_LEVERS)) and "sample_chunk" not in r.levers and "ODDE_OFFLOAD" not in r.exports and "ODDE_OFFLOAD" in r.unset
    assert r.line.name == F and r.mode == "big" and r.exports[modes.ALLOCATOR] == modes.EXPANDABLE
    big._ST["plan"] = None
    assert "no_dit_hoist" in modes.resolve("big", _stubs.TREE, big.plan(res, _query(tmp_path, 1478), {})).levers
    # the row-sharded line's plan gates the hoist only: sample_chunk stays at every size (big --n_gpu P unaffected)
    modes.MODE_LINES["big"] = modes.BIG_TP_LINE
    try:
        tp = modes.resolve("big", _stubs.TREE, {})
        pl = big.plan(tp, _query(tmp_path, 300, 506), {})
        assert pl["line"] == modes.BIG_TP_LINE and set(pl["policies"]) == {"no_dit_hoist"} and big.switches(modes.BIG_TP_LINE) == {"no_dit_hoist": False}
        assert "sample_chunk" in modes.resolve("big", _stubs.TREE, {}).levers
    finally:
        modes.MODE_LINES["big"] = F
        big._ST["plan"] = None


# ----------------------------------------------------------------------------------------------------------------- apply / census
class _Cfg:
    def __init__(self, chunk):
        self.infer_setting = types.SimpleNamespace(sample_diffusion_chunk_size=chunk)


class _Model:
    def __init__(self, chunk=5):
        self.configs = _Cfg(chunk)
        self.seen = []

    def stage(self, **kw):                                                     # the stock method: records the chunk size the model would read
        self.seen.append(self.configs.infer_setting.sample_diffusion_chunk_size if self.configs is not None else None)
        return ("coords", kw.get("N_sample"))


def _offload(installed=True):
    m = types.ModuleType("odde_offload"); m._INSTALLED = installed; m.CFG = {"stages": {"struct", "trunk", "conf"}}; m.STATS = {}
    return m


FAKE_SAMPLER = "e2b_fake_sampler_module"                                     # this test's own stock module: the CPU stubs' `opendde.model.opendde.OpenDDE`
                                                                             # (tests/_stubs.py) carries no sampler method, the real one needs torch


@pytest.fixture
def activated(monkeypatch):
    """Apply a big line the way stack._apply does (the allocator writer stubbed: its own tests are test_lever_mechanics.py; the census unit's
    site pointed at this module's fake sampler class), then export the line's switches; returns (res, rec)."""
    fake = types.ModuleType(FAKE_SAMPLER); fake.OpenDDE = _Model
    monkeypatch.setitem(sys.modules, FAKE_SAMPLER, fake)
    monkeypatch.setattr(big, "UNIT_MODULE", FAKE_SAMPLER)
    monkeypatch.setattr(_Model, "run_sample_diffusion_stage", _Model.stage, raising=False)

    def go(sel="big", env=None, offload=True):
        for k, v in (env or {}).items():
            monkeypatch.setenv(k, v)
        monkeypatch.setattr(alloc, "apply", lambda res, jobs=None: {"alloc_export": "present"})
        res = modes.resolve(sel, _stubs.TREE)
        facts = big.apply(res)
        assert facts == {"alloc_export": "present"}                              # the allocator writer's facts are the return value, unchanged
        for k, v in res.exports.items():
            monkeypatch.setenv(k, v)
        if offload is not None:
            monkeypatch.setitem(sys.modules, "odde_offload", _offload(offload))
        return res, big._ST["record"]
    return go


def _rep(res):
    return {"active": True, "mode": res.mode, "line": modes.describe_line(res), "line_name": None if res.mode else res.line.name,
            "levers_planned": list(res.levers), "levers_applied": ["served_levers_hook"], "levers_fallback": [], "partial": False}


def test_apply_records_the_line_through_the_core_and_arms_the_unit(activated):
    res, rec = activated()
    assert rec is not None and tuple(rec.levers) == ("pair_offload", "no_dit_hoist", "sample_chunk") and rec.exact == "band"
    assert rec.active_line(big.TAG).startswith("[opendde-opt] ACTIVE mode=big:pair_offload,no_dit_hoist,sample_chunk") and "base=fast" in rec.active_line(big.TAG)
    assert dict(rec.exact_per_lever) == {"pair_offload": "band", "no_dit_hoist": "bitwise", "sample_chunk": "band"}
    assert big._ST["line"] == F and big._ST["patch"] is not None and big._ST["allocator_conf"] == modes.EXPANDABLE
    a = big.applied()
    assert a["levers"] == ["pair_offload", "no_dit_hoist", "sample_chunk"] and a["allocator"]["conf"] == modes.EXPANDABLE
    from opt_core import mem
    assert {n: mem.LEVERS[n].strategy for n in ("pair_offload", "no_dit_hoist", "sample_chunk")} == big.STRATEGY
    assert set(big.STRATEGY.values()) <= {"F7.pair_offload", "F7.chunked_eval", "F7.hoist_off"}


def test_below_a_size_gate_the_record_names_the_switch(activated, monkeypatch):
    """The plan's decisions reach the core as switches: between the gates the record's line is the row without no_dit_hoist, off_by_flag
    names it; below the offload gate every memory lever is off by its gate — the record holds none, names all three, and is not partial."""
    monkeypatch.setitem(big._ST, "plan", _plan(1500))
    res, rec = activated(F)
    assert tuple(rec.levers) == ("pair_offload", "sample_chunk") and list(rec.off_by_flag) == ["no_dit_hoist"] and rec.exact == "band" and rec.allow_partial is False
    big._reset()
    monkeypatch.setitem(big._ST, "plan", _plan(1012))
    res, rec = activated(F, offload=None)                                          # the unit's module is not even imported below the gate
    assert tuple(rec.levers) == () and list(rec.off_by_flag) == ["pair_offload", "no_dit_hoist", "sample_chunk"] and not rec.refused
    assert "ODDE_OFFLOAD" not in res.exports and big.applied()["pair_offload_policy"]["on"] is False and big.applied()["no_dit_hoist_policy"]["on"] is False
    model = _Model(chunk=5)
    assert model.run_sample_diffusion_stage(N_sample=5) == ("coords", 5) and model.seen == [5]   # the census unit still delimits the prediction; the stock chunking (one batch) runs
    rep = big.refresh(_rep(res))
    assert rep["partial"] is False and not rep.get("levers_fallback") and rep["big"]["predictions"] == 1 and rep["big"]["census"]["ok"]
    assert rep["big"]["off_by_flag"] == ["pair_offload", "no_dit_hoist", "sample_chunk"] and rep["big"]["size_gate_tokens"]["item0"]["residue_tokens"] == 1012
    assert rep["big"]["gated_off"] == "pair_offload,no_dit_hoist,sample_chunk"                       # the EXIT tally's one scalar of the decision (big={… gated_off=…})
    assert any(f.startswith("big={") and "gated_off=pair_offload,no_dit_hoist,sample_chunk" in f and "exact=bitwise" in f for f in _report.tally_fields({"big": rep["big"]}))
    assert any(n.startswith("big: pair_offload off — largest item 1012 residue tokens < gate 1400") for n in rep["notes"])
    assert any(n.startswith("big: sample_chunk off — follows pair_offload below its size gate") for n in rep["notes"])
    lines = {ln.split("name=")[1].split(" ")[0]: ln for ln in _report.lever_lines(rep, {})}
    assert all(" state=off reason=below_gate:1012/1400 " in lines[r] for r in modes._OFFLOAD_LEVERS + ("sample_chunk",)) and " state=off reason=below_gate:1012/2565 " in lines["no_dit_hoist"]


def test_a_precondition_that_fails_refuses_by_name(monkeypatch):
    monkeypatch.setattr(alloc, "apply", lambda res, jobs=None: {})
    res = modes.resolve("big", _stubs.TREE)
    res.exports.pop("ODDE_OFFLOAD")                                              # a resolution that does not export the unit's switch
    with pytest.raises(big.BigRefusal, match="pair_offload.*the line's exports carry the unit's switches"):
        big.apply(res)


def test_every_prediction_is_a_census_unit_and_sample_chunk_acts_on_the_stock_config(activated):
    res, rec = activated()
    assert getattr(_Model.run_sample_diffusion_stage, "_big_stock", None) is _Model.stage      # the unit armed on the (fake) stock class: the wrapper over the stock method
    model = _Model(chunk=5)
    for _ in range(2):
        assert model.run_sample_diffusion_stage(N_sample=5) == ("coords", 5)                       # through the patched class = one census unit each
    assert model.seen == [1, 1] and model.configs.infer_setting.sample_diffusion_chunk_size == 5      # chunks of 1 inside the call, the stock value after it
    census = rec.census()
    assert census["n_units"] == 2 and census["ok"] and census["per_lever"]["sample_chunk"]["ran"] == 2 and census["per_lever"]["pair_offload"]["ran"] == 2
    rep = big.refresh(_rep(res))
    assert rep["partial"] is False and {"no_dit_hoist", "sample_chunk"} <= set(rep["levers_applied"])
    blk = rep["big"]
    assert blk["predictions"] == 2 and blk["kit_line"] == F and blk["mode_line"] == "big:pair_offload,no_dit_hoist,sample_chunk" and blk["n_gpu"] == 1
    assert blk["census"]["n_ok"] == 2 and blk["allocator_check"]["absent_at_predictions"] == 0 and blk["exact"] == "band"
    json.dumps(blk)                                                              # the manifest's kit.big block is JSON-ready
    ev = big.lever_evidence("sample_chunk")
    assert (ev["units"], ev["ran"], ev["fallback"], ev["samples"], ev["strategy"], ev["exact"]) == (2, 2, 0, 1, "F7.chunked_eval", "band")
    assert big.lever_evidence("pair_offload_trunk")["ran"] == 2 and big.lever_evidence("no_dit_hoist")["exact"] == "bitwise"
    assert big.lever_evidence("arm_u") is None
    lines = {ln.split("name=")[1].split(" ")[0]: ln for ln in _report.lever_lines(rep, {})}
    assert " state=on " in lines["no_dit_hoist"] and " state=on " in lines["sample_chunk"] and "state=off" in lines["dit_hoist"]


def test_n_sample_within_one_chunk_is_a_named_skip_not_partial(activated):
    res, rec = activated()                                                       # samples=1 (the row's setting): a one-sample prediction is one chunk either way
    model = _Model(chunk=5)
    big._prediction(model, _Model.stage, {"N_sample": 1})
    assert model.seen == [5]
    rep = big.refresh(_rep(res))
    assert rep["partial"] is False and rep["big"]["census"]["per_lever"]["sample_chunk"]["skipped"] == 1
    assert "sample_chunk" in rep.get("levers_inert", []) and "sample_chunk" not in rep["levers_applied"]


@pytest.mark.parametrize("case, words", [
    ("no_unit", "pair_offload: the offload unit is not installed"),
    ("hoist_env", "no_dit_hoist: ODDE_ADDON_LEVERS='dit_hoist,dit_align' is set"),
    ("no_config", "sample_chunk: N_sample or configs.infer_setting.sample_diffusion_chunk_size not reachable"),
    ("allocator", "allocator: PYTORCH_CUDA_ALLOC_CONF without expandable_segments:True at 1 predictions"),
    ("no_prediction", "no_dit_hoist: no prediction ran in this process"),
])
def test_a_lever_that_did_not_run_on_a_prediction_is_partial_by_name(activated, monkeypatch, case, words):
    res, rec = activated(offload=(case != "no_unit"))
    model = _Model(chunk=5)
    if case == "hoist_env":
        monkeypatch.setenv("ODDE_ADDON_LEVERS", "dit_hoist,dit_align")
    if case == "no_config":
        model.configs = None
    if case == "allocator":
        monkeypatch.setenv(modes.ALLOCATOR, "max_split_size_mb:128")
    if case != "no_prediction":
        big._prediction(model, _Model.stage, {"N_sample": 5})
    rep = big.refresh(_rep(res))
    assert rep["partial"] is True and any(f.startswith(words) for f in rep["levers_fallback"]), rep["levers_fallback"]
    name = words.split(":")[0]
    if name in registry.LEVERS:
        state, reason = _report.lever_state(name, rep)
        assert state == "skipped" and reason.startswith("fell back: " + name)


def test_no_variable_suppresses_the_census_partial(activated, monkeypatch):
    """`pred --allow-partial` (the exit code, cli) is the one opt-out: a caller's OPENDDE_BIG_ALLOW_PARTIAL=1 does not un-name a fallback."""
    res, rec = activated(offload=False)                                          # applied at allow_partial=False; a caller's OPENDDE_BIG_ALLOW_PARTIAL is refused by name before this point
    assert rec.allow_partial is False
    big._prediction(_Model(), _Model.stage, {"N_sample": 5})
    rep = big.refresh(_rep(res))
    assert rep["partial"] is True and rep["levers_fallback"]
    assert not any("ALLOWED" in n for n in rep.get("notes", []))


def test_the_policies_land_in_the_report(activated, tmp_path):
    res0 = modes.resolve("big", _stubs.TREE, {})
    big.plan(res0, _query(tmp_path, 750), {})                                   # 1500 tokens, between the gates: the decisions stay in this process for compose / apply
    res, rec = activated()
    assert "no_dit_hoist" not in rec.levers and {"dit_hoist", "dit_align"} <= set(res.levers) and "pair_offload" in rec.levers
    big._prediction(_Model(), _Model.stage, {"N_sample": 5})
    rep = big.refresh(_rep(res))
    assert rep["big"]["no_dit_hoist_policy"]["on"] is False and any(n.startswith("big: no_dit_hoist off") for n in rep["notes"])
    assert rep["big"]["pair_offload_policy"]["on"] is True and any(n.startswith("big: pair_offload on — largest item 1500 residue tokens >= gate 1400") for n in rep["notes"])
    assert rep["big"]["gated_off"] == "no_dit_hoist"
    assert rep["partial"] is False and rep["big"]["off_by_flag"] == ["no_dit_hoist"] and "off=no_dit_hoist" in rec.active_line(big.TAG)


def test_an_older_core_refuses_every_big_line_by_name_on_every_route(monkeypatch, capsys):
    """core_missing: on a core without the memory-lever registry (opt_core < 0.4.0) the big lines refuse BY NAME before anything is
    applied — resolve (every route resolves: cli pred / check, the env route's activate, the Python call) and the CLI's rc 3."""
    import opt_core.mem as memmod
    from opendde_opt import cli
    monkeypatch.delitem(memmod.LAZY_EXPORTS, "register")                                              # the core binds `register` on first access (opt_core.mem LAZY_EXPORTS):
    monkeypatch.delattr(memmod, "register", raising=False)                                             # an older core has neither the table entry nor the binding
    for sel in ("big", "BIG_F"):
        with pytest.raises(modes.OpenModeError, match=r"core_missing: opt_core .* carries no memory-lever registry .* nothing applied"):
            modes.resolve(sel, _stubs.TREE, {})
    assert modes.resolve("fast", _stubs.TREE, {}).line.name == "LSTAR2A"                                # the other lines are untouched
    monkeypatch.setenv("MODEL_OPT", _stubs.TREE)
    assert cli.main(["check", "--mode", "big"]) == cli.EXIT_NOT_ACTIVE
    assert "core_missing" in "".join(capsys.readouterr())


def test_an_absent_core_refuses_every_big_line_by_name():
    """core_missing:opt_core.mem — with no opt_core importable at all, `opendde_opt.modes` / `opendde_opt.big` still import (no core at
    module top) and every big line refuses BY NAME at resolve; the other lines resolve. (`python -S`: no site-packages, the package
    directory alone on the path.)"""
    import subprocess
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))         # opendde/opt
    code = ("import sys; sys.path[:] = [p for p in sys.path if 'opt_core' not in p]; sys.path.insert(0, %r)\n"
            "import importlib.util as u\n"
            "if u.find_spec('opt_core') is not None: print('CORE-PRESENT'); raise SystemExit(0)\n"
            "import opendde_opt.modes as m, opendde_opt.big\n"
            "for sel in ('big', 'BIG_F'):\n"
            "    try:\n        m.resolve(sel, '/T', {}); print('NO-REFUSAL', sel)\n"
            "    except m.OpenModeError as e: print('REFUSED', sel, str(e))\n"
            "print('FAST', m.resolve('fast', '/T', {}).line.name)\n") % pkg_root
    r = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    if "CORE-PRESENT" in r.stdout:
        pytest.skip("opt_core is importable even under `python -S` in this environment (installed into the stdlib path): the absent-core probe cannot be staged here")
    assert r.returncode == 0, r.stderr[-600:]
    assert r.stdout.count("REFUSED") == 2 and "NO-REFUSAL" not in r.stdout and "FAST LSTAR" in r.stdout, r.stdout
    assert r.stdout.count("core_missing:opt_core.mem") == 2, r.stdout


def test_the_fused_token_stack_is_off_every_big_line_by_name():
    """0.2.61: dit_fused + dit_lowp (the fused token stack's resident packed fp16 weight copy: +0.45 GiB allocated peak at 1,400-1,600 tokens on
    H100) are off EVERY big line — the static rows, the resident composition below the size gates, the row-sharded line — with their switches
    unset; fast keeps them; the sampler's other levers stay on the one-card lines."""
    fast = modes.LINES["LSTAR2A"]
    assert {"dit_fused", "dit_lowp"} <= set(fast.levers) and fast.exports["ODDE_DIT_FUSED"] == "1" and fast.exports["ODDE_DIT_LOWP"] == "fp16"
    rows = [modes.LINES[n] for n in modes.BIG_LINES]
    rows += [modes.big_line(n, mem) for n in modes.BIG_LINES                                            # every composition big.compose can build: each subset of the
             for mem in itertools.chain.from_iterable(itertools.combinations(modes.BIG_MEM_LEVERS[n], k) for k in range(len(modes.BIG_MEM_LEVERS[n]) + 1))]   # line's memory levers
    rows += [modes.line_without(modes.LINES[n], ("no_dit_hoist",)) for n in modes.BIG_LINES]                # an ablation route rebuilds through big_line too
    for ln in rows:
        assert not {"dit_fused", "dit_lowp"} & set(ln.levers), ln.name
        assert "ODDE_DIT_FUSED" not in ln.exports and "ODDE_DIT_LOWP" not in ln.exports and {"ODDE_DIT_FUSED", "ODDE_DIT_LOWP"} <= set(ln.unset), ln.name
        if ln.name != modes.BIG_TP_LINE:
            assert {"dit_attn_apb", "atom_attn_apb", "cond_dedupe", "atom_fused"} <= set(ln.levers), ln.name
