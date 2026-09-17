"""attn.pair_fused cell resolution under P12 / R4 (opt_core.kernels.cell_words): certified rows select exactly what they select in the recorded
land decisions (tests/fixtures/pair_fused_land_decisions.json); a key the table lists as MEASURED OFF (named_off[] / a candidate row not admitted)
is its land `no-cell:…` word qualified `+off(<why>)` and never engages; an UNKNOWN key engages the lever's SAFE settings (or an lnl piece's default
settings) and is named ONCE; nothing else changes.  CPU only: stacks are named (cc, triton)."""
import copy
import json
import os

import pytest

from opt_core.attn import pair_fused as PF
from opt_core.kernels import cell_words as CW
from opt_core.kernels import safe_settings as SS

HERE = os.path.dirname(os.path.abspath(__file__))
LAND = json.load(open(os.path.join(HERE, "fixtures", "pair_fused_land_decisions.json")))["decisions"]
CERTIFIED_CCS = ("8.0", "9.0", "10.0", "10.3")


@pytest.fixture
def clean():
    for n in PF._NETS.values(): n.reset()
    PF.SERVED_KEYS.clear()
    yield
    for n in PF._NETS.values(): n.reset()
    PF.SERVED_KEYS.clear()
    os.environ.pop(PF.ALLOW_CANDIDATE_ENV, None)


def _decide(k):
    impl, piece, key, cc, tt, var, allow = k.split("|")
    os.environ[PF.ALLOW_CANDIDATE_ENV] = "1" if allow == "1" else ""
    try:
        return PF.lookup_cell(impl, piece, [int(x) for x in key.split("x")], stack=(cc, tt), variant=None if var == "None" else var)
    finally:
        os.environ.pop(PF.ALLOW_CANDIDATE_ENV, None)


def _universe():
    return {(r["impl"], r["piece"], "x".join(map(str, r["key"]))) for r in PF.cells()["rows"]}


def test_land_served_rows_select_identically():
    """The certified-path gate: every (key, stack, variant, switch) a table row served in the land record is served by the SAME row id today."""
    served = {k: v for k, v in LAND.items() if v[0]}
    assert len(served) > 500 and {v[0] for v in served.values()} >= {"9.0|*", "8.0|*", "10.0|*", "10.3|*", "9.0|3.3", "9.0|3.7"}
    for k, (served_by, rid, _reason) in served.items():
        d = _decide(k)
        assert d.served and d.row["id"] == rid and d.served_by == served_by and d.reason == "", (k, d)
    assert {v[1] for v in served.values()} == {r["id"] for r in PF.cells()["rows"]}          # every row of the table is exercised


def test_land_refusals_on_certified_capabilities_stay_unserved_for_every_shape_a_kit_runs():
    """Zero change to certified compositions: on 8.0 / 9.0 / 10.0 / 10.3 every key some row lists (a shape a kit runs) that was refused on land is
    refused today — its land word, qualified `+off(<why>)` (measured off) or `+no_safe(<why>)` (the safe rows exclude the class) or unchanged
    (a variant without a safe cell)."""
    uni = _universe()
    n_off = n_nosafe = n_same = 0
    for k, (served_by, rid, reason) in LAND.items():
        impl, piece, key, cc, tt, var, allow = k.split("|")
        if served_by or cc not in CERTIFIED_CCS or (impl, piece, key) not in uni:
            continue
        d = _decide(k)
        assert not d.served, (k, d)
        assert d.reason == reason or d.reason.startswith(reason + "+"), (k, reason, d.reason)          # the land word is the head: kits routing on it route as before
        if CW.is_off_word(d.reason):
            n_off += 1; assert d.off and d.word().startswith("off:")
        elif "+no_safe(" in d.reason and "+no_safe(" not in reason:
            n_nosafe += 1
        else:
            n_same += 1; assert d.reason == reason
    assert n_off > 100 and n_same > 100, (n_off, n_nosafe, n_same)


@pytest.mark.parametrize("impl,piece,key,cc,why", [
    ("fpf", "prologue", (64, 4, 64), "9.0", "not-measured"),        # a c=64, 4 x 64 pair stack on H100: a candidate row, not admitted (the 4 x 32 template key is certified, r11)
    ("fpf", "prologue", (256, 4, 64), "8.0", "not-measured"),       # a c_z=256, 4 x 64 pair stack on A100: a named_off entry (no row)
    ("fpf", "transition", (384, 1536), "8.0", "slower"),           # r58: measured slower on A100 -> the row's own off word
    ("lnl", "gate_transpose", (777,), "10.0", "not-measured"),     # ... any key
])
def test_measured_off_keys_are_named_and_unserved(impl, piece, key, cc, why, clean, capfd):
    d = PF.lookup_cell(impl, piece, key, stack=(cc, "3.7"))
    head = f"no-cell:{impl}:{piece}:{'x'.join(map(str, key))}:{cc}|3.7"
    assert not d.served and d.off and d.reason == head + CW.off_suffix(why) and CW.off_why(d.reason) == why
    assert d.word() == f"off:{why}:{'x'.join(map(str, key))}" and d.word("c64") == f"off:{why}:c64"
    with pytest.raises(PF.Unsupported) as e:
        PF.find_cell(impl, piece, key, None, stack=(cc, "3.7"))
    assert e.value.reason == d.reason and capfd.readouterr().err == ""                          # nothing prints, nothing engages
    assert all(not n.on and n.note() is None for n in PF._NETS.values())


def test_candidate_switch_still_serves_candidates(clean):
    os.environ[PF.ALLOW_CANDIDATE_ENV] = "1"
    d = PF.lookup_cell("fpf", "prologue", (64, 4, 64), stack=("9.0", "3.7"))
    assert d.served and d.row["status"] == "candidate" and d.served_by == "9.0|*"


def test_fp32_stream_variants_are_measured_off_everywhere(clean):
    for cc in CERTIFIED_CCS + ("8.6",):
        d = PF.lookup_cell("fpf", "prologue", (128, 4, 32), stack=(cc, "3.7"), variant="v3_fp32z")
        assert d.off and d.reason == f"no-cell:fpf:prologue:128x4x32:{cc}|3.7+off(not-measured)", (cc, d)
        t = PF.lookup_cell("fpf", "transition", (128, 512), stack=(cc, "3.7"), variant="v1_fp32x")
        assert t.off and t.reason == f"no-cell:fpf:transition:128x512:{cc}|3.7+off(not-measured)", (cc, t)
        b = PF.lookup_cell("fpf", "prologue", (128, 4, 32), stack=(cc, "3.7"), variant="v3")     # the bf16 variant of the same key is untouched: its certified row,
        assert b.served and b.served_by == (f"{cc}|*" if cc in CERTIFIED_CCS else "8.0|*;%s(%s->8.0)" % (PF.INHERIT_TOKEN, cc))   # or (a capability without a column) the nearest measured column's row


def test_unknown_key_on_a_certified_capability_engages_the_safe_cell_named_once(clean, capfd):
    """(U): a shape no row and no named_off entry lists, on a capability with SAFE rows -> the SAFE cell serves, ONE line, census word."""
    d = PF.lookup_cell("fpf", "prologue", (32, 2, 16), stack=("9.0", "3.7"))
    assert d.served and d.safe and d.reason == "no_cell:32x2x16" and d.row["status"] == "safe" and d.row["variant"] == "v3"
    assert d.row["cfg"] == dict(SS.safe_row("pair_fused:prologue", "9.0")["settings"]) and d.word() == "safe:no_cell:32x2x16"
    assert capfd.readouterr().err == ""                                                        # lookup prints nothing
    row = PF.find_cell("fpf", "prologue", (32, 2, 16), None, stack=("9.0", "3.7"))
    assert row["status"] == "safe" and PF.row_key(row) == "safe"
    assert capfd.readouterr().err.strip() == "[opt_core/pair_fused:prologue] safe settings served (no_cell:32x2x16, cc 9.0, triton 3.7)"
    PF.find_cell("fpf", "prologue", (32, 2, 16), None, stack=("9.0", "3.7"))
    assert capfd.readouterr().err == ""                                                        # once per process
    assert PF.evidence_tail()["settings"] == "safe:no_cell:32x2x16"
    t = PF.lookup_cell("fpf", "transition", (96, 384), stack=("8.0", "3.6"))
    assert t.safe and t.row["cfg"] == dict(SS.safe_row("pair_fused:transition", "8.0")["settings"])


def test_unknown_key_the_safe_rows_exclude_keeps_the_no_safe_word(clean):
    d = PF.lookup_cell("fpf", "epilogue", (256, 8, 32), stack=("8.0", "3.3"))                  # c_z 256 on cc 8.0, no row for this triton line: the v2 class is measured slower, excluded BY NAME
    assert not d.served and not d.off and d.reason == "no-cell:fpf:epilogue:256x8x32:8.0|3.3+no_safe(c_z_above_128_slower_than_stock_on_cc8.0)"
    r = PF.lookup_cell("fpf", "epilogue", (256, 8, 32), stack=("8.0", "3.7"))                  # the triton-3.7 line has a MEASURED row (epilogue_v3, x2.3-2.9 vs the statements): served
    assert r.served and r.row["id"] == "r103" and r.row["variant"] == "epilogue_v3"
    v = PF.lookup_cell("fpf", "prologue", (32, 2, 16), stack=("9.0", "3.7"), variant="prologue_v4")   # a variant without a safe cell: plain no-cell
    assert not v.served and v.reason == "no-cell:fpf:prologue:32x2x16:9.0|3.7"


def test_unknown_lnl_key_engages_default_settings_named_once(clean, capfd):
    d = PF.lookup_cell("lnl", "ln_linear", (256,), stack=("9.0", "3.7"))
    assert d.served and d.served_by == "default" and d.row["status"] == "default" and d.row["variant"] == "ln_linear" and d.row["cfg"] == {}
    assert d.word() == "default:256" and PF.row_key(d.row) == "default"
    PF.find_cell("lnl", "ln_linear", (256,), None, stack=("9.0", "3.7"))
    err = capfd.readouterr().err.strip()
    assert err == "[opt_core/pair_fused:lnl] default settings (no row for ln_linear 256 on cc 9.0, cc 9.0, triton 3.7); tuned rows exist for cc 8.0, 9.0"
    PF.find_cell("lnl", "gate_transpose", (512,), None, stack=("9.0", "3.7"))
    assert capfd.readouterr().err == ""                                                        # one line per process for the impl
    assert PF.evidence_tail()["cells_note"] == "pair_fused:lnl:default:no_row"
    off = PF.lookup_cell("lnl", "ln_linear", (256,), stack=("10.0", "3.7"))                    # the same key on B200: measured off (named_off)
    assert off.off and not off.served


def test_unknown_capability_takes_the_nearest_measured_column_else_the_safe_rows_the_core_carries(clean, monkeypatch):
    """A capability no row names: at or above a measured column (cc 8.6 here) it runs the nearest measured column's certified row (8.0's,
    served_by carries the inherited token); BELOW every measured column (cc 7.5 here) it is served the lever's SAFE settings when
    safe_settings carries a row for it (an any-capability '*' row or its own), else the `no-cell` word — never a named_off refusal unless an
    entry says so."""
    d = PF.lookup_cell("fpf", "prologue", (128, 4, 32), stack=("8.6", "3.7"))
    assert d.served and d.row["cc"] == "8.0" and d.row["status"] == "certified" and d.served_by == "8.0|*;%s(8.6->8.0)" % PF.INHERIT_TOKEN
    has_safe = SS.safe_row("pair_fused:prologue", "7.5") is not None
    d = PF.lookup_cell("fpf", "prologue", (128, 4, 32), stack=("7.5", "3.7"))
    assert d.served == has_safe and (d.safe if has_safe else d.reason == "no-cell:fpf:prologue:128x4x32:7.5|3.7")
    rows = copy.deepcopy(SS.SAFE_ROWS)
    rows.setdefault("pair_fused:prologue", {})["7.5"] = {"settings": {"BI": 8, "BJ": 8, "num_warps": 4, "num_stages": 1}, "status": "SAFE:test"}
    monkeypatch.setattr(SS, "SAFE_ROWS", rows)
    d2 = PF.lookup_cell("fpf", "prologue", (128, 4, 32), stack=("7.5", "3.7"))
    assert d2.safe and d2.row["cfg"] == {"BI": 8, "BJ": 8, "num_warps": 4, "num_stages": 1}

def test_named_off_matcher_fields():
    e = {"impl": "fpf", "piece": "prologue", "key": [64, 4, 32], "cc": "8.0"}
    assert PF._entry_matches(e, "fpf", "prologue", (64, 4, 32), "8.0", None) and PF._entry_matches(e, "fpf", "prologue", (64, 4, 32), "8.0", "v3")
    assert not PF._entry_matches(e, "fpf", "prologue", (64, 4, 32), "9.0", None) and not PF._entry_matches(e, "fpf", "epilogue", (64, 4, 32), "8.0", None)
    v = {"impl": "fpf", "piece": "prologue", "variant": "v3_fp32z", "key": "*", "cc": "*"}
    assert PF._entry_matches(v, "fpf", "prologue", (1, 2, 3), "12.0", "v3_fp32z") and not PF._entry_matches(v, "fpf", "prologue", (1, 2, 3), "12.0", None)
    w = {"impl": "lnl", "piece": "*", "key": "*", "cc": "10.0"}
    assert PF._entry_matches(w, "lnl", "gate_transpose", (5,), "10.0", None) and not PF._entry_matches(w, "fpf", "prologue", (5,), "10.0", None)
    ids = [e["id"] for e in PF.cells()["named_off"]]
    assert len(ids) == len(set(ids)) and all(set(e) >= {"id", "impl", "piece", "key", "cc", "off", "evidence"} for e in PF.cells()["named_off"])


def test_build_failure_of_an_unknown_keys_safe_cell_is_the_levers_refusal(clean, capfd):
    """CANNOT RUN: the SAFE cell of an unknown key failing to BUILD is safe_settings case (iii) — Refused propagates (the mode refuses by name)."""
    class _Build(RuntimeError):                                          # a RuntimeError carrying triton's MLIR words (safe_settings.is_build_failure)
        pass
    seen = []

    def build(cfg):
        seen.append(dict(cfg)); raise _Build("PassManager::run failed")
    with pytest.raises(PF.Refused) as e:
        PF.run_cell("pair_fused:prologue", "9.0", "3.7", build, {"BI": 16}, dims={"c_z": 32, "H": 2, "D": 16})
    assert "pair_fused:prologue" in str(e.value) and "--mode off" in str(e.value)
    assert seen == [{"BI": 16}, dict(SS.safe_row("pair_fused:prologue", "9.0")["settings"])]      # the tuned cell, then the SAFE cell, then the refusal
    assert "safe settings served (build_failed:_Build, cc 9.0, triton 3.7)" in capfd.readouterr().err


def test_cell_words_grammar():
    assert CW.with_off("no-cell:x") == "no-cell:x+off(not-measured)" and CW.with_off("w", CW.SLOWER) == "w+off(slower)"
    assert CW.is_off_word("a+off(b)") and not CW.is_off_word("a+no_safe(b)") and not CW.is_off_word(None)
    assert CW.off_why("a+off(slower)") == "slower" and CW.off_why("a") == "" and CW.unverified_word("bf16_C512") == "unverified(bf16_C512)"



@pytest.mark.parametrize("impl,piece,key,cc", [
    ("fpf", "transition", (64, 256), "10.0"),        # the c=64 n=4 template transition on B200: named off NOT-MEASURED there, certified on 9.0 -> 9.0's row serves (inherited)
    ("lnl", "ln_linear", (128,), "10.3"),            # the lnl pieces on B300: named off not-measured (wildcard), certified below -> inherited
])
def test_not_measured_keys_on_a_capability_inherit_the_nearest_measured_columns_row(impl, piece, key, cc, clean):
    """NO-REGRESSION per-key rule: a key a capability never measured (off word not-measured, or no entry at all) is served by the nearest
    measured column's certified row for that key -- never a refusal where an arch-portable row exists; a MEASURED off entry (slower ...) on
    the capability itself still wins (test_measured_off_keys_are_named_and_unserved)."""
    col = PF.inherit_column(cc, impl, piece, key)
    assert col is not None
    d = PF.lookup_cell(impl, piece, key, stack=(cc, "3.7"))
    assert d.served and d.row["status"] == "certified" and d.row["cc"] == col and d.served_by.endswith("%s(%s->%s)" % (PF.INHERIT_TOKEN, cc, col)), (d.served_by, d.reason)
