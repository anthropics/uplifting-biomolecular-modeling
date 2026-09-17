"""kernels/transition: the transition-family provider -- its cell table, selection rules (default rule = row word / opt-in tier words / EXACT
FLOOR x1.00 / refusals by name / capture words), the forward and forward+backward passes, the row envelopes (admits), the int64 / int32 row-offset
audit pins of every carried kernel, the module census and the carried sub-packages' byte identity with the kit copies.  CPU only."""
import ast
import filecmp
import hashlib
import json
import os
import re

import pytest

from opt_core import kernels
from opt_core.kernels import transition as T

CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE_TREE = os.path.dirname(os.path.dirname(CORE_DIR))
PKG = os.path.join(CORE_DIR, "opt_core", "kernels", "transition")
CARDS = ("9.0", "8.0")
SIZES = (400, 800, 1200)
H100 = "H100:torch2.13.0+cu130/3.7.1"
A100 = "A100:torch2.13.0+cu130/3.7.1"
ESM_H100 = "H100:torch2.11.0+cu128/3.6.0"
ESM_A100 = "A100:torch2.11.0+cu128/3.6.0"
# every transition the release tree's torch engines run today: (dtype, cell word, family, direction) -- the family's shape table
SHAPES = [("bf16", "pair_c128_n4", "pair", "fwd"), ("fp32", "pair_c128_n4", "pair", "fwd"), ("bf16", "pair_c128_n2", "pair", "fwd"), ("fp32", "pair_c128_n2", "pair", "fwd"),
          ("bf16", "pair_c256_n4", "pair", "fwd"), ("fp32", "pair_c384_n4", "pair", "fwd"), ("bf16", "single_c384_n4", "single", "fwd"), ("fp32", "single_c384_n4", "single", "fwd"),
          ("bf16", "rows_c64_n4", "rows", "fwd"), ("fp32", "rows_c64_n4", "rows", "fwd"), ("bf16", "esmpair_c256_n4", "esmpair", "fwd"), ("bf16", "esmpair_c256_n4", "esmpair", "fwdbwd")]
MEASURED_CCS = {("fp32", "pair_c384_n4"): ("9.0",)}      # (dtype, cell) families measured on one card only (the fp32 c=384 pair stack: cc 9.0 rows to N 800)
MEASURED_TO = {("fp32", "pair_c384_n4"): 800}
DIMS = {"pair_c128_n4": (128, 512), "pair_c256_n4": (256, 1024), "pair_c128_n2": (128, 256), "pair_c384_n4": (384, 1536), "single_c384_n4": (384, 1536), "single_c384_n2": (384, 768), "pair_c256_n2": (256, 512), "single_c768_n4": (768, 3072), "rows_c64_n4": (64, 256), "rows_c64_n2": (64, 128),
        "esmpair_c256_n4": (256, 1024)}


def test_table_schema_rows_words():
    t = T.table()
    assert t["schema"] == "transition_cells/v1"
    for k in ("op", "key_grammar", "evidence", "stacks", "rows", "extra_arms", "tiers", "default_rule", "exact_rule", "capture_rule", "cells", "coverage"):
        assert k in t, k
    assert set(T.ROW_NAMES) == set(t["rows"]), sorted(set(T.ROW_NAMES) ^ set(t["rows"]))
    assert set(T.TIER_WORDS) == set(t["tiers"]), sorted(set(T.TIER_WORDS) ^ set(t["tiers"]))
    for name, row in t["rows"].items():
        for k in ("class", "capture_safe", "backward", "fallback", "serves", "numerics"):
            assert k in row, (name, k)
        assert row["class"] in ("fast", "exact", "stock", "schedule"), name
        assert row["fallback"] is None or row["fallback"] in T.ROW_NAMES, name
    assert {n for n, r in t["rows"].items() if r["class"] == "stock"} == set(T.STOCK_ROWS)
    assert {n for n, r in t["rows"].items() if r["class"] == "exact"} == set(T.EXACT_CLASS_ROWS) == {"flash_sm90a", "esm_fused_exact"}
    assert set(T.EXACT_GIVEN_LN_ROWS) == {"flash_sm90a", "v1"} and "flash_sm90a" in T.EXACT_CLASS_ROWS      # exact class given the caller's LayerNorm output
    assert {n for n, r in t["rows"].items() if r["class"] == "schedule"} == set(T.SCHEDULE_ROWS) == {"rowpair"}
    assert {n for n, r in t["rows"].items() if r["backward"]} == set(T.BACKWARD_ROWS)
    assert {n for n, r in t["rows"].items() if not r["capture_safe"]} == set(T.CAPTURE_UNSAFE_ROWS) == {"compile"}
    assert set(T.FORWARD_ONLY_ROWS) | set(T.BACKWARD_ROWS) | set(T.SCHEDULE_ROWS) == set(T.ROW_NAMES)
    assert not set(T.FORWARD_ONLY_ROWS) & set(T.BACKWARD_ROWS)
    assert len(t["cells"]) >= 200
    assert set(CARDS) <= {k.split("|")[0] for k in t["cells"]} and all(re.match(r"^\d+\.\d$", k.split("|")[0]) for k in t["cells"])   # the reference cards + any measured part's capability
    assert len(t["stacks"]) >= 4 and all(re.match(r"^(H100|H200|A100|A100_40|cc\d{2,3}):torch\d+\.\d+\.\d+\+cu\d+/\d+\.\d+\.\d+$", s) for s in t["stacks"]), list(t["stacks"])   # card word = the face's stack_word / norm_stack vocabulary


def test_every_cell_is_well_formed_and_its_winners_obey_the_rules():
    t = T.table()
    for key, cell in t["cells"].items():
        cc, dt, cw, n, timing, d = key.split("|")
        assert dt in ("bf16", "fp32") and cw in T.CELL_WORDS and n.startswith("N<=") and timing in ("eager", "graph") and d in T.PASS_WORDS, key
        assert cell["ref_stack"] in cell["stacks"] and cell["ref_stack"] in t["stacks"], key
        assert (cell["c"], cell["hidden"]) == DIMS[cw.split("+")[0]] == T.cell_dims(cw), key
        assert "+" not in cw or cw.split("+")[1] in T.FORMS, key
        for st, sc in cell["stacks"].items():
            assert st in t["stacks"], (key, st)
            for w in ("fast", "faithful", "exact", "big"):
                arm = sc.get(w)
                if arm is None:
                    continue
                row = T.split_word(arm)[0]
                assert row in T.ROW_NAMES, (key, st, w, arm)                       # winners are row words (kit arms measured for the record never win)
                if w == "exact" and row in T.STOCK_ROWS and sc.get("statement_na"):
                    continue                                                       # a kernel-only cell beyond the statement's memory ceiling: exact = the statement by name
                assert arm in sc["ms"], (key, st, w, arm)                           # ... with a number
                if d == "fwdbwd":
                    assert row in T.BACKWARD_ROWS, (key, st, w, arm)               # forward-only rows never win a fwd+bwd cell
                if timing == "graph":
                    assert row not in T.CAPTURE_UNSAFE_ROWS, (key, st, w, arm)
            ex = sc.get("exact")
            if ex is not None and T.split_word(ex)[0] not in T.STOCK_ROWS:          # EXACT FLOOR x1.00: bitwise AND not slower, else the stock arm by name
                assert sc["class"][ex] == "bitwise" and sc["x_stock"][ex] >= 1.0, (key, st, ex)
            if ex is not None and T.split_word(ex)[0] in T.STOCK_ROWS and not sc.get("statement_na"):   # the stock arm itself, or a stock-family row measured bitwise against it
                assert sc.get("exact_parity_flag") == "stock" or str(sc.get("exact_parity_flag")).startswith("bitwise_vs:"), (key, st, sc.get("exact_parity_flag"))
            if sc.get("statement_na"):                                                   # KERNEL-ONLY cell (the statement cannot run at this size): exact is the statement by name,
                assert ex in T.STOCK_ROWS and sc.get("stock_ms") is None and sc.get("fast") and T.split_word(sc["fast"])[0] not in T.STOCK_ROWS, (key, st)   # x undefined
            fa = sc.get("fast")
            if fa is not None and sc.get("stock_arm") in sc["ms"] and T.split_word(fa)[0] not in T.STOCK_ROWS:
                assert sc["ms"][fa] <= sc["ms"][sc["stock_arm"]] * 1.0000001, (key, st, fa)   # the fast winner is never slower than the stock arm it was timed with
            bg = sc.get("big")
            if bg is not None and T.split_word(bg)[0] not in T.STOCK_ROWS and not sc.get("statement_na"):
                assert (sc["x_stock"].get(bg) or 0) >= 1.0, (key, st, bg)
            for arm in sc["ms"]:
                assert arm.startswith("x:") or T.split_word(arm)[0] in T.ROW_NAMES, (key, st, arm)
                if arm.startswith("x:"):
                    assert arm in t["extra_arms"], (key, arm)
            assert not any(a.startswith(("kf1", "x:kf1")) for a in sc["ms"]), (key, st)       # the candidate GEMM never imported in the measured images: no number of it is in the table
        for k in ("fast", "exact", "big"):
            if cell.get(k) is not None:
                assert cell[k] == cell["stacks"][cell["ref_stack"]].get(k), (key, k)


@pytest.mark.parametrize("dt,cw,family,d", SHAPES)
def test_every_engine_shape_has_measured_cells_on_both_cards_and_the_tier_words_resolve(dt, cw, family, d):
    c, hidden = DIMS[cw]
    for cc in MEASURED_CCS.get((dt, cw), CARDS):
        stacks = (ESM_H100, ESM_A100) if family == "esmpair" else (H100, A100)
        st = stacks[0] if cc == "9.0" else stacks[1]
        for n in SIZES:
            key, size, beyond = T.cell_key(cc, dt, cw, n, "eager", d)
            assert key is not None, (cc, dt, cw, n, d)
            assert size == n or (beyond and size == MEASURED_TO.get((dt, cw)) < n), (cc, dt, cw, n, d, size)   # a family measured to a smaller size serves it flagged beyond_measured
            cell = T.table()["cells"][key]
            assert cell["measured"], key
            for tier in T.TIER_WORDS:
                try:
                    s = T.select(tier, c=c, hidden=hidden, n_tokens=n, dtype=dt, direction=d, family=family, residual=(family == "esmpair"), cc=cc, stack=st)
                except T.Refusal as e:
                    assert e.kind.startswith(("no_winner", "tier_winner_outside_envelope")), (key, tier, str(e))
                    continue
                assert s.tier == tier and s.word == tier and s.cell_key == key and s.row in T.ROW_NAMES, s
                assert s.stack_measured is True and s.ms is not None and s.x_stock is not None and s.beyond_measured == beyond, s
                if tier == "exact" and s.row not in T.STOCK_ROWS:
                    assert s.cls == "bitwise" and s.x_stock >= 1.0, s
                if tier == "faithful" and s.row not in T.STOCK_ROWS:
                    assert s.cls in ("stock", "bitwise") or cell["stacks"][st]["relrms"][T._arm_word(s.row, s.variant, T.cfg_word(s.cfg) if s.row == "v2" and s.cfg else None)] <= 3e-5, s   # faithful: the stock arm, a row bitwise to it, or an fp32-class arm
                if d == "fwdbwd":
                    assert s.backward, s


def test_default_rule_a_row_word_serves_exactly_that_row():
    for w in ("v2", "v2:fast", "v1", "v1:lnfused", "pf:fpf", "pf:lnl", "lnl", "af3_fused", "flash_sm90a", "flash_sm90a:kernel_ln", "torch_swiglu", "engine_module", "compile"):
        s = T.select(w, c=256, hidden=1024, n_tokens=800, cc="9.0", stack=H100)
        assert (s.row, s.variant) == (T.split_word(w)[0], T.split_word(w)[1]) and s.tier is None and s.word == w, s
        assert s.cell_key == "9.0|bf16|pair_c256_n4|N<=800|eager|fwd" and s.stack_measured is True, s
    s = T.select("v2@bm64bh64w4s2il1", c=128, hidden=256, cc="8.0", stack=A100)     # an explicit launch word serves without a cell (the caller's pin)
    assert s.row == "v2" and s.cfg == {"BM": 64, "BH": 64, "num_warps": 4, "num_stages": 2, "IL": 1} and s.cell_key is None
    s = T.select("v2", c=128, hidden=512, n_tokens=800, cc="8.0", stack=A100)          # no package configuration off cc 9.0: the measured sweep's best launch, named in the selection
    assert s.row == "v2" and s.cfg is not None and s.stack_measured is True and s.ms is not None, s
    assert T.cfg_word(s.cfg) in {T.split_word(a)[2] for a in T.table()["cells"]["8.0|bf16|pair_c128_n4|N<=800|eager|fwd"]["stacks"][A100]["ms"] if a.startswith("v2@")}
    for w, kw in (("esm_t15", {}), ("esm_t16", {}), ("esm_kd3", {}), ("esm_kd3:lean", {}), ("esm_t16_kd3", dict(direction="fwdbwd")), ("esm_t15_kd3", dict(direction="fwdbwd"))):
        s = T.select(w, c=256, hidden=1024, n_tokens=800, family="esmpair", residual=True, cc="9.0", stack=ESM_H100, **kw)
        assert s.row == T.split_word(w)[0], s


def test_no_word_is_refused_the_provider_never_chooses_for_a_caller_that_did_not_ask():
    with pytest.raises(T.Refusal) as e:
        T.select(None, c=128, hidden=512, n_tokens=800, cc="9.0")
    assert e.value.kind == "no_word" and e.value.fallback == "torch_swiglu"


@pytest.mark.parametrize("word,kw,kind,fallback", [
    ("bogus", dict(c=128, hidden=512), "unknown_row:bogus", "torch_swiglu"),
    ("v2:lean", dict(c=128, hidden=512), "unknown_variant:v2:lean", "torch_swiglu"),
    ("lnl@bm64bh64w4s2il1", dict(c=128, hidden=512), "cfg_word_only_for_v2", "torch_swiglu"),
    ("v2", dict(c=128, hidden=512, dtype="fp32"), "v2:dtype:fp32", "torch_swiglu"),
    ("v2", dict(c=96, hidden=192), "v2:cell_96x192_not_configured", "torch_swiglu"),
    ("v2@bm64bh48w4s2il0", dict(c=128, hidden=512), "v2:hidden_512_not_multiple_of_BH_48", "torch_swiglu"),
    ("v2", dict(c=128, hidden=512, rows_count=2 ** 31), "v2:rows_2147483648_int32_row_index", "torch_swiglu"),
    ("v2", dict(c=96, hidden=384, cc="8.0"), "v2:no_measured_launch_on_cc_8.0_for_96x384", "torch_swiglu"),
    ("lnl", dict(c=384, hidden=1536), "lnl:c384_outside_64/128/256", "torch_swiglu"),
    ("lnl", dict(c=128, hidden=512, direction="fwdbwd"), "forward_only:lnl", "torch_swiglu"),
    ("af3_fused", dict(c=96, hidden=192), "af3_fused:c96_outside_64/128/256/384", "torch_swiglu"),
    ("af3_fused", dict(c=128, hidden=512, mask=True), "row_mask_folded_by_v2_only", "torch_swiglu"),
    ("flash_sm90a", dict(c=128, hidden=512), "flash_sm90a:cell_128x512_not_256x1024", "torch_swiglu"),
    ("flash_sm90a", dict(c=256, hidden=1024, cc="8.0"), "flash_sm90a:cc_8.0_not_9.0", "torch_swiglu"),
    ("flash_sm90a", dict(c=256, hidden=1024, dtype="fp32"), "flash_sm90a:dtype:fp32", "torch_swiglu"),
    ("esm_t15", dict(c=256, hidden=1024, family="pair"), "esm_t15:residual_folded_only", "engine_module"),
    ("esm_t15", dict(c=128, hidden=512, family="esmpair", residual=True), "esm_t15:c128_not_256", "engine_module"),
    ("esm_t15", dict(c=256, hidden=1024, family="esmpair", residual=True, direction="fwdbwd"), "forward_only:esm_t15", "engine_module"),
    ("esm_kd3", dict(c=256, hidden=1024, family="esmpair", residual=True, dtype="fp32"), "esm_kd3:dtype:fp32", "engine_module"),
    ("esm_t16", dict(c=256, hidden=1024, cc="8.0"), "esm_t16:cc_8.0_not_9.0", "engine_module"),
    ("esm_t16_kd3", dict(c=256, hidden=1024, family="esmpair", residual=True, direction="fwdbwd", cc="8.0"), "esm_t16_kd3:cc_8.0_not_9.0", "esm_kd3"),
    ("esm_t16", dict(c=256, hidden=2048), "esm_t16:cell_256x2048_not_256x1024", "engine_module"),
    ("esm_fused_exact", dict(c=256, hidden=1024, family="esmpair", cc="8.6"), "esm_fused_exact:cc_8.6_unmeasured", "engine_module"),
    ("esm_fused_exact", dict(c=128, hidden=512, family="esmpair"), "esm_fused_exact:c128_not_256", "engine_module"),
    ("esm_fused_exact", dict(c=256, hidden=1000, family="esmpair"), "esm_fused_exact:hidden_1000_not_multiple_of_64", "engine_module"),
    ("esm_fused_exact", dict(c=256, hidden=1024, family="esmpair", dtype="fp32"), "esm_fused_exact:dtype:fp32", "engine_module"),
    ("esm_fused_exact", dict(c=256, hidden=1024, family="esmpair", direction="fwdbwd"), "forward_only:esm_fused_exact", "engine_module"),
    ("esm_fused_exact", dict(c=256, hidden=1024, family="esmpair", ln_given=True), "esm_fused_exact:x_ln_given_the_row_carries_its_statements_layernorm", "engine_module"),
    ("esm_fused_exact", dict(c=256, hidden=1024, family="esmpair", mask=True), "row_mask_folded_by_v2_only", "engine_module"),
    ("esm_fused_exact:kernel_ln", dict(c=256, hidden=1024, family="esmpair"), "unknown_variant:esm_fused_exact:kernel_ln", "engine_module"),
    ("compile", dict(c=128, hidden=512, capture=True), "capture_unsafe:compile", "torch_swiglu"),
    ("fast", dict(c=96, hidden=192, n_tokens=800), "no_cell:9.0|bf16|c96_h192_pair|eager|fwd", "torch_swiglu"),
    ("fast", dict(c=128, hidden=512), "no_cell:9.0|bf16|pair_c128_n4|eager|fwd", "torch_swiglu"),
])
def test_refusals_are_by_name_with_the_fallback_row(word, kw, kind, fallback):
    kw = dict(dict(cc="9.0", stack=H100), **kw)
    with pytest.raises(T.Refusal) as e:
        T.select(word, **kw)
    assert e.value.kind == kind, (str(e.value), kind)
    assert e.value.fallback == fallback and e.value.row == word


def test_tier_words_are_opt_in_and_carry_the_measured_numbers():
    s = T.select("fast", c=128, hidden=512, n_tokens=800, cc="9.0", stack=H100)
    cell = T.table()["cells"]["9.0|bf16|pair_c128_n4|N<=800|eager|fwd"]
    assert T._arm_word(s.row, s.variant, T.cfg_word(s.cfg) if s.cfg else None) == cell["stacks"][H100]["fast"] and s.x_stock == cell["stacks"][H100]["x_stock"][cell["stacks"][H100]["fast"]] > 1.0
    s2 = T.select("fast", c=128, hidden=512, n_tokens=800, cc="9.0", stack="H100:torch9.9.9+cu999/9.9.9")     # an unmeasured stack: the reference stack's winner, flagged
    assert s2.stack_measured is False and "reference_stack_numbers" in s2.words and s2.stack == cell["ref_stack"]
    s3 = T.select("fast", c=128, hidden=512, n_tokens=100000, cc="9.0", stack=H100)                            # above the largest measured size: served, flagged
    assert s3.beyond_measured is True and "beyond_measured" in s3.words
    with pytest.raises(T.Refusal) as e:                                                                        # a tier winner that is a kernel other than v2 cannot fold a row mask: v2 by name
        T.select("fast", c=128, hidden=256, n_tokens=800, dtype="fp32", mask=True, cc="9.0", stack=H100)
    assert e.value.kind.startswith("tier_winner_takes_no_mask") and e.value.fallback == "v2", str(e.value)
    s4 = T.select("af3_fused", c=128, hidden=512, n_tokens=800, capture=True, cc="9.0", stack=H100)
    assert "warm_before_capture" in s4.words and s4.capture_safe is True
    s5 = T.select("flash_sm90a", c=256, hidden=1024, n_tokens=800, cc="9.0", stack=H100)
    assert "exact_class_given_the_callers_ln_output" in s5.words
    assert "exact_class_given_the_callers_ln_output" not in T.select("flash_sm90a:kernel_ln", c=256, hidden=1024, n_tokens=800, cc="9.0", stack=H100).words
    s6 = T.select("esm_fused_exact", c=256, hidden=1024, family="esmpair", form="esmfused", cc="9.0", stack=H100)   # the row carries its statement's LayerNorm: exact class without x_ln
    assert s6.row == "esm_fused_exact" and "exact_class_given_the_callers_ln_output" not in s6.words and s6.backward is False and s6.capture_safe is True


def test_k384_cells_say_stock_wins_in_bf16_with_the_measured_numbers():
    """c=384 (hidden 1536): the carried one-kernel rows pad the register tile to 512 columns (a third more K-work), and lose to cuBLAS in bf16 at
    N^2 rows on both cards -- the cell's fast word is a stock row there, with the deficits kept as numbers; the single transition (N rows) and the
    fp32 form have their own measured winners."""
    t = T.table()
    for cc, st in (("9.0", H100), ("8.0", A100)):
        for n in SIZES:
            sc = t["cells"]["%s|bf16|pair_c384_n4|N<=%d|eager|fwd" % (cc, n)]["stacks"][st]
            assert T.split_word(sc["fast"])[0] in T.STOCK_ROWS, (cc, n, sc["fast"])
            losers = {a: x for a, x in sc["x_stock"].items() if T.split_word(a)[0] in ("v2", "af3_fused", "v1", "pf")}
            assert losers and max(losers.values()) < 1.0, (cc, n, losers)
    src = open(os.path.join(CORE_DIR, "opt_core", "kernels", "fpf_transition_v2", "kernel.py"), encoding="utf-8").read()
    assert "cp = triton.next_power_of_2(c)" in src
    assert "CP = triton.next_power_of_2(C)" in open(os.path.join(PKG, "af3", "af3_fused.py"), encoding="utf-8").read()


def test_rows_for_kit_reads_the_table():
    r = T.rows_for_kit(c=128, hidden=512, cc="9.0")
    assert set(r) == set(SIZES) and all(v["cell"].startswith("9.0|bf16|pair_c128_n4|") for v in r.values())
    r = T.rows_for_kit(c=256, hidden=1024, family="esmpair", direction="fwdbwd", cc="9.0")
    assert all(T.split_word(v["fast"])[0] in T.BACKWARD_ROWS for v in r.values())
    assert T.rows_for_kit(c=96, hidden=192)[400] is None


def test_admits_matches_the_row_envelopes():
    ok = lambda *a, **k: T.admits(*a, **k)[0]   # noqa: E731
    assert ok("v2", c=128, hidden=512) and ok("v2", c=64, hidden=128) and ok("v2", c=64, hidden=256) and ok("v2", c=256, hidden=1024) and ok("v2", c=384, hidden=1536)
    assert not ok("v2", c=128, hidden=256) and ok("v2@bm64bh64w4s2il1", c=128, hidden=256) and ok("v2", c=128, hidden=256, cc="8.0")   # (128,256): a launch word on 9.0; on 8.0 the bare word resolves to the sweep's launch in select()
    assert ok("v2", c=128, hidden=512, mask=True) and not ok("v1", c=128, hidden=512, mask=True)
    assert ok("v1", c=128, hidden=512, dtype="fp32") and not ok("v1", c=128, hidden=512, dtype="fp32", residual=True)
    assert ok("af3_fused", c=384, hidden=1536, dtype="fp32") and ok("lnl", c=256, hidden=1024) and not ok("lnl", c=384, hidden=1536)
    assert ok("flash_sm90a", c=256, hidden=1024) and not ok("flash_sm90a", c=256, hidden=1024, cc="8.0")
    assert ok("esm_t15", c=256, hidden=1024, residual=True) and ok("esm_t15", c=256, hidden=1024, residual=True, cc="8.0") and not ok("esm_t15", c=256, hidden=1024, residual=True, cc="7.5")
    assert ok("esm_kd3:lean", c=256, hidden=1024, residual=True, direction="fwdbwd") and ok("esm_t15_kd3", c=256, hidden=1024, residual=True, direction="fwdbwd")
    assert ok("esm_t16", c=256, hidden=1024) and ok("esm_t16", c=256, hidden=1024, residual=True) and not ok("esm_t16", c=256, hidden=1024, cc="8.0")
    assert ok("esm_t16_kd3", c=256, hidden=1024, residual=True, direction="fwdbwd") and ok("esm_t15_kd3", c=256, hidden=1024, residual=True, direction="fwdbwd", cc="8.0")
    assert ok("torch_swiglu", c=77, hidden=13, dtype="fp32", direction="fwdbwd", mask=True, residual=True) and ok("rowpair", c=128, hidden=512)
    assert ok("esm_fused_exact", c=256, hidden=1024, residual=True) and ok("esm_fused_exact", c=256, hidden=1024) and ok("esm_fused_exact", c=256, hidden=2048, residual=True)
    assert ok("esm_fused_exact", c=256, hidden=1024, cc="8.0") and not ok("esm_fused_exact", c=256, hidden=1024, cc="8.6")
    assert not ok("esm_fused_exact", c=384, hidden=1536) and not ok("esm_fused_exact", c=256, hidden=1024, dtype="fp32")
    assert not ok("esm_fused_exact", c=256, hidden=1024, direction="fwdbwd") and not ok("esm_fused_exact", c=256, hidden=1024, ln_given=True) and not ok("esm_fused_exact", c=256, hidden=96)
    assert T.admits("v2", c=128, hidden=512, rows_count=T.V2_ROW_LIMIT - 1)[0] and not T.admits("v2", c=128, hidden=512, rows_count=T.V2_ROW_LIMIT)[0]


def test_row_offsets_are_int64_in_every_carried_kernel_and_the_int32_row_index_is_bounded_by_name():
    """The audit: N^2 rows x c can exceed 2^31 elements (2048^2 x 512 = 2^31).  Every carried Triton kernel forms its row offsets in int64; v2's
    row INDEX is int32 and refused by name at 2^31 - 2^12 rows; the sealed sm_90a kernel addresses rows through TMA coordinates / 64-bit
    pointers (its LayerNorm prologue's row is a 64-bit long)."""
    v2 = open(os.path.join(CORE_DIR, "opt_core", "kernels", "fpf_transition_v2", "kernel.py"), encoding="utf-8").read()
    assert "rm64 = rm.to(tl.int64)" in v2 and 'raise Unsupported("rows"' in v2
    assert T.V2_ROW_LIMIT == 2 ** 31 - 2 ** 12
    t15 = open(os.path.join(PKG, "esm", "ef2_pair_v2.py"), encoding="utf-8").read()
    assert "offs_m = pid.to(tl.int64) * BM + tl.arange(0, BM)" in t15
    af3 = open(os.path.join(PKG, "af3", "af3_fused.py"), encoding="utf-8").read()
    assert "r64 = rows.to(tl.int64)" in af3 and "r64[:, None] * C" in af3
    agk = open(os.path.join(PKG, "esm", "ef2_autograd_kernels.py"), encoding="utf-8").read()
    assert re.search(r"row\s*=\s*tl\.program_id\(0\)\.to\(tl\.int64\)", agk) or "tl.int64" in agk
    lnl = open(os.path.join(CORE_DIR, "opt_core", "kernels", "lnl_fused.py"), encoding="utf-8").read()
    assert "rows[:, None].to(tl.int64) * C" in lnl
    v1 = open(os.path.join(CORE_DIR, "opt_core", "kernels", "fpf_transition", "transition.py"), encoding="utf-8").read()
    assert "tl.int64" in v1
    cu = open(os.path.join(PKG, "flash_sm90a", "csrc", "flash_transition_sm90.cu"), encoding="utf-8").read()
    assert "long row = (long(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;" in cu
    fx = open(os.path.join(PKG, "esm_fused", "__init__.py"), encoding="utf-8").read()
    assert "offs_m = pid.to(tl.int64) * BM + tl.arange(0, BM)" in fx and "tl.atomic" not in fx
    t16 = json.load(open(os.path.join(PKG, "esm", "ef2_t16", "sm_90a", "manifest.json")))
    cub = t16["cubins"]["ef2_transition_cute"]
    import hashlib
    assert hashlib.sha256(open(os.path.join(PKG, "esm", "ef2_t16", "sm_90a", cub["file"]), "rb").read()).hexdigest() == cub["cubin_sha256"]


def test_the_face_is_standard_library_at_import():
    tree = ast.parse(open(os.path.join(PKG, "__init__.py"), encoding="utf-8").read())
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:                                           # the one relative import: the cell-coverage census (itself standard library only)
                assert [a.name for a in node.names] == ["cell_census"], [a.name for a in node.names]
                continue
            top.add((node.module or "").split(".")[0])
    assert top <= {"json", "os", "re"}, top
    import subprocess
    import sys
    # a fresh interpreter (this session may already hold torch for other tests): importing the face pulls in no framework
    r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, %r); import opt_core.kernels.transition as T; "
                        "print(int('torch' in sys.modules), int('triton' in sys.modules), len(T.table()['cells']))" % CORE_DIR], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-400:]
    flags = r.stdout.split()
    assert flags[:2] == ["0", "0"], flags

def test_meta_census_names_the_package_and_its_runtime_imports_exist():
    meta = json.load(open(os.path.join(CORE_DIR, "opt_core", "kernels", "META", "transition.json"), encoding="utf-8"))
    assert meta["name"] == "transition" and meta["kind"] == "package" and re.match(r"^\d+\.\d+\.\d+$", meta["version"])
    for k in ("exports", "license", "mechanism", "not_carried", "runtime_imports", "runtime_imports_note", "served_note"):
        assert k in meta, k
    kdir = os.path.join(CORE_DIR, "opt_core", "kernels")
    for name in meta["runtime_imports"]:
        assert (os.path.isdir(os.path.join(kdir, name)) or os.path.isfile(os.path.join(kdir, name + ".py")) or os.path.isfile(os.path.join(CORE_DIR, "opt_core", "attn", name + ".py"))
                or os.path.isdir(os.path.join(CORE_DIR, "opt_core", "mem", name))), name
    for row, names in T._MODULES.items():
        for n in names:
            rel = n.replace("opt_core.", "").replace(".", os.sep)
            assert os.path.isfile(os.path.join(CORE_DIR, "opt_core", rel + ".py")) or os.path.isfile(os.path.join(CORE_DIR, "opt_core", rel, "__init__.py")), (row, n)
    assert hasattr(kernels, "__path__")


def test_sealed_row_manifest_and_payload_are_present_and_consistent():
    d = os.path.join(PKG, "flash_sm90a")
    keys = sorted(os.listdir(os.path.join(d, "prebuilt")))
    assert keys == ["torch2.13.0-cu130"], keys
    man = json.load(open(os.path.join(d, "prebuilt", keys[0], "manifest.json")))
    for k in ("module_name", "torch", "cuda", "python_tag", "so_sha256", "source_sha256"):
        assert k in man, k
    import hashlib
    so = [f for f in os.listdir(os.path.join(d, "prebuilt", keys[0])) if f.endswith(".so")]
    assert len(so) == 1
    assert hashlib.sha256(open(os.path.join(d, "prebuilt", keys[0], so[0]), "rb").read()).hexdigest() == man["so_sha256"]
    for fn, digest in man["source_sha256"].items():
        assert hashlib.sha256(open(os.path.join(d, "csrc", fn), "rb").read()).hexdigest() == digest, fn
    for lic in ("LICENSE.cutlass", "LICENSE.flash-attention", "LICENSE_NOTE.md"):
        assert os.path.isfile(os.path.join(d, lic)), lic
    assert os.path.isfile(os.path.join(PKG, "NOTICE.md"))


def _same_tree(a, b):
    cmp = filecmp.dircmp(a, b, ignore=["__pycache__"])
    if cmp.left_only or cmp.right_only or cmp.diff_files or cmp.funny_files:
        return False
    (_, mismatch, errors) = filecmp.cmpfiles(a, b, cmp.common_files, shallow=False)
    if mismatch or errors:
        return False
    return all(_same_tree(os.path.join(a, s), os.path.join(b, s)) for s in cmp.common_dirs)


CARRIED = ["flash_sm90a", "esm/ef2_pair_v2.py", "esm/ef2_autograd_kernels.py", "esm/ef2_t16_transition.py", "esm/ef2_t16_nvjit.py", "esm/ef2_t16", "af3/af3_fused.py"]
SEALED_UNIT = "fpf_flash_transition"      # flash_sm90a's kit directory carries the unit under its producer's package name: found by its manifest's module name


def _kit_copies(carried):
    """Every copy of a carried file / directory in the release tree outside the core, found by name (no engine is named here)."""
    core = os.path.join(RELEASE_TREE, "common")
    hits = []
    if carried == "flash_sm90a":
        for root, dirs, files in os.walk(RELEASE_TREE):
            dirs[:] = [x for x in dirs if x not in ("__pycache__", ".git")]
            if root.startswith(core):
                continue
            if "LICENSE_NOTE.md" in files and os.path.isdir(os.path.join(root, "csrc")) and any(f.startswith("flash_transition") for f in os.listdir(os.path.join(root, "csrc"))):
                hits.append(root)
        return hits
    name = os.path.basename(carried)
    for root, dirs, files in os.walk(RELEASE_TREE):
        dirs[:] = [x for x in dirs if x not in ("__pycache__", ".git")]
        if root.startswith(core):
            continue
        if name.endswith(".py"):
            if name in files:
                hits.append(os.path.join(root, name))
        elif name in dirs and os.path.isfile(os.path.join(root, name, "PROVENANCE.md")):
            hits.append(os.path.join(root, name))
    return hits


def _digest(path):
    """sha256 of a file; 'tree:<sha256>' over (relative path, file digest) pairs of a directory (no __pycache__)."""
    if os.path.isfile(path):
        return hashlib.sha256(open(path, "rb").read()).hexdigest()
    entries = []
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")          # pruned BEFORE descending: interpreter caches never enter the digest
        for f in files:
            if f.endswith((".pyc", ".pyo")):
                continue
            fp = os.path.join(root, f)
            entries.append((os.path.relpath(fp, path), hashlib.sha256(open(fp, "rb").read()).digest()))
    h = hashlib.sha256()
    for rel, d in sorted(entries):
        h.update(rel.encode()); h.update(b"\0"); h.update(d)
    return "tree:" + h.hexdigest()


@pytest.mark.parametrize("carried", CARRIED)
def test_carried_rows_are_byte_identical_to_their_kit_copies(carried):
    """The carried file / directory has the digest META records for it (the core's own consistency: always checked), and a kit copy found by NAME anywhere in
    the release tree with the SAME digest witnesses the byte identity.  Kit-side moves and edits never fail this gate: no copy in the checkout, or only
    copies with another digest (the kit edited its file after the carry), SKIP with the reason named -- the provider serves the carried bytes either way."""
    meta = json.load(open(os.path.join(CORE_DIR, "opt_core", "kernels", "META", "transition.json"), encoding="utf-8"))
    a = os.path.join(PKG, carried)
    assert os.path.exists(a), a
    recorded = meta["carried_digests"].get(carried)
    mine = _digest(a)
    assert recorded == mine, (carried, recorded, mine)           # the carried copy changed without its META digest: re-carry = copy + record
    copies = _kit_copies(carried)
    if not copies:
        pytest.skip("no kit copy of %s in this checkout (byte identity as recorded in META carried_digests)" % carried)
    same = [b for b in copies if _digest(b) == mine]
    if not same:
        pytest.skip("kit copies of %s differ from the carried digest %s..: %s -- the kit edited its file after the carry; the provider serves the carried bytes until re-carried"
                    % (carried, mine[:16], ", ".join(os.path.relpath(b, RELEASE_TREE) for b in copies)))
    for b in same:
        if os.path.isdir(a):
            assert _same_tree(a, b), (carried, os.path.relpath(b, RELEASE_TREE))
        else:
            assert filecmp.cmp(a, b, shallow=False), (carried, os.path.relpath(b, RELEASE_TREE))



OF3_STACK = "H100:torch2.10.0+cu128/3.6.0"


def test_liger_form_cells_resolve_by_form_and_the_plain_word_is_refused_under_the_other_form():
    t = T.table()
    liger = [k for k in t["cells"] if "+liger|" in k]
    assert liger, "no '+liger' cells"
    for key in liger:
        cc, dt, cw, n, timing, d = key.split("|")
        assert cc in ("9.0", "8.0", "10.0", "10.3") and dt == "bf16" and d == "fwd", key   # the engine's own images (H100 / A100) + the new parts' columns
    for cw, fam in (("pair_c128_n4", "pair"), ("pair_c128_n2", "pair"), ("rows_c64_n4", "rows")):
        c, hidden = DIMS[cw]
        assert T.cell_word(c, hidden, fam, "liger") == cw + "+liger" and T.cell_word(c, hidden, fam) == cw
        for n in SIZES:
            key, size, beyond = T.cell_key("9.0", "bf16", cw + "+liger", n, "eager", "fwd")
            assert key is not None and size == n and not beyond, (cw, n)
            cell = t["cells"][key]
            st = cell["ref_stack"]
            s = T.select("exact", c=c, hidden=hidden, n_tokens=n, cc="9.0", family=fam, form="liger", stack=st, ln_given=True)
            if cw.startswith("pair_c128"):
                assert (s.row, s.variant) == ("v1", "liger") and s.cls == "bitwise" and s.x_stock >= 1.0, s     # EXACT FLOOR x1.00 under the Liger form: v1:liger bitwise and not slower
            else:
                assert s.row in T.STOCK_ROWS, s                                                                  # (64, 256): v1 has no launch cell on that stack -> the stock arm by name
            f = T.select("fast", c=c, hidden=hidden, n_tokens=n, cc="9.0", family=fam, form="liger", stack=st)
            assert f.row in T.ROW_NAMES and f.ms is not None, f
            # the plain statement's exact word on the same shape stays what it was (v1 or the stock arm): the Liger cells do not touch the plain cells
            p = T.select("exact", c=c, hidden=hidden, n_tokens=n, cc="9.0", family=fam, stack=H100, ln_given=True)
            assert p.variant != "liger", p
        with pytest.raises(T.Refusal) as e:
            T.select("v1", c=c, hidden=hidden, n_tokens=400, cc="9.0", family=fam, form="liger")
        assert e.value.kind.startswith("v1:form_liger_needs_v1:liger"), e.value.kind
        with pytest.raises(T.Refusal) as e:
            T.select("v1:liger", c=c, hidden=hidden, n_tokens=400, cc="9.0", family=fam)
        assert e.value.kind.startswith("v1:form_swiglu_needs_v1"), e.value.kind
        s = T.select("v1:liger", c=c, hidden=hidden, n_tokens=400, cc="9.0", family=fam, form="liger", ln_given=True)      # the row word itself resolves (serving refuses by name where v1 has no launch cell)
        assert (s.row, s.variant) == ("v1", "liger") and s.cell_key.split("|")[2] == cw + "+liger", s
    with pytest.raises(T.Refusal) as e:
        T.select("fast", c=128, hidden=512, n_tokens=400, cc="9.0", form="gelu")
    assert e.value.kind.startswith("unknown_form"), e.value.kind
    assert T.cell_word(256, 1024, "pair", "liger") is None and T.cell_word(768, 3072, "single", "liger") is None and T.cell_word(384, 1536, "single", "liger") == "single_c384_n4+liger"    # shapes without Liger cells: row words only
    ok, why = T.admits("v1:liger", c=128, hidden=512)
    assert ok, why


ESM_FUSED_PARTS = {"ef2_pair_v2.py": ("_ld_chunk", "_bfly", "_warp_tree", "_stock_tree_sum", "_stock_stats"), "ef2_w4.py": ("_xhat_chunk",)}


def _function_text(src, name):
    """The exact source lines of the top-level function `name` (its decorator line included) or None."""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
            return "".join(lines[start:node.end_lineno])
    return None


def test_esm_fused_carries_the_kit_parts_byte_for_byte_and_states_its_arithmetic():
    """Row esm_fused_exact reuses six Triton device functions of the ESM-family inference kit (the fork's statistics tree and its bf16 x_hat
    chain): their texts inside esm_fused/__init__.py equal the kit files' (the carried esm/ef2_pair_v2.py for five of them, the kit's ef2_w4.py
    found by name in the release tree for _xhat_chunk); the kernel states the round-then-add epilogue and the chained W3 accumulation."""
    fx = open(os.path.join(PKG, "esm_fused", "__init__.py"), encoding="utf-8").read()
    for name in ESM_FUSED_PARTS["ef2_pair_v2.py"]:
        mine = _function_text(fx, name)
        theirs = _function_text(open(os.path.join(PKG, "esm", "ef2_pair_v2.py"), encoding="utf-8").read(), name)
        assert mine is not None and theirs is not None and mine == theirs, name
    copies = _kit_copies("ef2_w4.py")
    mine = _function_text(fx, "_xhat_chunk")
    assert mine is not None and "x_hat = ((x - mean[:, None]) * rstd[:, None]) * ln_w[None, :] + ln_b[None, :]" in mine
    if not copies:
        pytest.skip("no kit copy of ef2_w4.py in this checkout")
    for b in copies:
        theirs = _function_text(open(b, encoding="utf-8").read(), "_xhat_chunk")
        if theirs is not None:
            assert mine == theirs, os.path.relpath(b, RELEASE_TREE)
    assert "mean, rstd = _stock_stats(x0, x1, x2, x3, eps, 0)" in fx                      # TREE 0: the association fixed by code (no layout-dependent reduction)
    assert "y = acc.to(tl.bfloat16)" in fx and "o = xres + y" in fx                      # round-then-add: the statement's addmm epilogue
    assert "acc = tl.dot(h, w3t, acc)" in fx and "h = swiglu.to(tl.bfloat16)" in fx        # chained W3 accumulation over ascending hidden chunks; one SwiGLU rounding
    assert T._MODULES["esm_fused_exact"] == ("opt_core.kernels.transition.esm_fused",)
    r = T.rows()["esm_fused_exact"]
    assert r["class"] == "exact" and r["backward"] is False and r["capture_safe"] is True and r["fallback"] == "engine_module"


def test_esmfused_form_cells_resolve_by_form_and_the_exact_tier_is_the_bitwise_one_kernel_row():
    """'+esmfused' cells (the ESM-family fork's fused inference statement, cc 9.0): select(form='esmfused') resolves them; the exact tier is
    esm_fused_exact (bitwise, EXACT FLOOR x1.00), the fast tier a row word not slower than the statement; the row word serves by name under the
    plain form too (its class is a property of the '+esmfused' cells only); a card without such cells refuses tier words by name (no_cell), a
    card without a measured launch row refuses the row word by name."""
    t = T.table()
    keys = [k for k in t["cells"] if "+esmfused|" in k]
    assert keys, "no '+esmfused' cells"
    ccs = sorted({k.split("|")[0] for k in keys})
    assert "9.0" in ccs and set(ccs) <= set(T.ESM_FUSED_CCS), ccs
    for key in keys:
        cc, dt, cw, n, timing, d = key.split("|")
        assert (dt, cw, d) == ("bf16", "esmpair_c256_n4+esmfused", "fwd"), key
        cell = t["cells"][key]; st = cell["ref_stack"]; size = int(n[3:])
        assert T.cell_word(256, 1024, "esmpair", "esmfused") == cw and T.cell_word(256, 1024, "esmpair") == "esmpair_c256_n4"
        s = T.select("exact", c=256, hidden=1024, n_tokens=size, cc=cc, family="esmpair", form="esmfused", timing=timing, stack=st)
        sc = cell["stacks"][st]
        if sc["class"].get("esm_fused_exact") == "bitwise" and sc["ms"]["esm_fused_exact"] <= sc["stock_ms"] and not sc.get("exact_withheld"):
            assert s.row == "esm_fused_exact" and s.cls == "bitwise" and s.x_stock >= 1.0 and not s.beyond_measured, (key, s)     # EXACT FLOOR x1.00, bitwise
        else:                                                                                                                        # not bitwise at every size on the card (the
            assert s.row != "esm_fused_exact" and cell["exact"] == sc["stock_arm"], (key, s)                                         # sweep's exceptions recorded), or slower:
            if sc.get("exact_withheld"):                                                                                             # the exact word stays the stock arm by name,
                assert cell["faithful"] == "esm_fused_exact" and "sweep" in sc["exact_withheld"], key                                # the row is the faithful word
        f = T.select("fast", c=256, hidden=1024, n_tokens=size, cc=cc, family="esmpair", form="esmfused", timing=timing, stack=st)
        assert f.row in T.ROW_NAMES and f.x_stock >= 1.0, (key, f)
        assert sc["stock_arm"] == "engine_module", key
    for cc in ccs:
        w = T.select("esm_fused_exact", c=256, hidden=1024, n_tokens=800, cc=cc, family="esmpair")               # plain form: the row word by name, on the esmpair cells
        assert w.row == "esm_fused_exact" and (w.cell_key is None or "+esmfused" not in w.cell_key)
    u = T.select("exact", c=256, hidden=1024, n_tokens=800, cc="12.0", family="esmpair", form="esmfused")      # an unmeasured capability: the statement by cell (inherited column), no vouch there
    assert u.row in T.STOCK_ROWS and "inherited_cc:unmeasured" in u.words and "exact_unvouched_on_cc_12.0" in u.words, (u.row, u.words)
    with pytest.raises(T.Refusal) as e:
        T.select("esm_fused_exact", c=256, hidden=1024, n_tokens=800, cc="8.6", family="esmpair", form="esmfused")
    assert e.value.kind == "esm_fused_exact:cc_8.6_unmeasured" and e.value.fallback == "engine_module", (e.value.kind, e.value.fallback)


def test_split_and_cfg_words_round_trip():
    assert T.split_word("v2:fast@bm64bh64w4s2il1") == ("v2", "fast", "bm64bh64w4s2il1")
    assert T.split_word("pf:lnl") == ("pf", "lnl", None) and T.split_word("lnl") == ("lnl", None, None)
    cfg = T.cfg_of("bm128bh32w8s3il0")
    assert cfg == {"BM": 128, "BH": 32, "num_warps": 8, "num_stages": 3, "IL": 0} and T.cfg_word(cfg) == "bm128bh32w8s3il0"
    with pytest.raises(T.Refusal):
        T.cfg_of("BM64_BH64")
    assert T.cell_word(128, 512) == "pair_c128_n4" and T.cell_word(128, 256) == "pair_c128_n2" and T.cell_word(64, 256) == "rows_c64_n4"
    assert T.cell_word(384, 1536, "single") == "single_c384_n4" and T.cell_word(384, 1536, "pair") == "pair_c384_n4" and T.cell_word(256, 1024, "esmpair") == "esmpair_c256_n4"
    assert T.cell_word(256, 1024, "esmpair", "esmfused") == "esmpair_c256_n4+esmfused" and T.cell_word(128, 512, "pair", "esmfused") is None
    assert T.cell_word(96, 192) is None and T.cell_word(64, 96) is None
    assert T.cell_word(64, 128) == "rows_c64_n2" and T.cell_dims("rows_c64_n2") == (64, 128)                            # the template pair transition (n 2)
    assert T.cell_word(768, 3072, "single") == "single_c768_n4" and T.cell_dims("single_c768_n4") == (768, 3072)      # the wide single-track family (stock wins by measurement)
    assert T.admits("v1", c=768, hidden=3072, dtype="bf16", cc="9.0") == (False, "v1:c768_outside_64/128/256/384")     # measured x0.02-0.03 there: refused by name
    assert T.admits("v1", c=128, hidden=512, dtype="bf16", cc="9.0")[0] and T.admits("pf:fpf", c=384, hidden=1536, dtype="bf16", cc="9.0")[0]


def test_the_esm_subpackage_names_its_sibling_bindings():
    src = open(os.path.join(PKG, "esm", "__init__.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    names = [n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SIBLING_NAMES"]
    assert names and [e.value for e in names[0].value.elts] == ["ef2_t16_nvjit", "ef2_t16_transition"]
    for n in ("ef2_t16_nvjit", "ef2_t16_transition", "ef2_autograd_kernels", "ef2_pair_v2"):
        assert os.path.isfile(os.path.join(PKG, "esm", n + ".py")), n
    agk = open(os.path.join(PKG, "esm", "ef2_autograd_kernels.py"), encoding="utf-8").read()
    assert "import ef2_t16_transition as _t16" in agk                       # the top-level sibling import bind_names() answers
    t16 = open(os.path.join(PKG, "esm", "ef2_t16_transition.py"), encoding="utf-8").read()
    assert "import ef2_t16_nvjit as ef2_nvjit" in t16


def test_af3_fused_launch_pins_cover_every_measured_af3_cell_on_cc90_and_the_face_reads_them():
    """Row af3_fused launches ONE pinned tile per (cc, dtype, rows bucket, c, hidden, residual) -- no candidate timing at run time (launch_pins); every cc 9.0
    cell whose tier winner is af3_fused has a pin for its rows bucket (an unpinned key is never a tier winner); the pinned tile is the one the carried autotuner picked (same bytes, same ms); a card
    without pins (cc 8.0 today) returns None = the carried module's own launch, named in the rule."""
    t = T.table()
    pins = t["launch_pins"]["af3_fused"]
    assert len(pins) >= 30 and t["launch_pins_rule"] and T.AF3_AUTOTUNE_ENV in t["launch_pins_rule"]
    cards = sorted({k.split("|")[0] for k in pins})
    assert "9.0" in cards and set(cards) <= {"9.0", "8.0"}, cards
    for key, pin in pins.items():
        cc, dt, mb, c, h, res = key.split("|")
        assert cc in cards and dt in ("bf16", "fp32") and mb in ("MB0", "MB1", "MB2") and res in ("res0", "res1"), key
        for k in ("BM", "BH", "num_warps", "num_stages", "kernel"):
            assert k in pin, (key, k)
        assert pin["kernel"] == ("pow2" if int(c[1:]) <= 128 else "pad"), key                 # the carried module's own kernel choice (pow2 C <= 128, else the padded kernel)
        if str(pin.get("pin_source", "")).startswith("kit"):                                  # a kit's recorded / derived launch table entry imported read-only (provenance named)
            assert all(isinstance(pin[k], int) for k in ("BM", "BH", "num_warps", "num_stages")), (key, pin)
            continue
        assert pin["pinned_ms"] <= pin["tuned_ms"] * 1.10 + 0.01, (key, pin)                  # the pin is never slower than the autotuned launch in the same process
        assert pin["same_bytes_as_tuned"] is True or str(pin.get("pin_source", "")).startswith("sweep"), (key, pin)
    n_checked = 0
    for key, cell in t["cells"].items():
        cc, dt, cw, nb, timing, d = key.split("|")
        if cc not in cards or d != "fwd":
            continue
        arms = set()
        for sc in cell["stacks"].values():                                              # every cell where af3_fused is a TIER WINNER on some stack needs the pin (tier words never
            arms |= {sc[w] for w in ("fast", "faithful", "big") if sc.get(w) and sc[w].split(":")[0] == "af3_fused"}   # autotune); an unpinned key stays selectable by
        if not arms:                                                                    # the row word through the carried launch (launch_pins_rule names it) and the builder
            continue                                                                    # never lets it win a tier
        c, hidden = T.cell_dims(cw)
        n = int(nb[3:]); rows = n if cw.startswith("single") else n * n
        for arm in arms:
            res = arm.endswith(":res") or cw.startswith("esmpair")
            pin = T.af3_pin(cc, dt, rows, c, hidden, res)
            assert pin is not None, (key, arm)
            n_checked += 1
    assert n_checked >= 40 * len(cards), (n_checked, cards)
    assert T.af3_pin("10.0", "bf16", 640000, 128, 512, False) is None                     # a card without recorded pins: the carried launch there (launch_pins_rule names it)
    assert T.af3_rows_bucket(65535) == 0 and T.af3_rows_bucket(65536) == 1 and T.af3_rows_bucket(399999) == 1 and T.af3_rows_bucket(400000) == 2


def test_first_call_records_name_every_forward_row_on_the_reference_stack_and_pinning_removed_the_candidate_timing():
    t = T.table()
    fc = t["first_call"]
    assert t["first_call_rule"]
    ref = fc[H100]
    rows_seen = {k.split("|")[0].split("^")[0].split(":")[0].split("@")[0] for k in ref}
    for row in ("v2", "v1", "pf", "lnl", "af3_fused", "flash_sm90a", "esm_t16"):
        assert row in rows_seen, (row, sorted(rows_seen))
    for key, rec in ref.items():
        if "refused" in rec or "error" in rec:
            continue
        for k in ("cold_first_call_s", "warm_first_call_s", "steady_ms", "what"):
            assert rec.get(k) is not None, (key, k)
        assert rec["warm_first_call_s"] <= rec["cold_first_call_s"] + 0.5, (key, rec)         # the disk cache never costs more than a cold compile
        row = key.split("|")[0]
        if row in ("flash_sm90a", "flash_sm90a:kernel_ln", "esm_t16"):
            assert (rec["triton_cache_files_compiled_cold"] or 0) == 0 and rec["cold_first_call_s"] < 3.0, (key, rec)   # prebuilt binaries: nothing compiles
    # af3_fused: pinned (after) vs the carried autotuned launch (before, the bench env word) on the same cell
    for cellw, dt in (("pair_c128_n4", "bf16"), ("pair_c256_n4", "bf16"), ("pair_c128_n4", "fp32"), ("pair_c384_n4", "fp32"), ("rows_c64_n4", "bf16")):
        a = ref["af3_fused|%s|%s|fwd" % (cellw, dt)]; b = ref["af3_fused^tune|%s|%s|fwd" % (cellw, dt)]
        assert a["served_via"].startswith("pin:") and b["served_via"] == "autotune", (cellw, a.get("served_via"), b.get("served_via"))
        assert a["cold_first_call_s"] < b["cold_first_call_s"] / 2.5 and a["warm_first_call_s"] < b["warm_first_call_s"], (cellw, dt, a, b)
        assert (a["triton_cache_files_compiled_cold"] or 99) <= 12 < (b["triton_cache_files_compiled_cold"] or 0), (cellw, a, b)      # one variant compiled instead of 6-7
        assert abs(a["steady_ms"] - b["steady_ms"]) <= 0.05 * b["steady_ms"] + 0.02, (cellw, dt, a["steady_ms"], b["steady_ms"])   # the same tile serves: same steady speed


def test_the_exact_tier_word_is_stack_and_size_gated_and_refuses_by_name_to_the_module_floor_elsewhere():
    """An EXACT vouch (bitwise vs the statement) holds only on the stack and from the sizes it was measured on: every cell whose exact arm is a kernel row carries
    vouched_on = {stack: arm} (measured bitwise there at the cell's size) and the family's coverage carries exact_floor = {stack: smallest vouched n_tokens};
    select('exact') serves the kernel arm only on a vouched stack at n_tokens >= the floor, else it refuses BY NAME with the module floor (torch_swiglu |
    engine_module) as the fallback; fast / faithful / big are tolerance-class words and resolve as before."""
    t = T.table()
    assert t["exact_vouch_rule"]
    n_v = 0
    for key, cell in t["cells"].items():
        cc, dt, cw, nb, timing, d = key.split("|")
        v = cell.get("vouched_on", {})
        fam_key = "%s|%s|%s|%s|%s" % (cc, dt, cw, timing, d)
        for st, arm in v.items():
            assert T.split_word(arm)[0] not in T.STOCK_ROWS, (key, st, arm)
            if st in cell["stacks"]:                                                # measured (timed) on that stack: its own exact winner, class bitwise
                assert cell["stacks"][st].get("exact") == arm and str(cell["stacks"][st]["class"][arm]).startswith("bitwise"), (key, st, arm)
            else:                                                                   # vouch-only stack: a size sweep recorded the reference stack's exact arm bitwise at this size
                assert arm == cell["stacks"][cell["ref_stack"]].get("exact"), (key, st, arm)
                assert t["coverage"][fam_key]["exact_vouch_sizes"][st][str(int(nb[3:]))] is True, (key, st)
            n_v += 1
        for st, sc in cell["stacks"].items():                                    # and every kernel exact winner measured bitwise IS vouched (nothing silently dropped)
            e = sc.get("exact")
            if e and T.split_word(e)[0] not in T.STOCK_ROWS and str(sc["class"].get(e, "")).startswith("bitwise"):
                assert v.get(st) == e, (key, st, e)
        if v:
            c, hidden = T.cell_dims(cw)
            fam = "pair" if cw.startswith("pair") else "single" if cw.startswith("single") else "rows" if cw.startswith("rows") else "esmpair"
            form = cw.split("+", 1)[1] if "+" in cw else "swiglu"                 # the statement form is the cell word's suffix (liger | esmfused | ...)
            n = int(nb[3:])
            for st, arm in v.items():
                lg = T.split_word(arm)[0] in T.EXACT_GIVEN_LN_ROWS                # rows whose exact class needs the caller's LayerNorm output get it; rows carrying their own LN refuse it
                floor = T.exact_floor(key, st)
                assert floor is not None and floor <= n, (key, st, floor)
                band = t.get("exact_row_bands", {}).get(st)                        # a banded stack: the vouched cell serves only when the row count's remainder is in band
                if band and T.split_word(arm)[0] in band["rows"] and dt == band["dtype"]:
                    rows_here = n * n if not cw.split("+")[0].startswith("single") else n
                    if not (band["min_remainder"] <= rows_here % band["modulus"] <= band["max_remainder"]):
                        with pytest.raises(T.Refusal) as eb:
                            T.select("exact", c=c, hidden=hidden, n_tokens=n, dtype=dt, timing=timing, direction=d, family=fam, cc=cc, stack=st, form=form, ln_given=lg)
                        assert "remainder" in eb.value.kind, (key, st, eb.value.kind)
                        continue
                s = T.select("exact", c=c, hidden=hidden, n_tokens=n, dtype=dt, timing=timing, direction=d, family=fam, cc=cc, stack=st, form=form, ln_given=lg)
                assert s.row + ((":" + s.variant) if s.variant else "") == arm.split("@")[0], (key, st, arm, s.as_dict())
                assert s.stack_measured is (st in cell["stacks"]) and (s.stack_measured or "reference_stack_numbers" in s.words), (key, st, s.as_dict())
                with pytest.raises(T.Refusal) as r:                                # the same request on a stack the vouch was not recorded on
                    T.select("exact", c=c, hidden=hidden, n_tokens=n, dtype=dt, timing=timing, direction=d, family=fam, cc=cc, stack=st.split(":")[0] + ":torch9.9.9+cu999/9.9.9", form=form, ln_given=lg)
                assert r.value.kind.startswith("exact_vouch_not_recorded_on_") and r.value.fallback in ("torch_swiglu", "engine_module"), (key, r.value.kind, r.value.fallback)
                if floor > 20:                                                     # below the vouched floor on the vouched stack
                    kb, _, _ = T.cell_key(cc, dt, cw, floor - 1, timing, d)
                    if kb == key or t["cells"][kb].get("vouched_on", {}).get(st):
                        with pytest.raises(T.Refusal) as r2:
                            T.select("exact", c=c, hidden=hidden, n_tokens=floor - 1, dtype=dt, timing=timing, direction=d, family=fam, cc=cc, stack=st, form=form, ln_given=lg)
                        assert r2.value.kind.startswith("exact_vouch_below_%d_tokens_on_" % floor) or (st in t.get("exact_row_bands", {}) and "remainder" in r2.value.kind), (key, r2.value.kind)
                # the tolerance-class words are unaffected on the unvouched stack
                f = T.select("fast", c=c, hidden=hidden, n_tokens=n, dtype=dt, timing=timing, direction=d, family=fam, cc=cc, stack=st.split(":")[0] + ":torch9.9.9+cu999/9.9.9", form=form)
                assert f.row in T.ROW_NAMES
    assert n_v >= 40, n_v
    # the record on the AF3-family torch 2.10.0 / triton 3.6.0 A100 stack (cc 8.0 bf16 pair c128 n2 / n4, both statement forms): v1 / v1:liger are bitwise vs the
    # statement from 64 tokens up (size sweep 20 ... 1200) and NOT below (cuBLAS takes another GEMM at <= 48^2 rows) -> served from 64, refused by name below
    OF3A = "A100:torch2.10.0+cu128/3.6.0"
    for hidden in (256, 512):
        for form, fl in (("swiglu", False), ("liger", True)):
            for n in (20, 48, 63):
                with pytest.raises(T.Refusal) as r:
                    T.select("exact", c=128, hidden=hidden, n_tokens=n, cc="8.0", stack=OF3A, form=form, ln_given=fl)
                assert r.value.kind == "exact_vouch_below_64_tokens_on_" + OF3A and r.value.fallback == "torch_swiglu", (hidden, form, n, r.value.kind)
            for n in (64, 199, 400, 1200):
                s = T.select("exact", c=128, hidden=hidden, n_tokens=n, cc="8.0", stack=OF3A, form=form, ln_given=fl)
                assert (s.row, s.variant) == (("v1", "liger") if form == "liger" else ("v1", None)), (hidden, form, n, s.as_dict())
    # an AF3-family stack nobody measured stays refused by name at every size
    with pytest.raises(T.Refusal) as r:
        T.select("exact", c=128, hidden=512, n_tokens=800, cc="8.0", stack="A100:torch2.12.0+cu128/3.6.0")
    assert r.value.kind.startswith("exact_vouch_not_recorded_on_") and r.value.fallback == "torch_swiglu", r.value.kind
    # cc 9.0 (H100 torch 2.13.0 / triton 3.7.1): v1 (c128, c256) and flash_sm90a (c256) bitwise at every size 20 ... 1200 -> floor 20
    H = "H100:torch2.13.0+cu130/3.7.1"
    assert T.exact_floor("9.0|bf16|pair_c128_n4|N<=400|eager|fwd", H) == 20 and T.exact_floor("9.0|bf16|pair_c256_n4|N<=400|graph|fwd", H) == 20
    assert T.select("exact", c=128, hidden=512, n_tokens=257, cc="9.0", stack=H, ln_given=True).row == "v1"            # cell N<=400 (the N<=256 cell's exact word is the stock arm by its own numbers)
    assert T.select("exact", c=256, hidden=1024, n_tokens=257, cc="9.0", stack=H, ln_given=True).row == "flash_sm90a"


def test_every_operand_pack_is_cached_on_its_weights_object_never_by_shape():
    """MULTI-WEIGHT contract (CPU part): the face keeps every kernel operand pack / descriptor on the Weights instance it was built from (Weights.pack_for),
    so two weight sets of identical shapes can never share a pack; launch_pins carry tile words only (no operand); the face module holds no module-level
    dict of packs.  The GPU replay (A, B, A, C interleaved per row, bitwise / class checks) is recorded in the notes."""
    import ast as _ast
    src = open(os.path.join(PKG, "__init__.py"), encoding="utf-8").read()
    tree = _ast.parse(src)
    # 1. every pack the serving functions build goes through W.pack_for(...)
    serve_fns = [n for n in tree.body if isinstance(n, _ast.FunctionDef) and n.name.startswith("_serve_")]
    assert len(serve_fns) >= 6
    for fn in serve_fns:
        calls = [c for c in _ast.walk(fn) if isinstance(c, _ast.Call) and isinstance(c.func, _ast.Attribute)]
        builds = [c for c in calls if c.func.attr in ("pack_weights", "pack_transition_weights", "_pack_tensors", "_pack")]
        for b in builds:                                       # a raw pack builder inside a serving function must sit inside a pack_for(lambda: ...) argument
            inside = any(isinstance(c, _ast.Call) and isinstance(c.func, _ast.Attribute) and c.func.attr == "pack_for" and any(b in list(_ast.walk(arg)) for arg in c.args)
                         for c in calls)
            assert inside, (fn.name, b.func.attr)
    # 2. no module-level mutable dict that could hold operand packs keyed by shape (the flash EXTENSION-module cache by ABI key is the one allowed dict)
    top_dicts = [t.id for n in tree.body if isinstance(n, _ast.Assign) and isinstance(n.value, _ast.Dict) for t in n.targets if isinstance(t, _ast.Name)]
    assert set(top_dicts) <= {"_FLASH_EXT", "_MODULES", "LAST", "CELL_DIMS", "_PACK_LRU", "_RESIDENCY", "_PACK_STATS"} | {d for d in top_dicts if d.isupper() and "PACK" not in d and "CACHE" not in d}, top_dicts
    top_sets = sorted(n.targets[0].id for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and getattr(n.value.func, "id", "") == "set" and isinstance(n.targets[0], ast.Name))
    assert set(top_sets) <= {"_DEAD_INHERITED", "_PRIMED_ROWS", "_AUTO_PRIMED"}, top_sets      # process bookkeeping sets: dead inherited rows, primed rows, auto-primed classes (no tensors)
    # (_PACK_LRU is the big tier's residency store: keyed by the weight set's identity and holding the set itself -- never by shape; _PACK_STATS is bookkeeping)
    # 3. pack_for is per instance
    class _FakeT:                                              # a duck tensor for the standard-library face (no torch here)
        def __init__(self, shape): self.shape = shape; self.device = "cpu"
    a = T.Weights.__new__(T.Weights); a._packs = {}
    b = T.Weights.__new__(T.Weights); b._packs = {}
    n = {"k": 0}
    def builder():
        n["k"] += 1
        return object()
    pa1 = a.pack_for("v2", builder); pa2 = a.pack_for("v2", builder); pb = b.pack_for("v2", builder)
    assert pa1 is pa2 and pa1 is not pb and n["k"] == 2
    # 4. launch pins name tiles, never operands
    for key, pin in T.table()["launch_pins"]["af3_fused"].items():
        assert set(pin) <= {"BM", "BH", "num_warps", "num_stages", "kernel", "tuned_ms", "pinned_ms", "same_bytes_as_tuned", "autotune_first_call_s", "tuned_cfg", "pin_source", "sweep_ms"}, (key, sorted(pin))


def test_af3_fused_never_autotunes_at_call_time_unpinned_keys_refuse_by_name():
    """Row af3_fused launches only pinned tiles: every (card, dtype, c, hidden, residual, rows bucket) the face ADMITS has a launch pin, a key without one is
    refused by name in admits() and at serving (af3_fused:no_launch_pin:<key>, fallback row) -- the carried module's autotuned launch is reachable only under
    the bench env word."""
    import ast as _ast
    src = open(T.__file__, encoding="utf-8").read()
    fn = src[src.index("def _serve_af3("):src.index("\ndef ", src.index("def _serve_af3(") + 10)]
    assert fn.count("A.fused_transition(") == 1 and fn.index("A.fused_transition(") > fn.index("AF3_AUTOTUNE_ENV") and "no_launch_pin" in fn, "autotuned launch only under the env word"
    assert fn.index("A.fused_transition(") < fn.index("pin = af3_pin("), "the env-word branch returns before the pin lookup; no other autotune path"
    pins = T.table()["launch_pins"]["af3_fused"]
    cards = sorted({k.split("|")[0] for k in pins})
    n_ok = n_ref = 0
    for cc in cards + ["10.0"]:
        for dt in ("bf16", "fp32", "fp16"):
            for c in (64, 96, 128, 256, 384, 768):
                for f in (2, 4):
                    for res in (False, True):
                        for rows in (4096, 65536, 160000, 399999, 400000, 1440000, 4194304):
                            ok, why = T.admits("af3_fused", c=c, hidden=c * f, dtype=dt, cc=cc, residual=res, rows_count=rows)
                            pin = T.af3_pin(cc, dt, rows, c, c * f, res)
                            if ok:
                                assert pin is not None, (cc, dt, c, f, res, rows)
                                n_ok += 1
                            elif pin is None and why.startswith("af3_fused:no_launch_pin"):
                                assert T.af3_pin_key(cc, dt, rows, c, c * f, res).startswith(cc + "|" + dt), why
                                n_ref += 1
    assert n_ok >= 40 and n_ref >= 10, (n_ok, n_ref)
    ok, why = T.admits("af3_fused", c=256, hidden=512, dtype="bf16", cc="9.0", rows_count=640000)         # a pair c256 n2 shape: admissible structurally, no pin anywhere
    assert not ok and why == "af3_fused:no_launch_pin:9.0|bf16|c256|h512|res0", why                        # no bucket pinned at all for that (c, hidden)
    assert T.admits("af3_fused", c=64, hidden=256, dtype="bf16", cc="9.0", rows_count=640000)[0]            # the c64 n4 family is pinned
    with pytest.raises(T.Refusal) as e:                                                                     # select by row word at an unpinned key: refused with the fallback row
        T.select("af3_fused", c=256, hidden=512, n_tokens=800, cc="9.0", rows_count=640000)
    assert e.value.kind.startswith("af3_fused:no_launch_pin") and e.value.fallback in T.ROW_NAMES, (e.value.kind, e.value.fallback)
    for cc in ("9.0", "8.0"):                                                                               # the residual pair c128 n4 key an AF3-family kit binds, and the template n2
        for rows in (4096, 100000, 1478656):                                                                # key: pinned on both cards in every rows bucket (kit launch table import)
            for c, h, res in ((128, 512, True), (128, 512, False), (64, 128, False), (64, 128, True)):
                assert T.admits("af3_fused", c=c, hidden=h, dtype="bf16", cc=cc, residual=res, rows_count=rows) == (True, ""), (cc, rows, c, h, res)
                pin = T.af3_pin(cc, "bf16", rows, c, h, res)
                assert pin and pin["kernel"] == "pow2" and pin["BM"] and pin["BH"], (cc, rows, c, h, res, pin)
            s = T.select("af3_fused", c=128, hidden=512, n_tokens=int(rows ** 0.5), cc=cc, residual=True, rows_count=rows)
            assert s.row == "af3_fused", s


def test_big_is_the_fastest_row_unless_its_peak_exceeds_the_statements_and_the_literal_word_resolves():
    """Release policy for the memory tier: per cell and stack, big == fast unless the fast row's recorded peak exceeds the statement's recorded peak
    (by more than 5 % + 64 MiB; peaks under 1 GiB never count), in which case big is an admissible row whose peak does not (or the statement); and
    select('big', ...) -- the literal tier word kits bind -- resolves on both cards for every family."""
    t = T.table()
    n = n_eq = 0
    for key, cell in t["cells"].items():
        for st, sc in cell["stacks"].items():
            b, f = sc.get("big"), sc.get("fast")
            if not b or not f:
                continue
            n += 1
            assert b in sc["ms"], (key, st, b)
            stock = cell.get("stock_arm") or sc.get("stock_arm")
            sp = sc["peak_mib"].get(stock) if stock else None
            pf, pb = sc["peak_mib"].get(f), sc["peak_mib"].get(b)
            fits = lambda p: p is None or sp is None or p < 1024.0 or p <= sp * 1.05 + 64.0
            if b == f:
                n_eq += 1
                continue
            if fits(pf):                                                     # the fast row fits: big may differ only by parity (an incumbent within 5 %) and must fit too
                assert fits(pb) and sc["ms"][b] <= sc["ms"][f] * 1.05 + 1e-9, (key, st, f, b, sc["ms"][f], sc["ms"][b])
            else:                                                            # the fast row's peak exceeds the statement's: big is a fitting row or the statement
                assert fits(pb) or T.split_word(b)[0] in T.STOCK_ROWS, (key, st, f, b, pf, pb, sp)
    assert n >= 400 and n_eq >= 0.8 * n, (n, n_eq)
    served = 0
    for cc in ("9.0", "8.0"):
        for c, h, fam, dt in ((128, 512, "pair", "bf16"), (256, 1024, "pair", "bf16"), (128, 256, "pair", "bf16"), (384, 1536, "single", "bf16"), (64, 256, "rows", "bf16"),
                              (128, 512, "pair", "fp32"), (384, 1536, "pair", "fp32"), (256, 1024, "esmpair", "bf16")):
            for n_tok in (400, 800, 1200):
                s = T.select("big", c=c, hidden=h, n_tokens=n_tok, dtype=dt, cc=cc, family=fam, residual=(fam == "esmpair"), ln_given=True)
                assert s.tier == "big" and s.row in T.ROW_NAMES, (cc, c, h, fam, dt, n_tok, s)
                served += 1
    assert served == 48


def test_stack_words_with_library_suffixes_and_card_aliases_resolve_to_the_tables_keys():
    """Kits key other providers' cells on longer stack words ('.../cueq0.10.0') and various card spellings; the transition table keys on
    '<H100|A100>:torch<v>/<triton>' -- norm_stack() maps the former onto the latter so an exact vouch recorded for the stack is found."""
    assert T.norm_stack("H100:torch2.7.1+cu128/3.3.1/cueq0.10.0") == "H100:torch2.7.1+cu128/3.3.1"
    assert T.norm_stack("A100-80GB:torch2.13.0+cu130/3.7.1/cueq0.11.1") == "A100:torch2.13.0+cu130/3.7.1"
    assert T.norm_stack("NVIDIA H100 80GB HBM3:torch2.13.0+cu130/3.7.1") == "H100:torch2.13.0+cu130/3.7.1"
    assert T.norm_stack("H100:torch2.13.0+cu130/3.7.1") == "H100:torch2.13.0+cu130/3.7.1" and T.norm_stack(None) is None
    a = T.select("exact", c=128, hidden=512, n_tokens=512, cc="9.0", stack="H100:torch2.7.1+cu128/3.3.1", ln_given=True)
    b = T.select("exact", c=128, hidden=512, n_tokens=512, cc="9.0", stack="H100:torch2.7.1+cu128/3.3.1/cueq0.10.0", ln_given=True)
    assert a.row == b.row == "v1" and a.stack == b.stack and str(a.cls).startswith("bitwise"), (a, b)
    s = T.select("v1", c=128, hidden=512, n_tokens=512, cc="9.0", ln_given=True)
    assert "exact_class_given_the_callers_ln_output" in s.words


def test_af3_fused_row_floors_from_the_table_refuse_by_name_and_never_win_a_tier_below_them():
    """row_floors.af3_fused (a kit arch table encoded as data): below 64 rows the row refuses by name; the padded-tile path (c 256 / 384, bf16) refuses below
    4096 rows on cc 9.0 and always on cc 8.0; no tier word resolves to af3_fused where the floor refuses it."""
    t = T.table(); fl = t["row_floors"]["af3_fused"]
    assert fl["min_rows"] == 64 and fl["padded"]["9.0"]["min_rows"] == 4096 and fl["padded"]["8.0"] == "off" and fl["source"].startswith("kit table")
    assert T.admits("af3_fused", c=128, hidden=512, dtype="bf16", cc="9.0", rows_count=63) == (False, "af3_fused:below_64_rows")
    assert T.admits("af3_fused", c=128, hidden=512, dtype="bf16", cc="9.0", rows_count=64)[0]
    assert T.admits("af3_fused", c=384, hidden=1536, dtype="bf16", cc="9.0", rows_count=4095) == (False, "af3_fused:padded_path_below_4096_rows_on_cc9.0")
    assert T.admits("af3_fused", c=384, hidden=1536, dtype="bf16", cc="9.0", rows_count=4096)[0]
    assert T.admits("af3_fused", c=384, hidden=1536, dtype="bf16", cc="8.0", rows_count=1 << 20) == (False, "af3_fused:padded_path_off_on_cc8.0")
    assert T.admits("af3_fused", c=256, hidden=1024, dtype="bf16", cc="8.0", rows_count=1 << 20) == (False, "af3_fused:padded_path_off_on_cc8.0")
    assert T.admits("af3_fused", c=384, hidden=1536, dtype="fp32", cc="9.0", rows_count=1200)[0]          # the fp32 engine's cells are not the padded bf16 rule's
    for key, cell in t["cells"].items():
        cc, dt, cw, nb, timing, d = key.split("|")
        c, h = T.cell_dims(cw)
        n = int(nb[3:]); rows = n * n if not cw.startswith("single") else n
        for st, sc in cell["stacks"].items():
            for w in ("fast", "faithful", "big"):
                arm = sc.get(w)
                if arm and T.split_word(arm)[0] == "af3_fused":
                    assert T.admits("af3_fused", c=c, hidden=h, dtype=dt, cc=cc, rows_count=rows, residual=cw.startswith("esmpair"))[0], (key, st, w, arm)


def test_exact_row_bands_gate_the_exact_tier_by_the_row_counts_remainder_on_the_banded_stack_only():
    """exact_row_bands (a kit's cc 8.0 exact vouch on the torch 2.11.0+cu128 / triton 3.6.0 A100 stack, encoded as data): the exact tier serves v1 only when
    161424 <= (rows mod 2^20) <= 2^20 - 8192 there; outside the band it refuses by name to the statement; other stacks are unaffected."""
    t = T.table(); ST = "A100:torch2.11.0+cu128/3.6.0"
    band = t["exact_row_bands"][ST]
    assert band["modulus"] == 1 << 20 and band["min_remainder"] == 161424 and band["max_remainder"] == (1 << 20) - 8192 and "v1" in band["rows"] and band["source"].startswith("kit table")
    s = T.select("exact", c=128, hidden=512, n_tokens=800, cc="8.0", stack=ST, ln_given=True)               # 640000 rows: inside the band and a vouched cell
    assert s.row == "v1", s
    with pytest.raises(T.Refusal) as e:                                                                    # 400 tokens = 160000 rows < 161424: outside the band
        T.select("exact", c=128, hidden=512, n_tokens=400, cc="8.0", stack=ST, ln_given=True)
    assert e.value.kind.startswith("exact_rows_remainder_outside_the_vouched_band_on_") and e.value.fallback in T.STOCK_ROWS, (e.value.kind, e.value.fallback)
    with pytest.raises(T.Refusal) as e2:                                                                   # 1024 tokens = 2^20 rows: remainder 0
        T.select("exact", c=128, hidden=512, n_tokens=1024, cc="8.0", stack=ST, ln_given=True)
    assert "remainder" in e2.value.kind
    s2 = T.select("exact", c=128, hidden=512, n_tokens=400, cc="8.0", stack="A100:torch2.13.0+cu130/3.7.1", ln_given=True)   # the reference stack: no band, vouched from 64 tokens
    assert s2.row == "v1", s2
    f = T.select("fast", c=128, hidden=512, n_tokens=400, cc="8.0", stack=ST)                              # tolerance tiers are not banded
    assert f.row in T.ROW_NAMES and f.tier == "fast"


def test_withdrawn_exact_vouch_names_the_statement_by_cell_and_keeps_the_row_selectable_by_word():
    """An exact vouch withdrawn on a binding engine's in-model det report: the cells' exact word is the statement by cell on that stack (no refusal),
    vouched_on is cleared, the reason is recorded, the row stays selectable by word and the tolerance tiers are unchanged."""
    t = T.table(); ST = "H100:torch2.13.0+cu130/3.7.1"
    cells = [k for k in t["cells"] if k.startswith("9.0|bf16|rows_c64_n2|")]
    assert len(cells) >= 8, cells
    for k in cells:
        c = t["cells"][k]
        assert c["exact"] in T.STOCK_ROWS and ST not in (c.get("vouched_on") or {}) and ST in (c.get("exact_vouch_withdrawn") or {}), k
    for n in (256, 400, 800, 1200, 1536):
        s = T.select("exact", c=64, hidden=128, n_tokens=n, cc="9.0", stack=ST, family="rows", ln_given=True)
        assert s.row == "torch_swiglu", (n, s)
        f = T.select("fast", c=64, hidden=128, n_tokens=n, cc="9.0", stack=ST, family="rows")
        assert f.row == "v2", (n, f)
        w = T.select("v1", c=64, hidden=128, n_tokens=n, cc="9.0", stack=ST, family="rows", ln_given=True)
        assert w.row == "v1", (n, w)
    assert ST in t["coverage"]["9.0|bf16|rows_c64_n2|eager|fwd"].get("exact_vouch_withdrawn", {})


def test_unmeasured_capabilities_inherit_the_nearest_measured_columns_portable_rows_never_a_refusal_for_the_tolerance_tiers():
    """A capability with NO cell for a family key (12.0 / 11.0 -> the highest measured column at or below it that carries the family; 8.6 / 8.9 -> the
    8.0 column): fast | big name a source-compiled row of the inherited cell (never an sm_90a binary row, never af3_fused without a launch pin for that
    capability, never a refusal while a portable row exists) with the census token; exact names the statement by cell; below 8.0 nothing is inherited;
    a capability that owns the key (the 10.0 / 10.3 pair columns) is served from its own cell."""
    RTX = "RTX:torch2.13.0+cu130/3.7.1"
    ms = set(T.measured_ccs())
    assert ms >= {"9.0", "8.0"}
    assert T.inherit_cc("9.0") is None and T.inherit_cc("8.0") is None and T.inherit_cc("7.5") is None and T.inherit_cc("8.6") == "8.0"
    assert T.inherit_cc("12.0") == max((m for m in ms if float(m) <= 12.0), key=float)
    cases = [("pair", 128, 512, "bf16", "pair_c128_n4"), ("pair", 256, 1024, "bf16", "pair_c256_n4"), ("pair", 128, 512, "fp32", "pair_c128_n4"), ("single", 384, 1536, "bf16", "single_c384_n4"),
             ("rows", 64, 256, "bf16", "rows_c64_n4"), ("pair", 384, 1536, "bf16", "pair_c384_n4")]
    for cc in ("12.0", "11.0", "8.6"):
        for fam, c, h, dt, cw in cases:
            for timing in ("eager", "graph"):
                col = T.inherit_cc(cc, dt, cw, timing, "fwd")
                assert col is not None and float(col) <= float(cc), (cc, cw, timing, col)
                for w in ("fast", "big", "faithful"):
                    s = T.select(w, c=c, hidden=h, n_tokens=800, dtype=dt, cc=cc, family=fam, timing=timing, stack=RTX, ln_given=True)
                    assert s.cell_key.startswith(col + "|"), (cc, cw, timing, w, s.cell_key, col)
                    assert "inherited_cc:unmeasured" in s.words and ("inherited_from:%s" % col) in s.words, s.words
                    assert s.row not in T.NON_PORTABLE_ROWS, (cc, cw, w, s.row)
                    assert s.row != "af3_fused" or T.af3_pin(cc, dt, 800 * 800, c, h, False) is not None, (cc, cw, w, s.row)
                    if s.row in T.STOCK_ROWS:                                       # only when the inherited cell has no portable admitted row at all
                        cell = T.table()["cells"][s.cell_key]; sc = cell["stacks"][cell["ref_stack"]]
                        portable = [a for a in sc["ms"] if sc["ms"][a] is not None and not a.startswith("x:") and T.split_word(a)[0] not in T.STOCK_ROWS
                                    and T.portable_row(T.split_word(a)[0], cc, dtype=dt, c=c, hidden=h) and T.split_word(a)[0] != "esm_fused_exact"]
                        assert "no_portable_row" in s.words and not [a for a in portable if T.admits(a, c=c, hidden=h, dtype=dt, cc=col, ln_given=True)[0]], (cc, cw, w, s.words, portable)
                e = T.select("exact", c=c, hidden=h, n_tokens=800, dtype=dt, cc=cc, family=fam, timing=timing, stack=RTX, ln_given=True)
                assert e.row in T.STOCK_ROWS and ("exact_unvouched_on_cc_%s" % cc) in e.words and "inherited_cc:unmeasured" in e.words, (cc, cw, e.row, e.words)
    for cc in sorted(ms - {"9.0", "8.0"}):                                          # a measured new-part column: its own pair cells, no inheritance
        own = T.select("fast", c=128, hidden=512, n_tokens=800, cc=cc, family="pair", stack="cc%s:torch2.13.0+cu130/3.7.1" % cc.replace(".", ""))
        assert own.cell_key.startswith(cc + "|") and not any(w.startswith("inherited_") for w in own.words), (cc, own.cell_key, own.words)
    for n in (800, 1200):                                                           # a donor cell whose fast word is an sm_90a binary row (the 9.0 c 256 graph cells): substituted
        col = T.inherit_cc("12.0", "bf16", "pair_c256_n4", "graph", "fwd")
        s = T.select("fast", c=256, hidden=1024, n_tokens=n, cc="12.0", family="pair", timing="graph", stack=RTX)
        sub = [w for w in s.words if w.startswith("substituted_for:")]
        assert s.row not in T.NON_PORTABLE_ROWS and (col != "9.0" or not sub or T.split_word(sub[0][len("substituted_for:"):])[0] in T.NON_PORTABLE_ROWS or sub[0].endswith("af3_fused")), (s.row, s.words)
    d = T.select("fast", c=256, hidden=1024, n_tokens=800, cc="10.3", family="esmpair", residual=True, direction="fwdbwd")   # the ESM pair family has no 10.x column: inherited
    assert d.row in ("esm_kd3", "esm_t15_kd3") and "inherited_cc:unmeasured" in d.words, (d.row, d.words)
    with pytest.raises(T.Refusal) as r:
        T.select("fast", c=128, hidden=512, n_tokens=800, cc="7.5", family="pair")
    assert str(r.value.kind).startswith("no_cell"), r.value.kind
    ok, why = T.admits("esm_t15", c=256, hidden=1024, cc="10.3", residual=True)
    assert ok, why
    ok, why = T.admits("esm_t16", c=256, hidden=1024, cc="10.3", residual=True)
    assert not ok and why == "esm_t16:cc_10.3_not_9.0", why

def test_measured_capabilities_never_take_the_inheritance_path():
    for cc in T.measured_ccs():
        for key, cell in T.table()["cells"].items():
            if not key.startswith(cc + "|") or "+" in key.split("|")[2]:
                continue
            _cc, dt, cw, nb, timing, d = key.split("|")
            c, h = T.cell_dims(cw)
            fam = "esmpair" if cw.startswith("esmpair") else ("single" if cw.startswith("single") else ("rows" if cw.startswith("rows") else "pair"))
            for w in ("fast", "big"):
                try:
                    s = T.select(w, c=c, hidden=h, n_tokens=int(nb[3:]), dtype=dt, cc=cc, family=fam, timing=timing, direction=d, residual=fam == "esmpair", stack=cell["ref_stack"], ln_given=True)
                except T.Refusal:
                    continue
                assert not any(x.startswith("inherited_") for x in (s.words or ())), (key, w, s.words)
            break


def test_cc_10_3_with_ONE_injected_cell_still_inherits_the_portable_row_for_every_OTHER_key(monkeypatch):
    """Inheritance is decided per (capability, family key): a capability that owns a cell for ONE key (here an injected 12.0 pair c 256 family) still
    inherits every OTHER key (pair c 128 -> the highest measured column at or below it carrying the family, single c 384 -> the 9.0 column, ...) instead
    of falling to no_cell / the statement by name.  (Named for the audit that introduced it; 10.3 has real pair columns since the parts fold.)"""
    import copy
    t = copy.deepcopy(T.table())
    CC, ST = "12.0", "RTX:torch2.13.0+cu130/3.7.1"
    src = "9.0|bf16|pair_c256_n4|N<=800|eager|fwd"
    inj = "%s|bf16|pair_c256_n4|N<=800|eager|fwd" % CC
    t["cells"][inj] = copy.deepcopy(t["cells"][src])
    sc = t["cells"][inj]["stacks"][t["cells"][inj]["ref_stack"]]                      # a realistic own cell: its winner is a row that serves the part (never an sm_90a binary)
    port = next(a for a in sorted(sc["ms"], key=lambda a: sc["ms"][a] or 9e9) if T.split_word(a)[0] == "v2")
    for w in ("fast", "big", "faithful"):
        sc[w] = port; t["cells"][inj][w] = port
    t["coverage"]["%s|bf16|pair_c256_n4|eager|fwd" % CC] = {"sizes": [800], "engines": "", "fast": {"800": port}, "exact": {"800": t["cells"][inj].get("exact")}}
    monkeypatch.setattr(T, "table", lambda: t)
    assert CC in T.measured_ccs()
    own = T.select("fast", c=256, hidden=1024, n_tokens=800, cc=CC, family="pair", stack=ST)
    assert own.cell_key == inj and not any(w.startswith("inherited_") for w in own.words), (own.cell_key, own.words)      # its own cell: no inheritance
    for fam, c, h, dt, cw in (("pair", 128, 512, "bf16", "pair_c128_n4"), ("pair", 128, 256, "bf16", "pair_c128_n2"), ("rows", 64, 256, "bf16", "rows_c64_n4"), ("single", 384, 1536, "bf16", "single_c384_n4"),
                              ("pair", 128, 512, "fp32", "pair_c128_n4"), ("pair", 384, 1536, "bf16", "pair_c384_n4")):
        for w in ("fast", "big"):
            for timing in ("eager", "graph"):
                col = T.inherit_cc(CC, dt, cw, timing, "fwd")
                assert col is not None, (cw, timing)
                s = T.select(w, c=c, hidden=h, n_tokens=800, dtype=dt, cc=CC, family=fam, timing=timing, stack=ST, ln_given=True)
                assert s.cell_key.startswith(col + "|") and "inherited_cc:unmeasured" in s.words and ("inherited_from:%s" % col) in s.words, (fam, c, w, timing, s.cell_key, s.words)
                assert s.row not in T.NON_PORTABLE_ROWS and (s.row != "af3_fused" or T.af3_pin(CC, dt, 640000, c, h, False) is not None), (fam, c, w, s.row)
        e = T.select("exact", c=c, hidden=h, n_tokens=800, dtype=dt, cc=CC, family=fam, stack=ST, ln_given=True)
        assert e.row in T.STOCK_ROWS and ("exact_unvouched_on_cc_%s" % CC) in e.words, (fam, c, e.row, e.words)
    g = T.select("fast", c=256, hidden=1024, n_tokens=800, cc=CC, family="pair", timing="graph", stack=ST)     # the injected family has no graph cell: inherited too
    colg = T.inherit_cc(CC, "bf16", "pair_c256_n4", "graph", "fwd")
    assert g.cell_key.startswith(colg + "|") and ("inherited_from:%s" % colg) in g.words, (g.cell_key, g.words)
    assert T.inherit_cc(CC, "bf16", "pair_c256_n4", "eager", "fwd") is None
    assert T.inherit_cc(CC, "bf16", "pair_c128_n4", "eager", "fwd") == max((m for m in T.measured_ccs() if float(m) <= float(CC) and "%s|bf16|pair_c128_n4|eager|fwd" % m in t["coverage"]), key=float)
    assert T.inherit_cc("8.0", "fp32", "pair_c384_n4", "eager", "fwd") is None                                       # no measured column at or below the part carries it: nothing inherited upward across architectures

def test_an_inherited_row_whose_launch_raises_steps_aside_once_to_the_next_portable_row_then_the_statement(monkeypatch):
    """An INHERITED (unmeasured-capability) row that fails to build / launch on the new part: the call returns the donor cell's next portable row (or the
    statement by name), the census carries stepped_aside:error:<type>, and the dead row is not retried by the next call; OOM is re-raised; a Refusal at
    serve steps aside the same way.  The dispatcher is faked (no GPU here)."""
    torch = pytest.importorskip("torch")
    T._DEAD_INHERITED.clear()
    calls, notes = [], []
    real_record = T._CENSUS.record
    monkeypatch.setattr(T._CENSUS, "record", lambda prov, key, outcome, served, cell_id=None, note="", **kw: notes.append((outcome, served, note)))
    def fake_dispatch(x, W, sel, word, residual, x_ln, mask, binary_mask, out, form):
        served = T._arm_word(sel.row, sel.variant, T.cfg_word(sel.cfg) if (sel.cfg and sel.row == "v2") else None)
        calls.append(served)
        if sel.row == "v2":
            raise RuntimeError("ptxas fatal: simulated build failure on this part")
        if sel.row == "lnl":
            raise T.Refusal("lnl:build_failed:CompilationError", word, "torch_swiglu")
        return ("served-by", served), sel
    monkeypatch.setattr(T, "_dispatch", fake_dispatch)
    lin = torch.nn.Linear(128, 512, bias=False)
    W = T.pack(w_a=lin.weight, w_b=torch.nn.Linear(128, 512, bias=False).weight, w_o=torch.nn.Linear(512, 128, bias=False).weight,
               ln_w=torch.ones(128), ln_b=torch.zeros(128), eps=1e-5, device="cpu")
    x = torch.zeros(64, 128, dtype=torch.bfloat16)
    first = T.select("fast", c=128, hidden=512, n_tokens=800, cc="12.0", family="pair", stack="RTX:torch2.13.0+cu130/3.7.1")
    assert first.row == "v2" and "inherited_cc:unmeasured" in first.words, (first.row, first.words)
    (tag, served1), sel1 = T.transition(x, W, word="fast", n_tokens=800, family="pair", cc="12.0", stack="RTX:torch2.13.0+cu130/3.7.1")
    assert tag == "served-by" and T.split_word(served1)[0] not in ("v2", "lnl") and sel1.row == T.split_word(served1)[0], (served1, calls)
    assert calls[0].startswith("v2") and any(n[0] == "named_fallback" and n[2] == "stepped_aside:error:RuntimeError" for n in notes), (calls, notes)
    if any(c_.startswith("lnl") for c_ in calls):
        assert any(n[2] == "stepped_aside:error:Refusal" for n in notes), notes
    n_first = len(calls)
    calls.clear(); notes.clear()
    (tag2, served2), sel2 = T.transition(x, W, word="fast", n_tokens=800, family="pair", cc="12.0", stack="RTX:torch2.13.0+cu130/3.7.1")
    assert served2 == served1 and calls == [served1] and not any("stepped_aside" in str(n_[2]) for n_ in notes), (calls, notes)   # the dead rows are not retried in this process
    again = T.select("fast", c=128, hidden=512, n_tokens=800, cc="12.0", family="pair", stack="RTX:torch2.13.0+cu130/3.7.1")
    assert not again.row == "v2", again.row
    def all_fail(x, W, sel, word, residual, x_ln, mask, binary_mask, out, form):
        if sel.row in T.STOCK_ROWS:
            return ("statement", sel.row), sel
        raise RuntimeError("nothing builds here")
    monkeypatch.setattr(T, "_dispatch", all_fail)
    (tag3, row3), sel3 = T.transition(x, W, word="big", n_tokens=800, family="pair", cc="12.0", stack="RTX:torch2.13.0+cu130/3.7.1")
    assert tag3 == "statement" and row3 == "torch_swiglu" and sel3.row == "torch_swiglu" and "inherited_cc:unmeasured" in sel3.words, (row3, sel3.words)
    def oom(x, W, sel, word, residual, x_ln, mask, binary_mask, out, form):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. simulated") if hasattr(torch.cuda, "OutOfMemoryError") else MemoryError("out of memory")
    monkeypatch.setattr(T, "_dispatch", oom)
    T._DEAD_INHERITED.clear()
    with pytest.raises((getattr(torch.cuda, "OutOfMemoryError", MemoryError), MemoryError, RuntimeError)) as r:
        T.transition(x, W, word="fast", n_tokens=800, family="pair", cc="12.0", stack="RTX:torch2.13.0+cu130/3.7.1")   # an inherited row: OOM is never swallowed by the guard
    assert "out of memory" in str(r.value).lower()
    T._DEAD_INHERITED.clear()


def test_no_column_off_cc90_names_a_bare_v2_word_as_a_tier_winner_and_select_substitutes_one_if_it_did(monkeypatch):
    """A `v2` / `v2:fast` word WITHOUT a launch configuration names the package's cc 9.0 default tile; on any other capability that tile is unmeasured.  No cell
    column whose capability is not 9.0 may name such a word as fast | faithful | big (the builder writes the column's fastest measured `v2@...` word instead),
    and -- were one to appear -- select()'s measured-tier branch substitutes the column's fastest measured launch word (token substituted_for:), never cfg=None."""
    import copy
    t = T.table()
    bad = []
    for key, cell in t["cells"].items():
        if key.split("|")[0] == "9.0":
            continue
        for st, sc in cell["stacks"].items():
            for w in ("fast", "faithful", "big"):
                a = sc.get(w)
                if a and T.split_word(a)[0] == "v2" and T.split_word(a)[2] is None:
                    bad.append((key, st, w, a))
        for w in ("fast", "faithful", "big"):
            a = cell.get(w)
            if a and T.split_word(a)[0] == "v2" and T.split_word(a)[2] is None:
                bad.append((key, "top", w, a))
    assert not bad, bad[:6]
    # a resolved tier word off 9.0 never hands _serve_v2 a bare v2 (cfg None) from a MEASURED column
    for key, cell in t["cells"].items():
        cc, dt, cw, nb, timing, d = key.split("|")
        if cc == "9.0" or d != "fwd" or "+" in cw:
            continue
        fam = "single" if cw.startswith("single") else "rows" if cw.startswith("rows") else "esmpair" if cw.startswith("esmpair") else "pair"
        for st in cell["stacks"]:
            for w in ("fast", "big"):
                try:
                    s = T.select(w, c=cell["c"], hidden=cell["hidden"], n_tokens=int(nb[3:]), dtype=dt, cc=cc, family=fam, timing=timing, stack=st, ln_given=True,
                                 residual=fam == "esmpair")
                except T.Refusal:
                    continue
                assert not (s.row == "v2" and s.cfg is None), (key, st, w, s.arm)
    # glue: inject a bare v2 winner into an 8.0 column and check the substitution
    t2 = copy.deepcopy(t)
    key = "8.0|bf16|pair_c128_n4|N<=800|eager|fwd"
    cell = t2["cells"][key]; st = cell["ref_stack"]; sc = cell["stacks"][st]
    meas = sorted((a for a in sc["ms"] if sc["ms"][a] is not None and a.startswith("v2@")), key=lambda a: sc["ms"][a])
    assert meas, "the reference 8.0 column carries measured v2 launch words"
    sc["ms"]["v2"] = sc["ms"][meas[0]]; sc.setdefault("x", {})["v2"] = (sc.get("x") or {}).get(meas[0]); sc["fast"] = "v2"; sc["big"] = "v2"; cell["fast"] = "v2"; cell["big"] = "v2"
    monkeypatch.setattr(T, "table", lambda: t2)
    for w in ("fast", "big"):
        s = T.select(w, c=128, hidden=512, n_tokens=800, cc="8.0", family="pair", stack=st, ln_given=True)
        assert s.row == "v2" and s.cfg is not None and T.cfg_word(s.cfg) == meas[0].split("@")[1] and "substituted_for:v2" in s.words, (w, s.row, s.cfg, s.words)
    s9 = T.select("fast", c=128, hidden=512, n_tokens=800, cc="9.0", family="pair", stack="H100:torch2.13.0+cu130/3.7.1", ln_given=True)   # on 9.0 the bare word IS the measured package cell
    assert not any(w_.startswith("substituted_for:") for w_ in s9.words), s9.words


def test_pack_cache_residency_big_holds_one_weight_sets_packs_fast_keeps_every_layers(monkeypatch):
    """PACK RESIDENCY: under the big tier word (policy 'lru') the packs the rows build (bf16 operand copies, the v2 interleaved layout, ...) never grow
    with the layer count -- after N distinct weight sets at most OPT_CORE_TRANSITION_PACK_LRU (default 1) sets' packs are resident; under fast / exact / row
    words (policy 'layer') every weight set keeps its packs (speed).  Counted through the cache's own bookkeeping (no device needed); the module's own
    parameters are never touched.  OPT_CORE_TRANSITION_PACK_CACHE=layer|lru forces either policy for every word."""
    class FakeT(object):                                                   # a tensor-like with a byte size
        def __init__(self, n): self._n = n
        def numel(self): return self._n
        def element_size(self): return 2
    ONE = 3 * 1024 * 2 + 4096 * 2                                          # bytes of one weight set's packs in this test
    def packs_of(W):
        W.pack_for("op16", lambda: (FakeT(1024), FakeT(1024), FakeT(1024)))
        return W.pack_for("v2", lambda: {"wab": FakeT(4096)})
    monkeypatch.delenv(T.PACK_CACHE_ENV, raising=False); monkeypatch.delenv(T.PACK_LRU_ENV, raising=False)
    T.pack_cache_clear()
    assert T.pack_policy("big") == "lru" and T.pack_policy("fast") == "layer" and T.pack_policy("exact") == "layer" and T.pack_policy("v2") == "layer"
    sel = T.Selection.__new__(T.Selection); sel.tier = "big"; sel.word = "big"
    assert T.pack_policy(sel) == "lru"
    # big: N = 8 distinct weight sets -> resident <= one set's packs
    Ws = []
    for i in range(8):
        W = T.Weights.__new__(T.Weights); W._packs = {}; W._policy = T.pack_policy("big"); Ws.append(W)
        p = packs_of(W)
        assert p is packs_of(W)                                            # a second call of the same layer hits the store (no rebuild while resident)
        st = T.pack_cache_stats()
        assert st["lru_entries"] <= st["lru_keep"] == 1 and st["lru_bytes"] <= ONE + 64, st
        assert not W._packs                                               # nothing kept with the layer under lru
    assert T.pack_cache_stats()["evictions"] >= 7
    monkeypatch.setenv(T.PACK_LRU_ENV, "2"); T.pack_cache_clear()               # identity-keyed, never shape-keyed: two weight sets of one shape get their own packs
    pa, pb = packs_of(Ws[0]), packs_of(Ws[1])
    assert pa is not pb and packs_of(Ws[0]) is pa and packs_of(Ws[1]) is pb
    # a deeper store by env
    monkeypatch.setenv(T.PACK_LRU_ENV, "2"); T.pack_cache_clear()
    for W in Ws[:5]:
        packs_of(W)
    st = T.pack_cache_stats(); assert st["lru_entries"] == 2 and st["lru_bytes"] <= 2 * ONE + 64, st
    # fast: every layer keeps its packs
    monkeypatch.delenv(T.PACK_LRU_ENV, raising=False); T.pack_cache_clear()
    before = T.pack_cache_stats()["layer_bytes"]
    Wf = []
    for i in range(8):
        W = T.Weights.__new__(T.Weights); W._packs = {}; W._policy = T.pack_policy("fast"); Wf.append(W)
        packs_of(W)
        assert set(W._packs) == {"op16", "v2"}
    st = T.pack_cache_stats(); assert st["layer_bytes"] - before >= 8 * ONE and st["lru_entries"] == 0, st   # built into 8 per-layer caches (kept by the layers)
    # the knob forces either way
    monkeypatch.setenv(T.PACK_CACHE_ENV, "lru"); assert T.pack_policy("fast") == "lru" and T.pack_policy("v1") == "lru"
    monkeypatch.setenv(T.PACK_CACHE_ENV, "layer"); assert T.pack_policy("big") == "layer"
    monkeypatch.delenv(T.PACK_CACHE_ENV, raising=False); T.pack_cache_clear()
    # the serving path sets the policy from the resolved word: _dispatch / transition_autograd assign W._policy before any pack is built
    import inspect
    assert "W._policy = pack_policy(sel)" in inspect.getsource(T._dispatch) and "W._policy = pack_policy(" in inspect.getsource(T.transition_autograd)


def test_pack_residency_B7_a_memory_lean_mode_binding_exact_rows_bounds_every_pack_store(monkeypatch):
    """B7: a kit's big MODE that binds an exact-class row (word='exact', not the big tier word) asks pack_residency('lru') once; then EVERY pack the
    serving paths build through Weights.pack_for -- the v2 interleaved layout (fpf_transition_v2.kernel.pack_weights via _serve_v2's builder), the bf16
    operand copies (_op16 = wa16 / wb16 / wo16), the v1 / flash / af3 / lnl packs -- lives in the ONE bounded store: after two forwards over 8 distinct
    layers the resident pack bytes across all stores are <= one layer's worth; with the default (follow the word) under 'exact' all 8 layers keep theirs;
    the engineering knob still overrides both ways."""
    torch = pytest.importorskip("torch")
    monkeypatch.delenv(T.PACK_CACHE_ENV, raising=False); monkeypatch.delenv(T.PACK_LRU_ENV, raising=False)
    T.pack_cache_clear()
    c, hid = 64, 256

    def layer():
        m = {"ln_w": torch.ones(c), "ln_b": torch.zeros(c), "w_a": torch.randn(hid, c), "w_b": torch.randn(hid, c), "w_o": torch.randn(c, hid)}
        return T.pack(ln_w=m["ln_w"], ln_b=m["ln_b"], w_a=m["w_a"], w_b=m["w_b"], w_o=m["w_o"], eps=1e-5)

    def v2_like_pack(W):                                                      # the bytes _serve_v2's builder makes (W12 = [wa16; wb16] interleave-sized, W3, LN)
        return {"W12": torch.cat([W.wa16, W.wb16], 0).contiguous().clone(), "W3": W.wo16.clone(), "LN_W32": W.ln_w, "LN_B32": W.ln_b}

    def forward(layers, word):
        for W in layers:
            W._policy = T.pack_policy(word)                                   # exactly what _dispatch / transition_autograd do per call
            W._op16()                                                         # the bf16 operand copies
            W.pack_for("v2", lambda W=W: v2_like_pack(W))                     # the row's pack

    def resident(layers):
        st = T.pack_cache_stats()
        per_layer = sum(sum(T.pack_nbytes(p) for p in W._packs.values()) for W in layers)
        return per_layer + st["lru_bytes"], st
    one = None
    try:
        assert T.pack_residency() is None
        # (1) default: word 'exact' -> per-layer cache: 8 layers x (op16 + v2 pack) resident after two forwards
        layers = [layer() for _ in range(8)]
        forward(layers, "exact"); forward(layers, "exact")
        tot, st = resident(layers)
        one = sum(T.pack_nbytes(p) for p in layers[0]._packs.values())
        assert one > 0 and tot >= 8 * one and st["lru_bytes"] == 0, (tot, one, st)
        # (2) the mode asks lru once: every store bounded to ONE weight set's packs, after two forwards over 8 NEW layers (and the old ones keep nothing new)
        assert T.pack_residency("lru") is None and T.pack_residency() == "lru"
        layers2 = [layer() for _ in range(8)]
        forward(layers2, "exact"); forward(layers2, "exact")
        tot2, st2 = resident(layers2)
        assert all(not W._packs for W in layers2), [len(W._packs) for W in layers2]          # nothing kept per layer
        assert st2["lru_entries"] <= 1 and tot2 <= one + 64, (tot2, one, st2)                # <= one layer's worth (+ rounding slack)
        assert st2["residency"] == "lru"
        # a Selection-shaped word (tier exact) is treated the same
        class _Sel(object):
            tier = "exact"; word = "exact"
        assert T.pack_policy(_Sel()) == "lru" and T.pack_policy("v2:fast") == "lru" and T.pack_policy("big") == "lru"
        # (3) precedence: the engineering knob overrides the process setting both ways
        monkeypatch.setenv(T.PACK_CACHE_ENV, "layer"); assert T.pack_policy("exact") == "layer" and T.pack_policy("big") == "layer"
        monkeypatch.setenv(T.PACK_CACHE_ENV, "lru"); T.pack_residency("layer"); assert T.pack_policy("exact") == "lru"
        monkeypatch.delenv(T.PACK_CACHE_ENV); assert T.pack_policy("big") == "layer"                  # process setting 'layer' now wins over the word
        assert T.pack_residency(None) == "layer" and T.pack_policy("big") == "lru" and T.pack_policy("exact") == "layer"
        with pytest.raises(ValueError):
            T.pack_residency("sometimes")
    finally:
        T.pack_residency(None); T.pack_cache_clear()


def test_capture_first_call_of_a_row_with_process_init_is_never_initialised_inside_a_capture_and_a_capture_safe_measured_row_serves(monkeypatch):
    """|graph cells promise capturability.  Rows whose FIRST use in a process does host-side work (esm_t16: cubin load + install canary with pageable
    host->device copies; flash_sm90a: extension load + load check) are PRIMED outside any capture: transition.prime(), or automatically at the first eager
    call of a class whose cells name such a row.  Were such a row still un-primed when its first call happens INSIDE a capture, nothing is initialised
    there: the cell column's fastest other measured, admitted, capture-safe arm serves the call by name (census token), else a Refusal by name
    ('<row>:init_during_capture') -- never the raw 'cannot copy between CPU and CUDA tensors during capture' error.  P9: the bounded pack store never
    evicts while a capture is recording."""
    import types
    H = "H100:torch2.13.0+cu130/3.7.1"
    skw = dict(c=256, hidden=1024, n_tokens=400, dtype="bf16", direction="fwd", timing="graph", family="pair", residual=False, mask=False, device=None, stack=H,
               capture=True, rows_count=400 * 400, ln_given=True, form="swiglu", cc="9.0")
    sel = T.select("fast", **{k: v for k, v in skw.items() if k not in ("dtype", "direction", "device")}, dtype="bf16")
    assert sel.row in T.NEEDS_PRIME, (sel.row, sel.cell_key)                  # today's 9.0 pair c256 N<=400 graph cell names the CuTe row
    monkeypatch.setattr(T, "_PRIMED_ROWS", set()); monkeypatch.setattr(T, "_AUTO_PRIMED", set())
    engaged = {"t16": 0, "flash": 0}
    fake_t16 = types.SimpleNamespace(engage=lambda: engaged.__setitem__("t16", engaged["t16"] + 1) or {"state": "on"}, live=lambda: True, _STATE={"engaged": None})
    real_cm = T.carried_module
    monkeypatch.setattr(T, "carried_module", lambda name: fake_t16 if name == "esm_t16" else real_cm(name))
    monkeypatch.setattr(T, "_flash_ext", lambda: engaged.__setitem__("flash", engaged["flash"] + 1))
    # (1) inside a capture, un-primed: the substitute is a measured capture-safe arm of the same cell column, no init attempted
    monkeypatch.setattr(T, "_capturing", lambda: True)
    alt = T._capture_safe_substitute(sel, skw, sel.row)
    assert alt is not None and alt.row not in T.NEEDS_PRIME and alt.row not in T.STOCK_ROWS and alt.cell_key == sel.cell_key, (alt.row if alt else None)
    assert T.prime() == {"esm_t16": "refused:capturing", "flash_sm90a": "refused:capturing"} and engaged == {"t16": 0, "flash": 0}
    with pytest.raises(T.Refusal) as r:                                        # the serving branch itself refuses by name while capturing (x on a 'cuda' stand-in)
        xs = types.SimpleNamespace(is_cuda=True, shape=(400, 400, 256))
        T._dispatch(xs, types.SimpleNamespace(_policy=None), sel, "fast", False, None, None, None, None, "swiglu")
    assert r.value.kind == "%s:init_during_capture" % sel.row and engaged["t16"] == 0
    # (2) outside a capture: prime() engages each row ONCE; a primed flash row becomes an admissible substitute; primed rows dispatch normally
    monkeypatch.setattr(T, "_capturing", lambda: False)
    assert T.prime() == {"esm_t16": "primed", "flash_sm90a": "primed"} and engaged == {"t16": 1, "flash": 1}
    assert T.prime()["esm_t16"] == "primed" and engaged == {"t16": 1, "flash": 1} and T._row_primed("esm_t16") and T._row_primed("esm_t16_kd3") and T._row_primed("flash_sm90a")
    # (3) auto-prime at the first eager call of the class: every NEEDS_PRIME row any cell of the class names is primed once, then never again
    monkeypatch.setattr(T, "_PRIMED_ROWS", set()); monkeypatch.setattr(T, "_AUTO_PRIMED", set()); engaged.update(t16=0, flash=0)
    xe = types.SimpleNamespace(device="cuda:0", is_cuda=True)
    T._auto_prime(xe, sel, skw); T._auto_prime(xe, sel, skw)
    assert engaged == {"t16": 1, "flash": 1} and len(T._AUTO_PRIMED) == 1
    sel8 = T.select("fast", c=64, hidden=256, n_tokens=800, cc="8.0", family="rows", stack="A100:torch2.13.0+cu130/3.7.1", ln_given=True)
    T._auto_prime(types.SimpleNamespace(device="cuda:1", is_cuda=True), sel8, dict(skw, cc="8.0", c=64, hidden=256, family="rows"))
    assert engaged == {"t16": 1, "flash": 1}                                    # an 8.0 class names no such row: nothing engaged
    # (4) P9: no eviction from the bounded pack store while capturing; the deferred eviction happens at the next pack outside the capture
    import torch
    T.pack_cache_clear(); monkeypatch.setenv(T.PACK_CACHE_ENV, "lru"); monkeypatch.setenv(T.PACK_LRU_ENV, "1")
    Ws = [T.pack(w_a=torch.randn(16, 8), w_b=torch.randn(16, 8), w_o=torch.randn(8, 16)) for _ in range(3)]
    for W in Ws:
        W._policy = T.pack_policy("big")
    monkeypatch.setattr(T, "_capturing", lambda: True)
    Ws[0].pack_for("p", lambda: torch.zeros(4)); Ws[1].pack_for("p", lambda: torch.zeros(4))
    assert T.pack_cache_stats()["lru_entries"] == 2 and T.pack_cache_stats().get("deferred_evictions", 0) >= 1     # both resident: the capture recorded their addresses
    monkeypatch.setattr(T, "_capturing", lambda: False)
    Ws[2].pack_for("p", lambda: torch.zeros(4))
    assert T.pack_cache_stats()["lru_entries"] == 1                     # evicted down to the bound once outside the capture
    T.pack_cache_clear()
