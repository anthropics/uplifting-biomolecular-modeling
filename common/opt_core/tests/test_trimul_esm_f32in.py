"""kernels.trimul row esm_v5_fwd:f32in: row esm_v5_fwd's line behind a boundary cast for fp32 / tf32 callers.  No framework here: admission and
refusals by name, the table entry, and that no tier word or cell ever chooses it."""
import json

import pytest

from opt_core.kernels import trimul as T

ROW = "esm_v5_fwd:f32in"
H100 = "H100:2.13.0+cu130/3.7.1/cueq0.11.1"


def test_row_facts_and_table_entry():
    assert ROW in T.ROW_NAMES and ROW not in T.NEEDS_ESM_IMAGE and ROW not in T.BACKWARD_ROWS and ROW not in T.EXACT_ROWS and ROW not in T.STOCK_ROWS
    tab = T.table()
    row = tab["rows"][ROW]
    assert row["class"] == "fast" and row["exact_vs"] is None and row["backward"] is False and row["admits"]["c_z"] == [128, 256] and row["admits"]["n_min"] == 101
    assert "TOLERANCE" in row["numerics"] and row["fallback"].startswith("v4")
    for tier, lst in tab["tiers"].items():
        assert ROW not in json.dumps(lst), tier                                         # in no tier list
    named = timed = 0
    for key, cell in tab["cells"].items():                                             # cells carry its same-machine numbers (ms / class / numerics_class) where a race
        for tier in ("fast", "exact", "big"):                                        # timed it, but no tier word and no fast_order entry resolves to it
            named += (cell.get(tier) == ROW) + sum(v == ROW for v in (cell.get(tier + "_per_stack") or {}).values())
        named += ROW in (cell.get("fast_order") or [])
        timed += ROW in (cell.get("ms") or {})
    assert named == 0 and timed >= 20, (named, timed)


def test_admits_fp32_and_tf32_and_refuses_by_name():
    for prec in ("fp32", "tf32"):
        for c in (128, 256):
            for d in ("outgoing", "incoming"):
                for n in (128, 400, 800, 1200):
                    s = T.select("9.0", prec, c, c, n, d, word=ROW, stack=H100)
                    assert s.row == ROW and (s.cls == "fast" or s.cls.startswith("tol(")) and s.backward is False   # the row's class word, or the class a race measured on that stack
        assert T.select("8.0", prec, 256, 256, 800, "outgoing", word=ROW).row == ROW                          # cc 8.0: the line's pointer cell, admitted by name
        assert T.select("9.0", prec, 128, 128, 800, "outgoing", word=ROW, stack="H100:2.10.0+cu128/3.6.0/cueq0.10.0/ds4s").row == ROW
    R = T.Refusal
    for args, kind, fb in ((("9.0", "bf16", 256, 256, 800), "dtype:bf16", "esm_v5_fwd"),          # bf16 callers use the bf16 row
                           (("9.0", "fp16", 256, 256, 800), "dtype:fp16", "cueq"),
                           (("7.5", "fp32", 256, 256, 800), "cc:7.5<8.0", "cueq"),
                           (("9.0", "fp32", 384, 384, 800), "c_z:384", "cueq"),
                           (("9.0", "fp32", 64, 128, 800), "c_hidden=128!=c_z=64", "torch_math"),
                           (("9.0", "fp32", 256, 256, 64), "n<101", "cueq")):
        with pytest.raises(R) as e:
            T.admits(ROW, *args)
        assert e.value.kind.startswith(kind) and e.value.row == ROW and e.value.fallback == fb, (args, e.value.kind, e.value.fallback)
    with pytest.raises(R) as e:
        T.select("9.0", "fp32", 256, 256, 800, "outgoing", word=ROW, stack="H100:2.7.1+cu128/3.3.1/cueq0.10.0")
    assert e.value.kind.startswith("triton:3.3.1<3.6") and e.value.fallback == "v4"


def test_tier_words_never_choose_it():
    for word in T.TIER_WORDS:
        for prec in ("fp32", "tf32", "bf16"):
            for c in (128, 256):
                for n in (400, 800, 1200):
                    for st in (H100, "A100:2.13.0+cu130/3.7.1/cueq0.11.1", "H100:2.10.0+cu128/3.6.0/cueq0.10.0/ds4s", None):
                        try:
                            s = T.select("9.0" if (st or "H").startswith("H") else "8.0", prec, c, c, n, "outgoing", word=word, stack=st)
                        except T.Refusal:
                            continue
                        assert s.row != ROW
