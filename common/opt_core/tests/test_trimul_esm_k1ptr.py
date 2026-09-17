"""kernels.trimul rows esm_k1ptr / esm_k1ptr:f32in: admission words, selection by word, cells = the pointer-served esm_shapes numbers."""
import json
import pytest

from opt_core.kernels import trimul as P
from opt_core.kernels.trimul import esm_k1ptr as R


def test_rows_are_registered_and_described():
    assert "esm_k1ptr" in P.ROW_NAMES and "esm_k1ptr:f32in" in P.ROW_NAMES
    rows = P.table()["rows"]
    assert rows["esm_k1ptr"]["class"] == "fast" and rows["esm_k1ptr:f32in"]["class"] == "fast"
    assert "pointer" in rows["esm_k1ptr"]["package"]


@pytest.mark.parametrize("cc,prec,C", [("9.0", "bf16", 128), ("9.0", "bf16", 256), ("9.0", "bf16", 384), ("8.0", "bf16", 128)])
def test_admits_inside_the_envelope(cc, prec, C):
    assert R.refusal("esm_k1ptr", cc, prec, C, C, 800, triton="3.3.1") is None
    assert R.refusal("esm_k1ptr:f32in", cc, "fp32", C, C, 800, triton="3.3.1") is None
    assert R.refusal("esm_k1ptr:f32in", cc, "tf32", C, C, 800, triton="3.7.1") is None
    P.admits("esm_k1ptr", cc, "bf16", C, C, 800, triton="3.3.1")


@pytest.mark.parametrize("args,kw,word,fb", [
    (("esm_k1ptr", "7.5", "bf16", 128, 128, 800), {}, "cc:", "cueq"),
    (("esm_k1ptr", "9.0", "fp32", 128, 128, 800), {}, "dtype:", "esm_k1ptr:f32in"),
    (("esm_k1ptr:f32in", "9.0", "bf16", 128, 128, 800), {}, "dtype:", "esm_k1ptr"),
    (("esm_k1ptr", "9.0", "bf16", 64, 64, 800), {}, "shape:", "cueq"),
    (("esm_k1ptr", "9.0", "bf16", 128, 256, 800), {}, "shape:", "cueq"),
    (("esm_k1ptr", "9.0", "bf16", 256, 256, 800), {"triton": "3.2.0"}, "triton:", "v4"),
    (("esm_k1ptr", "9.0", "bf16", 384, 384, 800), {"batch": 4}, "batch:", "cueq"),
])
def test_refusals_are_by_name(args, kw, word, fb):
    r = R.refusal(*args, **kw)
    assert r is not None and r[0].startswith(word) and r[1] == fb, r
    with pytest.raises(P.Refusal) as e:
        P.admits(*args, **kw)
    assert e.value.kind.startswith(word)


def test_select_by_word_names_the_row():
    sel = P.select("9.0", "bf16", 128, 128, 800, "outgoing", word="esm_k1ptr", stack="H100:2.7.1+cu128/3.3.1/cueq0.10.0")
    assert sel.row == "esm_k1ptr"
    sel = P.select("9.0", "fp32", 384, 384, 800, "outgoing", word="esm_k1ptr:f32in", stack="H100:2.7.1+cu126/3.3.1/cueq0.10.0")
    assert sel.row == "esm_k1ptr:f32in"


def test_no_tier_word_names_the_rows_and_cells_equal_the_pointer_served_numbers():
    T = P.table()
    seen = 0
    for key, c in T["cells"].items():
        cc = key.split("|")[0]
        for w in ("fast", "exact", "big"):
            assert c.get(w) not in R.ROWS, key
            assert all(v not in R.ROWS for v in (c.get(w + "_per_stack") or {}).values()), key
        for src, dst in (("esm_shapes", "esm_k1ptr"), ("esm_shapes:f32in", "esm_k1ptr:f32in")):
            for st, v in (c.get("ms", {}).get(dst) or {}).items():
                if "/3.3." in st:                                        # pointer-served (triton-3.3) columns carry COPIES of the re-tiled row's numbers; the later
                    assert c["ms"][src][st] == v, (key, st)              # race rounds timed the row as its own arm on other columns (cc 8.0, the 2.13 H100
                                                                         # reference): those are its own numbers
                seen += 1
    assert seen >= 30, seen
